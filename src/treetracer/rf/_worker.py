"""Subprocess workers for RF and MDS computation.

Kept in a separate module so that ProcessPoolExecutor child processes only
need to import rapidtrees / numpy — not Dash, DMC, or the full callback stack.

Workers save results directly to disk to avoid pickling large matrices
back across the process boundary.
"""


def compute_rf(names, newicks, translate_maps, map_indices, save_path,
               is_rooted=True, progress=None):
    """Compute pairwise RF distances + the per-tree split presence matrix in
    a single rapidtrees call, and persist both to disk.

    Rooted inputs prefer ``rf_distance_with_rooted_facts_from_newick_iter`` so
    the same RapidTrees parse also retains MrHIPSTR heights and directly
    observed splits. Unrooted inputs, older RapidTrees installations, and
    rooted inputs incompatible with the strict facts contract prefer the
    generic sparse-snapshot endpoint when available, then retain the
    established dense route as a compatibility fallback. All routes use
    RapidTrees' interned u32 clade IDs for RF computation.

    ``is_rooted`` (forwarded as ``rooted=`` to rapidtrees):
      * True  → every internal-node descendant set is one clade. Honest
        about MCMC rooting variability — a bipartition rooted differently
        in different samples appears as two distinct rooted clades.
      * False → bipartition mode: each split divides taxa into two
        unordered sets, regardless of which side the (arbitrary) newick
        root is on. Correct mode for MrBayes/RevBayes unrooted output.
    Caller validates that the input ``.trees`` files all share rooting;
    see ``callbacks/compute.py:handle_compute_rf``.

    Two files are written:

    - ``save_path``:                        uint16 ``.npy`` n×n RF matrix.
    - ``<save_path stem>_snapshots.npz``:   the established ``presence``,
                                            ``leaf_names``, and
                                            ``bipartition_bits`` arrays plus a
                                            compact sparse representation when
                                            RapidTrees provides one. Rooted
                                            compatible inputs carry versioned
                                            ``rooted_facts_*`` arrays, whose
                                            clade rows are already sparse.

    Dense compatibility arrays remain during this additive migration for
    rollback and parity checks. Consumers prefer sparse rows; MrHIPSTR reads
    compact facts when present and only reparses source trees for generic
    sparse or legacy/incompatible snapshots.

    Returns ``(result_names, elapsed, rf_details)``. ``rf_details`` records
    the RF interpretation and whether the optional rooted-facts endpoint
    actually succeeded, so the parent console can report the observed path
    rather than infer it from the requested rooting mode. The large arrays
    stay on disk and are never pickled across the process boundary.
    """
    import time
    from pathlib import Path

    import numpy as np

    # Worker-log checkpoints for the rapidtrees path. Each major
    # phase (rapidtrees DLL import, the compute itself, the two
    # numpy.save calls) gets one log line — enough to pinpoint a
    # hang to one of those phases without forcing us to instrument
    # the Rust side. See treetracer._worker_log for the path /
    # rationale.
    try:
        from .._worker_log import log as wlog
    except Exception:  # noqa: BLE001
        wlog = lambda _msg: None  # noqa: E731 — best-effort logging

    t0 = time.time()
    wlog("compute_rf: importing rapidtrees RF wrappers")
    import rapidtrees

    from .rf import (
        rf_distance_with_rooted_facts_from_newick_iter,
        rf_distance_with_sparse_snapshots_from_newick_iter,
        rf_distance_with_snapshots_from_newick_iter,
    )
    from .rooted_facts import rooted_facts_npz_payload
    from .sparse_snapshots import (
        dense_clade_bits_from_sparse,
        dense_presence_from_sparse,
        sparse_snapshot_from_rooted_facts,
        sparse_snapshot_npz_payload,
    )

    wlog(
        "compute_rf: rapidtrees imported in "
        f"{time.time() - t0:.3f}s; calling pairwise RF"
    )
    call_t0 = time.time()
    rooted_facts = None
    sparse_snapshot = None
    sparse_snapshot_source = "unavailable"
    rooted_facts_status = (
        "pending" if is_rooted else "not_applicable_unrooted"
    )
    has_rooted_facts_endpoint = hasattr(
        rapidtrees,
        "pairwise_rf_with_rooted_facts_from_newick_iter",
    )
    if is_rooted and has_rooted_facts_endpoint:
        try:
            wlog("compute_rf: using RapidTrees rooted-facts endpoint")
            result_names, rf_matrix, rooted_facts = (
                rf_distance_with_rooted_facts_from_newick_iter(
                    names,
                    iter(newicks),
                    translate_maps,
                    map_indices,
                    progress=progress,
                )
            )
            rooted_facts_status = "used"
        except ValueError as exc:
            # Rooted facts deliberately require strict binary trees and an
            # explicit finite length on every non-root edge. Preserve RF/MCC
            # compatibility for other rooted inputs by retrying the generic
            # sparse endpoint (or the established dense endpoint on older
            # RapidTrees); MrHIPSTR retains its source-Newick fallback.
            wlog(
                "compute_rf: rooted facts unavailable for this dataset; "
                f"falling back to clade snapshots ({exc})"
            )
            rooted_facts = None
            rooted_facts_status = "input_incompatible"
    elif is_rooted:
        rooted_facts_status = "endpoint_unavailable"

    if rooted_facts is not None:
        sparse_snapshot = sparse_snapshot_from_rooted_facts(rooted_facts)
        sparse_snapshot_source = "rooted_facts"
        leaf_names = list(sparse_snapshot.leaf_names)
        presence = dense_presence_from_sparse(sparse_snapshot)
        bipartition_bits = dense_clade_bits_from_sparse(sparse_snapshot)
    elif hasattr(
        rapidtrees,
        "pairwise_rf_with_sparse_snapshots_from_newick_iter",
    ):
        wlog("compute_rf: using RapidTrees sparse-snapshot endpoint")
        (
            result_names,
            rf_matrix,
            sparse_snapshot,
        ) = rf_distance_with_sparse_snapshots_from_newick_iter(
            names,
            iter(newicks),
            translate_maps,
            map_indices,
            rooted=is_rooted,
            progress=progress,
        )
        sparse_snapshot_source = "sparse_endpoint"
        leaf_names = list(sparse_snapshot.leaf_names)
        presence = dense_presence_from_sparse(sparse_snapshot)
        bipartition_bits = dense_clade_bits_from_sparse(sparse_snapshot)
    else:
        (
            result_names,
            rf_matrix,
            presence,
            leaf_names,
            _n_bip,
            bipartition_bits,
        ) = rf_distance_with_snapshots_from_newick_iter(
            names,
            iter(newicks),
            translate_maps,
            map_indices,
            rooted=is_rooted,
            progress=progress,
        )
    wlog(
        f"compute_rf: rapidtrees returned in {time.time() - call_t0:.3f}s; "
        f"rf_matrix.shape={rf_matrix.shape}, presence.shape={presence.shape}, "
        f"rooted_facts={'yes' if rooted_facts is not None else 'no'}, "
        f"sparse_snapshot={sparse_snapshot_source}"
    )
    # rf_matrix is uint32 from Rust; downcast to uint16 for disk storage
    # (RF distances are bounded by 2*(n_taxa-3), trivially fits).
    wlog(f"compute_rf: saving uint16 distmat to {save_path!r}")
    np.save(save_path, rf_matrix.astype(np.uint16))

    # Save the presence matrix + leaf_names alongside, with a derived path
    # so a single registry entry implicitly knows where to find both.
    snap_path = Path(save_path).with_name(Path(save_path).stem + "_snapshots.npz")
    wlog(f"compute_rf: saving snapshot .npz to {str(snap_path)!r}")
    snapshot_arrays = {
        "presence": presence,
        "leaf_names": np.array(leaf_names),
        "bipartition_bits": bipartition_bits,
    }
    if rooted_facts is not None:
        snapshot_arrays.update(rooted_facts_npz_payload(rooted_facts))
    elif sparse_snapshot is not None:
        snapshot_arrays.update(sparse_snapshot_npz_payload(sparse_snapshot))
    np.savez(snap_path, **snapshot_arrays)
    wlog("compute_rf: both files written; returning")

    elapsed = time.time() - t0
    rf_details = {
        "rf_mode": (
            "rooted_clades" if is_rooted else "unrooted_bipartitions"
        ),
        "rooted_facts_used": rooted_facts is not None,
        "rooted_facts_status": rooted_facts_status,
        "sparse_snapshot_used": sparse_snapshot is not None,
        "sparse_snapshot_source": sparse_snapshot_source,
    }
    return list(result_names), elapsed, rf_details


def compute_mds_worker(matrix_path, n_components, progress_path=None):
    """Compute PCoA embedding from a matrix on disk.

    Reads the .npy file directly — avoids pickling large matrices.
    Returns (embedding_list, elapsed).
    """
    import json
    import time
    from pathlib import Path

    import numpy as np

    progress_file = Path(progress_path) if progress_path else None

    def write_progress(phase, fraction, label):
        if progress_file is None:
            return
        try:
            progress_file.write_text(json.dumps({
                "phase": phase,
                "fraction": max(0.0, min(float(fraction), 1.0)),
                "label": label,
            }))
        except OSError:
            pass

    t0 = time.time()
    from .mds import compute_mds
    write_progress("loading", 0.05, "loading distance matrix…")
    try:
        distance_matrix = np.load(matrix_path).astype(float)
        write_progress("loading", 0.12, "distance matrix loaded")
        embedding = compute_mds(
            distance_matrix,
            n_components=n_components,
            algorithm="pcoa_fast",
            progress=write_progress,
        )
        write_progress("finalizing", 0.98, "serializing coordinates…")
        elapsed = time.time() - t0
        write_progress("complete", 1.0, "MDS embedding complete")
        return embedding.tolist(), elapsed
    finally:
        if progress_file is not None:
            try:
                progress_file.unlink()
            except OSError:
                pass
