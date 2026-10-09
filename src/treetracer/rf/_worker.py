"""Subprocess workers for RF and MDS computation.

Kept in a separate module so that ProcessPoolExecutor child processes only
need to import rapidtrees / numpy — not Dash, DMC, or the full callback stack.

Workers save results directly to disk to avoid pickling large matrices
back across the process boundary.
"""


def compute_rf(names, newicks, translate_maps, map_indices, save_path,
               is_rooted=True, progress=None):
    """Compute pairwise RF distances and persist compact clade snapshots.

    One ``rf_distance_with_sparse_snapshots_from_newick_iter`` call returns the
    RF matrix and CSR clade rows. Rooted inputs also ask RapidTrees for branch
    lengths, from which ``rooted_facts_from_sparse`` derives MrHIPSTR heights
    and directly observed splits in NumPy. Rooted inputs that are not strictly
    binary keep the CSR snapshot from the same call. The persisted snapshot
    never contains a dense tree-by-clade presence matrix or an unpacked
    clade-by-taxon matrix.

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
    - ``<save_path stem>_snapshots.npz``: versioned rooted-facts arrays or a
      versioned generic CSR snapshot, including a bit-packed clade catalog.

    MrHIPSTR reads compact rooted facts when present and reparses source trees
    only when the generic CSR representation lacks heights and observed
    parent/child splits.

    Returns ``(result_names, elapsed, rf_details)``. ``rf_details`` records
    the RF interpretation and whether rooted facts were derived, so the
    parent console can report the observed path rather than infer it from
    the requested rooting mode. The large arrays stay on disk and are never
    pickled across the process boundary.
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
    from .rf import rf_distance_with_sparse_snapshots_from_newick_iter
    from .rooted_facts import rooted_facts_from_sparse, rooted_facts_npz_payload
    from .sparse_snapshots import sparse_snapshot_npz_payload

    wlog(
        "compute_rf: rapidtrees imported in "
        f"{time.time() - t0:.3f}s; calling pairwise RF"
    )
    call_t0 = time.time()
    result_names, rf_matrix, sparse_snapshot = (
        rf_distance_with_sparse_snapshots_from_newick_iter(
            names,
            iter(newicks),
            translate_maps,
            map_indices,
            rooted=is_rooted,
            progress=progress,
            lengths=is_rooted,
        )
    )
    rooted_facts = None
    rooted_facts_status = "not_applicable_unrooted"
    if is_rooted:
        try:
            rooted_facts = rooted_facts_from_sparse(sparse_snapshot)
            rooted_facts_status = "used"
        except ValueError as exc:
            # Rooted facts need strictly binary trees with finite heights.
            # Other rooted inputs keep RF/MCC support through the CSR
            # snapshot; MrHIPSTR retains its source-Newick ingestion path.
            wlog(
                "compute_rf: rooted facts unavailable for this dataset; "
                f"keeping clade snapshots ({exc})"
            )
            rooted_facts_status = "input_incompatible"
    sparse_snapshot_source = (
        "rooted_facts" if rooted_facts is not None else "sparse_endpoint"
    )
    snapshot_n_trees = sparse_snapshot.n_trees
    snapshot_n_clades = sparse_snapshot.n_clades
    snapshot_n_entries = sparse_snapshot.n_entries
    wlog(
        f"compute_rf: rapidtrees returned in {time.time() - call_t0:.3f}s; "
        f"rf_matrix.shape={rf_matrix.shape}, "
        f"snapshot_trees={snapshot_n_trees}, "
        f"snapshot_clades={snapshot_n_clades}, "
        f"snapshot_entries={snapshot_n_entries}, "
        f"rooted_facts={'yes' if rooted_facts is not None else 'no'}, "
        f"sparse_snapshot={sparse_snapshot_source}"
    )
    # rf_matrix is uint32 from Rust; downcast to uint16 for disk storage
    # (RF distances are bounded by 2*(n_taxa-3), trivially fits).
    wlog(f"compute_rf: saving uint16 distmat to {save_path!r}")
    np.save(save_path, rf_matrix.astype(np.uint16))

    # Save only the compact snapshot alongside the RF matrix. Rooted facts
    # already contain fixed-width sparse clade rows; all other inputs use CSR.
    snap_path = Path(save_path).with_name(Path(save_path).stem + "_snapshots.npz")
    wlog(f"compute_rf: saving snapshot .npz to {str(snap_path)!r}")
    if rooted_facts is not None:
        snapshot_arrays = rooted_facts_npz_payload(rooted_facts)
    else:
        snapshot_arrays = sparse_snapshot_npz_payload(sparse_snapshot)
    np.savez(snap_path, **snapshot_arrays)
    wlog("compute_rf: both files written; returning")

    elapsed = time.time() - t0
    rf_details = {
        "rf_mode": (
            "rooted_clades" if is_rooted else "unrooted_bipartitions"
        ),
        "rooted_facts_used": rooted_facts is not None,
        "rooted_facts_status": rooted_facts_status,
        "sparse_snapshot_used": True,
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
