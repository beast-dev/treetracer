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

    Routes through ``rf_distance_with_snapshots_from_newick_iter``, which
    calls rapidtrees' ``pairwise_rf_with_snapshots_from_newick_iter``. RF
    runs on u32 split IDs against a globally-deduped bipartition table, so
    a comparison is a single integer compare and a large run's working set
    stays in cache rather than DRAM.

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
    - ``<save_path stem>_snapshots.npz``:   ``presence`` (uint8, n_trees ×
                                            n_clades-or-bipartitions) plus
                                            ``leaf_names`` (alphabetical
                                            taxon list).

    The presence matrix isn't free to compute, but it's the sufficient
    statistic for every topology-based convergence diagnostic
    (Pseudo-ESS, Fréchet correlation ESS, ASDSF) — saving it now means
    those callers can read it from disk later without re-parsing .trees
    files or recomputing snapshots.

    Returns (result_names, elapsed). Both matrices stay on disk, never
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
    wlog("compute_rf: importing rapidtrees binding (.rf.rf_distance_with_snapshots_from_newick_iter)")
    from .rf import rf_distance_with_snapshots_from_newick_iter
    wlog(f"compute_rf: rapidtrees imported in {time.time() - t0:.3f}s; calling pairwise RF")
    call_t0 = time.time()
    result_names, rf_matrix, presence, leaf_names, _n_bip, bipartition_bits = (
        rf_distance_with_snapshots_from_newick_iter(
            names, iter(newicks), translate_maps, map_indices,
            rooted=is_rooted, progress=progress,
        )
    )
    wlog(
        f"compute_rf: rapidtrees returned in {time.time() - call_t0:.3f}s; "
        f"rf_matrix.shape={rf_matrix.shape}, presence.shape={presence.shape}"
    )
    # rf_matrix is uint32 from Rust; downcast to uint16 for disk storage
    # (RF distances are bounded by 2*(n_taxa-3), trivially fits).
    wlog(f"compute_rf: saving uint16 distmat to {save_path!r}")
    np.save(save_path, rf_matrix.astype(np.uint16))

    # Save the presence matrix + leaf_names alongside, with a derived path
    # so a single registry entry implicitly knows where to find both.
    snap_path = Path(save_path).with_name(Path(save_path).stem + "_snapshots.npz")
    wlog(f"compute_rf: saving snapshot .npz to {str(snap_path)!r}")
    np.savez(snap_path,
             presence=presence,
             leaf_names=np.array(leaf_names),
             bipartition_bits=bipartition_bits)
    wlog("compute_rf: both files written; returning")

    elapsed = time.time() - t0
    return list(result_names), elapsed


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
