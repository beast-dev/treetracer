"""Subprocess workers for RF and MDS computation.

Kept in a separate module so that ProcessPoolExecutor child processes only
need to import rapidtrees / numpy — not Dash, DMC, or the full callback stack.

Workers save results directly to disk to avoid pickling large matrices
back across the process boundary.
"""


def compute_rf(names, newicks, translate_maps, map_indices, save_path):
    """Compute pairwise RF distances + the per-tree split presence matrix in
    a single rapidtrees call, and persist both to disk.

    Routes through ``rf_distance_with_snapshots_from_newick_iter``, which
    calls rapidtrees' interned-snapshot pyfunction
    (``pairwise_rf_with_snapshots_interned_from_newick_iter``). RF runs on
    u32 split IDs against a globally-deduped bipartition table — at 1000+
    taxa this is roughly 5–10× faster than the legacy bitset path because
    the inner loop's working set fits in L1.

    Two files are written:

    - ``save_path``:                        uint16 ``.npy`` n×n RF matrix.
    - ``<save_path stem>_snapshots.npz``:   ``presence`` (uint8, n_trees ×
                                            n_bipartitions) plus
                                            ``leaf_names`` (alphabetical
                                            taxon list defining the bit
                                            width of each bipartition).

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

    t0 = time.time()
    from .rf import rf_distance_with_snapshots_from_newick_iter
    result_names, rf_matrix, presence, leaf_names, _n_bip = (
        rf_distance_with_snapshots_from_newick_iter(
            names, iter(newicks), translate_maps, map_indices, rooted=False,
        )
    )
    # rf_matrix is uint32 from Rust; downcast to uint16 for disk storage
    # (RF distances are bounded by 2*(n_taxa-3), trivially fits).
    np.save(save_path, rf_matrix.astype(np.uint16))

    # Save the presence matrix + leaf_names alongside, with a derived path
    # so a single registry entry implicitly knows where to find both.
    snap_path = Path(save_path).with_name(Path(save_path).stem + "_snapshots.npz")
    np.savez(snap_path, presence=presence, leaf_names=np.array(leaf_names))

    elapsed = time.time() - t0
    return list(result_names), elapsed


def compute_mds_worker(matrix_path, n_components):
    """Compute PCoA embedding from a matrix on disk.

    Reads the .npy file directly — avoids pickling large matrices.
    Returns (embedding_list, elapsed).
    """
    import time
    import numpy as np
    t0 = time.time()
    from .mds import compute_mds
    distance_matrix = np.load(matrix_path).astype(float)
    embedding = compute_mds(distance_matrix, n_components=n_components, algorithm="pcoa")
    elapsed = time.time() - t0
    return embedding.tolist(), elapsed
