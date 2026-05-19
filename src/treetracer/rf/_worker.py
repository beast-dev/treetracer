"""Subprocess workers for RF and MDS computation.

Kept in a separate module so that ProcessPoolExecutor child processes only
need to import rapidtrees / numpy — not Dash, DMC, or the full callback stack.

Workers save results directly to disk to avoid pickling large matrices
back across the process boundary.
"""


def compute_rf(names, newicks, translate_maps, map_indices, save_path,
               is_rooted=True):
    """Compute pairwise RF distances + the per-tree split presence matrix in
    a single rapidtrees call, and persist both to disk.

    Routes through ``rf_distance_with_snapshots_from_newick_iter``, which
    calls rapidtrees' interned-snapshot pyfunction
    (``pairwise_rf_with_snapshots_interned_from_newick_iter``). RF runs on
    u32 split IDs against a globally-deduped bipartition table — at 1000+
    taxa this is roughly 5–10× faster than the legacy bitset path because
    the inner loop's working set fits in L1.

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

    t0 = time.time()
    from .rf import rf_distance_with_snapshots_from_newick_iter
    result_names, rf_matrix, presence, leaf_names, _n_bip, bipartition_bits = (
        rf_distance_with_snapshots_from_newick_iter(
            names, iter(newicks), translate_maps, map_indices,
            rooted=is_rooted,
        )
    )
    # rf_matrix is uint32 from Rust; downcast to uint16 for disk storage
    # (RF distances are bounded by 2*(n_taxa-3), trivially fits).
    np.save(save_path, rf_matrix.astype(np.uint16))

    # Save the presence matrix + leaf_names alongside, with a derived path
    # so a single registry entry implicitly knows where to find both.
    snap_path = Path(save_path).with_name(Path(save_path).stem + "_snapshots.npz")
    np.savez(snap_path,
             presence=presence,
             leaf_names=np.array(leaf_names),
             bipartition_bits=bipartition_bits)

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
    embedding = compute_mds(distance_matrix, n_components=n_components, algorithm="pcoa_fast")
    elapsed = time.time() - t0
    return embedding.tolist(), elapsed
