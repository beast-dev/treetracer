"""Subprocess workers for RF and MDS computation.

Kept in a separate module so that ProcessPoolExecutor child processes only
need to import rapidtrees / numpy — not Dash, DMC, or the full callback stack.

Workers save results directly to disk to avoid pickling large matrices
back across the process boundary.
"""


def compute_rf(names, newicks, translate_maps, map_indices, save_path):
    """Compute pairwise RF distances and save matrix to disk as uint16 .npy.

    Uses the iterator API: each newick is parsed into a compact snapshot and
    the raw string is discarded. The result comes back as raw bytes (uint32)
    which numpy wraps with zero-copy — no intermediate list-of-lists.

    Returns (result_names, elapsed) — the matrix stays on disk, never pickled.
    """
    import time
    import numpy as np
    t0 = time.time()
    from .rf import rf_distance_from_newick_iter
    result_names, matrix = rf_distance_from_newick_iter(
        names, iter(newicks), translate_maps, map_indices, rooted=False,
    )
    # matrix is already a numpy uint32 array from the Rust side
    # Downcast to uint16 for disk storage (RF distances fit in uint16)
    np.save(save_path, matrix.astype(np.uint16))
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
