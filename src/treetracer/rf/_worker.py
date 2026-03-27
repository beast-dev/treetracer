"""Subprocess workers for RF and MDS computation.

Kept in a separate module so that ProcessPoolExecutor child processes only
need to import rapidtrees / numpy — not Dash, DMC, or the full callback stack.
"""


def compute_rf(names, newicks, translate_maps, map_indices):
    """Compute pairwise RF distances. Runs in a subprocess with its own GIL."""
    import time
    t0 = time.time()
    from .rf import rf_distance_from_newicks
    result_names, matrix = rf_distance_from_newicks(
        names, newicks, translate_maps, map_indices, rooted=False,
    )
    elapsed = time.time() - t0
    return list(result_names), matrix, elapsed


def compute_mds_worker(matrix, n_components):
    """Compute PCoA embedding. Runs in a subprocess with its own GIL."""
    import time
    import numpy as np
    t0 = time.time()
    from .mds import compute_mds
    distance_matrix = np.array(matrix, dtype=float)
    embedding = compute_mds(distance_matrix, n_components=n_components, algorithm="pcoa")
    elapsed = time.time() - t0
    return embedding.tolist(), elapsed
