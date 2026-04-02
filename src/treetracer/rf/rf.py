"""Robinson-Foulds distance computation backed by rapidtrees.

Uses the iterator API: each newick is parsed into a compact snapshot and
the raw string is discarded. The result is a numpy uint32 array via
zero-copy from the Rust side.
"""

from typing import Dict, List, Tuple

import numpy as np
import rapidtrees


# ---------------------------------------------------------------------------
# Core distance functions
# ---------------------------------------------------------------------------

def rf_distance_from_newick_iter(
    names: List[str],
    newick_iter,
    translate_maps: List[Dict[str, str]],
    map_indices: List[int] | None = None,
    rooted: bool = False,
) -> Tuple[List[str], np.ndarray]:
    """Compute pairwise RF distances from a lazy iterator of newick strings.

    Each newick is parsed into a compact snapshot and the raw string is
    discarded — only one newick is in memory at a time. The result is
    returned as a numpy uint32 array via zero-copy from the Rust side.

    Args:
        names: Tree identifiers (one per newick).
        newick_iter: Iterator yielding newick strings. Must yield exactly
            len(names) strings.
        translate_maps: List of translate maps. When all trees share the same
            map, pass a single-element list.
        map_indices: Per-tree index into *translate_maps*. Defaults to all-zero.
        rooted: If True compare clades; if False compare bipartitions.

    Returns:
        (names, matrix) where matrix is an n×n numpy uint32 array.
    """
    if map_indices is None:
        map_indices = [0] * len(names)
    names_out, matrix_bytes = rapidtrees.pairwise_rf_from_newick_iter(
        names, newick_iter, translate_maps, map_indices, rooted=rooted,
    )
    n = len(names_out)
    matrix = np.frombuffer(matrix_bytes, dtype=np.uint32).reshape(n, n).copy()
    return names_out, matrix


