"""Robinson-Foulds distance computation backed by rust_python_tree_distances.

Provides rf_distance_from_newicks for trees already in memory (newick strings
+ translate map), returning (names, matrix) where matrix is a symmetric
list-of-lists of ints.
"""

from typing import Dict, List, Tuple

import rust_python_tree_distances as rtd


# ---------------------------------------------------------------------------
# Core distance functions
# ---------------------------------------------------------------------------

def rf_distance_from_newicks(
    names: List[str],
    newicks: List[str],
    translate_maps: List[Dict[str, str]],
    map_indices: List[int] | None = None,
    rooted: bool = False,
) -> Tuple[List[str], List[List[int]]]:
    """Compute pairwise RF distances from newick strings and translate maps.

    Args:
        names: Tree identifiers (one per newick).
        newicks: Newick strings (may contain BEAST annotations).
        translate_maps: List of translate maps. When all trees share the same
            map, pass a single-element list.
        map_indices: Per-tree index into *translate_maps*. Defaults to all-zero
            (every tree uses the first map).
        rooted: If True compare clades (rooted RF); if False compare
            bipartitions (unrooted RF, matches R phangorn default).

    Returns:
        (names, matrix) — tree identifiers and symmetric distance matrix.
    """
    if map_indices is None:
        map_indices = [0] * len(newicks)
    return rtd.pairwise_rf_from_newicks(
        names, newicks, translate_maps, map_indices, rooted=rooted,
    )



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def matrix_to_dict(
    names: List[str],
    matrix: List[List[int]],
) -> Dict[str, Dict[str, int]]:
    """Convert (names, matrix) into a nested dict keyed by tree name."""
    return {
        names[i]: {names[j]: matrix[i][j] for j in range(len(names))}
        for i in range(len(names))
    }
