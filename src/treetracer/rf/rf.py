"""Robinson-Foulds distance computation backed by rust_python_tree_distances.

Provides two entry points:
  - rf_distance_from_newicks: for trees already in memory (newick strings + translate map)
  - rf_distance_from_file:    for trees still on disk (.trees NEXUS file)

Both return (names, matrix) where matrix is a symmetric list-of-lists of ints.
"""

from typing import Dict, Iterator, List, Tuple

import numpy as np
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


def rf_distance_from_newick_iter(
    names: List[str],
    newick_iter: Iterator[str],
    translate_maps: List[Dict[str, str]],
    map_indices: List[int] | None = None,
    rooted: bool = False,
) -> Tuple[List[str], List[List[int]]]:
    """Compute pairwise RF distances from a lazy iterator of newick strings.

    Unlike rf_distance_from_newicks, this never holds all newick strings in
    memory at once. The Rust side pulls one newick at a time from the
    iterator, parses it into a compact snapshot, and discards the raw string.

    Args:
        names: Tree identifiers (one per newick).
        newick_iter: Iterator yielding newick strings (may contain BEAST
            annotations). Must yield exactly len(names) strings.
        translate_maps: List of translate maps. When all trees share the same
            map, pass a single-element list.
        map_indices: Per-tree index into *translate_maps*. Defaults to all-zero
            (every tree uses the first map).
        rooted: If True compare clades (rooted RF); if False compare
            bipartitions (unrooted RF, matches R phangorn default).

    Returns:
        (names, matrix) — tree identifiers and symmetric numpy uint32 array.
    """
    if map_indices is None:
        map_indices = [0] * len(names)
    names_out, matrix_bytes = rtd.pairwise_rf_from_newick_iter(
        names, newick_iter, translate_maps, map_indices, rooted=rooted,
    )
    n = len(names_out)
    matrix = np.frombuffer(matrix_bytes, dtype=np.uint32).reshape(n, n).copy()
    return names_out, matrix


def rf_distance_from_file(
    trees_path: str,
    burnin_trees: int = 0,
    rooted: bool = False,
) -> Tuple[List[str], List[List[int]]]:
    """Compute pairwise RF distances from a .trees NEXUS file.

    Args:
        trees_path: Path to a BEAST/NEXUS .trees file.
        burnin_trees: Number of trees to skip at the beginning.
        rooted: If True compare clades; if False compare bipartitions.

    Returns:
        (names, matrix) — tree identifiers and symmetric distance matrix.
    """
    return rtd.pairwise_rf(
        [trees_path], burnin_trees=burnin_trees, use_real_taxa=True, rooted=rooted,
    )


def rf_distance_from_files(
    trees_paths: List[str],
    burnin_trees: int = 0,
    rooted: bool = False,
) -> Tuple[List[str], List[List[int]]]:
    """Compute pairwise RF distances across multiple .trees files.

    All files must share the same taxon set.

    Args:
        trees_paths: Paths to BEAST/NEXUS .trees files.
        burnin_trees: Number of trees to skip per file.
        rooted: If True compare clades; if False compare bipartitions.

    Returns:
        (names, matrix) — tree identifiers and symmetric distance matrix.
    """
    return rtd.pairwise_rf(
        trees_paths, burnin_trees=burnin_trees, use_real_taxa=True, rooted=rooted,
    )


# ---------------------------------------------------------------------------
# One-vs-many distance (for diagnostics trace)
# ---------------------------------------------------------------------------

def rf_distance_to_reference(
    names: List[str],
    newick_iter: Iterator[str],
    n_trees: int,
    ref_newick: str,
    translate_maps: List[Dict[str, str]],
    map_indices: List[int],
    ref_map_index: int = 0,
    rooted: bool = False,
) -> Tuple[List[str], np.ndarray]:
    """Compute RF distance of every tree to a single reference tree.

    Uses the Rust cross-block function for O(n) computation instead of
    building the full pairwise matrix.

    Args:
        names: Tree identifiers (one per newick in the iterator).
        newick_iter: Iterator yielding newick strings for all trees.
        n_trees: Number of trees the iterator will yield.
        ref_newick: Newick string of the reference tree.
        translate_maps: List of translate maps.
        map_indices: Per-tree index into translate_maps.
        ref_map_index: Index into translate_maps for the reference tree.
        rooted: If True compare clades; if False compare bipartitions.

    Returns:
        (names, distances) — tree names and 1-D array of RF distances
        to the reference tree.
    """
    result_bytes = rtd.pairwise_rf_cross_block(
        newick_iter,            # iter A = all trees
        n_trees,
        iter([ref_newick]),     # iter B = single reference
        1,
        translate_maps,
        map_indices,            # map indices for A
        [ref_map_index],        # map indices for B
        rooted=rooted,
    )
    # Result is n_trees x 1 matrix in row-major u32 bytes
    distances = np.frombuffer(result_bytes, dtype=np.uint32).copy()
    return names, distances


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def matrix_to_numpy(matrix: List[List[int]]) -> np.ndarray:
    """Convert the list-of-lists distance matrix to a numpy array."""
    return np.array(matrix, dtype=np.int32)


def matrix_to_dict(
    names: List[str],
    matrix: List[List[int]],
) -> Dict[str, Dict[str, int]]:
    """Convert (names, matrix) into a nested dict keyed by tree name."""
    return {
        names[i]: {names[j]: matrix[i][j] for j in range(len(names))}
        for i in range(len(names))
    }
