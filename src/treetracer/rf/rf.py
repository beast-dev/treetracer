"""Robinson-Foulds distance computation backed by rapidtrees.

Uses the iterator API: each newick is parsed into a compact snapshot and
the raw string is discarded. The result is a numpy uint32 array via
zero-copy from the Rust side.
"""

from typing import Dict, List, Tuple

import numpy as np
import rapidtrees
import math


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


def rf_distance_with_snapshots_from_newick_iter(
    names: List[str],
    newick_iter,
    translate_maps: List[Dict[str, str]],
    map_indices: List[int] | None = None,
    rooted: bool = False,
    progress=None,
) -> Tuple[List[str], np.ndarray, np.ndarray, List[str], int]:
    """Compute pairwise RF distances *and* the per-tree split presence matrix.

    Calls rapidtrees' interned-snapshot pyfunction, which builds an
    InternedSnapshots representation under the hood: every distinct
    bipartition is assigned a u32 ID and the RF inner loop runs on integers
    instead of multi-word bitset memcmps. At 1000+ taxa the speedup over the
    legacy bitset path is roughly 5–10× because the working set fits in L1
    and there are no pointer-chased heap reads in the merge.

    The presence matrix is returned as a byproduct: shape (n_trees, n_bip),
    uint8 with ``presence[i, j] == 1`` iff tree ``i`` contains bipartition
    ``j``. Bipartition columns are in ascending Bitset order so the same
    tree set always produces the same matrix. Downstream convergence
    diagnostics (Pseudo-ESS, Fréchet correlation ESS, ASDSF) consume this
    matrix directly without re-parsing the original .trees files.

    Returns:
        (names, rf_matrix, presence, leaf_names, n_bipartitions) where
        rf_matrix is uint32 (n_trees × n_trees), presence is uint8
        (n_trees × n_bipartitions), leaf_names is the alphabetically-sorted
        taxon list (length matches the bit width inside each bipartition).
    """
    if map_indices is None:
        map_indices = [0] * len(names)
    names_out, rf_bytes, leaf_names, n_bipartitions, presence_bytes, bip_bytes = (
        rapidtrees.pairwise_rf_with_snapshots_from_newick_iter(
            names, newick_iter, translate_maps, map_indices,
            rooted=rooted, progress=progress,
        )
    )
    n = len(names_out)
    rf_matrix = np.frombuffer(rf_bytes, dtype=np.uint32).reshape(n, n).copy()
    presence = np.frombuffer(presence_bytes, dtype=np.uint8).reshape(n, n_bipartitions).copy()
    
    #words_per_bip = len(bip_bytes) // (n_bipartitions * 8) if n_bipartitions > 0 else 0
    #bipartition_bits = np.frombuffer(bip_bytes, dtype=np.uint64).reshape(n_bipartitions, words_per_bip).copy()
    if n_bipartitions > 0:
        bytes_per_bip = math.ceil(len(list(leaf_names)) / 8)
        bip_arr = np.frombuffer(bip_bytes, dtype=np.uint8).reshape(n_bipartitions, bytes_per_bip)
        bipartition_bits = np.unpackbits(bip_arr, axis=1, bitorder='little')[:, :len(leaf_names)].copy()
    else:
        bipartition_bits = np.zeros((0, 0), dtype=np.uint8)
    
    return names_out, rf_matrix, presence, list(leaf_names), int(n_bipartitions), bipartition_bits


