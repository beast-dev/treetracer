"""Robinson-Foulds distance computation backed by rapidtrees.

Uses the iterator API: each newick is parsed into a compact snapshot and
the raw string is discarded. The result is a numpy uint32 array via
zero-copy from the Rust side.
"""

import math
from typing import Dict, List, Tuple

import numpy as np
import rapidtrees

from .rooted_facts import RootedFactsSnapshot, rooted_facts_from_sparse
from .sparse_snapshots import SparseSnapshot, decode_sparse_snapshot, dense_presence_from_sparse


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
) -> Tuple[List[str], np.ndarray, np.ndarray, List[str], int, np.ndarray]:
    """Return the historical dense snapshot representation.

    This wrapper is retained as a reference/testing API. The TreeTracer
    application does not call it or persist its dense arrays; production RF
    jobs use rooted facts or generic CSR snapshots.

    The presence matrix is expanded from RapidTrees' sparse rows: shape
    (n_trees, n_bip), uint8 with ``presence[i, j] == 1`` iff tree ``i``
    contains bipartition ``j``. Bipartition columns are in ascending Bitset
    order so the same tree set always produces the same matrix.

    Returns:
        (names, rf_matrix, presence, leaf_names, n_bipartitions,
        bipartition_bits) where rf_matrix is uint32 (n_trees × n_trees),
        presence is uint8 (n_trees × n_bipartitions), leaf_names is the
        alphabetically-sorted taxon list and bipartition_bits is the unpacked
        uint8 (n_bipartitions × n_taxa) clade membership.
    """
    names_out, rf_matrix, sparse = rf_distance_with_sparse_snapshots_from_newick_iter(
        names, newick_iter, translate_maps, map_indices,
        rooted=rooted, progress=progress,
    )
    n_bipartitions = sparse.n_clades
    if n_bipartitions > 0:
        bipartition_bits = np.unpackbits(
            sparse.packed_clades, axis=1, bitorder="little"
        )[:, :sparse.n_taxa].copy()
    else:
        bipartition_bits = np.zeros((0, 0), dtype=np.uint8)
    return (
        names_out,
        rf_matrix,
        dense_presence_from_sparse(sparse),
        list(sparse.leaf_names),
        int(n_bipartitions),
        bipartition_bits,
    )


def rf_distance_with_sparse_snapshots_from_newick_iter(
    names: List[str],
    newick_iter,
    translate_maps: List[Dict[str, str]],
    map_indices: List[int] | None = None,
    rooted: bool = False,
    progress=None,
    lengths: bool = False,
) -> tuple[List[str], np.ndarray, SparseSnapshot]:
    """Compute RF distances and retain compact CSR clade-presence rows.

    Calls ``rapidtrees.pairwise_rf_with_snapshots_from_newick_iter``, validates
    its versioned sparse payload and keeps both membership rows and clade
    bitsets packed. With ``lengths=True`` the snapshot also carries the branch
    length above each entry, which :func:`rf_distance_with_rooted_facts_from_newick_iter`
    turns into MrHIPSTR's rooted facts.
    """
    if map_indices is None:
        map_indices = [0] * len(names)
    (
        names_out,
        rf_bytes,
        leaf_names,
        n_clades,
        sparse,
        clade_bytes,
    ) = rapidtrees.pairwise_rf_with_snapshots_from_newick_iter(
        names,
        newick_iter,
        translate_maps,
        map_indices,
        rooted=rooted,
        progress=progress,
        lengths=lengths,
    )
    n_trees = len(names_out)
    expected_rf_bytes = n_trees * n_trees * np.dtype(np.uint32).itemsize
    if len(rf_bytes) != expected_rf_bytes:
        raise ValueError(
            "RapidTrees returned an invalid sparse-snapshot RF buffer: "
            f"{len(rf_bytes)} bytes != {expected_rf_bytes}"
        )
    rf_matrix = (
        np.frombuffer(rf_bytes, dtype=np.uint32)
        .reshape(n_trees, n_trees)
        .copy()
    )
    sparse_snapshot = decode_sparse_snapshot(
        tree_names=names_out,
        leaf_names=leaf_names,
        n_clades=n_clades,
        clade_bytes=clade_bytes,
        sparse=sparse,
        rooted=rooted,
    )
    return list(names_out), rf_matrix, sparse_snapshot


def rf_distance_with_rooted_facts_from_newick_iter(
    names: List[str],
    newick_iter,
    translate_maps: List[Dict[str, str]],
    map_indices: List[int] | None = None,
    progress=None,
) -> tuple[List[str], np.ndarray, RootedFactsSnapshot]:
    """Compute rooted RF distances and derive compact MrHIPSTR facts.

    One rooted RapidTrees pass returns the clade rows with their branch
    lengths; :func:`~treetracer.rf.rooted_facts.rooted_facts_from_sparse`
    derives heights and observed binary splits from them in NumPy, without
    reparsing the source Newicks.

    Raises:
        ValueError: If a tree is not strictly binary or a height is not a
            finite ``float32``. Callers that already hold the sparse snapshot
            should call :func:`rooted_facts_from_sparse` themselves and keep
            the snapshot instead.
    """
    names_out, rf_matrix, sparse = rf_distance_with_sparse_snapshots_from_newick_iter(
        names, newick_iter, translate_maps, map_indices,
        rooted=True, progress=progress, lengths=True,
    )
    return names_out, rf_matrix, rooted_facts_from_sparse(sparse)
