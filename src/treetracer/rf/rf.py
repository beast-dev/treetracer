"""Robinson-Foulds distance computation backed by rapidtrees.

Uses the iterator API: each newick is parsed into a compact snapshot and
the raw string is discarded. The result is a numpy uint32 array via
zero-copy from the Rust side.
"""

import math
from typing import Dict, List, Tuple

import numpy as np
import rapidtrees

from .rooted_facts import RootedFactsSnapshot, decode_rooted_facts_snapshot
from .sparse_snapshots import SparseSnapshot, decode_sparse_snapshot


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
    """Return the historical dense snapshot representation.

    This wrapper is retained as a reference/testing API. The TreeTracer
    application does not call it or persist its dense arrays; production RF
    jobs use rooted facts or generic CSR snapshots.

    Calls rapidtrees' interned-snapshot pyfunction, which builds an
    InternedSnapshots representation under the hood: every distinct
    bipartition is assigned a u32 ID and the RF inner loop runs on integers
    instead of multi-word bitset memcmps. At 1000+ taxa the speedup over the
    legacy bitset path is roughly 5–10× because the working set fits in L1
    and there are no pointer-chased heap reads in the merge.

    The presence matrix is returned as a byproduct: shape (n_trees, n_bip),
    uint8 with ``presence[i, j] == 1`` iff tree ``i`` contains bipartition
    ``j``. Bipartition columns are in ascending Bitset order so the same
    tree set always produces the same matrix.

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

    return (
        names_out,
        rf_matrix,
        presence,
        list(leaf_names),
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
) -> tuple[List[str], np.ndarray, SparseSnapshot]:
    """Compute RF distances and retain compact CSR clade-presence rows.

    This additive wrapper leaves the established dense wrapper unchanged.  It
    validates RapidTrees' versioned sparse payload and keeps both membership
    rows and clade bitsets packed.
    """
    if map_indices is None:
        map_indices = [0] * len(names)
    endpoint = getattr(
        rapidtrees,
        "pairwise_rf_with_sparse_snapshots_from_newick_iter",
        None,
    )
    if endpoint is None:
        raise RuntimeError(
            "the installed RapidTrees does not provide sparse snapshots; "
            "install RapidTrees 0.9.1 or newer"
        )
    (
        names_out,
        rf_bytes,
        leaf_names,
        n_clades,
        clade_bytes,
        sparse,
    ) = endpoint(
        names,
        newick_iter,
        translate_maps,
        map_indices,
        rooted=rooted,
        progress=progress,
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
    """Compute rooted RF distances and retain compact MrHIPSTR facts.

    This is an additive alternative to
    :func:`rf_distance_with_snapshots_from_newick_iter`. It calls RapidTrees'
    rooted-only version-3 endpoint, keeps clade membership sparse, keeps clade
    bitsets packed, and returns heights and directly observed binary splits
    without reparsing the source Newicks in Python.

    The existing dense-snapshot function and its callers are intentionally
    unchanged. Trees supplied here must be rooted, strictly binary, and have
    explicit finite branch lengths on every non-root edge.

    Returns:
        ``(names, rf_matrix, rooted_facts)`` where ``rf_matrix`` is a copied
        ``uint32`` square matrix and ``rooted_facts`` is a validated
        :class:`RootedFactsSnapshot`.
    """
    if map_indices is None:
        map_indices = [0] * len(names)
    endpoint = getattr(
        rapidtrees,
        "pairwise_rf_with_rooted_facts_from_newick_iter",
        None,
    )
    if endpoint is None:
        raise RuntimeError(
            "the installed RapidTrees does not provide rooted facts; "
            "install RapidTrees 0.9.1 or newer"
        )

    (
        names_out,
        rf_bytes,
        leaf_names,
        n_clades,
        clade_bytes,
        facts,
    ) = endpoint(
        names,
        newick_iter,
        translate_maps,
        map_indices,
        progress=progress,
    )
    n_trees = len(names_out)
    expected_rf_bytes = n_trees * n_trees * np.dtype(np.uint32).itemsize
    if len(rf_bytes) != expected_rf_bytes:
        raise ValueError(
            "RapidTrees returned an invalid rooted-facts RF buffer: "
            f"{len(rf_bytes)} bytes != {expected_rf_bytes}"
        )
    rf_matrix = (
        np.frombuffer(rf_bytes, dtype=np.uint32)
        .reshape(n_trees, n_trees)
        .copy()
    )
    rooted_facts = decode_rooted_facts_snapshot(
        tree_names=names_out,
        leaf_names=leaf_names,
        n_clades=n_clades,
        clade_bytes=clade_bytes,
        facts=facts,
    )
    return list(names_out), rf_matrix, rooted_facts
