"""MrHIPSTR's rooted facts, derived in NumPy from RapidTrees' rooted CSR rows.

``pairwise_rf_with_snapshots_from_newick_iter(rooted=True, lengths=True)`` returns which clades each tree holds and the branch length above each. A rooted tree is fixed by its clades, so the parent/child splits and node heights MrHIPSTR needs follow from those rows without reparsing the source Newicks or asking RapidTrees for anything MrHIPSTR-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .sparse_snapshots import SparseSnapshot


ROOTED_FACTS_FORMAT_VERSION = 3
_NPZ_KEYS = frozenset(
    {
        "rooted_facts_format_version",
        "rooted_facts_tree_names",
        "rooted_facts_leaf_names",
        "rooted_facts_n_clades",
        "rooted_facts_root_column",
        "rooted_facts_nodes_per_tree",
        "rooted_facts_splits_per_tree",
        "rooted_facts_n_observed_splits",
        "rooted_facts_packed_clades",
        "rooted_facts_clade_columns",
        "rooted_facts_node_heights",
        "rooted_facts_root_heights",
        "rooted_facts_split_ids",
        "rooted_facts_split_table",
    }
)


@dataclass(frozen=True, slots=True)
class RootedFactsSnapshot:
    """Version-3 rooted facts aligned to RapidTrees clade columns.

    ``packed_clades`` keeps the compact little-bit-order encoding returned by
    RapidTrees. The implicit all-taxa root is not a packed clade and is instead
    represented by ``root_column == n_clades`` in ``split_table``. Node and
    root heights remain compact read-only ``float32`` arrays; consensus
    aggregation promotes individual values into ``float64`` accumulators.
    """

    tree_names: tuple[str, ...]
    leaf_names: tuple[str, ...]
    n_clades: int
    packed_clades: np.ndarray
    root_column: int
    nodes_per_tree: int
    splits_per_tree: int
    n_observed_splits: int
    clade_columns: np.ndarray
    node_heights: np.ndarray
    root_heights: np.ndarray
    split_ids: np.ndarray
    split_table: np.ndarray

    @property
    def n_trees(self) -> int:
        return len(self.tree_names)

    @property
    def n_taxa(self) -> int:
        return len(self.leaf_names)


# Seed for the per-taxon hash weights. Any fixed value works: the hashes only locate a right child, and every split is then checked on the exact bitmasks.
_HASH_SEED = 20251001

# Trees are derived a block at a time, about this many clade entries per block, so the temporaries stay near a hundred MB however many trees there are.
_BLOCK_ENTRIES = 1 << 20


@dataclass(frozen=True, slots=True)
class _Catalog:
    """Per-clade lookups every block shares: taxon count, lowest taxon and an XOR hash of its taxa (the root's appended last), plus the hashes sorted for search."""

    sizes: np.ndarray
    first: np.ndarray
    hashes: np.ndarray
    by_hash: np.ndarray
    sorted_hashes: np.ndarray


def _catalog(packed_clades: np.ndarray, n_taxa: int) -> _Catalog:
    n_clades, n_bytes = packed_clades.shape
    byte_bits = np.unpackbits(
        np.arange(256, dtype=np.uint8)[:, None], axis=1, bitorder="little"
    ).astype(bool)
    # 16-bit keys let the row sorts run as radix sorts.
    small = np.uint16 if n_taxa < 2 ** 16 else np.uint32
    lowest_bit = np.argmax(byte_bits, axis=1)
    rng = np.random.default_rng(_HASH_SEED)
    weights = np.frombuffer(rng.bytes(8 * 8 * n_bytes), dtype=np.uint64)
    # byte_hash[j, v] is the XOR of the weights of the taxa that byte value v sets at byte j.
    byte_hash = np.bitwise_xor.reduce(
        np.where(byte_bits[None], weights.reshape(n_bytes, 1, 8), np.uint64(0)),
        axis=2,
    )

    sizes = np.bitwise_count(packed_clades).sum(axis=1, dtype=small)
    first_byte = np.argmax(packed_clades != 0, axis=1)
    first = (8 * first_byte + lowest_bit[packed_clades[np.arange(n_clades), first_byte]]).astype(small)
    hashes = np.zeros(n_clades + 1, dtype=np.uint64)
    for j in range(n_bytes):
        hashes[:-1] ^= byte_hash[j, packed_clades[:, j]]
    hashes[-1] = np.bitwise_xor.reduce(weights[:n_taxa])
    by_hash = np.argsort(hashes[:-1])
    return _Catalog(sizes, first, hashes, by_hash, hashes[by_hash])


def rooted_facts_from_sparse(sparse: SparseSnapshot) -> RootedFactsSnapshot:
    """Derive MrHIPSTR's rooted facts from rooted CSR rows that carry branch lengths.

    Every tree must be strictly binary, which a rooted row shows by holding exactly ``2L - 2`` clades. Within a tree, the clades whose lowest taxon is ``m`` lie on the path from tip ``m`` upwards, so sorting a row by (lowest taxon, size) puts each internal clade straight after its left child, the child holding its lowest taxon. The right child is the clade whose taxa are the parent's minus the left child's: an XOR hash names it, it sits at the end of its own lowest taxon's run, and the exact bitmasks confirm every split.

    Heights follow the convention of the RapidTrees rooted-facts prototype and of TreeTracer's source-tree reader:

    ```text
    root_height = max(root-to-tip distance)
    node_height = root_height - root-to-node distance
    ```

    Root distances are summed in ``float64`` from the root down, one addition per edge, and the heights rounded to ``float32``.

    Raises:
        ValueError: If the rows are unrooted or carry no lengths, a tree is not strictly binary, or a height is not a finite ``float32``. The caller keeps the plain sparse snapshot in that case.
    """
    if not sparse.rooted or sparse.lengths is None:
        raise ValueError("rooted facts need rooted clade rows with branch lengths")
    n_trees, n_taxa, n_clades = sparse.n_trees, sparse.n_taxa, sparse.n_clades
    width = 2 * n_taxa - 2
    n_splits = n_taxa - 1
    if np.any(np.diff(sparse.row_offsets) != width):
        raise ValueError(
            f"rooted facts need strictly binary trees, with {width} clades for {n_taxa} taxa"
        )
    if (n_clades + 1) * n_clades ** 2 >= 2 ** 64:
        raise ValueError("rooted facts support fewer than 2.6 million clades")
    columns = sparse.column_indices.reshape(n_trees, width)
    lengths = sparse.lengths.reshape(n_trees, width)
    catalog = _catalog(sparse.packed_clades, n_taxa)

    node_heights = np.empty((n_trees, width), dtype=np.float32)
    root_heights = np.empty(n_trees, dtype=np.float32)
    split_keys = np.empty((n_trees, n_splits), dtype=np.uint64)
    step = max(1, _BLOCK_ENTRIES // width)
    blocks = [slice(start, start + step) for start in range(0, n_trees, step)]
    for rows in blocks:
        _derive_block(columns[rows], lengths[rows], catalog, n_clades, node_heights[rows], root_heights[rows], split_keys[rows])
    if not (np.all(np.isfinite(node_heights)) and np.all(np.isfinite(root_heights))):
        raise ValueError("rooted facts need finite float32 heights")

    unique_keys = np.unique(split_keys)
    split_ids = np.empty((n_trees, n_splits), dtype=np.uint32)
    for rows in blocks:
        # Each row's keys are sorted, so its split IDs come out sorted too.
        split_ids[rows] = np.searchsorted(unique_keys, split_keys[rows])
    split_table = np.column_stack(
        [unique_keys // n_clades // n_clades, unique_keys // n_clades % n_clades, unique_keys % n_clades]
    ).astype(np.uint32)
    _check_split_table(sparse.packed_clades, n_taxa, split_table)

    for value in (columns, node_heights, root_heights, split_ids, split_table):
        value.setflags(write=False)
    return RootedFactsSnapshot(
        tree_names=sparse.tree_names,
        leaf_names=sparse.leaf_names,
        n_clades=n_clades,
        packed_clades=sparse.packed_clades,
        root_column=n_clades,
        nodes_per_tree=width,
        splits_per_tree=n_splits,
        n_observed_splits=len(split_table),
        clade_columns=columns,
        node_heights=node_heights,
        root_heights=root_heights,
        split_ids=split_ids,
        split_table=split_table,
    )


def _derive_block(
    columns: np.ndarray,
    lengths: np.ndarray,
    catalog: _Catalog,
    n_clades: int,
    node_heights: np.ndarray,
    root_heights: np.ndarray,
    split_keys: np.ndarray,
) -> None:
    """Fill one block of trees' heights and per-row sorted split keys."""
    # Positions are flat indices into the block's (n_trees, width) arrays.
    n_trees, width = columns.shape
    n_taxa = width // 2 + 1
    tree = np.arange(n_trees)[:, None]
    starts = tree * width
    cols = columns.ravel()
    sizes = catalog.sizes[cols]
    first = catalog.first[cols]

    # Two stable radix passes sort each row by (lowest taxon, size).
    by_size = np.argsort(sizes.reshape(n_trees, width), axis=1, kind="stable") + starts
    by_first = np.argsort(first[by_size], axis=1, kind="stable") + starts
    order = by_size.ravel()[by_first.ravel()]
    sorted_first = first[order].reshape(n_trees, width)
    # Every taxon starts a run at its own tip, so each row ends exactly one run per taxon, in taxon order.
    run_end = np.ones((n_trees, width), dtype=bool)
    run_end[:, :-1] = sorted_first[:, 1:] != sorted_first[:, :-1]
    run_end = np.flatnonzero(run_end).reshape(n_trees, n_taxa)

    # An internal clade follows its left child; the root's left child ends taxon 0's run.
    internal = np.flatnonzero(sizes[order] > 1).reshape(n_trees, n_taxa - 2)
    sentinel = n_trees * width
    parent_pos = np.hstack([order[internal], np.full((n_trees, 1), sentinel)])
    left_pos = order[np.hstack([internal - 1, run_end[:, :1]])]
    parent_cols = np.hstack([cols[parent_pos[:, :-1]], np.full((n_trees, 1), n_clades, dtype=cols.dtype)])
    left_cols = cols[left_pos]
    right_hashes = catalog.hashes[parent_cols] ^ catalog.hashes[left_cols]
    found = np.minimum(np.searchsorted(catalog.sorted_hashes, right_hashes), n_clades - 1)
    right_cols = catalog.by_hash[found]
    right_pos = order[run_end.ravel()[catalog.first[right_cols] + tree * n_taxa]]
    if np.any(catalog.hashes[right_cols] != right_hashes) or np.any(cols[right_pos] != right_cols):
        raise ValueError("rooted clade rows do not form binary trees")

    # Walk each tree from its root down, one level per step, so every distance is its parent's plus its own length, as when recursing.
    left_of = np.full(sentinel, -1, dtype=np.intp)
    right_of = np.full(sentinel, -1, dtype=np.intp)
    left_of[parent_pos[:, :-1]] = left_pos[:, :-1]
    right_of[parent_pos[:, :-1]] = right_pos[:, :-1]
    flat_lengths = lengths.ravel()
    distances = np.full(sentinel, np.nan)
    frontier = np.concatenate([left_pos[:, -1], right_pos[:, -1]])
    above = np.zeros(len(frontier))
    visited = 0
    for _ in range(width):
        if not len(frontier):
            break
        here = above + flat_lengths[frontier]
        distances[frontier] = here
        visited += len(frontier)
        inner = left_of[frontier] >= 0
        frontier, here = frontier[inner], here[inner]
        frontier = np.concatenate([left_of[frontier], right_of[frontier]])
        above = np.concatenate([here, here])
    if visited != sentinel or np.isnan(distances).any():
        raise ValueError("rooted clade rows do not form binary trees")
    distances = distances.reshape(n_trees, width)
    root_distance = np.where((sizes == 1).reshape(n_trees, width), distances, -np.inf).max(axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        node_heights[:] = root_distance[:, None] - distances
        root_heights[:] = root_distance

    lo = np.minimum(left_cols, right_cols).astype(np.uint64)
    hi = np.maximum(left_cols, right_cols).astype(np.uint64)
    keys = (parent_cols.astype(np.uint64) * n_clades + lo) * n_clades + hi
    keys.sort(axis=1)
    split_keys[:] = keys


def _check_split_table(
    packed_clades: np.ndarray,
    n_taxa: int,
    split_table: np.ndarray,
    chunk_rows: int = 1 << 16,
) -> None:
    """Confirm on the bitmasks that every split's children are disjoint and make up its parent."""
    root = np.packbits(np.ones(n_taxa, dtype=bool), bitorder="little")
    with_root = np.vstack([packed_clades, root])
    for start in range(0, len(split_table), chunk_rows):
        chunk = split_table[start : start + chunk_rows]
        parents, left, right = (with_root[chunk[:, k]] for k in range(3))
        if np.any(left & right) or np.any((left | right) != parents):
            raise ValueError("rooted clade rows do not form binary trees")


def rooted_facts_npz_payload(
    rooted_facts: RootedFactsSnapshot,
) -> dict[str, np.ndarray]:
    """Return non-object arrays suitable for addition to a snapshot ``.npz``."""
    if not isinstance(rooted_facts, RootedFactsSnapshot):
        raise TypeError("rooted_facts must be a RootedFactsSnapshot")
    return {
        "rooted_facts_format_version": np.asarray(
            ROOTED_FACTS_FORMAT_VERSION,
            dtype=np.uint8,
        ),
        "rooted_facts_tree_names": np.asarray(rooted_facts.tree_names),
        "rooted_facts_leaf_names": np.asarray(rooted_facts.leaf_names),
        "rooted_facts_n_clades": np.asarray(
            rooted_facts.n_clades,
            dtype=np.uint64,
        ),
        "rooted_facts_root_column": np.asarray(
            rooted_facts.root_column,
            dtype=np.uint64,
        ),
        "rooted_facts_nodes_per_tree": np.asarray(
            rooted_facts.nodes_per_tree,
            dtype=np.uint64,
        ),
        "rooted_facts_splits_per_tree": np.asarray(
            rooted_facts.splits_per_tree,
            dtype=np.uint64,
        ),
        "rooted_facts_n_observed_splits": np.asarray(
            rooted_facts.n_observed_splits,
            dtype=np.uint64,
        ),
        "rooted_facts_packed_clades": rooted_facts.packed_clades,
        "rooted_facts_clade_columns": rooted_facts.clade_columns,
        "rooted_facts_node_heights": rooted_facts.node_heights,
        "rooted_facts_root_heights": rooted_facts.root_heights,
        "rooted_facts_split_ids": rooted_facts.split_ids,
        "rooted_facts_split_table": rooted_facts.split_table,
    }


def snapshot_has_rooted_facts(snapshot: object) -> bool:
    """Return whether a persisted snapshot advertises rooted-facts arrays."""
    files = getattr(snapshot, "files", None)
    if files is None:
        keys = getattr(snapshot, "keys", None)
        if keys is None:
            return False
        files = keys()
    return "rooted_facts_format_version" in files


def _npz_scalar(snapshot: object, key: str) -> int:
    value = np.asarray(snapshot[key])
    if value.shape != () or value.dtype.kind not in "iu":
        raise ValueError(f"persisted {key!r} must be an integer scalar")
    result = int(value)
    if result < 0:
        raise ValueError(f"persisted {key!r} must be non-negative")
    return result


def _npz_array(
    snapshot: object,
    key: str,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> np.ndarray:
    value = np.asarray(snapshot[key])
    if value.dtype != dtype:
        raise TypeError(
            f"persisted {key!r} has dtype {value.dtype}; expected {dtype}"
        )
    if value.shape != shape:
        raise ValueError(
            f"persisted {key!r} has shape {value.shape}; expected {shape}"
        )
    if not value.flags.c_contiguous:
        value = np.ascontiguousarray(value)
    value.setflags(write=False)
    return value


def rooted_facts_from_npz(snapshot: object) -> RootedFactsSnapshot:
    """Load a validated rooted-facts sidecar from an open NumPy snapshot.

    The loader deliberately requires the format marker and every companion
    array. A partially written sidecar is an error rather than a reason to
    silently fall back to reparsing source trees.
    """
    files = getattr(snapshot, "files", None)
    if files is None:
        keys = getattr(snapshot, "keys", None)
        if keys is None:
            raise TypeError("snapshot must expose .files or .keys()")
        files = keys()
    missing = _NPZ_KEYS.difference(files)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"persisted rooted facts are missing: {missing_list}")

    format_version = _npz_scalar(snapshot, "rooted_facts_format_version")
    if format_version != ROOTED_FACTS_FORMAT_VERSION:
        raise ValueError(
            "unsupported persisted rooted-facts format version "
            f"{format_version}; expected {ROOTED_FACTS_FORMAT_VERSION}"
        )
    tree_names_array = np.asarray(snapshot["rooted_facts_tree_names"])
    leaf_names_array = np.asarray(snapshot["rooted_facts_leaf_names"])
    if tree_names_array.ndim != 1 or tree_names_array.dtype.kind not in "US":
        raise ValueError("persisted rooted-facts tree names must be text")
    if leaf_names_array.ndim != 1 or leaf_names_array.dtype.kind not in "US":
        raise ValueError("persisted rooted-facts leaf names must be text")
    tree_names = tuple(str(value) for value in tree_names_array)
    leaf_names = tuple(str(value) for value in leaf_names_array)
    if len(tree_names) < 2 or len(leaf_names) < 2:
        raise ValueError("persisted rooted facts require two trees and two taxa")

    n_clades = _npz_scalar(snapshot, "rooted_facts_n_clades")
    root_column = _npz_scalar(snapshot, "rooted_facts_root_column")
    nodes_per_tree = _npz_scalar(snapshot, "rooted_facts_nodes_per_tree")
    splits_per_tree = _npz_scalar(snapshot, "rooted_facts_splits_per_tree")
    n_observed_splits = _npz_scalar(
        snapshot,
        "rooted_facts_n_observed_splits",
    )
    expected_nodes = 2 * len(leaf_names) - 2
    expected_splits = len(leaf_names) - 1
    if n_clades <= 0 or root_column != n_clades:
        raise ValueError("persisted rooted facts have an invalid root column")
    if nodes_per_tree != expected_nodes or splits_per_tree != expected_splits:
        raise ValueError("persisted rooted-facts widths disagree with leaf names")
    if n_observed_splits <= 0:
        raise ValueError("persisted rooted facts contain no observed splits")

    n_trees = len(tree_names)
    packed_clades = _npz_array(
        snapshot,
        "rooted_facts_packed_clades",
        dtype=np.dtype(np.uint8),
        shape=(n_clades, (len(leaf_names) + 7) // 8),
    )
    clade_columns = _npz_array(
        snapshot,
        "rooted_facts_clade_columns",
        dtype=np.dtype(np.uint32),
        shape=(n_trees, nodes_per_tree),
    )
    node_heights = _npz_array(
        snapshot,
        "rooted_facts_node_heights",
        dtype=np.dtype(np.float32),
        shape=(n_trees, nodes_per_tree),
    )
    root_heights = _npz_array(
        snapshot,
        "rooted_facts_root_heights",
        dtype=np.dtype(np.float32),
        shape=(n_trees,),
    )
    split_ids = _npz_array(
        snapshot,
        "rooted_facts_split_ids",
        dtype=np.dtype(np.uint32),
        shape=(n_trees, splits_per_tree),
    )
    split_table = _npz_array(
        snapshot,
        "rooted_facts_split_table",
        dtype=np.dtype(np.uint32),
        shape=(n_observed_splits, 3),
    )

    if np.any(clade_columns >= n_clades) or (
        nodes_per_tree > 1
        and np.any(clade_columns[:, 1:] <= clade_columns[:, :-1])
    ):
        raise ValueError("persisted rooted-facts clade rows are invalid")
    if not np.all(np.isfinite(node_heights)) or not np.all(
        np.isfinite(root_heights)
    ):
        raise ValueError("persisted rooted-facts heights must be finite")
    if np.any(split_ids >= n_observed_splits) or (
        splits_per_tree > 1
        and np.any(split_ids[:, 1:] <= split_ids[:, :-1])
    ):
        raise ValueError("persisted rooted-facts split-ID rows are invalid")
    if (
        np.any(split_table[:, 0] > root_column)
        or np.any(split_table[:, 1] >= root_column)
        or np.any(split_table[:, 2] >= root_column)
        or np.any(split_table[:, 1] >= split_table[:, 2])
    ):
        raise ValueError("persisted rooted-facts split table is invalid")

    remainder = len(leaf_names) % 8
    if remainder:
        padding_mask = np.uint8(0xFF ^ ((1 << remainder) - 1))
        if np.any(packed_clades[:, -1] & padding_mask):
            raise ValueError("persisted packed clades have nonzero padding bits")

    return RootedFactsSnapshot(
        tree_names=tree_names,
        leaf_names=leaf_names,
        n_clades=n_clades,
        packed_clades=packed_clades,
        root_column=root_column,
        nodes_per_tree=nodes_per_tree,
        splits_per_tree=splits_per_tree,
        n_observed_splits=n_observed_splits,
        clade_columns=clade_columns,
        node_heights=node_heights,
        root_heights=root_heights,
        split_ids=split_ids,
        split_table=split_table,
    )
