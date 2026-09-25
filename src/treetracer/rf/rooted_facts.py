"""Typed decoding for RapidTrees' compact rooted-facts result.

The established dense-snapshot path remains in :mod:`treetracer.rf.rf`.
This module represents the additional version-3 payload returned by
``pairwise_rf_with_rooted_facts_from_newick_iter`` without expanding either
the sparse tree-by-clade rows or the packed clade bitsets.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


ROOTED_FACTS_FORMAT_VERSION = 3
_REQUIRED_FACT_KEYS = frozenset(
    {
        "format_version",
        "root_column",
        "nodes_per_tree",
        "splits_per_tree",
        "n_observed_splits",
        "clade_columns",
        "node_heights",
        "root_heights",
        "split_ids",
        "split_table",
    }
)
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
    """Decoded version-3 rooted facts aligned to RapidTrees clade columns.

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


def _integer_field(facts: Mapping[str, object], key: str) -> int:
    value = facts[key]
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"rooted-facts {key!r} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"rooted-facts {key!r} must be non-negative")
    return value


def _decode_buffer(
    value: object,
    *,
    key: str,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"rooted-facts {key!r} must be a byte buffer")
    expected_values = math.prod(shape)
    expected_bytes = expected_values * dtype.itemsize
    if len(value) != expected_bytes:
        raise ValueError(
            f"rooted-facts {key!r} has {len(value)} bytes; "
            f"expected {expected_bytes} for shape {shape}"
        )
    result = np.frombuffer(value, dtype=dtype).reshape(shape).copy()
    result.setflags(write=False)
    return result


def decode_rooted_facts_snapshot(
    *,
    tree_names: Sequence[object],
    leaf_names: Sequence[object],
    n_clades: int,
    clade_bytes: object,
    facts: Mapping[str, object],
) -> RootedFactsSnapshot:
    """Decode and validate RapidTrees' version-3 rooted-facts sidecar.

    The returned arrays own their memory and are read-only. Native-endian
    ``uint32`` and ``float32`` decoding deliberately mirrors RapidTrees' public
    wire contract.
    """
    if not isinstance(facts, Mapping):
        raise TypeError("rooted_facts must be a mapping")
    missing = _REQUIRED_FACT_KEYS.difference(facts)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"rooted-facts payload is missing: {missing_list}")

    format_version = _integer_field(facts, "format_version")
    if format_version != ROOTED_FACTS_FORMAT_VERSION:
        raise ValueError(
            "unsupported rooted-facts format version "
            f"{format_version}; expected {ROOTED_FACTS_FORMAT_VERSION}"
        )

    names = tuple(str(name) for name in tree_names)
    leaves = tuple(str(name) for name in leaf_names)
    if len(names) < 2:
        raise ValueError("rooted facts require at least two trees")
    if len(leaves) < 2:
        raise ValueError("rooted facts require at least two taxa")

    if isinstance(n_clades, (bool, np.bool_)) or not isinstance(
        n_clades, (int, np.integer)
    ):
        raise TypeError("n_clades must be an integer")
    n_clades = int(n_clades)
    if n_clades <= 0:
        raise ValueError("n_clades must be positive")

    root_column = _integer_field(facts, "root_column")
    nodes_per_tree = _integer_field(facts, "nodes_per_tree")
    splits_per_tree = _integer_field(facts, "splits_per_tree")
    n_observed_splits = _integer_field(facts, "n_observed_splits")
    expected_nodes = 2 * len(leaves) - 2
    expected_splits = len(leaves) - 1
    if root_column != n_clades:
        raise ValueError(
            "rooted-facts root_column must equal n_clades: "
            f"{root_column} != {n_clades}"
        )
    if nodes_per_tree != expected_nodes:
        raise ValueError(
            "rooted-facts nodes_per_tree disagrees with the taxon count: "
            f"{nodes_per_tree} != {expected_nodes}"
        )
    if splits_per_tree != expected_splits:
        raise ValueError(
            "rooted-facts splits_per_tree disagrees with the taxon count: "
            f"{splits_per_tree} != {expected_splits}"
        )
    if n_observed_splits == 0:
        raise ValueError("rooted facts contain no observed splits")

    n_trees = len(names)
    bytes_per_clade = (len(leaves) + 7) // 8
    packed_clades = _decode_buffer(
        clade_bytes,
        key="clade_bytes",
        dtype=np.dtype(np.uint8),
        shape=(n_clades, bytes_per_clade),
    )
    clade_columns = _decode_buffer(
        facts["clade_columns"],
        key="clade_columns",
        dtype=np.dtype(np.uint32),
        shape=(n_trees, nodes_per_tree),
    )
    node_heights = _decode_buffer(
        facts["node_heights"],
        key="node_heights",
        dtype=np.dtype(np.float32),
        shape=(n_trees, nodes_per_tree),
    )
    root_heights = _decode_buffer(
        facts["root_heights"],
        key="root_heights",
        dtype=np.dtype(np.float32),
        shape=(n_trees,),
    )
    split_ids = _decode_buffer(
        facts["split_ids"],
        key="split_ids",
        dtype=np.dtype(np.uint32),
        shape=(n_trees, splits_per_tree),
    )
    split_table = _decode_buffer(
        facts["split_table"],
        key="split_table",
        dtype=np.dtype(np.uint32),
        shape=(n_observed_splits, 3),
    )

    if np.any(clade_columns >= n_clades):
        raise ValueError("rooted-facts clade row references an invalid column")
    if nodes_per_tree > 1 and np.any(
        clade_columns[:, 1:] <= clade_columns[:, :-1]
    ):
        raise ValueError("rooted-facts clade rows must be sorted and unique")
    if not np.all(np.isfinite(node_heights)):
        raise ValueError("rooted-facts node heights must be finite")
    if not np.all(np.isfinite(root_heights)):
        raise ValueError("rooted-facts root heights must be finite")
    if np.any(split_ids >= n_observed_splits):
        raise ValueError("rooted-facts split row references an invalid split ID")
    if splits_per_tree > 1 and np.any(split_ids[:, 1:] <= split_ids[:, :-1]):
        raise ValueError("rooted-facts split-ID rows must be sorted and unique")

    parents = split_table[:, 0]
    left_children = split_table[:, 1]
    right_children = split_table[:, 2]
    if np.any(parents > root_column):
        raise ValueError("rooted-facts split parent is outside the clade catalog")
    if np.any(left_children >= root_column) or np.any(
        right_children >= root_column
    ):
        raise ValueError("rooted-facts split child is outside the clade catalog")
    if np.any(left_children >= right_children):
        raise ValueError("rooted-facts split children must be strictly ordered")

    remainder = len(leaves) % 8
    if remainder:
        padding_mask = np.uint8(0xFF ^ ((1 << remainder) - 1))
        if np.any(packed_clades[:, -1] & padding_mask):
            raise ValueError("packed clades contain nonzero taxon-padding bits")

    return RootedFactsSnapshot(
        tree_names=names,
        leaf_names=leaves,
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
