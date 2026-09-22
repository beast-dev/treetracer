"""Typed decoding for RapidTrees' compact rooted-facts result.

The established dense-snapshot path remains in :mod:`treetracer.rf.rf`.
This module represents the additional version-2 payload returned by
``pairwise_rf_with_rooted_facts_from_newick_iter`` without expanding either
the sparse tree-by-clade rows or the packed clade bitsets.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


ROOTED_FACTS_FORMAT_VERSION = 2
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


@dataclass(frozen=True, slots=True)
class RootedFactsSnapshot:
    """Decoded version-2 rooted facts aligned to RapidTrees clade columns.

    ``packed_clades`` keeps the compact little-bit-order encoding returned by
    RapidTrees. The implicit all-taxa root is not a packed clade and is instead
    represented by ``root_column == n_clades`` in ``split_table``.
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
    """Decode and validate RapidTrees' version-2 rooted-facts sidecar.

    The returned arrays own their memory and are read-only. Native-endian
    ``uint32`` and ``float64`` decoding deliberately mirrors RapidTrees' public
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
        dtype=np.dtype(np.float64),
        shape=(n_trees, nodes_per_tree),
    )
    root_heights = _decode_buffer(
        facts["root_heights"],
        key="root_heights",
        dtype=np.dtype(np.float64),
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
