"""Core data structures and algorithms for MrHIPSTR summary trees.

This module is deliberately independent of Dash and worker-process state. It
decodes RapidTrees' rooted-clade snapshot and ingests source Newicks into the
compact sufficient statistics used by the MrHIPSTR optimizer.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..clade_freq.layout import Node, parse_newick


_COUNT_CHUNK_ROWS = 64


@dataclass(frozen=True, slots=True)
class SelectedCladeCounts:
    """Clade counts over one non-empty selection of snapshot rows."""

    counts: np.ndarray
    active_columns: np.ndarray
    n_trees: int


@dataclass(frozen=True, slots=True)
class RootedCladeCatalog:
    """Active rooted clades decoded from one RapidTrees snapshot.

    ``clade_by_column`` and ``column_by_clade`` contain only real snapshot
    columns. ``clades`` additionally contains ``root_bits``, the implicit
    all-taxa root omitted by RapidTrees. Clades are ordered by cardinality and
    then by integer value so later dynamic-programming passes are deterministic.
    """

    leaf_names: tuple[str, ...]
    root_bits: int
    clades: tuple[int, ...]
    clade_by_column: dict[int, int]
    column_by_clade: dict[int, int]

    @property
    def n_taxa(self) -> int:
        return len(self.leaf_names)


@dataclass(frozen=True, slots=True)
class SourceTreeRecord:
    """One raw source Newick and the Translate map belonging to its file."""

    newick: str
    translate: Mapping[str, str] | None = None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class SourceTreeSummary:
    """Streaming aggregates extracted from selected source trees."""

    observed_splits: dict[int, tuple[tuple[int, int], ...]]
    height_sums: dict[int, float]
    observation_counts: dict[int, int]
    n_trees: int

    def mean_height(self, clade_bits: int) -> float:
        """Return the exact-clade arithmetic mean height."""
        return self.height_sums[clade_bits] / self.observation_counts[clade_bits]


def _normalise_snapshot_taxon_name(value: object) -> str:
    """Remove NEXUS quoting retained in RapidTrees ``leaf_names``."""
    if isinstance(value, (bytes, np.bytes_)):
        name = bytes(value).decode("utf-8")
    else:
        name = str(value)
    if len(name) >= 2 and name[0] == name[-1] and name[0] in ("'", '"'):
        quote = name[0]
        name = name[1:-1].replace(quote + quote, quote)
    return name


def count_selected_clades(
    presence: np.ndarray,
    selected_rows: Sequence[int] | np.ndarray,
    *,
    chunk_rows: int = _COUNT_CHUNK_ROWS,
) -> SelectedCladeCounts:
    """Count snapshot clades over selected rows using bounded memory.

    The function never constructs ``presence[selected_rows]`` for the complete
    selection. At most ``chunk_rows`` advanced-indexed rows are materialized at
    once, bounding the temporary buffer to approximately
    ``chunk_rows * n_columns * presence.itemsize`` bytes.

    Args:
        presence: RapidTrees binary ``(n_trees, n_columns)`` matrix.
        selected_rows: Unique snapshot row IDs in the posterior selection.
        chunk_rows: Maximum number of selected rows copied per summation block.

    Returns:
        Full per-column counts, the sorted IDs of columns with nonzero counts,
        and the selected-tree count. Returned arrays are read-only.
    """
    matrix = np.asarray(presence)
    if matrix.ndim != 2:
        raise ValueError(
            "presence must be a two-dimensional (n_trees, n_columns) matrix"
        )
    if matrix.dtype != np.uint8 and matrix.dtype != np.bool_:
        raise TypeError(
            "presence must have uint8 or bool dtype; "
            f"got {matrix.dtype}"
        )
    if (
        isinstance(chunk_rows, (bool, np.bool_))
        or not isinstance(chunk_rows, (int, np.integer))
        or int(chunk_rows) <= 0
    ):
        raise ValueError("chunk_rows must be a positive integer")
    chunk_rows = int(chunk_rows)

    rows = np.asarray(selected_rows)
    if rows.ndim != 1:
        raise ValueError("selected_rows must be a one-dimensional sequence")
    if rows.size == 0:
        raise ValueError("selected_rows is empty")
    if rows.dtype.kind not in "iu":
        raise TypeError("selected_rows must contain integer row IDs")
    rows = rows.astype(np.intp, copy=False)
    if np.any(rows < 0):
        raise IndexError("selected snapshot row IDs must be non-negative")
    if np.any(rows >= matrix.shape[0]):
        bad_row = int(rows[rows >= matrix.shape[0]][0])
        raise IndexError(
            f"selected snapshot row {bad_row} is outside a "
            f"{matrix.shape[0]}-row presence matrix"
        )
    if np.unique(rows).size != rows.size:
        raise ValueError("selected_rows contains duplicate row IDs")

    rows = np.sort(rows)
    counts = np.zeros(matrix.shape[1], dtype=np.int64)
    for start in range(0, rows.size, chunk_rows):
        block = matrix[rows[start:start + chunk_rows]]
        if np.any((block != 0) & (block != 1)):
            raise ValueError("selected presence rows must contain only 0 or 1")
        counts += block.sum(axis=0, dtype=np.int64)

    active_columns = np.flatnonzero(counts).astype(np.intp, copy=False)
    counts.setflags(write=False)
    active_columns.setflags(write=False)
    return SelectedCladeCounts(
        counts=counts,
        active_columns=active_columns,
        n_trees=int(rows.size),
    )


def decode_rooted_clade_catalog(
    *,
    bipartition_bits: np.ndarray,
    leaf_names: Sequence[object],
    active_columns: Sequence[int] | np.ndarray,
) -> RootedCladeCatalog:
    """Decode active rooted snapshot columns into Python integer bitsets.

    Args:
        bipartition_bits: RapidTrees ``(n_columns, n_taxa)`` binary matrix.
            Despite the persisted name, rows are descendant-taxon sets when
            the RF matrix was computed in rooted mode.
        leaf_names: Taxon name for each bit position.
        active_columns: Snapshot columns with a nonzero count in the selected
            posterior sample. Only these rows are decoded.

    Returns:
        A validated catalog containing both column mappings and a synthesized
        all-taxa root.

    Raises:
        TypeError: if the bit matrix or column IDs have incompatible dtypes.
        ValueError: if shapes, names, bits, clades, or singleton coverage are
            incompatible with a rooted RapidTrees snapshot.
        IndexError: if an active column lies outside the snapshot matrix.
    """
    bits = np.asarray(bipartition_bits)
    if bits.ndim != 2:
        raise ValueError(
            "bipartition_bits must be a two-dimensional "
            "(n_columns, n_taxa) matrix"
        )
    if bits.dtype != np.uint8 and bits.dtype != np.bool_:
        raise TypeError(
            "bipartition_bits must have uint8 or bool dtype; "
            f"got {bits.dtype}"
        )

    names_array = np.asarray(leaf_names)
    if names_array.ndim != 1:
        raise ValueError("leaf_names must be a one-dimensional sequence")
    if bits.shape[1] != names_array.shape[0]:
        raise ValueError(
            "bipartition_bits width does not match leaf_names: "
            f"{bits.shape[1]} != {names_array.shape[0]}"
        )
    if names_array.size < 2:
        raise ValueError("a rooted clade catalog requires at least two taxa")

    names = tuple(_normalise_snapshot_taxon_name(name) for name in names_array)
    if any(not name for name in names):
        raise ValueError("leaf_names contains an empty taxon name")
    if len(set(names)) != len(names):
        raise ValueError("leaf_names contains duplicate normalized taxon names")

    columns_array = np.asarray(active_columns)
    if columns_array.ndim != 1:
        raise ValueError("active_columns must be a one-dimensional sequence")
    if columns_array.size == 0:
        raise ValueError("active_columns is empty")
    if columns_array.dtype.kind not in "iu":
        raise TypeError("active_columns must contain integer column IDs")
    columns_array = columns_array.astype(np.intp, copy=False)
    if np.any(columns_array < 0):
        raise IndexError("active snapshot column IDs must be non-negative")
    if np.any(columns_array >= bits.shape[0]):
        bad_column = int(columns_array[columns_array >= bits.shape[0]][0])
        raise IndexError(
            f"active snapshot column {bad_column} is outside a "
            f"{bits.shape[0]}-column catalog"
        )
    if np.unique(columns_array).size != columns_array.size:
        raise ValueError("active_columns contains duplicate column IDs")

    columns_array = np.sort(columns_array)
    selected_bits = bits[columns_array]
    if np.any((selected_bits != 0) & (selected_bits != 1)):
        raise ValueError("active bipartition_bits rows must contain only 0 or 1")

    packed_rows = np.packbits(
        selected_bits.astype(np.uint8, copy=False),
        axis=1,
        bitorder="little",
    )
    decoded = tuple(
        int.from_bytes(row.tobytes(), byteorder="little")
        for row in packed_rows
    )

    n_taxa = len(names)
    root_bits = (1 << n_taxa) - 1
    clade_by_column: dict[int, int] = {}
    column_by_clade: dict[int, int] = {}
    for column, clade_bits in zip(columns_array.tolist(), decoded, strict=True):
        column = int(column)
        if clade_bits == 0:
            raise ValueError(f"snapshot column {column} encodes an empty clade")
        if clade_bits == root_bits:
            raise ValueError(
                f"snapshot column {column} encodes the all-taxa root; "
                "the rooted RapidTrees contract requires an implicit root"
            )
        previous = column_by_clade.get(clade_bits)
        if previous is not None:
            raise ValueError(
                f"snapshot columns {previous} and {column} encode the same clade"
            )
        clade_by_column[column] = clade_bits
        column_by_clade[clade_bits] = column

    missing_singletons = [
        names[index]
        for index in range(n_taxa)
        if (1 << index) not in column_by_clade
    ]
    if missing_singletons:
        sample = ", ".join(repr(name) for name in missing_singletons[:3])
        suffix = "…" if len(missing_singletons) > 3 else ""
        raise ValueError(
            "rooted snapshot is missing active singleton clades for "
            f"{sample}{suffix}"
        )

    clades = tuple(
        sorted(
            (*column_by_clade, root_bits),
            key=lambda clade: (clade.bit_count(), clade),
        )
    )
    return RootedCladeCatalog(
        leaf_names=names,
        root_bits=root_bits,
        clades=clades,
        clade_by_column=clade_by_column,
        column_by_clade=column_by_clade,
    )


def _validate_ingestion_inputs(
    catalog: RootedCladeCatalog,
    snapshot_counts: SelectedCladeCounts,
) -> None:
    counts = np.asarray(snapshot_counts.counts)
    active_columns = np.asarray(snapshot_counts.active_columns)
    if counts.ndim != 1 or counts.dtype.kind not in "iu":
        raise ValueError(
            "snapshot clade counts must be a one-dimensional integer array"
        )
    if active_columns.ndim != 1 or active_columns.dtype.kind not in "iu":
        raise ValueError(
            "active snapshot columns must be a one-dimensional integer array"
        )
    if snapshot_counts.n_trees <= 0:
        raise ValueError("snapshot tree count must be positive")
    if np.any(counts < 0) or np.any(counts > snapshot_counts.n_trees):
        raise ValueError("snapshot clade counts must lie between zero and n_trees")

    counted_active = np.flatnonzero(counts)
    if not np.array_equal(active_columns, counted_active):
        raise ValueError("active snapshot columns do not match nonzero clade counts")
    catalog_columns = np.fromiter(
        catalog.clade_by_column,
        dtype=np.intp,
        count=len(catalog.clade_by_column),
    )
    if not np.array_equal(active_columns, catalog_columns):
        raise ValueError("snapshot counts and rooted clade catalog use different columns")


def _tree_context(record: SourceTreeRecord, tree_number: int) -> str:
    return record.name or f"selected tree {tree_number}"


def _extract_tree_facts(
    record: SourceTreeRecord,
    *,
    tree_number: int,
    catalog: RootedCladeCatalog,
    taxon_index: Mapping[str, int],
) -> tuple[dict[int, float], dict[int, tuple[int, int]]]:
    """Parse and validate one tree, returning local heights and child splits."""
    context = _tree_context(record, tree_number)
    try:
        root = parse_newick(
            record.newick,
            translate=record.translate,
            require_branch_lengths=True,
        )
    except ValueError as exc:
        raise ValueError(f"{context}: {exc}") from exc

    nodes: list[Node] = []
    tips: list[Node] = []
    distances: dict[int, float] = {}
    internal_count = 0
    stack: list[tuple[Node, float]] = [(root, 0.0)]
    while stack:
        node, distance = stack.pop()
        if not math.isfinite(distance):
            raise ValueError(f"{context}: non-finite cumulative root distance")
        nodes.append(node)
        distances[id(node)] = distance
        if node.children:
            if len(node.children) != 2:
                raise ValueError(
                    f"{context}: every internal node must have exactly two "
                    f"children; found {len(node.children)}"
                )
            internal_count += 1
            for child in reversed(node.children):
                child_distance = distance + child.length
                if not math.isfinite(child_distance):
                    raise ValueError(
                        f"{context}: branch lengths overflow cumulative distance"
                    )
                stack.append((child, child_distance))
        else:
            tips.append(node)

    tip_counts = Counter(tip.name for tip in tips)
    duplicate_taxa = sorted(
        name for name, count in tip_counts.items() if count != 1
    )
    unexpected_taxa = sorted(set(tip_counts) - set(taxon_index))
    missing_taxa = [name for name in catalog.leaf_names if name not in tip_counts]
    if duplicate_taxa or unexpected_taxa or missing_taxa:
        details = []
        if duplicate_taxa:
            details.append(f"duplicate taxa={duplicate_taxa[:3]!r}")
        if unexpected_taxa:
            details.append(f"unexpected taxa={unexpected_taxa[:3]!r}")
        if missing_taxa:
            details.append(f"missing taxa={missing_taxa[:3]!r}")
        raise ValueError(f"{context}: taxon mismatch ({'; '.join(details)})")

    expected_internal_count = catalog.n_taxa - 1
    if internal_count != expected_internal_count:
        raise ValueError(
            f"{context}: expected {expected_internal_count} internal partitions; "
            f"found {internal_count}"
        )

    root_height = max(distances[id(tip)] for tip in tips)
    clade_by_node: dict[int, int] = {}
    tree_heights: dict[int, float] = {}
    tree_splits: dict[int, tuple[int, int]] = {}
    for node in reversed(nodes):
        if not node.children:
            clade_bits = 1 << taxon_index[node.name]
        else:
            left_bits = clade_by_node[id(node.children[0])]
            right_bits = clade_by_node[id(node.children[1])]
            if left_bits & right_bits:
                raise ValueError(f"{context}: child clades overlap")
            clade_bits = left_bits | right_bits
            tree_splits[clade_bits] = tuple(sorted((left_bits, right_bits)))

        if clade_bits in tree_heights:
            raise ValueError(f"{context}: the same clade occurs more than once")
        if node is not root and clade_bits not in catalog.column_by_clade:
            raise ValueError(
                f"{context}: parsed clade 0x{clade_bits:x} has no active "
                "RapidTrees snapshot column"
            )
        height = root_height - distances[id(node)]
        if not math.isfinite(height):
            raise ValueError(f"{context}: calculated a non-finite node height")
        clade_by_node[id(node)] = clade_bits
        tree_heights[clade_bits] = height

    if clade_by_node[id(root)] != catalog.root_bits:
        raise ValueError(f"{context}: root does not contain the complete taxon set")
    if len(tree_splits) != expected_internal_count:
        raise ValueError(
            f"{context}: expected {expected_internal_count} direct child splits; "
            f"found {len(tree_splits)}"
        )
    return tree_heights, tree_splits


def ingest_source_trees(
    records: Iterable[SourceTreeRecord],
    *,
    catalog: RootedCladeCatalog,
    snapshot_counts: SelectedCladeCounts,
) -> SourceTreeSummary:
    """Stream selected Newicks into splits and exact-clade height aggregates.

    Each record is parsed once and discarded after its local facts have been
    merged. The completed aggregate is cross-checked against every active
    RapidTrees snapshot count before it is returned.
    """
    _validate_ingestion_inputs(catalog, snapshot_counts)
    taxon_index = {name: index for index, name in enumerate(catalog.leaf_names)}

    split_sets: dict[int, set[tuple[int, int]]] = {}
    height_sums: dict[int, float] = {}
    observation_counts: dict[int, int] = {}
    n_ingested = 0

    for tree_number, record in enumerate(records, start=1):
        if tree_number > snapshot_counts.n_trees:
            raise ValueError(
                "source tree stream contains more trees than the snapshot selection"
            )
        if not isinstance(record, SourceTreeRecord):
            raise TypeError("source tree stream must yield SourceTreeRecord objects")
        tree_heights, tree_splits = _extract_tree_facts(
            record,
            tree_number=tree_number,
            catalog=catalog,
            taxon_index=taxon_index,
        )
        for clade_bits, height in tree_heights.items():
            height_sums[clade_bits] = height_sums.get(clade_bits, 0.0) + height
            observation_counts[clade_bits] = (
                observation_counts.get(clade_bits, 0) + 1
            )
        for parent_bits, child_pair in tree_splits.items():
            split_sets.setdefault(parent_bits, set()).add(child_pair)
        n_ingested = tree_number

    if n_ingested != snapshot_counts.n_trees:
        raise ValueError(
            f"source tree stream yielded {n_ingested} trees; "
            f"snapshot selection contains {snapshot_counts.n_trees}"
        )

    mismatches = []
    for clade_bits, column in catalog.column_by_clade.items():
        parsed_count = observation_counts.get(clade_bits, 0)
        snapshot_count = int(snapshot_counts.counts[column])
        if parsed_count != snapshot_count:
            mismatches.append(
                f"column {column} clade 0x{clade_bits:x}: "
                f"parsed={parsed_count}, snapshot={snapshot_count}"
            )
    root_count = observation_counts.get(catalog.root_bits, 0)
    if root_count != snapshot_counts.n_trees:
        mismatches.append(
            f"implicit root: parsed={root_count}, snapshot={snapshot_counts.n_trees}"
        )
    if mismatches:
        sample = "; ".join(mismatches[:3])
        suffix = "; …" if len(mismatches) > 3 else ""
        raise ValueError(f"parsed clade counts disagree with snapshot: {sample}{suffix}")

    ordered_clades = [
        clade_bits
        for clade_bits in catalog.clades
        if clade_bits in observation_counts
    ]
    observed_splits = {
        clade_bits: tuple(sorted(split_sets[clade_bits]))
        for clade_bits in ordered_clades
        if clade_bits in split_sets
    }
    return SourceTreeSummary(
        observed_splits=observed_splits,
        height_sums={
            clade_bits: height_sums[clade_bits] for clade_bits in ordered_clades
        },
        observation_counts={
            clade_bits: observation_counts[clade_bits]
            for clade_bits in ordered_clades
        },
        n_trees=n_ingested,
    )
