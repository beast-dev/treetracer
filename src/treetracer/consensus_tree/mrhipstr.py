"""Core data structures and algorithms for MrHIPSTR summary trees.

This module is deliberately independent of Dash and worker-process state. It
decodes RapidTrees' rooted-clade snapshot, ingests source Newicks into compact
sufficient statistics, selects HIPSTR/MrHIPSTR topologies with the supplied
Java formulation, and serializes mean-height MrHIPSTR trees as NEXUS.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..clade_freq.layout import Node, parse_newick


_COUNT_CHUNK_ROWS = 64
MAJORITY_RULE_REWARD = 1e10
NEGATIVE_BRANCH_TOLERANCE = 1e-12
_SAFE_NEXUS_TOKEN = re.compile(r"^[A-Za-z0-9_.+\-]+$")


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


@dataclass(frozen=True, slots=True)
class HipstrTopology:
    """A selected HIPSTR/MrHIPSTR topology and its score metadata.

    ``selected_splits`` contains one canonical child pair for every selected
    internal clade, including the implicit root. ``selected_clades`` contains
    every node clade in cardinality/integer order. The bonus-augmented
    ``objective_score`` is retained for reference validation only; it is not a
    posterior probability or log posterior.
    """

    root_bits: int
    selected_clades: tuple[int, ...]
    selected_splits: dict[int, tuple[int, int]]
    cols_in_consensus_tree: frozenset[int]
    objective_score: float
    log_clade_credibility: float
    majority_clade_count: int
    majority_rule: bool


@dataclass(frozen=True, slots=True)
class MrHipstrTree:
    """A mean-height MrHIPSTR tree and its serialized NEXUS document.

    Mapping keys are descendant-taxon clade bitsets. ``branch_lengths`` omits
    the root because the root has no parent edge. Tiny negative lengths caused
    by floating-point roundoff are written as zero; material negative lengths
    are retained and counted.
    """

    topology: HipstrTopology
    mean_heights: dict[int, float]
    clade_frequencies: dict[int, float]
    branch_lengths: dict[int, float]
    newick: str
    tree_line: str
    nexus_bytes: bytes
    negative_branch_count: int
    minimum_branch_length: float | None


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
        raise ValueError(
            "snapshot counts and rooted clade catalog use different columns"
        )


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
        raise ValueError(
            f"parsed clade counts disagree with snapshot: {sample}{suffix}"
        )

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


def _validate_topology_inputs(
    catalog: RootedCladeCatalog,
    snapshot_counts: SelectedCladeCounts,
    source_summary: SourceTreeSummary,
) -> dict[int, tuple[tuple[int, int], ...]]:
    """Validate and canonicalize the clade-split DAG used by HIPSTR."""
    _validate_ingestion_inputs(catalog, snapshot_counts)
    if not isinstance(source_summary, SourceTreeSummary):
        raise TypeError("source_summary must be a SourceTreeSummary")
    if source_summary.n_trees != snapshot_counts.n_trees:
        raise ValueError(
            "source summary and snapshot counts use different tree counts: "
            f"{source_summary.n_trees} != {snapshot_counts.n_trees}"
        )

    expected_root = (1 << catalog.n_taxa) - 1
    if catalog.root_bits != expected_root:
        raise ValueError("rooted clade catalog has an invalid all-taxa root")

    clade_set = set(catalog.clades)
    if len(clade_set) != len(catalog.clades):
        raise ValueError("rooted clade catalog contains duplicate clades")
    if catalog.root_bits not in clade_set:
        raise ValueError("rooted clade catalog is missing its implicit root")
    if catalog.root_bits in catalog.column_by_clade:
        raise ValueError("implicit root must not have a snapshot column")
    if set(catalog.column_by_clade) != clade_set - {catalog.root_bits}:
        raise ValueError("rooted clade catalog has incomplete column mappings")
    for clade_bits, column in catalog.column_by_clade.items():
        if catalog.clade_by_column.get(column) != clade_bits:
            raise ValueError("rooted clade catalog column mappings disagree")

    observed_count_clades = set(source_summary.observation_counts)
    if observed_count_clades != clade_set:
        missing = clade_set - observed_count_clades
        extra = observed_count_clades - clade_set
        raise ValueError(
            "source summary observation counts do not match the rooted clade "
            f"catalog (missing={len(missing)}, extra={len(extra)})"
        )
    for clade_bits, column in catalog.column_by_clade.items():
        parsed_count = source_summary.observation_counts[clade_bits]
        snapshot_count = int(snapshot_counts.counts[column])
        if parsed_count != snapshot_count:
            raise ValueError(
                "source summary clade count disagrees with snapshot for "
                f"clade 0x{clade_bits:x}: {parsed_count} != {snapshot_count}"
            )
        if clade_bits.bit_count() == 1 and snapshot_count != snapshot_counts.n_trees:
            raise ValueError(
                f"singleton clade 0x{clade_bits:x} is not present in every tree"
            )
    if (
        source_summary.observation_counts[catalog.root_bits]
        != snapshot_counts.n_trees
    ):
        raise ValueError("implicit root is not present in every source tree")

    extra_split_parents = set(source_summary.observed_splits) - clade_set
    if extra_split_parents:
        raise ValueError("source summary contains splits for unknown parent clades")

    canonical_splits: dict[int, tuple[tuple[int, int], ...]] = {}
    for parent_bits in sorted(clade_set, key=lambda clade: (clade.bit_count(), clade)):
        raw_splits = source_summary.observed_splits.get(parent_bits, ())
        if parent_bits.bit_count() == 1:
            if raw_splits:
                raise ValueError(
                    f"singleton clade 0x{parent_bits:x} cannot have child splits"
                )
            continue
        if not raw_splits:
            raise ValueError(
                f"non-singleton clade 0x{parent_bits:x} has no observed child split"
            )

        parent_splits: set[tuple[int, int]] = set()
        for raw_pair in raw_splits:
            try:
                left_bits, right_bits = raw_pair
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"clade 0x{parent_bits:x} has a malformed child split"
                ) from exc
            if (
                isinstance(left_bits, (bool, np.bool_))
                or isinstance(right_bits, (bool, np.bool_))
                or not isinstance(left_bits, (int, np.integer))
                or not isinstance(right_bits, (int, np.integer))
            ):
                raise ValueError(
                    f"clade 0x{parent_bits:x} has non-integer child clades"
                )
            left_bits = int(left_bits)
            right_bits = int(right_bits)
            if left_bits <= 0 or right_bits <= 0:
                raise ValueError("child clades must be non-empty")
            if left_bits & right_bits:
                raise ValueError(
                    f"child clades overlap under parent 0x{parent_bits:x}"
                )
            if left_bits | right_bits != parent_bits:
                raise ValueError(
                    f"child clades do not partition parent 0x{parent_bits:x}"
                )
            if left_bits not in clade_set or right_bits not in clade_set:
                raise ValueError(
                    f"parent 0x{parent_bits:x} references an unknown child clade"
                )
            parent_splits.add(tuple(sorted((left_bits, right_bits))))
        canonical_splits[parent_bits] = tuple(sorted(parent_splits))

    return canonical_splits


def compute_hipstr_topology(
    *,
    catalog: RootedCladeCatalog,
    snapshot_counts: SelectedCladeCounts,
    source_summary: SourceTreeSummary,
    majority_rule: bool,
) -> HipstrTopology:
    """Select a HIPSTR topology with the supplied Java recurrence.

    When ``majority_rule`` is true this is MrHIPSTR: every reached
    non-singleton clade with credibility strictly greater than ``0.5`` adds
    ``MAJORITY_RULE_REWARD`` to the dynamic-programming score. When false, the
    same observed-split search runs without that reward and yields HIPSTR.

    Exact objective ties are resolved by the lexicographically smallest
    canonical child-bitset pair. This makes the selected topology reproducible
    without changing the Java formulation or its optimum score.
    """
    if not isinstance(majority_rule, bool):
        raise TypeError("majority_rule must be a bool")
    observed_splits = _validate_topology_inputs(
        catalog,
        snapshot_counts,
        source_summary,
    )

    frequencies: dict[int, float] = {catalog.root_bits: 1.0}
    for clade_bits, column in catalog.column_by_clade.items():
        frequency = int(snapshot_counts.counts[column]) / snapshot_counts.n_trees
        if not math.isfinite(frequency) or not 0.0 < frequency <= 1.0:
            raise ValueError(
                f"clade 0x{clade_bits:x} has invalid frequency {frequency!r}"
            )
        frequencies[clade_bits] = frequency

    ordered_clades = tuple(
        sorted(catalog.clades, key=lambda clade: (clade.bit_count(), clade))
    )
    scores: dict[int, float] = {}
    backpointers: dict[int, tuple[int, int]] = {}
    for clade_bits in ordered_clades:
        if clade_bits.bit_count() == 1:
            # The Java implementation uses log(1) for a tip whenever it is a
            # child, so singleton subtrees contribute exactly zero.
            scores[clade_bits] = 0.0
            continue

        best_subtree_score = -math.inf
        best_split: tuple[int, int] | None = None
        for child_pair in observed_splits[clade_bits]:
            left_bits, right_bits = child_pair
            if left_bits not in scores or right_bits not in scores:
                raise ValueError(
                    f"child score unavailable while processing clade "
                    f"0x{clade_bits:x}"
                )
            candidate_score = scores[left_bits] + scores[right_bits]
            if (
                candidate_score > best_subtree_score
                or (
                    candidate_score == best_subtree_score
                    and (best_split is None or child_pair < best_split)
                )
            ):
                best_subtree_score = candidate_score
                best_split = child_pair

        if best_split is None:
            raise ValueError(
                f"non-singleton clade 0x{clade_bits:x} is unreachable"
            )
        frequency = frequencies[clade_bits]
        clade_score = math.log(frequency)
        if majority_rule and frequency > 0.5:
            clade_score += MAJORITY_RULE_REWARD
        score = clade_score + best_subtree_score
        if not math.isfinite(score):
            raise ValueError(
                f"dynamic-programming score is non-finite for clade "
                f"0x{clade_bits:x}"
            )
        scores[clade_bits] = score
        backpointers[clade_bits] = best_split

    selected: set[int] = set()
    selected_splits: dict[int, tuple[int, int]] = {}
    stack = [catalog.root_bits]
    while stack:
        clade_bits = stack.pop()
        if clade_bits in selected:
            raise ValueError(
                f"backtracking reached clade 0x{clade_bits:x} more than once"
            )
        selected.add(clade_bits)
        if clade_bits.bit_count() == 1:
            continue
        child_pair = backpointers.get(clade_bits)
        if child_pair is None:
            raise ValueError(
                f"selected clade 0x{clade_bits:x} has no backpointer"
            )
        selected_splits[clade_bits] = child_pair
        # Push the larger child first so the smaller canonical child is visited
        # first. The returned mappings are sorted again below for stable output.
        stack.append(child_pair[1])
        stack.append(child_pair[0])

    selected_singletons = {
        clade_bits for clade_bits in selected if clade_bits.bit_count() == 1
    }
    expected_singletons = {1 << index for index in range(catalog.n_taxa)}
    if selected_singletons != expected_singletons:
        raise ValueError("selected topology does not contain every taxon exactly once")
    if len(selected) != 2 * catalog.n_taxa - 1:
        raise ValueError("selected topology is not a fully resolved binary tree")
    if len(selected_splits) != catalog.n_taxa - 1:
        raise ValueError("selected topology has an invalid number of internal splits")

    selected_clades = tuple(
        sorted(selected, key=lambda clade: (clade.bit_count(), clade))
    )
    selected_splits = {
        clade_bits: selected_splits[clade_bits]
        for clade_bits in selected_clades
        if clade_bits in selected_splits
    }
    cols_in_consensus_tree = frozenset(
        catalog.column_by_clade[clade_bits]
        for clade_bits in selected_clades
        if clade_bits != catalog.root_bits
    )
    log_clade_credibility = math.fsum(
        math.log(frequencies[clade_bits])
        for clade_bits in selected_clades
        if clade_bits.bit_count() > 1
    )
    majority_clade_count = sum(
        1
        for clade_bits in selected_clades
        if (
            1 < clade_bits.bit_count() < catalog.n_taxa
            and frequencies[clade_bits] > 0.5
        )
    )

    return HipstrTopology(
        root_bits=catalog.root_bits,
        selected_clades=selected_clades,
        selected_splits=selected_splits,
        cols_in_consensus_tree=cols_in_consensus_tree,
        objective_score=scores[catalog.root_bits],
        log_clade_credibility=log_clade_credibility,
        majority_clade_count=majority_clade_count,
        majority_rule=majority_rule,
    )


def compute_mrhipstr_topology(
    *,
    catalog: RootedCladeCatalog,
    snapshot_counts: SelectedCladeCounts,
    source_summary: SourceTreeSummary,
) -> HipstrTopology:
    """Select an MrHIPSTR topology using the Java ``1E10`` formulation."""
    return compute_hipstr_topology(
        catalog=catalog,
        snapshot_counts=snapshot_counts,
        source_summary=source_summary,
        majority_rule=True,
    )


def _validate_selected_topology(
    *,
    topology: HipstrTopology,
    catalog: RootedCladeCatalog,
    snapshot_counts: SelectedCladeCounts,
    source_summary: SourceTreeSummary,
) -> dict[int, int]:
    """Validate a selected topology and return each non-root clade's parent."""
    if not isinstance(topology, HipstrTopology):
        raise TypeError("topology must be a HipstrTopology")
    if not topology.majority_rule:
        raise ValueError("mean-height MrHIPSTR export requires majority_rule=True")
    observed_splits = _validate_topology_inputs(
        catalog,
        snapshot_counts,
        source_summary,
    )
    if topology.root_bits != catalog.root_bits:
        raise ValueError("selected topology and rooted clade catalog disagree")

    selected = set(topology.selected_clades)
    if len(selected) != len(topology.selected_clades):
        raise ValueError("selected topology contains duplicate clades")
    if catalog.root_bits not in selected:
        raise ValueError("selected topology is missing the all-taxa root")
    unknown = selected - set(catalog.clades)
    if unknown:
        raise ValueError("selected topology contains clades outside the catalog")

    selected_internal = {
        clade_bits for clade_bits in selected if clade_bits.bit_count() > 1
    }
    if set(topology.selected_splits) != selected_internal:
        raise ValueError(
            "selected topology splits do not match its internal clades"
        )

    visited: set[int] = set()
    parent_by_clade: dict[int, int] = {}
    stack = [catalog.root_bits]
    while stack:
        parent_bits = stack.pop()
        if parent_bits in visited:
            raise ValueError(
                f"selected topology reaches clade 0x{parent_bits:x} more than once"
            )
        visited.add(parent_bits)
        if parent_bits.bit_count() == 1:
            continue

        child_pair = topology.selected_splits[parent_bits]
        if child_pair not in observed_splits[parent_bits]:
            raise ValueError(
                f"selected split for clade 0x{parent_bits:x} was not observed"
            )
        left_bits, right_bits = child_pair
        if left_bits not in selected or right_bits not in selected:
            raise ValueError("selected split references an unselected child clade")
        for child_bits in child_pair:
            if child_bits in parent_by_clade:
                raise ValueError(
                    f"selected clade 0x{child_bits:x} has more than one parent"
                )
            parent_by_clade[child_bits] = parent_bits
        stack.append(right_bits)
        stack.append(left_bits)

    if visited != selected:
        raise ValueError("selected topology contains clades unreachable from the root")
    expected_columns = frozenset(
        catalog.column_by_clade[clade_bits]
        for clade_bits in selected
        if clade_bits != catalog.root_bits
    )
    if topology.cols_in_consensus_tree != expected_columns:
        raise ValueError("selected topology snapshot columns are inconsistent")
    return parent_by_clade


def _quote_nexus_label(value: str) -> str:
    """Quote one Newick/NEXUS label with doubled embedded apostrophes."""
    if not value:
        raise ValueError("NEXUS labels must not be empty")
    return "'" + value.replace("'", "''") + "'"


def _format_nexus_token(value: str) -> str:
    if not value:
        raise ValueError("NEXUS tokens must not be empty")
    if _SAFE_NEXUS_TOKEN.fullmatch(value):
        return value
    return _quote_nexus_label(value)


def _resolve_output_tip_labels(
    catalog: RootedCladeCatalog,
    canonical_translate: Mapping[object, object] | None,
) -> tuple[dict[int, str], tuple[tuple[str, str], ...]]:
    """Return emitted tip labels and a canonical-order Translate table."""
    if canonical_translate is not None and not isinstance(
        canonical_translate,
        Mapping,
    ):
        raise TypeError("canonical_translate must be a mapping or None")
    if not canonical_translate:
        return (
            {
                1 << index: _quote_nexus_label(taxon)
                for index, taxon in enumerate(catalog.leaf_names)
            },
            (),
        )

    token_by_taxon: dict[str, str] = {}
    seen_tokens: set[str] = set()
    for raw_token, raw_taxon in canonical_translate.items():
        token = str(raw_token)
        taxon = _normalise_snapshot_taxon_name(raw_taxon)
        if not token:
            raise ValueError("canonical Translate map contains an empty token")
        if token in seen_tokens:
            raise ValueError("canonical Translate map contains duplicate tokens")
        if taxon in token_by_taxon:
            raise ValueError(
                "canonical Translate map assigns multiple tokens to taxon "
                f"{taxon!r}"
            )
        seen_tokens.add(token)
        token_by_taxon[taxon] = token

    expected_taxa = set(catalog.leaf_names)
    translated_taxa = set(token_by_taxon)
    if translated_taxa != expected_taxa:
        missing = sorted(expected_taxa - translated_taxa)
        extra = sorted(translated_taxa - expected_taxa)
        raise ValueError(
            "canonical Translate map does not match the rooted clade catalog "
            f"(missing={missing[:3]!r}, extra={extra[:3]!r})"
        )

    entries = tuple(
        (token_by_taxon[taxon], taxon) for taxon in catalog.leaf_names
    )
    labels = {
        1 << index: _format_nexus_token(token_by_taxon[taxon])
        for index, taxon in enumerate(catalog.leaf_names)
    }
    return labels, entries


def _format_finite_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"cannot serialize non-finite value {value!r}")
    if value == 0.0:
        return "0"
    return format(value, ".17g")


def _new_nexus_preamble(
    translate_entries: Sequence[tuple[str, str]],
) -> bytes:
    lines = ["#NEXUS", "", "Begin trees;"]
    if translate_entries:
        lines.append("    Translate")
        last_index = len(translate_entries) - 1
        for index, (token, taxon) in enumerate(translate_entries):
            separator = "," if index != last_index else ""
            lines.append(
                f"        {_format_nexus_token(token)} "
                f"{_quote_nexus_label(taxon)}{separator}"
            )
        lines.append("    ;")
    return ("\n".join(lines) + "\n").encode("utf-8")


def serialize_mrhipstr_tree(
    *,
    topology: HipstrTopology,
    catalog: RootedCladeCatalog,
    snapshot_counts: SelectedCladeCounts,
    source_summary: SourceTreeSummary,
    canonical_translate: Mapping[object, object] | None = None,
    nexus_preamble: bytes | None = None,
    tree_name: str = "MrHIPSTR",
) -> MrHipstrTree:
    """Apply exact-clade mean heights and serialize an MrHIPSTR NEXUS tree.

    ``canonical_translate`` maps emitted tip tokens to taxon names. When it is
    omitted, the Newick uses quoted taxon names directly. ``nexus_preamble``
    may supply an existing canonical preamble ending inside a ``Begin trees``
    block; otherwise a complete minimal preamble is generated.
    """
    parent_by_clade = _validate_selected_topology(
        topology=topology,
        catalog=catalog,
        snapshot_counts=snapshot_counts,
        source_summary=source_summary,
    )
    if not isinstance(tree_name, str):
        raise TypeError("tree_name must be a string")
    formatted_tree_name = _format_nexus_token(tree_name)
    tip_labels, translate_entries = _resolve_output_tip_labels(
        catalog,
        canonical_translate,
    )

    mean_heights: dict[int, float] = {}
    clade_frequencies: dict[int, float] = {}
    for clade_bits in topology.selected_clades:
        observation_count = source_summary.observation_counts.get(clade_bits)
        if observation_count is None or observation_count <= 0:
            raise ValueError(
                f"selected clade 0x{clade_bits:x} has no height observations"
            )
        if clade_bits not in source_summary.height_sums:
            raise ValueError(
                f"selected clade 0x{clade_bits:x} has no height sum"
            )
        mean_height = (
            source_summary.height_sums[clade_bits] / observation_count
        )
        if not math.isfinite(mean_height):
            raise ValueError(
                f"selected clade 0x{clade_bits:x} has non-finite mean height"
            )
        mean_heights[clade_bits] = mean_height
        clade_frequencies[clade_bits] = (
            observation_count / snapshot_counts.n_trees
        )

    branch_lengths: dict[int, float] = {}
    negative_branch_count = 0
    for child_bits in topology.selected_clades:
        if child_bits == catalog.root_bits:
            continue
        parent_bits = parent_by_clade[child_bits]
        branch_length = mean_heights[parent_bits] - mean_heights[child_bits]
        if not math.isfinite(branch_length):
            raise ValueError(
                f"edge to clade 0x{child_bits:x} has non-finite length"
            )
        if -NEGATIVE_BRANCH_TOLERANCE <= branch_length < 0.0:
            branch_length = 0.0
        elif branch_length < -NEGATIVE_BRANCH_TOLERANCE:
            negative_branch_count += 1
        branch_lengths[child_bits] = branch_length
    minimum_branch_length = (
        min(branch_lengths.values()) if branch_lengths else None
    )

    def node_suffix(clade_bits: int) -> str:
        annotations = []
        if clade_bits.bit_count() > 1:
            annotations.append(
                "posterior="
                + _format_finite_float(clade_frequencies[clade_bits])
            )
        annotations.append(
            "height_mean=" + _format_finite_float(mean_heights[clade_bits])
        )
        suffix = "[&" + ",".join(annotations) + "]"
        if clade_bits != catalog.root_bits:
            suffix += ":" + _format_finite_float(branch_lengths[clade_bits])
        return suffix

    parts: list[str] = []
    emit_stack: list[tuple[str, int | str]] = [
        ("node", catalog.root_bits)
    ]
    while emit_stack:
        action, value = emit_stack.pop()
        if action == "text":
            parts.append(str(value))
            continue

        clade_bits = int(value)
        if clade_bits.bit_count() == 1:
            parts.append(tip_labels[clade_bits] + node_suffix(clade_bits))
            continue

        left_bits, right_bits = topology.selected_splits[clade_bits]
        parts.append("(")
        emit_stack.append(("text", node_suffix(clade_bits)))
        emit_stack.append(("text", ")"))
        emit_stack.append(("node", right_bits))
        emit_stack.append(("text", ","))
        emit_stack.append(("node", left_bits))

    newick = "".join(parts) + ";"
    tree_metadata = ",".join(
        (
            "lnCladeCred="
            + _format_finite_float(topology.log_clade_credibility),
            "summaryMethod=MrHIPSTR",
            "heightMethod=mean",
            f"majorityClades={topology.majority_clade_count}",
            f"negativeBranches={negative_branch_count}",
        )
    )
    tree_line = (
        f"tree {formatted_tree_name} [&{tree_metadata}] = [&R] {newick}"
    )

    if nexus_preamble is None:
        preamble = _new_nexus_preamble(translate_entries)
    else:
        if not isinstance(nexus_preamble, bytes):
            raise TypeError("nexus_preamble must be bytes or None")
        preamble = nexus_preamble
        if not re.search(rb"(?i)\bbegin\s+trees\s*;", preamble):
            raise ValueError("nexus_preamble has no open Begin trees block")
        if preamble.rstrip().lower().endswith(b"end;"):
            raise ValueError("nexus_preamble already closes its final block")
        if not preamble.endswith(b"\n"):
            preamble += b"\n"
    nexus_bytes = preamble + tree_line.encode("utf-8") + b"\nEnd;\n"

    return MrHipstrTree(
        topology=topology,
        mean_heights=mean_heights,
        clade_frequencies=clade_frequencies,
        branch_lengths=branch_lengths,
        newick=newick,
        tree_line=tree_line,
        nexus_bytes=nexus_bytes,
        negative_branch_count=negative_branch_count,
        minimum_branch_length=minimum_branch_length,
    )
