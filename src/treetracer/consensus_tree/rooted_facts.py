"""Adapt RapidTrees rooted facts to TreeTracer's MrHIPSTR core contracts.

This is a parallel ingestion path. The established source-Newick parser in
``mrhipstr.py`` remains available and unchanged; callers opt into these
helpers only when they already hold a version-2 ``RootedFactsSnapshot``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..rf.rooted_facts import RootedFactsSnapshot
from .mrhipstr import (
    RootedCladeCatalog,
    SelectedCladeCounts,
    SourceTreeSummary,
    _normalise_snapshot_taxon_name,
    _validate_ingestion_inputs,
)


_COUNT_CHUNK_ROWS = 64


@dataclass(frozen=True, slots=True)
class RootedFactsMrHipstrInputs:
    """The three existing MrHIPSTR inputs derived without parsing Newick."""

    catalog: RootedCladeCatalog
    snapshot_counts: SelectedCladeCounts
    source_summary: SourceTreeSummary


def _normalise_selected_rows(
    selected_rows: Sequence[int] | np.ndarray,
    *,
    n_trees: int,
) -> np.ndarray:
    rows = np.asarray(selected_rows)
    if rows.ndim != 1:
        raise ValueError("selected_rows must be a one-dimensional sequence")
    if rows.size == 0:
        raise ValueError("selected_rows is empty")
    if rows.dtype.kind not in "iu":
        raise TypeError("selected_rows must contain integer row IDs")
    if np.any(rows < 0):
        raise IndexError("selected rooted-facts row IDs must be non-negative")
    if np.any(rows >= n_trees):
        bad_row = int(rows[rows >= n_trees][0])
        raise IndexError(
            f"selected rooted-facts row {bad_row} is outside a "
            f"{n_trees}-row snapshot"
        )
    rows = rows.astype(np.intp, copy=False)
    if np.unique(rows).size != rows.size:
        raise ValueError("selected_rows contains duplicate row IDs")
    # Preserve caller order. It determines floating-point height summation
    # order and therefore matches the established source-record stream.
    return rows


def _normalise_chunk_rows(chunk_rows: int) -> int:
    if (
        isinstance(chunk_rows, (bool, np.bool_))
        or not isinstance(chunk_rows, (int, np.integer))
        or int(chunk_rows) <= 0
    ):
        raise ValueError("chunk_rows must be a positive integer")
    return int(chunk_rows)


def _require_snapshot(value: object) -> RootedFactsSnapshot:
    if not isinstance(value, RootedFactsSnapshot):
        raise TypeError("rooted_facts must be a RootedFactsSnapshot")
    return value


def count_selected_clades_from_rooted_facts(
    rooted_facts: RootedFactsSnapshot,
    selected_rows: Sequence[int] | np.ndarray,
    *,
    chunk_rows: int = _COUNT_CHUNK_ROWS,
) -> SelectedCladeCounts:
    """Count clades from fixed-width sparse rows without making a dense matrix."""
    snapshot = _require_snapshot(rooted_facts)
    rows = _normalise_selected_rows(selected_rows, n_trees=snapshot.n_trees)
    chunk_rows = _normalise_chunk_rows(chunk_rows)

    counts = np.zeros(snapshot.n_clades, dtype=np.int64)
    for start in range(0, rows.size, chunk_rows):
        columns = snapshot.clade_columns[rows[start : start + chunk_rows]]
        if np.any(columns >= snapshot.n_clades):
            raise ValueError("rooted-facts clade row references an invalid column")
        if columns.shape[1] > 1 and np.any(
            columns[:, 1:] <= columns[:, :-1]
        ):
            raise ValueError("rooted-facts clade rows must be sorted and unique")
        counts += np.bincount(
            columns.reshape(-1).astype(np.intp, copy=False),
            minlength=snapshot.n_clades,
        )

    active_columns = np.flatnonzero(counts).astype(np.intp, copy=False)
    counts.setflags(write=False)
    active_columns.setflags(write=False)
    return SelectedCladeCounts(
        counts=counts,
        active_columns=active_columns,
        n_trees=int(rows.size),
    )


def decode_rooted_clade_catalog_from_rooted_facts(
    rooted_facts: RootedFactsSnapshot,
    snapshot_counts: SelectedCladeCounts,
) -> RootedCladeCatalog:
    """Decode only selected packed clades and synthesize the implicit root."""
    snapshot = _require_snapshot(rooted_facts)
    counts = np.asarray(snapshot_counts.counts)
    columns = np.asarray(snapshot_counts.active_columns)
    if counts.shape != (snapshot.n_clades,):
        raise ValueError(
            "rooted-facts clade counts have the wrong width: "
            f"{counts.shape} != ({snapshot.n_clades},)"
        )
    if columns.ndim != 1 or columns.dtype.kind not in "iu":
        raise ValueError(
            "active rooted-facts columns must be a one-dimensional integer array"
        )
    if columns.size == 0:
        raise ValueError("active rooted-facts columns are empty")
    columns = columns.astype(np.intp, copy=False)
    if np.any(columns < 0) or np.any(columns >= snapshot.n_clades):
        raise IndexError("active rooted-facts column lies outside the catalog")
    if np.unique(columns).size != columns.size:
        raise ValueError("active rooted-facts columns contain duplicates")
    columns = np.sort(columns)

    names = tuple(
        _normalise_snapshot_taxon_name(name) for name in snapshot.leaf_names
    )
    if any(not name for name in names):
        raise ValueError("rooted-facts leaf names contain an empty taxon name")
    if len(set(names)) != len(names):
        raise ValueError(
            "rooted-facts leaf names contain duplicate normalized taxon names"
        )

    n_taxa = len(names)
    root_bits = (1 << n_taxa) - 1
    decoded = tuple(
        int.from_bytes(row.tobytes(), byteorder="little")
        for row in snapshot.packed_clades[columns]
    )
    clade_by_column: dict[int, int] = {}
    column_by_clade: dict[int, int] = {}
    for column, clade_bits in zip(columns.tolist(), decoded, strict=True):
        column = int(column)
        if clade_bits == 0:
            raise ValueError(f"rooted-facts column {column} encodes an empty clade")
        if clade_bits == root_bits:
            raise ValueError(
                f"rooted-facts column {column} encodes the implicit root"
            )
        previous = column_by_clade.get(clade_bits)
        if previous is not None:
            raise ValueError(
                f"rooted-facts columns {previous} and {column} encode the same clade"
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
            "rooted facts are missing active singleton clades for "
            f"{sample}{suffix}"
        )

    catalog = RootedCladeCatalog(
        leaf_names=names,
        root_bits=root_bits,
        clades=tuple(
            sorted(
                (*column_by_clade, root_bits),
                key=lambda clade: (clade.bit_count(), clade),
            )
        ),
        clade_by_column=clade_by_column,
        column_by_clade=column_by_clade,
    )
    _validate_ingestion_inputs(catalog, snapshot_counts)
    return catalog


def source_summary_from_rooted_facts(
    rooted_facts: RootedFactsSnapshot,
    selected_rows: Sequence[int] | np.ndarray,
    *,
    catalog: RootedCladeCatalog,
    snapshot_counts: SelectedCladeCounts,
    chunk_rows: int = _COUNT_CHUNK_ROWS,
) -> SourceTreeSummary:
    """Aggregate selected heights and observed splits without parsing Newick."""
    snapshot = _require_snapshot(rooted_facts)
    rows = _normalise_selected_rows(selected_rows, n_trees=snapshot.n_trees)
    chunk_rows = _normalise_chunk_rows(chunk_rows)
    _validate_ingestion_inputs(catalog, snapshot_counts)
    if rows.size != snapshot_counts.n_trees:
        raise ValueError(
            "selected rooted-facts rows and clade counts use different tree counts"
        )
    if snapshot.root_column != snapshot.n_clades:
        raise ValueError("rooted-facts root sentinel does not equal n_clades")

    actual_counts = np.zeros(snapshot.n_clades, dtype=np.int64)
    height_sums_by_column = np.zeros(snapshot.n_clades, dtype=np.float64)
    observed_split_mask = np.zeros(
        snapshot.n_observed_splits,
        dtype=np.bool_,
    )
    for start in range(0, rows.size, chunk_rows):
        row_block = rows[start : start + chunk_rows]
        columns = snapshot.clade_columns[row_block]
        heights = snapshot.node_heights[row_block]
        if np.any(columns >= snapshot.n_clades):
            raise ValueError("rooted-facts clade row references an invalid column")
        if not np.all(np.isfinite(heights)):
            raise ValueError("rooted-facts node heights contain a non-finite value")
        flat_columns = columns.reshape(-1).astype(np.intp, copy=False)
        actual_counts += np.bincount(
            flat_columns,
            minlength=snapshot.n_clades,
        )
        # Unbuffered row-major updates reproduce the source-stream addition
        # order for every exact clade.
        np.add.at(
            height_sums_by_column,
            flat_columns,
            heights.reshape(-1),
        )

        split_ids = snapshot.split_ids[row_block]
        if np.any(split_ids >= snapshot.n_observed_splits):
            raise ValueError("rooted-facts row references an invalid split ID")
        observed_split_mask[split_ids.reshape(-1)] = True

    if not np.array_equal(actual_counts, snapshot_counts.counts):
        raise ValueError(
            "selected rooted-facts clade counts disagree with snapshot_counts"
        )
    if not np.all(np.isfinite(height_sums_by_column)):
        raise ValueError("rooted-facts clade height sums are non-finite")

    root_height_sum = 0.0
    for row in rows.tolist():
        root_height = float(snapshot.root_heights[row])
        if not math.isfinite(root_height):
            raise ValueError("rooted-facts root heights contain a non-finite value")
        root_height_sum += root_height
    if not math.isfinite(root_height_sum):
        raise ValueError("rooted-facts root height sum is non-finite")

    observation_counts: dict[int, int] = {
        catalog.root_bits: int(rows.size),
    }
    height_sums: dict[int, float] = {catalog.root_bits: root_height_sum}
    for clade_bits in catalog.clades:
        if clade_bits == catalog.root_bits:
            continue
        column = catalog.column_by_clade[clade_bits]
        count = int(snapshot_counts.counts[column])
        if count <= 0:
            raise ValueError("active rooted-facts clade has no observations")
        observation_counts[clade_bits] = count
        height_sums[clade_bits] = float(height_sums_by_column[column])

    split_sets: dict[int, set[tuple[int, int]]] = {}
    for split_id in np.flatnonzero(observed_split_mask).tolist():
        parent_column, left_column, right_column = (
            int(value) for value in snapshot.split_table[split_id]
        )
        parent_bits = (
            catalog.root_bits
            if parent_column == snapshot.root_column
            else catalog.clade_by_column.get(parent_column)
        )
        left_bits = catalog.clade_by_column.get(left_column)
        right_bits = catalog.clade_by_column.get(right_column)
        if parent_bits is None or left_bits is None or right_bits is None:
            raise ValueError(
                "selected rooted-facts split references a clade absent from "
                "the selected catalog"
            )
        split_sets.setdefault(parent_bits, set()).add(
            tuple(sorted((left_bits, right_bits)))
        )

    return SourceTreeSummary(
        observed_splits={
            clade_bits: tuple(sorted(split_sets[clade_bits]))
            for clade_bits in catalog.clades
            if clade_bits in split_sets
        },
        height_sums=height_sums,
        observation_counts=observation_counts,
        n_trees=int(rows.size),
    )


def prepare_mrhipstr_inputs_from_rooted_facts(
    rooted_facts: RootedFactsSnapshot,
    selected_rows: Sequence[int] | np.ndarray,
    *,
    chunk_rows: int = _COUNT_CHUNK_ROWS,
) -> RootedFactsMrHipstrInputs:
    """Build all existing MrHIPSTR inputs from one selected facts subset."""
    snapshot_counts = count_selected_clades_from_rooted_facts(
        rooted_facts,
        selected_rows,
        chunk_rows=chunk_rows,
    )
    catalog = decode_rooted_clade_catalog_from_rooted_facts(
        rooted_facts,
        snapshot_counts,
    )
    source_summary = source_summary_from_rooted_facts(
        rooted_facts,
        selected_rows,
        catalog=catalog,
        snapshot_counts=snapshot_counts,
        chunk_rows=chunk_rows,
    )
    return RootedFactsMrHipstrInputs(
        catalog=catalog,
        snapshot_counts=snapshot_counts,
        source_summary=source_summary,
    )
