"""Adapt generic sparse RF snapshots to the established MrHIPSTR inputs.

This is the middle compatibility path between full RapidTrees rooted facts and
legacy dense snapshots.  It obtains clade frequencies and the rooted clade
catalog without densifying the CSR rows; observed splits and node heights still
come from the existing source-Newick ingestion path.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..rf.sparse_snapshots import SparseSnapshot, count_sparse_columns
from .mrhipstr import (
    RootedCladeCatalog,
    SelectedCladeCounts,
    _normalise_snapshot_taxon_name,
    _validate_ingestion_inputs,
)


def _require_sparse_snapshot(value: object) -> SparseSnapshot:
    if not isinstance(value, SparseSnapshot):
        raise TypeError("sparse_snapshot must be a SparseSnapshot")
    if not value.rooted:
        raise ValueError("MrHIPSTR requires a rooted sparse snapshot")
    return value


def count_selected_clades_from_sparse_snapshot(
    sparse_snapshot: SparseSnapshot,
    selected_rows: Sequence[int] | np.ndarray,
) -> SelectedCladeCounts:
    """Count selected generic CSR rows under the MrHIPSTR contract."""
    snapshot = _require_sparse_snapshot(sparse_snapshot)
    rows = np.asarray(selected_rows)
    if rows.ndim != 1:
        raise ValueError("selected_rows must be a one-dimensional sequence")
    if rows.size == 0:
        raise ValueError("selected_rows is empty")
    if rows.dtype.kind not in "iu":
        raise TypeError("selected_rows must contain integer row IDs")
    rows = rows.astype(np.intp, copy=False)
    if np.any(rows < 0) or np.any(rows >= snapshot.n_trees):
        raise IndexError("selected sparse row is outside the snapshot")
    if np.unique(rows).size != rows.size:
        raise ValueError("selected_rows contains duplicate row IDs")

    counts = count_sparse_columns(snapshot, rows)
    active_columns = np.flatnonzero(counts).astype(np.intp, copy=False)
    counts.setflags(write=False)
    active_columns.setflags(write=False)
    return SelectedCladeCounts(
        counts=counts,
        active_columns=active_columns,
        n_trees=int(rows.size),
    )


def decode_rooted_clade_catalog_from_sparse_snapshot(
    sparse_snapshot: SparseSnapshot,
    snapshot_counts: SelectedCladeCounts,
) -> RootedCladeCatalog:
    """Decode only active packed clades from a generic sparse snapshot."""
    snapshot = _require_sparse_snapshot(sparse_snapshot)
    counts = np.asarray(snapshot_counts.counts)
    columns = np.asarray(snapshot_counts.active_columns)
    if counts.shape != (snapshot.n_clades,):
        raise ValueError(
            "sparse clade counts have the wrong width: "
            f"{counts.shape} != ({snapshot.n_clades},)"
        )
    if columns.ndim != 1 or columns.dtype.kind not in "iu":
        raise ValueError(
            "active sparse columns must be a one-dimensional integer array"
        )
    if columns.size == 0:
        raise ValueError("active sparse columns are empty")
    columns = columns.astype(np.intp, copy=False)
    if np.any(columns < 0) or np.any(columns >= snapshot.n_clades):
        raise IndexError("active sparse column lies outside the catalog")
    if np.unique(columns).size != columns.size:
        raise ValueError("active sparse columns contain duplicates")
    columns = np.sort(columns)

    names = tuple(
        _normalise_snapshot_taxon_name(name) for name in snapshot.leaf_names
    )
    if any(not name for name in names):
        raise ValueError("sparse leaf names contain an empty taxon name")
    if len(set(names)) != len(names):
        raise ValueError(
            "sparse leaf names contain duplicate normalized taxon names"
        )

    root_bits = (1 << len(names)) - 1
    decoded = tuple(
        int.from_bytes(row.tobytes(), byteorder="little")
        for row in snapshot.packed_clades[columns]
    )
    clade_by_column: dict[int, int] = {}
    column_by_clade: dict[int, int] = {}
    for column, clade_bits in zip(columns.tolist(), decoded, strict=True):
        column = int(column)
        if clade_bits == 0:
            raise ValueError(f"sparse column {column} encodes an empty clade")
        if clade_bits == root_bits:
            raise ValueError(
                f"sparse column {column} encodes the implicit root"
            )
        previous = column_by_clade.get(clade_bits)
        if previous is not None:
            raise ValueError(
                f"sparse columns {previous} and {column} encode the same clade"
            )
        clade_by_column[column] = clade_bits
        column_by_clade[clade_bits] = column

    missing_singletons = [
        names[index]
        for index in range(len(names))
        if (1 << index) not in column_by_clade
    ]
    if missing_singletons:
        sample = ", ".join(repr(name) for name in missing_singletons[:3])
        suffix = "…" if len(missing_singletons) > 3 else ""
        raise ValueError(
            "sparse snapshot is missing active singleton clades for "
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
