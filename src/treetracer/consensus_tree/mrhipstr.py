"""Core data structures and algorithms for MrHIPSTR summary trees.

This module is deliberately independent of Dash and worker-process state.  The
first implementation stage decodes the rooted-clade catalog stored in a
RapidTrees snapshot into compact Python integer bitsets.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


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
