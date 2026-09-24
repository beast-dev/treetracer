"""Validated compact presence snapshots shared by RF consumers.

RapidTrees 0.9.1 exposes tree-by-clade membership as compressed sparse rows
instead of a dense ``uint8[n_trees, n_clades]`` matrix.  This module is an
additive adapter: it understands the generic version-1 CSR payload and the
fixed-width sparse rows already carried by version-2 rooted facts.  Legacy
dense snapshots remain supported by their existing callers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .rooted_facts import (
    ROOTED_FACTS_FORMAT_VERSION,
    RootedFactsSnapshot,
    snapshot_has_rooted_facts,
)


SPARSE_SNAPSHOT_FORMAT_VERSION = 1
SPARSE_SNAPSHOT_ENCODING = "csr"
_REQUIRED_PAYLOAD_KEYS = frozenset(
    {
        "format_version",
        "encoding",
        "n_entries",
        "row_offsets",
        "column_indices",
    }
)
_NPZ_KEYS = frozenset(
    {
        "sparse_snapshot_format_version",
        "sparse_snapshot_encoding",
        "sparse_snapshot_tree_names",
        "sparse_snapshot_leaf_names",
        "sparse_snapshot_n_clades",
        "sparse_snapshot_n_entries",
        "sparse_snapshot_rooted",
        "sparse_snapshot_packed_clades",
        "sparse_snapshot_row_offsets",
        "sparse_snapshot_column_indices",
    }
)


@dataclass(frozen=True, slots=True)
class SparseSnapshot:
    """A clade catalog plus CSR tree-to-clade membership rows."""

    tree_names: tuple[str, ...]
    leaf_names: tuple[str, ...]
    n_clades: int
    packed_clades: np.ndarray
    row_offsets: np.ndarray
    column_indices: np.ndarray
    rooted: bool

    @property
    def n_trees(self) -> int:
        return len(self.tree_names)

    @property
    def n_taxa(self) -> int:
        return len(self.leaf_names)

    @property
    def n_entries(self) -> int:
        return len(self.column_indices)

    def row_columns(self, row: int) -> np.ndarray:
        """Return the sorted clade-column IDs present in one tree."""
        if isinstance(row, (bool, np.bool_)) or not isinstance(
            row,
            (int, np.integer),
        ):
            raise TypeError("sparse snapshot row must be an integer")
        row = int(row)
        if row < 0 or row >= self.n_trees:
            raise IndexError(
                f"sparse snapshot row {row} is outside {self.n_trees} trees"
            )
        start = int(self.row_offsets[row])
        stop = int(self.row_offsets[row + 1])
        return self.column_indices[start:stop]


def _mapping_integer(payload: Mapping[str, object], key: str) -> int:
    value = payload[key]
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"sparse snapshot {key!r} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"sparse snapshot {key!r} must be non-negative")
    return result


def _decode_buffer(
    value: object,
    *,
    key: str,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"sparse snapshot {key!r} must be a byte buffer")
    expected_bytes = math.prod(shape) * dtype.itemsize
    if len(value) != expected_bytes:
        raise ValueError(
            f"sparse snapshot {key!r} has {len(value)} bytes; "
            f"expected {expected_bytes} for shape {shape}"
        )
    result = np.frombuffer(value, dtype=dtype).reshape(shape).copy()
    result.setflags(write=False)
    return result


def _validate_sparse_arrays(
    *,
    n_clades: int,
    n_taxa: int,
    packed_clades: np.ndarray,
    row_offsets: np.ndarray,
    column_indices: np.ndarray,
) -> None:
    n_entries = len(column_indices)
    if int(row_offsets[0]) != 0 or int(row_offsets[-1]) != n_entries:
        raise ValueError(
            "sparse snapshot row offsets must start at zero and end at "
            "n_entries"
        )
    if np.any(row_offsets[1:] < row_offsets[:-1]):
        raise ValueError("sparse snapshot row offsets must be nondecreasing")
    if n_entries and np.any(column_indices >= n_clades):
        raise ValueError("sparse snapshot row references an invalid clade column")

    for row in range(len(row_offsets) - 1):
        start = int(row_offsets[row])
        stop = int(row_offsets[row + 1])
        values = column_indices[start:stop]
        if len(values) > 1 and np.any(values[1:] <= values[:-1]):
            raise ValueError(
                "sparse snapshot rows must contain sorted, unique columns"
            )

    remainder = n_taxa % 8
    if remainder and n_clades:
        padding_mask = np.uint8(0xFF ^ ((1 << remainder) - 1))
        if np.any(packed_clades[:, -1] & padding_mask):
            raise ValueError(
                "sparse snapshot packed clades have nonzero padding bits"
            )


def decode_sparse_snapshot(
    *,
    tree_names: Sequence[object],
    leaf_names: Sequence[object],
    n_clades: int,
    clade_bytes: object,
    sparse: Mapping[str, object],
    rooted: bool,
) -> SparseSnapshot:
    """Decode and validate RapidTrees' version-1 sparse snapshot payload."""
    if not isinstance(sparse, Mapping):
        raise TypeError("sparse snapshot payload must be a mapping")
    missing = _REQUIRED_PAYLOAD_KEYS.difference(sparse)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"sparse snapshot payload is missing: {missing_list}")

    format_version = _mapping_integer(sparse, "format_version")
    if format_version != SPARSE_SNAPSHOT_FORMAT_VERSION:
        raise ValueError(
            "unsupported sparse snapshot format version "
            f"{format_version}; expected {SPARSE_SNAPSHOT_FORMAT_VERSION}"
        )
    encoding = sparse["encoding"]
    if encoding != SPARSE_SNAPSHOT_ENCODING:
        raise ValueError(
            f"unsupported sparse snapshot encoding {encoding!r}; "
            f"expected {SPARSE_SNAPSHOT_ENCODING!r}"
        )
    n_entries = _mapping_integer(sparse, "n_entries")

    names = tuple(str(name) for name in tree_names)
    leaves = tuple(str(name) for name in leaf_names)
    if len(names) < 2:
        raise ValueError("sparse snapshots require at least two trees")
    if len(leaves) < 2:
        raise ValueError("sparse snapshots require at least two taxa")
    if isinstance(n_clades, (bool, np.bool_)) or not isinstance(
        n_clades,
        (int, np.integer),
    ):
        raise TypeError("n_clades must be an integer")
    n_clades = int(n_clades)
    if n_clades < 0:
        raise ValueError("n_clades must be non-negative")
    if not isinstance(rooted, (bool, np.bool_)):
        raise TypeError("rooted must be a boolean")

    packed_clades = _decode_buffer(
        clade_bytes,
        key="clade_bytes",
        dtype=np.dtype(np.uint8),
        shape=(n_clades, (len(leaves) + 7) // 8),
    )
    row_offsets = _decode_buffer(
        sparse["row_offsets"],
        key="row_offsets",
        dtype=np.dtype(np.uint64),
        shape=(len(names) + 1,),
    )
    column_indices = _decode_buffer(
        sparse["column_indices"],
        key="column_indices",
        dtype=np.dtype(np.uint32),
        shape=(n_entries,),
    )
    _validate_sparse_arrays(
        n_clades=n_clades,
        n_taxa=len(leaves),
        packed_clades=packed_clades,
        row_offsets=row_offsets,
        column_indices=column_indices,
    )
    return SparseSnapshot(
        tree_names=names,
        leaf_names=leaves,
        n_clades=n_clades,
        packed_clades=packed_clades,
        row_offsets=row_offsets,
        column_indices=column_indices,
        rooted=bool(rooted),
    )


def sparse_snapshot_from_rooted_facts(
    rooted_facts: RootedFactsSnapshot,
) -> SparseSnapshot:
    """Expose rooted-facts clade rows through the common sparse contract."""
    if not isinstance(rooted_facts, RootedFactsSnapshot):
        raise TypeError("rooted_facts must be a RootedFactsSnapshot")
    width = rooted_facts.nodes_per_tree
    row_offsets = np.arange(
        0,
        (rooted_facts.n_trees + 1) * width,
        width,
        dtype=np.uint64,
    )
    row_offsets.setflags(write=False)
    column_indices = rooted_facts.clade_columns.reshape(-1)
    column_indices.setflags(write=False)
    return SparseSnapshot(
        tree_names=rooted_facts.tree_names,
        leaf_names=rooted_facts.leaf_names,
        n_clades=rooted_facts.n_clades,
        packed_clades=rooted_facts.packed_clades,
        row_offsets=row_offsets,
        column_indices=column_indices,
        rooted=True,
    )


def sparse_snapshot_npz_payload(
    sparse_snapshot: SparseSnapshot,
) -> dict[str, np.ndarray]:
    """Return generic sparse arrays suitable for an existing snapshot NPZ."""
    if not isinstance(sparse_snapshot, SparseSnapshot):
        raise TypeError("sparse_snapshot must be a SparseSnapshot")
    return {
        "sparse_snapshot_format_version": np.asarray(
            SPARSE_SNAPSHOT_FORMAT_VERSION,
            dtype=np.uint8,
        ),
        "sparse_snapshot_encoding": np.asarray(SPARSE_SNAPSHOT_ENCODING),
        "sparse_snapshot_tree_names": np.asarray(sparse_snapshot.tree_names),
        "sparse_snapshot_leaf_names": np.asarray(sparse_snapshot.leaf_names),
        "sparse_snapshot_n_clades": np.asarray(
            sparse_snapshot.n_clades,
            dtype=np.uint64,
        ),
        "sparse_snapshot_n_entries": np.asarray(
            sparse_snapshot.n_entries,
            dtype=np.uint64,
        ),
        "sparse_snapshot_rooted": np.asarray(
            sparse_snapshot.rooted,
            dtype=np.bool_,
        ),
        "sparse_snapshot_packed_clades": sparse_snapshot.packed_clades,
        "sparse_snapshot_row_offsets": sparse_snapshot.row_offsets,
        "sparse_snapshot_column_indices": sparse_snapshot.column_indices,
    }


def _snapshot_files(snapshot: object) -> set[str]:
    files = getattr(snapshot, "files", None)
    if files is None:
        keys = getattr(snapshot, "keys", None)
        if keys is None:
            raise TypeError("snapshot must expose .files or .keys()")
        files = keys()
    return set(files)


def snapshot_has_generic_sparse_snapshot(snapshot: object) -> bool:
    """Return whether an NPZ advertises the generic CSR sidecar."""
    try:
        files = _snapshot_files(snapshot)
    except TypeError:
        return False
    return "sparse_snapshot_format_version" in files


def snapshot_has_sparse_presence(snapshot: object) -> bool:
    """Return whether generic CSR or rooted facts can supply sparse rows."""
    return snapshot_has_generic_sparse_snapshot(
        snapshot
    ) or snapshot_has_rooted_facts(snapshot)


def _npz_integer(snapshot: object, key: str) -> int:
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


def _sparse_snapshot_from_rooted_facts_npz(snapshot: object) -> SparseSnapshot:
    """Load only the rooted-facts arrays needed for sparse membership.

    MCC scoring and clade comparison do not need heights, split IDs, or the
    observed-split table. Avoiding those arrays keeps their I/O and memory
    exclusive to the full MrHIPSTR rooted-facts loader.
    """
    required = {
        "rooted_facts_format_version",
        "rooted_facts_tree_names",
        "rooted_facts_leaf_names",
        "rooted_facts_n_clades",
        "rooted_facts_nodes_per_tree",
        "rooted_facts_packed_clades",
        "rooted_facts_clade_columns",
    }
    missing = required.difference(_snapshot_files(snapshot))
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(
            "persisted rooted-facts sparse rows are missing: "
            f"{missing_list}"
        )
    format_version = _npz_integer(
        snapshot,
        "rooted_facts_format_version",
    )
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
        raise ValueError("persisted sparse snapshots require two trees and taxa")

    n_clades = _npz_integer(snapshot, "rooted_facts_n_clades")
    nodes_per_tree = _npz_integer(
        snapshot,
        "rooted_facts_nodes_per_tree",
    )
    expected_nodes = 2 * len(leaf_names) - 2
    if n_clades <= 0 or nodes_per_tree != expected_nodes:
        raise ValueError(
            "persisted rooted-facts sparse widths disagree with leaf names"
        )
    packed_clades = _npz_array(
        snapshot,
        "rooted_facts_packed_clades",
        dtype=np.dtype(np.uint8),
        shape=(n_clades, (len(leaf_names) + 7) // 8),
    )
    clade_rows = _npz_array(
        snapshot,
        "rooted_facts_clade_columns",
        dtype=np.dtype(np.uint32),
        shape=(len(tree_names), nodes_per_tree),
    )
    row_offsets = np.arange(
        0,
        (len(tree_names) + 1) * nodes_per_tree,
        nodes_per_tree,
        dtype=np.uint64,
    )
    column_indices = clade_rows.reshape(-1)
    row_offsets.setflags(write=False)
    column_indices.setflags(write=False)
    _validate_sparse_arrays(
        n_clades=n_clades,
        n_taxa=len(leaf_names),
        packed_clades=packed_clades,
        row_offsets=row_offsets,
        column_indices=column_indices,
    )
    return SparseSnapshot(
        tree_names=tree_names,
        leaf_names=leaf_names,
        n_clades=n_clades,
        packed_clades=packed_clades,
        row_offsets=row_offsets,
        column_indices=column_indices,
        rooted=True,
    )


def sparse_snapshot_from_npz(snapshot: object) -> SparseSnapshot:
    """Load generic CSR, or adapt rooted facts without duplicating its arrays."""
    if not snapshot_has_generic_sparse_snapshot(snapshot):
        if snapshot_has_rooted_facts(snapshot):
            return _sparse_snapshot_from_rooted_facts_npz(snapshot)
        raise KeyError("snapshot has no sparse presence representation")

    files = _snapshot_files(snapshot)
    missing = _NPZ_KEYS.difference(files)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"persisted sparse snapshot is missing: {missing_list}")
    format_version = _npz_integer(snapshot, "sparse_snapshot_format_version")
    if format_version != SPARSE_SNAPSHOT_FORMAT_VERSION:
        raise ValueError(
            "unsupported persisted sparse snapshot format version "
            f"{format_version}; expected {SPARSE_SNAPSHOT_FORMAT_VERSION}"
        )
    encoding_value = np.asarray(snapshot["sparse_snapshot_encoding"])
    if encoding_value.shape != () or encoding_value.dtype.kind not in "US":
        raise ValueError("persisted sparse snapshot encoding must be text")
    encoding = str(encoding_value)
    if encoding != SPARSE_SNAPSHOT_ENCODING:
        raise ValueError(
            f"unsupported persisted sparse snapshot encoding {encoding!r}"
        )

    tree_names_array = np.asarray(snapshot["sparse_snapshot_tree_names"])
    leaf_names_array = np.asarray(snapshot["sparse_snapshot_leaf_names"])
    if tree_names_array.ndim != 1 or tree_names_array.dtype.kind not in "US":
        raise ValueError("persisted sparse tree names must be text")
    if leaf_names_array.ndim != 1 or leaf_names_array.dtype.kind not in "US":
        raise ValueError("persisted sparse leaf names must be text")
    tree_names = tuple(str(value) for value in tree_names_array)
    leaf_names = tuple(str(value) for value in leaf_names_array)
    if len(tree_names) < 2 or len(leaf_names) < 2:
        raise ValueError("persisted sparse snapshots require two trees and taxa")

    rooted_value = np.asarray(snapshot["sparse_snapshot_rooted"])
    if rooted_value.shape != () or rooted_value.dtype != np.dtype(np.bool_):
        raise TypeError("persisted sparse snapshot rooted flag must be boolean")
    n_clades = _npz_integer(snapshot, "sparse_snapshot_n_clades")
    n_entries = _npz_integer(snapshot, "sparse_snapshot_n_entries")
    packed_clades = _npz_array(
        snapshot,
        "sparse_snapshot_packed_clades",
        dtype=np.dtype(np.uint8),
        shape=(n_clades, (len(leaf_names) + 7) // 8),
    )
    row_offsets = _npz_array(
        snapshot,
        "sparse_snapshot_row_offsets",
        dtype=np.dtype(np.uint64),
        shape=(len(tree_names) + 1,),
    )
    column_indices = _npz_array(
        snapshot,
        "sparse_snapshot_column_indices",
        dtype=np.dtype(np.uint32),
        shape=(n_entries,),
    )
    _validate_sparse_arrays(
        n_clades=n_clades,
        n_taxa=len(leaf_names),
        packed_clades=packed_clades,
        row_offsets=row_offsets,
        column_indices=column_indices,
    )
    return SparseSnapshot(
        tree_names=tree_names,
        leaf_names=leaf_names,
        n_clades=n_clades,
        packed_clades=packed_clades,
        row_offsets=row_offsets,
        column_indices=column_indices,
        rooted=bool(rooted_value),
    )


def _normalise_rows(
    sparse_snapshot: SparseSnapshot,
    rows: Sequence[int],
) -> np.ndarray:
    values = np.asarray(rows)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise TypeError("sparse snapshot row selection must be integer-valued")
    values = values.astype(np.intp, copy=False)
    if np.any(values < 0) or np.any(values >= sparse_snapshot.n_trees):
        raise IndexError("sparse snapshot row selection is outside the tree range")
    return values


def count_sparse_columns(
    sparse_snapshot: SparseSnapshot,
    rows: Sequence[int],
    *,
    columns: Sequence[int] | None = None,
) -> np.ndarray:
    """Count clade membership over selected rows without densifying them."""
    if not isinstance(sparse_snapshot, SparseSnapshot):
        raise TypeError("sparse_snapshot must be a SparseSnapshot")
    row_ids = _normalise_rows(sparse_snapshot, rows)
    if columns is None:
        counts = np.zeros(sparse_snapshot.n_clades, dtype=np.int64)
        for row in row_ids:
            counts[sparse_snapshot.row_columns(int(row))] += 1
        return counts

    requested = np.asarray(columns)
    if requested.ndim != 1 or requested.dtype.kind not in "iu":
        raise TypeError("requested sparse columns must be integer-valued")
    requested = requested.astype(np.uint32, copy=False)
    if len(requested) and (
        np.any(requested >= sparse_snapshot.n_clades)
        or len(np.unique(requested)) != len(requested)
    ):
        raise IndexError(
            "requested sparse columns must be unique and inside the catalog"
        )
    counts = np.zeros(len(requested), dtype=np.int64)
    if not len(requested):
        return counts
    for row in row_ids:
        row_columns = sparse_snapshot.row_columns(int(row))
        positions = np.searchsorted(row_columns, requested)
        valid = positions < len(row_columns)
        matches = np.zeros(len(requested), dtype=bool)
        matches[valid] = row_columns[positions[valid]] == requested[valid]
        counts[matches] += 1
    return counts


def dense_presence_from_sparse(
    sparse_snapshot: SparseSnapshot,
) -> np.ndarray:
    """Reconstruct the legacy dense presence matrix for compatibility."""
    presence = np.zeros(
        (sparse_snapshot.n_trees, sparse_snapshot.n_clades),
        dtype=np.uint8,
    )
    for row in range(sparse_snapshot.n_trees):
        presence[row, sparse_snapshot.row_columns(row)] = 1
    return presence


def dense_clade_bits_from_sparse(
    sparse_snapshot: SparseSnapshot,
) -> np.ndarray:
    """Reconstruct the legacy unpacked clade-bit matrix for compatibility."""
    if not sparse_snapshot.n_clades:
        return np.zeros((0, sparse_snapshot.n_taxa), dtype=np.uint8)
    return np.unpackbits(
        sparse_snapshot.packed_clades,
        axis=1,
        bitorder="little",
    )[:, : sparse_snapshot.n_taxa].copy()


def sparse_clade_tip_indices(
    sparse_snapshot: SparseSnapshot,
    columns: Sequence[int] | None = None,
    *,
    chunk_rows: int = 1024,
) -> list[tuple[int, ...]]:
    """Decode packed clades into canonical taxon-index tuples in chunks."""
    if columns is None:
        column_ids = np.arange(sparse_snapshot.n_clades, dtype=np.intp)
    else:
        column_ids = np.asarray(columns)
        if column_ids.ndim != 1 or column_ids.dtype.kind not in "iu":
            raise TypeError("clade columns must be a one-dimensional integer list")
        column_ids = column_ids.astype(np.intp, copy=False)
        if np.any(column_ids < 0) or np.any(
            column_ids >= sparse_snapshot.n_clades
        ):
            raise IndexError("clade column is outside the sparse catalog")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    result: list[tuple[int, ...]] = []
    for start in range(0, len(column_ids), chunk_rows):
        selected = sparse_snapshot.packed_clades[
            column_ids[start : start + chunk_rows]
        ]
        bits = np.unpackbits(selected, axis=1, bitorder="little")[
            :, : sparse_snapshot.n_taxa
        ]
        result.extend(
            tuple(int(index) for index in np.flatnonzero(row))
            for row in bits
        )
    return result
