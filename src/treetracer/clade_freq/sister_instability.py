"""Unified clade- and tip-level Sister-Clade Jaccard Instability.

Pure-NumPy implementation operating directly on TreeTracer's version-3
``RootedFactsSnapshot`` split tables (``packed_clades``, ``split_table``,
``split_ids``, ``leaf_names``) without importing Dash or reparsing trees.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..rf.rooted_facts import (
    ROOTED_FACTS_FORMAT_VERSION,
    snapshot_has_rooted_facts,
)


_PACKED_JACCARD_THRESHOLD = 256
_BINCOUNT_CHUNK_ROWS = 64

# Lowest set bit position (0..7) for each uint8 byte value under little-bit-order
# encoding (RapidTrees' ``packed_clades`` convention).
_LOWEST_BIT = np.array(
    [0 if value == 0 else (value & -value).bit_length() - 1 for value in range(256)],
    dtype=np.int32,
)


@dataclass(frozen=True, slots=True)
class CladeInstabilityMetrics:
    """Sister-clade Jaccard instability metrics for one consensus clade."""

    clade_instability: float
    instab_1: float | None
    instab_2: float | None
    between_shift: float | None


@dataclass(frozen=True, slots=True)
class _SplitArrays:
    """Minimal rooted-facts arrays required by sister-clade instability."""

    packed_clades: np.ndarray
    split_table: np.ndarray
    split_ids: np.ndarray
    leaf_names: tuple[str, ...]
    n_clades: int


def _snapshot_files(snapshot: object) -> set[str]:
    files = getattr(snapshot, "files", None)
    if files is None:
        keys = getattr(snapshot, "keys", None)
        if keys is None:
            raise TypeError("snapshot must expose .files or .keys()")
        files = keys()
    return set(files)


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
    return value


def _normalise_leaf_names(raw_leaf_names: Sequence[object]) -> tuple[str, ...]:
    """Strip syntactic NEXUS quotes so names match parsed consensus trees."""
    return tuple(str(name).strip("'\"") for name in raw_leaf_names)


def _load_split_arrays(
    snapshot: object,
    *,
    packed_clades: np.ndarray | None = None,
    leaf_names: Sequence[str] | None = None,
) -> _SplitArrays:
    """Selectively load only the v3 rooted-facts arrays needed for instability.

    Skips ``rooted_facts_clade_columns``, ``rooted_facts_node_heights``, and
    ``rooted_facts_root_heights`` to keep I/O and RAM low. Callers that already
    read ``packed_clades`` and ``leaf_names`` (e.g., via ``SparseSnapshot``)
    may pass them in to avoid reading those members twice.
    """
    if not snapshot_has_rooted_facts(snapshot):
        raise ValueError("snapshot does not contain version-3 rooted facts")

    files = _snapshot_files(snapshot)
    required = {
        "rooted_facts_format_version",
        "rooted_facts_n_clades",
        "rooted_facts_splits_per_tree",
        "rooted_facts_n_observed_splits",
        "rooted_facts_split_table",
        "rooted_facts_split_ids",
    }
    if packed_clades is None:
        required.add("rooted_facts_packed_clades")
    if leaf_names is None:
        required.add("rooted_facts_leaf_names")
    missing = required.difference(files)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"persisted rooted facts are missing: {missing_list}")

    format_version = _npz_scalar(snapshot, "rooted_facts_format_version")
    if format_version != ROOTED_FACTS_FORMAT_VERSION:
        raise ValueError(
            "unsupported persisted rooted-facts format version "
            f"{format_version}; expected {ROOTED_FACTS_FORMAT_VERSION}"
        )

    if leaf_names is None:
        leaf_names_array = np.asarray(snapshot["rooted_facts_leaf_names"])
        if leaf_names_array.ndim != 1 or leaf_names_array.dtype.kind not in "US":
            raise ValueError("persisted rooted-facts leaf names must be text")
        normalised_leaves = _normalise_leaf_names(leaf_names_array)
    else:
        normalised_leaves = _normalise_leaf_names(leaf_names)

    n_taxa = len(normalised_leaves)
    if n_taxa < 2:
        raise ValueError("sister-clade instability requires at least two taxa")

    n_clades = _npz_scalar(snapshot, "rooted_facts_n_clades")
    splits_per_tree = _npz_scalar(snapshot, "rooted_facts_splits_per_tree")
    n_observed_splits = _npz_scalar(snapshot, "rooted_facts_n_observed_splits")
    expected_splits = n_taxa - 1
    if n_clades <= 0:
        raise ValueError("persisted rooted facts have invalid clade count")
    if splits_per_tree != expected_splits:
        raise ValueError("persisted rooted-facts split width disagrees with leaf names")
    if n_observed_splits <= 0:
        raise ValueError("persisted rooted facts contain no observed splits")

    n_bytes = (n_taxa + 7) // 8
    if packed_clades is None:
        loaded_packed = _npz_array(
            snapshot,
            "rooted_facts_packed_clades",
            dtype=np.dtype(np.uint8),
            shape=(n_clades, n_bytes),
        )
    else:
        loaded_packed = np.asarray(packed_clades)
        if loaded_packed.dtype != np.dtype(np.uint8) or loaded_packed.shape != (
            n_clades,
            n_bytes,
        ):
            raise ValueError(
                "preloaded packed_clades disagrees with persisted rooted facts"
            )

    split_table = _npz_array(
        snapshot,
        "rooted_facts_split_table",
        dtype=np.dtype(np.uint32),
        shape=(n_observed_splits, 3),
    )
    split_ids = np.asarray(snapshot["rooted_facts_split_ids"])
    if (
        split_ids.dtype != np.dtype(np.uint32)
        or split_ids.ndim != 2
        or split_ids.shape[1] != splits_per_tree
        or split_ids.shape[0] < 1
    ):
        raise ValueError(
            "persisted 'rooted_facts_split_ids' has invalid dtype or shape"
        )
    if not split_ids.flags.c_contiguous:
        split_ids = np.ascontiguousarray(split_ids)

    return _SplitArrays(
        packed_clades=loaded_packed,
        split_table=split_table,
        split_ids=split_ids,
        leaf_names=normalised_leaves,
        n_clades=n_clades,
    )


def _normalise_row_indices(
    row_indices: Sequence[int],
    n_trees: int,
    *,
    name: str,
) -> np.ndarray:
    arr = np.asarray(row_indices)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if arr.ndim != 1 or arr.dtype.kind not in "iu":
        raise TypeError(f"{name} must be a 1-D sequence of integers")
    unique = np.unique(arr.astype(np.intp, copy=False))
    if unique[0] < 0 or unique[-1] >= n_trees:
        raise IndexError(f"{name} contains a row index outside [0, {n_trees})")
    return unique


def _bincount_splits(
    split_ids: np.ndarray,
    row_indices: np.ndarray,
    n_splits: int,
    *,
    chunk_rows: int = _BINCOUNT_CHUNK_ROWS,
) -> np.ndarray:
    counts = np.zeros(n_splits, dtype=np.int64)
    for start in range(0, len(row_indices), chunk_rows):
        chunk = split_ids[row_indices[start : start + chunk_rows]]
        counts += np.bincount(chunk.ravel(), minlength=n_splits)
    return counts


def _pairwise_jaccard(
    packed_clades: np.ndarray,
    sizes: np.ndarray,
    sisters: np.ndarray,
    n_taxa: int,
    *,
    packed_threshold: int = _PACKED_JACCARD_THRESHOLD,
) -> np.ndarray:
    """Compute the k x k Jaccard similarity matrix among sister clade columns.

    Uses packed bitwise ``AND + np.bitwise_count`` for small sister sets
    (``k <= packed_threshold``) and switches to FP32 ``U @ U.T`` BLAS matrix
    multiplication for large ``k`` so high-uncertainty tips stay fast.
    """
    k = len(sisters)
    if k == 0:
        return np.empty((0, 0), dtype=np.float64)
    if k == 1:
        return np.ones((1, 1), dtype=np.float64)

    subset = packed_clades[sisters]
    sister_sizes = sizes[sisters].astype(np.float64)
    if k <= packed_threshold:
        inter = np.bitwise_count(
            subset[:, None, :] & subset[None, :, :]
        ).sum(axis=2, dtype=np.float64)
    else:
        unpacked = np.unpackbits(subset, axis=1, bitorder="little")[
            :, :n_taxa
        ].astype(np.float32)
        inter = (unpacked @ unpacked.T).astype(np.float64)

    union = sister_sizes[:, None] + sister_sizes[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _first_set_bits(packed_row: np.ndarray, limit: int) -> list[int]:
    """Extract the first ``limit`` set taxon indices from one packed clade row."""
    found: list[int] = []
    for byte_idx in range(len(packed_row)):
        byte_val = int(packed_row[byte_idx])
        while byte_val:
            lsb = byte_val & -byte_val
            bit = lsb.bit_length() - 1
            found.append(byte_idx * 8 + bit)
            if len(found) >= limit:
                return found
            byte_val ^= lsb
    return found


def _format_top_sisters(
    sisters: np.ndarray,
    probs: np.ndarray | None,
    *,
    packed_clades: np.ndarray,
    sizes: np.ndarray,
    leaf_names: Sequence[str],
    min_prob: float = 0.05,
    max_items: int = 2,
) -> str:
    """Format up to ``max_items`` top observed sister clades concisely."""
    if probs is None:
        return "—"
    observed = np.flatnonzero(probs > 0.0)
    if len(observed) == 0:
        return "—"

    # Deterministic ordering: descending probability, then smaller clade, then column ID.
    ordered = sorted(
        observed.tolist(),
        key=lambda idx: (-float(probs[idx]), int(sizes[sisters[idx]]), int(sisters[idx])),
    )
    kept = [idx for idx in ordered if float(probs[idx]) >= min_prob][:max_items]
    if not kept:
        kept = ordered[:1]

    parts: list[str] = []
    for idx in kept:
        col = int(sisters[idx])
        m = int(sizes[col])
        pct = f"{round(float(probs[idx]) * 100)}%"
        bits = _first_set_bits(packed_clades[col], min(m, 2))
        if m == 1:
            parts.append(f"{leaf_names[bits[0]]} (cherry) {pct}")
        elif m == 2:
            parts.append(f"{{{leaf_names[bits[0]]}, {leaf_names[bits[1]]}}} {pct}")
        else:
            parts.append(f"Clade ({m} tips: {leaf_names[bits[0]]}, ...) {pct}")

    summary = ", ".join(parts)
    remaining = len(ordered) - len(kept)
    if remaining > 0:
        summary += f" + {remaining} more"
    return summary


def compute_sister_instability_from_arrays(
    *,
    packed_clades: np.ndarray,
    split_table: np.ndarray,
    split_ids: np.ndarray,
    leaf_names: Sequence[str],
    row_indices_1: Sequence[int],
    row_indices_2: Sequence[int],
    clade_columns: Sequence[int] = (),
    packed_threshold: int = _PACKED_JACCARD_THRESHOLD,
) -> tuple[dict[int, CladeInstabilityMetrics], list[dict[str, Any]]]:
    """Compute unified clade- and tip-level Sister-Clade Jaccard Instability."""
    normalised_leaves = _normalise_leaf_names(leaf_names)
    n_taxa = len(normalised_leaves)
    n_clades = int(packed_clades.shape[0])
    n_trees = int(split_ids.shape[0])
    n_splits = int(split_table.shape[0])

    r1 = _normalise_row_indices(row_indices_1, n_trees, name="row_indices_1")
    r2 = _normalise_row_indices(row_indices_2, n_trees, name="row_indices_2")
    r_comb = np.union1d(r1, r2)

    sizes = np.bitwise_count(packed_clades).sum(axis=1, dtype=np.int32)
    tip_cols = np.flatnonzero(sizes == 1)
    if len(tip_cols) > 0:
        tip_packed = packed_clades[tip_cols]
        first_byte = np.argmax(tip_packed != 0, axis=1)
        tip_taxa = 8 * first_byte + _LOWEST_BIT[
            tip_packed[np.arange(len(tip_cols)), first_byte]
        ]
        col_to_taxon = {
            int(col): int(taxon)
            for col, taxon in zip(tip_cols.tolist(), tip_taxa.tolist())
        }
    else:
        col_to_taxon = {}

    cons_col_set: set[int] = set()
    for raw_col in clade_columns:
        col = int(raw_col)
        if col < 0 or col >= n_clades:
            raise IndexError(
                f"clade column {col} is outside [0, {n_clades})"
            )
        cons_col_set.add(col)

    target_mask = np.zeros(n_clades, dtype=bool)
    if len(tip_cols) > 0:
        target_mask[tip_cols] = True
    if cons_col_set:
        target_mask[sorted(cons_col_set)] = True

    c1 = _bincount_splits(split_ids, r1, n_splits)
    c2 = _bincount_splits(split_ids, r2, n_splits)
    if len(r_comb) == len(r1) + len(r2):
        c_comb = c1 + c2
    else:
        c_comb = _bincount_splits(split_ids, r_comb, n_splits)

    active = np.flatnonzero(c_comb > 0)
    if len(active) == 0:
        return {}, []

    left = split_table[active, 1]
    right = split_table[active, 2]

    # Independent left/right emission (never if/elif) so cherries and splits
    # between two consensus clades emit both (left -> right) and (right -> left).
    left_hits = np.flatnonzero(target_mask[left])
    right_hits = np.flatnonzero(target_mask[right])

    target_cols = np.concatenate([left[left_hits], right[right_hits]])
    sister_cols = np.concatenate([right[left_hits], left[right_hits]])
    hit_splits = np.concatenate([active[left_hits], active[right_hits]])
    if len(target_cols) == 0:
        return {}, []

    order = np.argsort(target_cols, kind="stable")
    target_cols = target_cols[order]
    sister_cols = sister_cols[order]
    hit_splits = hit_splits[order]

    n1_stream = c1[hit_splits]
    n2_stream = c2[hit_splits]
    nc_stream = c_comb[hit_splits]

    boundaries = np.concatenate(
        [
            np.array([0], dtype=np.intp),
            np.flatnonzero(target_cols[1:] != target_cols[:-1]) + 1,
            np.array([len(target_cols)], dtype=np.intp),
        ]
    )

    clade_metrics: dict[int, CladeInstabilityMetrics] = {}
    tip_rows: list[dict[str, Any]] = []

    for group_idx in range(len(boundaries) - 1):
        lo = int(boundaries[group_idx])
        hi = int(boundaries[group_idx + 1])
        target_col = int(target_cols[lo])
        sisters = sister_cols[lo:hi]
        n1 = n1_stream[lo:hi]
        n2 = n2_stream[lo:hi]
        nc = nc_stream[lo:hi]

        m_comb = int(nc.sum())
        m1 = int(n1.sum())
        m2 = int(n2.sum())
        p_comb = nc.astype(np.float64) / float(m_comb)
        p1 = (n1.astype(np.float64) / float(m1)) if m1 > 0 else None
        p2 = (n2.astype(np.float64) / float(m2)) if m2 > 0 else None

        k_c = len(sisters)
        if k_c <= 1:
            i_comb = 0.0
            i1 = 0.0 if p1 is not None else None
            i2 = 0.0 if p2 is not None else None
            d_shift = 0.0 if (p1 is not None and p2 is not None) else None
        else:
            jaccard = _pairwise_jaccard(
                packed_clades,
                sizes,
                sisters,
                n_taxa,
                packed_threshold=packed_threshold,
            )
            i_comb = _clamp01(1.0 - float(p_comb @ jaccard @ p_comb))
            i1 = _clamp01(1.0 - float(p1 @ jaccard @ p1)) if p1 is not None else None
            i2 = _clamp01(1.0 - float(p2 @ jaccard @ p2)) if p2 is not None else None
            if p1 is not None and p2 is not None:
                diff = p1 - p2
                d_shift = _clamp01(0.5 * float(diff @ jaccard @ diff))
            else:
                d_shift = None

        if target_col in cons_col_set:
            clade_metrics[target_col] = CladeInstabilityMetrics(
                clade_instability=i_comb,
                instab_1=i1,
                instab_2=i2,
                between_shift=d_shift,
            )

        taxon_index = col_to_taxon.get(target_col)
        if taxon_index is not None:
            is_cherry = sizes[sisters] == 1
            p_cherry_1 = (
                _clamp01(float(p1[is_cherry].sum())) if p1 is not None else 0.0
            )
            p_cherry_2 = (
                _clamp01(float(p2[is_cherry].sum())) if p2 is not None else 0.0
            )
            tip_rows.append(
                {
                    "taxon_index": taxon_index,
                    "taxon": normalised_leaves[taxon_index],
                    "column_j": target_col,
                    "between_shift": float(d_shift if d_shift is not None else 0.0),
                    "instab_1": float(i1 if i1 is not None else 0.0),
                    "instab_2": float(i2 if i2 is not None else 0.0),
                    "combined_instability": float(i_comb),
                    "n_sisters_1": int(np.count_nonzero(n1 > 0)),
                    "n_sisters_2": int(np.count_nonzero(n2 > 0)),
                    "n_sisters_comb": int(k_c),
                    "p_cherry_1": p_cherry_1,
                    "p_cherry_2": p_cherry_2,
                    "top_sister_1": _format_top_sisters(
                        sisters,
                        p1,
                        packed_clades=packed_clades,
                        sizes=sizes,
                        leaf_names=normalised_leaves,
                    ),
                    "top_sister_2": _format_top_sisters(
                        sisters,
                        p2,
                        packed_clades=packed_clades,
                        sizes=sizes,
                        leaf_names=normalised_leaves,
                    ),
                }
            )

    tip_rows.sort(
        key=lambda row: (
            -row["between_shift"],
            -row["combined_instability"],
            row["taxon"],
        )
    )
    return clade_metrics, tip_rows


def compute_sister_instability_from_npz(
    snapshot: object,
    *,
    row_indices_1: Sequence[int],
    row_indices_2: Sequence[int],
    clade_columns: Sequence[int] = (),
    packed_clades: np.ndarray | None = None,
    leaf_names: Sequence[str] | None = None,
    packed_threshold: int = _PACKED_JACCARD_THRESHOLD,
) -> tuple[dict[int, CladeInstabilityMetrics], list[dict[str, Any]]]:
    """Load minimal v3 rooted-facts split arrays from ``snapshot`` and compute instability."""
    arrays = _load_split_arrays(
        snapshot,
        packed_clades=packed_clades,
        leaf_names=leaf_names,
    )
    return compute_sister_instability_from_arrays(
        packed_clades=arrays.packed_clades,
        split_table=arrays.split_table,
        split_ids=arrays.split_ids,
        leaf_names=arrays.leaf_names,
        row_indices_1=row_indices_1,
        row_indices_2=row_indices_2,
        clade_columns=clade_columns,
        packed_threshold=packed_threshold,
    )


_TSV_COLUMNS = (
    "taxon",
    "between_shift",
    "instab_1",
    "instab_2",
    "combined_instability",
    "n_sisters_1",
    "n_sisters_2",
    "n_sisters_comb",
    "p_cherry_1",
    "p_cherry_2",
    "top_sister_1",
    "top_sister_2",
)


def _tsv_cell(value: object) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def format_tip_instability_tsv(tip_rows: Sequence[dict[str, Any]]) -> str:
    """Serialize all tip instability rows to a tab-separated string for export."""
    lines = ["\t".join(_TSV_COLUMNS)]
    for row in tip_rows:
        lines.append(
            "\t".join(
                (
                    _tsv_cell(row["taxon"]),
                    f"{float(row['between_shift']):.6f}",
                    f"{float(row['instab_1']):.6f}",
                    f"{float(row['instab_2']):.6f}",
                    f"{float(row['combined_instability']):.6f}",
                    str(int(row["n_sisters_1"])),
                    str(int(row["n_sisters_2"])),
                    str(int(row["n_sisters_comb"])),
                    f"{float(row['p_cherry_1']):.6f}",
                    f"{float(row['p_cherry_2']):.6f}",
                    _tsv_cell(row["top_sister_1"]),
                    _tsv_cell(row["top_sister_2"]),
                )
            )
        )
    return "\n".join(lines) + "\n"
