"""Unified clade- and tip-level Sister-Clade Jaccard Instability.

Pure-NumPy engine over TreeTracer's v3 ``RootedFactsSnapshot`` split tables.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..rf.rooted_facts import (
    ROOTED_FACTS_FORMAT_VERSION,
    _npz_array,
    _npz_scalar,
    snapshot_has_rooted_facts,
)


_PACKED_JACCARD_THRESHOLD = 256
_BINCOUNT_CHUNK_ROWS = 64

# Lowest set bit (0..7) per uint8 byte under RapidTrees' little-bit-order packing.
_LOWEST_BIT = np.array(
    [0 if v == 0 else (v & -v).bit_length() - 1 for v in range(256)],
    dtype=np.int32,
)

_TSV_COLUMNS = (
    "taxon",
    "d_sis",
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


@dataclass(frozen=True, slots=True)
class CladeInstabilityMetrics:
    """Sister-clade Jaccard instability metrics for one consensus clade."""

    clade_instability: float
    instab_1: float | None
    instab_2: float | None
    d_sis: float | None


def _normalise_leaf_names(raw: Sequence[object]) -> tuple[str, ...]:
    return tuple(str(name).strip("'\"") for name in raw)


def _load_split_arrays(
    snapshot: object,
    *,
    packed_clades: np.ndarray | None = None,
    leaf_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Selectively read only v3 split arrays, skipping clade_columns and heights."""
    if not snapshot_has_rooted_facts(snapshot):
        raise ValueError("snapshot does not contain version-3 rooted facts")

    files = set(
        getattr(snapshot, "files", None)
        or snapshot.keys()  # type: ignore[union-attr]
    )
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
        raise ValueError(
            f"persisted rooted facts are missing: {', '.join(sorted(missing))}"
        )

    version = _npz_scalar(snapshot, "rooted_facts_format_version")
    if version != ROOTED_FACTS_FORMAT_VERSION:
        raise ValueError(
            f"unsupported persisted rooted-facts format version {version}; "
            f"expected {ROOTED_FACTS_FORMAT_VERSION}"
        )

    if leaf_names is None:
        raw_leaves = np.asarray(snapshot["rooted_facts_leaf_names"])
        if raw_leaves.ndim != 1 or raw_leaves.dtype.kind not in "US":
            raise ValueError("persisted rooted-facts leaf names must be text")
        leaves = _normalise_leaf_names(raw_leaves)
    else:
        leaves = _normalise_leaf_names(leaf_names)

    n_taxa = len(leaves)
    if n_taxa < 2:
        raise ValueError("sister-clade instability requires at least two taxa")

    n_clades = _npz_scalar(snapshot, "rooted_facts_n_clades")
    splits_per_tree = _npz_scalar(snapshot, "rooted_facts_splits_per_tree")
    n_observed_splits = _npz_scalar(snapshot, "rooted_facts_n_observed_splits")
    if n_clades <= 0 or splits_per_tree != n_taxa - 1 or n_observed_splits <= 0:
        raise ValueError("persisted rooted facts have invalid split dimensions")

    packed_shape = (n_clades, (n_taxa + 7) // 8)
    if packed_clades is None:
        loaded_packed = _npz_array(
            snapshot,
            "rooted_facts_packed_clades",
            dtype=np.dtype(np.uint8),
            shape=packed_shape,
        )
    else:
        loaded_packed = np.asarray(packed_clades)
        if loaded_packed.dtype != np.dtype(np.uint8) or loaded_packed.shape != packed_shape:
            raise ValueError("preloaded packed_clades disagrees with persisted rooted facts")

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
        raise ValueError("persisted 'rooted_facts_split_ids' has invalid dtype or shape")
    if not split_ids.flags.c_contiguous:
        split_ids = np.ascontiguousarray(split_ids)
    return loaded_packed, split_table, split_ids, leaves


def _normalise_row_indices(
    row_indices: Sequence[int], n_trees: int, *, name: str
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
    split_ids: np.ndarray, row_indices: np.ndarray, n_splits: int
) -> np.ndarray:
    counts = np.zeros(n_splits, dtype=np.int64)
    for start in range(0, len(row_indices), _BINCOUNT_CHUNK_ROWS):
        chunk = split_ids[row_indices[start : start + _BINCOUNT_CHUNK_ROWS]]
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
    k = len(sisters)
    if k <= 1:
        return np.ones((k, k), dtype=np.float64)
    subset = packed_clades[sisters]
    s = sizes[sisters].astype(np.float64)
    if k <= packed_threshold:
        inter = np.bitwise_count(subset[:, None, :] & subset[None, :, :]).sum(
            axis=2, dtype=np.float64
        )
    else:
        unpacked = np.unpackbits(subset, axis=1, bitorder="little")[
            :, :n_taxa
        ].astype(np.float32)
        inter = (unpacked @ unpacked.T).astype(np.float64)
    union = s[:, None] + s[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)


def _format_pct(prob: float) -> str:
    if prob >= 1.0 - 1e-12:
        return "100%"
    pct = round(prob * 100)
    if pct >= 100:
        return ">99%"
    if pct <= 0:
        return "<1%"
    return f"{pct}%"


def _format_top_sisters(
    sisters: np.ndarray,
    probs: np.ndarray | None,
    *,
    sizes: np.ndarray,
) -> str:
    if probs is None:
        return "—"
    observed = np.flatnonzero(probs > 0.0)
    if len(observed) == 0:
        return "—"

    ordered = sorted(
        observed.tolist(),
        key=lambda i: (-float(probs[i]), int(sizes[sisters[i]]), int(sisters[i])),
    )
    kept = [i for i in ordered if float(probs[i]) >= 0.05][:2] or ordered[:1]

    parts = []
    for i in kept:
        m = int(sizes[int(sisters[i])])
        pct = _format_pct(float(probs[i]))
        if m == 1:
            parts.append(f"1-tip cherry ({pct})")
        else:
            parts.append(f"{m}-tip sister ({pct})")
    return ", ".join(parts)


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
    leaves = _normalise_leaf_names(leaf_names)
    n_taxa = len(leaves)
    n_clades = int(packed_clades.shape[0])
    n_splits = int(split_table.shape[0])

    r1 = _normalise_row_indices(row_indices_1, split_ids.shape[0], name="row_indices_1")
    r2 = _normalise_row_indices(row_indices_2, split_ids.shape[0], name="row_indices_2")
    r_comb = np.union1d(r1, r2)

    sizes = np.bitwise_count(packed_clades).sum(axis=1, dtype=np.int32)
    tip_cols = np.flatnonzero(sizes == 1)
    if len(tip_cols):
        tip_packed = packed_clades[tip_cols]
        first_byte = np.argmax(tip_packed != 0, axis=1)
        tip_taxa = 8 * first_byte + _LOWEST_BIT[tip_packed[np.arange(len(tip_cols)), first_byte]]
        col_to_taxon = dict(zip(tip_cols.tolist(), tip_taxa.tolist()))
    else:
        col_to_taxon = {}

    cons_cols = {int(c) for c in clade_columns}
    if any(c < 0 or c >= n_clades for c in cons_cols):
        raise IndexError(f"clade column outside [0, {n_clades})")

    target_mask = np.zeros(n_clades, dtype=bool)
    target_mask[tip_cols] = True
    if cons_cols:
        target_mask[sorted(cons_cols)] = True

    c1 = _bincount_splits(split_ids, r1, n_splits)
    c2 = _bincount_splits(split_ids, r2, n_splits)
    c_comb = (c1 + c2) if len(r_comb) == len(r1) + len(r2) else _bincount_splits(split_ids, r_comb, n_splits)

    active = np.flatnonzero(c_comb > 0)
    if not len(active):
        return {}, []

    left = split_table[active, 1]
    right = split_table[active, 2]
    # Independent left/right masks so cherries and consensus-clade pairs emit both sides.
    left_hits = np.flatnonzero(target_mask[left])
    right_hits = np.flatnonzero(target_mask[right])

    target_cols = np.concatenate([left[left_hits], right[right_hits]])
    if not len(target_cols):
        return {}, []
    sister_cols = np.concatenate([right[left_hits], left[right_hits]])
    hit_splits = np.concatenate([active[left_hits], active[right_hits]])

    order = np.argsort(target_cols, kind="stable")
    target_cols, sister_cols, hit_splits = target_cols[order], sister_cols[order], hit_splits[order]
    n1_stream, n2_stream, nc_stream = c1[hit_splits], c2[hit_splits], c_comb[hit_splits]
    boundaries = np.concatenate(
        [[0], np.flatnonzero(target_cols[1:] != target_cols[:-1]) + 1, [len(target_cols)]]
    )

    clade_metrics: dict[int, CladeInstabilityMetrics] = {}
    tip_rows: list[dict[str, Any]] = []

    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        target_col = int(target_cols[lo])
        sisters = sister_cols[lo:hi]
        n1, n2, nc = n1_stream[lo:hi], n2_stream[lo:hi], nc_stream[lo:hi]

        m1, m2 = int(n1.sum()), int(n2.sum())
        p_comb = nc.astype(np.float64) / float(nc.sum())
        p1 = (n1.astype(np.float64) / float(m1)) if m1 > 0 else None
        p2 = (n2.astype(np.float64) / float(m2)) if m2 > 0 else None

        if len(sisters) <= 1:
            i_comb = 0.0
            i1 = 0.0 if p1 is not None else None
            i2 = 0.0 if p2 is not None else None
            d_sis = 0.0 if (p1 is not None and p2 is not None) else None
        else:
            jaccard = _pairwise_jaccard(
                packed_clades, sizes, sisters, n_taxa, packed_threshold=packed_threshold
            )
            i_comb = float(np.clip(1.0 - p_comb @ jaccard @ p_comb, 0.0, 1.0))
            i1 = float(np.clip(1.0 - p1 @ jaccard @ p1, 0.0, 1.0)) if p1 is not None else None
            i2 = float(np.clip(1.0 - p2 @ jaccard @ p2, 0.0, 1.0)) if p2 is not None else None
            if p1 is not None and p2 is not None:
                diff = p1 - p2
                d_sis = float(np.clip(0.5 * (diff @ jaccard @ diff), 0.0, 1.0))
            else:
                d_sis = None

        if target_col in cons_cols:
            clade_metrics[target_col] = CladeInstabilityMetrics(i_comb, i1, i2, d_sis)

        taxon_index = col_to_taxon.get(target_col)
        if taxon_index is not None:
            is_cherry = sizes[sisters] == 1
            tip_rows.append(
                {
                    "taxon_index": taxon_index,
                    "taxon": leaves[taxon_index],
                    "column_j": target_col,
                    "d_sis": float(d_sis or 0.0),
                    "instab_1": float(i1 or 0.0),
                    "instab_2": float(i2 or 0.0),
                    "combined_instability": i_comb,
                    "n_sisters_1": int(np.count_nonzero(n1 > 0)),
                    "n_sisters_2": int(np.count_nonzero(n2 > 0)),
                    "n_sisters_comb": int(len(sisters)),
                    "p_cherry_1": float(np.clip(p1[is_cherry].sum(), 0.0, 1.0)) if p1 is not None else 0.0,
                    "p_cherry_2": float(np.clip(p2[is_cherry].sum(), 0.0, 1.0)) if p2 is not None else 0.0,
                    "top_sister_1": _format_top_sisters(sisters, p1, sizes=sizes),
                    "top_sister_2": _format_top_sisters(sisters, p2, sizes=sizes),
                }
            )

    tip_rows.sort(
        key=lambda row: (-row["d_sis"], -row["combined_instability"], row["taxon"])
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
    """Load minimal v3 split arrays from ``snapshot`` and compute instability."""
    loaded_packed, split_table, split_ids, leaves = _load_split_arrays(
        snapshot, packed_clades=packed_clades, leaf_names=leaf_names
    )
    return compute_sister_instability_from_arrays(
        packed_clades=loaded_packed,
        split_table=split_table,
        split_ids=split_ids,
        leaf_names=leaves,
        row_indices_1=row_indices_1,
        row_indices_2=row_indices_2,
        clade_columns=clade_columns,
        packed_threshold=packed_threshold,
    )


def format_tip_instability_tsv(tip_rows: Sequence[dict[str, Any]]) -> str:
    """Serialize all tip instability rows to a tab-separated string."""
    clean = lambda v: str(v).replace("\t", " ").replace("\r", " ").replace("\n", " ")
    lines = ["\t".join(_TSV_COLUMNS)]
    for row in tip_rows:
        lines.append(
            "\t".join(
                (
                    clean(row["taxon"]),
                    f"{float(row['d_sis']):.6f}",
                    f"{float(row['instab_1']):.6f}",
                    f"{float(row['instab_2']):.6f}",
                    f"{float(row['combined_instability']):.6f}",
                    str(int(row["n_sisters_1"])),
                    str(int(row["n_sisters_2"])),
                    str(int(row["n_sisters_comb"])),
                    f"{float(row['p_cherry_1']):.6f}",
                    f"{float(row['p_cherry_2']):.6f}",
                    clean(row["top_sister_1"]),
                    clean(row["top_sister_2"]),
                )
            )
        )
    return "\n".join(lines) + "\n"
