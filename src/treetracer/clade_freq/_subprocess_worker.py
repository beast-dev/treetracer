"""Process-isolated clade-frequency comparison worker.

The UI ultimately displays only clades present in either selected consensus
tree. Decoding every bipartition in the RF snapshot into Python tuples was both
CPU- and memory-heavy, especially on older hardware. This worker decodes only
that small union of column IDs and returns compact numeric rows plus one shared
leaf-name vector.
"""

from __future__ import annotations

from typing import Any


def _counts_for_columns(
    *,
    sparse_snapshot,
    columns,
    cached_counts,
    cached_n_trees,
    tree_names,
    full_distmat_names,
):
    import numpy as np

    if cached_counts is not None and int(cached_n_trees or 0) > 0:
        counts = np.asarray(cached_counts, dtype=np.int64)
        if counts.shape != (len(columns),):
            raise ValueError(
                "cached clade counts do not match the selected column count"
            )
        return counts, int(cached_n_trees)

    if not tree_names or not full_distmat_names:
        raise ValueError(
            "clade counts are unavailable and no tree-name fallback was supplied"
        )
    name_to_index = {
        str(name): index for index, name in enumerate(full_distmat_names)
    }
    row_indices = [
        name_to_index[str(name)]
        for name in tree_names
        if str(name) in name_to_index
    ]
    if not row_indices:
        raise KeyError(
            "none of the selected consensus-tree names occur in the RF snapshot"
        )

    from ..rf.sparse_snapshots import count_sparse_columns

    counts = count_sparse_columns(
        sparse_snapshot,
        row_indices,
        columns=columns,
    )
    return np.asarray(counts, dtype=np.int64), len(row_indices)


def compute_clade_frequencies_worker_entry(
    *,
    snapshots_path: str,
    columns: list[int],
    counts_1: list[int] | None,
    counts_2: list[int] | None,
    n_trees_1: int,
    n_trees_2: int,
    tree_names_1: list[str] | None = None,
    tree_names_2: list[str] | None = None,
    full_distmat_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute frequencies and decode only requested snapshot columns."""
    import time

    import numpy as np

    started = time.perf_counter()
    columns = sorted({int(column) for column in columns})
    if any(column < 0 for column in columns):
        raise ValueError("clade column IDs must be non-negative")

    with np.load(snapshots_path, allow_pickle=False) as snapshot:
        from ..rf.sparse_snapshots import (
            snapshot_has_generic_sparse_snapshot,
            sparse_clade_tip_indices,
            sparse_snapshot_from_npz,
        )

        try:
            sparse_snapshot = sparse_snapshot_from_npz(snapshot)
        except KeyError as exc:
            raise ValueError(
                "clade-frequency comparison requires a sparse RF snapshot; "
                "recompute the RF matrix with RapidTrees 0.9.1 or newer"
            ) from exc
        n_clades = sparse_snapshot.n_clades
        snapshot_input_mode = (
            "sparse"
            if snapshot_has_generic_sparse_snapshot(snapshot)
            else "rooted_facts"
        )
        if columns and columns[-1] >= n_clades:
            raise IndexError(
                f"clade column {columns[-1]} is outside a "
                f"{n_clades}-column RF snapshot"
            )

        selected_counts_1, denominator_1 = _counts_for_columns(
            sparse_snapshot=sparse_snapshot,
            columns=columns,
            cached_counts=counts_1,
            cached_n_trees=n_trees_1,
            tree_names=tree_names_1,
            full_distmat_names=full_distmat_names,
        )
        selected_counts_2, denominator_2 = _counts_for_columns(
            sparse_snapshot=sparse_snapshot,
            columns=columns,
            cached_counts=counts_2,
            cached_n_trees=n_trees_2,
            tree_names=tree_names_2,
            full_distmat_names=full_distmat_names,
        )

        selected_keys = sparse_clade_tip_indices(
            sparse_snapshot,
            columns,
        )
        raw_leaf_names = sparse_snapshot.leaf_names
        leaf_names = [str(name).strip("'\"") for name in raw_leaf_names]

        frequencies_1 = selected_counts_1.astype(
            np.float64,
            copy=False,
        ) / np.float64(denominator_1)
        frequencies_2 = selected_counts_2.astype(
            np.float64,
            copy=False,
        ) / np.float64(denominator_2)

        rows = []
        for position, column in enumerate(columns):
            split_key = selected_keys[position]
            rows.append(
                {
                    "split_key": split_key,
                    "column_j": int(column),
                    "freq_1": float(frequencies_1[position]),
                    "freq_2": float(frequencies_2[position]),
                    "clade_size": len(split_key),
                }
            )

    rows.sort(
        key=lambda row: (row["freq_1"] + row["freq_2"]) / 2.0,
        reverse=True,
    )
    return {
        "rows": rows,
        "leaf_names": leaf_names,
        "n_trees_1": denominator_1,
        "n_trees_2": denominator_2,
        "snapshot_input_mode": snapshot_input_mode,
        "elapsed": time.perf_counter() - started,
    }
