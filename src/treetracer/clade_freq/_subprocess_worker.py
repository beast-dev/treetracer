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
    snapshot,
    *,
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

    presence = snapshot["presence"]
    counts = presence[np.ix_(row_indices, columns)].sum(axis=0)
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
        if "bipartition_bits" not in snapshot.files:
            raise KeyError(
                "RF snapshot has no bipartition_bits; recompute the RF matrix"
            )
        bits = snapshot["bipartition_bits"]
        if columns and columns[-1] >= bits.shape[0]:
            raise IndexError(
                f"clade column {columns[-1]} is outside a "
                f"{bits.shape[0]}-column RF snapshot"
            )

        selected_counts_1, denominator_1 = _counts_for_columns(
            snapshot,
            columns=columns,
            cached_counts=counts_1,
            cached_n_trees=n_trees_1,
            tree_names=tree_names_1,
            full_distmat_names=full_distmat_names,
        )
        selected_counts_2, denominator_2 = _counts_for_columns(
            snapshot,
            columns=columns,
            cached_counts=counts_2,
            cached_n_trees=n_trees_2,
            tree_names=tree_names_2,
            full_distmat_names=full_distmat_names,
        )

        selected_bits = bits[columns]
        leaf_names = [str(name).strip("'\"") for name in snapshot["leaf_names"]]

        rows = []
        for position, column in enumerate(columns):
            split_key = tuple(
                int(index)
                for index in np.flatnonzero(selected_bits[position])
            )
            freq_1 = float(selected_counts_1[position]) / denominator_1
            freq_2 = float(selected_counts_2[position]) / denominator_2
            rows.append(
                {
                    "split_key": split_key,
                    "column_j": int(column),
                    "freq_1": freq_1,
                    "freq_2": freq_2,
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
        "elapsed": time.perf_counter() - started,
    }
