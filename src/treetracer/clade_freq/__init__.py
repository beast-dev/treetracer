"""Clade frequency computation for the Clade Frequency Comparison feature.

Consumes the per-distmat snapshot written by ``rf._worker.compute_rf``
(compact sparse clade rows when available, with legacy ``presence`` and
``bipartition_bits`` compatibility) and produces a
DataFrame with one row per bipartition observed in either of two
groups of trees, containing per-group frequencies and the size of the
canonical side (the side NOT containing the alphabetically first
taxon, per rapidtrees' canonicalisation).

Architecture
------------
Two pieces of state live outside this module to keep this code hot-path
cheap:

1. ``state.get_canonical_keys(source_distmat)`` returns
       {"tuples":     list[tuple[int, ...]],
        "leaf_names": list[str]}
   for the distmat. Packed sparse clades (or legacy ``bipartition_bits``)
   are decoded to ``tuple[int]`` once per distmat per session, lazily on
   first use, so repeat comparisons pay no catalog-decode cost.

2. Each consensus tree registry entry carries
       counts:  np.int32 array, length n_bipartitions
       n_trees: int
   computed at consensus tree registration time. ``counts[j]`` is the number of
   trees in the user's consensus tree selection that contain bipartition ``j``;
   the frequency is just ``counts[j] / n_trees``. With this pre-
   computed, a Compare click does no row-sum work.

Same-distmat only
-----------------
Both consensus tree entries MUST share ``source_distmat``. The diagnostics UI
filters the Compare-clade dropdowns to the active RF matrix so this
is always true; ``compute_clade_frequencies`` raises ``ValueError``
on a mismatch as a defensive check. With shared distmat, both
``counts`` vectors index into the *same* rooted-clade column basis
and merging is a pure column-index alignment — no key hashing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import state


# Module-level export expected by older tests / callers.
def _bits_to_tip_indices(bipartition_bits: np.ndarray) -> list[tuple[int, ...]]:
    """Decode the ``(n_bipartitions, n_leaves) uint8`` matrix into a
    list of sorted ``tuple[int]`` of leaf indices on the canonical side.

    Standalone form of what ``state.get_canonical_keys`` does
    internally. Useful for tests and for benchmarks that need the
    decoder isolated from the cache.
    """
    return [tuple(np.flatnonzero(row).tolist()) for row in bipartition_bits]


def _normalise_counts(entry, source_distmat):
    """Return ``(counts, n_trees)`` for a registry entry, recomputing
    if the entry was registered without pre-computed counts (e.g. by
    an older session or a unit test).

    Pulls sparse rows, or legacy ``presence``, from the snapshot lazily —
    only happens on the slow path.
    """
    counts = entry.get("counts")
    n_trees = entry.get("n_trees") or 0
    if counts is not None and n_trees > 0:
        return np.asarray(counts), n_trees

    # Slow recompute path: do the row-sum on demand.
    snap_path = state.get_snapshots_path(source_distmat)
    full_names = state.get_distmat_names(source_distmat)
    name_to_idx = {n: i for i, n in enumerate(full_names)}
    tree_names = entry.get("tree_names") or []
    row_idx = [name_to_idx[n] for n in tree_names if n in name_to_idx]
    if not row_idx:
        raise KeyError(
            f"None of the consensus tree's tree names match {source_distmat}'s snapshot. "
            "The distmat may have been recomputed since the consensus tree was registered."
        )
    with np.load(snap_path, allow_pickle=False) as snap:
        from ..rf.sparse_snapshots import (
            count_sparse_columns,
            snapshot_has_sparse_presence,
            sparse_snapshot_from_npz,
        )

        if snapshot_has_sparse_presence(snap):
            sparse_snapshot = sparse_snapshot_from_npz(snap)
            if sparse_snapshot.tree_names != tuple(full_names):
                raise ValueError(
                    "persisted sparse-snapshot tree names disagree with "
                    "the RF registry ordering"
                )
            counts = count_sparse_columns(
                sparse_snapshot,
                row_idx,
            ).astype(np.int32)
        else:
            presence = snap["presence"]
            counts = presence[row_idx].sum(axis=0).astype(np.int32)
    return counts, len(row_idx)


def compute_clade_frequencies(entry1, entry2) -> pd.DataFrame:
    """Compute per-rooted-clade frequencies for two consensus tree registry entries.

    Both entries MUST share ``source_distmat`` — clade-frequency
    comparison is only meaningful for consensus trees computed from the same RF
    matrix, since column indices in rapidtrees' rooted-clade presence
    table are basis-specific to that matrix. The dropdowns in the
    Diagnostics UI enforce this by filtering consensus tree options to the active
    distmat (see ``callbacks.diagnostics.populate_consensus_tree_selects``); this
    function raises if a caller bypasses that filter.

    Args:
        entry1, entry2: consensus tree registry entries (dicts as returned by
            ``state.register_consensus_tree``). Each must carry at least
            ``source_distmat``; ``counts`` and ``n_trees`` are used
            when present, otherwise recomputed from the snapshot.

    Returns:
        DataFrame with columns:
            split_key   tuple[int]  rooted-clade descendant tip indices
                                    into ``state.get_canonical_keys(
                                    source_distmat)["leaf_names"]``.
            column_j    int         presence-matrix column index in the
                                    snapshot. Always populated.
            freq_1      float       frequency in group 1's trees.
            freq_2      float       frequency in group 2's trees.
            clade_size  int         ``len(split_key)``.
        Sorted by descending mean of the two frequencies.

    Raises:
        ValueError: if the two entries' ``source_distmat`` differ.
        KeyError, FileNotFoundError: on snapshot lookup failures.
    """
    src1 = entry1["source_distmat"]
    src2 = entry2["source_distmat"]
    if src1 != src2:
        raise ValueError(
            f"Cannot compare consensus trees from different RF matrices "
            f"({src1!r} vs {src2!r}); UI must filter the dropdowns to "
            f"the active distmat."
        )
    counts_1, n_trees_1 = _normalise_counts(entry1, src1)
    counts_2, n_trees_2 = _normalise_counts(entry2, src1)

    keys = state.get_canonical_keys(src1)
    tuples = keys["tuples"]

    # Column basis is identical, so ``counts_2`` indexes the same way
    # as ``counts_1`` — merge purely by column position.
    freqs_1 = counts_1.astype(np.float64) / np.float64(n_trees_1)
    freqs_2 = counts_2.astype(np.float64) / np.float64(n_trees_2)
    mask = (counts_1 > 0) | (counts_2 > 0)
    cols = np.flatnonzero(mask)
    rows = [
        {
            "split_key":  tuples[j],
            "column_j":   int(j),
            "freq_1":     float(freqs_1[j]),
            "freq_2":     float(freqs_2[j]),
            "clade_size": len(tuples[j]),
        }
        for j in cols
    ]
    df = pd.DataFrame(
        rows,
        columns=["split_key", "column_j", "freq_1", "freq_2", "clade_size"],
    )
    df["mean_freq"] = (df["freq_1"] + df["freq_2"]) / 2
    df = (
        df.sort_values("mean_freq", ascending=False)
          .drop(columns="mean_freq")
          .reset_index(drop=True)
    )
    return df
