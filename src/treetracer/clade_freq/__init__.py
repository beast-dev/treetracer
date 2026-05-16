"""Clade frequency computation for the Clade Frequency Comparison feature.

Consumes the per-distmat snapshot written by ``rf._worker.compute_rf``
(``presence``, ``leaf_names``, ``bipartition_bits``) and produces a
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
   for the distmat. The expensive ``bipartition_bits → tuple[int]``
   decode happens once per distmat per session (lazily on first call),
   so a Compare click pays at most one decode for a *new* distmat and
   zero for repeat clicks on the same distmat.

2. Each MCC registry entry carries
       counts:  np.int32 array, length n_bipartitions
       n_trees: int
   computed at MCC registration time. ``counts[j]`` is the number of
   trees in the user's MCC selection that contain bipartition ``j``;
   the frequency is just ``counts[j] / n_trees``. With this pre-
   computed, a Compare click does no row-sum work.

Same-distmat only
-----------------
Both MCC entries MUST share ``source_distmat``. The diagnostics UI
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

    Pulls ``presence`` from the snapshot lazily — only happens on the
    slow path.
    """
    counts = entry.get("counts")
    n_trees = entry.get("n_trees") or 0
    if counts is not None and n_trees > 0:
        return np.asarray(counts), n_trees

    # Slow recompute path: do the row-sum on demand.
    snap_path = state.get_snapshots_path(source_distmat)
    snap = np.load(snap_path, allow_pickle=False)
    presence = snap["presence"]
    full_names = state.get_distmat_names(source_distmat)
    name_to_idx = {n: i for i, n in enumerate(full_names)}
    tree_names = entry.get("tree_names") or []
    row_idx = [name_to_idx[n] for n in tree_names if n in name_to_idx]
    if not row_idx:
        raise KeyError(
            f"None of the MCC's tree names match {source_distmat}'s snapshot. "
            "The distmat may have been recomputed since the MCC was registered."
        )
    counts = presence[row_idx].sum(axis=0).astype(np.int32)
    return counts, len(row_idx)


def compute_clade_frequencies(entry1, entry2) -> pd.DataFrame:
    """Compute per-rooted-clade frequencies for two MCC registry entries.

    Both entries MUST share ``source_distmat`` — clade-frequency
    comparison is only meaningful for MCCs computed from the same RF
    matrix, since column indices in rapidtrees' rooted-clade presence
    table are basis-specific to that matrix. The dropdowns in the
    Diagnostics UI enforce this by filtering MCC options to the active
    distmat (see ``callbacks.diagnostics.populate_mcc_selects``); this
    function raises if a caller bypasses that filter.

    Args:
        entry1, entry2: MCC registry entries (dicts as returned by
            ``state.register_mcc``). Each must carry at least
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
            f"Cannot compare MCCs from different RF matrices "
            f"({src1!r} vs {src2!r}); UI must filter the dropdowns to "
            f"the active distmat."
        )
    counts_1, n_trees_1 = _normalise_counts(entry1, src1)
    counts_2, n_trees_2 = _normalise_counts(entry2, src1)

    keys = state.get_canonical_keys(src1)
    tuples = keys["tuples"]

    # Column basis is identical, so ``counts_2`` indexes the same way
    # as ``counts_1`` — merge purely by column position.
    freqs_1 = counts_1.astype(np.float64) / n_trees_1
    freqs_2 = counts_2.astype(np.float64) / n_trees_2
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
