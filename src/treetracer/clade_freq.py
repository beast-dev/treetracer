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

Same-distmat fast path
----------------------
When both MCC entries share a ``source_distmat`` (the common case —
comparing a Between and a Within MCC built from the same RF run), both
``counts`` vectors index into the *same* bipartition column basis.
We merge by column index — no frozenset / tuple hashing needed.

Cross-distmat fallback
----------------------
When the two distmats have the same leaf set, int-tuple keys are
comparable (rapidtrees sorts leaves alphabetically, so the integer
indices mean the same taxa across runs). When leaf sets differ, we
fall back to merging by frozenset-of-taxon-names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import state


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
    """Compute per-bipartition frequencies for two MCC registry entries.

    Args:
        entry1, entry2: MCC registry entries (dicts as returned by
            ``state.register_mcc``). Each must carry at least
            ``source_distmat``; ``counts`` and ``n_trees`` are used
            when present, otherwise recomputed from the snapshot.

    Returns:
        DataFrame with columns:
            split_key   tuple[int] | frozenset[str]  canonical side tip
                                    indices into leaf_names_1, or (for
                                    cross-distmat-only splits) a
                                    frozenset of taxon names.
            column_j    int | None  presence-matrix column index in
                                    ``entry1.source_distmat``'s snapshot
                                    (same-distmat fast path); ``None``
                                    for the cross-distmat fallback.
            freq_1      float       frequency in group 1's trees
            freq_2      float       frequency in group 2's trees
            clade_size  int         len(split_key)
        Sorted by descending mean of the two frequencies.

    Raises:
        KeyError, FileNotFoundError on snapshot lookup failures.
    """
    src1 = entry1["source_distmat"]
    src2 = entry2["source_distmat"]
    counts_1, n_trees_1 = _normalise_counts(entry1, src1)
    counts_2, n_trees_2 = _normalise_counts(entry2, src2)

    keys_1 = state.get_canonical_keys(src1)

    if src1 == src2:
        # Fast path: column basis is identical, so freq_2 indexes the
        # same way as freq_1. No key matching at all.
        freqs_1 = counts_1.astype(np.float64) / n_trees_1
        freqs_2 = counts_2.astype(np.float64) / n_trees_2
        mask = (counts_1 > 0) | (counts_2 > 0)
        cols = np.flatnonzero(mask)
        tuples = keys_1["tuples"]
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
    else:
        # Cross-distmat fallback: merge by frozenset of taxon names so
        # different column orderings between the two distmats line up.
        keys_2 = state.get_canonical_keys(src2)
        leaf_names_1 = keys_1["leaf_names"]
        leaf_names_2 = keys_2["leaf_names"]

        def names_of(idx_tuple, leaf_names):
            return frozenset(leaf_names[i] for i in idx_tuple)

        freq_1 = {}
        for j, idx_tuple in enumerate(keys_1["tuples"]):
            if counts_1[j] == 0:
                continue
            freq_1[names_of(idx_tuple, leaf_names_1)] = (
                float(counts_1[j]) / n_trees_1, idx_tuple,
            )
        freq_2 = {}
        for j, idx_tuple in enumerate(keys_2["tuples"]):
            if counts_2[j] == 0:
                continue
            freq_2[names_of(idx_tuple, leaf_names_2)] = float(counts_2[j]) / n_trees_2

        # Use group-1 leaf order for split_key indices so the click→
        # tanglegram path always resolves against ``entry1``'s leaves.
        rows = []
        for name_set, (f1, idx_tuple) in freq_1.items():
            rows.append({
                "split_key":  idx_tuple,
                "column_j":   None,
                "freq_1":     f1,
                "freq_2":     freq_2.get(name_set, 0.0),
                "clade_size": len(idx_tuple),
            })
        only_in_2 = set(freq_2) - set(freq_1)
        # For splits only in group 2 we don't have group-1 indices.
        # Fabricate an indirect representation: store the name-set as
        # a 'split_key' synonym (downstream callers should handle the
        # ``isinstance(split_key, frozenset)`` branch when needed).
        for name_set in only_in_2:
            rows.append({
                "split_key":  name_set,
                "column_j":   None,
                "freq_1":     0.0,
                "freq_2":     freq_2[name_set],
                "clade_size": len(name_set),
            })
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
