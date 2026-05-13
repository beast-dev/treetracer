"""Clade frequency computation for the Clade Frequency Comparison feature.

Given two groups of trees (each identified by a registry entry containing
a source_distmat key and a list of tree names), this module computes
per-bipartition frequencies for each group and returns a combined DataFrame
suitable for the scatter plot.

Bipartition representation
--------------------------
Each bipartition is stored in the snapshot as a row of uint64 words
(``bipartition_bits``, shape ``(n_bipartitions, words_per_bitset)``).
Bit k of word w is set if leaf (w*64 + k) — in ``leaf_names`` order — is
on the canonical side of the split.  We convert each row to a frozenset of
tip names (the smaller of the two partitions) to get a canonical, order-
independent key that is comparable across distmats.

Cross-distmat correctness
--------------------------
Two independent RF computations assign bipartition column indices
independently.  By canonicalising on frozenset of tip names we can merge
frequencies from two different distmats correctly, filling 0.0 for
bipartitions absent from one group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import state


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# def _bits_to_tip_sets(bipartition_bits: np.ndarray,
#                       leaf_names: list[str]) -> list[frozenset[str]]:
#     """Convert the bipartition_bits array to a list of canonical frozensets.

#     Args:
#         bipartition_bits: (n_bipartitions, words_per_bitset) uint64 ndarray.
#         leaf_names:       Ordered list of taxon names (length = n_leaves).

#     Returns:
#         List of length n_bipartitions.  Element j is the frozenset of tip
#         names on the smaller side of bipartition j (canonical split key).
#     """
#     n_bipartitions, words_per_bitset = bipartition_bits.shape
#     n_leaves = len(leaf_names)
#     leaf_arr = np.array(leaf_names)
#     canonical: list[frozenset[str]] = []

#     for j in range(n_bipartitions):
#         # Reconstruct the set of leaf indices on the "1" side of this split.
#         on_side: list[int] = []
#         for w in range(words_per_bitset):
#             word = int(bipartition_bits[j, w])
#             if word == 0:
#                 continue
#             for bit in range(64):
#                 leaf_idx = w * 64 + bit
#                 if leaf_idx >= n_leaves:
#                     break
#                 if word & (1 << bit):
#                     on_side.append(leaf_idx)

#         #side_a = frozenset(leaf_arr[on_side].tolist())
#         #side_b = frozenset(leaf_arr) - side_a
#         # Canonical = smaller partition (ties broken by which side has fewer
#         # tips; consistent within a run because leaf_names order is fixed).
#         #canonical.append(side_a if len(side_a) <= len(side_b) else side_b)

#         side_a = frozenset(leaf_arr[on_side].tolist())
#         side_b = frozenset(leaf_arr) - side_a
#         # Match rapidtrees canonicalisation: always store the side NOT containing
#         # leaf_names[0] (the alphabetically first taxon).
#         first_leaf = leaf_arr[0]
#         canonical.append(side_a if first_leaf not in side_a else side_b)


#     return canonical


def _bits_to_tip_sets(bipartition_bits: np.ndarray,
                      leaf_names: list[str]) -> list[frozenset[str]]:
    """Convert the bipartition_bits array to a list of canonical frozensets.

    In rapidtrees 0.5.0+, bipartition_bits is already decoded:
    shape (n_bipartitions, n_leaves) uint8, where bipartition_bits[j, i] == 1
    means leaf_names[i] is on the canonical side of bipartition j.
    """
    leaf_arr = np.array(leaf_names)
    canonical: list[frozenset[str]] = []
    for j in range(bipartition_bits.shape[0]):
        on_side = leaf_arr[bipartition_bits[j] == 1].tolist()
        canonical.append(frozenset(on_side))
    return canonical


def _load_group(source_distmat: str,
                tree_names: list[str]) -> tuple[dict[frozenset, float],
                                                dict[frozenset, int]]:
    """Load a snapshot and compute per-bipartition frequencies for a group.

    Args:
        source_distmat: Key into state._distmat_index.
        tree_names:     Names of the trees that form this group.

    Returns:
        freq_dict:  {canonical_split: frequency}  — frequency in [0, 1].
        size_dict:  {canonical_split: clade_size} — size of smaller partition.

    Raises:
        KeyError:        if any tree name is not found in the snapshot index.
        FileNotFoundError: if the snapshot .npz file does not exist.
    """
    snap_path = state.get_snapshots_path(source_distmat)
    snap = np.load(snap_path, allow_pickle=False)

    presence         = snap["presence"]          # (n_trees, n_splits) uint8
    leaf_names       = [str(n) for n in snap["leaf_names"]]
    bipartition_bits = snap["bipartition_bits"]  # (n_splits, n_leaves) uint8

    full_names  = state.get_distmat_names(source_distmat)
    name_to_idx = {n: i for i, n in enumerate(full_names)}

    missing = [n for n in tree_names if n not in name_to_idx]
    if missing:
        raise KeyError(
            f"Trees not found in snapshot for {source_distmat}: {missing}"
        )

    row_idx      = [name_to_idx[n] for n in tree_names]
    presence_sub = presence[row_idx]              # (n_sel, n_splits) uint8
    n_sel        = len(row_idx)

    # Column-wise frequencies — one numpy op over the selected rows.
    counts = presence_sub.sum(axis=0).astype(np.float64)  # (n_splits,)
    freqs  = counts / n_sel                                # (n_splits,)

    # Decode bipartition bit-vectors into canonical frozenset keys.
    canonical_keys = _bits_to_tip_sets(bipartition_bits, leaf_names)

    freq_dict: dict[frozenset, float] = {}
    size_dict: dict[frozenset, int]   = {}

    for j, key in enumerate(canonical_keys):
        if counts[j] == 0:
            continue                              # absent from all selected trees
        freq_dict[key] = float(freqs[j])
        size_dict[key] = len(key)

    return freq_dict, size_dict


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_clade_frequencies(
    source_distmat_1: str,
    tree_names_1: list[str],
    source_distmat_2: str,
    tree_names_2: list[str],
) -> pd.DataFrame:
    """Compute per-bipartition frequencies for two groups of trees.

    Args:
        source_distmat_1: distmat key for group 1 (from registry entry).
        tree_names_1:     tree names in group 1.
        source_distmat_2: distmat key for group 2 (from registry entry).
        tree_names_2:     tree names in group 2.

    Returns:
        DataFrame with columns:
            split_key   — frozenset[str], canonical split (smaller partition)
            freq_1      — float in [0, 1], frequency in group 1
            freq_2      — float in [0, 1], frequency in group 2
            clade_size  — int, number of tips in the smaller partition

        One row per bipartition observed in either group.  Bipartitions
        absent from a group get 0.0 for that group.
        Sorted by descending mean frequency (most shared clades first).
    """
    freq_1, size_1 = _load_group(source_distmat_1, tree_names_1)
    freq_2, size_2 = _load_group(source_distmat_2, tree_names_2)

    all_splits   = set(freq_1.keys()) | set(freq_2.keys())
    # size_1 wins on key conflict — same frozenset = same clade = same size.
    size_combined = {**size_2, **size_1}

    rows = [
        {
            "split_key":  split,
            "freq_1":     freq_1.get(split, 0.0),
            "freq_2":     freq_2.get(split, 0.0),
            "clade_size": size_combined[split],
        }
        for split in all_splits
    ]

    df = pd.DataFrame(rows, columns=["split_key", "freq_1", "freq_2", "clade_size"])
    df["mean_freq"] = (df["freq_1"] + df["freq_2"]) / 2
    df = (df.sort_values("mean_freq", ascending=False)
            .drop(columns="mean_freq")
            .reset_index(drop=True))
    return df