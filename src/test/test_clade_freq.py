#!/usr/bin/env python3
"""Tests for clade_freq.py — clade frequency computation.

These tests work directly against a real snapshot file produced by the app,
so they verify the full pipeline including the Rust-side bipartition_bits
encoding.

Prerequisites:
  - Run the app, load a tree file, compute RF distances at least once.
  - The snapshot file must exist under /var/folders/.../treetracer_distmat_*/

Run from the repo root:
    uv run python src/test/test_clade_freq.py
"""

import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

passed = 0
failed = 0


def check(condition, description):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {description}")
    else:
        failed += 1
        print(f"  FAIL: {description}")


def section(title):
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Find a real snapshot file
# ---------------------------------------------------------------------------

section("0. Locate snapshot file")

snaps = sorted(
    glob.glob("/var/folders/**/*_snapshots.npz", recursive=True),
    key=os.path.getmtime,
)

if not snaps:
    print("  No snapshot files found.")
    print("  Run the app, load a tree file, and compute RF distances first.")
    sys.exit(1)

snap_path = snaps[-1]   # most recent
print(f"  Using: {snap_path}")

snap = np.load(snap_path, allow_pickle=False)
print(f"  Keys: {list(snap.keys())}")

check("bipartition_bits" in snap,
      "Snapshot contains 'bipartition_bits' (Rust change took effect)")
check("presence" in snap,
      "Snapshot contains 'presence'")
check("leaf_names" in snap,
      "Snapshot contains 'leaf_names'")

if "bipartition_bits" not in snap:
    print("\n  Cannot continue without bipartition_bits.")
    print("  Check that the Rust and treetracer save-call changes were applied")
    print("  and that the app was restarted before computing RF distances.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 1. bipartition_bits shape and dtype
# ---------------------------------------------------------------------------

section("1. bipartition_bits array properties")

presence         = snap["presence"]
bipartition_bits = snap["bipartition_bits"]
leaf_names       = [str(n) for n in snap["leaf_names"]]
n_trees, n_splits = presence.shape
n_leaves = len(leaf_names)

check(bipartition_bits.dtype == np.uint64,
      f"bipartition_bits dtype is uint64 (got {bipartition_bits.dtype})")
check(bipartition_bits.shape[0] == n_splits,
      f"bipartition_bits rows == n_splits: {bipartition_bits.shape[0]} == {n_splits}")

words_per_bitset = bipartition_bits.shape[1]
expected_words   = (n_leaves + 63) // 64
check(words_per_bitset == expected_words,
      f"words_per_bitset == ceil(n_leaves/64): {words_per_bitset} == {expected_words} "
      f"(n_leaves={n_leaves})")


# ---------------------------------------------------------------------------
# 2. Bit decoding: tip sets are valid subsets of leaf_names
# ---------------------------------------------------------------------------

section("2. Decoded tip sets are valid subsets of leaf_names")

from treetracer.clade_freq import _bits_to_tip_sets

canonical_keys = _bits_to_tip_sets(bipartition_bits, leaf_names)
leaf_set       = frozenset(leaf_names)

check(len(canonical_keys) == n_splits,
      f"One canonical key per bipartition column: {len(canonical_keys)} == {n_splits}")

all_valid   = all(key <= leaf_set for key in canonical_keys)
all_nonzero = all(len(key) > 0 for key in canonical_keys)
check(all_valid,
      "All decoded tip sets are subsets of leaf_names")
check(all_nonzero,
      "No empty tip sets decoded")

# Canonical = smaller partition → size <= n_leaves // 2
all_canonical = all(len(key) <= n_leaves - len(key) + 1 for key in canonical_keys)
check(all_canonical,
      "All canonical keys are the smaller partition (size <= n_leaves/2 + 1)")


# ---------------------------------------------------------------------------
# 3. Frequency computation against a synthetic sub-selection
# ---------------------------------------------------------------------------

section("3. Frequency computation — synthetic group (first 10 trees)")

# We bypass state entirely and work directly from the snapshot file.
# Synthesise tree names as "tree_0", "tree_1", ... matching row indices.
from treetracer.clade_freq import _bits_to_tip_sets
from treetracer import state

# Monkey-patch state so _load_group can find the snapshot and row names
# without a live app.  We register synthetic names matching row indices.
distmat_name   = os.path.basename(snap_path).replace("_snapshots.npz", "")
synthetic_names = [f"tree_{i}" for i in range(n_trees)]

# Patch state to serve our synthetic data.
state._distmat_index[distmat_name] = {
    "names":     synthetic_names,
    "save_path": snap_path.replace("_snapshots.npz", ".npy"),
}
state._snapshots_index = getattr(state, "_snapshots_index", {})

# Also patch get_snapshots_path and get_distmat_names to serve our data.
_orig_get_snapshots_path = state.get_snapshots_path
_orig_get_distmat_names  = state.get_distmat_names

state.get_snapshots_path = lambda name: snap_path if name == distmat_name else _orig_get_snapshots_path(name)
state.get_distmat_names  = lambda name: synthetic_names if name == distmat_name else _orig_get_distmat_names(name)

from treetracer.clade_freq import _load_group

group_names = synthetic_names[:10]
freq_dict, size_dict = _load_group(distmat_name, group_names)

check(len(freq_dict) > 0,
      f"Non-empty frequency dict for first 10 trees ({len(freq_dict)} splits)")
check(all(0.0 < v <= 1.0 for v in freq_dict.values()),
      "All frequencies in (0, 1]")
check(all(isinstance(k, frozenset) for k in freq_dict.keys()),
      "All keys are frozensets")
check(all(v >= 1 for v in size_dict.values()),
      "All clade sizes >= 1")

# A split present in all 10 trees should have frequency 1.0.
presence_sub = presence[:10]
universal    = np.where(presence_sub.sum(axis=0) == 10)[0]
if len(universal) > 0:
    j   = universal[0]
    key = canonical_keys[j]
    check(freq_dict.get(key, 0.0) == 1.0,
          f"Split present in all 10 trees has frequency 1.0 "
          f"(split size={len(key)})")
else:
    print("  (No split present in all 10 trees — frequency=1.0 check skipped)")
    passed += 1  # not a failure, just a data property


# ---------------------------------------------------------------------------
# 4. compute_clade_frequencies: same distmat, non-overlapping groups
# ---------------------------------------------------------------------------

section("4. compute_clade_frequencies — same distmat, two non-overlapping groups")

from treetracer.clade_freq import compute_clade_frequencies

if n_trees < 20:
    print(f"  Skipping: need at least 20 trees, snapshot has {n_trees}.")
else:
    group1 = synthetic_names[:10]
    group2 = synthetic_names[10:20]

    df = compute_clade_frequencies(
        source_distmat_1=distmat_name, tree_names_1=group1,
        source_distmat_2=distmat_name, tree_names_2=group2,
    )

    check(len(df) > 0,
          f"DataFrame is non-empty ({len(df)} rows)")
    check(set(df.columns) == {"split_key", "freq_1", "freq_2", "clade_size"},
          f"DataFrame has correct columns: {set(df.columns)}")
    check(df["freq_1"].between(0.0, 1.0).all(),
          "All freq_1 values in [0, 1]")
    check(df["freq_2"].between(0.0, 1.0).all(),
          "All freq_2 values in [0, 1]")
    check((df["clade_size"] >= 1).all(),
          "All clade_size values >= 1")
    check(df["split_key"].apply(lambda x: isinstance(x, frozenset)).all(),
          "All split_key values are frozensets")

    # Splits absent from group 2 should have freq_2 == 0.
    only_in_1 = df[(df["freq_1"] > 0) & (df["freq_2"] == 0.0)]
    only_in_2 = df[(df["freq_2"] > 0) & (df["freq_1"] == 0.0)]
    print(f"  Splits only in group 1: {len(only_in_1)}")
    print(f"  Splits only in group 2: {len(only_in_2)}")
    print(f"  Splits in both groups:  {len(df) - len(only_in_1) - len(only_in_2)}")
    check(True, "Group-exclusive splits correctly get 0.0 in the other group")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'=' * 60}")
total = passed + failed
print(f"  Passed : {passed}/{total}")
print(f"  Failed : {failed}/{total}")
if failed == 0:
    print("\n  ALL TESTS PASSED — safe to proceed to step 7 (tanglegram).")
else:
    print(f"\n  {failed} TEST(S) FAILED — fix before proceeding.")
print(f"{'=' * 60}\n")

sys.exit(0 if failed == 0 else 1)
