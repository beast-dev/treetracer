#!/usr/bin/env python3
"""Tests for the MCC result registry added to state.py.

Verifies that:
  - register_mcc_result stores all fields correctly
  - get_mcc_registry_index returns lightweight summaries (no tree_names)
  - get_mcc_registry_entry returns the full entry including tree_names
  - Multiple entries coexist correctly under distinct uids
  - clear_all_mcc_trees wipes both _mcc_cache and _mcc_registry
  - cache_mcc_tree + register_mcc_result round-trip is consistent

Run from the repo root:
    uv run python src/test/test_mcc_registry.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from treetracer import state

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


def fresh_state():
    """Wipe both caches so each test section starts clean."""
    state.clear_all_mcc_trees()


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

DUMMY_NEXUS = b"#NEXUS\nbegin trees;\ntree t1 = (A,(B,C));\nEnd;\n"

ENTRY_A = dict(
    label="run1/STATE_5000",
    source_distmat="RF_001",
    tree_names=["run1/STATE_1000", "run1/STATE_2000", "run1/STATE_5000"],
)

ENTRY_B = dict(
    label="run2/STATE_8000",
    source_distmat="RF_001",
    tree_names=["run2/STATE_6000", "run2/STATE_7000", "run2/STATE_8000"],
)


# ---------------------------------------------------------------------------
# 1. Basic round-trip
# ---------------------------------------------------------------------------

section("1. Basic round-trip: cache + register + retrieve")
fresh_state()

uid_a = state.cache_mcc_tree(DUMMY_NEXUS)
state.register_mcc_result(uid=uid_a, **ENTRY_A)

check(state.has_cached_mcc_tree(uid_a),
      "cache_mcc_tree: uid present in _mcc_cache")
check(state.get_cached_mcc_tree(uid_a) == DUMMY_NEXUS,
      "cache_mcc_tree: NEXUS bytes round-trip correctly")

entry = state.get_mcc_registry_entry(uid_a)
check(entry is not None,
      "get_mcc_registry_entry: entry exists after register_mcc_result")
check(entry["label"] == ENTRY_A["label"],
      f"label stored correctly: {entry['label']!r}")
check(entry["source_distmat"] == ENTRY_A["source_distmat"],
      f"source_distmat stored correctly: {entry['source_distmat']!r}")
check(entry["tree_names"] == ENTRY_A["tree_names"],
      f"tree_names stored correctly ({len(entry['tree_names'])} entries)")
check(entry["n_trees"] == len(ENTRY_A["tree_names"]),
      f"n_trees matches len(tree_names): {entry['n_trees']}")


# ---------------------------------------------------------------------------
# 2. get_mcc_registry_index — lightweight view
# ---------------------------------------------------------------------------

section("2. get_mcc_registry_index returns lightweight summaries")
fresh_state()

uid_a = state.cache_mcc_tree(DUMMY_NEXUS)
state.register_mcc_result(uid=uid_a, **ENTRY_A)

index = state.get_mcc_registry_index()
check(uid_a in index,
      "uid present in registry index")
check(set(index[uid_a].keys()) == {"label", "source_distmat", "n_trees"},
      f"index entry has exactly the lightweight keys: {set(index[uid_a].keys())}")
check("tree_names" not in index[uid_a],
      "tree_names NOT present in lightweight index (kept server-side only)")
check(index[uid_a]["n_trees"] == len(ENTRY_A["tree_names"]),
      f"n_trees correct in index: {index[uid_a]['n_trees']}")


# ---------------------------------------------------------------------------
# 3. Multiple entries coexist
# ---------------------------------------------------------------------------

section("3. Multiple MCC entries coexist under distinct uids")
fresh_state()

uid_a = state.cache_mcc_tree(DUMMY_NEXUS)
state.register_mcc_result(uid=uid_a, **ENTRY_A)

uid_b = state.cache_mcc_tree(DUMMY_NEXUS)
state.register_mcc_result(uid=uid_b, **ENTRY_B)

check(uid_a != uid_b,
      "Two cache calls produce distinct uids")

index = state.get_mcc_registry_index()
check(len(index) == 2,
      f"Registry index contains 2 entries: {len(index)}")

entry_a = state.get_mcc_registry_entry(uid_a)
entry_b = state.get_mcc_registry_entry(uid_b)
check(entry_a["label"] == ENTRY_A["label"],
      f"Entry A label correct: {entry_a['label']!r}")
check(entry_b["label"] == ENTRY_B["label"],
      f"Entry B label correct: {entry_b['label']!r}")
check(entry_a["tree_names"] != entry_b["tree_names"],
      "Entries A and B have distinct tree_names lists")


# ---------------------------------------------------------------------------
# 4. clear_all_mcc_trees wipes both cache and registry
# ---------------------------------------------------------------------------

section("4. clear_all_mcc_trees wipes both _mcc_cache and _mcc_registry")
fresh_state()

uid_a = state.cache_mcc_tree(DUMMY_NEXUS)
state.register_mcc_result(uid=uid_a, **ENTRY_A)
uid_b = state.cache_mcc_tree(DUMMY_NEXUS)
state.register_mcc_result(uid=uid_b, **ENTRY_B)

state.clear_all_mcc_trees()

check(not state.has_cached_mcc_tree(uid_a),
      "uid_a absent from _mcc_cache after clear")
check(not state.has_cached_mcc_tree(uid_b),
      "uid_b absent from _mcc_cache after clear")
check(state.get_mcc_registry_entry(uid_a) is None,
      "uid_a absent from _mcc_registry after clear")
check(state.get_mcc_registry_entry(uid_b) is None,
      "uid_b absent from _mcc_registry after clear")
check(state.get_mcc_registry_index() == {},
      "get_mcc_registry_index returns empty dict after clear")


# ---------------------------------------------------------------------------
# 5. Missing uid returns None gracefully
# ---------------------------------------------------------------------------

section("5. Querying a nonexistent uid returns None gracefully")
fresh_state()

check(state.get_mcc_registry_entry("nonexistent-uid") is None,
      "get_mcc_registry_entry returns None for unknown uid")
check(state.get_cached_mcc_tree("nonexistent-uid") is None,
      "get_cached_mcc_tree returns None for unknown uid")


# ---------------------------------------------------------------------------
# 6. tree_names list is a copy (mutations don't affect the stored entry)
# ---------------------------------------------------------------------------

section("6. Stored tree_names list is an independent copy")
fresh_state()

names = ["run1/STATE_1000", "run1/STATE_2000"]
uid_a = state.cache_mcc_tree(DUMMY_NEXUS)
state.register_mcc_result(
    uid=uid_a, label="test", source_distmat="RF_001", tree_names=names
)
names.append("run1/STATE_3000")   # mutate the original list

entry = state.get_mcc_registry_entry(uid_a)
check(len(entry["tree_names"]) == 2,
      "Stored tree_names unaffected by mutation of original list "
      f"(stored={len(entry['tree_names'])}, original now={len(names)})")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'=' * 60}")
total = passed + failed
print(f"  Passed : {passed}/{total}")
print(f"  Failed : {failed}/{total}")
if failed == 0:
    print("\n  ALL TESTS PASSED — safe to proceed to step 4.")
else:
    print(f"\n  {failed} TEST(S) FAILED — fix before proceeding.")
print(f"{'=' * 60}\n")

sys.exit(0 if failed == 0 else 1)
