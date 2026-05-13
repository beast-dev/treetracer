#!/usr/bin/env python3
"""Tests for the step-4 UI wiring (Clade Frequency Comparison panel).

Because these are Dash callbacks we cannot invoke them directly without a
running server.  Instead we test the two pieces of logic that are pure
functions and easy to isolate:

  1. The option-list builder (same logic as populate_mcc_selects, extracted
     so we can unit-test it without Dash).
  2. The button-enable logic (toggle_compare_button).
  3. That get_mcc_registry_index() produces output with the exact shape the
     dropdown builder expects.

For a full end-to-end smoke test of the UI:
  - Run the app, compute RF + MDS, select trees, click "View MCC" in either
    the Within-run or Between-run tab.
  - Open the Diagnostics tab.
  - Verify that both MCC dropdowns now list the saved MCC tree.
  - Select the same (or different) entries in both dropdowns.
  - Verify the "Compare Clade Frequencies" button becomes enabled.

Run from the repo root:
    uv run python src/test/test_clade_freq_ui.py
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
    state.clear_all_mcc_trees()


# ---------------------------------------------------------------------------
# Extracted pure-Python versions of the two callback bodies
# (identical logic to what lives in diagnostics.py)
# ---------------------------------------------------------------------------

def populate_mcc_selects(registry):
    """Mirror of the populate_mcc_selects callback body."""
    if not registry:
        return [], True, [], True
    options = [
        {
            "value": uid,
            "label": f"{meta['label']}  ({meta['n_trees']} trees)",
        }
        for uid, meta in registry.items()
    ]
    return options, False, options, False


def toggle_compare_button(uid1, uid2):
    """Mirror of the toggle_compare_button callback body."""
    return not (uid1 and uid2)


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
# 1. Empty registry → dropdowns disabled
# ---------------------------------------------------------------------------

section("1. Empty registry produces disabled dropdowns")
fresh_state()

opts1, dis1, opts2, dis2 = populate_mcc_selects({})
check(opts1 == [] and opts2 == [],
      "Empty registry → both option lists are empty")
check(dis1 is True and dis2 is True,
      "Empty registry → both dropdowns disabled")


# ---------------------------------------------------------------------------
# 2. One registry entry → dropdowns populated and enabled
# ---------------------------------------------------------------------------

section("2. One registry entry → dropdowns populated and enabled")
fresh_state()

uid_a = state.cache_mcc_tree(DUMMY_NEXUS)
state.register_mcc_result(uid=uid_a, **ENTRY_A)
registry = state.get_mcc_registry_index()

opts1, dis1, opts2, dis2 = populate_mcc_selects(registry)

check(len(opts1) == 1 and len(opts2) == 1,
      f"One entry → one option in each dropdown (got {len(opts1)})")
check(dis1 is False and dis2 is False,
      "One entry → both dropdowns enabled")
check(opts1[0]["value"] == uid_a,
      f"Option value is the uid: {opts1[0]['value']!r}")
check(ENTRY_A["label"] in opts1[0]["label"],
      f"Option label contains MCC label: {opts1[0]['label']!r}")
check(str(ENTRY_A["tree_names"].__len__()) in opts1[0]["label"],
      f"Option label contains n_trees: {opts1[0]['label']!r}")


# ---------------------------------------------------------------------------
# 3. Two registry entries → both options appear in each dropdown
# ---------------------------------------------------------------------------

section("3. Two registry entries → both options in each dropdown")
fresh_state()

uid_a = state.cache_mcc_tree(DUMMY_NEXUS)
state.register_mcc_result(uid=uid_a, **ENTRY_A)
uid_b = state.cache_mcc_tree(DUMMY_NEXUS)
state.register_mcc_result(uid=uid_b, **ENTRY_B)
registry = state.get_mcc_registry_index()

opts1, dis1, opts2, dis2 = populate_mcc_selects(registry)

check(len(opts1) == 2 and len(opts2) == 2,
      f"Two entries → two options in each dropdown (got {len(opts1)})")
uids_in_opts = {o["value"] for o in opts1}
check(uid_a in uids_in_opts and uid_b in uids_in_opts,
      "Both uids present in dropdown options")


# ---------------------------------------------------------------------------
# 4. Compare button logic
# ---------------------------------------------------------------------------

section("4. Compare button enable/disable logic")

check(toggle_compare_button(None, None) is True,
      "Both None → button disabled")
check(toggle_compare_button(uid_a, None) is True,
      "Only Group 1 selected → button disabled")
check(toggle_compare_button(None, uid_b) is True,
      "Only Group 2 selected → button disabled")
check(toggle_compare_button(uid_a, uid_b) is False,
      "Both selected → button enabled")
check(toggle_compare_button(uid_a, uid_a) is False,
      "Same uid in both dropdowns → button enabled (allowed)")


# ---------------------------------------------------------------------------
# 5. Registry index shape matches what populate_mcc_selects expects
# ---------------------------------------------------------------------------

section("5. Registry index shape is correct for the dropdown builder")
fresh_state()

uid_a = state.cache_mcc_tree(DUMMY_NEXUS)
state.register_mcc_result(uid=uid_a, **ENTRY_A)
registry = state.get_mcc_registry_index()

check(isinstance(registry, dict),
      "get_mcc_registry_index returns a dict")
entry = registry[uid_a]
check("label" in entry,
      "Registry index entry has 'label' key")
check("source_distmat" in entry,
      "Registry index entry has 'source_distmat' key")
check("n_trees" in entry,
      "Registry index entry has 'n_trees' key")
check("tree_names" not in entry,
      "Registry index entry does NOT expose 'tree_names' (kept server-side)")


# ---------------------------------------------------------------------------
# 6. After clear, dropdowns go back to disabled
# ---------------------------------------------------------------------------

section("6. After clear_all_mcc_trees, dropdowns revert to disabled state")

state.clear_all_mcc_trees()
registry = state.get_mcc_registry_index()

opts1, dis1, opts2, dis2 = populate_mcc_selects(registry)
check(opts1 == [] and opts2 == [],
      "After clear → option lists empty")
check(dis1 is True and dis2 is True,
      "After clear → dropdowns disabled")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'=' * 60}")
total = passed + failed
print(f"  Passed : {passed}/{total}")
print(f"  Failed : {failed}/{total}")
if failed == 0:
    print("\n  ALL TESTS PASSED — safe to proceed to step 5.")
else:
    print(f"\n  {failed} TEST(S) FAILED — fix before proceeding.")
print(f"{'=' * 60}\n")

sys.exit(0 if failed == 0 else 1)
