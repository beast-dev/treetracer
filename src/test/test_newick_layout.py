#!/usr/bin/env python3
"""Tests for newick_layout.py — newick parser, layout engine, and NEXUS parser.

These tests are fully self-contained and require no running app or snapshot
files.  They use synthetic newick strings and NEXUS bytes.

Run from the repo root:
    uv run python src/test/test_newick_layout.py
"""

import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from treetracer.newick_layout import (
    _parse_newick, _apply_translate, _assign_layout,
    _collect_nodes, parse_nexus,
    build_tree_traces, build_connector_traces,
)

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


def approx(a, b, tol=1e-9):
    return math.isclose(a, b, abs_tol=tol)


# ---------------------------------------------------------------------------
# 1. Newick parsing — basic structure
# ---------------------------------------------------------------------------

section("1. Newick parsing — basic structure")

# Simple 4-tip tree: ((A:1,B:1):1,(C:2,D:2):2):0;
newick = "((A:1,B:1):1,(C:2,D:2):2):0"
root = _parse_newick(newick)
nodes = _collect_nodes(root)
tips = [n for n in nodes if not n.children]
internals = [n for n in nodes if n.children]

check(len(tips) == 4,
      f"4 tips parsed (got {len(tips)})")
check(len(internals) == 3,
      f"3 internal nodes parsed (got {len(internals)})")
check({t.name for t in tips} == {"A", "B", "C", "D"},
      f"Tip names correct: {{{', '.join(t.name for t in tips)}}}")

# Branch lengths
tip_by_name = {t.name: t for t in tips}
check(approx(tip_by_name["A"].length, 1.0),
      "A branch length = 1.0")
check(approx(tip_by_name["C"].length, 2.0),
      "C branch length = 2.0")


# ---------------------------------------------------------------------------
# 2. Newick parsing — metadata comments stripped
# ---------------------------------------------------------------------------

section("2. Metadata comments [&...] are stripped correctly")

newick_meta = "((A[&rate=0.1]:1,B[&rate=0.2]:1)[&posterior=0.9]:1,(C:2,D:2):2):0"
root = _parse_newick(newick_meta)
nodes = _collect_nodes(root)
tips = [n for n in nodes if not n.children]

check(len(tips) == 4,
      f"4 tips parsed with metadata comments (got {len(tips)})")
check({t.name for t in tips} == {"A", "B", "C", "D"},
      "Tip names correct after stripping comments")


# ---------------------------------------------------------------------------
# 3. Translate map application
# ---------------------------------------------------------------------------

section("3. Translate map application")

newick_tok = "((1:1,2:1):1,(3:2,4:2):2):0"
translate  = {"1": "Taxon_A", "2": "Taxon_B", "3": "Taxon_C", "4": "Taxon_D"}
root = _parse_newick(newick_tok)
for n in _collect_nodes(root):
    n.is_tip = not bool(n.children)
_apply_translate(root, translate)

tips = [n for n in _collect_nodes(root) if n.is_tip]
check({t.name for t in tips} == {"Taxon_A", "Taxon_B", "Taxon_C", "Taxon_D"},
      f"Translate map applied: {{{', '.join(t.name for t in tips)}}}")


# ---------------------------------------------------------------------------
# 4. Layout — x positions (cumulative branch length)
# ---------------------------------------------------------------------------

section("4. Layout — x positions are cumulative branch lengths")

newick = "((A:1,B:3):2,(C:1,D:1):1):0"
root = _parse_newick(newick)
for n in _collect_nodes(root):
    n.is_tip = not bool(n.children)
_assign_layout(root)

nodes = _collect_nodes(root)
tip_by_name = {n.name: n for n in nodes if n.is_tip}

check(approx(root.x, 0.0),
      "Root x = 0.0")
check(approx(tip_by_name["A"].x, 3.0),   # root(0) + internal(2) + A(1)
      f"Tip A x = 3.0 (got {tip_by_name['A'].x})")
check(approx(tip_by_name["B"].x, 5.0),   # root(0) + internal(2) + B(3)
      f"Tip B x = 5.0 (got {tip_by_name['B'].x})")
check(approx(tip_by_name["C"].x, 2.0),   # root(0) + internal(1) + C(1)
      f"Tip C x = 2.0 (got {tip_by_name['C'].x})")


# ---------------------------------------------------------------------------
# 5. Layout — y positions (tip ranks, internal = child mean)
# ---------------------------------------------------------------------------

section("5. Layout — y positions: tips ranked 0..n-1, internals at child mean")

newick = "((A:1,B:1):1,(C:1,D:1):1):0"
root = _parse_newick(newick)
for n in _collect_nodes(root):
    n.is_tip = not bool(n.children)
_assign_layout(root)

nodes = _collect_nodes(root)
tips = sorted([n for n in nodes if n.is_tip], key=lambda n: n.y)
tip_ys = [t.y for t in tips]

check(tip_ys == sorted(tip_ys),
      f"Tip y values are ordered: {tip_ys}")
check(set(tip_ys) == {0.0, 1.0, 2.0, 3.0},
      f"Tip y values are 0,1,2,3 (got {sorted(set(tip_ys))})")
check(approx(root.y, 1.5),
      f"Root y = mean of all tips = 1.5 (got {root.y})")


# ---------------------------------------------------------------------------
# 6. NEXUS parsing — translate block and newick extraction
# ---------------------------------------------------------------------------

section("6. NEXUS parsing — translate block and newick extraction")

nexus = b"""#NEXUS

Begin taxa;
    Dimensions ntax=4;
    Taxlabels
        Taxon_A Taxon_B Taxon_C Taxon_D
    ;
End;

Begin trees;
    Translate
        1 Taxon_A,
        2 Taxon_B,
        3 Taxon_C,
        4 Taxon_D
    ;
    tree STATE_1000 = ((1:1.0,2:1.0):0.5,(3:2.0,4:2.0):0.5):0.0;
End;
"""

root, translate = parse_nexus(nexus)
nodes = _collect_nodes(root)
tips  = [n for n in nodes if n.is_tip]

check(len(translate) == 4,
      f"Translate map has 4 entries (got {len(translate)})")
check(translate.get("1") == "Taxon_A",
      f"Translate['1'] = 'Taxon_A' (got {translate.get('1')!r})")
check({t.name for t in tips} == {"Taxon_A", "Taxon_B", "Taxon_C", "Taxon_D"},
      "Tip names resolved via translate map")
check(approx(root.x, 0.0),
      "Root x = 0.0 after layout")
check(len(tips) == 4,
      f"4 tips after NEXUS parse (got {len(tips)})")


# ---------------------------------------------------------------------------
# 7. build_tree_traces — basic structure
# ---------------------------------------------------------------------------

section("7. build_tree_traces — returns non-empty trace list")

nexus = b"""#NEXUS
Begin trees;
    tree t = ((A:1,B:1):1,(C:1,D:1):1):0;
End;
"""
root, _ = parse_nexus(nexus)
traces = build_tree_traces(root, highlight={"A", "B"})

check(len(traces) >= 2,
      f"At least 2 traces returned (branches + markers): {len(traces)}")

branch_trace = traces[0]
check(branch_trace["mode"] == "lines",
      "First trace is a line trace (branches)")

# Check highlighted tips get a separate trace with red colour.
highlight_trace = next(
    (t for t in traces if t.get("marker", {}).get("color") == "#e63946"), None
)
check(highlight_trace is not None,
      "Highlighted tip trace has red colour #e63946")


# ---------------------------------------------------------------------------
# 8. build_connector_traces — highlight and non-highlight connectors
# ---------------------------------------------------------------------------

section("8. build_connector_traces — connector line structure")

from treetracer.newick_layout import Node

# Manually construct two matching tip lists.
def make_tip(name, y):
    n = Node(name=name, y=y, is_tip=True)
    return n

tips_left  = [make_tip("A", 0.0), make_tip("B", 1.0),
               make_tip("C", 2.0), make_tip("D", 3.0)]
tips_right = [make_tip("A", 0.5), make_tip("B", 1.5),
               make_tip("C", 2.5), make_tip("D", 3.5)]

highlight = {"A", "B"}
traces = build_connector_traces(
    tips_left=tips_left, tips_right=tips_right,
    highlight=highlight, x_left=1.0, x_right=1.3,
)

check(len(traces) == 2,
      f"Two connector traces (highlighted + non-highlighted): {len(traces)}")

red_trace  = next((t for t in traces if "230,57,70" in t["line"]["color"]), None)
grey_trace = next((t for t in traces if "180,180,180" in t["line"]["color"]), None)
check(red_trace is not None,
      "Highlighted connector trace is red")
check(grey_trace is not None,
      "Non-highlighted connector trace is grey")

# Each connector = 3 points (x1, x2, None) per tip pair.
n_hi_connectors  = red_trace["x"].count(None)
n_lo_connectors  = grey_trace["x"].count(None)
check(n_hi_connectors == len(highlight),
      f"Red trace has {len(highlight)} connectors (got {n_hi_connectors})")
check(n_lo_connectors == len(tips_left) - len(highlight),
      f"Grey trace has {len(tips_left)-len(highlight)} connectors "
      f"(got {n_lo_connectors})")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'=' * 60}")
total = passed + failed
print(f"  Passed : {passed}/{total}")
print(f"  Failed : {failed}/{total}")
if failed == 0:
    print("\n  ALL TESTS PASSED — step 7 complete.")
else:
    print(f"\n  {failed} TEST(S) FAILED — fix before proceeding.")
print(f"{'=' * 60}\n")

sys.exit(0 if failed == 0 else 1)