"""Pure-synthetic tests for ``treetracer.newick_layout`` — the Newick
parser, the x/y layout engine, and the NEXUS parser used by the
tanglegram.

No fixtures from ``conftest.py`` because everything here is built from
hand-written newick/NEXUS strings. Doesn't depend on rapidtrees,
DendroPy, or the bundled ``test.trees`` data file.
"""

from __future__ import annotations

import math

import pytest

from treetracer.clade_freq.layout import (
    _assign_layout,
    _collect_nodes,
    parse_newick,
    parse_nexus,
    build_tree_traces,
)


def _approx(a, b, tol=1e-9):
    return math.isclose(a, b, abs_tol=tol)


# ---------------------------------------------------------------------------
# 1. Newick parsing — basic structure
# ---------------------------------------------------------------------------


def test_basic_newick_4_tips_3_internals():
    newick = "((A:1,B:1):1,(C:2,D:2):2):0"
    root = parse_newick(newick)
    nodes = _collect_nodes(root)
    tips = [n for n in nodes if not n.children]
    internals = [n for n in nodes if n.children]
    assert len(tips) == 4
    assert len(internals) == 3
    assert {t.name for t in tips} == {"A", "B", "C", "D"}


def test_basic_newick_branch_lengths_preserved():
    newick = "((A:1,B:1):1,(C:2,D:2):2):0"
    root = parse_newick(newick)
    tips = {n.name: n for n in _collect_nodes(root) if not n.children}
    assert _approx(tips["A"].length, 1.0)
    assert _approx(tips["C"].length, 2.0)


def test_public_parser_forwards_strict_branch_length_validation():
    root = parse_newick(
        "(A:1,B:2):0",
        require_branch_lengths=True,
    )
    assert all(
        node.has_length
        for node in _collect_nodes(root)
        if node is not root
    )

    with pytest.raises(ValueError, match="explicit branch length required"):
        parse_newick("(A,B:2):0", require_branch_lengths=True)


def test_strict_parser_accepts_supported_labels_annotations_and_lengths():
    root = parse_newick(
        "[&R] ('A taxon'[&rate=2]:+1e-3,\"B\"\"taxon\":-2.5E+1):0;",
        require_branch_lengths=True,
    )

    tips = {node.name: node for node in _collect_nodes(root) if node.is_tip}
    assert set(tips) == {"A taxon", 'B"taxon'}
    assert tips["A taxon"].length == pytest.approx(0.001)
    assert tips['B"taxon'].length == pytest.approx(-25.0)


@pytest.mark.parametrize(
    ("newick", "message"),
    [
        ("(A:,B:1):0;", "missing value"),
        ("(A:not-a-number,B:1):0;", "malformed branch length"),
        ("(A:nan,B:1):0;", "non-finite branch length"),
        ("(A:inf,B:1):0;", "non-finite branch length"),
    ],
)
def test_strict_parser_rejects_invalid_branch_lengths(newick, message):
    with pytest.raises(ValueError, match=message):
        parse_newick(newick, require_branch_lengths=True)


def test_parser_handles_a_caterpillar_deeper_than_recursion_limit():
    n_taxa = 1_101
    newick = "(T0:1,T1:1):1"
    for index in range(2, n_taxa):
        newick = f"({newick},T{index}:1):1"

    root = parse_newick(newick + ";", require_branch_lengths=True)
    nodes = _collect_nodes(root)

    assert len(nodes) == 2 * n_taxa - 1
    assert sum(node.is_tip for node in nodes) == n_taxa


# ---------------------------------------------------------------------------
# 2. Metadata comments [&...] stripped during tokenisation
# ---------------------------------------------------------------------------


def test_metadata_comments_stripped():
    """BEAST-style annotation comments must not bleed into tip names
    or break the structural parse."""
    newick = "((A[&rate=0.1]:1,B[&rate=0.2]:1)[&posterior=0.9]:1,(C:2,D:2):2):0"
    root = parse_newick(newick)
    tips = [n for n in _collect_nodes(root) if not n.children]
    assert len(tips) == 4
    assert {t.name for t in tips} == {"A", "B", "C", "D"}


# ---------------------------------------------------------------------------
# 3. Translate map application
# ---------------------------------------------------------------------------


def test_translate_map_replaces_integer_tokens():
    """Integer tip labels are replaced by their translate-map values."""
    newick = "((1:1,2:1):1,(3:2,4:2):2):0"
    translate = {"1": "Taxon_A", "2": "Taxon_B", "3": "Taxon_C", "4": "Taxon_D"}
    root = parse_newick(newick, translate=translate)
    tips = [n for n in _collect_nodes(root) if n.is_tip]
    assert {t.name for t in tips} == {"Taxon_A", "Taxon_B", "Taxon_C", "Taxon_D"}


# ---------------------------------------------------------------------------
# 4. Layout — x positions cumulative
# ---------------------------------------------------------------------------


def test_x_positions_are_cumulative_branch_lengths():
    newick = "((A:1,B:3):2,(C:1,D:1):1):0"
    root = parse_newick(newick)
    for n in _collect_nodes(root):
        n.is_tip = not bool(n.children)
    _assign_layout(root)
    tip_by_name = {n.name: n for n in _collect_nodes(root) if n.is_tip}
    assert _approx(root.x, 0.0)
    assert _approx(tip_by_name["A"].x, 3.0)   # 0 + 2 (internal) + 1 (A)
    assert _approx(tip_by_name["B"].x, 5.0)   # 0 + 2 + 3
    assert _approx(tip_by_name["C"].x, 2.0)   # 0 + 1 + 1


# ---------------------------------------------------------------------------
# 5. Layout — y positions
# ---------------------------------------------------------------------------


def test_y_positions_tips_ranked_internals_at_child_mean():
    """Tips get integer y ranks (after ladderization); internal nodes
    sit at the mean y of their direct children."""
    newick = "((A:1,B:1):1,(C:1,D:1):1):0"
    root = parse_newick(newick)
    for n in _collect_nodes(root):
        n.is_tip = not bool(n.children)
    _assign_layout(root)
    tips = [n for n in _collect_nodes(root) if n.is_tip]
    tip_ys = sorted(t.y for t in tips)
    assert tip_ys == [0.0, 1.0, 2.0, 3.0]
    assert _approx(root.y, 1.5)  # mean of {0,1,2,3}


# ---------------------------------------------------------------------------
# 6. NEXUS parsing — translate + newick extraction
# ---------------------------------------------------------------------------


def test_nexus_parse_extracts_translate_and_tree():
    """``parse_nexus`` pulls the Translate block, the tree line, and
    runs layout. End-to-end smoke for the tanglegram parser path."""
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
    tips = [n for n in nodes if n.is_tip]
    assert len(translate) == 4
    assert translate["1"] == "Taxon_A"
    assert len(tips) == 4
    assert {t.name for t in tips} == {"Taxon_A", "Taxon_B", "Taxon_C", "Taxon_D"}
    assert _approx(root.x, 0.0)


def test_nexus_parse_strips_quoted_translate_values():
    """NEXUS allows quoted taxon names; ``parse_nexus`` strips the
    outer quotes so downstream tip-name lookups don't mismatch
    against rapidtrees' canonical leaf order (which we normalise
    the same way — see ``state.get_canonical_keys``)."""
    nexus = b"""#NEXUS
Begin trees;
    Translate
        1 'A_taxon_2024-01',
        2 'B_taxon_2024-02'
    ;
    tree t = (1:1,2:1):0;
End;
"""
    root, translate = parse_nexus(nexus)
    tips = [n for n in _collect_nodes(root) if n.is_tip]
    assert translate["1"] == "A_taxon_2024-01"   # quotes stripped
    assert {t.name for t in tips} == {"A_taxon_2024-01", "B_taxon_2024-02"}


def test_nexus_parse_strips_direct_quoted_tip_labels():
    """Some consensus tree Newick bodies contain quoted taxon names directly,
    rather than integer labels resolved through a Translate block.
    Those names must still match the unquoted canonical leaf names used
    by the clade-frequency click -> tanglegram lookup."""
    nexus = b"""#NEXUS
Begin trees;
    tree t = ('A_taxon_2024-01':1,'B_taxon_2024-02':1):0;
End;
"""
    root, _ = parse_nexus(nexus)
    tips = [n for n in _collect_nodes(root) if n.is_tip]
    assert {t.name for t in tips} == {"A_taxon_2024-01", "B_taxon_2024-02"}

    traces = build_tree_traces(root, highlight={"A_taxon_2024-01"})
    red = next(
        (t for t in traces if t.get("marker", {}).get("color") == "#e63946"),
        None,
    )
    assert red is not None
    assert red["text"] == ["A_taxon_2024-01"]


# ---------------------------------------------------------------------------
# 7. Trace builders
# ---------------------------------------------------------------------------


def test_build_tree_traces_returns_branches_and_highlight_markers():
    """``build_tree_traces`` emits at least: one line trace for the
    branches, and a red marker trace for the highlighted tips."""
    nexus = b"""#NEXUS
Begin trees;
    tree t = ((A:1,B:1):1,(C:1,D:1):1):0;
End;
"""
    root, _ = parse_nexus(nexus)
    traces = build_tree_traces(root, highlight={"A", "B"})
    assert len(traces) >= 2
    branch_trace = traces[0]
    assert branch_trace["mode"] == "lines"
    # Highlighted tips → red marker trace.
    red = next(
        (t for t in traces if t.get("marker", {}).get("color") == "#e63946"),
        None,
    )
    assert red is not None, "no highlighted-tip marker trace found"


def test_build_tree_traces_empty_highlight_returns_skeleton_only():
    """With no highlight set, every tip lands in the grey 'low' trace
    and there's no red marker trace. This is the skeleton form
    cached by ``_get_tanglegram_layout`` and patched in place by
    ``draw_tanglegram`` on every click."""
    nexus = b"""#NEXUS
Begin trees;
    tree t = ((A:1,B:1):1,(C:1,D:1):1):0;
End;
"""
    root, _ = parse_nexus(nexus)
    traces = build_tree_traces(root, highlight=set())
    red = [t for t in traces if t.get("marker", {}).get("color") == "#e63946"]
    assert red == [], "skeleton should have no red highlight trace"
