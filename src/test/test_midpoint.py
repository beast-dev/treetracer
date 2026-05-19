"""Tests for ``treetracer.mcc._midpoint``.

Verifies the midpoint property: after re-rooting, the two leaves at
the ends of the longest path are equidistant from the new root.
That equidistance IS the definition of midpoint rooting, so it's the
right invariant to assert on.
"""

from __future__ import annotations

import pytest

from treetracer.mcc._midpoint import (
    midpoint_root_newick, _parse_newick, _leaves, _farthest_leaf,
)


def _root_to_leaf_distance(root, target_name):
    visited = {id(root)}
    stack = [(root, 0.0)]
    while stack:
        node, d = stack.pop()
        if node.name == target_name:
            return d
        for child, w in node.children:
            if id(child) not in visited:
                visited.add(id(child))
                stack.append((child, d + w))
    raise KeyError(f"leaf {target_name!r} not found")


def _diameter_endpoints(root):
    leaves = _leaves(root)
    u, _ = _farthest_leaf(leaves[0])
    v, D = _farthest_leaf(u)
    return u, v, D


def test_3_leaves_passthrough():
    # No meaningful midpoint with <3 leaves of structure — the function
    # returns the input unchanged for star-like trees at this size.
    nwk = "(A:0.5,B:0.5);"
    assert midpoint_root_newick(nwk) == nwk


def test_asymmetric_4leaf_midpoint_balanced():
    """Diameter is D↔A = 1.3; midpoint at 0.65 from each."""
    nwk = "(A:0.1,(B:0.1,(C:0.1,D:1.0):0.1):0.1);"
    out = midpoint_root_newick(nwk)
    root = _parse_newick(out)
    u, v, _ = _diameter_endpoints(root)
    d_u = _root_to_leaf_distance(root, u.name)
    d_v = _root_to_leaf_distance(root, v.name)
    assert d_u == pytest.approx(d_v, abs=1e-9)
    assert d_u == pytest.approx(0.65, abs=1e-9)


def test_balanced_tree_root_at_centre():
    """Already-balanced tree: midpoint should sit on the original root edge."""
    nwk = "((A:1,B:1):1,(C:1,D:1):1);"
    out = midpoint_root_newick(nwk)
    root = _parse_newick(out)
    # All leaves should be equidistant (root_to_X is 2.0 for each).
    for leaf in _leaves(root):
        d = _root_to_leaf_distance(root, leaf.name)
        assert d == pytest.approx(2.0, abs=1e-9), \
            f"{leaf.name} is at {d}, expected 2.0"


def test_polytomy_passes_through():
    """Trifurcation at the root with unequal branches — midpoint walks
    the longest pair, doesn't choke on the 3-child node."""
    nwk = "(A:0.1,B:0.2,(C:0.1,D:0.5):0.1);"
    out = midpoint_root_newick(nwk)
    root = _parse_newick(out)
    u, v, D = _diameter_endpoints(root)
    d_u = _root_to_leaf_distance(root, u.name)
    d_v = _root_to_leaf_distance(root, v.name)
    assert d_u == pytest.approx(d_v, abs=1e-9), \
        f"diameter endpoints not balanced: {u.name}={d_u}, {v.name}={d_v}"


def test_zero_length_branches_robust():
    """Trees with zero-length internal branches shouldn't crash; the
    midpoint algorithm handles zero edges by skipping over them."""
    nwk = "((A:1,B:0):0,(C:0,D:1):0);"
    out = midpoint_root_newick(nwk)
    root = _parse_newick(out)
    u, v, D = _diameter_endpoints(root)
    d_u = _root_to_leaf_distance(root, u.name)
    d_v = _root_to_leaf_distance(root, v.name)
    assert d_u == pytest.approx(d_v, abs=1e-9)
