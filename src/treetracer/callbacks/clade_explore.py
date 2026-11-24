"""Clade Exploration tab — consensus-tree clade-frequency comparison.

Split out of ``callbacks/diagnostics.py``: the clade-frequency scatter
+ tanglegram is exploratory phylogenetics, not a convergence
diagnostic, so it lives in its own tab and its own module. This module
owns the tab's RF-matrix selector, the per-matrix consensus tree table, the
two-consensus-tree Compare workflow, and the scatter / tanglegram rendering.
"""

import functools
from dataclasses import dataclass
from functools import partial
from typing import Any

from dash import dcc, html, callback, Input, Output, State, no_update, Patch, ALL, ctx
import dash_mantine_components as dmc
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from .. import state
from ..background_jobs import JobBusyError, JobRef, JobState, job_manager
from ..clade_freq.layout import parse_nexus, build_tree_traces, _collect_nodes
from ..clade_freq.sister_instability import format_tip_instability_tsv
from ..icons import icon
from ..logger import add_log, notif_id
from ..theme import get_template, DARK_TEMPLATE
from ..plot_utils import retheme_figure
from ..ui.widgets import stop_button
from . import persistent_worker
from ._helpers import _save_file_dialog
from .compute import _get_executor
from .job_reconcile import (
    is_compute_busy,
    terminal_delivery_marker,
    terminal_event_for_job,
)


# ---------------------------------------------------------------------------
# Tanglegram caches
# ---------------------------------------------------------------------------
# Re-parsing the consensus tree NEXUS bytes on every scatter click was the dominant
# cost of ``draw_tanglegram`` for big trees (~280 taxa = 100s of ms per
# click). These two caches plus a deterministic trace layout in the
# tanglegram figure let the callback Patch only the dynamic traces
# (highlight markers + connectors) when the consensus tree pair hasn't changed.


@functools.lru_cache(maxsize=64)
def _get_parsed_consensus_tree(uid):
    """Parse the cached NEXUS for a consensus tree uuid into a laid-out Node tree.

    Cached so repeat clicks on a tanglegram don't re-parse the same
    NEXUS file. Keyed on uuid — when a consensus tree is dropped from the LRU
    cache its uuid is recycled, but since the cached_consensus_tree key
    space is random-tokens, false hits are impossibly rare.
    """
    nexus = state.get_cached_consensus_tree(uid)
    if nexus is None:
        return None
    root, _translate = parse_nexus(nexus)
    return root


def _compute_yspans(root):
    """Map ``id(node)`` -> ``(y_lo, y_hi)``, the y-range of a node's
    descendant tips.

    The layout ranks tips by an in-order traversal, so every subtree's
    tips form one unbroken y-band. That turns an MRCA lookup into an
    O(depth) top-down descent — see ``_find_mrca``.
    """
    spans = {}

    def walk(node):
        if not node.children:
            spans[id(node)] = (node.y, node.y)
            return node.y, node.y
        lo = hi = None
        for child in node.children:
            clo, chi = walk(child)
            lo = clo if lo is None else min(lo, clo)
            hi = chi if hi is None else max(hi, chi)
        spans[id(node)] = (lo, hi)
        return lo, hi

    walk(root)
    return spans


def _find_mrca(root, spans, ymin, ymax):
    """Deepest node whose descendant-tip y-band covers ``[ymin, ymax]``
    — the MRCA of the tip set with that y-extent. Siblings have
    disjoint bands, so at most one child can cover the range; when none
    does, the set straddles this node's children and the node itself is
    the MRCA."""
    node = root
    while node.children:
        nxt = None
        for child in node.children:
            lo, hi = spans[id(child)]
            if lo <= ymin and hi >= ymax:
                nxt = child
                break
        if nxt is None:
            break
        node = nxt
    return node


def _subtree_branch_segments(mrca, x_offset, x_scale, x_flip):
    """L-shaped branch segments (in plot coords) for the subtree rooted
    at ``mrca`` — the same rectangular layout ``build_tree_traces``
    draws, restricted to the subtree so it can be recoloured as an
    overlay on top of the dimmed skeleton."""
    def tx(x):
        scaled = x * x_scale
        return (x_offset - scaled) if x_flip else (x_offset + scaled)

    xs, ys = [], []
    for node in _collect_nodes(mrca):
        for child in node.children:
            xs += [tx(node.x), tx(child.x), None]
            ys += [child.y, child.y, None]
        if node.children:
            child_ys = [c.y for c in node.children]
            xs += [tx(node.x), tx(node.x), None]
            ys += [min(child_ys), max(child_ys), None]
    return xs, ys


def _compute_node_sisters(root):
    """Map node id (and tip name) -> ``(parent, siblings)`` for O(1) sister lookup."""
    sisters = {}
    for node in _collect_nodes(root):
        if not node.children:
            continue
        for child in node.children:
            entry = (node, [sib for sib in node.children if sib is not child])
            sisters[id(child)] = entry
            if child.is_tip:
                sisters[child.name] = entry
    return sisters


def _sister_branch_segments(parent, siblings, x_offset, x_scale, x_flip):
    """Rectangular branch segments for a tip's immediate sister subtree(s)."""
    def tx(x):
        scaled = x * x_scale
        return (x_offset - scaled) if x_flip else (x_offset + scaled)

    xs, ys = [], []
    px = tx(parent.x)
    for sib in siblings:
        # Stem edge back to parent ensures single-tip (cherry) sisters draw visibly.
        xs += [px, tx(sib.x), None]
        ys += [sib.y, sib.y, None]
        if sib.children:
            sub_x, sub_y = _subtree_branch_segments(sib, x_offset, x_scale, x_flip)
            xs += sub_x
            ys += sub_y
    return xs, ys


def _branch_overlay_trace(xs, ys, color):
    return dict(
        type="scatter",
        x=xs,
        y=ys,
        mode="lines",
        line=dict(color=color, width=2),
        hoverinfo="skip",
        showlegend=False,
    )


@functools.lru_cache(maxsize=32)
def _get_tanglegram_layout(uid1, uid2):
    """Pre-computed layout values that don't depend on which clade is
    highlighted — scales, tip plot-coordinates, plot height.

    Returned dict keys:
        root1, root2    laid-out Node roots (from the parsed cache)
        scale1, scale2  per-tree x scale so both trees fit in [0, 1]
        right_start     x_offset of the right tree (gap + 1.0)
        tips1, tips2    dict[name, (plot_x, plot_y)] for fast highlight
                        lookups; plot_x is post-scale, post-flip
        max_y           tallest tip y across both trees, drives height
        skeleton_traces 4 static traces (left branches+all-tips,
                        right branches+all-tips); branches dimmed
        yspan1, yspan2  id(node) -> (y_lo, y_hi) per tree, for O(depth)
                        MRCA descent
    """
    root1 = _get_parsed_consensus_tree(uid1)
    root2 = _get_parsed_consensus_tree(uid2)
    if root1 is None or root2 is None:
        return None

    nodes1 = _collect_nodes(root1)
    nodes2 = _collect_nodes(root2)
    max_x1 = max(n.x for n in nodes1) or 1.0
    max_x2 = max(n.x for n in nodes2) or 1.0
    scale1 = 1.0 / max_x1
    scale2 = 1.0 / max_x2

    # GAP is the empty middle band where the red tip-connector lines
    # live. Bigger feels less cramped on dense (~280 tip) trees and
    # gives the eye room to follow each line.
    GAP = 0.6
    right_start = 1.0 + GAP

    # Skeleton: ``build_tree_traces(..., highlight=set())`` returns
    # exactly 2 traces (branches + all-grey-tips) since no tip lands in
    # the empty highlight set. That's our static base. Cast to plain
    # dicts for cleaner Patch interaction downstream.
    left_skeleton = build_tree_traces(
        root1, x_offset=0.0, x_scale=scale1, x_flip=False,
        highlight=set(),
    )
    right_skeleton = build_tree_traces(
        root2, x_offset=right_start + 1.0, x_scale=scale2, x_flip=True,
        highlight=set(),
    )

    def tip_plot_x(tip, x_offset, x_scale, x_flip):
        scaled = tip.x * x_scale
        return (x_offset - scaled) if x_flip else (x_offset + scaled)

    tips1 = {
        n.name: (tip_plot_x(n, 0.0, scale1, False), n.y)
        for n in nodes1 if n.is_tip
    }
    tips2 = {
        n.name: (tip_plot_x(n, right_start + 1.0, scale2, True), n.y)
        for n in nodes2 if n.is_tip
    }
    max_y = max(
        max((n.y for n in nodes1 if n.is_tip), default=0),
        max((n.y for n in nodes2 if n.is_tip), default=0),
    )

    # Dim the branch skeleton (trace 0 = left, 2 = right) so the
    # recoloured MRCA-subtree overlay drawn on top reads as the focus.
    # Grey tip markers (1, 3) are left as-is.
    skeleton = list(left_skeleton) + list(right_skeleton)
    skeleton[0]["line"]["color"] = _SKELETON_DIM
    skeleton[2]["line"]["color"] = _SKELETON_DIM

    return {
        "root1": root1,
        "root2": root2,
        "scale1": scale1,
        "scale2": scale2,
        "right_start": right_start,
        "tips1": tips1,
        "tips2": tips2,
        "max_y": max_y,
        "skeleton_traces": skeleton,
        "yspan1": _compute_yspans(root1),
        "yspan2": _compute_yspans(root2),
        "sisters1": _compute_node_sisters(root1),
        "sisters2": _compute_node_sisters(root2),
    }


# Trace indices in the assembled tanglegram figure (branches and sister
# connectors come before the red tip/connector overlays so the selected tip
# always draws cleanly on top). Skeleton indices (0, 1, 5, 6) are static.
#
#   0: left branches          (static, dimmed)
#   1: right branches         (static, dimmed)
#   2: clade MRCA branches    (dynamic) — blue, clade mode only
#   3: G1 sister branches     (dynamic) — blue, origin + monophyletic landing
#   4: G2 sister branches     (dynamic) — green, origin + monophyletic landing
#   5: left grey tips         (static)
#   6: right grey tips        (static)
#   7: blue G1 sister conns   (dynamic) — thin low-alpha, Connect sister ON
#   8: green G2 sister conns  (dynamic) — thin low-alpha, Connect sister ON
#   9: blue G1 sister tips    (dynamic) — origin; +right landing when ON
#  10: green G2 sister tips   (dynamic) — origin; +left landing when ON
#  11: left red highlight     (dynamic)
#  12: right red highlight    (dynamic)
#  13: red connectors         (dynamic)
#  14: clade MRCA markers     (dynamic) — clade mode only
#  15: G1 sister MRCA marker  (dynamic) — blue diamond
#  16: G2 sister MRCA marker  (dynamic) — green diamond
#  17: complement tips        (dynamic) — green, clade mode only
#  18: complement connectors  (dynamic) — green, clade mode only
_TANGLEGRAM_MRCA_SUBTREE          = 2
_TANGLEGRAM_SISTER_BRANCH_LEFT    = 3
_TANGLEGRAM_SISTER_BRANCH_RIGHT   = 4
_TANGLEGRAM_TIP_SISTER_CONN_LEFT  = 7
_TANGLEGRAM_TIP_SISTER_CONN_RIGHT = 8
_TANGLEGRAM_TIP_SISTER_TIPS_LEFT  = 9
_TANGLEGRAM_TIP_SISTER_TIPS_RIGHT = 10
_TANGLEGRAM_HIGHLIGHT_LEFT        = 11
_TANGLEGRAM_HIGHLIGHT_RIGHT       = 12
_TANGLEGRAM_CONNECTORS            = 13
_TANGLEGRAM_MRCA_MARKER           = 14
_TANGLEGRAM_SISTER_MRCA_LEFT      = 15
_TANGLEGRAM_SISTER_MRCA_RIGHT     = 16
_TANGLEGRAM_COMPLEMENT_TIPS       = 17
_TANGLEGRAM_COMPLEMENT_CONN       = 18

# Colours for the MRCA and tip-sister emphasis overlays.
_SKELETON_DIM            = "#c8c8c8"   # branches outside the highlighted subtree
_MRCA_ACCENT             = "#1c7ed6"   # blue: MRCA subtree or G1 consensus sister
_TIP_SISTER_ACCENT       = "#2f9e44"   # green: G2 consensus sister
_TIP_SISTER_CONN_LEFT    = "rgba(28,126,214,0.35)"
_TIP_SISTER_CONN_RIGHT   = "rgba(47,158,68,0.35)"
_MRCA_MARKER             = "#1864ab"   # blue diamond marker
_TIP_SISTER_MARKER_RIGHT = "#2b8a3e"   # green diamond marker

# Off-white backbone for dark mode: the light-grey ``_SKELETON_DIM``
# recedes on a white canvas, but needs to be lighter to stay legible
# against the dark plotly_dark canvas.
_SKELETON_DIM_DARK = "#cfcfcf"


def _is_dark():
    """True when the dark Plotly template is active (theme toggle)."""
    return get_template() == DARK_TEMPLATE


def _backbone_color():
    """Skeleton (backbone) branch colour for the current theme."""
    return _SKELETON_DIM_DARK if _is_dark() else _SKELETON_DIM


def _scatter_colorscale():
    """Clade-size colour scale. Viridis' dark low-end sinks into the dark
    canvas, so dark mode uses the higher-contrast Turbo."""
    return "Turbo" if _is_dark() else "Viridis"


def _highlight_overlay_trace(
    tips_by_name,
    highlight,
    *,
    color="#e63946",
    size=8,
    extra="",
    also_tips=None,
):
    """Build a coloured tip-marker overlay trace (optionally across both trees)."""
    xs, ys, names = [], [], []
    tip_maps = (tips_by_name, also_tips) if also_tips is not None else (tips_by_name,)
    for name in highlight:
        for tip_map in tip_maps:
            coord = tip_map.get(name)
            if coord is None:
                continue
            xs.append(coord[0])
            ys.append(coord[1])
            names.append(name)
    return dict(
        type="scatter",
        x=xs, y=ys,
        mode="markers",
        marker=dict(color=color, size=size),
        text=names,
        hovertemplate=f"%{{text}}<extra>{extra}</extra>",
        showlegend=False,
    )


def _connector_overlay_trace(
    tips1,
    tips2,
    highlight,
    *,
    color="rgba(230,57,70,0.7)",
    width=1.5,
    extra="",
):
    """Build a connector overlay between the two trees for tips present on both sides."""
    xs, ys, names = [], [], []
    for name in highlight:
        c1 = tips1.get(name)
        c2 = tips2.get(name)
        if c1 is None or c2 is None:
            continue
        xs += [c1[0], c2[0], None]
        ys += [c1[1], c2[1], None]
        names.append(name)
    return dict(
        type="scatter",
        x=xs, y=ys,
        mode="lines",
        line=dict(color=color, width=width),
        text=[n for n in names for _ in range(3)],
        hovertemplate=f"%{{text}}<extra>{extra}</extra>",
        showlegend=False,
    )


def _mrca_diamond_trace(xs, ys, texts, color):
    return dict(
        type="scatter",
        x=xs,
        y=ys,
        mode="markers",
        marker=dict(
            symbol="diamond",
            size=9,
            color=color,
            line=dict(color="white", width=1.5),
        ),
        text=texts,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )


def _sister_side_geometry(
    sib_entry,
    sister_names,
    *,
    origin_tree,
    landing_tree,
    connect: bool,
    show_origin: bool,
    label: str,
):
    """Build branch segments and MRCA diamond coords/labels for one sister clade."""
    bx, by, mx, my, mtext = [], [], [], [], []
    if not sib_entry or not sister_names or not (connect or show_origin):
        return bx, by, mx, my, mtext

    o_spans, o_xoff, o_xscale, o_xflip = origin_tree
    parent, siblings = sib_entry
    sx, sy = _sister_branch_segments(parent, siblings, o_xoff, o_xscale, o_xflip)
    bx += sx
    by += sy
    o_mrca = siblings[0] if len(siblings) == 1 else parent
    lo, hi = o_spans[id(o_mrca)]
    o_n = int(round(hi - lo)) + 1
    scaled = o_mrca.x * o_xscale
    mx.append((o_xoff - scaled) if o_xflip else (o_xoff + scaled))
    my.append(o_mrca.y)
    mtext.append(f"{label} MRCA — {o_n} tip{'s' if o_n != 1 else ''}")

    if not connect:
        return bx, by, mx, my, mtext

    l_root, l_spans, l_tips, l_sisters, l_xoff, l_xscale, l_xflip = landing_tree
    ys = [l_tips[n][1] for n in sister_names if n in l_tips]
    if ys:
        l_mrca = _find_mrca(l_root, l_spans, min(ys), max(ys))
        lo, hi = l_spans[id(l_mrca)]
        l_n = int(round(hi - lo)) + 1
        scaled = l_mrca.x * l_xscale
        mx.append((l_xoff - scaled) if l_xflip else (l_xoff + scaled))
        my.append(l_mrca.y)
        mtext.append(f"{label} MRCA — {l_n} tip{'s' if l_n != 1 else ''}")
        # Only colour landing branches when intact/monophyletic on the opposite tree
        # so a shattered sister does not paint half the opposite skeleton.
        if l_n == len(sister_names):
            l_parent_entry = l_sisters.get(id(l_mrca))
            if l_parent_entry:
                lsx, lsy = _sister_branch_segments(
                    l_parent_entry[0], [l_mrca], l_xoff, l_xscale, l_xflip
                )
            else:
                lsx, lsy = _subtree_branch_segments(l_mrca, l_xoff, l_xscale, l_xflip)
            bx += lsx
            by += lsy
    return bx, by, mx, my, mtext


def _sister_overlay_bundle(
    s1,
    s2,
    layout,
    *,
    connect: bool,
    show_origin_when_disconnected: bool,
):
    """Build all 8 sister traces: left/right branches, connectors, tips, and MRCA markers."""
    tips1, tips2 = layout["tips1"], layout["tips2"]
    right_xoff = layout["right_start"] + 1.0
    left_names = (
        [n.name for sib in s1[1] for n in _collect_nodes(sib) if n.is_tip]
        if s1
        else []
    )
    right_names = (
        [n.name for sib in s2[1] for n in _collect_nodes(sib) if n.is_tip]
        if s2
        else []
    )

    lbx, lby, lmx, lmy, lmt = _sister_side_geometry(
        s1,
        left_names,
        origin_tree=(layout["yspan1"], 0.0, layout["scale1"], False),
        landing_tree=(
            layout["root2"],
            layout["yspan2"],
            tips2,
            layout["sisters2"],
            right_xoff,
            layout["scale2"],
            True,
        ),
        connect=connect,
        show_origin=show_origin_when_disconnected,
        label="G1 sister",
    )
    rbx, rby, rmx, rmy, rmt = _sister_side_geometry(
        s2,
        right_names,
        origin_tree=(layout["yspan2"], right_xoff, layout["scale2"], True),
        landing_tree=(
            layout["root1"],
            layout["yspan1"],
            tips1,
            layout["sisters1"],
            0.0,
            layout["scale1"],
            False,
        ),
        connect=connect,
        show_origin=show_origin_when_disconnected,
        label="G2 sister",
    )

    show_left = left_names if (connect or show_origin_when_disconnected) else []
    show_right = right_names if (connect or show_origin_when_disconnected) else []
    branch_left = _branch_overlay_trace(lbx, lby, _MRCA_ACCENT)
    branch_right = _branch_overlay_trace(rbx, rby, _TIP_SISTER_ACCENT)
    conn_left = _connector_overlay_trace(
        tips1,
        tips2,
        left_names if connect else [],
        color=_TIP_SISTER_CONN_LEFT,
        width=1.0,
        extra="G1 sister",
    )
    conn_right = _connector_overlay_trace(
        tips1,
        tips2,
        right_names if connect else [],
        color=_TIP_SISTER_CONN_RIGHT,
        width=1.0,
        extra="G2 sister",
    )
    tips_left = _highlight_overlay_trace(
        tips1,
        show_left,
        color=_MRCA_ACCENT,
        size=7,
        extra="G1 sister",
        also_tips=tips2 if connect else None,
    )
    tips_right = _highlight_overlay_trace(
        tips2,
        show_right,
        color=_TIP_SISTER_ACCENT,
        size=7,
        extra="G2 sister",
        also_tips=tips1 if connect else None,
    )
    mrca_left = _mrca_diamond_trace(lmx, lmy, lmt, _MRCA_MARKER)
    mrca_right = _mrca_diamond_trace(rmx, rmy, rmt, _TIP_SISTER_MARKER_RIGHT)
    return (
        branch_left,
        branch_right,
        conn_left,
        conn_right,
        tips_left,
        tips_right,
        mrca_left,
        mrca_right,
    )


def _tip_sister_overlay_traces(taxon, layout, *, connect: bool):
    """Build all 8 sister overlay traces for a selected tip."""
    s1 = layout["sisters1"].get(taxon) if taxon else None
    s2 = layout["sisters2"].get(taxon) if taxon else None
    return _sister_overlay_bundle(
        s1, s2, layout, connect=connect, show_origin_when_disconnected=True
    )


def _clade_sister_overlay_traces(highlight, layout, *, connect: bool):
    """Build all 8 sister overlay traces for a selected clade on monophyletic sides only."""
    if not connect:
        return _sister_overlay_bundle(
            None, None, layout, connect=False, show_origin_when_disconnected=False
        )
    entries = []
    for root, spans, tips, sisters in (
        (layout["root1"], layout["yspan1"], layout["tips1"], layout["sisters1"]),
        (layout["root2"], layout["yspan2"], layout["tips2"], layout["sisters2"]),
    ):
        ys = [tips[name][1] for name in highlight if name in tips]
        if len(ys) != len(highlight) or not ys:
            entries.append(None)
            continue
        mrca = _find_mrca(root, spans, min(ys), max(ys))
        mrca_tips = {n.name for n in _collect_nodes(mrca) if n.is_tip}
        entries.append(sisters.get(id(mrca)) if mrca_tips == highlight else None)
    return _sister_overlay_bundle(
        entries[0], entries[1], layout, connect=True, show_origin_when_disconnected=False
    )

def _complement_tips_trace(complement, tips1, tips2):
    """Build the green-marker overlay trace for complementary tips on both trees.

    Looks up each complement tip in both ``tips1`` and ``tips2``
    independently so markers appear on both sides of the tanglegram,
    even in the concordant tree where the complement set is empty but
    the tips still exist as nodes. Returns a fully styled trace with
    empty coordinates when ``complement`` is empty, so callers can use
    it as the toggle-off placeholder too.
    """
    xs, ys, names = [], [], []
    for name in complement:
        for tree_tips in (tips1, tips2):
            coord = tree_tips.get(name)
            if coord is None:
                continue
            xs.append(coord[0])
            ys.append(coord[1])
            names.append(name)
    return dict(
        type="scatter",
        x=xs, y=ys,
        mode="markers",
        marker=dict(color="#2f9e44", size=8),
        text=names,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )


def _complement_connector_trace(tips1, tips2, complement1, complement2):
    """Build the green connector lines between complementary tips.

    Draws a line for every tip name that appears in either complement
    set and exists on both sides of the tanglegram.
    """
    all_complement = complement1 | complement2
    xs, ys, names = [], [], []
    for name in all_complement:
        c1 = tips1.get(name)
        c2 = tips2.get(name)
        if c1 is None or c2 is None:
            continue
        xs += [c1[0], c2[0], None]
        ys += [c1[1], c2[1], None]
        names.append(name)
    return dict(
        type="scatter",
        x=xs, y=ys,
        mode="lines",
        line=dict(color="rgba(47,158,68,0.7)", width=1.5),
        text=[n for n in names for _ in range(3)],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )

def _build_mrca_traces(highlight, layout):
    """Build the two MRCA-emphasis overlay traces from the cached
    layout and the highlighted tip set.

    Returns ``(subtree_trace, marker_trace, complement_per_tree)``
    where ``complement_per_tree`` is a dict with keys ``"tips1"`` and
    ``"tips2"``, each a set of tip names that are descendants of the
    MRCA in that tree but are NOT in ``highlight``. These are the
    "intruder" tips that make the clade non-monophyletic in the
    discordant tree — empty set when the tree contains the clade
    monophyletically.
    """
    sub_x, sub_y = [], []
    marker_x, marker_y, marker_text = [], [], []
    complement_per_tree = {"tips1": set(), "tips2": set()}

    for key, (root, spans, tips, x_offset, x_scale, x_flip) in zip(
        ("tips1", "tips2"),
        (
            (layout["root1"], layout["yspan1"], layout["tips1"],
             0.0, layout["scale1"], False),
            (layout["root2"], layout["yspan2"], layout["tips2"],
             layout["right_start"] + 1.0, layout["scale2"], True),
        ),
    ):
        ys = [tips[name][1] for name in highlight if name in tips]
        if not ys:
            continue
        mrca = _find_mrca(root, spans, min(ys), max(ys))
        seg_x, seg_y = _subtree_branch_segments(mrca, x_offset, x_scale, x_flip)
        sub_x += seg_x
        sub_y += seg_y
        lo, hi = spans[id(mrca)]
        n_tips = int(round(hi - lo)) + 1
        scaled = mrca.x * x_scale
        marker_x.append((x_offset - scaled) if x_flip else (x_offset + scaled))
        marker_y.append(mrca.y)
        marker_text.append(f"MRCA — {n_tips} tips")

        # Collect complement: all tips under this MRCA minus highlight
        mrca_tips = {n.name for n in _collect_nodes(mrca) if n.is_tip}
        complement_per_tree[key] = mrca_tips - highlight

    subtree_trace = dict(
        type="scatter",
        x=sub_x, y=sub_y,
        mode="lines",
        line=dict(color=_MRCA_ACCENT, width=2),
        hoverinfo="skip",
        showlegend=False,
    )
    marker_trace = dict(
        type="scatter",
        x=marker_x, y=marker_y,
        mode="markers",
        marker=dict(symbol="diamond", size=10, color=_MRCA_MARKER,
                    line=dict(color="white", width=1.5)),
        text=marker_text,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )
    return subtree_trace, marker_trace, complement_per_tree


def _tanglegram_title_children(label1, label2, highlight,
                               in_1=False, in_2=False):
    """Build the tanglegram's title as Dash children for the sticky
    ``clade-freq-tanglegram-title`` div above the graph.

    The title lives outside the figure so it doesn't scroll off the
    top of the viewport when the user expands the tree (the figure
    can grow to tens of thousands of pixels tall). Returns a list of
    ``html.Span`` elements — the consensus tree label whose tree contains the
    clade is drawn green, the other in default colour, and the
    middle segment shows the clade size.
    """
    green = "#2f9e44"
    def fmt(label, contains):
        style = {"fontWeight": "bold"}
        if contains:
            style["color"] = green
        return html.Span(label, style=style)
    return [
        fmt(label1, in_1),
        html.Span("  ←   "),
        html.Span(f"clade: {len(highlight)} tips"),
        html.Span("   →  "),
        fmt(label2, in_2),
    ]


def _g1_g2_sister_spans(sister_1: str, sister_2: str) -> list:
    """Color-coded compact G1 (blue) -> G2 (green) sister summary matching the tanglegram."""
    return [
        html.Span("● G1: ", style={"color": _MRCA_ACCENT, "fontWeight": 600}),
        html.Span(sister_1),
        html.Span("  →  ", style={"fontWeight": 600, "color": "#e63946"}),
        html.Span("● G2: ", style={"color": _TIP_SISTER_ACCENT, "fontWeight": 600}),
        html.Span(sister_2),
    ]


def _tanglegram_tip_title_children(taxon, tip_row):
    """Sticky tanglegram title for a selected tip showing Shift and color-coded G1 -> G2 sisters."""
    parts = [
        html.Span("Tip ", style={"fontWeight": 500}),
        dmc.Badge(
            taxon,
            color="red",
            variant="light",
            size="md",
            style={
                "fontFamily": "monospace",
                "textTransform": "none",
                "verticalAlign": "middle",
            },
        ),
    ]
    if tip_row is not None:
        parts.append(
            html.Span(
                [
                    html.Span(f"   Shift: {float(tip_row['between_shift']):.2f} · "),
                    *_g1_g2_sister_spans(
                        str(tip_row.get("top_sister_1", "—")),
                        str(tip_row.get("top_sister_2", "—")),
                    ),
                ],
                style={"fontSize": "14px", "fontWeight": 400},
            )
        )
    return parts


# Slider-default-aware tanglegram height. The slider value 1..5 is
# "pixels per tip" — fine for small trees, but at 1000+ taxa even
# px_per_tip=1 produces a ~1500 px figure that overflows the viewport
# on first render. Cap at ``_TANGLEGRAM_DEFAULT_MAX_HEIGHT`` while the
# user is at the default; the moment they step up the slider the cap
# lifts and the figure scales linearly with ``px_per_tip * n_tips``.
_TANGLEGRAM_DEFAULT_MAX_HEIGHT = 700  # px, fits typical browser viewports


def _tanglegram_height(px_per_tip, max_y):
    px_per_tip = px_per_tip or 1
    natural = int(max_y * px_per_tip) + 60
    if px_per_tip == 1:
        natural = min(natural, _TANGLEGRAM_DEFAULT_MAX_HEIGHT)
    return max(300, natural)


def _tanglegram_placeholder_fig():
    """The empty-state figure that lives in the tanglegram panel
    before any Compare+click has happened — also restored when the
    user clears all data."""
    return {
        "data": [],
        "layout": {
            "height": 200,
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
            "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
            "annotations": [{
                "text": "Select two consensus trees and click "
                        "<b>Compare Clade Frequencies</b>,"
                        " then click a dot in the scatter "
                        "above to draw the tanglegram.",
                "xref": "paper", "yref": "paper",
                "x": 0.5, "y": 0.5,
                "showarrow": False,
                "font": {"size": 13, "color": "#888"},
                "align": "center",
            }],
        },
    }


def clear_clade_freq_caches():
    """Drop every per-session cache used by the Clade Frequency
    Comparison feature.

    Called from ``sidebar.clear_uploads`` so a Clear-data click
    actually wipes the parsed-NEXUS and layout LRUs. Managed comparison
    payloads and their click-resolution maps live in ``state`` and are cleared
    by ``state.clear_all_analysis_results`` in the same reset transaction.
    """
    _get_parsed_consensus_tree.cache_clear()
    _get_tanglegram_layout.cache_clear()


def _format_shift_sigfig(value: float) -> str:
    """Format shift to 4 significant figures, snapping floating-point dust to 0."""
    v = float(value)
    return "0" if abs(v) < 1e-12 else f"{v:.4g}"


def _format_instability_hover(
    clade_instab,
    between_shift,
    instab_1=None,
    instab_2=None,
) -> str:
    if clade_instab is None:
        return ""
    if instab_1 is not None and instab_2 is not None:
        text = (
            f"<br>Instab: {float(clade_instab):.2f} "
            f"(G1 {float(instab_1):.2f}, G2 {float(instab_2):.2f})"
        )
    else:
        text = f"<br>Instab: {float(clade_instab):.2f}"
    if between_shift is not None:
        text += f"<br>Shift: {_format_shift_sigfig(between_shift)}"
    return text


def _scatter_customdata(df_plot: pd.DataFrame) -> list[list[Any]]:
    if df_plot.empty:
        return []
    hover = (
        df_plot["instability_hover"]
        if "instability_hover" in df_plot.columns
        else [""] * len(df_plot)
    )
    return [
        [int(sid), int(size), str(mem), str(inst)]
        for sid, size, mem, inst in zip(
            df_plot["split_id"],
            df_plot["clade_size"],
            df_plot["consensus_tree_membership"],
            hover,
        )
    ]


def _build_scatter_fig(df_plot, label1, label2):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1,
        line=dict(color="grey", width=1, dash="dash"),
        layer="below",
    )
    # ``Scattergl`` (WebGL) instead of ``Scatter`` (SVG). At ~40k
    # bipartitions per Compare click the SVG path creates one DOM node
    # per marker and freezes the browser; WebGL renders the same
    # point set in a single canvas frame. Trade-off: ``marker.line``
    # is not supported in WebGL — the white outline around each dot
    # is dropped, but the colour-coded fill is enough to distinguish
    # clade sizes on a dense scatter anyway.
    fig.add_trace(go.Scattergl(
        x=df_plot["freq_1"],
        y=df_plot["freq_2"],
        mode="markers",
        # Only one data trace + the colourbar already labels the axis,
        # so suppress the otherwise meaningless "trace 0" legend chip.
        showlegend=False,
        marker=dict(
            size=8,
            color=df_plot["clade_size"],
            colorscale=_scatter_colorscale(),
            showscale=True,
            colorbar=dict(title=dict(text="Clade size"), thickness=12),
            opacity=0.75,
        ),
        customdata=_scatter_customdata(df_plot),
        hovertemplate=(
            "<b>Clade (%{customdata[1]} tips)</b><br>"
            "Group 1: %{x:.3f}<br>"
            "Group 2: %{y:.3f}"
            "%{customdata[3]}<br>"
            "In: %{customdata[2]}"
            "<extra></extra>"
        ),
    ))
    # Trailing click-marker overlay (trace index 1). Empty until the
    # user clicks a point; ``update_click_marker`` Patches its x/y to
    # surround the clicked dot with a hollow red circle. We use
    # ``go.Scatter`` (SVG) for this — only ever one marker, so the
    # SVG cost is negligible, and SVG supports ``marker.line`` for the
    # ring outline (WebGL doesn't).
    fig.add_trace(go.Scatter(
        x=[], y=[],
        mode="markers",
        marker=dict(
            size=16,
            color="rgba(0,0,0,0)",
            line=dict(color="#e63946", width=2.5),
            symbol="circle",
        ),
        hoverinfo="skip",
        showlegend=False,
        name="selected",
    ))
    fig.update_layout(
        template=get_template(),
        xaxis=dict(title=f"Frequency — {label1}", range=[-0.02, 1.02]),
        yaxis=dict(title=f"Frequency — {label2}", range=[-0.02, 1.02]),
        margin=dict(l=60, r=20, t=30, b=50),
        height=450,
        # Pure ``event`` mode — Plotly fires clickData on every click
        # cleanly. We draw the click-marker ourselves via Patch (see
        # ``update_click_marker`` below) rather than relying on the
        # ``+select`` auto-grey, which on Scattergl is intermittent.
        # The ``store_scatter_click`` callback also stamps a nonce so
        # identical click payloads still propagate through dcc.Store.
        # dcc.Store.
        clickmode="event",
        uirevision="clade-freq-scatter",
    )
    return fig


def _filter_clade_frequency_frame(
    frame: pd.DataFrame,
    *,
    min_clade_size: int,
    exclusive_only: bool,
    color_by: str = "clade_size",
    min_instability: float = 0.0,
) -> pd.DataFrame:
    """Apply the user-facing scatter filters to comparison records."""
    if frame.empty:
        return frame
    if color_by == "clade_instability" and "clade_instability" in frame.columns:
        instab = pd.to_numeric(frame["clade_instability"], errors="coerce")
        visible = (
            (frame["clade_size"] >= 2)
            & instab.notna()
            & (instab >= float(min_instability))
        )
    else:
        visible = frame["clade_size"] >= min_clade_size
    if exclusive_only:
        visible &= frame["in_consensus_tree_1"].ne(
            frame["in_consensus_tree_2"]
        )
    return frame.loc[visible]


_TIP_FOOTER_DEFAULT_HINT = (
    "Click ⊙ on any taxon to highlight its jump in the tanglegram and filter "
    "the scatter to clades containing it."
)


def _build_scatter_patch(
    store_data,
    *,
    color_by,
    min_clade_size,
    min_instability,
    exclusive_only,
    tip_taxon: str | None = None,
    resolution: dict[int, Any] | None = None,
    click_data: dict[str, Any] | None = None,
) -> tuple[Patch, int]:
    """Build a Patch updating scatter trace 0 (points/color) and trace 1 (ring)."""
    df = pd.DataFrame(store_data)
    has_instab_col = (
        "clade_instability" in df.columns
        and pd.to_numeric(df["clade_instability"], errors="coerce").notna().any()
    )
    mode = (
        "clade_instability"
        if color_by == "clade_instability" and has_instab_col
        else "clade_size"
    )
    df_plot = _filter_clade_frequency_frame(
        df,
        min_clade_size=int(min_clade_size or 2),
        exclusive_only=bool(exclusive_only),
        color_by=mode,
        min_instability=float(min_instability or 0.0),
    )
    if tip_taxon is not None and resolution is not None and not df_plot.empty:
        keep = [
            tip_taxon in (resolution.get(int(sid)) or {}).get("tip_names", ())
            for sid in df_plot["split_id"]
        ]
        df_plot = df_plot.loc[keep]

    patch = Patch()
    patch["data"][0]["x"] = df_plot["freq_1"].tolist()
    patch["data"][0]["y"] = df_plot["freq_2"].tolist()
    patch["data"][0]["customdata"] = _scatter_customdata(df_plot)
    if mode == "clade_instability":
        patch["data"][0]["marker"]["color"] = (
            df_plot["clade_instability"].astype(float).tolist()
        )
        patch["data"][0]["marker"]["colorbar"]["title"]["text"] = "Instability"
        patch["data"][0]["marker"]["cauto"] = False
        patch["data"][0]["marker"]["cmin"] = 0.0
        patch["data"][0]["marker"]["cmax"] = 1.0
    else:
        patch["data"][0]["marker"]["color"] = df_plot["clade_size"].tolist()
        patch["data"][0]["marker"]["colorbar"]["title"]["text"] = "Clade size"
        patch["data"][0]["marker"]["cauto"] = True
        patch["data"][0]["marker"]["cmin"] = None
        patch["data"][0]["marker"]["cmax"] = None
    ring_x: list[float] = []
    ring_y: list[float] = []
    if (
        isinstance(click_data, dict)
        and click_data.get("kind", "clade") == "clade"
        and click_data.get("split_id") is not None
        and not df_plot.empty
        and int(click_data["split_id"]) in set(df_plot["split_id"].astype(int))
    ):
        x, y = click_data.get("x"), click_data.get("y")
        if x is not None and y is not None:
            ring_x, ring_y = [x], [y]
    patch["data"][1]["x"] = ring_x
    patch["data"][1]["y"] = ring_y
    return patch, len(df_plot)


def _shift_pill_class(value: float) -> str:
    if value >= 0.40:
        return "tt-shift-pill tt-shift-red"
    if value >= 0.10:
        return "tt-shift-pill tt-shift-amber"
    return "tt-shift-pill tt-shift-neutral"


def _sort_tip_rows(tip_rows: list[dict[str, Any]], sort_by: str | None) -> list[dict[str, Any]]:
    key = sort_by or "between_shift"
    if key == "taxon":
        return sorted(tip_rows, key=lambda r: str(r["taxon"]).lower())
    if key == "combined_instability":
        return sorted(
            tip_rows,
            key=lambda r: (-float(r["combined_instability"]), -float(r["between_shift"]), str(r["taxon"])),
        )
    if key == "instab_1":
        return sorted(
            tip_rows,
            key=lambda r: (-float(r["instab_1"]), -float(r["between_shift"]), str(r["taxon"])),
        )
    if key == "instab_2":
        return sorted(
            tip_rows,
            key=lambda r: (-float(r["instab_2"]), -float(r["between_shift"]), str(r["taxon"])),
        )
    return sorted(
        tip_rows,
        key=lambda r: (-float(r["between_shift"]), -float(r["combined_instability"]), str(r["taxon"])),
    )


def _tip_detail_footer(selected_row: dict[str, Any] | None):
    if selected_row is None:
        return dmc.Text(_TIP_FOOTER_DEFAULT_HINT, size="xs", c="dimmed")
    p1_pct = round(float(selected_row.get("p_cherry_1", 0.0)) * 100)
    p2_pct = round(float(selected_row.get("p_cherry_2", 0.0)) * 100)
    n_sis = int(selected_row.get("n_sisters_comb", 0))
    return html.Div(
        [
            dmc.Group(
                [
                    html.Span(
                        [
                            html.Strong("Selected: "),
                            html.Span(
                                selected_row["taxon"],
                                style={"fontFamily": "monospace", "fontWeight": 600, "color": "#e63946"},
                            ),
                        ]
                    ),
                    dmc.Text(
                        f"{n_sis} distinct sisters · Cherry: {p1_pct}% (G1) / {p2_pct}% (G2)",
                        size="xs",
                        c="dimmed",
                    ),
                ],
                justify="space-between",
                gap="xs",
                wrap="wrap",
            ),
            html.Div(
                _g1_g2_sister_spans(
                    str(selected_row.get("top_sister_1", "—")),
                    str(selected_row.get("top_sister_2", "—")),
                ),
                style={"marginTop": "2px", "overflowWrap": "anywhere"},
            ),
        ]
    )


@dataclass(frozen=True, slots=True)
class _CladeComparisonFinalizationContext:
    source_distmat: str
    uid_1: str
    uid_2: str
    label_1: str
    label_2: str
    consensus_columns_1: frozenset[int]
    consensus_columns_2: frozenset[int]
    min_clade_size: int
    exclusive_only: bool


def _cached_counts_for_columns(entry, columns):
    """Return a compact count slice or ``None`` for the worker fallback."""
    counts = entry.get("counts")
    n_trees = int(entry.get("n_trees") or 0)
    if counts is None or n_trees <= 0:
        return None, 0
    values = np.asarray(counts)
    if values.ndim != 1:
        raise ValueError("cached clade counts must be one-dimensional")
    if columns and columns[-1] >= len(values):
        raise ValueError("consensus-tree clade column is outside cached counts")
    return values[columns].astype(np.int64, copy=False).tolist(), n_trees


def _finalize_clade_comparison_job(
    ref: JobRef,
    result: Any,
    *,
    context: _CladeComparisonFinalizationContext,
) -> dict[str, Any]:
    """Publish compact scatter data and its server-side click resolution."""
    if not isinstance(result, dict):
        raise TypeError("clade-frequency worker returned a non-mapping result")
    rows = result.get("rows")
    leaf_names = result.get("leaf_names")
    if not isinstance(rows, list) or not isinstance(leaf_names, list):
        raise TypeError("clade-frequency worker returned an invalid payload")

    records = []
    resolution = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("clade-frequency row must be a mapping")
        column_j = int(row["column_j"])
        clade_size = int(row["clade_size"])
        if clade_size < 2:
            # Singletons live exclusively in the Tip Instability Table.
            continue
        in_1 = column_j in context.consensus_columns_1
        in_2 = column_j in context.consensus_columns_2
        if not (in_1 or in_2):
            continue
        if in_1 and in_2:
            membership = "both consensus trees"
        elif in_1:
            membership = f"{context.label_1} only"
        else:
            membership = f"{context.label_2} only"

        split_key = tuple(int(index) for index in row["split_key"])
        try:
            tip_names = tuple(str(leaf_names[index]) for index in split_key)
        except IndexError as exc:
            raise ValueError(
                "clade-frequency split references an unknown leaf index"
            ) from exc

        clade_instab = row.get("clade_instability")
        instab_1 = row.get("instab_1")
        instab_2 = row.get("instab_2")
        between_shift = row.get("between_shift")

        split_id = len(records)
        records.append(
            {
                "split_id": split_id,
                "freq_1": float(row["freq_1"]),
                "freq_2": float(row["freq_2"]),
                "clade_size": clade_size,
                "in_consensus_tree_1": in_1,
                "in_consensus_tree_2": in_2,
                "consensus_tree_membership": membership,
                "clade_instability": (
                    float(clade_instab) if clade_instab is not None else None
                ),
                "instab_1": float(instab_1) if instab_1 is not None else None,
                "instab_2": float(instab_2) if instab_2 is not None else None,
                "between_shift": (
                    float(between_shift) if between_shift is not None else None
                ),
                "instability_hover": _format_instability_hover(
                    clade_instab,
                    between_shift,
                    instab_1=instab_1,
                    instab_2=instab_2,
                ),
            }
        )
        resolution[split_id] = {
            "source_distmat": context.source_distmat,
            "column_j": column_j,
            "tip_names": tip_names,
        }

    if not records:
        raise ValueError(
            "None of the selected consensus-tree clades are available to plot"
        )
    frame = pd.DataFrame(records)
    plotted = _filter_clade_frequency_frame(
        frame,
        min_clade_size=context.min_clade_size,
        exclusive_only=context.exclusive_only,
    )
    figure = _build_scatter_fig(plotted, context.label_1, context.label_2)
    tip_rows = list(result.get("tip_rows") or [])
    has_instability = bool(result.get("has_instability") and tip_rows)
    state.store_clade_frequency_result(
        ref.job_id,
        {
            "records": records,
            "resolution": resolution,
            "figure": figure.to_dict(),
            "pair": (context.uid_1, context.uid_2),
            "label_1": context.label_1,
            "label_2": context.label_2,
            "tip_rows": tip_rows,
            "has_instability": has_instability,
        },
    )
    elapsed = float(result.get("elapsed", 0.0))
    snapshot_input_text = {
        "sparse": "generic sparse CSR snapshot",
        "rooted_facts": "rooted facts",
    }.get(
        result.get("snapshot_input_mode"),
        "worker-reported snapshot representation",
    )
    add_log(
        f"Compared {len(records)} consensus-tree clades for "
        f"{context.source_distmat} in {elapsed:.3f}s "
        f"using {snapshot_input_text}."
    )
    return {
        "result_key": ref.job_id,
        "source_distmat": context.source_distmat,
        "label_1": context.label_1,
        "label_2": context.label_2,
        "n_clades": len(records),
        "elapsed": elapsed,
    }


def _resolved_clade(click_data, expected_pair=None):
    """Resolve a browser split ID through its immutable managed result key."""
    if not isinstance(click_data, dict):
        return None
    result_key = click_data.get("result_key")
    split_id = click_data.get("split_id")
    if result_key is None or split_id is None:
        return None
    cached = state.get_clade_frequency_result(result_key)
    if cached is None:
        return None
    if expected_pair is not None and tuple(expected_pair) != tuple(
        cached.get("pair", ())
    ):
        return None
    return cached.get("resolution", {}).get(int(split_id))


def register_clade_explore_callbacks():
    # Monotonic counter that gets stamped on every scatter-plot click
    # payload (see ``store_scatter_click`` below). Without a unique
    # value, ``dcc.Store`` deduplicates identical click data and the
    # downstream tanglegram callback doesn't fire — producing the
    # "first click does nothing, second click works" behaviour.
    _click_counter = 0

    # ─── Clade Exploration RF Matrix selector ──────────────────────────
    # The tab's own matrix dropdown — clade-frequency comparison only
    # makes sense between consensus trees computed from the same RF matrix, so the
    # consensus tree table and the Compare dropdowns below all condition on it.
    # Mirrors the Diagnostics tab's selector; both are fed by the
    # shared ``distmat-store``.
    @callback(
        Output("clade-distmat-select", "data"),
        Output("clade-distmat-select", "value"),
        Output("clade-distmat-info", "children"),
        Input("distmat-store", "data"),
        State("clade-distmat-select", "value"),
    )
    def populate_clade_distmat_select(distmat_data, current_value):
        if not distmat_data:
            return [], None, ""
        options = [
            {"value": k,
             "label": (
                 f"{k} ({v.get('n_trees', '?')} trees, "
                 f"{'rooted' if v.get('is_rooted', True) else 'unrooted'})"
             )}
            for k, v in distmat_data.items()
        ]
        # Keep the user's pick if it's still around; otherwise default to
        # the most recently registered matrix.
        new_value = (
            current_value
            if current_value and current_value in distmat_data
            else list(distmat_data.keys())[-1]
        )
        meta = distmat_data.get(new_value, {})
        groups_per_file = meta.get("groups_per_file", {})
        n_runs = len({g for groups in groups_per_file.values() for g in groups})
        info = dmc.Group([
            dmc.Badge(f"{meta.get('n_trees', '?')} trees",
                      variant="light", color="grape", size="sm"),
            dmc.Badge(f"{n_runs} runs",
                      variant="light", color="teal", size="sm"),
        ], gap="xs")
        return options, new_value, info

    # ─── consensus trees registered for the selected matrix ──────────────────
    # Lists the consensus trees registered against the currently-selected RF
    # matrix, split into "Between-runs" and "Within-run" sub-tables.
    # Hidden when no consensus trees match the active matrix.
    @callback(
        Output("clade-consensus-tree-title", "children"),
        Output("clade-consensus-tree-list", "children"),
        Output("clade-consensus-tree-paper", "style"),
        Input("consensus-tree-registry-store", "data"),
        Input("clade-distmat-select", "value"),
    )
    def render_clade_consensus_tree_panel(registry, selected_matrix):
        from .consensus_tree_list import _table_for
        if not registry or not selected_matrix:
            return html.Div(), html.Div(), {"display": "none"}
        matched = [e for e in registry
                   if e.get("source_distmat") == selected_matrix]
        if not matched:
            return html.Div(), html.Div(), {"display": "none"}
        title = dmc.Title(f"Consensus trees for {selected_matrix}", order=5)
        table = _table_for(matched, show_mode=True, source="clade")
        return title, table, {}

    # ------ Empty-state notice: visible only while the consensus tree registry is
    # wholly empty, hidden as soon as any consensus tree exists. Registry-only (not
    # matrix-scoped), unlike ``render_clade_consensus_tree_panel`` above.

    @callback(
        Output("clade-consensus-tree-empty-paper", "style"),
        Input("consensus-tree-registry-store", "data"),
    )
    def toggle_clade_consensus_tree_empty_state(registry):
        # Visible branch keeps textAlign so the rewrite doesn't drop the
        # centering baked into the Paper's style prop.
        return {"display": "none"} if registry else {"textAlign": "center"}

    # ------ Hide the Clade Frequency panel when the consensus tree registry is
    # empty (covers the "user deleted every consensus tree" case — the other two
    # writers, Compare-button success and Clear-Data, handle the show
    # and full-reset paths respectively). ``no_update`` when there are
    # still consensus trees so we never fight the Compare path that just opened
    # the panel.

    @callback(
        Output("clade-freq-output-paper", "style", allow_duplicate=True),
        Input("consensus-tree-registry-store", "data"),
        prevent_initial_call=True,
    )
    def hide_clade_freq_when_no_consensus_trees(registry):
        if registry:
            return no_update
        return {"display": "none"}

    # ------ Clade Frequency Comparison: populate dropdowns ------

    @callback(
        Output("clade-freq-consensus-tree-select-1", "data"),
        Output("clade-freq-consensus-tree-select-1", "disabled"),
        Output("clade-freq-consensus-tree-select-1", "value", allow_duplicate=True),
        Output("clade-freq-consensus-tree-select-2", "data"),
        Output("clade-freq-consensus-tree-select-2", "disabled"),
        Output("clade-freq-consensus-tree-select-2", "value", allow_duplicate=True),
        Input("consensus-tree-registry-store", "data"),
        Input("clade-distmat-select", "value"),
        State("clade-freq-consensus-tree-select-1", "value"),
        State("clade-freq-consensus-tree-select-2", "value"),
        prevent_initial_call=True,
    )
    def populate_consensus_tree_selects(registry, selected_matrix, sel1, sel2):
        """Rebuild the consensus-tree dropdown options whenever a new consensus tree is
        saved OR the active distmat changes.

        **Filtered by the active distmat**: clade-frequency comparison
        only makes sense between consensus trees computed from the SAME RF matrix
        (= same trees, same column basis in the rooted-clade presence
        table). Showing consensus trees from other matrices in the dropdowns would
        let the user pick a meaningless cross-matrix pair, so we
        restrict each dropdown's options to ``e["source_distmat"] ==
        selected_matrix``.

        When the user switches the active distmat, any selection that
        no longer belongs to the new matrix is cleared so the Compare
        button doesn't fire on a phantom pair.
        """
        if not registry or not selected_matrix:
            return [], True, None, [], True, None

        matched = [
            e for e in registry
            if e.get("source_distmat") == selected_matrix
        ]
        if not matched:
            return [], True, None, [], True, None

        options = [
            {
                "value": e["uuid"],
                "label": f"{e['name']}  ({e['n_trees']} trees)",
            }
            for e in matched
        ]
        valid_uids = {e["uuid"] for e in matched}
        new_sel1 = sel1 if sel1 in valid_uids else None
        new_sel2 = sel2 if sel2 in valid_uids else None
        return options, False, new_sel1, options, False, new_sel2

    # ------ Clade Frequency Comparison: toggle sliders & filter scatter ------

    @callback(
        Output("clade-freq-size-slider-wrap", "style"),
        Output("clade-freq-instability-slider-wrap", "style"),
        Input("clade-freq-color-by", "value"),
    )
    def toggle_clade_freq_slider_wrappers(color_by):
        if color_by == "clade_instability":
            return {"display": "none"}, {}
        return {}, {"display": "none"}

    @callback(
        Output("clade-freq-scatter", "figure", allow_duplicate=True),
        Output("clade-freq-tip-filter-banner", "style"),
        Output("clade-freq-tip-filter-banner-text", "children"),
        Input("clade-freq-color-by", "value"),
        Input("clade-freq-min-clade-size", "value"),
        Input("clade-freq-min-instability", "value"),
        Input("clade-freq-exclusive-only", "checked"),
        Input("tip-instability-filter-store", "data"),
        State("clade-freq-data-store", "data"),
        State("clade-freq-click-store", "data"),
        prevent_initial_call=True,
    )
    def filter_clade_freq_plot(
        color_by,
        min_clade_size,
        min_instability,
        exclusive_only,
        tip_filter,
        store_data,
        click_data,
    ):
        """Patch scatter ``data[0]`` and tip-filter banner, stacking tip + control filters."""
        hide_banner = {"display": "none"}
        if not store_data:
            return no_update, hide_banner, ""

        tip_taxon = None
        resolution = None
        if isinstance(tip_filter, dict) and tip_filter.get("taxon"):
            cached = state.get_clade_frequency_result(tip_filter.get("result_key"))
            if cached is not None:
                tip_taxon = str(tip_filter["taxon"])
                resolution = cached.get("resolution", {})

        patch, n_shown = _build_scatter_patch(
            store_data,
            color_by=color_by,
            min_clade_size=min_clade_size,
            min_instability=min_instability,
            exclusive_only=exclusive_only,
            tip_taxon=tip_taxon,
            resolution=resolution,
            click_data=click_data,
        )
        if tip_taxon is None:
            return patch, hide_banner, ""
        if n_shown:
            banner_text = (
                f"Showing {n_shown} clade"
                f"{'s' if n_shown != 1 else ''} containing {tip_taxon}"
            )
        else:
            banner_text = f"No clades contain {tip_taxon} under current filter"
        return patch, {}, banner_text

    # ------ Tip Instability Table: reactive sync, render, pagination, TSV ------

    @callback(
        Output("clade-freq-color-by", "data"),
        Output("clade-freq-color-by", "value"),
        Output("clade-freq-color-by-tooltip", "disabled"),
        Output("tip-instability-sort", "value"),
        Output("tip-instability-search", "value"),
        Output("tip-instability-limit-store", "data", allow_duplicate=True),
        Output("tip-instability-filter-store", "data", allow_duplicate=True),
        Output("tip-instability-export-btn", "disabled"),
        Output("tip-instability-controls", "style"),
        Input("clade-freq-result-key-store", "data"),
        State("clade-freq-color-by", "value"),
        prevent_initial_call=True,
    )
    def sync_instability_controls_on_result(result_key, current_color_by):
        """Sync dropdown availability, default sort, and resets on Compare/Clear."""
        cached = state.get_clade_frequency_result(result_key) if result_key else None
        has_instab = bool(cached and cached.get("has_instability"))
        color_options = [
            {"value": "clade_size", "label": "Clade size"},
            {
                "value": "clade_instability",
                "label": "Clade instability",
                "disabled": not has_instab,
            },
        ]
        # Always start/reset to Clade size so freshly built scatter matches dropdown.
        color_value = no_update if current_color_by == "clade_size" else "clade_size"
        tooltip_disabled = has_instab or cached is None

        default_sort = "between_shift"
        if has_instab:
            tip_rows = cached.get("tip_rows") or []
            max_shift = max((float(r["between_shift"]) for r in tip_rows), default=0.0)
            if max_shift <= 1e-12:
                default_sort = "combined_instability"

        controls_style = {} if has_instab else {"display": "none"}
        return (
            color_options,
            color_value,
            tooltip_disabled,
            default_sort,
            "",
            50,
            None,
            not has_instab,
            controls_style,
        )

    @callback(
        Output("tip-instability-limit-store", "data", allow_duplicate=True),
        Input("tip-instability-show-more", "n_clicks"),
        State("tip-instability-limit-store", "data"),
        prevent_initial_call=True,
    )
    def bump_tip_instability_limit(n_clicks, current_limit):
        if not n_clicks:
            return no_update
        return int(current_limit or 50) + 50

    @callback(
        Output("tip-instability-table-body", "children"),
        Output("tip-instability-badge", "children"),
        Output("tip-instability-badge", "style"),
        Output("tip-instability-summary", "children"),
        Output("tip-instability-show-more", "style"),
        Output("tip-instability-detail-footer", "children"),
        Input("clade-freq-result-key-store", "data"),
        Input("tip-instability-search", "value"),
        Input("tip-instability-sort", "value"),
        Input("tip-instability-limit-store", "data"),
        Input("tip-instability-filter-store", "data"),
        prevent_initial_call=True,
    )
    def render_tip_instability_table(
        result_key,
        search_text,
        sort_by,
        limit,
        tip_filter,
    ):
        cached = state.get_clade_frequency_result(result_key) if result_key else None
        if cached is None:
            return (
                "",
                "",
                {"display": "none"},
                "",
                {"display": "none"},
                _tip_detail_footer(None),
            )

        if not cached.get("has_instability"):
            empty = html.Div(
                [
                    icon("tabler:info-circle", size=24, style={"opacity": 0.6}),
                    dmc.Text("Tip Instability Unavailable", fw=600, size="sm"),
                    dmc.Text(
                        "Immediate sister-clade Jaccard instability requires strictly "
                        "binary rooted trees with persisted RootedFactsSnapshot splits. "
                        "Unrooted bipartition scatter and tanglegram views remain active.",
                        size="xs",
                        c="dimmed",
                    ),
                ],
                className="tt-tip-instability-empty",
            )
            return empty, "", {"display": "none"}, "", {"display": "none"}, ""

        tip_rows = cached.get("tip_rows") or []
        n_shifting = sum(1 for r in tip_rows if float(r["between_shift"]) > 0.05)
        n_unstable = sum(1 for r in tip_rows if float(r["combined_instability"]) > 0.05)
        if n_shifting:
            badge_text = f"{n_shifting} shifting"
        elif n_unstable:
            badge_text = f"{n_unstable} unstable"
        else:
            badge_text = "stable"

        query = (search_text or "").strip().lower()
        filtered = (
            [r for r in tip_rows if query in str(r["taxon"]).lower()]
            if query
            else tip_rows
        )
        ordered = _sort_tip_rows(filtered, sort_by)
        cap = max(50, int(limit or 50))
        shown = ordered[:cap]

        selected_taxon = (
            tip_filter.get("taxon")
            if isinstance(tip_filter, dict) and tip_filter.get("result_key") == result_key
            else None
        )
        selected_row = (
            next((r for r in tip_rows if r["taxon"] == selected_taxon), None)
            if selected_taxon
            else None
        )

        body_rows = []
        for row in shown:
            taxon = str(row["taxon"])
            is_selected = taxon == selected_taxon
            shift_val = float(row["between_shift"])
            body_rows.append(
                dmc.TableTr(
                    [
                        dmc.TableTd(
                            html.Span(taxon, className="tt-tip-taxon-text", title=taxon)
                        ),
                        dmc.TableTd(
                            html.Span(
                                f"{shift_val:.2f}",
                                className=_shift_pill_class(shift_val),
                            ),
                            className="tt-tip-col-shift",
                        ),
                        dmc.TableTd(
                            f"{float(row['instab_1']):.2f}",
                            className="tt-tip-col-num",
                        ),
                        dmc.TableTd(
                            f"{float(row['instab_2']):.2f}",
                            className="tt-tip-col-num",
                        ),
                        dmc.TableTd(
                            f"{float(row['combined_instability']):.2f}",
                            className="tt-tip-col-num",
                        ),
                        dmc.TableTd(
                            dmc.ActionIcon(
                                "⊙",
                                id={"type": "tip-instability-focus", "taxon": taxon},
                                size="xs",
                                variant="filled" if is_selected else "subtle",
                                color="red" if is_selected else "gray",
                                **{"aria-label": f"Focus {taxon}"},
                            ),
                            className="tt-tip-col-action",
                        ),
                    ],
                    className="tt-tip-row-selected" if is_selected else "",
                )
            )

        if body_rows:
            table = dmc.Table(
                [
                    dmc.TableThead(
                        dmc.TableTr(
                            [
                                dmc.TableTh("Taxon"),
                                dmc.TableTh("Shift", className="tt-tip-col-shift"),
                                dmc.TableTh("G1", className="tt-tip-col-num"),
                                dmc.TableTh("G2", className="tt-tip-col-num"),
                                dmc.TableTh("Comb.", className="tt-tip-col-num"),
                                dmc.TableTh("", className="tt-tip-col-action"),
                            ]
                        )
                    ),
                    dmc.TableTbody(body_rows),
                ],
                className="tt-tip-instability-table",
                highlightOnHover=True,
            )
            summary = (
                f"Showing {len(shown)} of {len(ordered)} tips · "
                f"{n_shifting} with Shift > 0.05 · {n_unstable} with Comb. > 0.05"
            )
        else:
            table = dmc.Text(
                f"No taxa matching '{(search_text or '').strip()}'.",
                size="xs",
                c="dimmed",
                ta="center",
                mt="md",
            )
            summary = f"0 of {len(tip_rows)} tips"

        show_more_style = {} if len(shown) < len(ordered) else {"display": "none"}
        return (
            table,
            badge_text,
            {},
            summary,
            show_more_style,
            _tip_detail_footer(selected_row),
        )

    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("tip-instability-export-btn", "n_clicks"),
        State("clade-freq-result-key-store", "data"),
        prevent_initial_call=True,
    )
    def export_tip_instability_tsv(n_clicks, result_key):
        if not n_clicks or not result_key:
            return no_update
        cached = state.get_clade_frequency_result(result_key)
        if not cached or not cached.get("has_instability"):
            return no_update
        tip_rows = cached.get("tip_rows") or []
        if not tip_rows:
            return no_update
        label_1 = str(cached.get("label_1", "G1")).replace("/", "_").replace(" ", "_")
        label_2 = str(cached.get("label_2", "G2")).replace("/", "_").replace(" ", "_")
        path = _save_file_dialog(
            default_filename=f"{label_1}_vs_{label_2}_tip_instability.tsv"
        )
        if not path:
            return no_update
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(format_tip_instability_tsv(tip_rows))
        except OSError as exc:
            add_log(f"Failed to export tip instability TSV to {path}: {exc}", "ERROR")
            return dmc.Notification(
                title="Export Failed",
                message=str(exc),
                color="red",
                action="show",
                autoClose=4000,
                id=notif_id(),
            )
        add_log(f"Exported tip instability ({len(tip_rows)} tips) to {path}")
        return dmc.Notification(
            title="Tip Instability Exported",
            message=f"Saved {len(tip_rows)} tips to {path}",
            color="green",
            action="show",
            autoClose=3000,
            id=notif_id(),
        )

    # ------ Clade Frequency Comparison: enable Compare button ------

    @callback(
        Output("clade-freq-compare-button", "disabled"),
        Input("clade-freq-consensus-tree-select-1", "value"),
        Input("clade-freq-consensus-tree-select-2", "value"),
        Input("compute-busy-store", "data"),
    )
    def toggle_compare_button(uid1, uid2, compute_busy):
        """Enable the Compare button only when both dropdowns have a selection."""
        return bool(is_compute_busy(compute_busy) or not (uid1 and uid2))

    # ------ Clade Frequency Comparison: compute and plot ------

    @callback(
        Output("clade-freq-plot", "children", allow_duplicate=True),
        Output("clade-freq-data-store", "data", allow_duplicate=True),
        Output("clade-freq-output-paper", "style", allow_duplicate=True),
        Output("clade-freq-compare-button", "disabled", allow_duplicate=True),
        Output("clade-freq-job-store", "data"),
        Output("clade-freq-result-key-store", "data", allow_duplicate=True),
        Output("clade-freq-click-store", "data", allow_duplicate=True),
        Input("clade-freq-compare-button", "n_clicks"),
        State("clade-freq-consensus-tree-select-1", "value"),
        State("clade-freq-consensus-tree-select-2", "value"),
        State("clade-freq-min-clade-size", "value"),
        State("clade-freq-exclusive-only", "checked"),
        prevent_initial_call=True,
    )
    def compute_and_plot_clade_frequencies(
        n_clicks,
        uid1,
        uid2,
        min_clade_size,
        exclusive_only,
    ):
        """Prepare and submit a selective, process-isolated comparison."""
        if not n_clicks or not uid1 or not uid2:
            return (no_update,) * 7

        def error(message):
            return (
                dmc.Text(message, c="red", size="sm"),
                no_update,
                {},
                False,
                no_update,
                no_update,
                no_update,
            )

        entry1 = state.get_consensus_tree_registry_entry(uid1)
        entry2 = state.get_consensus_tree_registry_entry(uid2)

        if entry1 is None or entry2 is None:
            return error(
                "One or both selected consensus trees are no longer available. "
                "Please recompute them."
            )

        try:
            source_1 = str(entry1["source_distmat"])
            source_2 = str(entry2["source_distmat"])
            if source_1 != source_2:
                raise ValueError(
                    "Selected consensus trees belong to different RF matrices."
                )
            columns_1 = frozenset(
                int(column)
                for column in (entry1.get("cols_in_consensus_tree") or [])
            )
            columns_2 = frozenset(
                int(column)
                for column in (entry2.get("cols_in_consensus_tree") or [])
            )
            columns = sorted(columns_1 | columns_2)
            if not columns:
                raise ValueError(
                    "Selected consensus trees have no cached clade columns."
                )
            counts_1, n_trees_1 = _cached_counts_for_columns(entry1, columns)
            counts_2, n_trees_2 = _cached_counts_for_columns(entry2, columns)
            needs_fallback = counts_1 is None or counts_2 is None
            distmat_names = state.get_distmat_names(source_1)
            name_to_idx = {str(n): i for i, n in enumerate(distmat_names)}
            row_indices_1 = [
                name_to_idx[str(n)]
                for n in (entry1.get("tree_names") or [])
                if str(n) in name_to_idx
            ] or None
            row_indices_2 = [
                name_to_idx[str(n)]
                for n in (entry2.get("tree_names") or [])
                if str(n) in name_to_idx
            ] or None
            full_names = list(distmat_names) if needs_fallback else None
            snapshots_path = str(state.get_snapshots_path(source_1))
        except (KeyError, ValueError, FileNotFoundError) as exc:
            return error(f"Error preparing clade comparison: {exc}")

        try:
            min_size = max(1, int(min_clade_size or 2))
        except (TypeError, ValueError):
            min_size = 2
        context = _CladeComparisonFinalizationContext(
            source_distmat=source_1,
            uid_1=str(uid1),
            uid_2=str(uid2),
            label_1=str(entry1["name"]),
            label_2=str(entry2["name"]),
            consensus_columns_1=columns_1,
            consensus_columns_2=columns_2,
            min_clade_size=min_size,
            exclusive_only=bool(exclusive_only),
        )

        try:
            job_ref = job_manager.submit(
                _get_executor(),
                "clade_compare",
                persistent_worker.submit_job,
                "compute_clade_frequencies",
                snapshots_path=snapshots_path,
                columns=columns,
                counts_1=counts_1,
                counts_2=counts_2,
                n_trees_1=n_trees_1,
                n_trees_2=n_trees_2,
                tree_names_1=(
                    list(entry1.get("tree_names") or [])
                    if counts_1 is None
                    else None
                ),
                tree_names_2=(
                    list(entry2.get("tree_names") or [])
                    if counts_2 is None
                    else None
                ),
                full_distmat_names=full_names,
                row_indices_1=row_indices_1,
                row_indices_2=row_indices_2,
                metadata={
                    "display_name": "Clade Frequency Comparison",
                    "source_distmat": source_1,
                },
                finalizer=partial(
                    _finalize_clade_comparison_job,
                    context=context,
                ),
                cancel_exceptions=(persistent_worker.JobCancelled,),
            )
        except JobBusyError as exc:
            message = (
                f"Another computation ({exc.active.kind.replace('_', ' ').upper()}) "
                "is still finishing. Please wait for it to complete."
            )
            add_log(message, "WARNING")
            return error(message)

        spinner = dmc.Group(
            [
                dmc.Loader(size="sm", type="dots"),
                dmc.Text(
                    f"Comparing {len(columns)} consensus-tree clades…",
                    size="sm",
                    c="dimmed",
                ),
                stop_button("clade-compare"),
            ],
            gap="sm",
        )
        return (
            spinner,
            None,
            {},
            True,
            job_ref.as_dict(),
            None,
            None,
        )

    @callback(
        Output("clade-freq-plot", "children", allow_duplicate=True),
        Output("clade-freq-data-store", "data", allow_duplicate=True),
        Output("clade-freq-output-paper", "style", allow_duplicate=True),
        Output("clade-freq-result-key-store", "data", allow_duplicate=True),
        Output("clade-freq-click-store", "data", allow_duplicate=True),
        Output(
            {"type": "compute-terminal-receipt", "kind": "clade-compare"},
            "data",
        ),
        Input("compute-terminal-event-store", "data"),
        Input("clade-freq-job-store", "data"),
        prevent_initial_call=True,
    )
    def render_clade_frequency_terminal_event(terminal_event, job_data):
        event = terminal_event_for_job(
            terminal_event,
            job_data,
            expected_kind="clade_compare",
        )
        if event is None:
            return (no_update,) * 6

        terminal_state = JobState(str(event["state"]))
        payload = event["payload"]
        first_delivery = int(event["delivery_attempt"]) == 1
        store_data = no_update
        result_key = no_update
        if terminal_state is JobState.CANCELLED:
            if first_delivery:
                add_log("Clade comparison cancelled by user.", "WARNING")
            output = dmc.Alert(
                title="Clade comparison cancelled",
                children=dmc.Text("Stopped before completion.", size="sm"),
                color="gray",
                variant="light",
            )
        elif terminal_state is JobState.FAILED:
            message = str(payload.get("message", "Unknown error"))
            if first_delivery:
                add_log(f"Clade comparison failed: {message}", "ERROR")
            output = dmc.Text(
                f"Error computing clade frequencies: {message}",
                c="red",
                size="sm",
            )
        else:
            cached = state.get_clade_frequency_result(
                payload.get("result_key")
            )
            if cached is None:
                output = dmc.Text(
                    "Clade comparison result is no longer available. "
                    "Please recompute it.",
                    c="red",
                    size="sm",
                )
            else:
                store_data = cached["records"]
                result_key = payload["result_key"]
                output = dcc.Graph(
                    id="clade-freq-scatter",
                    figure=cached["figure"],
                    config={"displayModeBar": False, "doubleClick": "reset"},
                    style={"width": "100%"},
                )

        return (
            output,
            store_data,
            {},
            result_key,
            None,
            terminal_delivery_marker(event),
        )

    @callback(
        Output("clade-freq-click-store", "data"),
        Output("clade-freq-scatter", "clickData"),
        Input("clade-freq-scatter", "clickData"),
        State("clade-freq-result-key-store", "data"),
        State("clade-freq-click-store", "data"),
        State("tip-instability-filter-store", "data"),
        prevent_initial_call=True,
    )
    def store_scatter_click(click_data, result_key, current_click, tip_filter):
        """Forward a scatter click to the click store; clicking the selected dot toggles it off.

        Clears ``clade-freq-scatter.clickData`` back to ``None`` on every handled click
        because Dash's renderer deep-equals ``clickData`` against the previous value and
        otherwise swallows a second click on the exact same dot.
        """
        nonlocal _click_counter
        if not result_key or not click_data or not click_data.get("points"):
            return no_update, no_update
        result_key_s = str(result_key)
        points = click_data["points"]
        point = points[0]
        custom = point.get("customdata")
        if custom is None:
            return no_update, None
        try:
            clicked_split = int(custom[0])
            clicked_size = int(custom[1])
            clicked_x = float(point["x"])
            clicked_y = float(point["y"])
        except (TypeError, ValueError, IndexError, KeyError):
            return no_update, None

        if (
            isinstance(current_click, dict)
            and current_click.get("kind", "clade") == "clade"
            and str(current_click.get("result_key") or "") == result_key_s
            and current_click.get("split_id") is not None
            and any(
                pt.get("customdata")
                and str(pt["customdata"][0]) == str(int(current_click["split_id"]))
                for pt in points
            )
        ):
            if (
                isinstance(tip_filter, dict)
                and tip_filter.get("taxon")
                and str(tip_filter.get("result_key") or "") == result_key_s
            ):
                _click_counter += 1
                return (
                    {
                        "kind": "tip",
                        "taxon": str(tip_filter["taxon"]),
                        "result_key": result_key_s,
                        "_t": _click_counter,
                    },
                    None,
                )
            return None, None

        _click_counter += 1
        return (
            {
                "kind":       "clade",
                "split_id":   clicked_split,
                "clade_size": clicked_size,
                "x":          clicked_x,
                "y":          clicked_y,
                "result_key": result_key_s,
                "_t":         _click_counter,
            },
            None,
        )

    @callback(
        Output("clade-freq-click-store", "data", allow_duplicate=True),
        Output("tip-instability-filter-store", "data", allow_duplicate=True),
        Input({"type": "tip-instability-focus", "taxon": ALL}, "n_clicks"),
        State("clade-freq-result-key-store", "data"),
        State("tip-instability-filter-store", "data"),
        prevent_initial_call=True,
    )
    def select_tip_instability_row(n_clicks_list, result_key, tip_filter):
        """Filter the scatter to a tip and focus it in the tanglegram (or toggle off)."""
        nonlocal _click_counter
        if not n_clicks_list or not any(n_clicks_list) or not result_key:
            return no_update, no_update
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or triggered.get("type") != "tip-instability-focus":
            return no_update, no_update
        taxon = str(triggered["taxon"])
        if isinstance(tip_filter, dict) and tip_filter.get("taxon") == taxon:
            return None, None
        _click_counter += 1
        return (
            {
                "kind": "tip",
                "taxon": taxon,
                "result_key": str(result_key),
                "_t": _click_counter,
            },
            {"taxon": taxon, "result_key": str(result_key)},
        )

    @callback(
        Output("clade-freq-click-store", "data", allow_duplicate=True),
        Output("tip-instability-filter-store", "data", allow_duplicate=True),
        Input("clade-freq-tip-banner-dismiss", "n_clicks"),
        State("clade-freq-click-store", "data"),
        prevent_initial_call=True,
    )
    def dismiss_tip_selection_banner(n_clicks, click_data):
        if not n_clicks:
            return no_update, no_update
        if isinstance(click_data, dict) and click_data.get("kind", "clade") == "clade":
            return no_update, None
        return None, None

    @callback(
        Output("clade-freq-scatter", "figure", allow_duplicate=True),
        Input("clade-freq-click-store", "data"),
        State("clade-freq-data-store", "data"),
        prevent_initial_call=True,
    )
    def update_click_marker(click_data, store_data):
        """Patch only scatter overlay trace 1 (single-clade ring) without altering data[0]."""
        if not store_data:
            return no_update
        patch = Patch()
        if isinstance(click_data, dict) and click_data.get("kind", "clade") == "clade":
            x = click_data.get("x")
            y = click_data.get("y")
            if x is None or y is None:
                return no_update
            patch["data"][1]["x"] = [x]
            patch["data"][1]["y"] = [y]
            return patch
        patch["data"][1]["x"] = []
        patch["data"][1]["y"] = []
        return patch

    @callback(
        Output("clade-freq-tanglegram", "figure"),
        Output("clade-freq-tanglegram-pair-store", "data"),
        Output("clade-freq-tanglegram-title", "children"),
        Output("tanglegram-complement-toggle", "disabled"),
        Input("clade-freq-click-store", "data"),
        State("clade-freq-consensus-tree-select-1", "value"),
        State("clade-freq-consensus-tree-select-2", "value"),
        State("tanglegram-yscale-slider", "value"),
        State("clade-freq-tanglegram-pair-store", "data"),
        State("tanglegram-complement-toggle", "checked"),
        State("tanglegram-sister-toggle", "checked"),
        prevent_initial_call=True,
    )
    def draw_tanglegram(
        click_data,
        uid1,
        uid2,
        px_per_tip,
        current_pair,
        complement_on,
        sister_on,
    ):
        """Draw or patch the tanglegram for a clicked clade, selected tip, or dismiss."""
        same_pair = bool(uid1 and uid2 and current_pair == [uid1, uid2])
        if not click_data:
            if not same_pair:
                return no_update, no_update, no_update, False
            # Clear dynamic overlays back to the dimmed skeleton on dismiss / filter change.
            patch = Patch()
            for idx in (
                _TANGLEGRAM_MRCA_SUBTREE,
                _TANGLEGRAM_SISTER_BRANCH_LEFT,
                _TANGLEGRAM_SISTER_BRANCH_RIGHT,
            ):
                patch["data"][idx]["x"] = []
                patch["data"][idx]["y"] = []
            for idx in (
                _TANGLEGRAM_TIP_SISTER_CONN_LEFT,
                _TANGLEGRAM_TIP_SISTER_CONN_RIGHT,
                _TANGLEGRAM_TIP_SISTER_TIPS_LEFT,
                _TANGLEGRAM_TIP_SISTER_TIPS_RIGHT,
                _TANGLEGRAM_HIGHLIGHT_LEFT,
                _TANGLEGRAM_HIGHLIGHT_RIGHT,
                _TANGLEGRAM_CONNECTORS,
                _TANGLEGRAM_MRCA_MARKER,
                _TANGLEGRAM_SISTER_MRCA_LEFT,
                _TANGLEGRAM_SISTER_MRCA_RIGHT,
                _TANGLEGRAM_COMPLEMENT_TIPS,
                _TANGLEGRAM_COMPLEMENT_CONN,
            ):
                patch["data"][idx]["x"] = []
                patch["data"][idx]["y"] = []
                patch["data"][idx]["text"] = []
            return patch, no_update, "", False

        if not uid1 or not uid2:
            return no_update, no_update, no_update, no_update

        layout = _get_tanglegram_layout(uid1, uid2)
        if layout is None:
            return no_update, no_update, no_update, no_update

        tips1 = layout["tips1"]
        tips2 = layout["tips2"]
        right_start = layout["right_start"]
        kind = click_data.get("kind", "clade")

        if kind == "tip":
            cached = state.get_clade_frequency_result(click_data.get("result_key"))
            if cached is None or tuple(cached.get("pair", ())) != (uid1, uid2):
                return no_update, no_update, no_update, no_update
            taxon = str(click_data.get("taxon") or "")
            if not taxon:
                return no_update, no_update, no_update, no_update

            highlight = {taxon}
            hl_left = _highlight_overlay_trace(tips1, highlight)
            hl_right = _highlight_overlay_trace(tips2, highlight)
            connectors = _connector_overlay_trace(tips1, tips2, highlight)
            mrca_subtree = _branch_overlay_trace([], [], _MRCA_ACCENT)
            mrca_marker = _mrca_diamond_trace([], [], [], _MRCA_MARKER)
            (
                sister_branch_left,
                sister_branch_right,
                tip_sister_conn_left,
                tip_sister_conn_right,
                tip_sister_tips_left,
                tip_sister_tips_right,
                sister_mrca_left,
                sister_mrca_right,
            ) = _tip_sister_overlay_traces(taxon, layout, connect=bool(sister_on))
            comp_tips = _complement_tips_trace(set(), tips1, tips2)
            comp_conn = _complement_connector_trace(tips1, tips2, set(), set())
            tip_row = next(
                (r for r in (cached.get("tip_rows") or []) if r["taxon"] == taxon),
                None,
            )
            title_children = _tanglegram_tip_title_children(taxon, tip_row)
            complement_disabled = True
        else:
            resolved = _resolved_clade(click_data, expected_pair=(uid1, uid2))
            if resolved is None:
                return no_update, no_update, no_update, no_update
            column_j = int(resolved["column_j"])
            highlight = set(resolved["tip_names"])

            entry1 = state.get_consensus_tree_registry_entry(uid1)
            entry2 = state.get_consensus_tree_registry_entry(uid2)
            cols1 = set(entry1.get("cols_in_consensus_tree") or []) if entry1 else set()
            cols2 = set(entry2.get("cols_in_consensus_tree") or []) if entry2 else set()
            in_1 = column_j in cols1
            in_2 = column_j in cols2
            label1 = entry1["name"] if entry1 else "Group 1"
            label2 = entry2["name"] if entry2 else "Group 2"

            hl_left = _highlight_overlay_trace(tips1, highlight)
            hl_right = _highlight_overlay_trace(tips2, highlight)
            connectors = _connector_overlay_trace(tips1, tips2, highlight)
            mrca_subtree, mrca_marker, complement_per_tree = _build_mrca_traces(
                highlight, layout
            )
            (
                sister_branch_left,
                sister_branch_right,
                tip_sister_conn_left,
                tip_sister_conn_right,
                tip_sister_tips_left,
                tip_sister_tips_right,
                sister_mrca_left,
                sister_mrca_right,
            ) = _clade_sister_overlay_traces(highlight, layout, connect=bool(sister_on))
            complement1 = complement_per_tree["tips1"]
            complement2 = complement_per_tree["tips2"]
            title_children = _tanglegram_title_children(
                label1, label2, highlight, in_1=in_1, in_2=in_2,
            )
            if complement_on:
                comp_tips = _complement_tips_trace(complement1 | complement2, tips1, tips2)
                comp_conn = _complement_connector_trace(
                    tips1, tips2, complement1, complement2,
                )
            else:
                comp_tips = _complement_tips_trace(set(), tips1, tips2)
                comp_conn = _complement_connector_trace(tips1, tips2, set(), set())
            complement_disabled = False

        if same_pair:
            patch = Patch()
            for idx, trace in (
                (_TANGLEGRAM_MRCA_SUBTREE,        mrca_subtree),
                (_TANGLEGRAM_SISTER_BRANCH_LEFT,  sister_branch_left),
                (_TANGLEGRAM_SISTER_BRANCH_RIGHT, sister_branch_right),
            ):
                patch["data"][idx]["x"] = trace["x"]
                patch["data"][idx]["y"] = trace["y"]
            for idx, trace in (
                (_TANGLEGRAM_TIP_SISTER_CONN_LEFT,  tip_sister_conn_left),
                (_TANGLEGRAM_TIP_SISTER_CONN_RIGHT, tip_sister_conn_right),
                (_TANGLEGRAM_TIP_SISTER_TIPS_LEFT,  tip_sister_tips_left),
                (_TANGLEGRAM_TIP_SISTER_TIPS_RIGHT, tip_sister_tips_right),
                (_TANGLEGRAM_HIGHLIGHT_LEFT,        hl_left),
                (_TANGLEGRAM_HIGHLIGHT_RIGHT,       hl_right),
                (_TANGLEGRAM_CONNECTORS,            connectors),
                (_TANGLEGRAM_MRCA_MARKER,           mrca_marker),
                (_TANGLEGRAM_SISTER_MRCA_LEFT,      sister_mrca_left),
                (_TANGLEGRAM_SISTER_MRCA_RIGHT,     sister_mrca_right),
                (_TANGLEGRAM_COMPLEMENT_TIPS,       comp_tips),
                (_TANGLEGRAM_COMPLEMENT_CONN,       comp_conn),
            ):
                patch["data"][idx]["x"] = trace["x"]
                patch["data"][idx]["y"] = trace["y"]
                patch["data"][idx]["text"] = trace["text"]
            return patch, no_update, title_children, complement_disabled

        skel = layout["skeleton_traces"]
        traces = [
            skel[0], skel[2], mrca_subtree, sister_branch_left, sister_branch_right,
            skel[1], skel[3], tip_sister_conn_left, tip_sister_conn_right,
            tip_sister_tips_left, tip_sister_tips_right,
            hl_left, hl_right, connectors, mrca_marker,
            sister_mrca_left, sister_mrca_right,
            comp_tips, comp_conn,
        ]
        height = _tanglegram_height(px_per_tip, layout["max_y"])
        fig = go.Figure(data=[
            t if isinstance(t, go.Scatter) else go.Scatter(**t)
            for t in traces
        ])
        backbone = _backbone_color()
        fig.data[0].line.color = backbone
        fig.data[1].line.color = backbone
        fig.update_layout(
            template=get_template(),
            height=height,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(visible=False, range=[-0.05, right_start + 1.05]),
            yaxis=dict(visible=False),
            hovermode="closest",
            uirevision="clade-freq-tanglegram",
        )
        return fig, [uid1, uid2], title_children, complement_disabled

    # ------ theme toggle → re-theme the Clade Exploration plots ------
    # Mirrors treespace's ``update_plot_theme``: rebuild with the active
    # template so backgrounds/axes follow light/dark. Beyond the template
    # we also flip the theme-sensitive trace colours that aren't
    # template-derived — the scatter's colour scale and the tanglegram's
    # backbone branches.
    @callback(
        Output("clade-freq-scatter", "figure", allow_duplicate=True),
        Input("plotly-template-store", "data"),
        State("clade-freq-scatter", "figure"),
        prevent_initial_call=True,
    )
    def retheme_clade_scatter(_template, current_fig):
        if not current_fig:
            return no_update
        fig = retheme_figure(current_fig, skip_invalid=True)
        # The clade-size scale isn't template-derived; swap it too. The
        # data trace (index 0) is the only one carrying a colour scale.
        if len(fig.data):
            try:
                fig.data[0].marker.colorscale = _scatter_colorscale()
            except (AttributeError, ValueError):
                pass
        return fig

    @callback(
        Output("clade-freq-tanglegram", "figure", allow_duplicate=True),
        Input("plotly-template-store", "data"),
        State("clade-freq-tanglegram", "figure"),
        prevent_initial_call=True,
    )
    def retheme_clade_tanglegram(_template, current_fig):
        if not current_fig or not current_fig.get("data"):
            return no_update
        fig = retheme_figure(current_fig, skip_invalid=True)
        backbone = _backbone_color()
        for idx in (0, 1):   # the two skeleton (backbone) branch traces
            if idx < len(fig.data):
                try:
                    fig.data[idx].line.color = backbone
                except (AttributeError, ValueError):
                    pass
        return fig

    @callback(
        Output("clade-freq-tanglegram", "figure", allow_duplicate=True),
        Input("tanglegram-yscale-slider", "value"),
        State("clade-freq-consensus-tree-select-1", "value"),
        State("clade-freq-consensus-tree-select-2", "value"),
        prevent_initial_call=True,
    )
    def update_tanglegram_height(px_per_tip, uid1, uid2):
        """Slide-to-resize. The tanglegram's height scales with the
        number of tips × the slider value. Patches only
        ``layout.height`` so the 11-trace figure doesn't get rebuilt
        on every slider drag tick.

        No-ops when no consensus tree pair is selected yet (slider has nothing
        to resize against) or when the layout cache is cold (no
        click has rendered the tanglegram yet).
        """
        if not uid1 or not uid2:
            return no_update
        layout = _get_tanglegram_layout(uid1, uid2)
        if layout is None:
            return no_update
        patch = Patch()
        patch["layout"]["height"] = _tanglegram_height(
            px_per_tip, layout["max_y"]
        )
        return patch

    @callback(
        Output("clade-freq-tanglegram", "figure", allow_duplicate=True),
        Input("tanglegram-complement-toggle", "checked"),
        State("clade-freq-click-store", "data"),
        State("clade-freq-consensus-tree-select-1", "value"),
        State("clade-freq-consensus-tree-select-2", "value"),
        prevent_initial_call=True,
    )
    def toggle_complement_highlights(checked, click_data, uid1, uid2):
        """Show or hide the complementary-tip green overlay when the
        toggle is flipped, without requiring a new scatter click.

        Patches only the two green overlay traces (9/10); the grey tip
        base (3/4) is left untouched because the green markers simply
        draw on top of it. Hiding empties the green x/y/text; showing
        resolves the last clicked split from its managed result and rebuilds
        them. Their styling was baked in at first render, so
        the patch never re-sends marker/line/mode.
        """
        # Need a rendered clade tanglegram — Clade complement is disabled in tip mode.
        if not click_data or click_data.get("kind") == "tip" or not uid1 or not uid2:
            return no_update
        layout = _get_tanglegram_layout(uid1, uid2)
        if layout is None:
            return no_update

        patch = Patch()
        if not checked:
            for idx in (_TANGLEGRAM_COMPLEMENT_TIPS, _TANGLEGRAM_COMPLEMENT_CONN):
                patch["data"][idx]["x"] = []
                patch["data"][idx]["y"] = []
                patch["data"][idx]["text"] = []
            return patch

        resolved = _resolved_clade(click_data, expected_pair=(uid1, uid2))
        if resolved is None:
            return no_update
        highlight = set(resolved["tip_names"])

        _, _, complement_per_tree = _build_mrca_traces(highlight, layout)
        complement1 = complement_per_tree["tips1"]
        complement2 = complement_per_tree["tips2"]

        tips1 = layout["tips1"]
        tips2 = layout["tips2"]

        comp_tips = _complement_tips_trace(complement1 | complement2, tips1, tips2)
        comp_conn = _complement_connector_trace(tips1, tips2, complement1, complement2)

        for idx, trace in (
            (_TANGLEGRAM_COMPLEMENT_TIPS, comp_tips),
            (_TANGLEGRAM_COMPLEMENT_CONN, comp_conn),
        ):
            patch["data"][idx]["x"] = trace["x"]
            patch["data"][idx]["y"] = trace["y"]
            patch["data"][idx]["text"] = trace["text"]
        return patch

    @callback(
        Output("clade-freq-tanglegram", "figure", allow_duplicate=True),
        Input("tanglegram-sister-toggle", "checked"),
        State("clade-freq-click-store", "data"),
        State("clade-freq-consensus-tree-select-1", "value"),
        State("clade-freq-consensus-tree-select-2", "value"),
        prevent_initial_call=True,
    )
    def toggle_sister_connectors(checked, click_data, uid1, uid2):
        """Show or hide cross-tree G1/G2 sister connectors and landing tips."""
        if not click_data or not uid1 or not uid2:
            return no_update
        layout = _get_tanglegram_layout(uid1, uid2)
        if layout is None:
            return no_update

        if click_data.get("kind") == "tip":
            taxon = str(click_data.get("taxon") or "")
            if not taxon:
                return no_update
            (
                branch_left,
                branch_right,
                conn_left,
                conn_right,
                tips_left,
                tips_right,
                mrca_left,
                mrca_right,
            ) = _tip_sister_overlay_traces(taxon, layout, connect=bool(checked))
        else:
            resolved = _resolved_clade(click_data, expected_pair=(uid1, uid2))
            if resolved is None:
                return no_update
            (
                branch_left,
                branch_right,
                conn_left,
                conn_right,
                tips_left,
                tips_right,
                mrca_left,
                mrca_right,
            ) = _clade_sister_overlay_traces(
                set(resolved["tip_names"]), layout, connect=bool(checked)
            )

        patch = Patch()
        for idx, trace in (
            (_TANGLEGRAM_SISTER_BRANCH_LEFT,  branch_left),
            (_TANGLEGRAM_SISTER_BRANCH_RIGHT, branch_right),
        ):
            patch["data"][idx]["x"] = trace["x"]
            patch["data"][idx]["y"] = trace["y"]
        for idx, trace in (
            (_TANGLEGRAM_TIP_SISTER_CONN_LEFT,  conn_left),
            (_TANGLEGRAM_TIP_SISTER_CONN_RIGHT, conn_right),
            (_TANGLEGRAM_TIP_SISTER_TIPS_LEFT,  tips_left),
            (_TANGLEGRAM_TIP_SISTER_TIPS_RIGHT, tips_right),
            (_TANGLEGRAM_SISTER_MRCA_LEFT,      mrca_left),
            (_TANGLEGRAM_SISTER_MRCA_RIGHT,     mrca_right),
        ):
            patch["data"][idx]["x"] = trace["x"]
            patch["data"][idx]["y"] = trace["y"]
            patch["data"][idx]["text"] = trace["text"]
        return patch
