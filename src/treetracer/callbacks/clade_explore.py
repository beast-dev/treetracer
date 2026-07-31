"""Clade Exploration tab — consensus-tree clade-frequency comparison.

Split out of ``callbacks/diagnostics.py``: the clade-frequency scatter
+ tanglegram is exploratory phylogenetics, not a convergence
diagnostic, so it lives in its own tab and its own module. This module
owns the tab's RF-matrix selector, the per-matrix consensus tree table, the
two-consensus-tree Compare workflow, and the scatter / tanglegram rendering.
"""

import functools

from dash import (dcc, html, callback, Input, Output, State, no_update, Patch,
                  clientside_callback)
import dash_mantine_components as dmc
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from .. import state
from ..clade_freq import compute_clade_frequencies
from ..clade_freq.layout import parse_nexus, build_tree_traces, _collect_nodes
from ..theme import get_template, DARK_TEMPLATE
from ..plot_utils import retheme_figure


# Server-side resolution table for the Clade Frequency scatter →
# tanglegram round-trip. The scatter's ``customdata`` carries an
# integer ``split_id`` (the row index in the DataFrame produced by
# ``compute_clade_frequencies``); this dict maps that id to a
# ``(source_distmat, column_j, split_key)`` triple so the click
# handler can both resolve the clade's tip names AND check consensus tree
# membership via ``column_j in cols_in_consensus_tree`` (the rapidtrees-encoded
# rooted-clade column index in that distmat's snapshot). Rebuilt on
# every Compare click.
#
# Keeping the keys server-side avoids serialising tens of thousands
# of tuples through the browser store on every click.
_split_resolution: dict[int, tuple[str, int, tuple]] = {}


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
    }


# Trace indices in the assembled tanglegram figure. Every branch trace
# comes first, so the tip markers — drawn afterwards — sit on top of
# the MRCA-subtree recolour instead of being hidden under it. The
# skeleton (indices 0, 1, 3, 4) is static; the rest are dynamic
# overlays the click callback patches.
#
#   0: left branches         (static, dimmed)
#   1: right branches        (static, dimmed)
#   2: MRCA subtree branches (dynamic) — both trees, accent colour
#   3: left grey tips        (static)
#   4: right grey tips       (static)
#   5: left red highlight    (dynamic)
#   6: right red highlight   (dynamic)
#   7: red connectors        (dynamic)
#   8: MRCA node markers     (dynamic) — both trees
#   9: complement tips       (dynamic) — green, both trees
#  10: complement connectors (dynamic) — green
_TANGLEGRAM_MRCA_SUBTREE    = 2
_TANGLEGRAM_HIGHLIGHT_LEFT  = 5
_TANGLEGRAM_HIGHLIGHT_RIGHT = 6
_TANGLEGRAM_CONNECTORS      = 7
_TANGLEGRAM_MRCA_MARKER     = 8
_TANGLEGRAM_COMPLEMENT_TIPS = 9
_TANGLEGRAM_COMPLEMENT_CONN = 10

# Colours for the MRCA emphasis overlay.
_SKELETON_DIM = "#c8c8c8"   # branches outside the MRCA subtree
_MRCA_ACCENT  = "#1c7ed6"   # branches inside the MRCA subtree
_MRCA_MARKER  = "#1864ab"   # the MRCA node marker

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


def _highlight_overlay_trace(tips_by_name, highlight):
    """Build the red-marker overlay trace for one tree.

    ``tips_by_name`` is the layout cache's tip-name → (x, y) dict.
    Returns the trace as a plain dict so Patch can splat its
    individual fields without ever going through a go.Scatter
    constructor."""
    xs, ys, names = [], [], []
    for name in highlight:
        coord = tips_by_name.get(name)
        if coord is None:
            continue
        xs.append(coord[0])
        ys.append(coord[1])
        names.append(name)
    return dict(
        type="scatter",
        x=xs, y=ys,
        mode="markers",
        marker=dict(color="#e63946", size=8),
        text=names,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )


def _connector_overlay_trace(tips1, tips2, highlight):
    """Build the red-line connector overlay between the two trees for
    all tip names in *highlight* that exist on both sides."""
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
        line=dict(color="rgba(230,57,70,0.7)", width=1.5),
        text=[n for n in names for _ in range(3)],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
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
    actually wipes the bipartition→tip-set decode, the parsed-NEXUS
    LRU, and the click→split lookup table — they're keyed on consensus tree
    uuids that are about to disappear from ``state._consensus_tree_cache``.
    """
    _get_parsed_consensus_tree.cache_clear()
    _get_tanglegram_layout.cache_clear()
    _split_resolution.clear()


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
            colorbar=dict(title="Clade size", thickness=12),
            opacity=0.75,
        ),
        # customdata carries the row's split_id (integer key into the
        # per-distmat canonical_keys cache) and clade_size. Using an
        # integer ID dodges the previous fragile comma-joined-string
        # round-trip — taxon names with embedded commas no longer break
        # the click→tanglegram path.
        # ``.values.tolist()`` converts the numpy int32 array to native
        # Python ints in nested lists. Scattergl serialises this more
        # reliably through clickData than a raw numpy 2D array — without
        # it some Plotly versions drop customdata or pass it as a flat
        # array, which breaks the click→tanglegram resolution below.
        customdata=df_plot[
            ["split_id", "clade_size", "consensus_tree_membership"]
        ].values.tolist(),
        hovertemplate=(
            "<b>Clade (%{customdata[1]} tips)</b><br>"
            "Group 1: %{x:.3f}<br>"
            "Group 2: %{y:.3f}<br>"
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
    )
    return fig


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

    # ------ Clade Frequency Comparison: filter the scatter ------

    @callback(
        Output("clade-freq-scatter", "figure", allow_duplicate=True),
        Input("clade-freq-min-clade-size", "value"),
        State("clade-freq-data-store", "data"),
        prevent_initial_call=True,
    )
    def filter_clade_freq_plot(min_clade_size, store_data):
        """Filter the scatter on min-clade-size slider changes.

        Patches only ``data[0]`` (the Scattergl trace) — x/y/customdata/
        marker.color — so Plotly doesn't rebuild the figure or remount
        the dcc.Graph on every drag tick. The click-marker overlay
        (trace 1) is left intact, which means a previously-clicked
        point's red ring can land on a filtered-away coordinate; the
        user just re-clicks if they want it on a currently-visible
        point.
        """
        if not store_data:
            return no_update

        df = pd.DataFrame(store_data)
        min_size = int(min_clade_size or 2)
        df_plot = df[df["clade_size"] >= min_size]

        patch = Patch()
        patch["data"][0]["x"] = df_plot["freq_1"].tolist()
        patch["data"][0]["y"] = df_plot["freq_2"].tolist()
        patch["data"][0]["customdata"] = (
            df_plot[["split_id", "clade_size", "consensus_tree_membership"]].values.tolist()
        )
        patch["data"][0]["marker"]["color"] = df_plot["clade_size"].tolist()
        return patch

    # ------ Clade Frequency Comparison: enable Compare button ------

    @callback(
        Output("clade-freq-compare-button", "disabled"),
        Input("clade-freq-consensus-tree-select-1", "value"),
        Input("clade-freq-consensus-tree-select-2", "value"),
    )
    def toggle_compare_button(uid1, uid2):
        """Enable the Compare button only when both dropdowns have a selection."""
        return not (uid1 and uid2)

    # ------ Clade Frequency Comparison: compute and plot ------

    # Instant feedback on click: flip the output paper visible so the
    # ``dcc.Loading`` wrapper around ``clade-freq-plot`` (defined in
    # ui/panels/clade_freq.py) can render its spinner while the slow
    # server compute below runs. Without this the paper stays hidden
    # until the compute returns, so the user sees no loader at all.
    # Runs clientside (no Python roundtrip) so the spinner appears
    # within a frame of the click.
    clientside_callback(
        """
        function(n_clicks) {
            if (!n_clicks) return window.dash_clientside.no_update;
            return {};
        }
        """,
        Output("clade-freq-output-paper", "style", allow_duplicate=True),
        Input("clade-freq-compare-button", "n_clicks"),
        prevent_initial_call=True,
    )

    @callback(
        Output("clade-freq-plot", "children"),
        Output("clade-freq-data-store", "data"),
        # Toggle the output Paper visible only on success; stays
        # hidden on any error path or before the first successful
        # Compare. Cleared by the sidebar's Clear-data flow.
        Output("clade-freq-output-paper", "style"),
        Input("clade-freq-compare-button", "n_clicks"),
        State("clade-freq-consensus-tree-select-1", "value"),
        State("clade-freq-consensus-tree-select-2", "value"),
        State("clade-freq-min-clade-size", "value"),
        prevent_initial_call=True,
    )
    def compute_and_plot_clade_frequencies(n_clicks, uid1, uid2, min_clade_size):
        """Compute clade frequencies for the two selected consensus tree groups and
        render a scatter plot (freq group 1 vs freq group 2).

        Each dot is one bipartition observed in either group. Dot colour
        encodes clade_size — the number of tips in the monophyletic
        descendant side of the bipartition in the consensus tree(s) that
        contain it (so it matches what the tanglegram highlights when
        the dot is clicked). Clicking a dot triggers the tanglegram
        callback.
        """
        if not uid1 or not uid2:
            return no_update, no_update, no_update

        entry1 = state.get_consensus_tree_registry_entry(uid1)
        entry2 = state.get_consensus_tree_registry_entry(uid2)

        if entry1 is None or entry2 is None:
            return dmc.Text(
                "One or both selected consensus trees are no longer available. "
                "Please recompute them.",
                c="red", size="sm",
            ), no_update, no_update

        try:
            df = compute_clade_frequencies(entry1, entry2)
        except (KeyError, FileNotFoundError) as e:
            return dmc.Text(
                f"Error computing clade frequencies: {e}",
                c="red", size="sm",
            ), no_update, no_update

        # Annotate every rooted clade with whether it's present in
        # consensus tree 1 / consensus tree 2. The dropdowns in ``populate_consensus_tree_selects``
        # restrict choices to the active distmat, so by construction
        # ``entry1.source_distmat == entry2.source_distmat`` and both
        # ``cols_in_consensus_tree`` lists are in the same column basis as the
        # DataFrame's ``column_j`` — a pure int-in-set check.
        src1 = entry1["source_distmat"]
        cols_in_consensus_tree_1 = entry1.get("cols_in_consensus_tree") or []
        cols_in_consensus_tree_2 = entry2.get("cols_in_consensus_tree") or []
        df["in_consensus_tree_1"] = df["column_j"].isin(set(cols_in_consensus_tree_1))
        df["in_consensus_tree_2"] = df["column_j"].isin(set(cols_in_consensus_tree_2))

        # Show only clades that are present in at least one of the two
        # consensus trees — keeps the tanglegram meaningful when the user clicks.
        df = df[df["in_consensus_tree_1"] | df["in_consensus_tree_2"]].reset_index(drop=True)

        # No clade-size re-stamping in rooted mode: each rapidtrees
        # column is already a rooted clade with one specific descendant
        # set, so the size computed in ``compute_clade_frequencies``
        # (``len(split_key)``) is already the size we want to display.

        if df.empty:
            return dmc.Text(
                "None of the bipartitions observed in the two groups "
                "is a clade of either consensus tree — nothing to plot.",
                c="dimmed", size="sm",
            ), no_update, no_update

        # Pre-render a human-readable membership label per row for
        # the scatter hover. Stored in the DataFrame so the slider
        # callback can patch ``customdata`` without rebuilding the
        # mapping.
        label1_h = entry1["name"]
        label2_h = entry2["name"]
        membership_labels = []
        for in1, in2 in zip(df["in_consensus_tree_1"], df["in_consensus_tree_2"]):
            if in1 and in2:
                membership_labels.append("both consensus trees")
            elif in1:
                membership_labels.append(f"{label1_h} only")
            else:
                membership_labels.append(f"{label2_h} only")
        df["consensus_tree_membership"] = membership_labels

        # Integer row id replaces the old fragile comma-joined string.
        # The click-handler + tanglegram callbacks resolve split_id to
        # tip names via state.get_canonical_keys at render time.
        df["split_id"] = np.arange(len(df), dtype=np.int32)

        # Refresh the click-resolution table: split_id → (distmat,
        # column_j, split_key). ``column_j`` is the rapidtrees-encoded
        # rooted-clade column index in the snapshot; the click handler
        # uses it for the in_1/in_2 check against each consensus tree's
        # ``cols_in_consensus_tree``. ``split_key`` is the descendant-set tuple
        # of leaf indices, used only to resolve tip names for the
        # highlight overlay.
        _split_resolution.clear()
        for split_id, col_j, key in zip(
            df["split_id"].tolist(),
            df["column_j"].tolist(),
            df["split_key"].tolist(),
        ):
            _split_resolution[int(split_id)] = (src1, int(col_j), key)

        label1 = entry1["name"]
        label2 = entry2["name"]

        # Serialise for the slider callback. split_key is a tuple[int]
        # — not JSON-serialisable, so it stays server-side and we only
        # ship the integer id through the browser. The in_consensus_tree_1/
        # in_consensus_tree_2 booleans ride along so any future filter or
        # colour-coding callback can consume them without re-running
        # the clade-membership check.
        store_data = df[[
            "split_id", "freq_1", "freq_2", "clade_size",
            "in_consensus_tree_1", "in_consensus_tree_2", "consensus_tree_membership",
        ]].to_dict("records")

        min_size = int(min_clade_size or 2)
        df_plot = df[df["clade_size"] >= min_size]

        fig = _build_scatter_fig(df_plot, label1, label2)
        # Success path: reveal the output paper.
        return (
            dcc.Graph(
                id="clade-freq-scatter",
                figure=fig,
                config={"displayModeBar": False},
                style={"width": "100%"},
            ),
            store_data,
            {},
        )

    @callback(
        Output("clade-freq-click-store", "data"),
        Input("clade-freq-scatter", "clickData"),
        prevent_initial_call=True,
    )
    def store_scatter_click(click_data):
        """Forward a scatter plot click to the click store.

        ``customdata`` is ``[split_id, clade_size]``. The split_id is an
        integer row index in the DataFrame produced by the compute
        callback; ``draw_tanglegram`` uses it together with the
        per-distmat canonical-keys cache to resolve the actual tip
        names to highlight.

        The ``_t`` nonce is set to a unique counter each time so
        ``dcc.Store`` does not deduplicate identical click payloads
        (e.g. clicking the same point twice). Without it, Plotly's
        first click on a point sometimes appears to "do nothing"
        because the store value matches the previous click.
        """
        nonlocal _click_counter
        if not click_data or not click_data.get("points"):
            return no_update
        point = click_data["points"][0]
        custom = point.get("customdata")
        if custom is None:
            return no_update
        try:
            _click_counter += 1
            return {
                "split_id":   int(custom[0]),
                "clade_size": int(custom[1]),
                "x":          float(point["x"]),
                "y":          float(point["y"]),
                "_t":         _click_counter,
            }
        except (TypeError, ValueError, IndexError, KeyError):
            return no_update

    @callback(
        Output("clade-freq-scatter", "figure", allow_duplicate=True),
        Input("clade-freq-click-store", "data"),
        prevent_initial_call=True,
    )
    def update_click_marker(click_data):
        """Patch only the overlay trace (index 1) on the scatter to
        place a hollow red circle around the clicked point.

        Returns a ``Patch`` so Plotly never redraws the 40k-point
        Scattergl trace — only the single-point overlay updates.
        Rebuilds via the Compare button reset this overlay back to
        empty, which is the right behaviour (a fresh comparison
        clears the previous click).
        """
        if not click_data:
            return no_update
        x = click_data.get("x")
        y = click_data.get("y")
        if x is None or y is None:
            return no_update
        patch = Patch()
        patch["data"][1]["x"] = [x]
        patch["data"][1]["y"] = [y]
        return patch

    @callback(
        Output("clade-freq-tanglegram", "figure"),
        Output("clade-freq-tanglegram-pair-store", "data"),
        Output("clade-freq-tanglegram-title", "children"),
        Input("clade-freq-click-store", "data"),
        State("clade-freq-consensus-tree-select-1", "value"),
        State("clade-freq-consensus-tree-select-2", "value"),
        State("tanglegram-yscale-slider", "value"),
        State("clade-freq-tanglegram-pair-store", "data"),
        State("tanglegram-complement-toggle", "checked"),
        prevent_initial_call=True,
    )
    def draw_tanglegram(click_data, uid1, uid2, px_per_tip, current_pair, complement_on):
        """Draw a tanglegram of the two consensus trees when a clade dot is clicked.

        Two render paths:

        * **First click on a new consensus tree pair** — build the full figure
          (11 traces: static skeleton for both trees + dynamic overlays
          for the MRCA subtree, the highlight, the connectors, the MRCA
          node markers, and the green complement tips/connectors).
          Returns a fresh figure dict and stamps the new pair into the
          tanglegram-pair-store.
        * **Subsequent clicks on the same pair** — return a
          ``dash.Patch`` that updates only the 7 dynamic traces'
          x/y/text and the title. The static skeleton (≈300 line
          segments + 280 tip markers per tree) is never re-sent.

        The clicked split is identified by an integer ``split_id``;
        the actual tip names are resolved server-side via
        ``_split_resolution`` (rebuilt by the Compare callback) and
        the per-distmat canonical-keys cache.
        """
        if not click_data or not uid1 or not uid2:
            return no_update, no_update, no_update

        # ── Resolve the click via _split_resolution ───────────────────────
        # Yields ``(src, column_j, split_key)`` where ``split_key`` is the
        # tuple of leaf indices that make up the rooted clade.
        split_id = click_data.get("split_id")
        if split_id is None:
            return no_update, no_update, no_update
        resolved = _split_resolution.get(int(split_id))
        if resolved is None:
            # Click store survived a Compare-button reset and we no
            # longer know which split this is. Drop the request
            # quietly; the next Compare repopulates _split_resolution.
            return no_update, no_update, no_update
        src, column_j, split_key = resolved

        try:
            canonical = state.get_canonical_keys(src)
        except (KeyError, FileNotFoundError):
            return no_update, no_update, no_update
        leaf_names = canonical["leaf_names"]

        # Single highlight (same on both trees): consensus tree 1 and consensus tree 2 are
        # both anchored to ``src`` (the Compare-clade dropdowns
        # filter to the active distmat), so a column in the rooted
        # presence table represents the same descendant set in both.
        highlight = {leaf_names[i] for i in split_key}

        # Containment is an O(1) ``column_j ∈ cols_in_consensus_tree`` check
        # straight off the registry entries.
        entry1 = state.get_consensus_tree_registry_entry(uid1)
        entry2 = state.get_consensus_tree_registry_entry(uid2)
        cols1 = set(entry1.get("cols_in_consensus_tree") or []) if entry1 else set()
        cols2 = set(entry2.get("cols_in_consensus_tree") or []) if entry2 else set()
        in_1 = column_j in cols1
        in_2 = column_j in cols2

        # ── Look up (or build) the static tanglegram layout ────────────────
        layout = _get_tanglegram_layout(uid1, uid2)
        if layout is None:
            # consensus tree NEXUS bytes evicted from cache; user must recompute.
            return no_update, no_update, no_update

        tips1 = layout["tips1"]
        tips2 = layout["tips2"]
        right_start = layout["right_start"]

        # Group labels for the sticky title above the graph.
        label1 = entry1["name"] if entry1 else "Group 1"
        label2 = entry2["name"] if entry2 else "Group 2"

        # ── Build the dynamic overlay traces + the sticky title ──────────
        hl_left  = _highlight_overlay_trace(tips1, highlight)
        hl_right = _highlight_overlay_trace(tips2, highlight)
        connectors = _connector_overlay_trace(tips1, tips2, highlight)
        mrca_subtree, mrca_marker, complement_per_tree = _build_mrca_traces(highlight, layout)
        complement1 = complement_per_tree["tips1"]
        complement2 = complement_per_tree["tips2"]
        title_children = _tanglegram_title_children(
            label1, label2, highlight, in_1=in_1, in_2=in_2,
        )

        # Build the green overlays fully styled in BOTH branches —
        # empty data when the toggle is off — so the styling is baked
        # into the figure once at first render and later patches only
        # ever touch x/y/text. The grey tip base (traces 3/4) is never
        # reduced: the green markers (size 8, opaque) sit on a higher
        # trace index and fully occlude the grey tips (size 6) beneath,
        # exactly like the red highlight overlay already does.
        if complement_on:
            comp_tips = _complement_tips_trace(complement1 | complement2, tips1, tips2)
            comp_conn = _complement_connector_trace(
                tips1, tips2, complement1, complement2,
            )
        else:
            comp_tips = _complement_tips_trace(set(), tips1, tips2)
            comp_conn = _complement_connector_trace(tips1, tips2, set(), set())

        same_pair = current_pair == [uid1, uid2]
        if same_pair:
            # Patch only the dynamic overlays' geometry. The static
            # skeleton (branches 0/1, grey tips 3/4) and the green
            # overlays' styling are already in the figure, so the patch
            # carries nothing but the x/y/text that actually changed.
            patch = Patch()
            patch["data"][_TANGLEGRAM_MRCA_SUBTREE]["x"] = mrca_subtree["x"]
            patch["data"][_TANGLEGRAM_MRCA_SUBTREE]["y"] = mrca_subtree["y"]
            for idx, trace in (
                (_TANGLEGRAM_HIGHLIGHT_LEFT,  hl_left),
                (_TANGLEGRAM_HIGHLIGHT_RIGHT, hl_right),
                (_TANGLEGRAM_CONNECTORS,      connectors),
                (_TANGLEGRAM_MRCA_MARKER,     mrca_marker),
                (_TANGLEGRAM_COMPLEMENT_TIPS, comp_tips),
                (_TANGLEGRAM_COMPLEMENT_CONN, comp_conn),
            ):
                patch["data"][idx]["x"] = trace["x"]
                patch["data"][idx]["y"] = trace["y"]
                patch["data"][idx]["text"] = trace["text"]
            return patch, no_update, title_children

        # ── First time this pair is rendered — build the full figure ─────
        # Trace order must match the _TANGLEGRAM_* index constants:
        # branches first (skeleton 0-1, MRCA subtree 2), then the tip
        # markers (grey 3-4, red 5-6) so the dots draw over the branch
        # colour, then connectors (7), MRCA markers (8), and finally the
        # green complement overlays (9-10) on top. skel[1]/skel[3] are
        # the full static grey-tip sets — the green markers simply draw
        # over them. skeleton_traces is [L branch, L tips, R branch, R tips].
        skel = layout["skeleton_traces"]
        traces = [
            skel[0], skel[2], mrca_subtree,
            skel[1], skel[3], hl_left, hl_right,
            connectors, mrca_marker,
            comp_tips, comp_conn,
        ]
        height = _tanglegram_height(px_per_tip, layout["max_y"])
        fig = go.Figure(data=[
            t if isinstance(t, go.Scatter) else go.Scatter(**t)
            for t in traces
        ])
        # Backbone (skeleton branch) colour follows the theme — an
        # off-white reads against the dark canvas in dark mode. Set it on
        # the built figure (traces 0/1) rather than the cached layout so
        # the lru-cached skeleton dicts stay theme-agnostic.
        backbone = _backbone_color()
        fig.data[0].line.color = backbone
        fig.data[1].line.color = backbone
        fig.update_layout(
            template=get_template(),
            height=height,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(visible=False,
                       range=[-0.05, right_start + 1.05]),
            yaxis=dict(visible=False),
            hovermode="closest",
        )
        return fig, [uid1, uid2], title_children

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
        resolves the last clicked split from ``_split_resolution`` and
        rebuilds them. Their styling was baked in at first render, so
        the patch never re-sends marker/line/mode.
        """
        # Need a rendered tanglegram — a prior click plus a live layout
        # — to patch against; otherwise there are no green traces yet.
        if not click_data or not uid1 or not uid2:
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

        split_id = click_data.get("split_id")
        if split_id is None:
            return no_update
        resolved = _split_resolution.get(int(split_id))
        if resolved is None:
            return no_update

        src, _column_j, split_key = resolved
        try:
            canonical = state.get_canonical_keys(src)
        except (KeyError, FileNotFoundError):
            return no_update
        leaf_names = canonical["leaf_names"]
        highlight = {leaf_names[i] for i in split_key}

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
