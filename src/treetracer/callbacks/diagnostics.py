from dash import dcc, html, callback, Input, Output, State, no_update, ctx, ALL, Patch
import dash_mantine_components as dmc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd

from ..logger import add_log, notif_id
from ..db.tree_service import get_tree_service
from ..ess.rf_trace import compute_rf_trace_data
from ..ess import compute_pseudo_ess
from .. import state
from ..theme import get_template
from ..clade_freq import compute_clade_frequencies


# Server-side resolution table for the Clade Frequency scatter →
# tanglegram round-trip. The scatter's ``customdata`` carries an
# integer ``split_id`` (the row index in the DataFrame produced by
# ``compute_clade_frequencies``); this dict maps that id to a
# ``(source_distmat, split_key)`` pair so the click handler can
# resolve back to actual tip names. Rebuilt on every Compare click.
#
# Keeping the keys server-side avoids serialising 40k frozensets of
# strings (or tuples of ints) through the browser store, and dodges
# the previous fragile ", ".join(sorted(s)) → ",".split() round-trip.
_split_resolution: dict[int, tuple[str, object]] = {}

from ..newick_layout import parse_nexus, build_tree_traces, build_connector_traces, _collect_nodes
from ._helpers import _save_file_dialog


# ---------------------------------------------------------------------------
# Tanglegram caches
# ---------------------------------------------------------------------------
# Re-parsing the MCC NEXUS bytes on every scatter click was the dominant
# cost of ``draw_tanglegram`` for big trees (~280 taxa = 100s of ms per
# click). These two caches plus a deterministic trace layout in the
# tanglegram figure let the callback Patch only the dynamic traces
# (highlight markers + connectors) when the MCC pair hasn't changed.

import functools


@functools.lru_cache(maxsize=64)
def _get_parsed_mcc(uid):
    """Parse the cached NEXUS for an MCC uuid into a laid-out Node tree.

    Cached so repeat clicks on a tanglegram don't re-parse the same
    NEXUS file. Keyed on uuid — when an MCC is dropped from the LRU
    cache its uuid is recycled, but since the cached_mcc_tree key
    space is random-tokens, false hits are impossibly rare.
    """
    nexus = state.get_cached_mcc_tree(uid)
    if nexus is None:
        return None
    root, _translate = parse_nexus(nexus)
    return root


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
                        right branches+all-tips)
    """
    root1 = _get_parsed_mcc(uid1)
    root2 = _get_parsed_mcc(uid2)
    if root1 is None or root2 is None:
        return None

    nodes1 = _collect_nodes(root1)
    nodes2 = _collect_nodes(root2)
    max_x1 = max(n.x for n in nodes1) or 1.0
    max_x2 = max(n.x for n in nodes2) or 1.0
    scale1 = 1.0 / max_x1
    scale2 = 1.0 / max_x2

    GAP = 0.3
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

    return {
        "root1": root1,
        "root2": root2,
        "scale1": scale1,
        "scale2": scale2,
        "right_start": right_start,
        "tips1": tips1,
        "tips2": tips2,
        "max_y": max_y,
        "skeleton_traces": list(left_skeleton) + list(right_skeleton),
    }


# Dynamic-trace indices in the assembled tanglegram figure. The static
# skeleton occupies the first 4 indices (2 per tree, lines + grey
# markers); indices 4, 5, 6 are the dynamic overlays the click
# callback patches.
#
#   0: left branches      (static)
#   1: left grey tips     (static)
#   2: right branches     (static)
#   3: right grey tips    (static)
#   4: left red highlight (dynamic)
#   5: right red highlight(dynamic)
#   6: red connectors     (dynamic)
_TANGLEGRAM_HIGHLIGHT_LEFT  = 4
_TANGLEGRAM_HIGHLIGHT_RIGHT = 5
_TANGLEGRAM_CONNECTORS      = 6


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
        line=dict(color="rgba(230,57,70,0.7)", width=2),
        text=[n for n in names for _ in range(3)],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )


def _tanglegram_title(label1, label2, highlight):
    return (
        f"<b>{label1}</b> ← "
        f"  clade: {len(highlight)} tips  "
        f"→ <b>{label2}</b>"
    )


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
                "text": "Select two MCC trees and click "
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
    LRU, and the click→split lookup table — they're keyed on MCC
    uuids that are about to disappear from ``state._mcc_cache``.
    """
    _get_parsed_mcc.cache_clear()
    _get_tanglegram_layout.cache_clear()
    _split_resolution.clear()


def _build_rf_trace_fig(trace_df, ref_group, ref_position, burnin=0):
    """Build the RF trace figure with optional burnin zoom."""
    all_groups = sorted(trace_df['group'].unique().tolist())
    colors = px.colors.qualitative.Dark24[:len(all_groups)]
    color_map = dict(zip(all_groups, colors))

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        column_widths=[0.8, 0.2],
        horizontal_spacing=0.02,
    )

    value_col = 'rf_distance'

    for group in all_groups:
        gdf = trace_df[trace_df['group'] == group]
        vals = gdf[value_col].values.astype(float)
        fig.add_trace(go.Scatter(
            x=gdf['treenum'], y=vals,
            mode='lines', name=group,
            line=dict(color=color_map[group], width=1),
            legendgroup=group,
        ), row=1, col=1)

        # KDE: use post-burnin values if burnin is set
        kde_vals = vals
        if burnin > 0:
            post = gdf[gdf['treenum'] > burnin][value_col].values.astype(float)
            if len(post) > 1:
                kde_vals = post

        if len(kde_vals) > 1 and np.std(kde_vals) > 0:
            kde = gaussian_kde(kde_vals)
            y_grid = np.linspace(kde_vals.min(), kde_vals.max(), 200)
            density = kde(y_grid)
            fig.add_trace(go.Scatter(
                x=density, y=y_grid, mode='lines',
                line=dict(color=color_map[group], width=1),
                fill='tozerox', opacity=0.3,
                legendgroup=group, showlegend=False,
            ), row=1, col=2)

    # Apply burnin zoom
    if burnin > 0:
        post_burnin = trace_df[trace_df['treenum'] > burnin]
        if len(post_burnin) > 0:
            max_treenum = int(trace_df['treenum'].max())
            fig.update_xaxes(range=[burnin, max_treenum], row=1, col=1)
            all_post = post_burnin[value_col].values.astype(float)
            ymin, ymax = all_post.min(), all_post.max()
            ypad = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
            fig.update_yaxes(range=[ymin - ypad, ymax + ypad], row=1, col=1)

    ref_label = f"{ref_position} tree of {ref_group}"
    fig.update_layout(
        template=get_template(),
        xaxis_title="Tree number",
        yaxis_title=f"RF distance to {ref_label}",
        xaxis2_title="Density",
        margin=dict(l=60, r=20, t=30, b=40),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

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
        marker=dict(
            size=8,
            color=df_plot["clade_size"],
            colorscale="Viridis",
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
        customdata=df_plot[["split_id", "clade_size"]].values.tolist(),
        hovertemplate=(
            "<b>Clade (%{customdata[1]} tips)</b><br>"
            "Group 1: %{x:.3f}<br>"
            "Group 2: %{y:.3f}"
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
        template="simple_white",
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

def register_diagnostics_callbacks():
    # Monotonic counter that gets stamped on every scatter-plot click
    # payload (see ``store_scatter_click`` below). Without a unique
    # value, ``dcc.Store`` deduplicates identical click data and the
    # downstream tanglegram callback doesn't fire — producing the
    # "first click does nothing, second click works" behaviour.
    _click_counter = 0

    @callback(
        Output("lnl-trace-plot", "children"),
        Output("export-lnl-trace-button", "disabled", allow_duplicate=True),
        Output("lnl-burnin-input", "value"),
        Output("rf-burnin-input", "value"),
        Input("tree-offset-store", "data"),
        Input("lnl-burnin-input", "value"),
        Input("diagnostics-distmat-select", "value"),
        Input("plotly-template-store", "data"),
        State("distmat-store", "data"),
        prevent_initial_call=True,
    )
    def update_lnl_trace(stored_summaries, burnin, selected_matrix,
                         _template, distmat_data):
        """Render log-posterior trace plot for the SELECTED RF matrix.

        The Diagnostics tab is unified around the header RF-matrix
        selector — every section, including this one, conditions on
        that pick. With no matrix selected the plot shows a hint and
        nothing else; once a matrix is chosen, traces are restricted
        to the *exact* set of trees that went into it (by tree-name
        match against the matrix's stored ``names`` list — see
        ``state.get_distmat_names``). This keeps the lnP plot in lock-
        step with the matrix's downsample.

        The ``plotly-template-store`` Input is unused inside the
        function body — it's only there so dark-mode toggles trigger a
        re-render, which then re-reads ``get_template()`` at figure
        build time.
        """
        if not stored_summaries:
            return dmc.Text(
                "No trees loaded yet.",
                c="dimmed", size="sm", style={"padding": "20px"},
            ), True, 0, 0

        if not selected_matrix or not distmat_data or selected_matrix not in distmat_data:
            return (
                dmc.Text(
                    "Pick an RF matrix above to see the log-posterior "
                    "trace for the trees that went into it.",
                    c="dimmed", size="sm", style={"padding": "20px"},
                ),
                True, no_update, no_update,
            )

        # When trees are loaded/changed, set burnin to 10% of max per-group tree count
        triggered = ctx.triggered_id
        if triggered == "tree-offset-store":
            max_trees = max(
                s.get("total_trees", 0) for s in stored_summaries.values()
            )
            default_burnin = max(0, int(max_trees * 0.1))
            burnin = default_burnin
            burnin_out = (default_burnin, default_burnin)
        else:
            burnin_out = (no_update, no_update)

        tree_service = get_tree_service()
        all_file_sources = list(stored_summaries.keys())
        try:
            tree_names = list(state.get_distmat_names(selected_matrix))
        except KeyError:
            return (
                dmc.Text(
                    f"Matrix {selected_matrix!r} is no longer available.",
                    c="red", size="sm", style={"padding": "20px"},
                ),
                True, no_update, no_update,
            )
        traces = tree_service.get_metadata_traces(
            file_sources=all_file_sources,
            tree_names=tree_names,
        )

        if not traces:
            return (dmc.Text(
                "No log-posterior data found in tree annotations.",
                c="dimmed", size="sm", style={"padding": "20px"},
            ), True, *burnin_out)

        # Pick first available field (prefer lnP > lnL > posterior > joint > loglikelihood)
        preferred = ['lnP', 'lnL', 'posterior', 'joint', 'loglikelihood']
        field_name = None
        for f in preferred:
            if f in traces:
                field_name = f
                break
        if field_name is None:
            field_name = next(iter(traces))

        trace_df = traces[field_name]
        groups = sorted(trace_df['group'].unique().tolist())
        colors = px.colors.qualitative.Dark24[:len(groups)]
        color_map = dict(zip(groups, colors))

        fig = make_subplots(
            rows=1, cols=2, shared_yaxes=True,
            column_widths=[0.8, 0.2],
            horizontal_spacing=0.02,
        )
        for group in groups:
            gdf = trace_df[trace_df['group'] == group]
            vals = gdf['value'].values
            fig.add_trace(go.Scatter(
                x=gdf['treenum'],
                y=vals,
                mode='lines',
                name=group,
                line=dict(color=color_map[group], width=1),
                legendgroup=group,
            ), row=1, col=1)
            if len(vals) > 1 and np.std(vals) > 0:
                kde = gaussian_kde(vals)
                y_grid = np.linspace(vals.min(), vals.max(), 200)
                density = kde(y_grid)
                fig.add_trace(go.Scatter(
                    x=density, y=y_grid,
                    mode='lines',
                    line=dict(color=color_map[group], width=1),
                    fill='tozerox',
                    opacity=0.3,
                    legendgroup=group,
                    showlegend=False,
                ), row=1, col=2)

        # Apply burnin: filter data for KDE and set x-axis range
        try:
            burnin = int(burnin) if burnin else 0
        except (ValueError, TypeError):
            burnin = 0
        if burnin > 0:
            post_burnin = trace_df[trace_df['treenum'] > burnin]
            if len(post_burnin) > 0:
                # Recompute KDE using only post-burnin values
                fig.data = []  # clear traces, rebuild with filtered KDE
                for group in groups:
                    gdf = trace_df[trace_df['group'] == group]
                    vals = gdf['value'].values
                    # Trace line: show all data (full range)
                    fig.add_trace(go.Scatter(
                        x=gdf['treenum'], y=vals,
                        mode='lines', name=group,
                        line=dict(color=color_map[group], width=1),
                        legendgroup=group,
                    ), row=1, col=1)
                    # KDE: only post-burnin
                    post_vals = gdf[gdf['treenum'] > burnin]['value'].values
                    if len(post_vals) > 1 and np.std(post_vals) > 0:
                        kde = gaussian_kde(post_vals)
                        y_grid = np.linspace(post_vals.min(), post_vals.max(), 200)
                        density = kde(y_grid)
                        fig.add_trace(go.Scatter(
                            x=density, y=y_grid, mode='lines',
                            line=dict(color=color_map[group], width=1),
                            fill='tozerox', opacity=0.3,
                            legendgroup=group, showlegend=False,
                        ), row=1, col=2)

                # Zoom x-axis to post-burnin range
                max_treenum = int(trace_df['treenum'].max())
                fig.update_xaxes(range=[burnin, max_treenum], row=1, col=1)

                # Zoom y-axis to post-burnin value range with 5% padding
                all_post = post_burnin['value'].values
                ymin, ymax = all_post.min(), all_post.max()
                ypad = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
                fig.update_yaxes(range=[ymin - ypad, ymax + ypad], row=1, col=1)

        fig.update_layout(
            template=get_template(),
            xaxis_title="Tree number",
            yaxis_title=field_name,
            xaxis2_title="Density",
            margin=dict(l=60, r=20, t=30, b=40),
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return (dcc.Graph(id="lnl-trace-graph", figure=fig, config={"displayModeBar": False}),
                False, *burnin_out)

    @callback(
        Output("rf-reference-group-select", "data", allow_duplicate=True),
        Output("rf-reference-group-select", "value", allow_duplicate=True),
        Input("tree-offset-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_rf_trace_controls(stored_summaries):
        """Populate group dropdown when trees are loaded."""
        if not stored_summaries:
            return [], None

        # Collect all groups across all loaded files
        all_groups = []
        for summary in stored_summaries.values():
            all_groups.extend(summary.get("groups", []))
        all_groups = sorted(set(all_groups))

        group_options = [{"value": g, "label": g} for g in all_groups]
        # Default to last group alphabetically
        default_group = all_groups[-1] if all_groups else None
        return group_options, default_group

    @callback(
        Output("rf-reference-group-select", "data", allow_duplicate=True),
        Output("rf-reference-group-select", "value", allow_duplicate=True),
        Input("diagnostics-distmat-select", "value"),
        State("distmat-store", "data"),
        prevent_initial_call=True,
    )
    def populate_groups_from_distmat(selected_matrix, stored_distmats):
        """Populate the reference-group dropdown from the SELECTED RF matrix.

        Driven by the shared header selector so the RF-trace section
        only offers groups that actually live in the matrix the user is
        currently inspecting.
        """
        if not stored_distmats or not selected_matrix:
            return no_update, no_update

        compact = stored_distmats.get(selected_matrix)
        if not compact:
            return no_update, no_update

        groups_per_file = compact.get("groups_per_file", {})
        all_groups = sorted(set(g for groups in groups_per_file.values() for g in groups))
        if not all_groups:
            # Fallback: use file_breakdown keys
            all_groups = sorted(compact.get("file_breakdown", {}).keys())
        if not all_groups:
            return no_update, no_update

        group_options = [{"value": g, "label": g} for g in all_groups]
        default_group = all_groups[-1] if all_groups else None
        return group_options, default_group

    @callback(
        Output("rf-trace-plot", "children"),
        Output("rf-trace-store", "data"),
        Output("notifications-container", "children", allow_duplicate=True),
        Output("export-rf-trace-button", "disabled", allow_duplicate=True),
        Input("compute-rf-trace-button", "n_clicks"),
        State("tree-offset-store", "data"),
        State("rf-reference-group-select", "value"),
        State("rf-reference-position-select", "value"),
        State("distmat-store", "data"),
        State("diagnostics-distmat-select", "value"),
        State("rf-burnin-input", "value"),
        prevent_initial_call=True,
    )
    def compute_rf_trace(n_clicks, stored_summaries, ref_group, ref_position,
                         stored_distmats, selected_matrix, burnin):
        """Compute RF distance of every tree to a single shared reference tree using pre-computed distance matrix."""
        if not n_clicks:
            return no_update, no_update, no_update, no_update

        if not ref_group:
            return (
                dmc.Text("Please select a reference group.", c="red"),
                no_update,
                no_update,
                no_update,
            )

        if not stored_distmats:
            return (
                dmc.Text("Please compute RF distances first (Distances tab).", c="red"),
                no_update, no_update, no_update,
            )

        if not selected_matrix or selected_matrix not in stored_distmats:
            return (
                dmc.Text("Please pick an RF matrix at the top of the page.", c="red"),
                no_update, no_update, no_update,
            )

        result, ref_name = compute_rf_trace_data(selected_matrix, ref_group, ref_position)

        # If result is a string, it's an error message
        if isinstance(result, str):
            return dmc.Text(result, c="red"), no_update, no_update, no_update

        trace_df = result

        try:
            burnin = int(burnin) if burnin else 0
        except (ValueError, TypeError):
            burnin = 0
        fig = _build_rf_trace_fig(trace_df, ref_group, ref_position, burnin=burnin)

        notification = dmc.Notification(
            title="RF Trace Computed",
            message=f"Computed RF distances for {len(trace_df)} trees to {ref_position} tree of {ref_group}.",
            color="green",
            action="show",
            autoClose=3000,
            id=notif_id(),
        )

        store_data = trace_df.to_dict("records")

        return (
            dcc.Graph(id="rf-trace-graph", figure=fig, config={"displayModeBar": False}),
            store_data,
            notification,
            False,
        )

    # Re-render RF trace plot when burnin changes
    @callback(
        Output("rf-trace-plot", "children", allow_duplicate=True),
        Input("rf-burnin-input", "value"),
        Input("plotly-template-store", "data"),
        State("rf-trace-store", "data"),
        State("rf-reference-group-select", "value"),
        State("rf-reference-position-select", "value"),
        prevent_initial_call=True,
    )
    def update_rf_trace_burnin(burnin, _, store_data, ref_group, ref_position):
        if not store_data:
            return no_update
        try:
            burnin = int(burnin) if burnin else 0
        except (ValueError, TypeError):
            burnin = 0

        trace_df = pd.DataFrame(store_data)
        fig = _build_rf_trace_fig(trace_df, ref_group or "", ref_position or "last", burnin)
        return dcc.Graph(id="rf-trace-graph", figure=fig, config={"displayModeBar": False})

    # ------ EXPORT DIAGNOSTICS PLOTS AS PDF ------

    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("export-lnl-trace-button", "n_clicks"),
        State("lnl-trace-graph", "figure"),
        prevent_initial_call=True,
    )
    def export_lnl_trace_pdf(n_clicks, fig_dict):
        if not n_clicks or not fig_dict:
            return no_update
        path = _save_file_dialog(default_filename="lnP_trace.pdf")
        if not path:
            return no_update
        fig = go.Figure(fig_dict)
        fig.update_layout(template=get_template())
        fig.write_image(path, width=1200, height=400, scale=2)
        add_log(f"Exported lnP trace plot to {path}")
        return dmc.Notification(
            title="lnP Trace Exported",
            message=f"Saved to {path}",
            color="green",
            action="show",
            autoClose=3000,
            id=notif_id(),
        )

    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("export-rf-trace-button", "n_clicks"),
        State("rf-trace-graph", "figure"),
        prevent_initial_call=True,
    )
    def export_rf_trace_pdf(n_clicks, fig_dict):
        if not n_clicks or not fig_dict:
            return no_update
        path = _save_file_dialog(default_filename="rf_trace.pdf")
        if not path:
            return no_update
        fig = go.Figure(fig_dict)
        fig.update_layout(template=get_template())
        fig.write_image(path, width=1200, height=400, scale=2)
        add_log(f"Exported RF trace plot to {path}")
        return dmc.Notification(
            title="RF Trace Exported",
            message=f"Saved to {path}",
            color="green",
            action="show",
            autoClose=3000,
            id=notif_id(),
        )

    # ─── Shared Diagnostics RF Matrix selector ─────────────────────────
    # One dropdown at the top of the tab feeds every section below.
    # `update_lnl_trace`, the RF-trace callbacks, and the Pseudo-ESS
    # callbacks all read this id.
    @callback(
        Output("diagnostics-distmat-select", "data"),
        Output("diagnostics-distmat-select", "value"),
        Output("diagnostics-distmat-info", "children"),
        Input("distmat-store", "data"),
        State("diagnostics-distmat-select", "value"),
    )
    def populate_diagnostics_distmat_select(distmat_data, current_value):
        if not distmat_data:
            return [], None, ""
        options = [
            {"value": k,
             "label": f"{k} ({v.get('n_trees', '?')} trees)"}
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

    # ─── MCC trees registered for the selected matrix ──────────────────
    # Renders a panel at the bottom of the Diagnostics tab listing the
    # MCC trees registered against the currently-selected RF matrix,
    # split into "Between-runs" and "Within-run" sub-tables. Hidden
    # when no MCCs match the active matrix. The Tab itself decides
    # which subset is interesting; here we surface both since the user
    # is viewing the matrix as a whole. Follow-up PRs can wire
    # post-processing actions onto a clicked row.
    @callback(
        Output("diagnostics-mcc-list", "children"),
        Output("diagnostics-mcc-paper", "style"),
        Input("mcc-registry-store", "data"),
        Input("diagnostics-distmat-select", "value"),
    )
    def render_diagnostics_mcc_panel(registry, selected_matrix):
        from .mcc_list import _table_for
        if not registry or not selected_matrix:
            return html.Div(), {"display": "none"}
        matched = [e for e in registry
                   if e.get("source_distmat") == selected_matrix]
        if not matched:
            return html.Div(), {"display": "none"}
        return dmc.Stack([
            dmc.Title(f"MCC trees for {selected_matrix}", order=5),
            _table_for(matched, show_mode=True),
        ], gap="sm"), {}

    # ─── Pseudo-ESS section ────────────────────────────────────────────
    # Two callbacks own the per-run table + Compute button. Both react
    # to ``diagnostics-distmat-select`` (the shared header dropdown).

    @callback(
        Output("ess-runs-table", "children"),
        Input("diagnostics-distmat-select", "value"),
    )
    def render_ess_runs_table(selected_matrix):
        if not selected_matrix:
            return html.Div(
                dmc.Text(
                    "Pick an RF matrix above to list its runs.",
                    c="dimmed", size="sm",
                ),
                style={"padding": "10px"},
            )

        # Server-side helper walks the matrix's tree-name list and
        # buckets by group prefix. Returns [(group, count), ...].
        groups = state.get_distmat_groups_with_counts(selected_matrix)
        if not groups:
            return dmc.Text(
                f"Matrix {selected_matrix!r} has no recognisable runs.",
                c="dimmed", size="sm",
            )

        rows = []
        for group, count in groups:
            rows.append(
                dmc.TableTr([
                    dmc.TableTd(group),
                    dmc.TableTd(str(count)),
                    dmc.TableTd(
                        dmc.Checkbox(
                            id={"type": "ess-run-checkbox", "index": group},
                            checked=True,
                        )
                    ),
                ])
            )

        return dmc.Table(
            [
                dmc.TableThead(
                    dmc.TableTr([
                        dmc.TableTh("Run"),
                        dmc.TableTh("Trees"),
                        dmc.TableTh("Select"),
                    ])
                ),
                dmc.TableTbody(rows),
            ],
            striped=True,
            highlightOnHover=True,
            withTableBorder=True,
            withColumnBorders=True,
        )

    @callback(
        Output("compute-pseudo-ess-button", "disabled"),
        Input("diagnostics-distmat-select", "value"),
        Input({"type": "ess-run-checkbox", "index": ALL}, "checked"),
    )
    def toggle_compute_pseudo_ess_button(selected_matrix, checks):
        # Disabled until a matrix is selected AND at least one run is checked.
        if not selected_matrix:
            return True
        if not checks or not any(checks):
            return True
        return False

    @callback(
        Output("pseudo-ess-output", "children"),
        Input("compute-pseudo-ess-button", "n_clicks"),
        State("diagnostics-distmat-select", "value"),
        State("ess-n-refs-input", "value"),
        State("ess-burnin-input", "value"),
        State({"type": "ess-run-checkbox", "index": ALL}, "checked"),
        State({"type": "ess-run-checkbox", "index": ALL}, "id"),
        prevent_initial_call=True,
    )
    def compute_pseudo_ess_for_runs(n_clicks, selected_matrix, n_refs, burnin, checks, ids):
        """Compute Pseudo-ESS per ticked run + an optional Combined row.

        For each ticked run we slice the cached RF distance matrix to the
        rows/cols of trees whose name has the ``"<run>/"`` prefix
        (dropping the first ``burnin`` of those rows), then feed that
        submatrix to :func:`treetracer.ess.compute_pseudo_ess`. Burn-in
        is applied *per run* so chains of different lengths don't get
        clipped against a global tree-index threshold. If more than one
        run is ticked we also compute the same diagnostic on the union
        of their (post-burn-in) tree indices — the "Combined" row.
        """
        if not n_clicks or not selected_matrix:
            return no_update

        ticked = [i["index"] for i, c in zip(ids, checks) if c]
        if not ticked:
            return dmc.Text(
                "No runs selected.", c="dimmed", size="sm",
            )

        try:
            names, distmat = state.load_distmat(selected_matrix)
        except KeyError:
            return dmc.Text(
                f"Matrix {selected_matrix!r} is no longer available.",
                c="red", size="sm",
            )

        # Bucket row indices by group prefix once (matrix-row order
        # matches MCMC iteration order within each chain).
        group_to_indices = {}
        for i, tree_name in enumerate(names):
            grp = str(tree_name).split("/", 1)[0]
            group_to_indices.setdefault(grp, []).append(i)

        try:
            n_refs_int = int(n_refs) if n_refs else 100
        except (ValueError, TypeError):
            n_refs_int = 100

        try:
            burnin_int = max(0, int(burnin)) if burnin else 0
        except (ValueError, TypeError):
            burnin_int = 0

        # Stoplight thresholds match the Lanfear paper's rough rule of
        # thumb: <100 is unreliable, <200 is borderline, ≥200 is the
        # "you can trust this" zone.
        def _ess_cell(v):
            if np.isnan(v):
                return dmc.TableTd("—")
            if v < 100:
                color = "red"
            elif v < 200:
                color = "orange"
            else:
                color = "green"
            return dmc.TableTd(
                dmc.Text(f"{v:.1f}", c=color, fw=600, span=True)
            )

        def _row_for(label, indices, burnin_label):
            sub = distmat[np.ix_(indices, indices)]
            res = compute_pseudo_ess(sub, n_refs=n_refs_int, seed=0)
            valid = res["ess_values"][~np.isnan(res["ess_values"])]
            if valid.size:
                mn = float(valid.min())
                q1, q2, q3 = np.quantile(valid, [0.25, 0.5, 0.75])
                mx = float(valid.max())
            else:
                mn = q1 = q2 = q3 = mx = float("nan")

            return dmc.TableTr([
                dmc.TableTd(label),
                dmc.TableTd(str(len(indices))),
                dmc.TableTd(burnin_label),
                _ess_cell(mn),
                _ess_cell(q1),
                _ess_cell(q2),
                _ess_cell(q3),
                _ess_cell(mx),
                dmc.TableTd(str(res["n_refs_used"])),
            ])

        rows = []
        all_indices = []
        skipped = []
        for grp in ticked:
            idx = group_to_indices.get(grp, [])
            # Burn-in is per chain — first ``burnin_int`` trees of THIS run.
            idx = idx[burnin_int:]
            if len(idx) < 4:
                skipped.append(grp)
                continue
            rows.append(_row_for(grp, idx, str(burnin_int)))
            all_indices.extend(idx)

        if len(ticked) - len(skipped) > 1 and all_indices:
            # Sort to preserve MCMC order across the union — important so
            # the autocorrelation in each reference's RF trace is meaningful.
            combined_idx = sorted(set(all_indices))
            # Per-run burn-in was already applied before union, so the
            # Combined label reads "Nx<burnin>" to make clear it isn't a
            # single global cut.
            rows.append(_row_for(
                "Combined", combined_idx,
                f"{len(ticked) - len(skipped)}×{burnin_int}",
            ))

        if not rows:
            msg = "Burn-in leaves fewer than 4 trees per run; nothing to compute."
            return dmc.Text(msg, c="dimmed", size="sm")

        return dmc.Table(
            [
                dmc.TableThead(
                    dmc.TableTr([
                        dmc.TableTh("Run"),
                        dmc.TableTh("Trees"),
                        dmc.TableTh("Burn-in"),
                        dmc.TableTh("Min"),
                        dmc.TableTh("Q1"),
                        dmc.TableTh("Q2 (median)"),
                        dmc.TableTh("Q3"),
                        dmc.TableTh("Max"),
                        dmc.TableTh("# refs"),
                    ])
                ),
                dmc.TableTbody(rows),
            ],
            striped=True,
            highlightOnHover=True,
            withTableBorder=True,
            withColumnBorders=True,
        )


    # ------ Clade Frequency Comparison: populate dropdowns ------
 
    @callback(
        Output("clade-freq-mcc-select-1", "data"),
        Output("clade-freq-mcc-select-1", "disabled"),
        Output("clade-freq-mcc-select-2", "data"),
        Output("clade-freq-mcc-select-2", "disabled"),
        Input("mcc-registry-store", "data"),
    )
    def populate_mcc_selects(registry):
        """Rebuild the MCC-tree dropdown options whenever a new MCC is saved.

        The registry list has the shape returned by get_mcc_registry():
            [{"uuid": str, "name": str, "n_trees": int, ...}, ...]

        Each option's value is the MCC's uuid (used to retrieve the cached
        NEXUS bytes and registry entry). The label shows the structured name
        and tree count so the user can see which selection each MCC summarises.
        """
        if not registry:
            return [], True, [], True

        options = [
            {
                "value": e["uuid"],
                "label": f"{e['name']}  ({e['n_trees']} trees)",
            }
            for e in registry
        ]
        return options, False, options, False
 
    # ------ Clade Frequency Comparison: enable Compare button ------
    
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
            df_plot[["split_id", "clade_size"]].values.tolist()
        )
        patch["data"][0]["marker"]["color"] = df_plot["clade_size"].tolist()
        return patch

    @callback(
        Output("clade-freq-compare-button", "disabled"),
        Input("clade-freq-mcc-select-1", "value"),
        Input("clade-freq-mcc-select-2", "value"),
    )
    def toggle_compare_button(uid1, uid2):
        """Enable the Compare button only when both dropdowns have a selection."""
        return not (uid1 and uid2)

    # ------ Clade Frequency Comparison: compute and plot ------

    @callback(
        Output("clade-freq-plot", "children"),
        Output("clade-freq-data-store", "data"),
        # Toggle the output Paper visible only on success; stays
        # hidden on any error path or before the first successful
        # Compare. Cleared by the sidebar's Clear-data flow.
        Output("clade-freq-output-paper", "style"),
        Input("clade-freq-compare-button", "n_clicks"),
        State("clade-freq-mcc-select-1", "value"),
        State("clade-freq-mcc-select-2", "value"),
        State("clade-freq-min-clade-size", "value"),
        prevent_initial_call=True,
    )
    def compute_and_plot_clade_frequencies(n_clicks, uid1, uid2, min_clade_size):
        """Compute clade frequencies for the two selected MCC groups and
        render a scatter plot (freq group 1 vs freq group 2).

        Each dot is one bipartition observed in either group. Dot colour
        encodes clade_size (number of tips in the canonical side).
        Clicking a dot triggers the tanglegram callback.
        """
        if not uid1 or not uid2:
            return no_update, no_update, no_update

        entry1 = state.get_mcc_registry_entry(uid1)
        entry2 = state.get_mcc_registry_entry(uid2)

        if entry1 is None or entry2 is None:
            return dmc.Text(
                "One or both selected MCC trees are no longer available. "
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

        # Integer row id replaces the old fragile comma-joined string.
        # The click-handler + tanglegram callbacks resolve split_id to
        # tip names via state.get_canonical_keys at render time.
        df["split_id"] = np.arange(len(df), dtype=np.int32)

        # Refresh the click-resolution table: split_id → (distmat,
        # split_key). split_key is tuple[int] for the fast same-distmat
        # case (resolved via canonical_keys["leaf_names"]) or
        # frozenset[str] for the cross-distmat fallback (already names).
        _split_resolution.clear()
        src1 = entry1["source_distmat"]
        for split_id, key in zip(df["split_id"].tolist(), df["split_key"].tolist()):
            _split_resolution[int(split_id)] = (src1, key)

        label1 = entry1["name"]
        label2 = entry2["name"]

        # Serialise for the slider callback. split_key is a tuple[int]
        # (or, in the rare cross-distmat fallback, a frozenset[str]) —
        # neither is JSON-serialisable, so it stays server-side and we
        # only ship the integer id through the browser.
        store_data = df[["split_id", "freq_1", "freq_2", "clade_size"]].to_dict("records")

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
        Input("clade-freq-click-store", "data"),
        State("clade-freq-mcc-select-1", "value"),
        State("clade-freq-mcc-select-2", "value"),
        State("tanglegram-yscale-slider", "value"),
        State("clade-freq-tanglegram-pair-store", "data"),
        prevent_initial_call=True,
    )
    def draw_tanglegram(click_data, uid1, uid2, px_per_tip, current_pair):
        """Draw a tanglegram of the two MCC trees when a clade dot is clicked.

        Two render paths:

        * **First click on a new MCC pair** — build the full figure
          (7 traces: static skeleton for both trees + dynamic overlays
          for the highlight and connectors). Returns a fresh figure
          dict and stamps the new pair into the tanglegram-pair-store.
        * **Subsequent clicks on the same pair** — return a
          ``dash.Patch`` that updates only the 3 dynamic traces and
          the title annotation. The static skeleton (≈300 line
          segments + 280 tip markers per tree) is never re-sent.

        The clicked split is identified by an integer ``split_id``;
        the actual tip names are resolved server-side via
        ``_split_resolution`` (rebuilt by the Compare callback) and
        the per-distmat canonical-keys cache.
        """
        if not click_data or not uid1 or not uid2:
            return no_update, no_update

        # ── Resolve the highlight tip-name set ─────────────────────────────
        split_id = click_data.get("split_id")
        if split_id is None:
            return no_update, no_update
        resolved = _split_resolution.get(int(split_id))
        if resolved is None:
            # Click store survived a Compare-button reset and we no
            # longer know which split this is. Drop the request
            # quietly; the next Compare repopulates _split_resolution.
            return no_update, no_update
        src, split_key = resolved
        if isinstance(split_key, frozenset):
            highlight = set(split_key)
        else:
            try:
                canonical = state.get_canonical_keys(src)
            except (KeyError, FileNotFoundError):
                return no_update, no_update
            leaf_names = canonical["leaf_names"]
            highlight = {leaf_names[i] for i in split_key}

        # ── Look up (or build) the static tanglegram layout ────────────────
        layout = _get_tanglegram_layout(uid1, uid2)
        if layout is None:
            # MCC NEXUS bytes evicted from cache; user must recompute.
            return no_update, no_update

        tips1 = layout["tips1"]
        tips2 = layout["tips2"]
        right_start = layout["right_start"]

        # Group labels for the title annotation.
        entry1 = state.get_mcc_registry_entry(uid1)
        entry2 = state.get_mcc_registry_entry(uid2)
        label1 = entry1["name"] if entry1 else "Group 1"
        label2 = entry2["name"] if entry2 else "Group 2"

        # ── Build the three dynamic traces (highlight + connectors) ───────
        hl_left  = _highlight_overlay_trace(tips1, highlight)
        hl_right = _highlight_overlay_trace(tips2, highlight)
        connectors = _connector_overlay_trace(tips1, tips2, highlight)
        title = _tanglegram_title(label1, label2, highlight)

        same_pair = current_pair == [uid1, uid2]
        if same_pair:
            # ── Patch-only update — never re-sends the static skeleton ───
            patch = Patch()
            for idx, trace in (
                (_TANGLEGRAM_HIGHLIGHT_LEFT,  hl_left),
                (_TANGLEGRAM_HIGHLIGHT_RIGHT, hl_right),
                (_TANGLEGRAM_CONNECTORS,      connectors),
            ):
                patch["data"][idx]["x"] = trace["x"]
                patch["data"][idx]["y"] = trace["y"]
                patch["data"][idx]["text"] = trace["text"]
            patch["layout"]["annotations"][0]["text"] = title
            return patch, no_update

        # ── First time this pair is rendered — build the full figure ─────
        traces = list(layout["skeleton_traces"]) + [hl_left, hl_right, connectors]
        px_per_tip = px_per_tip or 1
        height = max(300, int(layout["max_y"] * px_per_tip) + 60)

        fig = go.Figure(data=[
            t if isinstance(t, go.Scatter) else go.Scatter(**t)
            for t in traces
        ])
        fig.update_layout(
            template="simple_white",
            height=height,
            margin=dict(l=10, r=10, t=40, b=10),
            # Content lives in [0, right_start + 1.0] (left tree
            # 0–1, gap 1–1.3, right tree 1.3–2.3). Use a tiny equal
            # padding on both sides so the two trees stay centred
            # in the panel — the previous ``-1.05`` left edge was
            # asymmetric and shoved the tanglegram visibly right.
            xaxis=dict(visible=False,
                       range=[-0.05, right_start + 1.05]),
            yaxis=dict(visible=False),
            hovermode="closest",
            annotations=[
                dict(
                    x=0.5, y=1.02, xref="paper", yref="paper",
                    text=title,
                    showarrow=False,
                    font=dict(size=12),
                    xanchor="center",
                ),
            ],
        )
        return fig, [uid1, uid2]

    @callback(
        Output("clade-freq-tanglegram", "figure", allow_duplicate=True),
        Input("tanglegram-yscale-slider", "value"),
        State("clade-freq-mcc-select-1", "value"),
        State("clade-freq-mcc-select-2", "value"),
        prevent_initial_call=True,
    )
    def update_tanglegram_height(px_per_tip, uid1, uid2):
        """Slide-to-resize. The tanglegram's height scales with the
        number of tips × the slider value. Patches only
        ``layout.height`` so the 7-trace figure doesn't get rebuilt
        on every slider drag tick.

        No-ops when no MCC pair is selected yet (slider has nothing
        to resize against) or when the layout cache is cold (no
        click has rendered the tanglegram yet).
        """
        if not uid1 or not uid2:
            return no_update
        layout = _get_tanglegram_layout(uid1, uid2)
        if layout is None:
            return no_update
        px_per_tip = px_per_tip or 1
        patch = Patch()
        patch["layout"]["height"] = max(300, int(layout["max_y"] * px_per_tip) + 60)
        return patch

