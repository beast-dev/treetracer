import re

from dash import html, callback, clientside_callback, Input, Output, Patch, State, no_update
import dash_mantine_components as dmc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from ..logger import add_log
from ..state import get_mds_result
from ..theme import get_template
from ..plot_utils import (
    make_plot_grid, add_trace_multiplot_interleaved, placeholder_fig,
    retheme_figure,
)
from ..mcc._canonical_remap import (
    _substitute_newick_labels,
    _build_canonical_remaps,
)


# Matches the `tree NAME` token at the start of a NEXUS tree line.
_TREE_NAME_RE = re.compile(r'^(\s*tree\s+)([^\s=]+)', re.IGNORECASE)


def _sanitize_tree_name_token(text):
    """Make a string safe to use as an unquoted NEXUS tree name."""
    return re.sub(r'[^A-Za-z0-9_.]', '_', text)


def _rewrite_tree_line(line, new_name, label_remap):
    """Take a raw NEXUS ``tree <name> = <newick>;`` line and emit it with
    the name replaced and any integer labels remapped via ``label_remap``
    (source int → canonical int). Pass an empty dict to leave labels alone.
    """
    renamed = _TREE_NAME_RE.sub(
        lambda m: f'{m.group(1)}{new_name}',
        line,
        count=1,
    )
    return _substitute_newick_labels(renamed, label_remap)


# Trace-layout invariants set by ``add_trace_multiplot_interleaved``:
# the last 8 traces are 2 trailing overlay bundles of 4 each, in this
# exact order:
#   -8 … -5 → selection overlays (3D + xy + xz + yz)
#   -4 … -1 → MCC overlays       (3D + xy + xz + yz)
SELECTION_OVERLAY_OFFSETS = (-8, -7, -6, -5)
MCC_OVERLAY_OFFSETS = (-4, -3, -2, -1)
N_SELECTION_OVERLAYS = 4
N_MCC_OVERLAYS = 4
N_TRAILING_OVERLAYS = N_SELECTION_OVERLAYS + N_MCC_OVERLAYS


def _panels_2d(x, y, z):
    """Return [(xcol, ycol), ...] for the 3 right-column 2D panels in order:
    (x,y), (x,z), (y,z). Same order add_trace_multiplot_interleaved uses,
    which is what makes the trailing overlay offsets meaningful."""
    return [(x, y), (x, z), (y, z)]


def _overlay_panels_data(df, selected_pairs, x, y, z):
    """Build (xs, ys, customdata) for each of the 4 selection overlays.

    Returns a 4-tuple in trace order:
        index 0 → 3D overlay  (x, y, z arrays + customdata)
        index 1 → 2D x-y overlay  (x, y arrays + customdata)
        index 2 → 2D x-z overlay
        index 3 → 2D y-z overlay

    When nothing is selected the arrays are empty so the trace stays present
    but draws nothing.
    """
    empty_3d = {"x": [], "y": [], "z": [], "customdata": []}
    empty_2d = {"x": [], "y": [], "customdata": []}
    if not selected_pairs:
        return [dict(empty_3d), dict(empty_2d), dict(empty_2d), dict(empty_2d)]

    selected_set = {(g, int(t)) for g, t in selected_pairs}
    keys = list(zip(df["group"], df["treenum"].astype(int)))
    mask = pd.Series([k in selected_set for k in keys], index=df.index)
    sel = df[mask]
    if len(sel) == 0:
        return [dict(empty_3d), dict(empty_2d), dict(empty_2d), dict(empty_2d)]

    labels = sel["tree"].str.split("/").str[-1].str.strip().tolist()
    treenums = sel["treenum"].astype(int).tolist()
    groups = sel["group"].tolist()
    cd = list(zip(treenums, labels, groups))

    return [
        {"x": sel[x].tolist(), "y": sel[y].tolist(),
         "z": sel[z].tolist(), "customdata": cd},
        {"x": sel[x].tolist(), "y": sel[y].tolist(), "customdata": cd},
        {"x": sel[x].tolist(), "y": sel[z].tolist(), "customdata": cd},
        {"x": sel[y].tolist(), "y": sel[z].tolist(), "customdata": cd},
    ]


def _mcc_overlay_panels_data(df, registry, source_distmat, x, y, z):
    """Same shape as ``_overlay_panels_data`` but pulled from the MCC
    registry, filtered to **Between**-mode entries belonging to
    *source_distmat*.

    Within-mode MCCs are deliberately excluded so MCCs computed on the
    within-run tab don't leak as green rings onto the between-runs plot
    (and vice-versa — the within-run side does its own filtering).

    Each entry contributes one point at ``mcc_tree.(group, treenum)``,
    looked up in *df*. ``customdata[1]`` carries the MCC's registered
    name so the hover reads "Tree #N: <RF_001_Between_MCC_2>".
    """
    pairs = []
    name_by_pair = {}
    for e in registry:
        if e.get("source_distmat") != source_distmat:
            continue
        if e.get("mode") != "Between":
            continue
        mt = e.get("mcc_tree") or {}
        g, t = mt.get("group"), mt.get("treenum")
        if g is None or t is None:
            continue
        pair = (g, int(t))
        pairs.append(pair)
        name_by_pair[pair] = e.get("name", "")

    empty_3d = {"x": [], "y": [], "z": [], "customdata": []}
    empty_2d = {"x": [], "y": [], "customdata": []}
    if not pairs:
        return [dict(empty_3d), dict(empty_2d), dict(empty_2d), dict(empty_2d)]

    pair_set = set(pairs)
    keys = list(zip(df["group"], df["treenum"].astype(int)))
    mask = pd.Series([k in pair_set for k in keys], index=df.index)
    sel = df[mask]
    if len(sel) == 0:
        return [dict(empty_3d), dict(empty_2d), dict(empty_2d), dict(empty_2d)]

    treenums = sel["treenum"].astype(int).tolist()
    groups = sel["group"].tolist()
    labels = [name_by_pair.get((g, t), "") for g, t in zip(groups, treenums)]
    cd = list(zip(treenums, labels, groups))

    return [
        {"x": sel[x].tolist(), "y": sel[y].tolist(),
         "z": sel[z].tolist(), "customdata": cd},
        {"x": sel[x].tolist(), "y": sel[y].tolist(), "customdata": cd},
        {"x": sel[x].tolist(), "y": sel[z].tolist(), "customdata": cd},
        {"x": sel[y].tolist(), "y": sel[z].tolist(), "customdata": cd},
    ]


def _resolve_source_distmat(selected_key, mds_results):
    """Pull ``source_distmat`` out of the active MDS result metadata."""
    if not selected_key or not mds_results or selected_key not in mds_results:
        return None
    return (mds_results[selected_key] or {}).get("source_distmat")


def _stamp_overlay_bundle(fig, panel_data, offsets):
    """Write a 4-tuple of panel-data dicts (3D + xy + xz + yz) into the
    overlay traces at *offsets* (negative indices into ``fig.data``)."""
    if not panel_data:
        return
    d3, dxy, dxz, dyz = panel_data
    n = len(fig.data)
    o3d, oxy, oxz, oyz = offsets
    fig.data[n + o3d].x = d3["x"]
    fig.data[n + o3d].y = d3["y"]
    fig.data[n + o3d].z = d3["z"]
    fig.data[n + o3d].customdata = d3["customdata"]
    for offset, d in zip((oxy, oxz, oyz), (dxy, dxz, dyz)):
        idx = n + offset
        fig.data[idx].x = d["x"]
        fig.data[idx].y = d["y"]
        fig.data[idx].customdata = d["customdata"]


def _patch_overlay_bundle(patch, n_traces, panel_data, offsets):
    """Same as ``_stamp_overlay_bundle`` but writes through a ``dash.Patch``
    object — used by patching callbacks that don't rebuild the figure."""
    if not panel_data:
        return
    d3, dxy, dxz, dyz = panel_data
    o3d, oxy, oxz, oyz = offsets
    patch["data"][n_traces + o3d]["x"] = d3["x"]
    patch["data"][n_traces + o3d]["y"] = d3["y"]
    patch["data"][n_traces + o3d]["z"] = d3["z"]
    patch["data"][n_traces + o3d]["customdata"] = d3["customdata"]
    for offset, d in zip((oxy, oxz, oyz), (dxy, dxz, dyz)):
        idx = n_traces + offset
        patch["data"][idx]["x"] = d["x"]
        patch["data"][idx]["y"] = d["y"]
        patch["data"][idx]["customdata"] = d["customdata"]


def register_treespace_callbacks():
    # Populate the MDS-result selector dropdown
    @callback(
        Output("treespace-result-select", "data"),
        Output("treespace-result-select", "value"),
        Input("mds-result-store", "data"),
        State("treespace-result-select", "value"),
    )
    def populate_result_selector(results, current_value):
        if not results:
            return [], None
        options = [
            {"value": k,
             "label": f"{v.get('filename', k)} ({v['rows']} trees, {len(v.get('groups', []))} runs)"}
            for k, v in results.items()
        ]
        if current_value and current_value in results:
            return options, current_value
        return options, list(results.keys())[-1]

    # When a result is picked, configure controls and reset the plot canvas.
    @callback(
        Output("plot-config-store", "data", allow_duplicate=True),
        Output("dim-x-select", "data"),
        Output("dim-x-select", "value"),
        Output("dim-y-select", "data"),
        Output("dim-y-select", "value"),
        Output("dim-z-select", "data"),
        Output("dim-z-select", "value"),
        Output("treenum-slider", "min"),
        Output("treenum-slider", "max"),
        Output("treenum-slider", "value"),
        Output("treenum-slider", "marks"),
        Output("treespace-info", "children"),
        Output("treespace-controls-paper", "style"),
        Output("graph", "figure", allow_duplicate=True),
        Output("treespace-selected-trees-store", "data", allow_duplicate=True),
        Output("treespace-export-trees", "disabled", allow_duplicate=True),
        Output("treespace-view-mcc", "disabled", allow_duplicate=True),
        Output("treespace-selection-info", "children", allow_duplicate=True),
        Input("treespace-result-select", "value"),
        State("mds-result-store", "data"),
        prevent_initial_call=True,
    )
    def configure_for_selected_result(selected_key, results):
        empty_state = (
            {},
            [], None,
            [], None,
            [], None,
            1, 100, [1, 100], [],
            html.Div(),
            {"display": "none"},
            placeholder_fig("No MDS result selected. Compute an MDS in the Compute tab."),
            [],
            True,
            True,
            html.Div(),
        )
        if not selected_key or not results or selected_key not in results:
            return empty_state

        mds_result = get_mds_result(selected_key)
        if not mds_result or not mds_result.get("data"):
            return empty_state

        metadata = mds_result["metadata"]
        combined_df = pd.DataFrame(mds_result["data"])
        mdscols = metadata["dimensions"]
        groups = combined_df["group"].unique().tolist()
        group_colors = px.colors.qualitative.Dark24[:len(groups)]
        color_dict = {g: c for g, c in zip(groups, group_colors)}
        MIN_TREENUM = metadata["MIN_TREENUM"]
        MAX_TREENUM = metadata["MAX_TREENUM"]

        plot_config = {
            "combined_data": mds_result["data"],
            "mdscols": mdscols,
            "min_treenum": MIN_TREENUM,
            "max_treenum": MAX_TREENUM,
            "groups": groups,
            "color_dict": color_dict,
        }

        dim_options = [{"value": col, "label": col} for col in mdscols]
        z_default = mdscols[2] if len(mdscols) > 2 else mdscols[0]

        marks = [
            {"value": MIN_TREENUM, "label": str(MIN_TREENUM)},
            {"value": MAX_TREENUM, "label": str(MAX_TREENUM)},
        ]

        info = dmc.Group([
            dmc.Badge(f"Trees: {len(combined_df)}",
                      variant="light", color="grape", size="sm"),
            dmc.Badge(f"Runs: {len(groups)}",
                      variant="light", color="teal", size="sm"),
        ], gap="xs")

        # ``no_update`` for the graph figure: the new dim values + slider
        # range fire the auto-update callback below, which renders the
        # real multiplot. Holding the previous figure until that lands
        # keeps the panel from flickering through a placeholder.
        return (
            plot_config,
            dim_options, mdscols[0],
            dim_options, mdscols[1],
            dim_options, z_default,
            MIN_TREENUM, MAX_TREENUM, [MIN_TREENUM, MAX_TREENUM], marks,
            info,
            {"display": "flex"},
            no_update,
            [],
            True,
            True,
            html.Div(),
        )

    # Auto-rebuild: any change to the dim selectors, the treenum range,
    # or the lines toggle re-renders the multiplot. Matches the always-
    # live UX of the within-run tab. When ``configure_for_selected_result``
    # populates the dim defaults + slider range, Dash coalesces all
    # changed Inputs into a single firing of this callback, so a fresh
    # result switch costs one rebuild — not five.
    @callback(
        Output("graph", "figure", allow_duplicate=True),
        Input("dim-x-select", "value"),
        Input("dim-y-select", "value"),
        Input("dim-z-select", "value"),
        Input("treenum-slider", "value"),
        Input("show-lines-checkbox", "checked"),
        State("graph", "figure"),
        State("plot-config-store", "data"),
        State("treespace-dragmode", "value"),
        State("treespace-selected-trees-store", "data"),
        State("mcc-registry-store", "data"),
        State("treespace-result-select", "value"),
        State("mds-result-store", "data"),
        prevent_initial_call=True,
    )
    def auto_update_graph(dim_x, dim_y, dim_z, treenum_range,
                          show_lines, current_fig, plot_config, dragmode,
                          selected, mcc_registry, selected_key,
                          mds_results):
        if not plot_config or not all([dim_x, dim_y, dim_z]) or not treenum_range:
            return no_update

        combined_df = pd.DataFrame(plot_config["combined_data"])

        filtered_dff = combined_df[
            (combined_df["treenum"] >= treenum_range[0])
            & (combined_df["treenum"] <= treenum_range[1])
        ]
        mds_selected = [dim_x, dim_y, dim_z]
        add_log(f"Plotting {len(filtered_dff)} trees (range {treenum_range[0]}-{treenum_range[1]}), dims: {mds_selected}")

        # Create new plot with filtered data. The interleaved variant splits
        # each group's points into chunks and stacks them by chunk-index so
        # no single run sits entirely on top of the others in the 2D panels.
        # It also appends 8 trailing overlay traces — 4 selection (red) +
        # 4 MCC (green).
        fig = make_plot_grid()
        add_trace_multiplot_interleaved(
            fig, filtered_dff, dim_x, dim_y, dim_z,
            plot_config["groups"], plot_config["color_dict"],
            show_lines=show_lines,
        )
        fig.update_layout(dragmode=dragmode or "zoom")

        # Try to preserve visibility settings if updating an existing plot.
        # Skip when the count differs (after dim/range/show-lines toggles the
        # number of chunk traces can change; a mismatch means we just emit a
        # fresh figure).
        if current_fig and "data" in current_fig:
            old_data = current_fig["data"]
            if len(old_data) == len(fig.data):
                for i in range(len(fig.data)):
                    old_vis = old_data[i].get("visible") if isinstance(old_data[i], dict) else None
                    if old_vis is not None:
                        fig.data[i].visible = old_vis

        # Re-apply current selection so the red overlays survive a rebuild.
        # Filter against the FILTERED dataframe so selected trees outside the
        # treenum-range stay invisible (matching how their normal markers
        # would have been hidden too).
        if selected:
            _stamp_overlay_bundle(
                fig, _overlay_panels_data(filtered_dff, selected,
                                          dim_x, dim_y, dim_z),
                SELECTION_OVERLAY_OFFSETS,
            )

        # Re-apply registered MCCs (green rings) for the current matrix.
        source_distmat = _resolve_source_distmat(selected_key, mds_results)
        if mcc_registry and source_distmat:
            _stamp_overlay_bundle(
                fig,
                _mcc_overlay_panels_data(filtered_dff, mcc_registry,
                                         source_distmat, dim_x, dim_y, dim_z),
                MCC_OVERLAY_OFFSETS,
            )

        return fig

    # ------ click / box / lasso → selection store ------
    @callback(
        Output("treespace-selected-trees-store", "data"),
        Input("graph", "clickData"),
        State("treespace-selected-trees-store", "data"),
        prevent_initial_call=True,
    )
    def handle_click_select(click_data, current_selection):
        if not click_data:
            return no_update
        point = click_data["points"][0]
        cd = point.get("customdata")
        if not cd or len(cd) < 3:
            return no_update
        try:
            treenum = int(cd[0])
            group = cd[2]
        except (ValueError, TypeError, IndexError):
            return no_update

        selected = {(g, int(t)) for g, t in (current_selection or [])}
        key = (group, treenum)
        if key in selected:
            selected.discard(key)
        else:
            selected.add(key)
        return sorted(([g, t] for g, t in selected), key=lambda p: (str(p[0]), p[1]))

    @callback(
        Output("treespace-selected-trees-store", "data", allow_duplicate=True),
        Input("graph", "selectedData"),
        State("treespace-selected-trees-store", "data"),
        prevent_initial_call=True,
    )
    def handle_region_select(selected_data, current_selection):
        if not selected_data or not selected_data.get("points"):
            return no_update
        new_pairs = set()
        for point in selected_data["points"]:
            cd = point.get("customdata")
            if not cd or len(cd) < 3:
                continue
            try:
                new_pairs.add((cd[2], int(cd[0])))
            except (ValueError, TypeError, IndexError):
                continue
        if not new_pairs:
            return no_update
        selected = {(g, int(t)) for g, t in (current_selection or [])}
        selected |= new_pairs
        return sorted(([g, t] for g, t in selected), key=lambda p: (str(p[0]), p[1]))

    @callback(
        Output("treespace-selected-trees-store", "data", allow_duplicate=True),
        Input("treespace-clear-selection", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_selection(n_clicks):
        if not n_clicks:
            return no_update
        return []

    # ------ dragmode → patch layout.dragmode (preserves zoom) ------
    @callback(
        Output("graph", "figure", allow_duplicate=True),
        Input("treespace-dragmode", "value"),
        prevent_initial_call=True,
    )
    def update_dragmode(dragmode):
        if not dragmode:
            return no_update
        patch = Patch()
        patch["layout"]["dragmode"] = dragmode
        return patch

    # ------ theme toggle → rebuild figure with active template ------
    @callback(
        Output("graph", "figure", allow_duplicate=True),
        Input("plotly-template-store", "data"),
        State("graph", "figure"),
        prevent_initial_call=True,
    )
    def update_plot_theme(_, current_fig):
        if not current_fig:
            return no_update
        return retheme_figure(current_fig, skip_invalid=True)

    # ------ selection-info badge + Export-trees / Export-MCC enable ------
    @callback(
        Output("treespace-selection-info", "children"),
        Output("treespace-export-trees", "disabled"),
        Output("treespace-view-mcc", "disabled"),
        Input("treespace-selected-trees-store", "data"),
    )
    def update_selection_info(selected):
        if not selected:
            return html.Div(), True, True
        return (
            dmc.Badge(f"Selected: {len(selected)} trees",
                      color="red", variant="light", size="lg"),
            False,
            False,
        )

    # ------ selection store change → patch only the last 4 traces ------
    @callback(
        Output("graph", "figure", allow_duplicate=True),
        Input("treespace-selected-trees-store", "data"),
        State("graph", "figure"),
        State("dim-x-select", "value"),
        State("dim-y-select", "value"),
        State("dim-z-select", "value"),
        State("plot-config-store", "data"),
        prevent_initial_call=True,
    )
    def update_selection_overlay(selected, current_fig,
                                 dim_x, dim_y, dim_z, plot_config):
        if not current_fig or not plot_config:
            return no_update
        n_traces = len(current_fig.get("data", []))
        # The trailing overlay block is 8 traces (4 selection + 4 MCC).
        # Bail if the graph hasn't been plotted yet or the trace count
        # doesn't have room for them.
        if n_traces < N_TRAILING_OVERLAYS:
            return no_update
        if not all([dim_x, dim_y, dim_z]):
            return no_update

        combined_df = pd.DataFrame(plot_config["combined_data"])
        sel_data = _overlay_panels_data(combined_df, selected, dim_x, dim_y, dim_z)

        patch = Patch()
        # Drop the dashed box/lasso rectangle Plotly persists in
        # layout.selections after a box-select.
        patch["layout"]["selections"] = []
        # Reset selectedpoints on every 2D trace so Plotly's auto-fade on
        # out-of-box markers doesn't mute points after box-select. Scatter3d
        # doesn't support selectedpoints — skipping the 3D traces avoids a
        # ValueError when go.Figure later reconstructs the dict (e.g. PDF
        # export).
        traces = current_fig.get("data", [])
        for i in range(n_traces):
            if traces[i].get("type") == "scatter3d":
                continue
            patch["data"][i]["selectedpoints"] = None

        _patch_overlay_bundle(patch, n_traces, sel_data,
                              SELECTION_OVERLAY_OFFSETS)
        return patch

    # ------ MCC registry change → patch only the green overlays ------
    @callback(
        Output("graph", "figure", allow_duplicate=True),
        Input("mcc-registry-store", "data"),
        State("graph", "figure"),
        State("dim-x-select", "value"),
        State("dim-y-select", "value"),
        State("dim-z-select", "value"),
        State("plot-config-store", "data"),
        State("treespace-result-select", "value"),
        State("mds-result-store", "data"),
        prevent_initial_call=True,
    )
    def update_mcc_overlay(registry, current_fig, dim_x, dim_y, dim_z,
                           plot_config, selected_key, mds_results):
        if not current_fig or not plot_config:
            return no_update
        n_traces = len(current_fig.get("data", []))
        if n_traces < N_TRAILING_OVERLAYS:
            return no_update
        if not all([dim_x, dim_y, dim_z]):
            return no_update

        source_distmat = _resolve_source_distmat(selected_key, mds_results)
        combined_df = pd.DataFrame(plot_config["combined_data"])
        mcc_data = _mcc_overlay_panels_data(
            combined_df, registry or [], source_distmat,
            dim_x, dim_y, dim_z,
        )

        patch = Patch()
        _patch_overlay_bundle(patch, n_traces, mcc_data,
                              MCC_OVERLAY_OFFSETS)
        return patch

    # ------ reset zoom button ------
    @callback(
        Output("graph", "figure", allow_duplicate=True),
        Input("treespace-reset-button", "n_clicks"),
        State("dim-x-select", "value"),
        State("dim-y-select", "value"),
        State("dim-z-select", "value"),
        State("treenum-slider", "value"),
        State("show-lines-checkbox", "checked"),
        State("plot-config-store", "data"),
        State("treespace-selected-trees-store", "data"),
        State("treespace-dragmode", "value"),
        State("mcc-registry-store", "data"),
        State("treespace-result-select", "value"),
        State("mds-result-store", "data"),
        prevent_initial_call=True,
    )
    def reset_axes(n_clicks, dim_x, dim_y, dim_z, treenum_range,
                   show_lines, plot_config, selected, dragmode,
                   mcc_registry, selected_key, mds_results):
        if not n_clicks or not plot_config or not all([dim_x, dim_y, dim_z]):
            return no_update

        combined_df = pd.DataFrame(plot_config["combined_data"])
        filtered_dff = combined_df[
            (combined_df["treenum"] >= treenum_range[0])
            & (combined_df["treenum"] <= treenum_range[1])
        ]

        fig = make_plot_grid()
        add_trace_multiplot_interleaved(
            fig, filtered_dff, dim_x, dim_y, dim_z,
            plot_config["groups"], plot_config["color_dict"],
            show_lines=show_lines,
        )
        fig.update_layout(dragmode=dragmode or "zoom")
        # Force a fresh uirevision so reset *does* throw away the user's zoom.
        fig.update_layout(uirevision=f"reset-{n_clicks}")

        # Re-apply current selection so the overlay survives the reset.
        if selected:
            _stamp_overlay_bundle(
                fig, _overlay_panels_data(filtered_dff, selected,
                                          dim_x, dim_y, dim_z),
                SELECTION_OVERLAY_OFFSETS,
            )
        # Re-apply registered MCCs (green rings).
        source_distmat = _resolve_source_distmat(selected_key, mds_results)
        if mcc_registry and source_distmat:
            _stamp_overlay_bundle(
                fig,
                _mcc_overlay_panels_data(filtered_dff, mcc_registry,
                                         source_distmat, dim_x, dim_y, dim_z),
                MCC_OVERLAY_OFFSETS,
            )
        return fig

    # ------ export selected trees as a .trees NEXUS file ------
    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("treespace-export-trees", "n_clicks"),
        State("treespace-selected-trees-store", "data"),
        State("plot-config-store", "data"),
        prevent_initial_call=True,
    )
    def export_selected_trees(n_clicks, selected_pairs, plot_config):
        from ..logger import notif_id
        from ..db.tree_service import get_tree_service
        from ._helpers import _save_file_dialog
        if not n_clicks or not selected_pairs or not plot_config:
            return no_update

        combined_df = pd.DataFrame(plot_config["combined_data"])
        selected_set = {(g, int(t)) for g, t in selected_pairs}
        keys = list(zip(combined_df["group"], combined_df["treenum"].astype(int)))
        mask = pd.Series([k in selected_set for k in keys], index=combined_df.index)
        sel_df = combined_df[mask]
        tree_names = sel_df["tree"].tolist()
        if not tree_names:
            return dmc.Notification(title="Export Error",
                                    message="No matching trees found.",
                                    color="red", action="show",
                                    autoClose=4000, id=notif_id())

        tree_service = get_tree_service()
        tree_service.db_manager.flush()
        all_trees = tree_service.db_manager._trees
        matched = all_trees[all_trees["name"].isin(tree_names)].sort_values("id")
        if len(matched) == 0:
            return dmc.Notification(
                title="Export Error",
                message="Selected trees not found in database. They may have been cleared.",
                color="red", action="show", autoClose=4000, id=notif_id())

        # Selected trees can come from multiple source files with different
        # translate tables. We pick the FIRST matched row's source as
        # canonical, write its preamble verbatim (which already includes the
        # `Translate` block), and remap the integer taxon labels in every
        # non-canonical tree's newick so they line up with the canonical
        # numbering. Tree names get prefixed with the group identifier so
        # cross-run STATE_X collisions are visible in the output.
        canonical_source = matched["file_source"].iloc[0]
        canonical_preamble = tree_service.db_manager._source_preambles.get(canonical_source)
        unique_sources = list(matched["file_source"].unique())
        remaps, missing_taxa = _build_canonical_remaps(
            unique_sources,
            tree_service.db_manager.get_translate_map,
            canonical_source,
        )
        if missing_taxa:
            sample = ", ".join(sorted(missing_taxa)[:5])
            more = "…" if len(missing_taxa) > 5 else ""
            return dmc.Notification(
                title="Export Error",
                message=(
                    f"Cannot align translate tables: taxa [{sample}{more}] "
                    f"are present in some selected runs but not in "
                    f"{canonical_source}'s Translate block. "
                    "Either deselect those runs or pick selections from runs "
                    "with matching taxa."
                ),
                color="red", action="show", autoClose=8000, id=notif_id())

        path = _save_file_dialog(default_filename=f"selected_{len(matched)}_trees.trees")
        if not path:
            return no_update

        try:
            with open(path, "wb") as out:
                if canonical_preamble:
                    out.write(canonical_preamble)
                else:
                    out.write(b"#NEXUS\n\nbegin trees;\n")
                for _, row in matched.iterrows():
                    file_source = row["file_source"]
                    line = tree_service.db_manager._read_newick(
                        file_source,
                        int(row["line_offset"]),
                        int(row["line_length"]),
                    )
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    db_name = row["name"]                      # "<group>/<orig>"
                    group_name = row["group_name"]
                    original_name = (
                        db_name.split("/", 1)[1]
                        if "/" in db_name else db_name
                    )
                    new_name = _sanitize_tree_name_token(
                        f"{group_name}_{original_name}"
                    )
                    rewritten = _rewrite_tree_line(
                        line, new_name, remaps.get(file_source, {}),
                    )
                    out.write(rewritten.encode("utf-8"))
                    if not rewritten.endswith("\n"):
                        out.write(b"\n")
                out.write(b"End;\n")
        except Exception as e:
            return dmc.Notification(title="Export Error", message=str(e),
                                    color="red", action="show",
                                    autoClose=6000, id=notif_id())

        add_log(f"Exported {len(matched)} selected trees to {path}")
        return dmc.Notification(title="Trees Exported",
                                message=f"Exported {len(matched)} trees to {path}",
                                color="green", action="show",
                                autoClose=4000, id=notif_id())

    # ------ View MCC tree — thin submit handler ------
    # Validates input, builds the matched-record list + MCC-coord
    # lookup, then hands off to the persistent worker via
    # ``mcc_compute.submit_mcc_job``. Completion is handled by
    # ``mcc_compute.poll_mcc_completion`` which fans the result back to
    # this tab's view-mcc-store, dismisses the loading overlay, and
    # re-enables the button.
    @callback(
        Output("treespace-loading-overlay", "visible", allow_duplicate=True),
        Output("treespace-view-mcc", "disabled", allow_duplicate=True),
        # MCC polling uses its own interval (see navbar.py) so this
        # handler and poll_mcc_completion don't collide with the RF/MDS
        # poll on a shared allow_duplicate output.
        Output("mcc-poll-interval", "disabled", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
        Input("treespace-view-mcc", "n_clicks"),
        State("treespace-selected-trees-store", "data"),
        State("plot-config-store", "data"),
        State("treespace-result-select", "value"),
        State("mds-result-store", "data"),
        prevent_initial_call=True,
    )
    def view_mcc_tree(n_clicks, selected_pairs, plot_config,
                      selected_key, results):
        from ..logger import notif_id
        from ..db.tree_service import get_tree_service
        from . import mcc_compute

        if not n_clicks or not selected_pairs or not plot_config:
            return no_update, no_update, no_update, no_update

        def _err(msg, autoclose=5000):
            return (False, False, no_update, dmc.Notification(
                title="MCC Error", message=msg,
                color="red", action="show", autoClose=autoclose,
                id=notif_id(),
            ))

        results = results or {}
        if not selected_key or selected_key not in results:
            return _err("No MDS result is currently selected.")
        source_distmat = (results[selected_key] or {}).get("source_distmat")
        if not source_distmat:
            return _err("No RF/snapshot data is associated with this MDS result.")

        combined_df = pd.DataFrame(plot_config["combined_data"])
        selected_set = {(g, int(t)) for g, t in selected_pairs}
        keys = list(zip(combined_df["group"], combined_df["treenum"].astype(int)))
        mask = pd.Series([k in selected_set for k in keys], index=combined_df.index)
        sel_df = combined_df[mask]
        tree_names = sel_df["tree"].tolist()
        if not tree_names:
            return _err("No matching trees found.", autoclose=4000)

        tree_service = get_tree_service()
        tree_service.db_manager.flush()
        all_trees = tree_service.db_manager._trees
        matched = all_trees[all_trees["name"].isin(tree_names)].sort_values("id")
        if len(matched) == 0:
            return _err(
                "Selected trees not found in database. They may have been cleared.",
                autoclose=4000,
            )

        # Plain-dict records the persistent worker can pickle.
        matched_records = matched[[
            "name", "file_source", "line_offset", "line_length", "metadata",
        ]].to_dict("records")
        for rec in matched_records:
            rec["line_offset"] = int(rec["line_offset"])
            rec["line_length"] = int(rec["line_length"])

        # (group, treenum) per tree-name so the poll callback can put the
        # green ring on the MCC's dot.
        mcc_coord_by_tree_name = {
            row["tree"]: (row["group"], int(row["treenum"]))
            for _, row in combined_df.iterrows()
        }

        mcc_compute.submit_mcc_job(
            matched_records=matched_records,
            source_distmat=source_distmat,
            mode="Between",
            selection=[[g, int(t)] for g, t in selected_pairs],
            run=None,
            mcc_coord_by_tree_name=mcc_coord_by_tree_name,
            store_target="treespace-view-mcc-store",
        )

        # Return: overlay on, button disabled, polling enabled, no
        # notification yet (notification fires when compute finishes).
        return True, True, False, no_update

    # The per-tab clientside ``window.open`` that used to live here is
    # gone — it was a duplicate of the one in within_run.py and the
    # one in mcc_list.py. They all now route through
    # ``mcc-peartree-open-store`` and the single clientside callback
    # in ``callbacks/rename_mcc.py`` does the actual ``window.open``.
    # The poll callback in ``mcc_compute.py`` still writes the freshly
    # registered MCC's ``{uuid, name}`` to ``treespace-view-mcc-store``
    # — it's now picked up by ``forward_compute_to_modal`` in
    # ``rename_mcc.py``, which opens the rename modal (with
    # ``after='view'``) so the user can confirm or edit the auto-name
    # before peartree opens on Save.

    # ------ export PDF ------
    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("treespace-export-pdf", "n_clicks"),
        State("graph", "figure"),
        prevent_initial_call=True,
    )
    def export_pdf(n_clicks, fig_dict):
        from ..logger import notif_id
        from ._helpers import _save_file_dialog
        if not n_clicks or not fig_dict:
            return no_update
        path = _save_file_dialog(default_filename="treespace.pdf")
        if not path:
            return no_update
        # skip_invalid=True drops any browser-only properties that Plotly's
        # Python validator rejects (e.g. selectedpoints accidentally left on
        # a 3D trace by an old Patch).
        fig = go.Figure(fig_dict, skip_invalid=True)
        fig.update_layout(template=get_template())
        fig.write_image(path, width=1800, height=1200, scale=2)
        add_log(f"Exported between-run plot to {path}")
        return dmc.Notification(title="PDF Exported",
                                message=f"Saved to {path}",
                                color="green", action="show",
                                autoClose=3000, id=notif_id())
