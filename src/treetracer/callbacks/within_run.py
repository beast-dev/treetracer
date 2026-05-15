"""Within-run analysis callbacks and figure rendering.

The figure is built as 3 linked 2D panels (x-y, x-z, y-z) sharing axes via
``matches``. The trace layout is fixed at 9 traces — three panels of
``{out-of-range, in-range}`` plus three trailing selection overlays — and that
constant trace count is load-bearing:

* It lets ``update_selection_overlay`` rewrite only the last 3 traces' x/y/
  customdata via ``dash.Patch`` whenever the selection store changes, so a
  click / box / lasso never rebuilds the figure (zoom and pan stay put).
* The same is done for the show-lines checkbox — a one-property Patch on the
  three in-range traces' ``mode``.
* Selection-related state that Plotly persists across renders
  (``layout.selections`` and per-trace ``selectedpoints``) is cleared on every
  selection-store update; otherwise the dashed bounding box from box-select
  sticks around and Plotly fades the overlay circles outside it.

Triggers that DO need a real rebuild (dim selectors, treenum range,
color-gradient toggle) read the user's current zoom out of
``current_fig.layout.{x,y}axis*.range`` and pin it on the new figure —
``uirevision`` alone has proven unreliable across full figure replacements.
"""

from dash import callback, clientside_callback, Input, Output, Patch, State, no_update, ctx, html
import dash_mantine_components as dmc
from ..icons import icon
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from ..theme import get_template
from ..plot_utils import retheme_figure


TREETRACER_BLUE = "#228be6"

# Fixed 12-trace shape: 3 panels × {out-of-range, in-range} + 3 selection
# overlays (red) + 3 MCC overlays (neon green). The patching callbacks
# address each bundle by its fixed negative offsets so a selection
# change or MCC-registry change never rebuilds the figure.
N_TRACES = 12
SELECTION_OVERLAY_OFFSETS = (-6, -5, -4)
MCC_OVERLAY_OFFSETS = (-3, -2, -1)
MCC_OVERLAY_COLOR = "#39ff14"


def _get_active_result(selected_key, results_index):
    """Resolve a between-run MDS result from the server-side store.

    The dcc.Store (``results_index``) only has metadata; full coordinate data
    is read from ``state._mds_results`` via ``get_mds_result``. The returned
    dict has the shape ``{"metadata": {...}, "data": [...]}``, where ``data``
    is a list of per-tree dicts including ``group``, ``treenum``, ``tree``,
    and the MDS dimension columns.
    """
    if not selected_key or not results_index or selected_key not in results_index:
        return None
    from ..state import get_mds_result
    return get_mds_result(selected_key)


def _filter_to_run(mds_result, selected_run):
    """Return the rows of ``mds_result['data']`` belonging to one run, plus
    the global axis ranges computed from the *unfiltered* dataframe so the
    within-run plot can be pinned to the between-run coordinate window.

    Returns ``(df_run, axis_ranges)`` or ``(None, None)`` when the inputs are
    incomplete.
    """
    if mds_result is None or not selected_run:
        return None, None
    df_full = pd.DataFrame(mds_result["data"])
    if df_full.empty or "group" not in df_full.columns:
        return None, None

    dimensions = mds_result.get("metadata", {}).get("dimensions") or [
        c for c in df_full.columns if c.startswith("MDS")
    ]
    axis_ranges = {
        col: (float(df_full[col].min()), float(df_full[col].max()))
        for col in dimensions if col in df_full.columns
    }

    df_run = df_full[df_full["group"] == selected_run].reset_index(drop=True)
    if df_run.empty:
        return None, axis_ranges

    if "treenum" not in df_run.columns or df_run["treenum"].isna().any():
        df_run["treenum"] = range(1, len(df_run) + 1)
    df_run["treenum"] = df_run["treenum"].astype(int)
    return df_run, axis_ranges


def _empty_panel():
    return {"x": [], "y": [], "customdata": []}


def _filter_mccs_for_within(registry, results, selected_key, selected_run):
    """Subset of the MCC registry that should ring-overlay in this run.

    Filters by ``source_distmat`` (taken from the active MDS result),
    ``mode == 'Within'``, and ``run == selected_run``. Returns an empty
    list when any required input is missing.
    """
    if not registry or not selected_key or not selected_run:
        return []
    if not results or selected_key not in results:
        return []
    source_distmat = (results[selected_key] or {}).get("source_distmat")
    if not source_distmat:
        return []
    return [e for e in registry
            if e.get("source_distmat") == source_distmat
            and e.get("mode") == "Within"
            and e.get("run") == selected_run]


def _selection_panels_data(df, selected_treenums, x, y, z):
    """Per-panel x / y / customdata for the 3 selection-overlay traces.

    Order matches the ``panels`` list in ``_make_within_run_figure``:
        index 0 → (x, y) panel
        index 1 → (x, z) panel
        index 2 → (y, z) panel

    When nothing is selected (or no rows match), every panel gets empty arrays
    — the trace stays present but draws nothing.
    """
    if not selected_treenums:
        return [_empty_panel(), _empty_panel(), _empty_panel()]
    sel = df[df["treenum"].isin(selected_treenums)]
    if len(sel) == 0:
        return [_empty_panel(), _empty_panel(), _empty_panel()]
    labels = sel["tree"].str.split("/").str[-1].str.strip().tolist()
    treenums = sel["treenum"].astype(int).tolist()
    cd = list(zip(treenums, labels))
    return [
        {"x": sel[x].tolist(), "y": sel[y].tolist(), "customdata": cd},
        {"x": sel[x].tolist(), "y": sel[z].tolist(), "customdata": cd},
        {"x": sel[y].tolist(), "y": sel[z].tolist(), "customdata": cd},
    ]


def _in_range_marker(df_in, color_gradient, show_cb, colorscale="Blues"):
    """Marker dict for an in-range trace.

    Lives outside ``_make_within_run_figure`` so the gradient-toggle handler
    can build identical markers without rebuilding the figure.
    """
    if len(df_in) > 0 and color_gradient:
        tmin = int(df_in["treenum"].min())
        tmax = int(df_in["treenum"].max())
        return dict(
            color=df_in["treenum"].tolist(), colorscale=colorscale, size=7,
            cmin=tmin, cmax=tmax,
            colorbar=dict(
                title="Tree #", x=1.02, len=0.9,
                tick0=tmin, dtick=max(1, (tmax - tmin) // 5),
            ) if show_cb else None,
            showscale=show_cb,
        )
    return dict(color=TREETRACER_BLUE, size=7)


def _selection_overlay_trace(xs, ys, customdata):
    """Hollow red-outline marker trace used as a selection overlay. Empty
    arrays are valid — they leave the trace present-but-invisible so the
    Patch in ``update_selection_overlay`` always has a stable target.

    Uses ``Scattergl`` so the overlay lives on the SAME WebGL canvas as
    the data traces. Within one canvas, trace insertion order wins;
    overlays are added after data, so they always draw on top. Earlier
    we tried SVG ``Scatter`` + ``zorder`` instead, but Plotly's WebGL
    canvas was rendering ABOVE the SVG layer in subplots and the data
    obscured the ring (Scattergl also rejects ``zorder``).

    Explicit ``selected`` / ``unselected`` styles pin opacity to 1 in
    both states. Without them, a fresh box-select stamps
    ``selectedpoints`` on this trace and Plotly fades overlay circles
    outside the new box (default unselected opacity ≈ 0.2).
    """
    return go.Scattergl(
        x=xs, y=ys,
        mode="markers",
        marker=dict(
            size=12,
            color="rgba(0,0,0,0)",
            line=dict(color="red", width=1.5),
        ),
        customdata=customdata,
        hovertemplate="Tree #%{customdata[0]}: %{customdata[1]}<extra>selected</extra>",
        showlegend=False,
        selected=dict(marker=dict(opacity=1)),
        unselected=dict(marker=dict(opacity=1)),
    )


def _mcc_overlay_trace(xs, ys, customdata):
    """Neon-green hollow ring used as the registered-MCC overlay. Same
    shape and renderer as the selection overlay so the patch callbacks
    address it the same way; just different colour, line width, and
    hovertemplate. Added to the figure AFTER the selection overlay, so
    within the shared WebGL canvas the MCC ring draws on top of both
    the data points and the (also Scattergl) selection ring."""
    return go.Scattergl(
        x=xs, y=ys,
        mode="markers",
        marker=dict(
            size=14,
            color="rgba(0,0,0,0)",
            line=dict(color=MCC_OVERLAY_COLOR, width=3.0),
        ),
        customdata=customdata,
        hovertemplate="Tree #%{customdata[0]}: %{customdata[1]}<extra>MCC</extra>",
        showlegend=False,
        selected=dict(marker=dict(opacity=1)),
        unselected=dict(marker=dict(opacity=1)),
    )


def _mcc_panels_data(df, mcc_entries, x, y, z):
    """Per-panel x / y / customdata for the 3 MCC-overlay traces.

    *mcc_entries* is a pre-filtered list of registry entries (already
    matching this run + matrix). For each entry we look up the
    ``(treenum)`` of its MCC tree in *df* and emit one point per panel.
    customdata[1] holds the registered MCC name so hover reads
    "Tree #N: <RF_001_Within_runA_MCC_2>".
    """
    if not mcc_entries:
        return [_empty_panel(), _empty_panel(), _empty_panel()]

    treenum_to_name = {}
    for e in mcc_entries:
        mt = e.get("mcc_tree") or {}
        t = mt.get("treenum")
        if t is None:
            continue
        treenum_to_name[int(t)] = e.get("name", "")

    if not treenum_to_name:
        return [_empty_panel(), _empty_panel(), _empty_panel()]

    sel = df[df["treenum"].astype(int).isin(treenum_to_name)]
    if len(sel) == 0:
        return [_empty_panel(), _empty_panel(), _empty_panel()]

    treenums = sel["treenum"].astype(int).tolist()
    names = [treenum_to_name.get(t, "") for t in treenums]
    cd = list(zip(treenums, names))
    return [
        {"x": sel[x].tolist(), "y": sel[y].tolist(), "customdata": cd},
        {"x": sel[x].tolist(), "y": sel[z].tolist(), "customdata": cd},
        {"x": sel[y].tolist(), "y": sel[z].tolist(), "customdata": cd},
    ]


def _make_within_run_figure(df, x, y, z, show_lines=True,
                            selected_treenums=None, treenum_range=None,
                            color_gradient=True, dragmode="zoom",
                            axis_ranges=None, mcc_entries=None):
    """Build the 3-panel within-run figure with a fixed 12-trace shape::

        0  panel (x,y)  out-of-range  (lightgrey, hover-disabled)
        1  panel (x,y)  in-range      (gradient or flat)
        2  panel (x,z)  out-of-range
        3  panel (x,z)  in-range
        4  panel (y,z)  out-of-range
        5  panel (y,z)  in-range
        6  panel (x,y)  selection overlay   (red outline)
        7  panel (x,z)  selection overlay
        8  panel (y,z)  selection overlay
        9  panel (x,y)  MCC overlay         (neon-green outline)
        10 panel (x,z)  MCC overlay
        11 panel (y,z)  MCC overlay

    Empty data is emitted as ``[]`` rather than dropping the trace so the
    count stays constant — that's what lets the patch callbacks rewrite
    the trailing overlay bundles by index.
    """
    range_str = f" (trees {treenum_range[0]}–{treenum_range[1]})" if treenum_range else ""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f"{x} vs {y}{range_str}",
            f"{x} vs {z}{range_str}",
            f"{y} vs {z}{range_str}",
        ],
        horizontal_spacing=0.06,
    )
    for ann in fig.layout.annotations:
        ann.font.size = 11

    mode = "lines+markers" if show_lines else "markers"
    colorscale = "Blues"
    tree_labels = df["tree"].str.split("/").str[-1].str.strip()

    if treenum_range is not None:
        in_mask = (df["treenum"] >= treenum_range[0]) & (df["treenum"] <= treenum_range[1])
    else:
        in_mask = pd.Series(True, index=df.index)
    out_mask = ~in_mask
    df_in = df[in_mask]
    df_out = df[out_mask]

    panels = [
        (x, y, 1, 1, True),
        (x, z, 1, 2, False),
        (y, z, 1, 3, False),
    ]

    for xcol, ycol, row, col, show_cb in panels:
        # Out-of-range — always emitted, even when empty. ``Scattergl``
        # so an 8k-tree run renders in one WebGL frame instead of
        # spawning a DOM node per marker.
        fig.add_trace(go.Scattergl(
            x=df_out[xcol].tolist() if len(df_out) else [],
            y=df_out[ycol].tolist() if len(df_out) else [],
            mode="markers",
            marker=dict(color="lightgrey", size=5, opacity=0.4),
            hoverinfo="skip",
            showlegend=False,
            customdata=(list(zip(df_out["treenum"], tree_labels[out_mask]))
                        if len(df_out) else []),
            selected=dict(marker=dict(opacity=0.4)),
            unselected=dict(marker=dict(opacity=0.4)),
        ), row=row, col=col)

        # In-range — always emitted, even when empty. Same WebGL
        # treatment; the gradient colourbar still renders correctly
        # against a Scattergl trace.
        marker_dict = _in_range_marker(df_in, color_gradient, show_cb, colorscale)

        fig.add_trace(go.Scattergl(
            x=df_in[xcol].tolist() if len(df_in) else [],
            y=df_in[ycol].tolist() if len(df_in) else [],
            mode=mode,
            marker=marker_dict,
            line=dict(color="rgba(120,120,120,0.4)", width=1),
            hovertemplate="Tree #%{customdata[0]}: %{customdata[1]}<extra></extra>",
            customdata=(list(zip(df_in["treenum"], tree_labels[in_mask]))
                        if len(df_in) else []),
            showlegend=False,
            selected=dict(marker=dict(opacity=1)),
            unselected=dict(marker=dict(opacity=1)),
        ), row=row, col=col)

    sel_data = _selection_panels_data(df, selected_treenums, x, y, z)
    for (_, _, row, col, _), d in zip(panels, sel_data):
        fig.add_trace(_selection_overlay_trace(d["x"], d["y"], d["customdata"]),
                      row=row, col=col)

    mcc_data = _mcc_panels_data(df, mcc_entries or [], x, y, z)
    for (_, _, row, col, _), d in zip(panels, mcc_data):
        fig.add_trace(_mcc_overlay_trace(d["x"], d["y"], d["customdata"]),
                      row=row, col=col)

    fig.update_layout(
        xaxis2=dict(matches='x'),
        xaxis3=dict(matches='y'),
        yaxis3=dict(matches='y2'),
    )
    fig.update_xaxes(title_text=x, row=1, col=1)
    fig.update_yaxes(title_text=y, row=1, col=1)
    fig.update_xaxes(title_text=x, row=1, col=2)
    fig.update_yaxes(title_text=z, row=1, col=2)
    fig.update_xaxes(title_text=y, row=1, col=3)
    fig.update_yaxes(title_text=z, row=1, col=3)

    if axis_ranges:
        if x in axis_ranges:
            fig.update_xaxes(range=axis_ranges[x], row=1, col=1)
            fig.update_xaxes(range=axis_ranges[x], row=1, col=2)
        if y in axis_ranges:
            fig.update_yaxes(range=axis_ranges[y], row=1, col=1)
            fig.update_xaxes(range=axis_ranges[y], row=1, col=3)
        if z in axis_ranges:
            fig.update_yaxes(range=axis_ranges[z], row=1, col=2)
            fig.update_yaxes(range=axis_ranges[z], row=1, col=3)

    fig.update_layout(
        template=get_template(),
        margin=dict(l=50, r=40, t=40, b=45),
        dragmode=dragmode,
        uirevision="within-run",
    )
    return fig


def register_within_run_callbacks():

    # ------ selectors ------
    @callback(
        Output("within-run-result-select", "data"),
        Output("within-run-result-select", "value"),
        Input("mds-result-store", "data"),
        State("within-run-result-select", "value"),
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

    @callback(
        Output("within-run-run-select", "data"),
        Output("within-run-run-select", "value"),
        Input("within-run-result-select", "value"),
        State("mds-result-store", "data"),
        State("within-run-run-select", "value"),
    )
    def populate_run_selector(selected_key, results, current_run):
        if not selected_key or not results or selected_key not in results:
            return [], None
        groups = results[selected_key].get("groups", [])
        if not groups:
            return [], None
        options = [{"value": g, "label": g} for g in groups]
        if current_run and current_run in groups:
            return options, current_run
        return options, groups[0]

    # ------ initial render ------
    @callback(
        Output("within-run-controls-paper", "style"),
        Output("within-run-dim-x", "data"),
        Output("within-run-dim-x", "value"),
        Output("within-run-dim-y", "data"),
        Output("within-run-dim-y", "value"),
        Output("within-run-dim-z", "data"),
        Output("within-run-dim-z", "value"),
        Output("within-run-treenum-slider", "min"),
        Output("within-run-treenum-slider", "max"),
        Output("within-run-treenum-slider", "value"),
        Output("within-run-treenum-slider", "marks"),
        Output("within-run-info", "children"),
        Output("within-run-graph", "figure"),
        Output("within-run-selected-trees-store", "data", allow_duplicate=True),
        Output("within-run-treenum-range-store", "data", allow_duplicate=True),
        Output("within-run-anim-interval", "disabled", allow_duplicate=True),
        Output("within-run-play-button", "children", allow_duplicate=True),
        Input("within-run-result-select", "value"),
        Input("within-run-run-select", "value"),
        State("mds-result-store", "data"),
        State("mcc-registry-store", "data"),
        prevent_initial_call=True,
    )
    def load_result_for_visualization(selected_key, selected_run, results,
                                      mcc_registry):
        result = _get_active_result(selected_key, results)
        if result is None or not selected_run:
            return (no_update,) * 17

        df_run, axis_ranges = _filter_to_run(result, selected_run)
        if df_run is None:
            return (no_update,) * 17

        mdscols = result.get("metadata", {}).get("dimensions") or [
            c for c in df_run.columns if c.startswith("MDS")
        ]
        n = len(df_run)
        z_default = mdscols[2] if len(mdscols) > 2 else mdscols[0]
        dim_options = [{"value": col, "label": col} for col in mdscols]
        marks = [{"value": max(1, round(n * i / 10)),
                  "label": str(max(1, round(n * i / 10)))}
                 for i in range(11)]

        mcc_entries = _filter_mccs_for_within(
            mcc_registry, results, selected_key, selected_run)
        fig = _make_within_run_figure(
            df_run, mdscols[0], mdscols[1], z_default,
            treenum_range=[1, n],
            axis_ranges=axis_ranges,
            mcc_entries=mcc_entries,
        )

        info = dmc.Group([
            dmc.Badge(f"Result: {result.get('metadata', {}).get('filename', '?')}",
                      variant="light", color="blue", size="lg"),
            dmc.Badge(f"Run: {selected_run}", variant="light", color="teal", size="lg"),
            dmc.Badge(f"Trees: {n}", variant="light", color="grape", size="lg"),
        ], gap="sm")

        return (
            {"display": "flex"},
            dim_options, mdscols[0],
            dim_options, mdscols[1],
            dim_options, z_default,
            1, n, [1, n], marks,
            info,
            fig,
            [],
            [1, n],
            True,
            icon("tabler:player-play-filled", size=18),
        )

    # ------ slider plumbing ------
    @callback(
        Output("within-run-treenum-range-store", "data"),
        Input("within-run-treenum-slider", "value"),
        prevent_initial_call=True,
    )
    def update_treenum_range(slider_value):
        if not slider_value:
            return no_update
        return slider_value

    @callback(
        Output("within-run-selected-trees-store", "data", allow_duplicate=True),
        Input("within-run-treenum-range-store", "data"),
        State("within-run-selected-trees-store", "data"),
        prevent_initial_call=True,
    )
    def filter_selection_on_range_change(treenum_range, current_selection):
        if not current_selection or not treenum_range:
            return no_update
        range_min, range_max = treenum_range
        filtered = [t for t in current_selection if range_min <= t <= range_max]
        if len(filtered) == len(current_selection):
            return no_update
        return filtered

    @callback(
        Output("within-run-treenum-slider", "minRange"),
        Input("within-run-window-size", "value"),
        prevent_initial_call=True,
    )
    def update_min_range(window_size):
        try:
            val = int(window_size) if window_size else 1
        except (ValueError, TypeError):
            return no_update
        if val < 1:
            return no_update
        return val

    # ------ animation ------
    @callback(
        Output("within-run-anim-interval", "disabled"),
        Output("within-run-play-button", "children"),
        Output("within-run-treenum-slider", "value", allow_duplicate=True),
        Input("within-run-play-button", "n_clicks"),
        State("within-run-anim-interval", "disabled"),
        State("within-run-treenum-slider", "min"),
        State("within-run-treenum-slider", "max"),
        State("within-run-window-size", "value"),
        prevent_initial_call=True,
    )
    def toggle_playback(n_clicks, currently_disabled, slider_min, slider_max, window_size):
        if not n_clicks:
            return no_update, no_update, no_update
        if currently_disabled:
            w = int(window_size) if window_size else 100
            return (
                False,
                icon("tabler:player-pause-filled", size=18),
                [slider_min, min(slider_min + w, slider_max)],
            )
        return (
            True,
            icon("tabler:player-play-filled", size=18),
            no_update,
        )

    @callback(
        Output("within-run-treenum-slider", "value", allow_duplicate=True),
        Output("within-run-anim-interval", "disabled", allow_duplicate=True),
        Output("within-run-play-button", "children", allow_duplicate=True),
        Input("within-run-anim-interval", "n_intervals"),
        State("within-run-treenum-slider", "value"),
        State("within-run-treenum-slider", "min"),
        State("within-run-treenum-slider", "max"),
        State("within-run-window-size", "value"),
        prevent_initial_call=True,
    )
    def advance_animation(n_intervals, current_value, slider_min, slider_max, window_size):
        if not current_value or not window_size:
            return no_update, no_update, no_update
        w = int(window_size)
        stride = max(1, w // 2)
        new_start = current_value[0] + stride
        new_end = new_start + w
        if new_start >= slider_max:
            return (
                [slider_min, min(slider_min + w, slider_max)],
                True,
                icon("tabler:player-play-filled", size=18),
            )
        if new_end > slider_max:
            new_end = slider_max
        return [new_start, new_end], no_update, no_update

    # ------ click / box / lasso → selection store ------
    @callback(
        Output("within-run-selected-trees-store", "data"),
        Input("within-run-graph", "clickData"),
        State("within-run-selected-trees-store", "data"),
        State("within-run-treenum-range-store", "data"),
        prevent_initial_call=True,
    )
    def handle_click_select(click_data, current_selection, treenum_range):
        if not click_data:
            return no_update
        point = click_data["points"][0]
        clicked_treenum = None
        if point.get("customdata"):
            try:
                clicked_treenum = int(point["customdata"][0])
            except (ValueError, TypeError, IndexError):
                pass
        if clicked_treenum is None:
            return no_update
        if treenum_range:
            if clicked_treenum < treenum_range[0] or clicked_treenum > treenum_range[1]:
                return no_update
        selected = set(current_selection or [])
        if clicked_treenum in selected:
            selected.discard(clicked_treenum)
        else:
            selected.add(clicked_treenum)
        return sorted(selected)

    @callback(
        Output("within-run-selected-trees-store", "data", allow_duplicate=True),
        Input("within-run-graph", "selectedData"),
        State("within-run-selected-trees-store", "data"),
        State("within-run-treenum-range-store", "data"),
        prevent_initial_call=True,
    )
    def handle_region_select(selected_data, current_selection, treenum_range):
        if not selected_data or not selected_data.get("points"):
            return no_update
        range_min = treenum_range[0] if treenum_range else -float("inf")
        range_max = treenum_range[1] if treenum_range else float("inf")
        new_treenums = set()
        for point in selected_data["points"]:
            if point.get("customdata"):
                try:
                    tn = int(point["customdata"][0])
                    if range_min <= tn <= range_max:
                        new_treenums.add(tn)
                except (ValueError, TypeError, IndexError):
                    pass
        if not new_treenums:
            return no_update
        selected = set(current_selection or [])
        selected |= new_treenums
        return sorted(selected)

    @callback(
        Output("within-run-selected-trees-store", "data", allow_duplicate=True),
        Input("within-run-clear-selection", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_selection(n_clicks):
        if not n_clicks:
            return no_update
        return []

    # ------ dragmode toggle ------
    @callback(
        Output("within-run-graph", "figure", allow_duplicate=True),
        Input("within-run-dragmode", "value"),
        State("within-run-graph", "figure"),
        prevent_initial_call=True,
    )
    def update_dragmode(dragmode, current_fig):
        if not current_fig or not dragmode:
            return no_update
        fig = go.Figure(current_fig)
        fig.update_layout(dragmode=dragmode)
        return fig

    # ------ theme toggle → rebuild figure with active template ------
    @callback(
        Output("within-run-graph", "figure", allow_duplicate=True),
        Input("plotly-template-store", "data"),
        State("within-run-graph", "figure"),
        prevent_initial_call=True,
    )
    def update_plot_theme(_, current_fig):
        if not current_fig:
            return no_update
        return retheme_figure(current_fig)

    # ------ selection-info badge ------
    @callback(
        Output("within-run-selection-info", "children"),
        Output("within-run-export-trees", "disabled"),
        Output("within-run-view-mcc", "disabled"),
        Input("within-run-selected-trees-store", "data"),
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

    # ------ export selected trees ------
    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("within-run-export-trees", "n_clicks"),
        State("within-run-selected-trees-store", "data"),
        State("within-run-result-select", "value"),
        State("within-run-run-select", "value"),
        State("mds-result-store", "data"),
        prevent_initial_call=True,
    )
    def export_selected_trees(n_clicks, selected_treenums, selected_key, selected_run, results):
        from ..logger import add_log, notif_id
        if not n_clicks or not selected_treenums:
            return no_update
        mds_result = _get_active_result(selected_key, results)
        if not mds_result or not selected_run:
            return no_update
        df_run, _ = _filter_to_run(mds_result, selected_run)
        if df_run is None:
            return no_update
        sel_df = df_run[df_run["treenum"].isin(selected_treenums)]
        tree_names = sel_df["tree"].tolist()
        if not tree_names:
            return dmc.Notification(title="Export Error", message="No matching trees found.",
                                    color="red", action="show", autoClose=4000, id=notif_id())

        from ..db.tree_service import get_tree_service
        from ._helpers import _save_file_dialog

        tree_service = get_tree_service()
        tree_service.db_manager.flush()
        all_trees = tree_service.db_manager._trees
        matched = all_trees[all_trees["name"].isin(tree_names)].sort_values("id")
        if len(matched) == 0:
            return dmc.Notification(
                title="Export Error",
                message="Selected trees not found in database. They may have been cleared.",
                color="red", action="show", autoClose=4000, id=notif_id())

        file_source = matched["file_source"].iloc[0]
        path = _save_file_dialog(default_filename=f"selected_{len(matched)}_trees.trees")
        if not path:
            return no_update

        preamble = tree_service.db_manager._source_preambles.get(file_source)
        try:
            with open(path, "wb") as out:
                if preamble:
                    out.write(preamble)
                for _, row in matched.iterrows():
                    line = tree_service.db_manager._read_newick(
                        row["file_source"], int(row["line_offset"]), int(row["line_length"])
                    )
                    out.write(line.encode("utf-8") if isinstance(line, str) else line)
                    out.write(b"\n")
                out.write(b"End;\n")
        except Exception as e:
            return dmc.Notification(title="Export Error", message=str(e),
                                    color="red", action="show", autoClose=6000, id=notif_id())

        add_log(f"Exported {len(matched)} selected trees to {path}")
        return dmc.Notification(title="Trees Exported",
                                message=f"Exported {len(matched)} trees to {path}",
                                color="green", action="show", autoClose=4000, id=notif_id())

    # ------ View MCC tree in a peartree window ------
    # Same selection plumbing as export_selected_trees, but instead of
    # writing a NEXUS file we hand the matched DataFrame to
    # ``mcc.assemble_mcc_nexus`` and stash the bytes in the in-memory
    # MCC cache. The view-mcc store gets {"uuid", "name"}, which a
    # clientside callback below picks up to open /peartree/<uuid> in a
    # new browser window.
    @callback(
        Output("within-run-view-mcc-store", "data"),
        Output("mcc-registry-store", "data", allow_duplicate=True),
        Output("within-run-selected-trees-store", "data", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
        Input("within-run-view-mcc", "n_clicks"),
        State("within-run-selected-trees-store", "data"),
        State("within-run-result-select", "value"),
        State("within-run-run-select", "value"),
        State("mds-result-store", "data"),
        prevent_initial_call=True,
    )
    def view_mcc_tree(n_clicks, selected_treenums, selected_key, selected_run, results):
        from ..logger import add_log, notif_id
        from .. import state as _state
        if not n_clicks or not selected_treenums:
            return no_update, no_update, no_update, no_update
        mds_result = _get_active_result(selected_key, results)
        if not mds_result or not selected_run:
            return no_update, no_update, no_update, no_update

        source_distmat = (mds_result.get("metadata") or {}).get("source_distmat")
        if not source_distmat:
            return no_update, no_update, no_update, dmc.Notification(
                title="MCC Error",
                message="No RF/snapshot data is associated with this MDS result.",
                color="red", action="show", autoClose=5000, id=notif_id())

        df_run, _ = _filter_to_run(mds_result, selected_run)
        if df_run is None:
            return no_update, no_update, no_update, no_update
        sel_df = df_run[df_run["treenum"].isin(selected_treenums)]
        tree_names = sel_df["tree"].tolist()
        if not tree_names:
            return no_update, no_update, no_update, dmc.Notification(
                title="MCC Error", message="No matching trees found.",
                color="red", action="show", autoClose=4000, id=notif_id())

        from ..db.tree_service import get_tree_service
        from ..mcc import assemble_mcc_nexus, extract_log_posterior

        tree_service = get_tree_service()
        tree_service.db_manager.flush()
        all_trees = tree_service.db_manager._trees
        matched = all_trees[all_trees["name"].isin(tree_names)].sort_values("id")
        if len(matched) == 0:
            return no_update, no_update, no_update, dmc.Notification(
                title="MCC Error",
                message="Selected trees not found in database. They may have been cleared.",
                color="red", action="show", autoClose=4000, id=notif_id())

        try:
            (nexus_bytes, mcc_row, log_clade_cred, counts,
             cols_in_mcc, missing_taxa) = assemble_mcc_nexus(
                matched, tree_service.db_manager, source_distmat,
            )
        except Exception as e:
            return no_update, no_update, no_update, dmc.Notification(
                title="MCC Error", message=str(e),
                color="red", action="show", autoClose=6000, id=notif_id())

        if missing_taxa:
            sample = ", ".join(sorted(missing_taxa)[:5])
            more = "…" if len(missing_taxa) > 5 else ""
            return no_update, no_update, no_update, dmc.Notification(
                title="MCC Error",
                message=(
                    f"Cannot align translate tables: taxa [{sample}{more}] "
                    "are present in some selected runs but not in the "
                    "canonical Translate block."
                ),
                color="red", action="show", autoClose=8000, id=notif_id())

        mcc_tree_name = mcc_row["name"]
        uid = _state.cache_mcc_tree(nexus_bytes)

        # Look up the MCC's treenum in this run's dataframe so the green
        # ring lands on the right point.
        mcc_in_df = df_run[df_run["tree"] == mcc_tree_name]
        mcc_treenum = (int(mcc_in_df.iloc[0]["treenum"])
                       if not mcc_in_df.empty else None)

        entry = _state.register_mcc(
            source_distmat=source_distmat,
            mode="Within",
            run=selected_run,
            uuid=uid,
            mcc_tree={
                "group": selected_run,
                "treenum": mcc_treenum,
                "tree_name": mcc_tree_name,
            },
            selection=[[selected_run, int(t)] for t in selected_treenums],
            log_clade_credibility=(None if log_clade_cred is None
                                   else float(log_clade_cred)),
            mcc_log_posterior=extract_log_posterior(mcc_row),
            tree_names=tree_names,
            counts=counts,
            cols_in_mcc=cols_in_mcc,
        )
        registered_name = entry["name"]
        add_log(
            f"Cached MCC tree '{mcc_tree_name}' (from {len(matched)} selected) "
            f"as {uid}; registered as {registered_name}"
        )
        notification = dmc.Notification(
            title="MCC Tree Ready",
            message=(
                f"MCC tree {registered_name} (from {len(matched)} selected) "
                "— opening in PearTree…"
            ),
            color="green", action="show", autoClose=4000, id=notif_id())
        # Clear the red selection ring once the MCC is registered.
        return ({"uuid": uid, "name": registered_name},
                _state.get_mcc_registry(),
                [],
                notification)

    # Clientside: in pywebview desktop mode call the Python-side JS API
    # to spawn a sibling native window; in ``--browser`` mode fall back
    # to a regular ``window.open`` that opens a new browser tab. Same
    # behaviour as the between-run tab.
    clientside_callback(
        """
        function(payload) {
            if (payload && payload.uuid) {
                const name = payload.name || '';
                if (window.pywebview && window.pywebview.api
                    && window.pywebview.api.open_peartree) {
                    window.pywebview.api.open_peartree(payload.uuid, name);
                } else {
                    const url = '/peartree/' + payload.uuid
                              + '?name=' + encodeURIComponent(name);
                    const features = 'width=1200,height=800,resizable=yes,scrollbars=yes';
                    window.open(url, 'peartree-' + payload.uuid, features);
                }
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("within-run-view-mcc-store", "data", allow_duplicate=True),
        Input("within-run-view-mcc-store", "data"),
        prevent_initial_call=True,
    )

    # ------ export PDF ------
    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("within-run-export-pdf", "n_clicks"),
        State("within-run-graph", "figure"),
        prevent_initial_call=True,
    )
    def export_within_run_pdf(n_clicks, fig_dict):
        from ..logger import add_log, notif_id
        if not n_clicks or not fig_dict:
            return no_update
        from ._helpers import _save_file_dialog
        path = _save_file_dialog(default_filename="within_run.pdf")
        if not path:
            return no_update
        fig = go.Figure(fig_dict)
        fig.update_layout(template=get_template())
        fig.write_image(path, width=1800, height=500, scale=2)
        add_log(f"Exported within-run plot to {path}")
        return dmc.Notification(title="PDF Exported", message=f"Saved to {path}",
                                color="green", action="show", autoClose=3000,
                                id=notif_id())

    # ------ main figure rebuild — selection / show_lines / color_gradient
    # are States; each has its own targeted callback below. ------
    @callback(
        Output("within-run-graph", "figure", allow_duplicate=True),
        Input("within-run-dim-x", "value"),
        Input("within-run-dim-y", "value"),
        Input("within-run-dim-z", "value"),
        Input("within-run-treenum-range-store", "data"),
        State("within-run-color-gradient", "checked"),
        State("within-run-show-lines", "checked"),
        State("within-run-selected-trees-store", "data"),
        State("within-run-result-select", "value"),
        State("within-run-run-select", "value"),
        State("mds-result-store", "data"),
        State("within-run-dragmode", "value"),
        State("mcc-registry-store", "data"),
        prevent_initial_call=True,
    )
    def auto_update_plot(dim_x, dim_y, dim_z, treenum_range,
                         color_gradient, show_lines, selected, selected_key,
                         selected_run, results, dragmode, mcc_registry):
        mds_result = _get_active_result(selected_key, results)
        if not mds_result or not selected_run or not all([dim_x, dim_y, dim_z]):
            return no_update
        df_run, axis_ranges = _filter_to_run(mds_result, selected_run)
        if df_run is None:
            return no_update
        selected_set = set(selected) if selected else None

        # Only re-pin to the global axis extent when the user changed an axis
        # dimension. For everything else we leave ranges off so Plotly's
        # uirevision can keep the user's current zoom — and because trace
        # count is constant, uirevision works for these rebuilds.
        triggered = ctx.triggered_id
        dim_triggered = triggered in (
            "within-run-dim-x", "within-run-dim-y", "within-run-dim-z",
        )
        fig_axis_ranges = axis_ranges if dim_triggered else None

        mcc_entries = _filter_mccs_for_within(
            mcc_registry, results, selected_key, selected_run)
        return _make_within_run_figure(
            df_run, dim_x, dim_y, dim_z, show_lines,
            selected_treenums=selected_set,
            treenum_range=treenum_range,
            color_gradient=color_gradient,
            dragmode=dragmode or "zoom",
            axis_ranges=fig_axis_ranges,
            mcc_entries=mcc_entries,
        )

    # ------ selection store change → patch only the last 3 traces ------
    @callback(
        Output("within-run-graph", "figure", allow_duplicate=True),
        Input("within-run-selected-trees-store", "data"),
        State("within-run-graph", "figure"),
        State("within-run-dim-x", "value"),
        State("within-run-dim-y", "value"),
        State("within-run-dim-z", "value"),
        State("within-run-result-select", "value"),
        State("within-run-run-select", "value"),
        State("mds-result-store", "data"),
        prevent_initial_call=True,
    )
    def update_selection_overlay(selected, current_fig,
                                 dim_x, dim_y, dim_z,
                                 selected_key, selected_run, results):
        if not current_fig:
            return no_update
        # Bail if the figure isn't the expected 9-trace shape — patching by
        # index against an unexpected layout would clobber the wrong traces.
        n_traces = len(current_fig.get("data", []))
        if n_traces != N_TRACES:
            return no_update
        if not all([dim_x, dim_y, dim_z]) or not selected_run:
            return no_update
        mds_result = _get_active_result(selected_key, results)
        if not mds_result:
            return no_update
        df_run, _ = _filter_to_run(mds_result, selected_run)
        if df_run is None:
            return no_update

        sel_data = _selection_panels_data(df_run, selected, dim_x, dim_y, dim_z)
        patch = Patch()
        # Clear the dashed box/lasso rectangle Plotly persists in
        # layout.selections after a box-select — without this the rectangle
        # sticks around once the selection lands.
        patch["layout"]["selections"] = []
        # Plotly also stamps selectedpoints on every trace inside the box,
        # which kicks in unselected-marker styling (default opacity ≈ 0.2)
        # on points outside the latest box. Reset to None so Plotly treats
        # all traces as "no active selection" — the red-outline overlay
        # traces are the only thing that should mark which trees are picked.
        for i in range(n_traces):
            patch["data"][i]["selectedpoints"] = None
        for offset, d in zip(SELECTION_OVERLAY_OFFSETS, sel_data):
            idx = n_traces + offset
            patch["data"][idx]["x"] = d["x"]
            patch["data"][idx]["y"] = d["y"]
            patch["data"][idx]["customdata"] = d["customdata"]
        return patch

    # ------ MCC registry change → patch only the green overlays ------
    @callback(
        Output("within-run-graph", "figure", allow_duplicate=True),
        Input("mcc-registry-store", "data"),
        State("within-run-graph", "figure"),
        State("within-run-dim-x", "value"),
        State("within-run-dim-y", "value"),
        State("within-run-dim-z", "value"),
        State("within-run-result-select", "value"),
        State("within-run-run-select", "value"),
        State("mds-result-store", "data"),
        prevent_initial_call=True,
    )
    def update_mcc_overlay(mcc_registry, current_fig, dim_x, dim_y, dim_z,
                           selected_key, selected_run, results):
        if not current_fig:
            return no_update
        n_traces = len(current_fig.get("data", []))
        if n_traces != N_TRACES:
            return no_update
        if not all([dim_x, dim_y, dim_z]) or not selected_run:
            return no_update
        mds_result = _get_active_result(selected_key, results)
        if not mds_result:
            return no_update
        df_run, _ = _filter_to_run(mds_result, selected_run)
        if df_run is None:
            return no_update

        mcc_entries = _filter_mccs_for_within(
            mcc_registry, results, selected_key, selected_run)
        mcc_data = _mcc_panels_data(df_run, mcc_entries, dim_x, dim_y, dim_z)
        patch = Patch()
        for offset, d in zip(MCC_OVERLAY_OFFSETS, mcc_data):
            idx = n_traces + offset
            patch["data"][idx]["x"] = d["x"]
            patch["data"][idx]["y"] = d["y"]
            patch["data"][idx]["customdata"] = d["customdata"]
        return patch

    # ------ color_gradient toggle → full rebuild, zoom pinned from layout ------
    # Patching the marker (whole-dict or per-property) ran into Plotly
    # react/merge edge cases: stale colorscale/cmin/cmax leaked across a
    # toggle off → on round-trip, leaving the wrong palette. Cleanest fix is
    # to rebuild the figure (so the marker dict is constructed from scratch
    # exactly like the initial render) and pin the new axes to whatever range
    # the user had zoomed to — read straight out of current_fig.layout.
    @callback(
        Output("within-run-graph", "figure", allow_duplicate=True),
        Input("within-run-color-gradient", "checked"),
        State("within-run-graph", "figure"),
        State("within-run-dim-x", "value"),
        State("within-run-dim-y", "value"),
        State("within-run-dim-z", "value"),
        State("within-run-treenum-range-store", "data"),
        State("within-run-show-lines", "checked"),
        State("within-run-selected-trees-store", "data"),
        State("within-run-result-select", "value"),
        State("within-run-run-select", "value"),
        State("mds-result-store", "data"),
        State("within-run-dragmode", "value"),
        State("mcc-registry-store", "data"),
        prevent_initial_call=True,
    )
    def update_color_gradient(color_gradient, current_fig,
                              dim_x, dim_y, dim_z, treenum_range, show_lines,
                              selected, selected_key, selected_run,
                              results, dragmode, mcc_registry):
        if (not current_fig or not all([dim_x, dim_y, dim_z])
                or not selected_run):
            return no_update
        mds_result = _get_active_result(selected_key, results)
        if not mds_result:
            return no_update
        df_run, axis_ranges = _filter_to_run(mds_result, selected_run)
        if df_run is None:
            return no_update
        selected_set = set(selected) if selected else None

        # Capture the user's current zoom from each axis. dcc.Graph round-trips
        # user-driven range changes back into layout.{x,y}axis*.range, so a
        # State read here gives us the exact viewport the user is looking at.
        layout = current_fig.get("layout", {}) or {}
        user_ranges = {}
        for axis_key in ("xaxis", "yaxis", "xaxis2", "yaxis2",
                         "xaxis3", "yaxis3"):
            rng = (layout.get(axis_key) or {}).get("range")
            if rng is not None:
                user_ranges[axis_key] = rng

        mcc_entries = _filter_mccs_for_within(
            mcc_registry, results, selected_key, selected_run)
        fig = _make_within_run_figure(
            df_run, dim_x, dim_y, dim_z, show_lines,
            selected_treenums=selected_set,
            treenum_range=treenum_range,
            color_gradient=color_gradient,
            dragmode=dragmode or "zoom",
            axis_ranges=axis_ranges,  # global extent — user_ranges may override
            mcc_entries=mcc_entries,
        )
        for axis_key, rng in user_ranges.items():
            getattr(fig.layout, axis_key).range = rng
        return fig

    # ------ show_lines toggle → patch only the in-range trace mode ------
    # Doing this as a Patch (rather than letting auto_update_plot rebuild the
    # whole figure) keeps the user's zoom intact across the toggle. Plotly's
    # uirevision is unreliable when the entire figure dict is replaced via
    # dcc.Graph.figure, so we avoid the rebuild on this trigger.
    @callback(
        Output("within-run-graph", "figure", allow_duplicate=True),
        Input("within-run-show-lines", "checked"),
        State("within-run-graph", "figure"),
        prevent_initial_call=True,
    )
    def update_show_lines(show_lines, current_fig):
        if not current_fig:
            return no_update
        n_traces = len(current_fig.get("data", []))
        if n_traces != N_TRACES:
            return no_update
        mode = "lines+markers" if show_lines else "markers"
        patch = Patch()
        # In-range traces sit at indices 1, 3, 5 (every 2nd trace, after each
        # panel's out-of-range trace). Out-of-range and selection overlays
        # are always markers-only and don't react to the lines checkbox.
        for idx in (1, 3, 5):
            patch["data"][idx]["mode"] = mode
        return patch

    # ------ reset zoom button ------
    @callback(
        Output("within-run-graph", "figure", allow_duplicate=True),
        Input("within-run-reset-button", "n_clicks"),
        State("within-run-dim-x", "value"),
        State("within-run-dim-y", "value"),
        State("within-run-dim-z", "value"),
        State("within-run-treenum-range-store", "data"),
        State("within-run-show-lines", "checked"),
        State("within-run-color-gradient", "checked"),
        State("within-run-result-select", "value"),
        State("within-run-run-select", "value"),
        State("mds-result-store", "data"),
        State("within-run-selected-trees-store", "data"),
        State("within-run-dragmode", "value"),
        State("mcc-registry-store", "data"),
        prevent_initial_call=True,
    )
    def reset_axes(n_clicks, dim_x, dim_y, dim_z, treenum_range,
                   show_lines, color_gradient, selected_key, selected_run,
                   results, selected, dragmode, mcc_registry):
        mds_result = _get_active_result(selected_key, results)
        if (not n_clicks or not mds_result or not selected_run
                or not all([dim_x, dim_y, dim_z])):
            return no_update
        df_run, axis_ranges = _filter_to_run(mds_result, selected_run)
        if df_run is None:
            return no_update
        selected_set = set(selected) if selected else None

        mcc_entries = _filter_mccs_for_within(
            mcc_registry, results, selected_key, selected_run)
        fig = _make_within_run_figure(
            df_run, dim_x, dim_y, dim_z, show_lines,
            selected_treenums=selected_set,
            treenum_range=treenum_range,
            color_gradient=color_gradient,
            dragmode=dragmode or "zoom",
            axis_ranges=axis_ranges,
            mcc_entries=mcc_entries,
        )
        # Force a fresh uirevision so reset *does* throw away the user's zoom.
        fig.update_layout(uirevision=f"reset-{n_clicks}")
        return fig
