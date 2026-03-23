from dash import dcc, html, callback, Input, Output, State, no_update, ctx
import dash_mantine_components as dmc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from ..logger import add_log
from ..db.tree_service import get_tree_service


TREETRACER_BLUE = "#228be6"


def _make_within_run_figure(df, x, y, z, show_lines=True,
                            highlighted_treenum=None, treenum_range=None,
                            color_gradient=True):
    """Build 3 linked 2D scatterplots in subplots with matched axes."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f"{x} vs {y}", f"{x} vs {z}", f"{y} vs {z}"],
        horizontal_spacing=0.06,
    )

    mode = "lines+markers" if show_lines else "markers"
    colorscale = "Blues"

    # Split into in-range and out-of-range points
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
        # Out-of-range points: greyed out, no lines, not clickable
        if len(df_out) > 0:
            fig.add_trace(go.Scatter(
                x=df_out[xcol].values, y=df_out[ycol].values,
                mode="markers",
                marker=dict(color="lightgrey", size=5, opacity=0.4),
                hoverinfo="skip",
                showlegend=False,
                customdata=df_out[["treenum"]].values.tolist(),
            ), row=row, col=col)

        # In-range points: colored by treenum gradient or flat color
        if len(df_in) > 0:
            if color_gradient:
                marker_dict = dict(
                    color=df_in["treenum"].values, colorscale=colorscale, size=7,
                    colorbar=dict(title="Tree #", x=1.02, len=0.9) if show_cb else None,
                    showscale=show_cb,
                )
            else:
                marker_dict = dict(color=TREETRACER_BLUE, size=7)

            fig.add_trace(go.Scatter(
                x=df_in[xcol].values, y=df_in[ycol].values,
                mode=mode,
                marker=marker_dict,
                line=dict(color="rgba(120,120,120,0.4)", width=1),
                hovertemplate="Tree: %{customdata[0]}<extra></extra>",
                customdata=df_in[["treenum"]].values.tolist(),
                showlegend=False,
            ), row=row, col=col)

        # Red-outlined highlight marker for the selected point
        if highlighted_treenum is not None:
            sel = df[df["treenum"] == highlighted_treenum]
            if len(sel) > 0:
                fig.add_trace(go.Scatter(
                    x=sel[xcol].values, y=sel[ycol].values,
                    mode="markers",
                    marker=dict(
                        size=14,
                        color="rgba(0,0,0,0)",
                        line=dict(color="red", width=3),
                    ),
                    customdata=sel[["treenum"]].values.tolist(),
                    hovertemplate=f"Tree: {highlighted_treenum}<extra>selected</extra>",
                    showlegend=False,
                ), row=row, col=col)

    # Sync axes by shared dimension:
    # Plot 1: xaxis (X), yaxis (Y)
    # Plot 2: xaxis2 (X), yaxis2 (Z)
    # Plot 3: xaxis3 (Y), yaxis3 (Z)
    fig.update_layout(
        xaxis2=dict(matches='x'),    # plot2 x = plot1 x (X dim)
        xaxis3=dict(matches='y'),    # plot3 x = plot1 y (Y dim)
        yaxis3=dict(matches='y2'),   # plot3 y = plot2 y (Z dim)
    )

    # Axis labels
    fig.update_xaxes(title_text=x, row=1, col=1)
    fig.update_yaxes(title_text=y, row=1, col=1)
    fig.update_xaxes(title_text=x, row=1, col=2)
    fig.update_yaxes(title_text=z, row=1, col=2)
    fig.update_xaxes(title_text=y, row=1, col=3)
    fig.update_yaxes(title_text=z, row=1, col=3)

    fig.update_layout(
        template="simple_white",
        margin=dict(l=50, r=40, t=40, b=45),
        dragmode="zoom",
        uirevision="within-run",
    )

    return fig


def register_within_run_callbacks():
    # Populate run selector when files are loaded
    @callback(
        Output("within-run-select", "data"),
        Output("within-run-select", "value"),
        Input("tree-offset-store", "data"),
        State("within-run-select", "value"),
    )
    def populate_run_selector(stored_summaries, current_value):
        if not stored_summaries:
            return [], None
        options = [{"value": f, "label": f} for f in stored_summaries.keys()]
        if current_value and current_value in stored_summaries:
            return options, current_value
        return options, None

    # Show run info and enable compute button when a run is selected
    @callback(
        Output("within-run-info", "children", allow_duplicate=True),
        Output("within-run-compute-button", "disabled"),
        Input("within-run-select", "value"),
        State("tree-offset-store", "data"),
        prevent_initial_call=True,
    )
    def display_run_info(selected_file, stored_summaries):
        if not selected_file or not stored_summaries:
            return html.Div(), True
        summary = stored_summaries.get(selected_file, {})
        if not summary:
            return html.Div(), True
        info = dmc.Group([
            dmc.Badge(f"File: {selected_file}", variant="light", color="blue", size="lg"),
            dmc.Badge(f"Taxa: {summary.get('n_taxa', 0)}", variant="light", color="teal", size="lg"),
            dmc.Badge(f"Trees: {summary.get('total_trees', 0)}", variant="light", color="grape", size="lg"),
        ], gap="sm")
        return info, False

    # Compute RF + MDS for a single run
    @callback(
        Output("within-run-mds-store", "data"),
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
        Output("within-run-min-range", "max"),
        Output("notifications-container", "children", allow_duplicate=True),
        Output("within-run-graph", "figure"),
        Input("within-run-compute-button", "n_clicks"),
        State("within-run-select", "value"),
        State("tree-offset-store", "data"),
        prevent_initial_call=True,
    )
    def compute_within_run(n_clicks, selected_file, stored_summaries):
        _no = (no_update,) * 15
        if not n_clicks or not selected_file or not stored_summaries:
            return _no

        summary = stored_summaries.get(selected_file, {})
        total_trees = summary.get("total_trees", 0)
        sample_size = total_trees

        add_log(f"Within-run analysis: {selected_file}, {sample_size} trees")
        tree_service = get_tree_service()

        sample = tree_service.get_sample_for_analysis(
            file_sources=[selected_file],
            sample_size=sample_size,
            strategy="random",
        )
        sampled_trees = sample["trees"]
        add_log(f"Within-run: retrieved {len(sampled_trees)} trees")

        def _err(msg, title="Within-run Error"):
            """Return error tuple with notification at position 13 (0-indexed)."""
            out = [no_update] * 15
            out[13] = dmc.Notification(title=title, message=msg, color="red",
                                       action="show", autoClose=6000, id="within-run-notification")
            return tuple(out)

        if len(sampled_trees) < 2:
            msg = "Not enough trees for within-run analysis."
            add_log(msg, "ERROR")
            return _err(msg)

        names = [t["name"] for t in sampled_trees]
        newicks = tree_service.prepare_trees_for_rf_analysis(sampled_trees)
        tmap = tree_service.db_manager.get_translate_map(selected_file)
        translate_maps = [tmap] if tmap else []
        map_indices = [0] * len(sampled_trees)

        try:
            import time as _time
            t0 = _time.time()
            from ..rf.rf import rf_distance_from_newicks
            result_names, matrix = rf_distance_from_newicks(
                names, newicks, translate_maps, map_indices, rooted=False)
            rf_elapsed = _time.time() - t0
            add_log(f"Within-run RF: {len(result_names)} trees in {rf_elapsed:.2f}s")
        except Exception as e:
            msg = f"Within-run RF failed: {e}"
            add_log(msg, "ERROR")
            return _err(msg, "RF Error")

        n = len(result_names)
        dist_array = np.array(matrix, dtype=float)

        try:
            t0 = _time.time()
            from ..rf.mds import compute_mds
            n_components = min(6, n - 1)
            embedding = compute_mds(dist_array, n_components=n_components, algorithm="pcoa")
            mds_elapsed = _time.time() - t0
            add_log(f"Within-run MDS: {n_components} components in {mds_elapsed:.2f}s")
        except Exception as e:
            msg = f"Within-run MDS failed: {e}"
            add_log(msg, "ERROR")
            return _err(msg, "MDS Error")

        mdscols = [f"MDS{i+1}" for i in range(n_components)]
        mds_df = pd.DataFrame(embedding, columns=mdscols)
        mds_df["tree"] = result_names
        mds_df["treenum"] = range(1, n + 1)

        mds_result = {
            "file": selected_file,
            "dimensions": mdscols,
            "n_trees": n,
            "data": mds_df.to_dict("records"),
        }

        z_default = mdscols[2] if len(mdscols) > 2 else mdscols[0]
        dim_options = [{"value": col, "label": col} for col in mdscols]
        marks = [{"value": max(1, round(n * i / 10)), "label": str(max(1, round(n * i / 10)))}
                 for i in range(11)]

        fig = _make_within_run_figure(mds_df, mdscols[0], mdscols[1], z_default,
                                      treenum_range=[1, n])

        notification = dmc.Notification(
            title="Within-run Analysis Complete",
            message=(f"RF ({rf_elapsed:.2f}s) + MDS ({mds_elapsed:.2f}s) for "
                     f"{n} trees from {selected_file}."),
            color="green", action="show", autoClose=4000,
            id="within-run-notification",
        )

        return (
            mds_result,                    # within-run-mds-store
            {"display": "block"},          # within-run-controls-paper style
            dim_options, mdscols[0],       # dim-x data, value
            dim_options, mdscols[1],       # dim-y data, value
            dim_options, z_default,        # dim-z data, value
            1, n, [1, n], marks,           # slider min, max, value, marks
            n,                             # min-range max
            notification,                  # notifications
            fig,                           # graph figure
        )

    # Pipe slider value to the range store
    @callback(
        Output("within-run-treenum-range-store", "data"),
        Input("within-run-treenum-slider", "value"),
        prevent_initial_call=True,
    )
    def update_treenum_range(slider_value):
        if not slider_value:
            return no_update
        return slider_value

    # Update minRange on the slider when the NumberInput changes
    @callback(
        Output("within-run-treenum-slider", "minRange"),
        Input("within-run-min-range", "value"),
        prevent_initial_call=True,
    )
    def update_min_range(min_range):
        try:
            min_range = int(min_range) if min_range else 1
        except (ValueError, TypeError):
            return no_update
        if min_range < 1:
            return no_update
        return min_range

    # Click on a point → store its treenum (toggle on re-click)
    @callback(
        Output("within-run-highlight-store", "data"),
        Input("within-run-graph", "clickData"),
        State("within-run-highlight-store", "data"),
        State("within-run-mds-store", "data"),
        State("within-run-treenum-range-store", "data"),
        prevent_initial_call=True,
    )
    def store_highlight(click_data, current_highlight, mds_result, treenum_range):
        if not click_data or not mds_result:
            return no_update
        point = click_data["points"][0]

        clicked_treenum = None

        # Try customdata first
        if point.get("customdata"):
            try:
                clicked_treenum = int(point["customdata"][0])
            except (ValueError, TypeError, IndexError):
                pass

        # Fall back to marker.color (works when color gradient is on)
        if clicked_treenum is None and "marker.color" in point:
            try:
                clicked_treenum = int(point["marker.color"])
            except (ValueError, TypeError):
                pass

        # Last resort: use pointIndex to look up treenum from the data
        if clicked_treenum is None and "pointIndex" in point:
            df = pd.DataFrame(mds_result["data"])
            if treenum_range:
                in_mask = (df["treenum"] >= treenum_range[0]) & (df["treenum"] <= treenum_range[1])
            else:
                in_mask = pd.Series(True, index=df.index)

            curve = point.get("curveNumber", 0)
            has_out = (~in_mask).any()
            traces_per_panel = (1 if has_out else 0) + 1 + (1 if current_highlight else 0)
            local_curve = curve % traces_per_panel if traces_per_panel > 0 else 0

            idx = point["pointIndex"]
            if has_out and local_curve == 0:
                out_df = df[~in_mask].reset_index(drop=True)
                if idx < len(out_df):
                    clicked_treenum = int(out_df.iloc[idx]["treenum"])
            elif local_curve == (1 if has_out else 0):
                in_df = df[in_mask].reset_index(drop=True)
                if idx < len(in_df):
                    clicked_treenum = int(in_df.iloc[idx]["treenum"])

        if clicked_treenum is None:
            return no_update
        if current_highlight == clicked_treenum:
            return None
        return clicked_treenum

    # Auto-update plot when any control changes
    @callback(
        Output("within-run-graph", "figure", allow_duplicate=True),
        Input("within-run-dim-x", "value"),
        Input("within-run-dim-y", "value"),
        Input("within-run-dim-z", "value"),
        Input("within-run-treenum-range-store", "data"),
        Input("within-run-show-lines", "checked"),
        Input("within-run-color-gradient", "checked"),
        Input("within-run-highlight-store", "data"),
        State("within-run-mds-store", "data"),
        prevent_initial_call=True,
    )
    def auto_update_plot(dim_x, dim_y, dim_z, treenum_range,
                         show_lines, color_gradient, highlighted, mds_result):
        if not mds_result or not all([dim_x, dim_y, dim_z]):
            return no_update

        df = pd.DataFrame(mds_result["data"])

        return _make_within_run_figure(df, dim_x, dim_y, dim_z, show_lines,
                                       highlighted_treenum=highlighted,
                                       treenum_range=treenum_range,
                                       color_gradient=color_gradient)

    # Reset Axes button — force zoom reset by changing uirevision
    @callback(
        Output("within-run-graph", "figure", allow_duplicate=True),
        Input("within-run-reset-button", "n_clicks"),
        State("within-run-dim-x", "value"),
        State("within-run-dim-y", "value"),
        State("within-run-dim-z", "value"),
        State("within-run-treenum-range-store", "data"),
        State("within-run-show-lines", "checked"),
        State("within-run-color-gradient", "checked"),
        State("within-run-mds-store", "data"),
        State("within-run-highlight-store", "data"),
        prevent_initial_call=True,
    )
    def reset_axes(n_clicks, dim_x, dim_y, dim_z, treenum_range,
                   show_lines, color_gradient, mds_result, highlighted):
        if not n_clicks or not mds_result or not all([dim_x, dim_y, dim_z]):
            return no_update

        df = pd.DataFrame(mds_result["data"])

        fig = _make_within_run_figure(df, dim_x, dim_y, dim_z,
                                      show_lines, highlighted_treenum=highlighted,
                                      treenum_range=treenum_range,
                                      color_gradient=color_gradient)
        fig.update_layout(uirevision=f"reset-{n_clicks}")
        return fig
