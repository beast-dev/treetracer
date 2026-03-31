from dash import dcc, html, callback, Input, Output, State, no_update, ctx
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


TREETRACER_BLUE = "#228be6"


def _make_within_run_figure(df, x, y, z, show_lines=True,
                            selected_treenums=None, treenum_range=None,
                            color_gradient=True, dragmode="zoom"):
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
                customdata=list(zip(df_out["treenum"], df_out["tree"].str.split("/").str[-1].str.strip())),
                selected=dict(marker=dict(opacity=0.4)),
                unselected=dict(marker=dict(opacity=0.4)),
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
                hovertemplate="Tree #%{customdata[0]}: %{customdata[1]}<extra></extra>",
                customdata=list(zip(df_in["treenum"], df_in["tree"].str.split("/").str[-1].str.strip())),
                showlegend=False,
                selected=dict(marker=dict(opacity=1)),
                unselected=dict(marker=dict(opacity=1)),
            ), row=row, col=col)

        # Red-outlined markers for selected trees
        if selected_treenums:
            sel = df[df["treenum"].isin(selected_treenums)]
            if len(sel) > 0:
                fig.add_trace(go.Scatter(
                    x=sel[xcol].values, y=sel[ycol].values,
                    mode="markers",
                    marker=dict(
                        size=12,
                        color="rgba(0,0,0,0)",
                        line=dict(color="red", width=1.5),
                    ),
                    customdata=list(zip(sel["treenum"], sel["tree"].str.split("/").str[-1].str.strip())),
                    hovertemplate="Tree #%{customdata[0]}: %{customdata[1]}<extra>selected</extra>",
                    showlegend=False,
                ), row=row, col=col)

    # Sync axes by shared dimension
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

    fig.update_layout(
        template="simple_white",
        margin=dict(l=50, r=40, t=40, b=45),
        dragmode=dragmode,
        uirevision="within-run",
    )

    return fig


def _get_active_result(selected_key, results):
    """Helper to resolve the active MDS result from store."""
    if not selected_key or not results or selected_key not in results:
        return None
    return results[selected_key]


def register_within_run_callbacks():

    # Populate result selector from computed within-run MDS results
    @callback(
        Output("within-run-result-select", "data"),
        Output("within-run-result-select", "value"),
        Input("within-run-mds-results-store", "data"),
        State("within-run-result-select", "value"),
    )
    def populate_result_selector(results, current_value):
        if not results:
            return [], None
        options = [
            {"value": k, "label": f"{v.get('file', k)} ({v['n_trees']} trees) [{v.get('source_distmat', '?')}]"}
            for k, v in results.items()
        ]
        if current_value and current_value in results:
            return options, current_value
        return options, list(results.keys())[-1]

    # When a result is selected, set up controls and render initial figure
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
        State("within-run-mds-results-store", "data"),
        prevent_initial_call=True,
    )
    def load_result_for_visualization(selected_key, results):
        result = _get_active_result(selected_key, results)
        if result is None:
            return (no_update,) * 17

        mdscols = result["dimensions"]
        n = result["n_trees"]
        mds_df = pd.DataFrame(result["data"])

        z_default = mdscols[2] if len(mdscols) > 2 else mdscols[0]
        dim_options = [{"value": col, "label": col} for col in mdscols]
        marks = [{"value": max(1, round(n * i / 10)), "label": str(max(1, round(n * i / 10)))}
                 for i in range(11)]

        fig = _make_within_run_figure(mds_df, mdscols[0], mdscols[1], z_default,
                                      treenum_range=[1, n])

        info = dmc.Group([
            dmc.Badge(f"File: {result.get('file', '?')}", variant="light", color="blue", size="lg"),
            dmc.Badge(f"Source: {result.get('source_distmat', '?')}", variant="light", color="teal", size="lg"),
            dmc.Badge(f"Trees: {n}", variant="light", color="grape", size="lg"),
        ], gap="sm")

        return (
            {"display": "flex"},           # controls row visible
            dim_options, mdscols[0],       # dim-x
            dim_options, mdscols[1],       # dim-y
            dim_options, z_default,        # dim-z
            1, n, [1, n], marks,           # slider
            info,                          # info badges
            fig,                           # graph
            [],                            # clear selection
            [1, n],                        # reset treenum range
            True,                          # disable animation interval
            DashIconify(icon="tabler:player-play-filled", width=18),  # reset play button
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

    # Auto-filter selection when range changes — remove trees outside the new window
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
            return no_update  # nothing changed
        return filtered

    # Sync slider minRange to the window size input
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

    # ------ SLIDING WINDOW ANIMATION ------

    # Play/Pause toggle
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
            # Start playing: snap slider to [min, min+window] and enable interval
            w = int(window_size) if window_size else 100
            return (
                False,
                DashIconify(icon="tabler:player-pause-filled", width=18),
                [slider_min, min(slider_min + w, slider_max)],
            )
        else:
            # Pause
            return (
                True,
                DashIconify(icon="tabler:player-play-filled", width=18),
                no_update,
            )

    # Advance the sliding window on each interval tick
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
            # Reached the end — stop and reset to start
            return (
                [slider_min, min(slider_min + w, slider_max)],
                True,
                DashIconify(icon="tabler:player-play-filled", width=18),
            )

        # Clamp end to slider max
        if new_end > slider_max:
            new_end = slider_max

        return [new_start, new_end], no_update, no_update

    # ------ UNIFIED SELECTION (click + box + lasso) ------

    # Click on a point → toggle it in the selection set
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

        # Ignore clicks on out-of-range (greyed out) points
        if treenum_range:
            if clicked_treenum < treenum_range[0] or clicked_treenum > treenum_range[1]:
                return no_update

        selected = set(current_selection or [])
        if clicked_treenum in selected:
            selected.discard(clicked_treenum)
        else:
            selected.add(clicked_treenum)
        return sorted(selected)

    # Box/lasso selection → add selected points to the selection set
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

        # Only select points within the current range slider window
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

    # Clear selection
    @callback(
        Output("within-run-selected-trees-store", "data", allow_duplicate=True),
        Input("within-run-clear-selection", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_selection(n_clicks):
        if not n_clicks:
            return no_update
        return []

    # Dragmode toggle (zoom / box select / lasso)
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

    # Selection info display + export button enable
    @callback(
        Output("within-run-selection-info", "children"),
        Output("within-run-export-trees", "disabled"),
        Input("within-run-selected-trees-store", "data"),
    )
    def update_selection_info(selected):
        if not selected:
            return html.Div(), True
        return dmc.Badge(f"Selected: {len(selected)} trees", color="red", variant="light", size="lg"), False

    # Export selected trees as NEXUS .trees file
    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("within-run-export-trees", "n_clicks"),
        State("within-run-selected-trees-store", "data"),
        State("within-run-result-select", "value"),
        State("within-run-mds-results-store", "data"),
        prevent_initial_call=True,
    )
    def export_selected_trees(n_clicks, selected_treenums, selected_key, results):
        if not n_clicks or not selected_treenums:
            return no_update

        mds_result = _get_active_result(selected_key, results)
        if not mds_result:
            return no_update

        # Map treenums back to tree names
        mds_df = pd.DataFrame(mds_result["data"])
        sel_df = mds_df[mds_df["treenum"].isin(selected_treenums)]
        tree_names = sel_df["tree"].tolist()

        if not tree_names:
            return dmc.Notification(title="Export Error", message="No matching trees found.",
                                    color="red", action="show", autoClose=4000, id="export-trees-notification")

        # Look up trees in the tree manager and export
        from ..db.tree_service import get_tree_service
        from ._helpers import _save_file_dialog

        tree_service = get_tree_service()
        tree_service.db_manager.flush()
        all_trees = tree_service.db_manager._trees
        matched = all_trees[all_trees["name"].isin(tree_names)].sort_values("id")

        if len(matched) == 0:
            return dmc.Notification(title="Export Error",
                                    message="Selected trees not found in database. They may have been cleared.",
                                    color="red", action="show", autoClose=4000, id="export-trees-notification")

        file_source = matched["file_source"].iloc[0]
        path = _save_file_dialog(default_filename=f"selected_{len(matched)}_trees.trees")
        if not path:
            return no_update

        # Write NEXUS file
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
                                    color="red", action="show", autoClose=6000, id="export-trees-notification")

        from ..logger import add_log
        add_log(f"Exported {len(matched)} selected trees to {path}")
        return dmc.Notification(title="Trees Exported",
                                message=f"Exported {len(matched)} trees to {path}",
                                color="green", action="show", autoClose=4000, id="export-trees-notification")

    # ------ PLOT RENDERING ------

    # Auto-update plot when any control changes
    @callback(
        Output("within-run-graph", "figure", allow_duplicate=True),
        Input("within-run-dim-x", "value"),
        Input("within-run-dim-y", "value"),
        Input("within-run-dim-z", "value"),
        Input("within-run-treenum-range-store", "data"),
        Input("within-run-show-lines", "checked"),
        Input("within-run-color-gradient", "checked"),
        Input("within-run-selected-trees-store", "data"),
        State("within-run-result-select", "value"),
        State("within-run-mds-results-store", "data"),
        State("within-run-dragmode", "value"),
        prevent_initial_call=True,
    )
    def auto_update_plot(dim_x, dim_y, dim_z, treenum_range,
                         show_lines, color_gradient, selected,
                         selected_key, results, dragmode):
        mds_result = _get_active_result(selected_key, results)
        if not mds_result or not all([dim_x, dim_y, dim_z]):
            return no_update

        df = pd.DataFrame(mds_result["data"])
        selected_set = set(selected) if selected else None

        return _make_within_run_figure(df, dim_x, dim_y, dim_z, show_lines,
                                       selected_treenums=selected_set,
                                       treenum_range=treenum_range,
                                       color_gradient=color_gradient,
                                       dragmode=dragmode or "zoom")

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
        State("within-run-result-select", "value"),
        State("within-run-mds-results-store", "data"),
        State("within-run-selected-trees-store", "data"),
        State("within-run-dragmode", "value"),
        prevent_initial_call=True,
    )
    def reset_axes(n_clicks, dim_x, dim_y, dim_z, treenum_range,
                   show_lines, color_gradient, selected_key, results, selected, dragmode):
        mds_result = _get_active_result(selected_key, results)
        if not n_clicks or not mds_result or not all([dim_x, dim_y, dim_z]):
            return no_update

        df = pd.DataFrame(mds_result["data"])
        selected_set = set(selected) if selected else None

        fig = _make_within_run_figure(df, dim_x, dim_y, dim_z,
                                      show_lines, selected_treenums=selected_set,
                                      treenum_range=treenum_range,
                                      color_gradient=color_gradient,
                                      dragmode=dragmode or "zoom")
        fig.update_layout(uirevision=f"reset-{n_clicks}")
        return fig
