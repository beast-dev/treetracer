from dash import dcc, html, callback, Input, Output, State, no_update, ctx
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


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
                customdata=list(zip(df_out["treenum"], df_out["tree"].str.split("/").str[-1].str.strip())),
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
        dragmode="zoom",
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
        Output("within-run-min-range", "max"),
        Output("within-run-info", "children"),
        Output("within-run-graph", "figure"),
        Output("within-run-highlight-store", "data", allow_duplicate=True),
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
            return (no_update,) * 18

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
            {"display": "block"},          # controls-paper style
            dim_options, mdscols[0],       # dim-x
            dim_options, mdscols[1],       # dim-y
            dim_options, z_default,        # dim-z
            1, n, [1, n], marks,           # slider
            n,                             # min-range max
            info,                          # info badges
            fig,                           # graph
            None,                          # reset highlight
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

    # Click on a point → store its treenum (toggle on re-click)
    @callback(
        Output("within-run-highlight-store", "data"),
        Input("within-run-graph", "clickData"),
        State("within-run-highlight-store", "data"),
        State("within-run-result-select", "value"),
        State("within-run-mds-results-store", "data"),
        State("within-run-treenum-range-store", "data"),
        prevent_initial_call=True,
    )
    def store_highlight(click_data, current_highlight, selected_key, results, treenum_range):
        mds_result = _get_active_result(selected_key, results)
        if not click_data or not mds_result:
            return no_update
        point = click_data["points"][0]

        clicked_treenum = None

        if point.get("customdata"):
            try:
                clicked_treenum = int(point["customdata"][0])
            except (ValueError, TypeError, IndexError):
                pass

        if clicked_treenum is None and "marker.color" in point:
            try:
                clicked_treenum = int(point["marker.color"])
            except (ValueError, TypeError):
                pass

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
        State("within-run-result-select", "value"),
        State("within-run-mds-results-store", "data"),
        prevent_initial_call=True,
    )
    def auto_update_plot(dim_x, dim_y, dim_z, treenum_range,
                         show_lines, color_gradient, highlighted,
                         selected_key, results):
        mds_result = _get_active_result(selected_key, results)
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
        State("within-run-result-select", "value"),
        State("within-run-mds-results-store", "data"),
        State("within-run-highlight-store", "data"),
        prevent_initial_call=True,
    )
    def reset_axes(n_clicks, dim_x, dim_y, dim_z, treenum_range,
                   show_lines, color_gradient, selected_key, results, highlighted):
        mds_result = _get_active_result(selected_key, results)
        if not n_clicks or not mds_result or not all([dim_x, dim_y, dim_z]):
            return no_update

        df = pd.DataFrame(mds_result["data"])

        fig = _make_within_run_figure(df, dim_x, dim_y, dim_z,
                                      show_lines, highlighted_treenum=highlighted,
                                      treenum_range=treenum_range,
                                      color_gradient=color_gradient)
        fig.update_layout(uirevision=f"reset-{n_clicks}")
        return fig
