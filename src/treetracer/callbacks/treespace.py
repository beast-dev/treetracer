from dash import dcc, html, callback, Input, Output, State, no_update
import dash_mantine_components as dmc
import plotly.express as px
import pandas as pd

from ..logger import add_log
from ..state import get_mds_result
from ..plot_utils import make_plot_grid, add_trace_multiplot_interleaved


def _placeholder(text):
    return html.Div(
        text,
        style={
            "text-align": "center",
            "color": "#666",
            "font-size": "18px",
            "padding": "100px",
            "height": "calc(100vh - 280px)",
            "display": "flex",
            "align-items": "center",
            "justify-content": "center",
        },
    )


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
        # Default to most recently added.
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
        Output("plot-container", "children", allow_duplicate=True),
        Output("plot-button", "children", allow_duplicate=True),
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
            [_placeholder("No MDS result selected. Compute an MDS in the Compute tab.")],
            "Plot",
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

        return (
            plot_config,
            dim_options, mdscols[0],
            dim_options, mdscols[1],
            dim_options, z_default,
            MIN_TREENUM, MAX_TREENUM, [MIN_TREENUM, MAX_TREENUM], marks,
            info,
            {"display": "flex"},
            [_placeholder("Click 'Plot' to visualize data")],
            "Plot",
        )

    # Plot button — explicit user trigger so dropdown / slider changes don't
    # auto-rebuild the (heavy) multiplot.
    @callback(
        Output("plot-container", "children", allow_duplicate=True),
        Output("plot-button", "children", allow_duplicate=True),
        Input("plot-button", "n_clicks"),
        State("dim-x-select", "value"),
        State("dim-y-select", "value"),
        State("dim-z-select", "value"),
        State("treenum-slider", "value"),
        State("show-lines-checkbox", "checked"),
        State("plot-container", "children"),
        State("plot-config-store", "data"),
        prevent_initial_call=True,
    )
    def update_graph_on_button_click(n_clicks, dim_x, dim_y, dim_z, treenum_range,
                                     show_lines, current_plot, plot_config):
        if not n_clicks or not plot_config or not all([dim_x, dim_y, dim_z]):
            return no_update, no_update

        combined_df = pd.DataFrame(plot_config["combined_data"])

        filtered_dff = combined_df[
            (combined_df["treenum"] >= treenum_range[0])
            & (combined_df["treenum"] <= treenum_range[1])
        ]
        mds_selected = [dim_x, dim_y, dim_z]
        add_log(f"Plotting {len(filtered_dff)} trees (range {treenum_range[0]}-{treenum_range[1]}), dims: {mds_selected}")

        x, y, z = dim_x, dim_y, dim_z

        # Create new plot with filtered data. The interleaved variant splits
        # each group's points into chunks and stacks them by chunk-index so
        # no single run sits entirely on top of the others in the 2D panels.
        fig = make_plot_grid()
        add_trace_multiplot_interleaved(
            fig, filtered_dff, x, y, z, plot_config["groups"], plot_config["color_dict"],
            show_lines=show_lines,
        )

        # Try to preserve visibility settings if updating existing plot
        if (
            current_plot
            and len(current_plot) > 0
            and hasattr(current_plot[0], 'children')
            and hasattr(current_plot[0].children, 'figure')
        ):
            current_figure = current_plot[0].children.figure
            if (
                current_figure
                and "data" in current_figure
                and len(current_figure["data"]) == len(fig.data)
            ):
                for i in range(len(fig.data)):
                    if "visible" in current_figure["data"][i]:
                        fig.data[i].visible = current_figure["data"][i]["visible"]

        plot_component = dcc.Graph(figure=fig, id="graph",
                                   style={"height": "calc(100vh - 280px)"})
        return [plot_component], "Update Plot"
