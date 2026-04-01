from dash import dcc, html, callback, Input, Output, State, no_update
import dash_mantine_components as dmc
import plotly.express as px
import pandas as pd

from ..logger import add_log
from ..plot_utils import make_plot_grid, add_trace_multiplot


def register_treespace_callbacks():
    # Auto-generate plot config when MDS result changes
    @callback(
        Output("plot-config-store", "data"),
        Input("mds-result-store", "data"),
    )
    def generate_plot_config_from_mds(mds_results):
        if not mds_results:
            return {}

        # Use the most recently added MDS result
        last_key = list(mds_results.keys())[-1]
        mds_result = mds_results[last_key]
        if not mds_result.get("data"):
            return {}

        metadata = mds_result["metadata"]
        combined_df = pd.DataFrame(mds_result["data"])
        mdscols = metadata["dimensions"]
        groups = combined_df["group"].unique().tolist()
        group_colors = px.colors.qualitative.Dark24[:len(groups)]
        color_dict = {g: c for g, c in zip(groups, group_colors)}

        return {
            "combined_data": mds_result["data"],
            "mdscols": mdscols,
            "min_treenum": metadata["MIN_TREENUM"],
            "max_treenum": metadata["MAX_TREENUM"],
            "groups": groups,
            "color_dict": color_dict,
        }

    # Display plots when plot config is ready
    @callback(
        Output("plot-display", "children", allow_duplicate=True),
        Input("plot-config-store", "data"),
        prevent_initial_call=True,
    )
    def display_plots(plot_config):
        if not plot_config or not plot_config.get("combined_data"):
            return html.Div("No MDS result available. Compute MDS in the Compute tab.")

        plot_div = []
        combined_df = pd.DataFrame(plot_config["combined_data"])
        mdscols = plot_config["mdscols"]
        MIN_TREENUM = plot_config["min_treenum"]
        MAX_TREENUM = plot_config["max_treenum"]
        groups = plot_config["groups"]
        color_dict = plot_config["color_dict"]

        # Controls row with axis selects, slider, checkbox, and plot button
        dim_options = [{"value": col, "label": col} for col in mdscols]
        controls = dmc.Paper(
            dmc.Group(
                [
                    dmc.Select(
                        label="X", id="dim-x-select",
                        data=dim_options, value=mdscols[0],
                        size="xs", w=120,
                    ),
                    dmc.Select(
                        label="Y", id="dim-y-select",
                        data=dim_options, value=mdscols[1],
                        size="xs", w=120,
                    ),
                    dmc.Select(
                        label="Z", id="dim-z-select",
                        data=dim_options, value=mdscols[2],
                        size="xs", w=120,
                    ),
                    dmc.Stack(
                        [
                            dmc.Text("Tree Number Range:", size="sm", fw=500),
                            dmc.RangeSlider(
                                id="treenum-slider",
                                min=MIN_TREENUM,
                                max=MAX_TREENUM,
                                value=[MIN_TREENUM, MAX_TREENUM],
                                marks=[
                                    {"value": MIN_TREENUM, "label": str(MIN_TREENUM)},
                                    {"value": MAX_TREENUM, "label": str(MAX_TREENUM)},
                                ],
                                step=1,
                                styles={"label": {"top": "unset", "bottom": "-2rem"}},
                            ),
                        ],
                        gap="xs",
                        style={"flex": 1},
                    ),
                    dmc.Checkbox(
                        label="Show lines",
                        id="show-lines-checkbox",
                        checked=True,
                    ),
                    dmc.Button(
                        "Plot",
                        id="plot-button",
                        variant="filled",
                        color="blue",
                        size="md",
                    ),
                ],
                align="flex-end",
                gap="lg",
            ),
            withBorder=True,
            p="md",
            radius="sm",
            mb="sm",
        )

        plot_div.append(controls)
        plot_div.append(
            html.Div(
                id="plot-container",
                children=[
                    html.Div(
                        "Click 'Plot' to visualize data",
                        style={
                            "text-align": "center",
                            "color": "#666",
                            "font-size": "18px",
                            "padding": "100px",
                            "height": "calc(100vh - 250px)",
                            "display": "flex",
                            "align-items": "center",
                            "justify-content": "center",
                        }
                    )
                ],
                style={
                    "width": "95%",
                    "display": "inline-block",
                    "vertical-align": "top",
                },
            )
        )

        return html.Div(plot_div)

    # Button-triggered plot update callback
    @callback(
        [
            Output("plot-container", "children", allow_duplicate=True),
            Output("plot-button", "children", allow_duplicate=True),
        ],
        Input("plot-button", "n_clicks"),
        [
            State("dim-x-select", "value"),
            State("dim-y-select", "value"),
            State("dim-z-select", "value"),
            State(component_id="treenum-slider", component_property="value"),
            State("show-lines-checkbox", "checked"),
            State("plot-container", "children"),
            State("plot-config-store", "data")
        ],
        prevent_initial_call=True,
    )
    def update_graph_on_button_click(n_clicks, dim_x, dim_y, dim_z, treenum_range, show_lines, current_plot, plot_config):
        if not n_clicks or not plot_config or not all([dim_x, dim_y, dim_z]):
            return no_update, no_update

        # Filter data based on current control values
        combined_df = pd.DataFrame(plot_config["combined_data"])

        filtered_dff = combined_df[
            (combined_df["treenum"] >= treenum_range[0]) & (combined_df["treenum"] <= treenum_range[1])
        ]
        mds_selected = [dim_x, dim_y, dim_z]
        add_log(f"Plotting {len(filtered_dff)} trees (range {treenum_range[0]}-{treenum_range[1]}), dims: {mds_selected}")

        x, y, z = dim_x, dim_y, dim_z

        # Create new plot with filtered data
        fig = make_plot_grid()
        add_trace_multiplot(
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

        # Create the plot component
        plot_component = dcc.Graph(figure=fig, id="graph", style={"height": "calc(100vh - 250px)"})

        # Update button text to indicate it's been used
        button_text = "Update Plot"

        return [plot_component], button_text
