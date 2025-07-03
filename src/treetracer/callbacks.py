from dash import dcc, html, callback, Input, Output, State, no_update
from .plot_utils import make_plot_grid, add_trace_multiplot
import dash_mantine_components as dmc
import plotly.express as px
import json
import base64
import io
import pandas as pd


# Helper functions for single responsibilities
def validate_file(filename):
    """Validate if file is acceptable format."""
    if not filename.endswith(".tsv"):
        return f"Only TSV files are allowed. '{filename}' was rejected."
    return None


def parse_uploaded_file(content, filename):
    """Parse uploaded file content into DataFrame."""
    try:
        content_type, content_string = content.split(",")
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep="\t")
        return df, None
    except Exception as e:
        return None, f"Error parsing {filename}: {str(e)}"


def transform_dataframe(df, filename):
    """Transform DataFrame with required columns and calculations."""
    # Column renaming
    df.columns = [x.replace("V", "MDS") for x in df.columns]
    mdscols = sorted([x for x in df.columns if "MDS" in x])
    
    # Group mapping
    group_mapping = {val: idx for idx, val in enumerate(df["group"].unique())}
    df["group_col"] = df["group"].map(group_mapping)
    df["file"] = filename
    
    # Renumber trees
    df["treenum"] = df.groupby("group").cumcount() + 1
    df["size"] = 6
    
    return df, mdscols


def create_file_metadata(df, filename, mdscols):
    """Create metadata for uploaded file."""
    return {
        "filename": filename,
        "rows": len(df),
        "dimensions": mdscols,
        "groups": df["group"].unique().tolist(),
        "MIN_TREENUM": int(df["treenum"].min()),
        "MAX_TREENUM": int(df["treenum"].max()),
    }


def combine_dataframes(file_data, dataframes_dict, selected_files):
    """Combine multiple dataframes into one."""
    combined_data = []
    for item in file_data:
        if item["filename"] in selected_files:
            df_data = dataframes_dict.get(item["filename"])
            if df_data:
                combined_data.extend(df_data)
    
    if not combined_data:
        return None
    
    combined_df = pd.DataFrame(combined_data)
    if len(selected_files) > 1:
        combined_df["group"] = combined_df["file"] + "/" + combined_df["group"]
    
    return combined_df


def create_plot_config(combined_df, file_data, selected_files):
    """Create plot configuration from combined data."""
    # Get metadata from first selected file
    mdscols = None
    min_treenum = float('inf')
    max_treenum = float('-inf')
    
    for item in file_data:
        if item["filename"] in selected_files:
            mdscols = item["dimensions"]
            min_treenum = min(min_treenum, item["MIN_TREENUM"])
            max_treenum = max(max_treenum, item["MAX_TREENUM"])
    
    groups = combined_df["group"].unique().tolist()
    group_colors = px.colors.qualitative.Dark24[:len(groups)]
    color_dict = {g: c for g, c in zip(groups, group_colors)}
    
    return {
        "combined_data": combined_df.to_dict('records'),
        "mdscols": mdscols,
        "min_treenum": int(min_treenum),
        "max_treenum": int(max_treenum),
        "groups": groups,
        "color_dict": color_dict
    }


def register_callbacks(app):
    # Sidebar collapse callback
    @callback(
        Output("appshell", "navbar"),
        Input("burger", "opened"),
        State("appshell", "navbar"),
    )
    def toggle_navbar(opened, navbar):
        navbar["collapsed"] = {"mobile": not opened}
        return navbar

    # Callback to handle file uploads
    @callback(
        [
            Output("uploaded-files-store", "data"),
            Output("dataframes-store", "data"),
            Output("validation-alert", "title"),
            Output("validation-alert", "style"),
        ],
        Input("upload-data-button", "contents"),
        State("upload-data-button", "filename"),
        State("upload-data-button", "last_modified"),
        State("uploaded-files-store", "data"),
        State("dataframes-store", "data"),
        prevent_initial_call=True,
    )
    def update_uploaded_files(contents, filenames, dates, stored_files, stored_dataframes):
        if contents is None:
            return no_update, no_update, no_update, no_update

        # Initialize or get existing data
        file_data = stored_files or []
        dataframes_dict = stored_dataframes or {}
        existing_filenames = [item["filename"] for item in file_data]
        error_message = ""

        for content, filename, date in zip(contents, filenames, dates):
            # Single responsibility: File validation
            validation_error = validate_file(filename)
            if validation_error:
                error_message = validation_error
                continue

            if filename not in existing_filenames:
                # Single responsibility: File parsing
                df, parse_error = parse_uploaded_file(content, filename)
                if parse_error:
                    error_message = parse_error
                    continue

                # Single responsibility: Data transformation
                df, mdscols = transform_dataframe(df, filename)

                # Single responsibility: Data storage
                dataframes_dict[filename] = df.to_dict('records')

                # Single responsibility: Metadata creation
                metadata = create_file_metadata(df, filename, mdscols)
                metadata["date"] = date
                file_data.append(metadata)

        # Configure alert display
        alert_style = {"display": "block"} if error_message else {"display": "none"}
        return file_data, dataframes_dict, error_message, alert_style

    # Callback to update the MultiSelect with uploaded filenames
    @callback(
        Output("upload-placeholder", "children"),
        Input("uploaded-files-store", "data"),
        prevent_initial_call=True,
    )
    def update_multiselect(file_data):
        if not file_data:
            return html.Div("No files uploaded yet.")

        if not file_data:
            return html.Div("No files uploaded yet.")

        # Get all filenames
        filenames = [item["filename"] for item in file_data]

        # Create the MultiSelect component
        multiselect = dmc.MultiSelect(
            id="files-multiselect",
            label="Uploaded TSV Files",
            description="Select files to process",
            data=filenames,
            value=[],  # Initially select no files
            style={"width": "100%"},
            hidePickedOptions=True,
        )

        return multiselect

    # Callback to display information about selected files
    @callback(
        Output("data-info-display", "children", allow_duplicate=True),
        Input("files-multiselect", "value"),
        State("uploaded-files-store", "data"),
        State("dataframes-store", "data"),
        prevent_initial_call=True,
    )
    def display_file_info(selected_files, file_data, dataframes_dict):
        if not selected_files or not file_data:
            return html.Div("No files selected.")
        file_info = []

        for item in file_data:
            if item["filename"] in selected_files:
                df_data = dataframes_dict.get(item["filename"])
                if df_data is not None:
                    file_info.append(
                        dmc.Paper(
                            children=[
                                dmc.Text(
                                    f"Filename: {item['filename']}",
                                ),
                                dmc.Text(
                                    f"Number of Dimensions: {len(item['dimensions'])}"
                                ),
                                dmc.Text(f"Dimensions: {item['dimensions']}"),
                                dmc.Text(f"Number of Groups: {len(item['groups'])}"),
                                dmc.Text("Groups: " + ", ".join(item["groups"])),
                                dmc.Text(
                                    f"Tree Range: {item['MIN_TREENUM']}-{item['MAX_TREENUM']}"
                                ),
                                dmc.Text(f"Total Number of Trees: {item['rows']}"),
                                dmc.Space(h=10),
                            ],
                            p="md",
                            shadow="xs",
                            withBorder=True,
                            mt=10,
                        )
                    )

        return html.Div(file_info)

    # Callback to clear uploaded files
    @callback(
        [
            Output("uploaded-files-store", "data", allow_duplicate=True),
            Output("dataframes-store", "data", allow_duplicate=True),
            Output("plot-config-store", "data", allow_duplicate=True),
            Output("data-info-display", "children", allow_duplicate=True),
            Output("plot-display", "children", allow_duplicate=True),
        ],
        Input("clear-data-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_uploads(n_clicks):
        if n_clicks:
            return (
                [],
                {},
                {},
                html.Div("Files cleared."),
                html.Div("Files cleared."),
            )
        return no_update, no_update, no_update, no_update, no_update

    # ------- PLOT CALLBACK

    # Callback to prepare plot data and store configuration
    @callback(
        Output("plot-config-store", "data", allow_duplicate=True),
        Input("files-multiselect", "value"),
        State("uploaded-files-store", "data"),
        State("dataframes-store", "data"),
        prevent_initial_call=True,
    )
    def prepare_plot_config(selected_files, file_data, dataframes_dict):
        if not selected_files or not file_data:
            return {}

        # Single responsibility: Data combination
        combined_df = combine_dataframes(file_data, dataframes_dict, selected_files)
        if combined_df is None:
            return {}

        # Single responsibility: Plot configuration
        return create_plot_config(combined_df, file_data, selected_files)

    # Callback to display plots
    @callback(
        Output("plot-display", "children", allow_duplicate=True),
        Input("plot-config-store", "data"),
        prevent_initial_call=True,
    )
    def display_plots(plot_config):
        if not plot_config or not plot_config.get("combined_data"):
            return html.Div("No files selected.")
            
        plot_div = []
        combined_df = pd.DataFrame(plot_config["combined_data"])
        mdscols = plot_config["mdscols"]
        MIN_TREENUM = plot_config["min_treenum"]
        MAX_TREENUM = plot_config["max_treenum"]
        groups = plot_config["groups"]
        color_dict = plot_config["color_dict"]

# No default plot - will be populated by button click

        # Controls row with slider, checkbox, and plot button
        controls = html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Dimensions:",
                            style={"display": "block", "margin-bottom": "2px"},
                        ),
                        dcc.Checklist(
                            options=mdscols,
                            value=mdscols[:3],
                            id="dimensions-box",
                            inline=True,
                        ),
                    ],
                    style={"width": "20%", "padding": "5px"},
                ),
                # tree num slider
                html.Div(
                    [
                        html.Label("Filter by Tree Number Range:"),
                        dcc.RangeSlider(
                            id="treenum-slider",
                            min=MIN_TREENUM,
                            max=MAX_TREENUM,
                            value=[MIN_TREENUM, MAX_TREENUM],
                            marks={
                                MIN_TREENUM: str(MIN_TREENUM),
                                MAX_TREENUM: str(MAX_TREENUM),
                            },
                            step=1,
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                    style={"width": "60%", "padding": "5px"},
                ),
                # plot button
                html.Div(
                    [
                        dmc.Button(
                            "Plot",
                            id="plot-button",
                            variant="filled",
                            color="blue",
                            size="md",
                            style={"margin-top": "20px"},
                        ),
                    ],
                    style={"width": "20%", "padding": "5px", "text-align": "center"},
                ),
            ],
            style={"display": "flex", "align-items": "center", "padding": "5px 0"},
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
                            "height": "75vh",
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
            State(component_id="dimensions-box", component_property="value"),
            State(component_id="treenum-slider", component_property="value"),
            State("plot-container", "children"),
            State("plot-config-store", "data")
        ],
        prevent_initial_call=True,
    )
    def update_graph_on_button_click(n_clicks, mds_selected, treenum_range, current_plot, plot_config):
        if not n_clicks or not plot_config or len(mds_selected) != 3:
            return no_update, no_update

        # Filter data based on current control values
        combined_df = pd.DataFrame(plot_config["combined_data"])
        filtered_dff = combined_df[
            (combined_df["treenum"] >= treenum_range[0]) & (combined_df["treenum"] <= treenum_range[1])
        ]

        x, y, z = mds_selected

        # Create new plot with filtered data
        fig = make_plot_grid()
        add_trace_multiplot(
            fig, filtered_dff, x, y, z, plot_config["groups"], plot_config["color_dict"]
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
        plot_component = dcc.Graph(figure=fig, id="graph", style={"height": "75vh"})
        
        # Update button text to indicate it's been used
        button_text = "Update Plot"
        
        return [plot_component], button_text
