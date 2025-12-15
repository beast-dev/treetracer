from dash import dcc, html, callback, Input, Output, State, no_update, ALL, MATCH
from .plot_utils import make_plot_grid, add_trace_multiplot
import dash_mantine_components as dmc
import plotly.express as px
import json
import base64
import io
import pandas as pd
from sklearn.manifold import MDS
import numpy as np


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

def parse_uploaded_distmat(content, filename):
    """Parse uploaded file content into DataFrame."""
    try:
        content_type, content_string = content.split(",")
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep="\t", index_col=0)
        #print(df.index)
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
    #prevent_initial_call=True,
    )
    def update_uploaded_files(contents, filenames, dates, stored_files, stored_dataframes):
        # Some debugging stuff
        #print("Trace callback TRIGGERED")
        #print(f"Contents: {contents is not None}")
        #print(f"Filenames: {filenames}")
        #print(f"Number of files: {len(filenames) if filenames else 0}")
        
        if contents is None:
            return no_update, no_update, no_update, no_update

        # Initialize or get existing data
        file_data = stored_files or []
        dataframes_dict = stored_dataframes or {}
        existing_filenames = [item["filename"] for item in file_data]
        error_message = ""

        # Some debugging stuff
        #print(f"Existing filenames: {existing_filenames}")
        #print(f"Processing {len(filenames)} files")

        for i, (content, filename, date) in enumerate(zip(contents, filenames, dates)):
            print(f"Processing file {i+1}: {filename}")
            
            # Single responsibility: File validation
            validation_error = validate_file(filename)
            if validation_error:
                print(f"Validation error for {filename}: {validation_error}")
                error_message = validation_error
                continue

            if filename not in existing_filenames:
                print(f"Processing new file: {filename}")
                # Single responsibility: File parsing
                df, parse_error = parse_uploaded_file(content, filename)
                if parse_error:
                    print(f"Parse error for {filename}: {parse_error}")
                    error_message = parse_error
                    continue

                # Single responsibility: Data transformation
                df, mdscols = transform_dataframe(df, filename)
                print(f"Transformed {filename}, shape: {df.shape}")

                # Single responsibility: Data storage
                dataframes_dict[filename] = df.to_dict('records')

                # Single responsibility: Metadata creation
                metadata = create_file_metadata(df, filename, mdscols)
                metadata["date"] = date
                file_data.append(metadata)
                print(f"Added metadata for {filename}")
            else:
                print(f"File {filename} already exists, skipping")

        # Some debugging stuff
        #print(f"Final file_data length: {len(file_data)}")
        #print(f"Final dataframes_dict keys: {list(dataframes_dict.keys())}")

        # Configure alert display
        alert_style = {"display": "block"} if error_message else {"display": "none"}
        return file_data, dataframes_dict, error_message, alert_style
    
    # Callback to handle distance matrix uploads
    @callback(
        Output("distmat-store", "data"),
        Output("distmat-validation-alert", "title"),
        Output("distmat-validation-alert", "style"),
        Input("upload-distmat-button", "contents"),
        State("upload-distmat-button", "filename"),
        State("upload-distmat-button", "last_modified"),
        State("distmat-store", "data"),
        #prevent_initial_call=True,
    )
    def update_uploaded_distmat(contents, filenames, dates, stored_distmats):
        # Some debugging stuff
        #print("Distmat callback TRIGGERED")
        if contents is None:
            return no_update, no_update, no_update

        stored_distmats = stored_distmats or {}
        error_message = ""

        for content, filename, date in zip(contents, filenames, dates):
            validation_error = validate_file(filename)
            if validation_error:
                error_message = validation_error
                continue

            if filename not in stored_distmats:
                df, parse_error = parse_uploaded_distmat(content, filename)
                if parse_error:
                    error_message = parse_error
                    continue

                # Store the dataframe as dict
                #stored_distmats[filename] = df.to_dict('records')
                stored_distmats[filename] = df.to_dict(orient='index')
                #print(stored_distmats[filename].keys()) 


        alert_style = {"display": "block"} if error_message else {"display": "none"}
        return stored_distmats, error_message, alert_style


    # Callbacks to update the MultiSelect with uploaded filenames 

    # Callback 1: Create the initial MultiSelect component
    @callback(
        Output("upload-placeholder", "children"),
        Input("uploaded-files-store", "data"),
        Input("distmat-store", "data"),
        prevent_initial_call=True,
    )
    def create_multiselect_component(file_data, distmat_data):
        print("create_multiselect_component callback triggered")
        
        # Check if we have any files at all
        if not file_data and not distmat_data:
            print("No files found, returning 'No files uploaded yet'")
            return html.Div("No files uploaded yet.")
        
        # Get initial filenames
        trace_filenames = [item["filename"] for item in file_data] if file_data else []
        distmat_filenames = list(distmat_data.keys()) if distmat_data else []
        all_filenames = trace_filenames + distmat_filenames
        
        if not all_filenames:
            print("No filenames found, returning 'No files uploaded yet'")
            return html.Div("No files uploaded yet.")
        
        # Create the multiselect component
        multiselect = dmc.MultiSelect(
            id="files-multiselect",
            label="Uploaded TSV Files",
            description="Select files to process",
            data=all_filenames,
            value=[],
            style={"width": "100%"},
            hidePickedOptions=True,
            searchable=True,
        )
        
        print(f"Created initial multiselect with data: {all_filenames}")
        return multiselect

    # Callback 2: Update MultiSelect data while preserving selections
    @callback(
    [Output("files-multiselect", "data"), Output("files-multiselect", "value")],
    [Input("uploaded-files-store", "data"), Input("distmat-store", "data")],
    State("files-multiselect", "value"),
    prevent_initial_call=True,
    )
    def update_multiselect_data(file_data, distmat_data, current_value):
        print("update_multiselect_data callback triggered")
        print(f"file_data: {[item['filename'] for item in file_data] if file_data else 'None'}")
        print(f"distmat_data keys: {list(distmat_data.keys()) if distmat_data else 'None'}")
        print(f"current_value: {current_value}")
        
        # Get all filenames
        trace_filenames = [item["filename"] for item in file_data] if file_data else []
        distmat_filenames = list(distmat_data.keys()) if distmat_data else []
        all_filenames = trace_filenames + distmat_filenames
        
        # If no files, return no_update to avoid errors
        if not all_filenames:
            print("No files available, returning no_update")
            return no_update, no_update
        
        # Preserve current selections that are still valid
        preserved_value = []
        if current_value:
            preserved_value = [f for f in current_value if f in all_filenames]
        
        print(f"Updated multiselect data: {all_filenames}, preserved selections: {preserved_value}")
        return all_filenames, preserved_value


    # Callback to display information about selected files
    @callback(
        Output("data-info-display", "children", allow_duplicate=True),
        Input("files-multiselect", "value"),
        State("uploaded-files-store", "data"),
        State("dataframes-store", "data"),
        State("distmat-store", "data"), 
        prevent_initial_call=True,
    )
    def display_file_info(selected_files, file_data, dataframes_dict, distmat_data):
        if not selected_files:
            return html.Div("No files selected.")
        
        file_info = []

        for filename in selected_files:
            # Check if it's a trace file (including MDS files)
            trace_file = next((item for item in (file_data or []) if item["filename"] == filename), None)
            
            if trace_file:
                # Handle trace files
                df_data = dataframes_dict.get(filename)
                if df_data is not None:
                    # Check if this is an MDS file
                    is_mds_file = filename.endswith('_MDS.tsv')
                    file_type = "MDS Results" if is_mds_file else "Tree Traces"
                    file_color = "purple" if is_mds_file else "blue"
                    
                    file_info.append(
                        dmc.Paper(
                            children=[
                                dmc.Text(f"Filename: {filename}", fw=500),
                                dmc.Text(f"File Type: {file_type}", c=file_color),
                                dmc.Text(f"Number of Dimensions: {len(trace_file['dimensions'])}"),
                                dmc.Text(f"Dimensions: {trace_file['dimensions']}"),
                                dmc.Text(f"Number of Groups: {len(trace_file['groups'])}"),
                                dmc.Text("Groups: " + ", ".join(trace_file["groups"])),
                                dmc.Text(f"Tree Range: {trace_file['MIN_TREENUM']}-{trace_file['MAX_TREENUM']}"),
                                dmc.Text(f"Total Number of Trees: {trace_file['rows']}"),
                                dmc.Space(h=10),
                            ],
                            p="md",
                            shadow="xs",
                            withBorder=True,
                            mt=10,
                        )
                    )
            
            elif distmat_data and filename in distmat_data:
                # Handle distance matrix files (only show button if not already processed)
                distmat_df_data = distmat_data[filename]
                df = pd.DataFrame(distmat_df_data)
                
                # Check if MDS version already exists
                mds_filename = filename.replace('.tsv', '_MDS.tsv')
                mds_exists = any(item["filename"] == mds_filename for item in (file_data or []))
                
                button_content = []
                if not mds_exists:
                    button_content.append(
                        dmc.Button(
                            "Process Distance Matrix",
                            id={"type": "process-distmat", "filename": filename},
                            variant="filled",
                            color="green",
                            size="sm",
                            fullWidth=True,
                        )
                    )
                else:
                    button_content.append(
                        dmc.Text(f"✓ MDS computed as: {mds_filename}", c="green", fw=500)
                    )
                
                file_info.append(
                    dmc.Paper(
                        children=[
                            dmc.Text(f"Filename: {filename}", fw=500),
                            dmc.Text(f"File Type: Distance Matrix", c="green"),
                            dmc.Text(f"Shape: {df.shape[0]} x {df.shape[1]}"),
                            dmc.Space(h=10),
                        ] + button_content,
                        p="md",
                        shadow="xs",
                        withBorder=True,
                        mt=10,
                    )
                )

        if not file_info:
            return html.Div("Selected files not found in uploaded data.")
        
        return html.Div(file_info)

    # Callback to compute MDS from distance matrix

    @callback(
    [
        Output("uploaded-files-store", "data", allow_duplicate=True),
        Output("dataframes-store", "data", allow_duplicate=True),
        #Output("upload-placeholder", "children", allow_duplicate=True), 
    ],
    Input({"type": "process-distmat", "filename": ALL}, "n_clicks"),
    [
        State("distmat-store", "data"),
        State("uploaded-files-store", "data"),
        State("dataframes-store", "data"),
    ],
    prevent_initial_call=True,
    )
    def process_distance_matrix_mds(n_clicks_list, distmat_data, file_data, dataframes_dict):
        print("Compute MDS triggered")
        print(f"n_clicks_list: {n_clicks_list}")
        print(f"distmat_data keys: {list(distmat_data.keys()) if distmat_data else 'None'}")
        
        # Check if any button was clicked
        if not any(n_clicks_list) or not distmat_data:
            print("Early return: no clicks or no distmat_data")
            return no_update, no_update#, no_update
        
        # Find which button was clicked
        from dash import ctx
        if not ctx.triggered:
            print("Early return: no ctx.triggered")
            return no_update, no_update#, no_update
        
        print(f"ctx.triggered: {ctx.triggered}")
        print(f"ctx.triggered_id: {ctx.triggered_id}")
        
        # Extract filename from the triggered button
        triggered_id = ctx.triggered_id
        if not triggered_id or "filename" not in triggered_id:
            print("Early return: no filename in triggered_id")
            return no_update, no_update#, no_update
        
        filename = triggered_id["filename"]
        print(f"Processing filename: {filename}")
        
        # Get the distance matrix data
        if filename not in distmat_data:
            print(f"Early return: {filename} not in distmat_data")
            return no_update, no_update#, no_update
        
        print("Getting distance matrix data...")
        distmat_df_data = distmat_data[filename]
        distmat_df = pd.DataFrame(distmat_df_data)
        print(distmat_df.index)
        print(f"Distance matrix shape: {distmat_df.shape}")
        
        # Convert to numpy array for MDS (assuming it's a square distance matrix)
        distance_matrix = distmat_df.values
        print(f"Distance matrix array shape: {distance_matrix.shape}")
        
        # Check if it's actually a square matrix
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            print(f"Error: Distance matrix is not square: {distance_matrix.shape}")
            return no_update, no_update#, no_update
        
        # Perform MDS
        print("Computing MDS...")
        try:
            n_components = min(3, distance_matrix.shape[0] - 1)  
            print(f"Using {n_components} components for MDS")
            mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42, verbose = 1)
            embedding = mds.fit_transform(distance_matrix)
            print(f"MDS embedding shape: {embedding.shape}")
        except Exception as e:
            print(f"Error computing MDS: {e}")
            return no_update, no_update#, no_update
        
        # Create a new dataframe in the same format as trace files
        mds_filename = filename.replace('.tsv', '_MDS.tsv')
        print(f"Creating MDS file: {mds_filename}")
        
        # Extract tree names from the distance matrix
        tree_names = distmat_df.index.astype(str).tolist()
        print(tree_names)

        # MDS result
        mds_df = pd.DataFrame(embedding, columns=[f"MDS{i+1}" for i in range(n_components)])
        mds_df["tree"] = tree_names

        # CLEAN group column
        mds_df["group"] = mds_df["tree"].apply(lambda x: str(x).split("_")[0].strip())
        mds_df["group"] = mds_df["group"].astype(str)  # enforce dtype

        # Build group_col just for internal use
        group_mapping = {val: idx for idx, val in enumerate(sorted(mds_df["group"].unique()))}
        mds_df["group_col"] = mds_df["group"].map(group_mapping)

        # Add rest of fields
        mds_df["file"] = mds_filename
        mds_df["treenum"] = mds_df.groupby("group").cumcount() + 1
        mds_df["size"] = 6

        print(f"Created MDS dataframe with shape: {mds_df.shape}")
        print(f"MDS dataframe columns: {list(mds_df.columns)}")
        
        # Get MDS columns for metadata
        mdscols = [f'MDS{i+1}' for i in range(n_components)]
        
        # Create metadata for the MDS file
        mds_metadata = {
            "filename": mds_filename,
            "rows": len(mds_df),
            "dimensions": mdscols,
            "groups": mds_df["group"].unique().tolist(),
            "MIN_TREENUM": int(mds_df["treenum"].min()),
            "MAX_TREENUM": int(mds_df["treenum"].max()),
            "date": None,
        }
        
        print(f"Created metadata: {mds_metadata}")
        
        # Update stored data
        updated_file_data = (file_data or []).copy()
        updated_dataframes_dict = (dataframes_dict or {}).copy()
        
        # Add to file data if not already present
        existing_filenames = [item["filename"] for item in updated_file_data]
        if mds_filename not in existing_filenames:
            updated_file_data.append(mds_metadata)
            print(f"Added {mds_filename} to file_data")
        else:
            print(f"{mds_filename} already exists in file_data")
        
        # Add to dataframes dict
        updated_dataframes_dict[mds_filename] = mds_df.to_dict('records')
        print(f"Added {mds_filename} to dataframes_dict")
        
        print("MDS processing completed successfully")
        return updated_file_data, updated_dataframes_dict

    # Callback to clear uploaded files
    @callback(
    [
        Output("uploaded-files-store", "data", allow_duplicate=True),
        Output("distmat-store", "data", allow_duplicate=True),
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
                {}, 
                html.Div("Files cleared."),
                html.Div("Files cleared."),
            )
        return no_update, no_update, no_update, no_update, no_update, no_update 


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
        print(f"DEBUG: Button clicked! n_clicks={n_clicks}")
        print(f"DEBUG: treenum_range={treenum_range}")
        print(f"DEBUG: mds_selected={mds_selected}")

        if not n_clicks or not plot_config or len(mds_selected) != 3:
            print("DEBUG: Early return - validation failed")
            return no_update, no_update

        # Filter data based on current control values
        combined_df = pd.DataFrame(plot_config["combined_data"])
        print(f"DEBUG: combined_df shape before filter: {combined_df.shape}")
        print(f"DEBUG: treenum range in data: {combined_df['treenum'].min()} to {combined_df['treenum'].max()}")

        filtered_dff = combined_df[
            (combined_df["treenum"] >= treenum_range[0]) & (combined_df["treenum"] <= treenum_range[1])
        ]
        print(f"DEBUG: filtered_dff shape after filter: {filtered_dff.shape}")

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
