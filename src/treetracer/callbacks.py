from dash import dcc, html, callback, Input, Output, State, no_update, ALL, MATCH
from .plot_utils import make_plot_grid, add_trace_multiplot
from .logger import add_log, get_logs, clear_logs
from .db.tree_service import get_tree_service
import dash_mantine_components as dmc
import plotly.express as px
import json
import base64
import io
import os
import sys

import subprocess
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


def _open_file_dialog():
    """Open a native file picker and return the selected path."""
    if sys.platform == "darwin":
        cmd = [
            "osascript", "-e",
            'POSIX path of (choose file of type {"trees"} '
            'with prompt "Select a .trees file")',
        ]
    else:
        cmd = [
            sys.executable, "-c",
            "import tkinter as tk; from tkinter import filedialog; "
            "root = tk.Tk(); root.withdraw(); "
            "print(filedialog.askopenfilename("
            "title='Select a .trees file', "
            "filetypes=[('Trees files', '*.trees'), ('All files', '*.*')])); "
            "root.destroy()",
        ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


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

    # ------ LOG PANEL CALLBACKS ------

    @callback(
        Output("log-footer", "style"),
        Output("log-panel-visible", "data"),
        Output("appshell", "footer"),
        Input("log-toggle-button", "n_clicks"),
        State("log-panel-visible", "data"),
        prevent_initial_call=True,
    )
    def toggle_log_panel(n_clicks, is_visible):
        if n_clicks:
            new_state = not is_visible
            style = {"display": "block"} if new_state else {"display": "none"}
            footer = {"height": 250} if new_state else {"height": 0}
            return style, new_state, footer
        return no_update, no_update, no_update

    @callback(
        Output("log-content", "children"),
        Input("log-poll-interval", "n_intervals"),
        State("log-panel-visible", "data"),
    )
    def update_log_display(n_intervals, is_visible):
        if not is_visible:
            return no_update
        logs = get_logs()
        if not logs:
            return dmc.Text("No log entries yet.", c="dimmed", size="sm",
                            style={"padding": "8px"})
        level_colors = {
            "INFO": "#58a6ff",
            "WARNING": "#d29922",
            "ERROR": "#f85149",
        }
        elements = []
        for entry in logs:
            color = level_colors.get(entry["level"], "#8b949e")
            elements.append(
                html.Div([
                    html.Span(f"[{entry['timestamp']}] ",
                              style={"color": "#8b949e"}),
                    html.Span(f"{entry['level']}: ",
                              style={"color": color, "fontWeight": "bold"}),
                    html.Span(entry["message"],
                              style={"color": "#c9d1d9"}),
                ], style={"marginBottom": "2px"})
            )
        return elements

    @callback(
        Output("log-content", "children", allow_duplicate=True),
        Input("log-clear-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_log_display(n_clicks):
        if n_clicks:
            clear_logs()
            return dmc.Text("No log entries yet.", c="dimmed", size="sm",
                            style={"padding": "8px"})
        return no_update

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
            add_log(f"Processing file {i+1}: {filename}")

            # Single responsibility: File validation
            validation_error = validate_file(filename)
            if validation_error:
                add_log(f"Validation error: {validation_error}", "ERROR")
                error_message = validation_error
                continue

            if filename not in existing_filenames:
                # Single responsibility: File parsing
                df, parse_error = parse_uploaded_file(content, filename)
                if parse_error:
                    add_log(f"Parse error for {filename}: {parse_error}", "ERROR")
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
                add_log(f"Loaded {filename}: {df.shape[0]} rows, {len(df['group'].unique())} groups")
            else:
                add_log(f"File {filename} already exists, skipping", "WARNING")

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
                add_log(f"Uploading distance matrix: {filename}")
                df, parse_error = parse_uploaded_distmat(content, filename)
                if parse_error:
                    add_log(f"Parse error for {filename}: {parse_error}", "ERROR")
                    error_message = parse_error
                    continue

                # Store the dataframe as dict
                stored_distmats[filename] = df.to_dict(orient='index')
                add_log(f"Loaded distance matrix {filename}: {df.shape[0]}x{df.shape[1]}")
                #print(stored_distmats[filename].keys()) 


        alert_style = {"display": "block"} if error_message else {"display": "none"}
        return stored_distmats, error_message, alert_style


    # Instantly switch to Data tab when Load Trees is clicked
    from dash import clientside_callback, ClientsideFunction
    clientside_callback(
        """function(n_clicks) { return "data"; }""",
        Output("main-tabs", "value"),
        Input("load-trees-button", "n_clicks"),
        prevent_initial_call=True,
    )

    # Callback to load .trees files via native file dialog
    @callback(
        Output("tree-offset-store", "data"),
        Output("trees-validation-alert", "title"),
        Output("trees-validation-alert", "style"),
        Output("trees-info-display", "children", allow_duplicate=True),
        Output("data-info-display", "children", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
        Input("load-trees-button", "n_clicks"),
        State("tree-offset-store", "data"),
        prevent_initial_call=True,
    )
    def handle_trees_load(n_clicks, stored_summaries):
        if not n_clicks:
            return no_update, no_update, no_update, no_update, no_update, no_update

        file_path = _open_file_dialog()
        if not file_path:
            return no_update, no_update, no_update, no_update, no_update, no_update

        stored_summaries = stored_summaries or {}
        filename = os.path.basename(file_path)

        if not file_path.endswith(".trees"):
            msg = f"Only .trees files are allowed. '{filename}' was rejected."
            add_log(msg, "ERROR")
            return no_update, msg, {"display": "block"}, no_update, no_update, no_update

        if filename in stored_summaries:
            add_log(f"File {filename} already loaded, skipping", "WARNING")
            return no_update, no_update, no_update, no_update, no_update, no_update

        print(f"Loading {filename}...")
        add_log(f"Loading {filename}...")

        try:
            tree_service = get_tree_service()
            result = tree_service.load_nexus_file(
                file_path, file_source=filename
            )

            if result["success"]:
                # Store only lightweight summary, not every offset row
                trees_per_group = {}
                offset_df = tree_service.db_manager._trees
                file_rows = offset_df[offset_df["file_source"] == filename]
                if len(file_rows) > 0:
                    trees_per_group = (
                        file_rows.groupby("group_name", observed=True)
                        .size()
                        .to_dict()
                    )

                # Fetch translate map early so we can store n_taxa
                new_translate = tree_service.db_manager.get_translate_map(filename)

                stored_summaries[filename] = {
                    "total_trees": result["trees_loaded"],
                    "n_taxa": len(new_translate) if new_translate else 0,
                    "groups": list(trees_per_group.keys()),
                    "trees_per_group": trees_per_group,
                    "path": file_path,
                }
                print(f"Loaded {filename}: {result['trees_loaded']} trees")
                add_log(
                    f"Loaded {filename}: {result['trees_loaded']} trees"
                )

                # Check for taxa mismatch against previously loaded files
                taxa_warning = None
                if new_translate and stored_summaries:
                    new_taxa = set(new_translate.values())
                    mismatched_files = []
                    for other_file in stored_summaries:
                        if other_file == filename:
                            continue
                        other_translate = tree_service.db_manager.get_translate_map(other_file)
                        if other_translate is None:
                            continue
                        other_taxa = set(other_translate.values())
                        if new_taxa != other_taxa:
                            mismatched_files.append(
                                f"{other_file} has {len(other_taxa)} taxa"
                            )

                    if mismatched_files:
                        taxa_warning = (
                            f"{filename} has {len(new_taxa)} taxa but "
                            + ", ".join(mismatched_files)
                        )
                        add_log(f"Taxa mismatch warning: {taxa_warning}", "WARNING")

                if taxa_warning:
                    notification = dmc.Notification(
                        title="Trees Loaded — Taxa Mismatch",
                        message=f"Loaded {result['trees_loaded']} trees from {filename}. WARNING: {taxa_warning}",
                        color="yellow",
                        action="show",
                        autoClose=8000,
                        id="load-notification",
                    )
                else:
                    notification = dmc.Notification(
                        title="Trees Loaded",
                        message=f"Loaded {result['trees_loaded']} trees from {filename}.",
                        color="green",
                        action="show",
                        autoClose=4000,
                        id="load-notification",
                    )
                return stored_summaries, "", {"display": "none"}, no_update, html.Div(), notification
            else:
                msg = f"Error loading {filename}: {result.get('error', 'Unknown error')}"
                add_log(msg, "ERROR")
                return no_update, msg, {"display": "block"}, no_update, no_update, no_update
        except Exception as e:
            msg = f"Error processing {filename}: {str(e)}"
            add_log(msg, "ERROR")
            return no_update, msg, {"display": "block"}, no_update, no_update, no_update

    # Callback to display loaded trees info in the Data tab
    @callback(
        Output("trees-info-display", "children"),
        Input("tree-offset-store", "data"),
    )
    def display_trees_info(stored_summaries):
        if not stored_summaries:
            return html.Div()

        cards = []
        for filename, summary in stored_summaries.items():
            total_trees = summary["total_trees"]
            groups = summary["groups"]
            trees_per_group = summary["trees_per_group"]

            group_lines = [
                dmc.Text(f"  {g}: {count} trees", size="sm")
                for g, count in trees_per_group.items()
            ]

            cards.append(
                dmc.Paper(
                    children=[
                        dmc.Text(f"Filename: {filename}", fw=500),
                        dmc.Text("File Type: Nexus Trees", c="green"),
                        dmc.Text(f"Total Trees: {total_trees}"),
                        dmc.Text(f"Number of Groups: {len(groups)}"),
                        dmc.Text("Groups: " + ", ".join(groups)),
                        dmc.Space(h=5),
                        dmc.Text("Trees per Group:", fw=500, size="sm"),
                        *group_lines,
                        dmc.Space(h=10),
                        dmc.Group([
                            dmc.NumberInput(
                                id={"type": "downsample-input", "index": filename},
                                value=1000,
                                min=1,
                                step=1,
                                style={"width": "120px"},
                            ),
                            dmc.Button(
                                "Downsample Trees",
                                id={"type": "downsample-btn", "index": filename},
                                variant="filled",
                                color="orange",
                                size="sm",
                            ),
                            dmc.Button(
                                "Reset",
                                id={"type": "reset-trees-btn", "index": filename},
                                variant="outline",
                                color="red",
                                size="sm",
                            ),
                        ]),
                    ],
                    p="md",
                    shadow="xs",
                    withBorder=True,
                    mt=10,
                )
            )

        return html.Div(cards)

    # Callback to render the Compute tab table
    @callback(
        Output("compute-trees-table", "children"),
        Output("compute-rf-button", "disabled"),
        Input("tree-offset-store", "data"),
    )
    def render_compute_table(stored_summaries):
        if not stored_summaries:
            return html.Div(
                dmc.Text("No .trees files loaded yet.", c="dimmed"),
                style={"padding": "20px"},
            ), True

        rows = []
        for filename, summary in stored_summaries.items():
            rows.append(
                dmc.TableTr([
                    dmc.TableTd(filename),
                    dmc.TableTd(str(summary.get("n_taxa", "—"))),
                    dmc.TableTd(str(summary.get("total_trees", 0))),
                    dmc.TableTd(
                        dmc.Checkbox(
                            id={"type": "compute-tree-checkbox", "index": filename},
                            checked=True,
                        )
                    ),
                ])
            )

        table = dmc.Table(
            [
                dmc.TableThead(
                    dmc.TableTr([
                        dmc.TableTh("File"),
                        dmc.TableTh("Taxa"),
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

        return table, False

    # Callback to downsample trees for a given file
    @callback(
        Output("tree-offset-store", "data", allow_duplicate=True),
        Output("trees-info-display", "children", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
        Input({"type": "downsample-btn", "index": ALL}, "n_clicks"),
        State({"type": "downsample-input", "index": ALL}, "value"),
        State("tree-offset-store", "data"),
        prevent_initial_call=True,
    )
    def downsample_trees(n_clicks_list, n_value_list, stored_summaries):
        if not any(n_clicks_list):
            return no_update, no_update, no_update

        from dash import ctx
        triggered_id = ctx.triggered_id
        if not triggered_id:
            return no_update, no_update, no_update

        filename = triggered_id["index"]

        # Find the matching value from the ALL list
        n_value = None
        for i, inp in enumerate(ctx.inputs_list[0]):
            if inp["id"]["index"] == filename:
                n_value = n_value_list[i]
                break

        if not n_value:
            return no_update, no_update, no_update

        n = int(n_value)

        # Check if requested sample size exceeds available trees
        stored_summaries = stored_summaries or {}
        current_total = stored_summaries.get(filename, {}).get("total_trees", 0)
        if n >= current_total:
            msg = f"Requested {n} trees but {filename} only has {current_total}. No downsampling performed."
            print(msg)
            add_log(msg, "WARNING")
            notification = dmc.Notification(
                title="Downsample Skipped",
                message=msg,
                color="yellow",
                action="show",
                autoClose=8000,
                id="downsample-skip-notification",
            )
            return no_update, no_update, notification

        print(f"Downsampling {filename} to {n} trees...")
        add_log(f"Downsampling {filename} to {n} trees...")

        tree_service = get_tree_service()
        tree_service.db_manager.downsample_trees(filename, n)

        # Recompute summary from the DataFrame
        offset_df = tree_service.db_manager._trees
        file_rows = offset_df[offset_df["file_source"] == filename]
        trees_per_group = {}
        if len(file_rows) > 0:
            trees_per_group = (
                file_rows.groupby("group_name", observed=True)
                .size()
                .to_dict()
            )

        if filename in stored_summaries:
            stored_summaries[filename]["total_trees"] = len(file_rows)
            stored_summaries[filename]["groups"] = list(trees_per_group.keys())
            stored_summaries[filename]["trees_per_group"] = trees_per_group

        print(f"Downsampled {filename} to {len(file_rows)} trees")
        add_log(f"Downsampled {filename} to {len(file_rows)} trees")
        notification = dmc.Notification(
            title="Trees Downsampled",
            message=f"Downsampled {filename} to {len(file_rows)} trees.",
            color="orange",
            action="show",
            autoClose=4000,
            id="downsample-notification",
        )
        return stored_summaries, no_update, notification

    # Callback to reset trees to original file contents
    @callback(
        Output("tree-offset-store", "data", allow_duplicate=True),
        Output("trees-info-display", "children", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
        Input({"type": "reset-trees-btn", "index": ALL}, "n_clicks"),
        State("tree-offset-store", "data"),
        prevent_initial_call=True,
    )
    def reset_trees(n_clicks_list, stored_summaries):
        if not any(n_clicks_list):
            return no_update, no_update, no_update

        from dash import ctx
        triggered_id = ctx.triggered_id
        if not triggered_id:
            return no_update, no_update, no_update

        filename = triggered_id["index"]
        stored_summaries = stored_summaries or {}
        if filename not in stored_summaries:
            return no_update, no_update, no_update

        file_path = stored_summaries[filename].get("path")
        if not file_path:
            return no_update, no_update, no_update

        print(f"Resetting {filename}...")
        add_log(f"Resetting {filename}...")

        tree_service = get_tree_service()

        # Clear existing trees for this file and reload from disk
        tree_service.db_manager.clear_trees(file_source=filename)
        result = tree_service.load_nexus_file(file_path, file_source=filename)

        if not result["success"]:
            add_log(f"Reset failed for {filename}: {result.get('error')}", "ERROR")
            return no_update, no_update, no_update

        # Recompute summary
        offset_df = tree_service.db_manager._trees
        file_rows = offset_df[offset_df["file_source"] == filename]
        trees_per_group = {}
        if len(file_rows) > 0:
            trees_per_group = (
                file_rows.groupby("group_name", observed=True)
                .size()
                .to_dict()
            )

        stored_summaries[filename]["total_trees"] = len(file_rows)
        stored_summaries[filename]["groups"] = list(trees_per_group.keys())
        stored_summaries[filename]["trees_per_group"] = trees_per_group

        print(f"Reset {filename}: reloaded {len(file_rows)} trees from disk")
        add_log(f"Reset {filename}: reloaded {len(file_rows)} trees from disk")
        notification = dmc.Notification(
            title="Trees Reset",
            message=f"Reloaded {len(file_rows)} trees from {filename}.",
            color="red",
            action="show",
            autoClose=4000,
            id="reset-notification",
        )
        return stored_summaries, no_update, notification

    # Callbacks to update the MultiSelect with uploaded filenames

    # Callback 1: Create the initial MultiSelect component
    @callback(
        Output("upload-placeholder", "children"),
        Input("uploaded-files-store", "data"),
        Input("distmat-store", "data"),
        prevent_initial_call=True,
    )
    def create_multiselect_component(file_data, distmat_data):
        # Check if we have any files at all
        if not file_data and not distmat_data:
            return html.Div("No files uploaded yet.")

        # Get initial filenames
        trace_filenames = [item["filename"] for item in file_data] if file_data else []
        distmat_filenames = list(distmat_data.keys()) if distmat_data else []
        all_filenames = trace_filenames + distmat_filenames

        if not all_filenames:
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
        
        return multiselect

    # Callback 2: Update MultiSelect data while preserving selections
    @callback(
    [Output("files-multiselect", "data"), Output("files-multiselect", "value")],
    [Input("uploaded-files-store", "data"), Input("distmat-store", "data")],
    State("files-multiselect", "value"),
    prevent_initial_call=True,
    )
    def update_multiselect_data(file_data, distmat_data, current_value):
        # Get all filenames
        trace_filenames = [item["filename"] for item in file_data] if file_data else []
        distmat_filenames = list(distmat_data.keys()) if distmat_data else []
        all_filenames = trace_filenames + distmat_filenames

        # If no files, return no_update to avoid errors
        if not all_filenames:
            return no_update, no_update

        # Preserve current selections that are still valid
        preserved_value = []
        if current_value:
            preserved_value = [f for f in current_value if f in all_filenames]

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
        # Check if any button was clicked
        if not any(n_clicks_list) or not distmat_data:
            return no_update, no_update

        # Find which button was clicked
        from dash import ctx
        if not ctx.triggered:
            return no_update, no_update

        # Extract filename from the triggered button
        triggered_id = ctx.triggered_id
        if not triggered_id or "filename" not in triggered_id:
            return no_update, no_update

        filename = triggered_id["filename"]

        # Get the distance matrix data
        if filename not in distmat_data:
            return no_update, no_update

        distmat_df_data = distmat_data[filename]
        distmat_df = pd.DataFrame(distmat_df_data)
        add_log(f"Distance matrix {filename}: {distmat_df.shape[0]}x{distmat_df.shape[1]}")

        # Convert to numpy array for MDS (assuming it's a square distance matrix)
        distance_matrix = distmat_df.values

        # Check if it's actually a square matrix
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            add_log(f"Distance matrix is not square: {distance_matrix.shape}", "ERROR")
            return no_update, no_update

        # Perform MDS
        try:
            n_components = min(3, distance_matrix.shape[0] - 1)
            add_log(f"Computing MDS with {n_components} components...")
            mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42, verbose=1)
            embedding = mds.fit_transform(distance_matrix)
            add_log(f"MDS completed: {embedding.shape[0]} points in {embedding.shape[1]}D space")
        except Exception as e:
            add_log(f"MDS computation failed: {str(e)}", "ERROR")
            return no_update, no_update

        # Create a new dataframe in the same format as trace files
        mds_filename = filename.replace('.tsv', '_MDS.tsv')

        # Extract tree names from the distance matrix
        tree_names = distmat_df.index.astype(str).tolist()

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

        # Update stored data
        updated_file_data = (file_data or []).copy()
        updated_dataframes_dict = (dataframes_dict or {}).copy()

        # Add to file data if not already present
        existing_filenames = [item["filename"] for item in updated_file_data]
        if mds_filename not in existing_filenames:
            updated_file_data.append(mds_metadata)

        # Add to dataframes dict
        updated_dataframes_dict[mds_filename] = mds_df.to_dict('records')

        add_log(f"Created {mds_filename}: {len(mds_df)} trees, {len(mds_df['group'].unique())} groups")
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
        Output("upload-data-button", "contents"),
        Output("upload-distmat-button", "contents"),
        Output("tree-offset-store", "data", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
    ],
    Input("clear-data-button", "n_clicks"),
    prevent_initial_call=True,
    )
    def clear_uploads(n_clicks):
        if n_clicks:
            add_log("Data cleared")
            notification = dmc.Notification(
                title="Data Cleared",
                message="All uploaded files and plots have been cleared.",
                color="blue",
                action="show",
                autoClose=4000,
                id="clear-notification",
            )
            return (
                [],
                {},
                {},
                {},
                html.Div(),
                html.Div(),
                None,
                None,
                {},
                notification,
            )
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update


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
        add_log(f"Plotting {len(filtered_dff)} trees (range {treenum_range[0]}-{treenum_range[1]}), dims: {mds_selected}")

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
        plot_component = dcc.Graph(figure=fig, id="graph", style={"height": "calc(100vh - 250px)"})
        
        # Update button text to indicate it's been used
        button_text = "Update Plot"
        
        return [plot_component], button_text
