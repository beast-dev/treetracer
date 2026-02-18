from dash import dcc, html, callback, Input, Output, State, no_update, ALL
from .plot_utils import make_plot_grid, add_trace_multiplot
from .logger import add_log, get_logs, clear_logs
from .db.tree_service import get_tree_service
import dash_mantine_components as dmc
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

import subprocess
import pandas as pd
from sklearn.manifold import MDS


def _save_file_dialog(default_filename="output.tsv"):
    """Open a native save-file dialog and return the chosen path."""
    if sys.platform == "darwin":
        cmd = [
            "osascript", "-e",
            'POSIX path of (choose file name with prompt '
            '"Save file as" default name "' + default_filename + '")',
        ]
    else:
        cmd = [
            sys.executable, "-c",
            "import tkinter as tk; from tkinter import filedialog; "
            "root = tk.Tk(); root.withdraw(); "
            "print(filedialog.asksaveasfilename("
            "title='Save file as', "
            "initialfile='" + default_filename + "', "
            "defaultextension='.tsv', "
            "filetypes=[('TSV files', '*.tsv'), ('All files', '*.*')])); "
            "root.destroy()",
        ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


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


def _open_tsv_dialog():
    """Open a native file picker filtered to .tsv files and return the selected path."""
    if sys.platform == "darwin":
        cmd = [
            "osascript", "-e",
            'POSIX path of (choose file of type {"tsv","tab"} '
            'with prompt "Select a .tsv file")',
        ]
    else:
        cmd = [
            sys.executable, "-c",
            "import tkinter as tk; from tkinter import filedialog; "
            "root = tk.Tk(); root.withdraw(); "
            "print(filedialog.askopenfilename("
            "title='Select a .tsv file', "
            "filetypes=[('TSV files', '*.tsv'), ('All files', '*.*')])); "
            "root.destroy()",
        ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _validate_group_names(names):
    """Check that every tree name contains at least one '/' as a group delimiter.

    Returns (True, None) if valid, (False, error_message) if any name lacks a group.
    """
    bad_names = [n for n in names if "/" not in str(n)]
    if bad_names:
        preview = ", ".join(str(n) for n in bad_names[:5])
        if len(bad_names) > 5:
            preview += f" ... ({len(bad_names)} total)"
        return False, (
            "Tree names must include a group prefix (e.g., 'group1/tree_name'). "
            f"Found names without '/': {preview}"
        )
    return True, None


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

    # Show loading overlay instantly when Load Trees is clicked
    from dash import clientside_callback
    clientside_callback(
        """function(n) { return true; }""",
        Output("sidebar-loading-overlay", "visible"),
        Input("load-trees-button", "n_clicks"),
        prevent_initial_call=True,
    )

    # Callback to load .trees files via native file dialog
    @callback(
        Output("tree-offset-store", "data"),
        Output("trees-validation-alert", "title"),
        Output("trees-validation-alert", "style"),
        Output("sidebar-trees-display", "children", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
        Output("sidebar-loading-overlay", "visible", allow_duplicate=True),
        Input("load-trees-button", "n_clicks"),
        State("tree-offset-store", "data"),
        prevent_initial_call=True,
    )
    def handle_trees_load(n_clicks, stored_summaries):
        if not n_clicks:
            return no_update, no_update, no_update, no_update, no_update, no_update

        file_path = _open_file_dialog()
        if not file_path:
            return no_update, no_update, no_update, no_update, no_update, False

        stored_summaries = stored_summaries or {}
        filename = os.path.basename(file_path)

        if not file_path.endswith(".trees"):
            msg = f"Only .trees files are allowed. '{filename}' was rejected."
            add_log(msg, "ERROR")
            return no_update, msg, {"display": "block"}, no_update, no_update, False

        if filename in stored_summaries:
            add_log(f"File {filename} already loaded, skipping", "WARNING")
            return no_update, no_update, no_update, no_update, no_update, False

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
                        autoClose=3000,
                        id="load-notification",
                    )
                else:
                    notification = dmc.Notification(
                        title="Trees Loaded",
                        message=f"Loaded {result['trees_loaded']} trees from {filename}.",
                        color="green",
                        action="show",
                        autoClose=3000,
                        id="load-notification",
                    )
                return stored_summaries, "", {"display": "none"}, no_update, notification, False
            else:
                msg = f"Error loading {filename}: {result.get('error', 'Unknown error')}"
                add_log(msg, "ERROR")
                return no_update, msg, {"display": "block"}, no_update, no_update, False
        except Exception as e:
            msg = f"Error processing {filename}: {str(e)}"
            add_log(msg, "ERROR")
            return no_update, msg, {"display": "block"}, no_update, no_update, False

    # Callback to display loaded trees info in the sidebar
    @callback(
        Output("sidebar-trees-display", "children"),
        Input("tree-offset-store", "data"),
    )
    def display_trees_info(stored_summaries):
        if not stored_summaries:
            return dmc.Text(
                "No trees loaded. Click the upload button above to load a .trees file.",
                c="dimmed",
                size="sm",
                style={"padding": "10px"},
            )

        items = []
        for filename, summary in stored_summaries.items():
            total_trees = summary["total_trees"]
            n_taxa = summary.get("n_taxa", 0)
            trees_per_group = summary["trees_per_group"]

            group_lines = [
                dmc.Text(f"{g}: {count}", size="xs", c="dimmed")
                for g, count in trees_per_group.items()
            ]

            panel_content = dmc.Stack([
                dmc.Text(f"Taxa: {n_taxa}", size="xs"),
                dmc.Text("Groups:", size="xs", fw=500),
                *group_lines,
                dmc.Divider(my="xs"),
                dmc.Group([
                    dmc.NumberInput(
                        id={"type": "downsample-input", "index": filename},
                        value=1000,
                        min=1,
                        step=1,
                        size="xs",
                        style={"width": "80px"},
                    ),
                    dmc.Button(
                        "Downsample",
                        id={"type": "downsample-btn", "index": filename},
                        variant="filled",
                        color="orange",
                        size="compact-xs",
                    ),
                    dmc.Button(
                        "Reset",
                        id={"type": "reset-trees-btn", "index": filename},
                        variant="outline",
                        color="red",
                        size="compact-xs",
                    ),
                ], gap="xs"),
            ], gap="xs")

            items.append(
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl(
                            dmc.Group([
                                dmc.Text(filename, size="sm", fw=500, style={"flex": 1}),
                                dmc.Badge(str(total_trees), size="sm", variant="light"),
                            ], gap="xs"),
                        ),
                        dmc.AccordionPanel(panel_content),
                    ],
                    value=filename,
                )
            )

        return dmc.Accordion(
            items,
            multiple=True,
            variant="separated",
            value=list(stored_summaries.keys()),
        )

    # Callback to render the Compute tab RF table
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

    # Callback to validate taxa and trigger RF computation
    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("compute-rf-output", "children"),
        Output("distmat-store", "data", allow_duplicate=True),
        Output("export-rf-button", "disabled"),
        Input("compute-rf-button", "n_clicks"),
        State({"type": "compute-tree-checkbox", "index": ALL}, "checked"),
        State({"type": "compute-tree-checkbox", "index": ALL}, "id"),
        State("tree-offset-store", "data"),
        State("distmat-store", "data"),
        prevent_initial_call=True,
    )
    def handle_compute_rf(n_clicks, checked_list, id_list, stored_summaries,
                          stored_distmats):
        if not n_clicks or not stored_summaries:
            return no_update, no_update, no_update, no_update

        # Determine which files are selected
        selected_files = [
            id_item["index"]
            for id_item, checked in zip(id_list, checked_list)
            if checked
        ]

        add_log(f"Compute RF requested for {len(selected_files)} file(s): {', '.join(selected_files)}")

        if len(selected_files) < 1:
            msg = "At least 1 file must be selected to compute RF distances."
            add_log(msg, "WARNING")
            return dmc.Notification(
                title="Selection Error",
                message=msg,
                color="yellow",
                action="show",
                autoClose=6000,
                id="compute-rf-notification",
            ), no_update, no_update, no_update

        # Collect taxa counts for selected files
        taxa_counts = {}
        for filename in selected_files:
            summary = stored_summaries.get(filename, {})
            taxa_counts[filename] = summary.get("n_taxa", 0)

        add_log(f"Taxa counts: {taxa_counts}")

        unique_counts = set(taxa_counts.values())
        if len(unique_counts) > 1:
            details = "; ".join(
                f"{fname}: {count} taxa" for fname, count in taxa_counts.items()
            )
            msg = f"Selected files have different numbers of taxa ({details}). All files must share the same taxa set to compute RF distances."
            add_log(f"Taxa mismatch — aborting RF computation: {details}", "ERROR")
            return dmc.Notification(
                title="Taxa Mismatch",
                message=msg,
                color="red",
                action="show",
                autoClose=6000,
                id="compute-rf-notification",
            ), no_update, no_update, no_update

        # --- All taxa counts match — run RF computation ---
        n_taxa = unique_counts.pop()
        add_log(f"Taxa validation passed: all {len(selected_files)} files have {n_taxa} taxa")

        tree_service = get_tree_service()

        # Compute total trees across selected files
        total_trees = sum(
            stored_summaries[f].get("total_trees", 0) for f in selected_files
        )
        add_log(f"Fetching all {total_trees} trees from {len(selected_files)} files...")

        # Fetch all trees from selected files
        sample = tree_service.get_sample_for_analysis(
            file_sources=selected_files,
            sample_size=total_trees,
            strategy="random",
        )
        sampled_trees = sample["trees"]
        add_log(f"Retrieved {len(sampled_trees)} trees for RF computation")

        if len(sampled_trees) < 2:
            msg = "Not enough trees retrieved for RF computation."
            add_log(msg, "ERROR")
            return dmc.Notification(
                title="RF Error",
                message=msg,
                color="red",
                action="show",
                autoClose=6000,
                id="compute-rf-notification",
            ), no_update, no_update, no_update

        # Build names, newicks, translate maps, and map indices
        names = [t["name"] for t in sampled_trees]
        newicks = tree_service.prepare_trees_for_rf_analysis(sampled_trees)

        # Collect unique translate maps and build per-tree map indices
        translate_maps = []
        file_to_map_idx = {}
        for fname in selected_files:
            tmap = tree_service.db_manager.get_translate_map(fname)
            if tmap is not None:
                file_to_map_idx[fname] = len(translate_maps)
                translate_maps.append(tmap)

        map_indices = [
            file_to_map_idx.get(t["file_source"], 0) for t in sampled_trees
        ]

        add_log(
            f"Starting RF computation: {len(names)} trees, "
            f"{len(translate_maps)} translate map(s), {n_taxa} taxa"
        )

        try:
            import time as _time
            t0 = _time.time()

            from .rf.rf import rf_distance_from_newicks, matrix_to_dict

            result_names, matrix = rf_distance_from_newicks(
                names, newicks, translate_maps, map_indices, rooted=False,
            )
            elapsed = _time.time() - t0
            add_log(
                f"RF computation complete: {len(result_names)} trees, "
                f"{elapsed:.2f}s"
            )
        except Exception as e:
            msg = f"RF computation failed: {e}"
            add_log(msg, "ERROR")
            return dmc.Notification(
                title="RF Computation Error",
                message=msg,
                color="red",
                action="show",
                autoClose=6000,
                id="compute-rf-notification",
            ), dmc.Text(msg, c="red"), no_update, no_update

        # Convert to dict-of-dicts and store in distmat-store (single matrix only)
        distmat_dict = matrix_to_dict(result_names, matrix)
        rf_filename = "RF_distances.tsv"

        stored_distmats = {rf_filename: distmat_dict}
        add_log(f"Stored RF distance matrix as '{rf_filename}' ({len(result_names)}x{len(result_names)})")

        notification = dmc.Notification(
            title="RF Distances Computed",
            message=f"Computed {len(result_names)}x{len(result_names)} RF distance matrix in {elapsed:.2f}s.",
            color="green",
            action="show",
            autoClose=3000,
            id="compute-rf-notification",
        )

        output_indicator = dmc.Alert(
            title="RF Distance Matrix",
            children=dmc.Text(
                f"{rf_filename}: {len(result_names)} x {len(result_names)} trees",
                size="sm",
            ),
            color="green",
            variant="light",
        )

        return notification, output_indicator, stored_distmats, False

    # Callback to downsample trees for a given file
    @callback(
        Output("tree-offset-store", "data", allow_duplicate=True),
        Output("sidebar-trees-display", "children", allow_duplicate=True),
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
                autoClose=4000,
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
        Output("sidebar-trees-display", "children", allow_duplicate=True),
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
            color="orange",
            action="show",
            autoClose=4000,
            id="reset-notification",
        )
        return stored_summaries, no_update, notification

    # ------ MDS SECTION ON COMPUTE TAB ------

    # Enable MDS button when a distance matrix is available
    @callback(
        Output("compute-mds-button", "disabled"),
        Output("mds-status-text", "children"),
        Input("distmat-store", "data"),
    )
    def toggle_mds_button(distmat_data):
        if not distmat_data:
            return True, dmc.Text("No distance matrix computed yet.", c="dimmed",style={"padding": "20px"})
        name = next(iter(distmat_data))
        df = pd.DataFrame(distmat_data[name])
        return False, dmc.Text(f"Distance matrix: {name} ({df.shape[0]}x{df.shape[1]})", c="green")

    # Compute MDS from distance matrix
    @callback(
        Output("mds-result-store", "data"),
        Output("compute-mds-output", "children"),
        Output("notifications-container", "children", allow_duplicate=True),
        Output("export-mds-button", "disabled"),
        Output("plot-config-store", "data", allow_duplicate=True),
        Input("compute-mds-button", "n_clicks"),
        State("distmat-store", "data"),
        prevent_initial_call=True,
    )
    def handle_compute_mds(n_clicks, distmat_data):
        if not n_clicks or not distmat_data:
            return no_update, no_update, no_update, no_update, no_update

        selected_distmat = next(iter(distmat_data))
        add_log(f"Computing MDS from {selected_distmat}...")

        distmat_df_data = distmat_data[selected_distmat]
        distmat_df = pd.DataFrame(distmat_df_data)
        add_log(f"Distance matrix {selected_distmat}: {distmat_df.shape[0]}x{distmat_df.shape[1]}")

        distance_matrix = distmat_df.values

        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            msg = f"Distance matrix is not square: {distance_matrix.shape}"
            add_log(msg, "ERROR")
            return no_update, dmc.Text(msg, c="red"), no_update, no_update, no_update

        try:
            import time as _time
            t0 = _time.time()
            n_components = min(6, distance_matrix.shape[0] - 1)
            add_log(f"Computing MDS with {n_components} components...")
            mds = MDS(n_components=n_components, metric='precomputed', random_state=42, verbose=1, n_init=1, init="random", normalized_stress="auto")
            embedding = mds.fit_transform(distance_matrix)
            elapsed = _time.time() - t0
            add_log(f"MDS completed in {elapsed:.2f}s: {embedding.shape[0]} points in {embedding.shape[1]}D space")
        except Exception as e:
            msg = f"MDS computation failed: {str(e)}"
            add_log(msg, "ERROR")
            return no_update, dmc.Text(msg, c="red"), no_update, no_update, no_update

        # Build MDS result dataframe
        tree_names = distmat_df.index.astype(str).tolist()
        mdscols = [f"MDS{i+1}" for i in range(n_components)]

        mds_df = pd.DataFrame(embedding, columns=mdscols)
        mds_df["tree"] = tree_names
        mds_df["group"] = mds_df["tree"].apply(lambda x: str(x).split("/")[0].strip())
        mds_df["group"] = mds_df["group"].astype(str)

        group_mapping = {val: idx for idx, val in enumerate(sorted(mds_df["group"].unique()))}
        mds_df["group_col"] = mds_df["group"].map(group_mapping)
        mds_df["treenum"] = mds_df.groupby("group").cumcount() + 1
        mds_df["size"] = 6

        mds_filename = selected_distmat.replace('.tsv', '_MDS.tsv')
        mds_df["file"] = mds_filename

        # Build metadata
        metadata = {
            "filename": mds_filename,
            "source_distmat": selected_distmat,
            "rows": len(mds_df),
            "dimensions": mdscols,
            "groups": mds_df["group"].unique().tolist(),
            "MIN_TREENUM": int(mds_df["treenum"].min()),
            "MAX_TREENUM": int(mds_df["treenum"].max()),
        }

        mds_result = {
            "metadata": metadata,
            "data": mds_df.to_dict("records"),
        }

        add_log(f"Created MDS result '{mds_filename}': {len(mds_df)} trees, {len(mds_df['group'].unique())} groups")

        n_groups = len(mds_df["group"].unique())
        output_indicator = dmc.Alert(
            title="MDS Embedding",
            children=dmc.Text(
                f"{mds_filename}: {len(mds_df)} points, {n_components}D, {n_groups} groups",
                size="sm",
            ),
            color="blue",
            variant="light",
        )

        notification = dmc.Notification(
            title="MDS Computed",
            message=f"MDS embedding from {selected_distmat}: {len(mds_df)} points, {n_components} dimensions in {elapsed:.2f}s.",
            color="green",
            action="show",
            autoClose=3000,
            id="compute-mds-notification",
        )

        return mds_result, output_indicator, notification, False, {}

    # ------ CLEAR DATA ------

    @callback(
        Output("distmat-store", "data", allow_duplicate=True),
        Output("plot-config-store", "data", allow_duplicate=True),
        Output("mds-result-store", "data", allow_duplicate=True),
        Output("plot-display", "children", allow_duplicate=True),
        Output("tree-offset-store", "data", allow_duplicate=True),
        Output("compute-rf-output", "children", allow_duplicate=True),
        Output("compute-mds-output", "children", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
        Output("export-rf-button", "disabled", allow_duplicate=True),
        Output("export-mds-button", "disabled", allow_duplicate=True),
        Output("sidebar-trees-display", "children", allow_duplicate=True),
        Output("rf-trace-store", "data", allow_duplicate=True),
        Output("lnl-trace-plot", "children", allow_duplicate=True),
        Output("rf-trace-plot", "children", allow_duplicate=True),
        Input("clear-data-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_uploads(n_clicks):
        if n_clicks:
            add_log("Data cleared")
            # Also clear the tree service database
            try:
                tree_service = get_tree_service()
                tree_service.db_manager.clear_trees()
            except Exception:
                pass
            notification = dmc.Notification(
                title="Data Cleared",
                message="All data and plots have been cleared.",
                color="blue",
                action="show",
                autoClose=3000,
                id="clear-notification",
            )
            empty_sidebar = dmc.Text(
                "No trees loaded. Click the upload button above to load a .trees file.",
                c="dimmed",
                size="sm",
                style={"padding": "10px"},
            )
            return (
                {},          # distmat-store
                {},          # plot-config-store
                None,        # mds-result-store
                html.Div(),  # plot-display
                {},          # tree-offset-store
                html.Div(),  # compute-rf-output
                html.Div(),  # compute-mds-output
                notification,
                True,        # export-rf-button disabled
                True,        # export-mds-button disabled
                empty_sidebar,   # sidebar-trees-display
                None,            # rf-trace-store
                html.Div(),      # lnl-trace-plot
                html.Div(),      # rf-trace-plot
            )
        return (no_update,) * 14

    # ------ EXPORT CALLBACKS ------

    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("export-rf-button", "n_clicks"),
        State("distmat-store", "data"),
        prevent_initial_call=True,
    )
    def export_rf_matrix(n_clicks, distmat_data):
        if not n_clicks or not distmat_data:
            return no_update
        rf_filename = next(iter(distmat_data))
        path = _save_file_dialog(default_filename=rf_filename)
        if not path:
            return no_update
        df = pd.DataFrame(distmat_data[rf_filename])
        df.to_csv(path, sep="\t")
        add_log(f"Exported RF distance matrix to {path}")
        return dmc.Notification(
            title="RF Matrix Exported",
            message=f"Saved to {path}",
            color="green",
            action="show",
            autoClose=3000,
            id="export-rf-notification",
        )

    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("export-mds-button", "n_clicks"),
        State("mds-result-store", "data"),
        prevent_initial_call=True,
    )
    def export_mds(n_clicks, mds_result):
        if not n_clicks or not mds_result or not mds_result.get("data"):
            return no_update
        metadata = mds_result["metadata"]
        path = _save_file_dialog(default_filename=metadata["filename"])
        if not path:
            return no_update
        mds_df = pd.DataFrame(mds_result["data"])
        cols = metadata["dimensions"] + ["tree", "group", "treenum"]
        export_df = mds_df[[c for c in cols if c in mds_df.columns]]
        export_df.to_csv(path, sep="\t", index=False)
        add_log(f"Exported MDS result to {path}")
        return dmc.Notification(
            title="MDS Exported",
            message=f"Saved to {path}",
            color="green",
            action="show",
            autoClose=3000,
            id="export-mds-notification",
        )

    # ------ LOAD RF / MDS FROM FILE ------

    @callback(
        Output("distmat-store", "data", allow_duplicate=True),
        Output("compute-rf-output", "children", allow_duplicate=True),
        Output("export-rf-button", "disabled", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
        Input("load-rf-button", "n_clicks"),
        State("distmat-store", "data"),
        prevent_initial_call=True,
    )
    def load_rf_matrix(n_clicks, stored_distmats):
        if not n_clicks:
            return no_update, no_update, no_update, no_update

        file_path = _open_tsv_dialog()
        if not file_path:
            return no_update, no_update, no_update, no_update

        try:
            df = pd.read_csv(file_path, sep="\t", index_col=0)
        except Exception as e:
            msg = f"Failed to read file: {e}"
            add_log(msg, "ERROR")
            return no_update, no_update, no_update, dmc.Notification(
                title="Load Error", message=msg, color="red",
                action="show", autoClose=8000, id="load-rf-notification",
            )

        # Validate tree names have group prefix
        all_names = list(df.index.astype(str)) + list(df.columns.astype(str))
        valid, err_msg = _validate_group_names(all_names)
        if not valid:
            add_log(f"RF load validation failed: {err_msg}", "ERROR")
            return no_update, no_update, no_update, dmc.Notification(
                title="Invalid Tree Names", message=err_msg, color="red",
                action="show", autoClose=8000, id="load-rf-notification",
            )

        filename = os.path.basename(file_path)
        stored_distmats = stored_distmats or {}
        stored_distmats[filename] = df.to_dict()
        add_log(f"Loaded RF distance matrix from {filename}: {df.shape[0]}x{df.shape[1]}")

        output_indicator = dmc.Alert(
            title="RF Distance Matrix",
            children=dmc.Text(
                f"{filename}: {df.shape[0]} x {df.shape[1]} trees",
                size="sm",
            ),
            color="green",
            variant="light",
        )

        notification = dmc.Notification(
            title="RF Matrix Loaded",
            message=f"Loaded {df.shape[0]}x{df.shape[1]} distance matrix from {filename}.",
            color="green",
            action="show",
            autoClose=3000,
            id="load-rf-notification",
        )

        return stored_distmats, output_indicator, False, notification

    @callback(
        Output("mds-result-store", "data", allow_duplicate=True),
        Output("compute-mds-output", "children", allow_duplicate=True),
        Output("export-mds-button", "disabled", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
        Input("load-mds-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def load_mds(n_clicks):
        if not n_clicks:
            return no_update, no_update, no_update, no_update

        file_path = _open_tsv_dialog()
        if not file_path:
            return no_update, no_update, no_update, no_update

        try:
            mds_df = pd.read_csv(file_path, sep="\t")
        except Exception as e:
            msg = f"Failed to read file: {e}"
            add_log(msg, "ERROR")
            return no_update, no_update, no_update, dmc.Notification(
                title="Load Error", message=msg, color="red",
                action="show", autoClose=8000, id="load-mds-notification",
            )

        if "group" not in mds_df.columns:
            msg = "MDS file must contain a 'group' column."
            add_log(msg, "ERROR")
            return no_update, no_update, no_update, dmc.Notification(
                title="Invalid MDS File", message=msg, color="red",
                action="show", autoClose=8000, id="load-mds-notification",
            )

        if mds_df["group"].isna().any() or (mds_df["group"].astype(str).str.strip() == "").any():
            msg = "The 'group' column must not contain empty values."
            add_log(msg, "ERROR")
            return no_update, no_update, no_update, dmc.Notification(
                title="Invalid MDS File", message=msg, color="red",
                action="show", autoClose=8000, id="load-mds-notification",
            )

        mds_df["group"] = mds_df["group"].astype(str)

        group_mapping = {val: idx for idx, val in enumerate(sorted(mds_df["group"].unique()))}
        mds_df["group_col"] = mds_df["group"].map(group_mapping)

        if "treenum" not in mds_df.columns:
            mds_df["treenum"] = mds_df.groupby("group").cumcount() + 1
        mds_df["size"] = 6

        mds_filename = os.path.basename(file_path)
        mds_df["file"] = mds_filename

        # Detect MDS dimension columns
        mdscols = [c for c in mds_df.columns if c.startswith("MDS")]
        if not mdscols:
            msg = "No MDS dimension columns found (expected columns starting with 'MDS')."
            add_log(msg, "ERROR")
            return no_update, no_update, no_update, dmc.Notification(
                title="Invalid MDS File", message=msg, color="red",
                action="show", autoClose=8000, id="load-mds-notification",
            )

        metadata = {
            "filename": mds_filename,
            "source_distmat": "loaded_from_file",
            "rows": len(mds_df),
            "dimensions": mdscols,
            "groups": mds_df["group"].unique().tolist(),
            "MIN_TREENUM": int(mds_df["treenum"].min()),
            "MAX_TREENUM": int(mds_df["treenum"].max()),
        }

        mds_result = {
            "metadata": metadata,
            "data": mds_df.to_dict("records"),
        }

        n_groups = len(mds_df["group"].unique())
        add_log(f"Loaded MDS from {mds_filename}: {len(mds_df)} points, {len(mdscols)}D, {n_groups} groups")

        output_indicator = dmc.Alert(
            title="MDS Embedding",
            children=dmc.Text(
                f"{mds_filename}: {len(mds_df)} points, {len(mdscols)}D, {n_groups} groups",
                size="sm",
            ),
            color="blue",
            variant="light",
        )

        notification = dmc.Notification(
            title="MDS Loaded",
            message=f"Loaded MDS from {mds_filename}: {len(mds_df)} points, {len(mdscols)} dimensions, {n_groups} groups.",
            color="green",
            action="show",
            autoClose=6000,
            id="load-mds-notification",
        )

        return mds_result, output_indicator, False, notification

    # ------- TREE SPACE TAB CALLBACKS ------

    # Auto-generate plot config when MDS result changes
    @callback(
        Output("plot-config-store", "data"),
        Input("mds-result-store", "data"),
    )
    def generate_plot_config_from_mds(mds_result):
        if not mds_result or not mds_result.get("data"):
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
                # show lines checkbox
                html.Div(
                    [
                        dmc.Checkbox(
                            label="Show lines",
                            id="show-lines-checkbox",
                            checked=True,
                        ),
                    ],
                    style={"width": "10%", "padding": "5px", "display": "flex", "align-items": "center"},
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
                    style={"width": "10%", "padding": "5px", "text-align": "center"},
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
            State("show-lines-checkbox", "checked"),
            State("plot-container", "children"),
            State("plot-config-store", "data")
        ],
        prevent_initial_call=True,
    )
    def update_graph_on_button_click(n_clicks, mds_selected, treenum_range, show_lines, current_plot, plot_config):
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

    # ------ DIAGNOSTICS TAB CALLBACKS ------

    @callback(
        Output("lnl-trace-plot", "children"),
        Input("tree-offset-store", "data"),
    )
    def update_lnl_trace(stored_summaries):
        """Render log-likelihood trace plot when trees are loaded/changed."""
        if not stored_summaries:
            return dmc.Text(
                "No trees loaded yet.",
                c="dimmed", size="sm", style={"padding": "20px"},
            )

        tree_service = get_tree_service()
        file_sources = list(stored_summaries.keys())
        traces = tree_service.get_metadata_traces(file_sources)

        if not traces:
            return dmc.Text(
                "No log-likelihood data found in tree annotations.",
                c="dimmed", size="sm", style={"padding": "20px"},
            )

        # Pick first available field (prefer lnP > lnL > posterior > joint > loglikelihood)
        preferred = ['lnP', 'lnL', 'posterior', 'joint', 'loglikelihood']
        field_name = None
        for f in preferred:
            if f in traces:
                field_name = f
                break
        if field_name is None:
            field_name = next(iter(traces))

        trace_df = traces[field_name]
        groups = sorted(trace_df['group'].unique().tolist())
        colors = px.colors.qualitative.Dark24[:len(groups)]
        color_map = dict(zip(groups, colors))

        fig = go.Figure()
        for group in groups:
            gdf = trace_df[trace_df['group'] == group]
            fig.add_trace(go.Scatter(
                x=gdf['treenum'],
                y=gdf['value'],
                mode='lines',
                name=group,
                line=dict(color=color_map[group], width=1),
            ))

        fig.update_layout(
            xaxis_title="Tree number",
            yaxis_title=field_name,
            margin=dict(l=60, r=20, t=30, b=40),
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return dcc.Graph(figure=fig, config={"displayModeBar": False})

    @callback(
        Output("compute-rf-trace-button", "disabled"),
        Output("rf-reference-select", "data"),
        Input("tree-offset-store", "data"),
    )
    def toggle_rf_trace_controls(stored_summaries):
        """Enable/disable RF trace controls based on loaded trees."""
        if not stored_summaries:
            return True, [
                {"value": "last", "label": "Last tree (default)"},
                {"value": "first", "label": "First tree"},
            ]

        options = [
            {"value": "last", "label": "Last tree (default)"},
            {"value": "first", "label": "First tree"},
        ]
        return False, options

    @callback(
        Output("rf-trace-plot", "children"),
        Output("rf-trace-store", "data"),
        Output("notifications-container", "children", allow_duplicate=True),
        Input("compute-rf-trace-button", "n_clicks"),
        State("tree-offset-store", "data"),
        State("rf-reference-select", "value"),
        prevent_initial_call=True,
    )
    def compute_rf_trace(n_clicks, stored_summaries, ref_choice):
        """Compute RF distance of every tree to a reference tree."""
        if not n_clicks or not stored_summaries:
            return no_update, no_update, no_update

        import time as _time

        tree_service = get_tree_service()
        file_sources = list(stored_summaries.keys())

        add_log(f"Computing RF trace to {ref_choice} tree...")

        all_records = []

        for file_source in file_sources:
            summary = stored_summaries[file_source]
            total = summary["total_trees"]

            # Get reference tree
            if ref_choice == "first":
                # Get first tree by id order
                tree_service.db_manager.flush()
                df = tree_service.db_manager._trees
                file_df = df[df['file_source'] == file_source].sort_values('id')
                if len(file_df) == 0:
                    continue
                first_row = file_df.iloc[:1]
                ref_trees = tree_service.db_manager._resolve_newick(first_row)
            else:
                ref_trees = tree_service.db_manager.get_last_trees(file_source, limit=1)

            if not ref_trees:
                add_log(f"No reference tree found for {file_source}", "WARNING")
                continue

            ref_tree = ref_trees[0]
            ref_newick = ref_tree['newick']

            # Get all tree IDs for this file (in order)
            tree_service.db_manager.flush()
            file_df = tree_service.db_manager._trees[
                tree_service.db_manager._trees['file_source'] == file_source
            ].sort_values('id')

            tree_ids = file_df['id'].tolist()
            names = file_df['name'].tolist()
            groups = file_df['group_name'].tolist()
            n_trees = len(tree_ids)

            # Build translate map info
            tmap = tree_service.db_manager.get_translate_map(file_source)
            translate_maps = [tmap] if tmap else [{}]
            map_indices = [0] * n_trees

            # Get newick iterator
            newick_iter = tree_service.db_manager.iter_newicks(tree_ids)

            try:
                from .rf.rf import rf_distance_to_reference

                t0 = _time.time()
                _, distances = rf_distance_to_reference(
                    names, newick_iter, n_trees, ref_newick,
                    translate_maps, map_indices,
                    ref_map_index=0, rooted=False,
                )
                elapsed = _time.time() - t0
                add_log(f"RF trace for {file_source}: {n_trees} trees in {elapsed:.2f}s")
            except Exception as e:
                msg = f"RF trace computation failed for {file_source}: {e}"
                add_log(msg, "ERROR")
                return dmc.Text(msg, c="red"), no_update, dmc.Notification(
                    title="RF Trace Error", message=msg, color="red",
                    action="show", autoClose=6000, id="rf-trace-notification",
                )

            for i in range(n_trees):
                all_records.append({
                    'treenum': i + 1,
                    'rf_distance': int(distances[i]),
                    'group': groups[i],
                    'name': names[i],
                    'file_source': file_source,
                })

        if not all_records:
            return dmc.Text("No trees available for RF trace.", c="dimmed"), no_update, no_update

        trace_df = pd.DataFrame(all_records)

        # Build plot
        all_groups = sorted(trace_df['group'].unique().tolist())
        colors = px.colors.qualitative.Dark24[:len(all_groups)]
        color_map = dict(zip(all_groups, colors))

        fig = go.Figure()
        for group in all_groups:
            gdf = trace_df[trace_df['group'] == group]
            fig.add_trace(go.Scatter(
                x=gdf['treenum'],
                y=gdf['rf_distance'],
                mode='lines',
                name=group,
                line=dict(color=color_map[group], width=1),
            ))

        ref_label = "last" if ref_choice == "last" else "first"
        fig.update_layout(
            xaxis_title="Tree number",
            yaxis_title=f"RF distance to {ref_label} tree",
            margin=dict(l=60, r=20, t=30, b=40),
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        notification = dmc.Notification(
            title="RF Trace Computed",
            message=f"Computed RF distances for {len(all_records)} trees to {ref_label} tree.",
            color="green",
            action="show",
            autoClose=3000,
            id="rf-trace-notification",
        )

        # Store result for potential reuse
        store_data = trace_df.to_dict("records")

        return (
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            store_data,
            notification,
        )
