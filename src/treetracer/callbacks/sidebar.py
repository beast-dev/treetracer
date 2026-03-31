from dash import html, callback, clientside_callback, Input, Output, State, no_update, ALL, ctx
import dash_mantine_components as dmc
import os

from ..logger import add_log
from ..db.tree_service import get_tree_service
from ..state import clear_all_distmats
from ._helpers import _open_file_dialog


def register_sidebar_callbacks():
    # Show loading overlay instantly when Load Trees is clicked
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

        # If a file with the same name is already loaded, append _N suffix before the extension
        original_filename = filename
        if filename in stored_summaries:
            name_base, name_ext = os.path.splitext(filename)
            n = 2
            while f"{name_base}_{n}{name_ext}" in stored_summaries:
                n += 1
            filename = f"{name_base}_{n}{name_ext}"
            add_log(f"File with name '{original_filename}' already loaded. Using '{filename}' as group name.", "WARNING")

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

                rename_note = ""
                if filename != original_filename:
                    rename_note = f" (renamed from '{original_filename}')"

                if taxa_warning:
                    notification = dmc.Notification(
                        title="Trees Loaded — Taxa Mismatch",
                        message=f"Loaded {result['trees_loaded']} trees as '{filename}'{rename_note}. WARNING: {taxa_warning}",
                        color="yellow",
                        action="show",
                        autoClose=5000,
                        id="load-notification",
                    )
                else:
                    notification = dmc.Notification(
                        title="Trees Loaded" if not rename_note else "Trees Loaded (Renamed)",
                        message=f"Loaded {result['trees_loaded']} trees as '{filename}'{rename_note}.",
                        color="green" if not rename_note else "yellow",
                        action="show",
                        autoClose=5000 if rename_note else 3000,
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
        Output("compute-rf-trace-button", "disabled", allow_duplicate=True),
        Output("export-lnl-trace-button", "disabled", allow_duplicate=True),
        Output("export-rf-trace-button", "disabled", allow_duplicate=True),
        Output("within-run-mds-results-store", "data", allow_duplicate=True),
        Output("within-run-treenum-range-store", "data", allow_duplicate=True),
        Output("within-run-controls-paper", "style", allow_duplicate=True),
        Output("within-run-info", "children", allow_duplicate=True),
        Output("compute-wr-mds-output", "children", allow_duplicate=True),
        Output("within-run-graph", "figure", allow_duplicate=True),
        Output("within-run-selected-trees-store", "data", allow_duplicate=True),
        Input("clear-data-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_uploads(n_clicks):
        if n_clicks:
            add_log("Data cleared")
            # Clear all server-side distance matrices from disk
            clear_all_distmats()
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
                True,            # compute-rf-trace-button disabled
                True,            # export-lnl-trace-button disabled
                True,            # export-rf-trace-button disabled
                {},              # within-run-mds-results-store (empty dict)
                None,            # within-run-treenum-range-store
                {"display": "none"},  # within-run-controls-paper style
                html.Div(),      # within-run-info
                html.Div(),      # compute-wr-mds-output
                {},              # within-run-graph (empty figure)
                [],              # within-run-selected-trees-store
            )
        return (no_update,) * 24
