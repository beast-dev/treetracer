from dash import html, callback, clientside_callback, Input, Output, State, no_update, ALL, ctx
import dash_mantine_components as dmc
import os

from ..logger import add_log, notif_id
from ..db.tree_service import get_tree_service
from ..state import (
    clear_all_analysis_results,
    clear_all_consensus_trees,
    clear_all_distmats,
    clear_all_mds_results,
)
from ..plot_utils import placeholder_fig
from .clade_explore import clear_clade_freq_caches, _tanglegram_placeholder_fig
from ._helpers import _open_file_dialog


MAX_TREE_FILES = 10


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

        file_paths = _open_file_dialog()
        if not file_paths:
            return no_update, no_update, no_update, no_update, no_update, False

        stored_summaries = stored_summaries or {}
        if len(stored_summaries) >= MAX_TREE_FILES:
            msg = (
                f"Tree file limit reached ({MAX_TREE_FILES}). "
                "Remove a loaded file or clear data before loading more."
            )
            add_log(msg, "WARNING")
            notification = dmc.Notification(
                title="Tree File Limit Reached",
                message=msg,
                color="yellow",
                action="show",
                autoClose=5000,
                id=notif_id(),
            )
            return no_update, msg, {"display": "block"}, no_update, notification, False

        tree_service = get_tree_service()
        loaded_count = 0
        errors = []
        taxa_warnings = []
        limit_skips = []

        for file_path in file_paths:
            filename = os.path.basename(file_path)

            if not file_path.endswith((".trees", ".t")):
                errors.append(f"'{filename}' is not a .trees or .t file")
                add_log(f"Skipped {filename}: not a .trees or .t file", "WARNING")
                continue

            if len(stored_summaries) >= MAX_TREE_FILES:
                limit_skips.append(filename)
                add_log(
                    f"Skipped {filename}: tree file limit "
                    f"({MAX_TREE_FILES}) already reached",
                    "WARNING",
                )
                continue

            # Deduplicate names
            original_filename = filename
            if filename in stored_summaries:
                name_base, name_ext = os.path.splitext(filename)
                n = 2
                while f"{name_base}_{n}{name_ext}" in stored_summaries:
                    n += 1
                filename = f"{name_base}_{n}{name_ext}"
                add_log(f"'{original_filename}' already loaded. Using '{filename}'.", "WARNING")

            add_log(f"Loading {filename}...")

            try:
                result = tree_service.load_nexus_file(file_path, file_source=filename)

                if not result["success"]:
                    errors.append(f"'{filename}': {result.get('error', 'Unknown error')}")
                    add_log(f"Error loading {filename}: {result.get('error')}", "ERROR")
                    continue

                summary = tree_service.compute_file_summary(filename)
                new_translate = tree_service.db_manager.get_translate_map(filename)

                stored_summaries[filename] = {
                    "total_trees": summary["total_trees"],
                    "original_total": summary["total_trees"],  # preserved across burn-in / downsample
                    "n_taxa": len(new_translate) if new_translate else 0,
                    "groups": summary["groups"],
                    "trees_per_group": summary["trees_per_group"],
                    "path": file_path,
                    "burnin": 0,
                    "is_rooted": summary["is_rooted"],
                }
                loaded_count += 1
                rooting_label = (
                    "ROOTED" if summary["is_rooted"] else "UNROOTED"
                )
                n_taxa = len(new_translate) if new_translate else 0
                taxa_text = f", {n_taxa} taxa" if n_taxa else ""
                add_log(
                    f"Loaded {filename}: {result['trees_loaded']} trees"
                    f"{taxa_text}; trees detected as {rooting_label}."
                )

                # Check taxa mismatch
                if new_translate:
                    new_taxa = set(new_translate.values())
                    for other_file in stored_summaries:
                        if other_file == filename:
                            continue
                        other_translate = tree_service.db_manager.get_translate_map(other_file)
                        if other_translate and set(other_translate.values()) != new_taxa:
                            taxa_warnings.append(f"{filename} vs {other_file}")
                            break

            except Exception as e:
                errors.append(f"'{filename}': {str(e)}")
                add_log(f"Error processing {filename}: {e}", "ERROR")

        if loaded_count == 0 and (errors or limit_skips):
            msg_parts = []
            if errors:
                msg_parts.append("Failed to load: " + "; ".join(errors))
            if limit_skips:
                msg_parts.append(
                    f"TreeTracer supports up to {MAX_TREE_FILES} loaded "
                    f"tree files; skipped {len(limit_skips)} file(s)."
                )
            msg = " ".join(msg_parts)
            notification = dmc.Notification(
                title="No Trees Loaded",
                message=msg,
                color="yellow",
                action="show",
                autoClose=5000,
                id=notif_id(),
            )
            return no_update, msg, {"display": "block"}, no_update, notification, False

        # Build notification
        msg_parts = [f"Loaded {loaded_count} file(s)"]
        if errors:
            msg_parts.append(f"{len(errors)} failed")
        if limit_skips:
            msg_parts.append(f"{len(limit_skips)} skipped at limit")
        if taxa_warnings:
            msg_parts.append("taxa mismatch detected")
            add_log(f"Taxa mismatch: {', '.join(taxa_warnings)}", "WARNING")

        color = "green"
        if taxa_warnings or errors or limit_skips:
            color = "yellow"

        notification_message = f"Successfully loaded {loaded_count} .trees file(s)."
        if limit_skips:
            notification_message += (
                f" Maximum loaded tree files: {MAX_TREE_FILES}."
            )
        notification = dmc.Notification(
            title=" — ".join(msg_parts),
            message=notification_message,
            color=color,
            action="show",
            autoClose=4000,
            id=notif_id(),
        )
        return stored_summaries, "", {"display": "none"}, no_update, notification, False

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

        def _stat_row(label, value):
            """One label/value row for a card's stat block — dimmed
            label on the left, value right-aligned."""
            return dmc.Group(
                [
                    dmc.Text(label, size="xs", c="dimmed"),
                    dmc.Text(value, size="xs", fw=500),
                ],
                justify="space-between",
                gap="xs",
            )

        items = []
        for filename, summary in stored_summaries.items():
            total_trees = int(summary["total_trees"])
            original_total = int(summary.get("original_total", total_trees))
            n_taxa = int(summary.get("n_taxa", 0))
            burnin_total = int(summary.get("burnin", 0))
            is_rooted = summary.get("is_rooted", True)

            # Rooting badge — defaults to "rooted" for pre-feature
            # summaries that don't carry the flag.
            rooting_badge = dmc.Badge(
                "rooted" if is_rooted else "unrooted",
                size="sm",
                variant="light",
                color="blue" if is_rooted else "violet",
            )

            # Tree count — folds in the original count once burn-in or
            # downsampling has changed it.
            if total_trees == original_total:
                trees_label = f"{total_trees:,} trees"
            else:
                trees_label = f"{total_trees:,} of {original_total:,} trees"

            # Header: the filename is the card's identity. It's clamped
            # to 2 lines while the card is collapsed, and shown in full
            # (wrapped, no truncation) once expanded — the un-clamp is
            # done by ``.tt-card-filename`` in assets/treetracer.css,
            # keyed off the accordion control's ``aria-expanded``. A
            # quiet meta line (rooting + tree count) sits underneath.
            header_content = dmc.Stack(
                [
                    dmc.Text(
                        filename,
                        size="sm",
                        fw=600,
                        className="tt-card-filename",
                    ),
                    dmc.Group(
                        [
                            rooting_badge,
                            dmc.Text(trees_label, size="xs", c="dimmed"),
                        ],
                        gap="xs",
                    ),
                ],
                gap=4,
            )

            # Body: a compact stat list, then the burn-in / downsample /
            # reset controls.
            panel_content = dmc.Stack([
                _stat_row("Taxa", f"{n_taxa:,}"),
                _stat_row("Burn-in dropped", f"{burnin_total:,}"),
                dmc.Divider(my="xs"),
                # Burn-in row — applied first; drops the first N trees by MCMC order.
                dmc.Group([
                    dmc.NumberInput(
                        id={"type": "burnin-input", "index": filename},
                        value=0,
                        min=0,
                        step=1,
                        size="xs",
                        style={"width": "80px"},
                        placeholder="Burn-in",
                    ),
                    dmc.Button(
                        "Apply Burn-in",
                        id={"type": "burnin-btn", "index": filename},
                        variant="light",
                        color="grape",
                        size="compact-xs",
                    ),
                ], gap="xs"),
                # Downsample row — operates on whatever's left after burn-in.
                dmc.Group([
                    dmc.NumberInput(
                        id={"type": "downsample-input", "index": filename},
                        value=1000,
                        min=1,
                        step=1,
                        size="xs",
                        style={"width": "80px"},
                        placeholder="Downsample",
                    ),
                    dmc.Button(
                        "Downsample",
                        id={"type": "downsample-btn", "index": filename},
                        variant="light",
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
                        # Header row: a Remove (✕) button to the LEFT of
                        # the accordion control, opposite the expand
                        # chevron. The ✕ is a sibling of — not nested
                        # inside — AccordionControl, so a click removes
                        # the file without toggling the card.
                        # ``minWidth: 0`` lets the control's flex column
                        # shrink to the sidebar width.
                        dmc.Group(
                            [
                                dmc.ActionIcon(
                                    "✕",
                                    id={"type": "remove-file-btn",
                                        "index": filename},
                                    variant="subtle",
                                    color="gray",
                                    size="md",
                                    style={"flexShrink": 0},
                                ),
                                dmc.AccordionControl(
                                    header_content,
                                    style={"flex": 1, "minWidth": 0},
                                ),
                            ],
                            gap="xs",
                            wrap="nowrap",
                            align="center",
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
            value=[],
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
            add_log(msg, "WARNING")
            notification = dmc.Notification(
                title="Downsample Skipped",
                message=msg,
                color="yellow",
                action="show",
                autoClose=4000,
                id=notif_id(),
            )
            return no_update, no_update, notification

        add_log(f"Downsampling {filename} to {n} trees...")

        tree_service = get_tree_service()
        tree_service.db_manager.downsample_trees(filename, n)

        # Recompute summary from the DataFrame
        summary = tree_service.compute_file_summary(filename)
        if filename in stored_summaries:
            stored_summaries[filename].update(summary)

        add_log(f"Downsampled {filename} to {summary['total_trees']} trees")
        notification = dmc.Notification(
            title="Trees Downsampled",
            message=f"Downsampled {filename} to {summary['total_trees']} trees.",
            color="orange",
            action="show",
            autoClose=4000,
            id=notif_id(),
        )
        return stored_summaries, no_update, notification

    # Callback to apply burn-in to a file (drops the first N trees by MCMC order)
    @callback(
        Output("tree-offset-store", "data", allow_duplicate=True),
        Output("sidebar-trees-display", "children", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
        Input({"type": "burnin-btn", "index": ALL}, "n_clicks"),
        State({"type": "burnin-input", "index": ALL}, "value"),
        State("tree-offset-store", "data"),
        prevent_initial_call=True,
    )
    def apply_burnin(n_clicks_list, n_value_list, stored_summaries):
        if not any(n_clicks_list):
            return no_update, no_update, no_update

        triggered_id = ctx.triggered_id
        if not triggered_id:
            return no_update, no_update, no_update

        filename = triggered_id["index"]

        n_value = None
        for i, inp in enumerate(ctx.inputs_list[0]):
            if inp["id"]["index"] == filename:
                n_value = n_value_list[i]
                break

        if not n_value or int(n_value) <= 0:
            return no_update, no_update, no_update

        n = int(n_value)
        stored_summaries = stored_summaries or {}
        current_total = stored_summaries.get(filename, {}).get("total_trees", 0)
        if n >= current_total:
            msg = (
                f"Burn-in of {n} would drop all {current_total} remaining trees "
                f"in {filename}. Skipped."
            )
            add_log(msg, "WARNING")
            notification = dmc.Notification(
                title="Burn-in Skipped",
                message=msg,
                color="yellow",
                action="show",
                autoClose=4000,
                id=notif_id(),
            )
            return no_update, no_update, notification

        add_log(f"Applying burn-in of {n} to {filename}...")

        tree_service = get_tree_service()
        tree_service.db_manager.apply_burnin(filename, n)

        summary = tree_service.compute_file_summary(filename)
        if filename in stored_summaries:
            prev_burnin = stored_summaries[filename].get("burnin", 0)
            stored_summaries[filename].update(summary)
            stored_summaries[filename]["burnin"] = prev_burnin + n

        add_log(
            f"Burn-in applied: {filename} now has {summary['total_trees']} trees "
            f"(cumulative burn-in: {stored_summaries[filename]['burnin']})"
        )
        notification = dmc.Notification(
            title="Burn-in Applied",
            message=(
                f"Dropped {n} trees from {filename}; "
                f"{summary['total_trees']} remaining "
                f"(cumulative burn-in: {stored_summaries[filename]['burnin']})."
            ),
            color="grape",
            action="show",
            autoClose=4000,
            id=notif_id(),
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

        add_log(f"Resetting {filename}...")

        tree_service = get_tree_service()

        # Clear existing trees for this file and reload from disk
        tree_service.db_manager.clear_trees(file_source=filename)
        result = tree_service.load_nexus_file(file_path, file_source=filename)

        if not result["success"]:
            add_log(f"Reset failed for {filename}: {result.get('error')}", "ERROR")
            return no_update, no_update, no_update

        # Recompute summary; reset clears any accumulated burn-in and
        # re-baselines original_total to the freshly-reloaded count.
        summary = tree_service.compute_file_summary(filename)
        stored_summaries[filename].update(summary)
        stored_summaries[filename]["burnin"] = 0
        stored_summaries[filename]["original_total"] = summary["total_trees"]

        add_log(f"Reset {filename}: reloaded {summary['total_trees']} trees from disk")
        notification = dmc.Notification(
            title="Trees Reset",
            message=f"Reloaded {summary['total_trees']} trees from {filename}.",
            color="orange",
            action="show",
            autoClose=4000,
            id=notif_id(),
        )
        return stored_summaries, no_update, notification

    # Callback to remove a single loaded file (its trees + sidebar card).
    # Computed RF / MDS / consensus tree results are intentionally left intact —
    # they're self-contained snapshots; "Clear Data" is the wipe-all path.
    @callback(
        Output("tree-offset-store", "data", allow_duplicate=True),
        Output("sidebar-trees-display", "children", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
        Input({"type": "remove-file-btn", "index": ALL}, "n_clicks"),
        State("tree-offset-store", "data"),
        prevent_initial_call=True,
    )
    def remove_file(n_clicks_list, stored_summaries):
        if not any(n_clicks_list):
            return no_update, no_update, no_update

        triggered_id = ctx.triggered_id
        if not triggered_id:
            return no_update, no_update, no_update

        filename = triggered_id["index"]
        stored_summaries = stored_summaries or {}
        if filename not in stored_summaries:
            return no_update, no_update, no_update

        add_log(f"Removing {filename}...")

        tree_service = get_tree_service()
        tree_service.db_manager.clear_trees(file_source=filename)
        del stored_summaries[filename]

        add_log(f"Removed {filename}")
        notification = dmc.Notification(
            title="File Removed",
            message=f"Removed {filename} from loaded trees.",
            color="blue",
            action="show",
            autoClose=4000,
            id=notif_id(),
        )
        return stored_summaries, no_update, notification

    # ------ CLEAR DATA ------

    @callback(
        Output("distmat-store", "data", allow_duplicate=True),
        Output("plot-config-store", "data", allow_duplicate=True),
        Output("mds-result-store", "data", allow_duplicate=True),
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
        Output("within-run-treenum-range-store", "data", allow_duplicate=True),
        Output("within-run-controls-paper", "style", allow_duplicate=True),
        Output("within-run-info", "children", allow_duplicate=True),
        Output("within-run-graph", "figure", allow_duplicate=True),
        Output("within-run-selected-trees-store", "data", allow_duplicate=True),
        Output("pseudo-ess-output", "children", allow_duplicate=True),
        Output("consensus-tree-registry-store", "data", allow_duplicate=True),
        Output("treespace-selected-trees-store", "data", allow_duplicate=True),
        # Clade Frequency Comparison surface — scatter, tanglegram,
        # the three stores backing them, and both consensus tree dropdown values.
        Output("clade-freq-plot", "children", allow_duplicate=True),
        Output("clade-freq-tanglegram", "figure", allow_duplicate=True),
        Output("clade-freq-tanglegram-title", "children", allow_duplicate=True),
        Output("clade-freq-data-store", "data", allow_duplicate=True),
        Output("clade-freq-click-store", "data", allow_duplicate=True),
        Output("clade-freq-result-key-store", "data", allow_duplicate=True),
        Output("clade-freq-tanglegram-pair-store", "data", allow_duplicate=True),
        Output("clade-freq-consensus-tree-select-1", "value", allow_duplicate=True),
        Output("clade-freq-consensus-tree-select-2", "value", allow_duplicate=True),
        Output("clade-freq-output-paper", "style", allow_duplicate=True),
        # Clear every browser job identity with the server-side record. Those
        # store changes wake the central reconciler, which alone settles its
        # interval and busy state after reset.
        Output("rf-job-store", "data", allow_duplicate=True),
        Output("mds-job-store", "data", allow_duplicate=True),
        Output("pseudo-ess-job-store", "data", allow_duplicate=True),
        Output("consensus-job-store", "data", allow_duplicate=True),
        Output("rf-trace-job-store", "data", allow_duplicate=True),
        Output("clade-freq-job-store", "data", allow_duplicate=True),
        Output("rf-progress-path", "data", allow_duplicate=True),
        Output("mds-progress-path", "data", allow_duplicate=True),
        Output("treespace-loading-overlay", "visible", allow_duplicate=True),
        Output("within-run-loading-overlay", "visible", allow_duplicate=True),
        Input("clear-data-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_uploads(n_clicks):
        if n_clicks:
            add_log("Data cleared")
            # Cancel any in-flight subprocess job — descriptors point
            # into the DB we're about to wipe, and we don't want a late
            # finalizer to publish results into freshly cleared state.
            from . import compute
            resetters = (("managed", compute.reset),)
            for label, resetter in resetters:
                try:
                    resetter()
                except Exception as exc:
                    add_log(
                        f"Could not reset {label} computation: {exc}",
                        "WARNING",
                    )
            # Clear all server-side distance matrices from disk
            clear_all_distmats()
            clear_all_mds_results()
            clear_all_consensus_trees()
            clear_all_analysis_results()
            # Wipe the in-process clade-freq caches (parsed NEXUS
            # trees, tanglegram layouts, click→split lookup). They're
            # keyed on consensus tree uuids that no longer exist after the calls
            # above and would otherwise return stale data.
            try:
                clear_clade_freq_caches()
            except Exception:
                pass
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
                id=notif_id(),
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
                {},          # mds-result-store
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
                None,            # within-run-treenum-range-store
                {"display": "none"},  # within-run-controls-paper style
                html.Div(),      # within-run-info
                placeholder_fig(  # within-run-graph: same empty-state as between-run
                    "No MDS result selected. Compute an MDS in the Compute tab."
                ),
                [],              # within-run-selected-trees-store
                html.Div(),      # pseudo-ess-output
                [],              # consensus-tree-registry-store
                [],              # treespace-selected-trees-store
                html.Div(),      # clade-freq-plot (scatter)
                # tanglegram placeholder fig — kept in sync with the
                # initial figure defined in ui.py's _add_clade_freq_panel
                # via the diagnostics helper.
                _tanglegram_placeholder_fig(),
                None,            # clade-freq-tanglegram-title.children
                None,            # clade-freq-data-store
                None,            # clade-freq-click-store
                None,            # clade-freq-result-key-store
                None,            # clade-freq-tanglegram-pair-store
                None,            # clade-freq-consensus-tree-select-1.value
                None,            # clade-freq-consensus-tree-select-2.value
                {"display": "none"},  # clade-freq-output-paper.style
                None,            # rf-job-store
                None,            # mds-job-store
                None,            # pseudo-ess-job-store
                None,            # consensus-job-store
                None,            # rf-trace-job-store
                None,            # clade-freq-job-store
                None,            # rf-progress-path
                None,            # mds-progress-path
                False,           # treespace-loading-overlay visible
                False,           # within-run-loading-overlay visible
            )
        return (no_update,) * 44
