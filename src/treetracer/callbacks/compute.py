from dash import html, callback, Input, Output, State, no_update, ALL
import dash_mantine_components as dmc
import os
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

from ..logger import add_log
from ..db.tree_service import get_tree_service
from ..state import save_distmat, load_distmat, load_distmat_as_lists, get_distmat_index, next_distmat_name
from ._helpers import _save_file_dialog, _open_tsv_dialog, _validate_group_names


# Separate-process computation — has its own GIL, so the main process stays responsive.
# The executor is created lazily to avoid spawning processes at import time.
_executor = None
_rf_future = None   # concurrent.futures.Future for RF job
_rf_meta = {}       # metadata needed by poll_completion to save RF result
_mds_future = None  # concurrent.futures.Future for MDS job
_mds_meta = {}      # metadata needed by poll_completion to build MDS result


def _get_executor():
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(max_workers=1)
    return _executor


def register_compute_callbacks():
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

    # Callback to validate taxa, extract data, and start RF computation in background
    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("compute-rf-output", "children"),
        Output("compute-poll-interval", "disabled"),
        Output("compute-rf-button", "disabled", allow_duplicate=True),
        Input("compute-rf-button", "n_clicks"),
        State({"type": "compute-tree-checkbox", "index": ALL}, "checked"),
        State({"type": "compute-tree-checkbox", "index": ALL}, "id"),
        State("tree-offset-store", "data"),
        prevent_initial_call=True,
    )
    def handle_compute_rf(n_clicks, checked_list, id_list, stored_summaries):
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

        # --- All taxa counts match — extract data then submit RF to subprocess ---
        n_taxa = unique_counts.pop()
        add_log(f"Taxa validation passed: all {len(selected_files)} files have {n_taxa} taxa")

        tree_service = get_tree_service()

        # Data extraction (disk I/O, fast — runs in main process)
        total_trees = sum(
            stored_summaries[f].get("total_trees", 0) for f in selected_files
        )
        add_log(f"Fetching all {total_trees} trees from {len(selected_files)} files...")

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
                title="RF Error", message=msg, color="red",
                action="show", autoClose=6000, id="compute-rf-notification",
            ), no_update, no_update, no_update

        names = [t["name"] for t in sampled_trees]
        newicks = tree_service.prepare_trees_for_rf_analysis(sampled_trees)

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

        # Build file breakdown: {filename: n_trees} for each source file
        file_breakdown = {}
        for t in sampled_trees:
            fs = t["file_source"]
            file_breakdown[fs] = file_breakdown.get(fs, 0) + 1

        rf_name = next_distmat_name()
        add_log(
            f"Starting RF computation ({rf_name}) in background process: {len(names)} trees, "
            f"{len(translate_maps)} translate map(s), {n_taxa} taxa"
        )

        # Store metadata for poll_completion
        _rf_meta["name"] = rf_name
        _rf_meta["file_breakdown"] = file_breakdown

        # Submit RF computation to a separate process (own GIL — main process stays free)
        from ..rf._worker import compute_rf
        global _rf_future
        _rf_future = _get_executor().submit(
            compute_rf, names, newicks, translate_maps, map_indices,
        )

        # Return immediately: show computing indicator, enable polling, disable button
        breakdown_str = ", ".join(f"{f}: {n}" for f, n in file_breakdown.items())
        computing_indicator = dmc.Alert(
            title=f"Computing RF Distances ({rf_name})...",
            children=dmc.Text(
                f"Computing {len(names)}x{len(names)} RF distance matrix in background. "
                f"Files: {breakdown_str}",
                size="sm",
            ),
            color="blue",
            variant="light",
        )
        return no_update, computing_indicator, False, True

    # ------ MDS SECTION ON COMPUTE TAB ------

    # Populate matrix selector and enable MDS button when distance matrices are available
    @callback(
        Output("compute-mds-button", "disabled"),
        Output("mds-status-text", "children"),
        Output("mds-distmat-select", "data"),
        Output("mds-distmat-select", "value"),
        Input("distmat-store", "data"),
    )
    def toggle_mds_button(distmat_data):
        if not distmat_data:
            return True, dmc.Text("No distance matrix computed yet.", c="dimmed", style={"padding": "20px"}), [], None
        options = [
            {"value": k, "label": f"{k} — {len(v['names'])} trees"}
            for k, v in distmat_data.items()
        ]
        last_key = list(distmat_data.keys())[-1]
        status = dmc.Text(f"{len(distmat_data)} distance matrix(es) available", c="green")
        return False, status, options, last_key

    # Show file breakdown badges when a matrix is selected
    @callback(
        Output("mds-distmat-info", "children"),
        Input("mds-distmat-select", "value"),
        State("distmat-store", "data"),
        prevent_initial_call=True,
    )
    def show_distmat_info(selected, distmat_data):
        if not selected or not distmat_data or selected not in distmat_data:
            return html.Div()
        breakdown = distmat_data[selected].get("file_breakdown", {})
        n = len(distmat_data[selected].get("names", []))
        badges = [
            dmc.Badge(f"{fname}: {count} trees", variant="light", color="blue", size="lg")
            for fname, count in breakdown.items()
        ]
        badges.append(dmc.Badge(f"Total: {n} trees", variant="light", color="grape", size="lg"))
        return dmc.Group(badges, gap="xs", mt="xs")

    # Start MDS computation in background process
    @callback(
        Output("compute-mds-output", "children"),
        Output("compute-poll-interval", "disabled", allow_duplicate=True),
        Output("compute-mds-button", "disabled", allow_duplicate=True),
        Input("compute-mds-button", "n_clicks"),
        State("mds-distmat-select", "value"),
        prevent_initial_call=True,
    )
    def handle_compute_mds(n_clicks, selected_distmat):
        if not n_clicks or not selected_distmat:
            return no_update, no_update, no_update

        try:
            tree_names, matrix_lists = load_distmat_as_lists(selected_distmat)
        except KeyError:
            msg = "Selected distance matrix not available. Please recompute RF distances."
            add_log(msg, "ERROR")
            return dmc.Text(msg, c="red"), no_update, no_update

        n = len(tree_names)
        n_components = min(6, n - 1)
        add_log(f"Computing MDS from {selected_distmat} ({n}x{n}) in background process...")

        # Store metadata for poll_completion to build the result
        _mds_meta["tree_names"] = tree_names
        _mds_meta["selected_distmat"] = selected_distmat
        _mds_meta["n_components"] = n_components

        # Submit to subprocess
        from ..rf._worker import compute_mds_worker
        global _mds_future
        _mds_future = _get_executor().submit(
            compute_mds_worker, matrix_lists, n_components,
        )

        computing_indicator = dmc.Alert(
            title="Computing MDS Embedding...",
            children=dmc.Text(
                f"Computing PCoA with {n_components} components for {n} trees in background.",
                size="sm",
            ),
            color="blue",
            variant="light",
        )
        return computing_indicator, False, True

    # ------ UNIFIED POLL CALLBACK (RF + MDS) ------
    # Single callback handles both RF and MDS completion to avoid
    # duplicate-output conflicts from two callbacks sharing the same Input.

    @callback(
        # RF outputs
        Output("compute-rf-output", "children", allow_duplicate=True),
        Output("distmat-store", "data", allow_duplicate=True),
        Output("export-rf-button", "disabled"),
        Output("compute-rf-trace-button", "disabled", allow_duplicate=True),
        Output("compute-rf-button", "disabled", allow_duplicate=True),
        # MDS outputs
        Output("mds-result-store", "data"),
        Output("compute-mds-output", "children", allow_duplicate=True),
        Output("export-mds-button", "disabled"),
        Output("plot-config-store", "data", allow_duplicate=True),
        Output("compute-mds-button", "disabled", allow_duplicate=True),
        # Shared outputs
        Output("notifications-container", "children", allow_duplicate=True),
        Output("compute-poll-interval", "disabled", allow_duplicate=True),
        Input("compute-poll-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def poll_completion(n_intervals):
        global _rf_future, _mds_future

        # --- defaults: no change ---
        rf_out = [no_update] * 5   # rf-output, distmat, export-rf, rf-trace-btn, rf-btn
        mds_out = [no_update] * 5  # mds-store, mds-output, export-mds, plot-config, mds-btn
        notif = no_update

        # --- Check RF ---
        if _rf_future is not None and _rf_future.done():
            try:
                result_names, matrix, elapsed = _rf_future.result()
            except Exception as e:
                msg = f"RF computation failed: {e}"
                add_log(msg, "ERROR")
                _rf_future = None
                rf_out = [
                    dmc.Text(msg, c="red"),  # compute-rf-output
                    no_update,               # distmat-store
                    no_update,               # export-rf-button
                    no_update,               # compute-rf-trace-button
                    False,                   # re-enable compute-rf-button
                ]
                notif = dmc.Notification(title="RF Computation Error", message=msg,
                                         color="red", action="show", autoClose=6000,
                                         id="compute-rf-notification")
            else:
                _rf_future = None
                rf_name = _rf_meta.get("name", "RF")
                file_breakdown = _rf_meta.get("file_breakdown", {})
                save_distmat(rf_name, result_names, matrix, file_breakdown=file_breakdown)
                add_log(f"Stored RF distance matrix as '{rf_name}' ({len(result_names)}x{len(result_names)})")
                add_log(f"RF computation took {elapsed:.2f}s")
                rf_out = [
                    dmc.Alert(title=f"RF Distance Matrix ({rf_name})",
                              children=dmc.Text(f"{len(result_names)} x {len(result_names)} trees", size="sm"),
                              color="green", variant="light"),
                    get_distmat_index(),
                    False,   # export-rf-button enabled
                    False,   # compute-rf-trace-button enabled
                    False,   # re-enable compute-rf-button
                ]
                notif = dmc.Notification(
                    title=f"RF Distances Computed ({rf_name})",
                    message=f"Computed {len(result_names)}x{len(result_names)} RF distance matrix in {elapsed:.2f}s.",
                    color="green", action="show", autoClose=3000, id="compute-rf-notification")

        # --- Check MDS ---
        if _mds_future is not None and _mds_future.done():
            try:
                embedding_list, elapsed = _mds_future.result()
            except Exception as e:
                msg = f"MDS computation failed: {e}"
                add_log(msg, "ERROR")
                _mds_future = None
                mds_out = [
                    no_update,
                    dmc.Text(msg, c="red"),
                    no_update, no_update,
                    False,   # re-enable mds button
                ]
                # Only overwrite notif if RF didn't already set one
                if notif is no_update:
                    notif = dmc.Notification(title="MDS Error", message=msg, color="red",
                                             action="show", autoClose=6000, id="compute-mds-notification")
            else:
                _mds_future = None
                tree_names = [str(n) for n in _mds_meta["tree_names"]]
                selected_distmat = _mds_meta["selected_distmat"]
                n_components = _mds_meta["n_components"]

                mdscols = [f"MDS{i+1}" for i in range(n_components)]
                mds_df = pd.DataFrame(embedding_list, columns=mdscols)
                mds_df["tree"] = tree_names
                mds_df["group"] = mds_df["tree"].apply(lambda x: str(x).split("/")[0].strip())
                mds_df["group"] = mds_df["group"].astype(str)
                group_mapping = {val: idx for idx, val in enumerate(sorted(mds_df["group"].unique()))}
                mds_df["group_col"] = mds_df["group"].map(group_mapping)
                mds_df["treenum"] = mds_df.groupby("group").cumcount() + 1
                mds_df["size"] = 6
                mds_filename = selected_distmat.replace('.tsv', '_MDS.tsv')
                mds_df["file"] = mds_filename

                metadata = {
                    "filename": mds_filename, "source_distmat": selected_distmat,
                    "rows": len(mds_df), "dimensions": mdscols,
                    "groups": mds_df["group"].unique().tolist(),
                    "MIN_TREENUM": int(mds_df["treenum"].min()),
                    "MAX_TREENUM": int(mds_df["treenum"].max()),
                }
                mds_result = {"metadata": metadata, "data": mds_df.to_dict("records")}

                n_groups = len(mds_df["group"].unique())
                add_log(f"PCoA completed in {elapsed:.2f}s: {len(mds_df)} points, {n_components}D, {n_groups} groups")

                mds_out = [
                    mds_result,
                    dmc.Alert(title="MDS Embedding",
                              children=dmc.Text(f"{mds_filename}: {len(mds_df)} points, {n_components}D, {n_groups} groups", size="sm"),
                              color="blue", variant="light"),
                    False,   # export-mds-button enabled
                    {},      # plot-config-store reset
                    False,   # re-enable mds button
                ]
                if notif is no_update:
                    notif = dmc.Notification(
                        title="MDS Computed",
                        message=f"PCoA: {len(mds_df)} points, {n_components}D in {elapsed:.2f}s.",
                        color="green", action="show", autoClose=3000, id="compute-mds-notification")

        # --- Disable interval only when no jobs are pending ---
        any_running = ((_rf_future is not None and not _rf_future.done()) or
                       (_mds_future is not None and not _mds_future.done()))
        should_disable = not any_running and (rf_out[0] is not no_update or mds_out[0] is not no_update)
        # If nothing finished this tick, keep interval as-is
        poll_disabled = should_disable if (rf_out[0] is not no_update or mds_out[0] is not no_update) else no_update

        return (*rf_out, *mds_out, notif, poll_disabled)

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
        try:
            names, arr = load_distmat(rf_filename)
        except KeyError:
            add_log("No matrix available for export.", "ERROR")
            return no_update
        df = pd.DataFrame(arr, index=names, columns=names)
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
        Output("compute-rf-trace-button", "disabled", allow_duplicate=True),
        Input("load-rf-button", "n_clicks"),
        State("distmat-store", "data"),
        prevent_initial_call=True,
    )
    def load_rf_matrix(n_clicks, stored_distmats):
        if not n_clicks:
            return no_update, no_update, no_update, no_update, no_update

        file_path = _open_tsv_dialog()
        if not file_path:
            return no_update, no_update, no_update, no_update, no_update

        try:
            df = pd.read_csv(file_path, sep="\t", index_col=0)
        except Exception as e:
            msg = f"Failed to read file: {e}"
            add_log(msg, "ERROR")
            return no_update, no_update, no_update, dmc.Notification(
                title="Load Error", message=msg, color="red",
                action="show", autoClose=8000, id="load-rf-notification",
            ), no_update

        # Validate tree names have group prefix
        all_names = list(df.index.astype(str)) + list(df.columns.astype(str))
        valid, err_msg = _validate_group_names(all_names)
        if not valid:
            add_log(f"RF load validation failed: {err_msg}", "ERROR")
            return no_update, no_update, no_update, dmc.Notification(
                title="Invalid Tree Names", message=err_msg, color="red",
                action="show", autoClose=8000, id="load-rf-notification",
            ), no_update

        filename = os.path.basename(file_path)
        names = list(df.index.astype(str))
        # Build file breakdown from group prefixes in tree names
        file_breakdown = {}
        for n in names:
            group = n.split("/")[0] if "/" in n else "(ungrouped)"
            file_breakdown[group] = file_breakdown.get(group, 0) + 1
        save_distmat(filename, names, df.values.tolist(), file_breakdown=file_breakdown)
        stored_distmats = get_distmat_index()
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

        return stored_distmats, output_indicator, False, notification, False

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
