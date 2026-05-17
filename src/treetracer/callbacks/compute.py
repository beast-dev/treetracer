from dash import html, callback, Input, Output, State, no_update, ALL
import dash_mantine_components as dmc
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

from ..logger import add_log, notif_id
from ..db.tree_service import get_tree_service
from ..state import (load_distmat, get_distmat_index, next_distmat_name,
                      get_distmat_path, register_distmat, get_distmat_file_path,
                      get_distmat_groups_per_file,
                      store_mds_result, get_mds_results_index,
                      clear_all_mds_results)
from ._helpers import _save_file_dialog, extract_group


# RF compute happens in a SUBPROCESS, not a thread. See
# ``_spawn_subprocess_worker`` below for the rationale; tl;dr is that
# ``rapidtrees`` holds the GIL during its iterator-consumption phase
# for several seconds, which triggers macOS's main-thread watchdog
# (the spinning beach ball) when we share a process with the GUI.
#
# The ``ThreadPoolExecutor`` is still here, but it just owns a
# concurrent.futures.Future that's blocking on ``subprocess.Popen.wait``.
# That wait releases the GIL the whole time, so the parent stays
# perfectly responsive while the child does the work.
#
# MDS still runs in the same thread executor without a subprocess —
# scipy's LAPACK calls release the GIL natively, so there's no beach
# ball risk and the subprocess startup overhead isn't worth it.
_executor = None
_rf_future = None      # concurrent.futures.Future for RF job
_rf_meta = {}          # metadata needed by poll_completion to save RF result
_mds_future = None     # concurrent.futures.Future for between-run MDS job
_mds_meta = {}         # metadata needed by poll_completion to build MDS result


def _get_executor():
    global _executor
    if _executor is None:
        import atexit
        _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="treetracer-compute")
        atexit.register(_shutdown_executor)
    return _executor


def _rf_pipeline(selected_files, save_path, rf_name):
    """Parent-side preparation + persistent-worker dispatch for an RF compute.

    The parent does only what is cheap on the GIL-shared GUI process:
    a vectorised pandas slice to pull the offset metadata for the
    selected trees, plus a few small dict lookups. Then hands off to
    ``compute_rf_worker_entry`` via the persistent worker subprocess
    (see ``callbacks/persistent_worker.py``).

    Wall-clock cost in the parent: ~50 ms for 5000 trees, well below
    the macOS beach-ball threshold.
    """
    import time
    from . import persistent_worker

    t0 = time.time()
    tree_service = get_tree_service()
    db_manager = tree_service.db_manager

    add_log(f"[{rf_name}] Preparing tree descriptors for {len(selected_files)} file(s)...")
    db_manager.flush()
    df = db_manager._trees
    mask = df["file_source"].isin(selected_files)
    selected_df = df[mask]
    if len(selected_df) < 2:
        raise RuntimeError("Not enough trees retrieved for RF computation.")

    # Vectorised extraction — no Python-level row loop, no disk reads.
    tree_descriptors = selected_df[[
        "name", "newick_offset", "newick_length", "file_source", "group_name",
    ]].to_dict("records")
    # Pandas gives us int64/int32 numpy scalars; the worker side
    # expects plain Python ints (cleaner pickle, no numpy dependency
    # if we ever simplify the worker).
    for d in tree_descriptors:
        d["newick_offset"] = int(d["newick_offset"])
        d["newick_length"] = int(d["newick_length"])

    # File-path map for the worker's disk reads (paths registered by
    # ``TreeManagerPandas.register_source_file`` at upload time).
    source_file_paths = {
        fs: db_manager._source_files[fs]
        for fs in selected_files
        if fs in db_manager._source_files
    }

    # Translate maps keyed by file_source — keeps the worker's job
    # purely descriptor-driven, no DB query needed on the other side.
    translate_maps = {
        fs: db_manager.get_translate_map(fs) for fs in selected_files
    }

    add_log(
        f"[{rf_name}] Dispatching to persistent worker "
        f"({len(tree_descriptors)} trees, {len(source_file_paths)} source file(s))..."
    )

    result = persistent_worker.submit_job(
        "compute_rf",
        tree_descriptors=tree_descriptors,
        source_file_paths=source_file_paths,
        translate_maps=translate_maps,
        save_path=save_path,
        rf_name=rf_name,
    )
    # Add parent-side total wall-time (includes IPC round-trip).
    result["total_elapsed"] = time.time() - t0
    return result


def _shutdown_executor():
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None


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
            burnin = int(summary.get("burnin", 0))
            total_trees = int(summary.get("total_trees", 0))
            original_total = int(summary.get("original_total", total_trees + burnin))

            # Burn-in cell: "N (of M)" with "(of M)" rendered in light grey.
            burnin_cell = dmc.TableTd([
                str(burnin),
                " ",
                html.Span(
                    f"(of {original_total})",
                    style={
                        "color": "var(--mantine-color-gray-6)",
                        "fontSize": "0.85em",
                    },
                ),
            ])

            rows.append(
                dmc.TableTr([
                    dmc.TableTd(filename),
                    dmc.TableTd(str(summary.get("n_taxa", "—"))),
                    burnin_cell,
                    dmc.TableTd(str(total_trees)),
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
                        dmc.TableTh("Burn-in"),
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
                id=notif_id(),
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
                id=notif_id(),
            ), no_update, no_update, no_update

        # --- Taxa validation passed — submit the pipeline to a worker thread ---
        # The worker does the DB fetch + newick prep + RF compute. This
        # callback returns the indicator in ~10ms, matching the MDS path.
        n_taxa = unique_counts.pop()
        add_log(f"Taxa validation passed: all {len(selected_files)} files have {n_taxa} taxa")

        # Expected counts from in-memory summaries (no disk hit).
        expected_total = sum(
            stored_summaries[f].get("total_trees", 0) for f in selected_files
        )
        expected_breakdown_str = ", ".join(
            f"{f}: {stored_summaries[f].get('total_trees', 0)}"
            for f in selected_files
        )

        rf_name = next_distmat_name()
        save_path = get_distmat_path(rf_name)
        _rf_meta["name"] = rf_name
        _rf_meta["save_path"] = save_path

        global _rf_future
        _rf_future = _get_executor().submit(
            _rf_pipeline, selected_files, save_path, rf_name,
        )

        computing_indicator = dmc.Alert(
            title=f"Computing RF Distances ({rf_name})...",
            children=dmc.Text(
                f"Computing {expected_total}×{expected_total} RF distance matrix in background. "
                f"Files: {expected_breakdown_str}",
                size="sm",
            ),
            color="blue",
            variant="light",
        )
        return no_update, computing_indicator, False, True

    # ------ RF MATRIX LIST (right column) ------

    @callback(
        Output("rf-matrix-select", "data"),
        Output("rf-matrix-select", "value"),
        Output("rf-matrix-count", "children"),
        Output("rf-matrix-count", "color"),
        Output("export-rf-button", "disabled"),
        Input("distmat-store", "data"),
    )
    def update_rf_matrix_list(distmat_data):
        if not distmat_data:
            return [], None, "0", "gray", True
        options = [
            {"value": k, "label": f"{k} — {v['n_trees']} trees"}
            for k, v in distmat_data.items()
        ]
        last_key = list(distmat_data.keys())[-1]
        count = str(len(distmat_data))
        return options, last_key, count, "blue", False

    # Show file breakdown badges when an RF matrix is selected
    @callback(
        Output("rf-matrix-info", "children"),
        Input("rf-matrix-select", "value"),
        State("distmat-store", "data"),
        prevent_initial_call=True,
    )
    def show_rf_matrix_info(selected, distmat_data):
        if not selected or not distmat_data or selected not in distmat_data:
            return html.Div()
        breakdown = distmat_data[selected].get("file_breakdown", {})
        n = distmat_data[selected].get("n_trees", 0)
        badges = [
            dmc.Badge(f"{f}: {c}", variant="light", color="blue", size="sm")
            for f, c in breakdown.items()
        ]
        badges.append(dmc.Badge(f"Total: {n}", variant="light", color="grape", size="sm"))
        return dmc.Group(badges, gap=4)

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
            {"value": k, "label": f"{k} — {v['n_trees']} trees"}
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
        n = distmat_data[selected].get("n_trees", 0)
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
            tree_names, _ = load_distmat(selected_distmat)
        except KeyError:
            msg = "Selected distance matrix not available. Please recompute RF distances."
            add_log(msg, "ERROR")
            return dmc.Text(msg, c="red"), no_update, no_update

        n = len(tree_names)
        n_components = min(6, n - 1)
        add_log(f"Computing MDS from {selected_distmat} ({n}x{n}) in background process...")

        _mds_meta["tree_names"] = tree_names
        _mds_meta["selected_distmat"] = selected_distmat
        _mds_meta["n_components"] = n_components

        # Route through the persistent worker — same pattern as RF/MCC/
        # Pseudo-ESS. Per-compute IPC overhead is ~100ms, dwarfed by the
        # ARPACK eigsh on a 5k×5k matrix; the win is a single unified
        # background-compute pattern and clean process isolation.
        from . import persistent_worker
        matrix_path = get_distmat_file_path(selected_distmat)
        global _mds_future
        _mds_future = _get_executor().submit(
            persistent_worker.submit_job,
            "compute_mds",
            matrix_path=str(matrix_path),
            n_components=n_components,
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

    # ------ POLL + PROCESS: checks futures, processes results in one round trip ------
    # Processing is fast (<20ms) since workers save to disk — no large pickle transfer.

    @callback(
        # RF outputs (5)
        Output("compute-rf-output", "children", allow_duplicate=True),
        Output("distmat-store", "data", allow_duplicate=True),
        Output("export-rf-button", "disabled", allow_duplicate=True),
        Output("compute-rf-trace-button", "disabled", allow_duplicate=True),
        Output("compute-rf-button", "disabled", allow_duplicate=True),
        # Between-run MDS outputs (5)
        Output("mds-result-store", "data"),
        Output("compute-mds-output", "children", allow_duplicate=True),
        Output("export-mds-button", "disabled"),
        Output("plot-config-store", "data", allow_duplicate=True),
        Output("compute-mds-button", "disabled", allow_duplicate=True),
        # Shared outputs (2)
        Output("notifications-container", "children", allow_duplicate=True),
        Output("compute-poll-interval", "disabled", allow_duplicate=True),
        # Auto-collapse sidebar on RF success (3 outputs mirror the
        # shell sidebar-toggle callback's outputs).
        Output("navbar", "style", allow_duplicate=True),
        Output("sidebar-visible", "data", allow_duplicate=True),
        Output("appshell", "navbar", allow_duplicate=True),
        Input("compute-poll-interval", "n_intervals"),
        State("sidebar-visible", "data"),
        prevent_initial_call=True,
    )
    def poll_completion(n_intervals, sidebar_visible):
        global _rf_future, _mds_future

        rf_done = _rf_future is not None and _rf_future.done()
        mds_done = _mds_future is not None and _mds_future.done()

        if not rf_done and not mds_done:
            return (no_update,) * 15

        rf_out = [no_update] * 5
        mds_out = [no_update] * 5
        notif = no_update
        # Sidebar outputs: (navbar.style, sidebar-visible, appshell.navbar).
        # Only flipped on RF success when the sidebar is currently open;
        # everything else stays no_update so we don't fight the user's
        # last toggle.
        sidebar_out = [no_update, no_update, no_update]

        # --- Process RF ---
        if rf_done:
            try:
                pipeline = _rf_future.result()
            except Exception as e:
                msg = f"RF computation failed: {e}"
                add_log(msg, "ERROR")
                _rf_future = None
                rf_out = [dmc.Text(msg, c="red"), no_update, no_update, no_update, False]
                notif = dmc.Notification(title="RF Computation Error", message=msg,
                                         color="red", action="show", autoClose=6000,
                                         id=notif_id())
            else:
                _rf_future = None
                rf_name = _rf_meta.get("name", "RF")
                result_names = pipeline["result_names"]
                file_breakdown = pipeline["file_breakdown"]
                groups_per_file = pipeline["groups_per_file"]
                elapsed = pipeline["total_elapsed"]
                compute_elapsed = pipeline["compute_elapsed"]
                # Matrix already saved to disk by the worker — just register it
                register_distmat(rf_name, result_names, _rf_meta["save_path"],
                                 file_breakdown=file_breakdown,
                                 groups_per_file=groups_per_file)
                add_log(f"Stored RF distance matrix as '{rf_name}' ({len(result_names)}x{len(result_names)})")
                add_log(f"RF pipeline took {elapsed:.2f}s (rapidtrees compute {compute_elapsed:.2f}s)")
                rf_out = [
                    dmc.Alert(title=f"RF Distance Matrix ({rf_name})",
                              children=dmc.Text(f"{len(result_names)} x {len(result_names)} trees", size="sm"),
                              color="green", variant="light"),
                    get_distmat_index(),
                    False, False, False,
                ]
                notif = dmc.Notification(
                    title=f"RF Distances Computed ({rf_name})",
                    message=f"Computed {len(result_names)}x{len(result_names)} RF distance matrix in {elapsed:.2f}s.",
                    color="green", action="show", autoClose=3000, id=notif_id())
                # Auto-collapse the sidebar to give the results area room.
                if sidebar_visible:
                    sidebar_out = [
                        {"display": "none"},
                        False,
                        {"width": 0, "breakpoint": "sm",
                         "collapsed": {"mobile": True}},
                    ]

        # --- Process MDS ---
        if mds_done:
            try:
                embedding_list, elapsed = _mds_future.result()
            except Exception as e:
                msg = f"MDS computation failed: {e}"
                add_log(msg, "ERROR")
                _mds_future = None
                mds_out = [no_update, dmc.Text(msg, c="red"), no_update, no_update, False]
                if notif is no_update:
                    notif = dmc.Notification(title="MDS Error", message=msg, color="red",
                                             action="show", autoClose=6000, id=notif_id())
            else:
                _mds_future = None
                tree_names = [str(n) for n in _mds_meta["tree_names"]]
                selected_distmat = _mds_meta["selected_distmat"]
                n_components = _mds_meta["n_components"]

                mdscols = [f"MDS{i+1}" for i in range(n_components)]
                mds_df = pd.DataFrame(embedding_list, columns=mdscols)
                mds_df["tree"] = tree_names
                mds_df["group"] = mds_df["tree"].apply(extract_group)
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
                mds_entry = {"metadata": metadata, "data": mds_df.to_dict("records")}

                # Store full result server-side, send only metadata through dcc.Store
                store_mds_result(mds_filename, mds_entry)

                n_groups = len(mds_df["group"].unique())
                add_log(f"PCoA completed in {elapsed:.2f}s: {len(mds_df)} points, {n_components}D, {n_groups} groups")

                mds_out = [
                    get_mds_results_index(),  # lightweight metadata only
                    dmc.Alert(title="MDS Embedding Complete",
                              children=dmc.Text(f"{mds_filename}: {len(mds_df)} points, {n_components}D, {n_groups} groups", size="sm"),
                              color="green", variant="light"),
                    False, {}, False,
                ]
                if notif is no_update:
                    notif = dmc.Notification(
                        title="MDS Computed",
                        message=f"PCoA: {len(mds_df)} points, {n_components}D in {elapsed:.2f}s.",
                        color="green", action="show", autoClose=3000, id=notif_id())

        # Re-enable interval if any jobs are still running
        any_running = ((_rf_future is not None and not _rf_future.done()) or
                       (_mds_future is not None and not _mds_future.done()))
        poll_disabled = not any_running

        return (*rf_out, *mds_out, notif, poll_disabled, *sidebar_out)

    # ------ EXPORT CALLBACKS ------

    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("export-rf-button", "n_clicks"),
        State("rf-matrix-select", "value"),
        prevent_initial_call=True,
    )
    def export_rf_matrix(n_clicks, selected_matrix):
        if not n_clicks or not selected_matrix:
            return no_update
        path = _save_file_dialog(default_filename=f"{selected_matrix}.tsv")
        if not path:
            return no_update
        try:
            names, arr = load_distmat(selected_matrix)
        except KeyError:
            add_log(f"Matrix '{selected_matrix}' not available for export.", "ERROR")
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
            id=notif_id(),
        )

    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("export-mds-button", "n_clicks"),
        State("mds-result-store", "data"),
        State("mds-result-select", "value"),
        prevent_initial_call=True,
    )
    def export_mds(n_clicks, mds_index, selected_mds):
        if not n_clicks or not mds_index or not selected_mds:
            return no_update
        from ..state import get_mds_result
        entry = get_mds_result(selected_mds)
        if not entry or not entry.get("data"):
            return no_update
        metadata = entry["metadata"]
        path = _save_file_dialog(default_filename=metadata["filename"])
        if not path:
            return no_update
        mds_df = pd.DataFrame(entry["data"])
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
            id=notif_id(),
        )

    # ------ BETWEEN-RUN MDS RESULT LIST (right column) ------

    @callback(
        Output("mds-result-select", "data"),
        Output("mds-result-select", "value"),
        Output("mds-result-count", "children"),
        Output("mds-result-count", "color"),
        Output("export-mds-button", "disabled", allow_duplicate=True),
        Input("mds-result-store", "data"),
        prevent_initial_call=True,
    )
    def update_mds_result_list(mds_index):
        if not mds_index:
            return [], None, "0", "gray", True
        options = []
        for key, meta in mds_index.items():
            n_groups = len(meta.get("groups", []))
            label = f"{key} — {meta.get('rows', '?')} trees, {n_groups} groups"
            options.append({"value": key, "label": label})
        last_key = list(mds_index.keys())[-1]
        count = str(len(mds_index))
        return options, last_key, count, "blue", False

    @callback(
        Output("mds-result-info", "children"),
        Input("mds-result-select", "value"),
        State("mds-result-store", "data"),
        prevent_initial_call=True,
    )
    def show_mds_result_info(selected, mds_index):
        if not selected or not mds_index or selected not in mds_index:
            return html.Div()
        meta = mds_index[selected]
        badges = [
            dmc.Badge(f"{g}", variant="light", color="blue", size="sm")
            for g in meta.get("groups", [])
        ]
        badges.append(dmc.Badge(f"{meta.get('rows', '?')} trees", variant="light", color="grape", size="sm"))
        return dmc.Group(badges, gap=4)

