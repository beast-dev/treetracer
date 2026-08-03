from dash import html, callback, Input, Output, State, no_update, ALL, ctx
import dash_mantine_components as dmc
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from ..logger import add_log, notif_id
from ..background_jobs import JobBusyError, JobRef, JobState, job_manager
from ..db.tree_service import get_tree_service
from ..state import (load_distmat, get_distmat_index, next_distmat_name,
                      get_distmat_path, register_distmat, get_distmat_file_path,
                      get_distmat_names, store_mds_result,
                      get_mds_results_index)
from ..ui.widgets import computing_banner
from ._helpers import _save_file_dialog, extract_group
from .._worker_log import log as _wlog
from . import persistent_worker


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
# The executor threads block on persistent-worker IPC. The heavy RF/MDS
# work runs in the worker subprocess, keeping the pywebview/Dash process
# responsive while compute jobs are in flight.
_executor = None


def _mds_export_filename(source_distmat):
    stem = source_distmat[:-4] if source_distmat.endswith(".tsv") else source_distmat
    return f"{stem}_MDS.tsv"


def _job_ref_from_store(data, *, expected_kind=None):
    """Parse a browser job reference defensively.

    Browser stores can be empty, stale, or manually modified.  Invalid data is
    treated as absent; ``JobManager`` performs the authoritative generation
    check on every operation.
    """
    if not isinstance(data, dict):
        return None
    try:
        ref = JobRef.from_dict(data)
    except (KeyError, TypeError, ValueError):
        return None
    if expected_kind is not None and ref.kind != expected_kind:
        return None
    return ref


def _latest_rf_mds_ref(rf_job_data, mds_job_data):
    refs = [
        ref
        for ref in (
            _job_ref_from_store(rf_job_data, expected_kind="rf"),
            _job_ref_from_store(mds_job_data, expected_kind="mds"),
        )
        if ref is not None
    ]
    return max(refs, key=lambda ref: ref.generation, default=None)


def _ack_applied_job(applied_data):
    """Acknowledge a terminal event known to have reached browser state."""
    ref = _job_ref_from_store(applied_data)
    if ref is None:
        return False
    try:
        revision = int(applied_data["terminal_revision"])
    except (KeyError, TypeError, ValueError):
        return False
    return job_manager.acknowledge(ref, revision)


def _terminal_progress_from_store(job_data, expected_kind):
    ref = _job_ref_from_store(job_data, expected_kind=expected_kind)
    if ref is None:
        return None
    snapshot = job_manager.snapshot(ref)
    if snapshot is None or snapshot.terminal is None:
        return None
    progress = snapshot.progress
    if progress is None:
        return None
    label = {
        JobState.SUCCEEDED: "complete",
        JobState.FAILED: "failed",
        JobState.CANCELLED: "cancelled",
    }[snapshot.state]
    return progress.fraction * 100.0, label


def _get_executor():
    global _executor
    if _executor is None:
        import atexit
        _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="treetracer-compute")
        atexit.register(_shutdown_executor)
    return _executor


def _rf_pipeline(selected_files, save_path, rf_name, is_rooted):
    """Parent-side preparation + persistent-worker dispatch for an RF compute.

    The parent does only what is cheap on the GIL-shared GUI process:
    a vectorised pandas slice to pull the offset metadata for the
    selected trees, plus a few small dict lookups. Then hands off to
    ``compute_rf_worker_entry`` via the persistent worker subprocess
    (see ``callbacks/persistent_worker.py``).

    Wall-clock cost in the parent: ~50 ms for 5000 trees, well below
    the macOS beach-ball threshold.

    ``is_rooted`` is the consensus tree rooting convention of the selected
    files (caller validates that they all agree). The worker forwards
    this to ``rapidtrees.pairwise_rf_with_snapshots_from_newick_iter``
    so the presence matrix's columns are rooted clades (True) or
    bipartitions (False).
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
        f"({len(tree_descriptors)} trees, {len(source_file_paths)} source file(s), "
        f"{'rooted' if is_rooted else 'unrooted'} mode)..."
    )

    # Sidecar file the worker will keep up-to-date with the rapidtrees
    # ProgressCounter state every ~100ms. The Dash poll callback
    # (``update_rf_progress``) reads this file to drive the progress bar
    # in the computing banner.
    progress_path = save_path + ".progress"

    _wlog(f"[parent] _rf_pipeline: about to call persistent_worker.submit_job for {rf_name!r}")
    result = persistent_worker.submit_job(
        "compute_rf",
        tree_descriptors=tree_descriptors,
        source_file_paths=source_file_paths,
        translate_maps=translate_maps,
        save_path=save_path,
        progress_path=progress_path,
        rf_name=rf_name,
        is_rooted=is_rooted,
    )
    _wlog(f"[parent] _rf_pipeline: submit_job returned for {rf_name!r}; result keys={sorted(result.keys())}")
    # Add parent-side total wall-time (includes IPC round-trip).
    result["total_elapsed"] = time.time() - t0
    result["is_rooted"] = is_rooted
    result["progress_path"] = progress_path
    result["save_path"] = save_path
    _wlog(f"[parent] _rf_pipeline: returning result; total_elapsed={result['total_elapsed']:.3f}s")
    return result


def _finalize_rf_job(_ref, pipeline):
    """Persist one RF result and return its small terminal UI payload."""
    return _publish_rf_result(pipeline)


def _publish_rf_result(pipeline):
    rf_name = pipeline["rf_name"]
    result_names = pipeline["result_names"]
    file_breakdown = pipeline["file_breakdown"]
    groups_per_file = pipeline["groups_per_file"]
    elapsed = pipeline["total_elapsed"]
    compute_elapsed = pipeline["compute_elapsed"]
    is_rooted = pipeline.get("is_rooted", True)

    # The matrix is already on disk; this is the exactly-once publication step.
    register_distmat(
        rf_name,
        result_names,
        pipeline["save_path"],
        file_breakdown=file_breakdown,
        groups_per_file=groups_per_file,
        is_rooted=is_rooted,
    )
    add_log(
        f"Stored RF distance matrix as '{rf_name}' "
        f"({len(result_names)}x{len(result_names)})"
    )
    add_log(
        f"RF pipeline took {elapsed:.2f}s "
        f"(rapidtrees compute {compute_elapsed:.2f}s)"
    )
    return {
        "rf_name": rf_name,
        "n_trees": len(result_names),
        "elapsed": elapsed,
        "compute_elapsed": compute_elapsed,
        "distmat_index": get_distmat_index(),
    }


def _mds_pipeline(
    matrix_path,
    progress_path,
    tree_names,
    selected_distmat,
    n_components,
):
    embedding_list, elapsed = persistent_worker.submit_job(
        "compute_mds",
        matrix_path=str(matrix_path),
        n_components=n_components,
        progress_path=progress_path,
    )
    return {
        "embedding": embedding_list,
        "elapsed": elapsed,
        "tree_names": [str(name) for name in tree_names],
        "selected_distmat": selected_distmat,
        "n_components": n_components,
    }


def _finalize_mds_job(_ref, result):
    """Persist one MDS result and return its small terminal UI payload."""
    return _publish_mds_result(result)


def _publish_mds_result(result):
    embedding_list = result["embedding"]
    elapsed = result["elapsed"]
    tree_names = result["tree_names"]
    selected_distmat = result["selected_distmat"]
    n_components = result["n_components"]

    mdscols = [f"MDS{i + 1}" for i in range(n_components)]
    mds_df = pd.DataFrame(embedding_list, columns=mdscols)
    mds_df["tree"] = tree_names
    mds_df["group"] = mds_df["tree"].apply(extract_group).astype(str)
    group_mapping = {
        value: index
        for index, value in enumerate(sorted(mds_df["group"].unique()))
    }
    mds_df["group_col"] = mds_df["group"].map(group_mapping)
    mds_df["treenum"] = mds_df.groupby("group").cumcount() + 1
    mds_df["size"] = 6
    mds_filename = _mds_export_filename(selected_distmat)
    mds_df["file"] = mds_filename

    metadata = {
        "filename": mds_filename,
        "source_distmat": selected_distmat,
        "rows": len(mds_df),
        "dimensions": mdscols,
        "groups": mds_df["group"].unique().tolist(),
        "MIN_TREENUM": int(mds_df["treenum"].min()),
        "MAX_TREENUM": int(mds_df["treenum"].max()),
    }
    store_mds_result(
        mds_filename,
        {"metadata": metadata, "data": mds_df.to_dict("records")},
    )

    n_groups = len(metadata["groups"])
    add_log(
        f"PCoA completed in {elapsed:.2f}s: {len(mds_df)} points, "
        f"{n_components}D, {n_groups} groups"
    )
    return {
        "mds_results_index": get_mds_results_index(),
        "mds_filename": mds_filename,
        "n_points": len(mds_df),
        "n_components": n_components,
        "n_groups": n_groups,
        "elapsed": elapsed,
    }


def _shutdown_executor():
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None


def reset():
    """Invalidate any managed job before Clear Data wipes its inputs.

    Removing the manager record prevents a late worker result from
    publishing into freshly cleared application state. The persistent-worker
    kill interrupts native work when possible.
    """
    active = job_manager.active_ref()
    if active is None:
        return
    persistent_worker.cancel_current_job()
    job_manager.invalidate(active)


def register_compute_callbacks():
    # Stop button — interrupt whatever compute is currently running by
    # killing the persistent worker (see
    # ``persistent_worker.cancel_current_job``). One callback serves
    # every banner's Stop button via the {"type": "compute-stop", ...}
    # pattern-matching id. The kill makes the in-flight job's future
    # raise JobCancelled; the per-compute poll callbacks below catch it
    # and clear the banner ~one 100ms tick later.
    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input({"type": "compute-stop", "which": ALL}, "n_clicks"),
        State("rf-progress-path", "data"),
        State("mds-progress-path", "data"),
        prevent_initial_call=True,
    )
    def handle_compute_stop(_stop_clicks, rf_progress_path, mds_progress_path):
        # Pattern-matching inputs fire both on real clicks and whenever
        # a Stop button is added to / removed from the layout (the
        # banners are dynamic). Only a real click carries a truthy
        # n_clicks in the trigger; layout-change fires carry None.
        if not any(t.get("value") for t in ctx.triggered):
            return no_update
        active = job_manager.active_ref()
        managed_cancelled = (
            active is not None
            and job_manager.mark_cancelled(
                active,
                message="Computation cancelled by user",
            )
        )
        worker_cancelled = persistent_worker.cancel_current_job()
        if managed_cancelled or worker_cancelled:
            # Cancel hard-kills the worker subprocess, so the progress
            # writer thread's ``finally`` block (which normally unlinks
            # the sidecar file) never runs. Tidy up here. The file is
            # ~80B so a leak is cosmetic, but next compute reuses the
            # same naming pattern — better to start clean.
            import contextlib
            from pathlib import Path
            for progress_path in (rf_progress_path, mds_progress_path):
                if progress_path:
                    with contextlib.suppress(OSError):
                        Path(progress_path).unlink()
            add_log("Compute cancelled by user — worker subprocess killed.",
                    "WARNING")
            return dmc.Notification(
                title="Stopping computation",
                message="The running computation is being cancelled.",
                color="yellow", action="show", autoClose=3000, id=notif_id(),
            )
        add_log("Stop clicked but no computation was running.", "WARNING")
        return no_update

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
                    dmc.TableTd(
                        html.Div(
                            filename,
                            className="tt-compute-filename",
                            title=filename,
                        ),
                        className="tt-compute-file-cell",
                    ),
                    dmc.TableTd(str(summary.get("n_taxa", "—")), className="tt-compute-number-cell"),
                    burnin_cell,
                    dmc.TableTd(str(total_trees), className="tt-compute-number-cell"),
                    dmc.TableTd(
                        dmc.Checkbox(
                            id={"type": "compute-tree-checkbox", "index": filename},
                            checked=True,
                        ),
                        className="tt-compute-select-cell",
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
            layout="fixed",
            className="tt-compute-table",
        )

        return table, False

    @callback(
        Output("compute-export-drawer", "opened"),
        Input("open-rf-export-drawer", "n_clicks"),
        Input("open-mds-export-drawer", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_export_drawer(_rf_clicks, _mds_clicks):
        if not any(t.get("value") for t in ctx.triggered):
            return no_update
        return True

    # Callback to validate taxa, extract data, and start RF computation in background
    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("compute-rf-output", "children"),
        Output("compute-poll-interval", "disabled"),
        Output("compute-rf-button", "disabled", allow_duplicate=True),
        # Progress-path Store — populated when an RF compute actually
        # starts so ``update_rf_progress`` knows which sidecar file to
        # tail. Early returns leave it at no_update.
        Output("rf-progress-path", "data", allow_duplicate=True),
        Output("rf-job-store", "data"),
        Input("compute-rf-button", "n_clicks"),
        State({"type": "compute-tree-checkbox", "index": ALL}, "checked"),
        State({"type": "compute-tree-checkbox", "index": ALL}, "id"),
        State("tree-offset-store", "data"),
        State("compute-applied-job-store", "data"),
        prevent_initial_call=True,
    )
    def handle_compute_rf(
        n_clicks,
        checked_list,
        id_list,
        stored_summaries,
        applied_job,
    ):
        if not n_clicks or not stored_summaries:
            return (no_update,) * 6

        # The marker is written in the same browser response that rendered the
        # prior terminal UI. A fast next click may beat the acknowledgement
        # callback, so acknowledge idempotently here before submitting too.
        _ack_applied_job(applied_job)

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
            ), no_update, no_update, no_update, no_update, no_update

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
            ), no_update, no_update, no_update, no_update, no_update

        # --- Rooting consistency check ---
        # RF over rooted clades and RF over bipartitions are different
        # quantities; comparing them across a mixed selection is
        # mathematically meaningless. Reject before submission.
        rooting_per_file = {
            fname: bool(stored_summaries.get(fname, {}).get("is_rooted", True))
            for fname in selected_files
        }
        unique_rootings = set(rooting_per_file.values())
        if len(unique_rootings) > 1:
            rooted_files = [f for f, r in rooting_per_file.items() if r]
            unrooted_files = [f for f, r in rooting_per_file.items() if not r]
            msg = (
                "Selected files mix rooted and unrooted trees. RF distances "
                "across rooting conventions are not comparable. "
                f"Rooted: {', '.join(rooted_files)}. "
                f"Unrooted: {', '.join(unrooted_files)}."
            )
            add_log(f"Rooting mismatch — aborting RF computation: {msg}", "ERROR")
            return dmc.Notification(
                title="Rooting Mismatch",
                message=msg,
                color="red", action="show", autoClose=8000, id=notif_id(),
            ), no_update, no_update, no_update, no_update, no_update
        selected_is_rooted = unique_rootings.pop()

        # --- Taxa validation passed — submit the pipeline to a worker thread ---
        # The worker does the DB fetch + newick prep + RF compute. This
        # callback returns the indicator in ~10ms, matching the MDS path.
        n_taxa = unique_counts.pop()
        add_log(
            f"Validation passed: all {len(selected_files)} files have {n_taxa} taxa, "
            f"{'rooted' if selected_is_rooted else 'unrooted'} mode."
        )

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
        progress_path = save_path + ".progress"
        try:
            job_ref = job_manager.submit(
                _get_executor(),
                "rf",
                _rf_pipeline,
                selected_files,
                save_path,
                rf_name,
                selected_is_rooted,
                metadata={
                    "display_name": rf_name,
                    "progress_path": progress_path,
                },
                finalizer=_finalize_rf_job,
                cancel_exceptions=(persistent_worker.JobCancelled,),
            )
        except JobBusyError as exc:
            msg = (
                f"Another computation ({exc.active.kind.upper()}) is still "
                "finishing. Please wait for it to complete."
            )
            add_log(msg, "WARNING")
            return (
                dmc.Notification(
                    title="Computation already running",
                    message=msg,
                    color="yellow",
                    action="show",
                    autoClose=4000,
                    id=notif_id(),
                ),
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
            )

        computing_indicator = computing_banner(
            title=f"Computing RF Distances ({rf_name})...",
            message=(
                f"Computing {expected_total}×{expected_total} RF distance "
                f"matrix in background. Files: {expected_breakdown_str}"
            ),
            which="rf",
            show_progress=True,
        )
        # Hand the same path ``_rf_pipeline`` derives down to the
        # Store so ``update_rf_progress`` reads from the right file.
        return (
            no_update,
            computing_indicator,
            False,
            True,
            progress_path,
            job_ref.as_dict(),
        )

    # Per-tick reader for the RF progress sidecar file. Runs off the
    # same ``compute-poll-interval`` as ``poll_completion`` but writes
    # to disjoint Outputs (the progress-bar value + label), so it
    # doesn't trip Dash 4.x's same-input/same-output duplicate check.
    @callback(
        Output("rf-progress-bar", "value"),
        Output("rf-progress-label", "children"),
        Input("compute-poll-interval", "n_intervals"),
        State("rf-progress-path", "data"),
        State("rf-job-store", "data"),
        prevent_initial_call=True,
    )
    def update_rf_progress(_n, progress_path, job_data):
        terminal_progress = _terminal_progress_from_store(job_data, "rf")
        if terminal_progress is not None:
            return terminal_progress
        if not progress_path:
            return no_update, no_update
        import json
        from pathlib import Path
        try:
            data = json.loads(Path(progress_path).read_text())
        except (OSError, ValueError):
            # File doesn't exist yet, was just deleted, or caught
            # mid-write — try again next tick. ValueError covers
            # JSONDecodeError (subclass) too.
            terminal_progress = _terminal_progress_from_store(job_data, "rf")
            return terminal_progress or (no_update, no_update)
        val = int(data.get("value", 0))
        tot = int(data.get("total", 0))
        frac = max(0.0, min(float(data.get("fraction", 0.0)), 1.0))
        phase = data.get("phase", "computing")
        ref = _job_ref_from_store(job_data, expected_kind="rf")
        if ref is not None:
            job_manager.update_progress(ref, frac, phase)
        if tot == 0:
            return 0, "starting…"
        pct = frac * 100.0
        if phase == "finalizing":
            label = f"{val:,} / {tot:,} pairs — finalizing…"
        else:
            label = f"{val:,} / {tot:,} pairs ({pct:.1f}%)"
        return pct, label

    @callback(
        Output("mds-progress-bar", "value"),
        Output("mds-progress-label", "children"),
        Input("compute-poll-interval", "n_intervals"),
        State("mds-progress-path", "data"),
        State("mds-job-store", "data"),
        prevent_initial_call=True,
    )
    def update_mds_progress(_n, progress_path, job_data):
        terminal_progress = _terminal_progress_from_store(job_data, "mds")
        if terminal_progress is not None:
            return terminal_progress
        if not progress_path:
            return no_update, no_update
        import json
        from pathlib import Path
        try:
            data = json.loads(Path(progress_path).read_text())
        except (OSError, ValueError):
            terminal_progress = _terminal_progress_from_store(job_data, "mds")
            return terminal_progress or (no_update, no_update)

        frac = max(0.0, min(float(data.get("fraction", 0.0)), 1.0))
        pct = frac * 100.0
        phase = data.get("phase", "computing")
        label = data.get("label") or phase.replace("_", " ")
        ref = _job_ref_from_store(job_data, expected_kind="mds")
        if ref is not None:
            job_manager.update_progress(ref, frac, phase, label)
        return pct, f"{label} ({pct:.0f}%)"

    # ------ RF MATRIX LIST (right column) ------

    @callback(
        Output("rf-matrix-select", "data"),
        Output("rf-matrix-select", "value"),
        Output("rf-matrix-count", "children"),
        Output("rf-matrix-count", "color"),
        Output("rf-matrix-count-main", "children"),
        Output("rf-matrix-count-main", "color"),
        Output("export-rf-button", "disabled"),
        Input("distmat-store", "data"),
    )
    def update_rf_matrix_list(distmat_data):
        if not distmat_data:
            return [], None, "0", "gray", "0", "gray", True
        options = [
            {"value": k, "label":
                f"{k} — {v['n_trees']} trees "
                f"({'rooted' if v.get('is_rooted', True) else 'unrooted'})"}
            for k, v in distmat_data.items()
        ]
        last_key = list(distmat_data.keys())[-1]
        count = str(len(distmat_data))
        return options, last_key, count, "blue", count, "blue", False

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
            {"value": k, "label":
                f"{k} — {v['n_trees']} trees "
                f"({'rooted' if v.get('is_rooted', True) else 'unrooted'})"}
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
        Output("mds-progress-path", "data", allow_duplicate=True),
        Output("mds-job-store", "data"),
        Input("compute-mds-button", "n_clicks"),
        State("mds-distmat-select", "value"),
        State("compute-applied-job-store", "data"),
        prevent_initial_call=True,
    )
    def handle_compute_mds(n_clicks, selected_distmat, applied_job):
        if not n_clicks or not selected_distmat:
            return (no_update,) * 5

        _ack_applied_job(applied_job)

        try:
            # The worker loads the matrix from its registered path. Pull only
            # the small name vector here instead of reading the full n×n array
            # into the UI process just to determine dimensions and labels.
            tree_names = list(get_distmat_names(selected_distmat))
        except KeyError:
            msg = "Selected distance matrix not available. Please recompute RF distances."
            add_log(msg, "ERROR")
            return dmc.Text(msg, c="red"), no_update, no_update, no_update, no_update

        n = len(tree_names)
        n_components = min(6, n - 1)
        add_log(f"Computing MDS from {selected_distmat} ({n}x{n}) in background process...")

        # Route through the persistent worker — same pattern as RF/consensus tree/
        # Pseudo-ESS. Per-compute IPC overhead is ~100ms, dwarfed by the
        # ARPACK eigsh on a 5k×5k matrix; the win is a single unified
        # background-compute pattern and clean process isolation.
        matrix_path = get_distmat_file_path(selected_distmat)
        progress_path = str(matrix_path) + ".mds.progress"
        try:
            job_ref = job_manager.submit(
                _get_executor(),
                "mds",
                _mds_pipeline,
                matrix_path,
                progress_path,
                tree_names,
                selected_distmat,
                n_components,
                metadata={
                    "display_name": _mds_export_filename(selected_distmat),
                    "progress_path": progress_path,
                },
                finalizer=_finalize_mds_job,
                cancel_exceptions=(persistent_worker.JobCancelled,),
            )
        except JobBusyError as exc:
            msg = (
                f"Another computation ({exc.active.kind.upper()}) is still "
                "finishing. Please wait for it to complete."
            )
            add_log(msg, "WARNING")
            return (
                dmc.Alert(
                    title="Computation already running",
                    children=dmc.Text(msg, size="sm"),
                    color="yellow",
                    variant="light",
                ),
                no_update,
                False,
                no_update,
                no_update,
            )

        computing_indicator = computing_banner(
            title="Computing MDS Embedding...",
            message=(
                f"Computing PCoA with {n_components} components for "
                f"{n} trees in background."
            ),
            which="mds",
            show_progress=True,
        )
        return computing_indicator, False, True, progress_path, job_ref.as_dict()

    # ------ POLL + RENDER: replay sticky terminal state until browser ack ------

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
        # Shared outputs (3). The applied marker lands in the same browser
        # response as the result UI; only that marker authorizes the separate
        # acknowledgement callback to consume the retained terminal event.
        Output("notifications-container", "children", allow_duplicate=True),
        Output("compute-poll-interval", "disabled", allow_duplicate=True),
        Output("compute-applied-job-store", "data", allow_duplicate=True),
        # Auto-collapse sidebar on RF success (3 outputs mirror the
        # shell sidebar-toggle callback's outputs).
        Output("navbar", "style", allow_duplicate=True),
        Output("sidebar-visible", "data", allow_duplicate=True),
        Output("appshell", "navbar", allow_duplicate=True),
        Input("compute-poll-interval", "n_intervals"),
        Input("rf-job-store", "data"),
        Input("mds-job-store", "data"),
        State("sidebar-visible", "data"),
        prevent_initial_call=True,
    )
    def poll_completion(
        n_intervals,
        rf_job_data,
        mds_job_data,
        sidebar_visible,
    ):
        ref = _latest_rf_mds_ref(rf_job_data, mds_job_data)
        if ref is None:
            return (no_update,) * 16

        snapshot = job_manager.snapshot_for_delivery(ref)
        if snapshot is None or snapshot.acknowledged:
            return (no_update,) * 16

        if snapshot.terminal is None:
            if (n_intervals or 0) % 10 == 0:
                _wlog(
                    f"[parent] poll_completion tick={n_intervals}: "
                    f"job={ref.job_id}/{ref.kind}/generation-{ref.generation}, "
                    f"state={snapshot.state.value}"
                )
            return (no_update,) * 16

        terminal = snapshot.terminal
        payload = terminal.payload
        _wlog(
            f"[parent] poll_completion tick={n_intervals}: "
            f"delivering job={ref.job_id}/{ref.kind}/generation-{ref.generation}, "
            f"state={terminal.state.value}, revision={terminal.revision}, "
            f"attempt={snapshot.delivery_attempt}"
        )

        rf_out = [no_update] * 5
        mds_out = [no_update] * 5
        notif = no_update
        # Sidebar outputs: (navbar.style, sidebar-visible, appshell.navbar).
        # Only flipped on RF success when the sidebar is currently open;
        # everything else stays no_update so we don't fight the user's
        # last toggle.
        sidebar_out = [no_update, no_update, no_update]

        if ref.kind == "rf":
            if terminal.state is JobState.CANCELLED:
                add_log("RF computation cancelled by user.", "WARNING")
                rf_out = [
                    dmc.Alert(
                        title="RF computation cancelled",
                        children=dmc.Text("Stopped before completion.", size="sm"),
                        color="gray", variant="light",
                    ),
                    no_update, no_update, no_update, False,
                ]
            elif terminal.state is JobState.FAILED:
                msg = f"RF computation failed: {payload.get('message', 'Unknown error')}"
                add_log(msg, "ERROR")
                rf_out = [dmc.Text(msg, c="red"), no_update, no_update, no_update, False]
                notif = dmc.Notification(
                    title="RF Computation Error",
                    message=msg,
                    color="red",
                    action="show",
                    autoClose=6000,
                    id=f"rf-terminal-{ref.job_id}",
                )
            else:
                rf_name = payload["rf_name"]
                n_trees = payload["n_trees"]
                elapsed = payload["elapsed"]
                rf_out = [
                    dmc.Alert(title=f"RF Distance Matrix ({rf_name})",
                              children=dmc.Text(f"{n_trees} x {n_trees} trees", size="sm"),
                              color="green", variant="light"),
                    payload["distmat_index"],
                    False, False, False,
                ]
                notif = dmc.Notification(
                    title=f"RF Distances Computed ({rf_name})",
                    message=(
                        f"Computed {n_trees}x{n_trees} RF distance matrix "
                        f"in {elapsed:.2f}s."
                    ),
                    color="green",
                    action="show",
                    autoClose=3000,
                    id=f"rf-terminal-{ref.job_id}",
                )
                # Auto-collapse the sidebar to give the results area room.
                if sidebar_visible:
                    sidebar_out = [
                        {"display": "none"},
                        False,
                        {"width": 0, "breakpoint": "sm",
                         "collapsed": {"mobile": True}},
                    ]

        else:
            if terminal.state is JobState.CANCELLED:
                add_log("MDS computation cancelled by user.", "WARNING")
                mds_out = [
                    no_update,
                    dmc.Alert(
                        title="MDS computation cancelled",
                        children=dmc.Text("Stopped before completion.", size="sm"),
                        color="gray", variant="light",
                    ),
                    no_update, no_update, False,
                ]
            elif terminal.state is JobState.FAILED:
                msg = f"MDS computation failed: {payload.get('message', 'Unknown error')}"
                add_log(msg, "ERROR")
                mds_out = [no_update, dmc.Text(msg, c="red"), no_update, no_update, False]
                notif = dmc.Notification(
                    title="MDS Error",
                    message=msg,
                    color="red",
                    action="show",
                    autoClose=6000,
                    id=f"mds-terminal-{ref.job_id}",
                )
            else:
                mds_filename = payload["mds_filename"]
                n_points = payload["n_points"]
                n_components = payload["n_components"]
                n_groups = payload["n_groups"]
                elapsed = payload["elapsed"]
                mds_out = [
                    payload["mds_results_index"],
                    dmc.Alert(title="MDS Embedding Complete",
                              children=dmc.Text(f"{mds_filename}: {n_points} points, {n_components}D, {n_groups} groups", size="sm"),
                              color="green", variant="light"),
                    False, {}, False,
                ]
                notif = dmc.Notification(
                    title="MDS Computed",
                    message=(
                        f"PCoA: {n_points} points, {n_components}D "
                        f"in {elapsed:.2f}s."
                    ),
                    color="green",
                    action="show",
                    autoClose=3000,
                    id=f"mds-terminal-{ref.job_id}",
                )

        applied = snapshot.terminal_delivery_marker()
        return (*rf_out, *mds_out, notif, True, applied, *sidebar_out)

    @callback(
        Output("compute-job-ack-store", "data"),
        Input("compute-applied-job-store", "data"),
        prevent_initial_call=True,
    )
    def acknowledge_terminal_job(applied_job):
        if not isinstance(applied_job, dict):
            return no_update
        acknowledged = _ack_applied_job(applied_job)
        return {**applied_job, "acknowledged": acknowledged}

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
        default_filename = metadata.get("filename") or selected_mds
        if not default_filename.endswith(".tsv"):
            default_filename = _mds_export_filename(
                metadata.get("source_distmat") or selected_mds
            )
        path = _save_file_dialog(default_filename=default_filename)
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
        Output("mds-result-count-main", "children"),
        Output("mds-result-count-main", "color"),
        Output("export-mds-button", "disabled", allow_duplicate=True),
        Input("mds-result-store", "data"),
        prevent_initial_call=True,
    )
    def update_mds_result_list(mds_index):
        if not mds_index:
            return [], None, "0", "gray", "0", "gray", True
        options = []
        for key, meta in mds_index.items():
            n_groups = len(meta.get("groups", []))
            label = f"{key} — {meta.get('rows', '?')} trees, {n_groups} groups"
            options.append({"value": key, "label": label})
        last_key = list(mds_index.keys())[-1]
        count = str(len(mds_index))
        return options, last_key, count, "blue", count, "blue", False

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
