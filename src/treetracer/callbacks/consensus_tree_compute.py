"""Managed consensus-tree dispatch, publication, and terminal polling.

Both the Between-run (``treespace``) and Within-run (``within_run``)
tabs have a "View consensus tree" button. They used to each call
``consensus_tree.assemble_consensus_tree_nexus`` synchronously inside the click callback,
which blocked the Dash request thread for hundreds of ms (or a few
seconds on big presence matrices) with no visual feedback.

This module factors out the shared pieces so both tabs:

* Click → enqueue a consensus tree compute job on the persistent worker, show a
  loading overlay over the active tab, disable the View consensus tree button.
* Wait → a dedicated interval reads a sticky ``JobManager`` snapshot. The
  success finalizer caches the NEXUS bytes and registers the tree exactly once;
  polling only renders the retained terminal payload and routes it to the
  originating tab.

The per-tab callbacks ``view_consensus_tree`` in ``treespace.py`` and
``within_run.py`` shrink to ~30 lines each — they're only responsible
for validating input, building ``matched_records`` + tab-specific
metadata, and calling ``submit_consensus_tree_job`` here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import partial
from typing import Any

import dash_mantine_components as dmc
from dash import Input, Output, callback, no_update

from .. import state as _state
from ..background_jobs import JobRef, JobState, job_manager
from ..logger import add_log
from ..consensus_tree import extract_log_posterior
from . import persistent_worker
from .compute import _get_executor, _job_ref_from_store


_TREESPACE_TARGET = "treespace-view-consensus-tree-store"
_WITHIN_RUN_TARGET = "within-run-view-consensus-tree-store"
_STORE_TARGETS = frozenset({_TREESPACE_TARGET, _WITHIN_RUN_TARGET})


@dataclass(frozen=True, slots=True)
class _ConsensusFinalizationContext:
    """Parent-only state needed for exactly-once consensus publication."""

    source_distmat: str
    mode: str
    run: str | None
    selection: list[Any]
    tree_names: tuple[str, ...]
    coord_by_tree_name: dict[str, tuple[Any, int]]


class ConsensusTaxaAlignmentError(ValueError):
    """Raised when selected source files cannot share one Translate table."""


def reset() -> None:
    """Invalidate an active consensus job before application state clears."""
    active = job_manager.active_ref()
    if active is None or active.kind != "consensus":
        return
    persistent_worker.cancel_current_job()
    job_manager.invalidate(active)


def _missing_taxa_message(missing_taxa: Any) -> str:
    missing = sorted(str(name) for name in missing_taxa)
    sample = ", ".join(missing[:5])
    more = "…" if len(missing) > 5 else ""
    return (
        f"Cannot align translate tables: taxa [{sample}{more}] are present "
        "in some selected runs but not in the canonical Translate block."
    )


def _finalize_consensus_tree_job(
    _ref: JobRef,
    result: Any,
    *,
    context: _ConsensusFinalizationContext,
) -> dict[str, Any]:
    """Publish one worker result and return a small browser payload."""
    if not isinstance(result, dict):
        raise TypeError("consensus-tree worker returned a non-mapping result")
    if result.get("missing_taxa"):
        raise ConsensusTaxaAlignmentError(
            _missing_taxa_message(result["missing_taxa"])
        )

    nexus_bytes = result.get("nexus_bytes")
    consensus_tree_row = result.get("consensus_tree_row")
    if not isinstance(nexus_bytes, (bytes, bytearray)):
        raise TypeError("consensus-tree worker result is missing NEXUS bytes")
    if not isinstance(consensus_tree_row, dict):
        raise TypeError("consensus-tree worker result is missing its tree row")

    consensus_tree_name = str(consensus_tree_row["name"])
    coord = context.coord_by_tree_name.get(consensus_tree_name)
    if coord is None:
        consensus_tree_group, consensus_treenum = None, None
    else:
        consensus_tree_group, consensus_treenum = coord

    log_clade_cred = result.get("log_clade_credibility")
    log_clade_cred = (
        None if log_clade_cred is None else float(log_clade_cred)
    )
    consensus_tree_log_posterior = extract_log_posterior(consensus_tree_row)

    # This finalizer runs behind JobManager's publication barrier and can only
    # be claimed once. Poll retries never execute these state mutations again.
    uid = _state.cache_consensus_tree(bytes(nexus_bytes))
    entry = _state.register_consensus_tree(
        source_distmat=context.source_distmat,
        mode=context.mode,
        run=context.run,
        uuid=uid,
        consensus_tree={
            "group": consensus_tree_group,
            "treenum": consensus_treenum,
            "tree_name": consensus_tree_name,
        },
        selection=context.selection,
        log_clade_credibility=log_clade_cred,
        consensus_tree_log_posterior=consensus_tree_log_posterior,
        tree_names=context.tree_names,
        counts=result.get("counts"),
        cols_in_consensus_tree=result.get("cols_in_consensus_tree"),
    )
    registered_name = entry["name"]
    n_trees = len(context.tree_names)
    add_log(
        f"Cached consensus tree '{consensus_tree_name}' "
        f"(from {n_trees} selected) as {uid}; registered as {registered_name}"
    )
    return {
        "uuid": uid,
        "name": registered_name,
        "consensus_tree_name": consensus_tree_name,
        "n_trees": n_trees,
        "mode": context.mode,
    }


def submit_consensus_tree_job(
    *,
    matched_records: list,
    source_distmat: str,
    mode: str,
    selection: list,
    run: str | None,
    consensus_tree_coord_by_tree_name: dict[str, tuple],
    store_target: str,
) -> JobRef:
    """Enqueue a consensus tree compute job. Called by both tab callbacks.

    Args:
        matched_records: per-tree dicts (``name``, ``file_source``,
            ``line_offset``, ``line_length``, ``metadata``) in DB order.
            The parent extracts these from a vectorised DataFrame slice.
        source_distmat: distmat name the selection was drawn from.
        mode: ``"Between"`` or ``"Within"`` — propagated into the
            registry entry's mode field and the per-tab labelling.
        selection: serialisable representation of the user's selection,
            stored on the registry entry so the consensus tree list table can
            re-create the orange selection ring.
        run: the run/group name for Within mode, else ``None``.
        consensus_tree_coord_by_tree_name: ``{tree_name: (group, treenum)}`` —
            consulted by the polling callback to populate
            ``consensus_tree.treenum`` (used to put the green ring on the
            consensus tree's MDS dot).
        store_target: ``"treespace-view-consensus-tree-store"`` or
            ``"within-run-view-consensus-tree-store"`` — tells the polling
            callback which tab's clientside ``window.open`` to fire.
    """
    if not matched_records:
        raise ValueError("matched_records must not be empty")
    if store_target not in _STORE_TARGETS:
        raise ValueError(f"unsupported consensus-tree store target: {store_target}")

    tree_service = _get_tree_service()
    db_manager = tree_service.db_manager

    # ── Build the kwargs the worker needs ──────────────────────────────
    unique_sources = list(dict.fromkeys(r["file_source"] for r in matched_records))

    snapshots_path = _state.get_snapshots_path(source_distmat)
    full_distmat_names = _state.get_distmat_names(source_distmat)

    source_file_paths = {
        fs: db_manager._source_files[fs]
        for fs in unique_sources
        if fs in db_manager._source_files
    }
    translate_maps = {
        fs: db_manager.get_translate_map(fs) for fs in unique_sources
    }
    source_preambles = {
        fs: db_manager._source_preambles.get(fs)
        for fs in unique_sources
        if fs in getattr(db_manager, "_source_preambles", {})
    }

    tree_names = tuple(str(r["name"]) for r in matched_records)
    context = _ConsensusFinalizationContext(
        mode=str(mode),
        run=None if run is None else str(run),
        source_distmat=str(source_distmat),
        selection=copy.deepcopy(selection),
        tree_names=tree_names,
        coord_by_tree_name={
            str(name): (coord[0], int(coord[1]))
            for name, coord in consensus_tree_coord_by_tree_name.items()
        },
    )

    # Lift the rooting flag off the distmat registry. Defaults to True
    # for pre-feature distmats; the worker decides based on this whether
    # to midpoint-root the chosen consensus tree newick before display.
    distmat_is_rooted = _state.get_distmat_is_rooted(source_distmat)

    add_log(
        f"[consensus tree/{mode}] Dispatching to persistent worker "
        f"({len(matched_records)} trees, source {source_distmat}, "
        f"{'rooted' if distmat_is_rooted else 'unrooted+midpoint-root'} mode)..."
    )
    ref = job_manager.submit(
        _get_executor(),
        "consensus",
        persistent_worker.submit_job,
        "compute_consensus_tree",
        matched_records=matched_records,
        source_distmat=source_distmat,
        snapshots_path=str(snapshots_path),
        full_distmat_names=list(full_distmat_names),
        translate_maps=translate_maps,
        source_file_paths=source_file_paths,
        source_preambles=source_preambles,
        is_rooted=distmat_is_rooted,
        metadata={
            "display_name": f"{mode} consensus tree",
            "mode": mode,
            "source_distmat": source_distmat,
            "store_target": store_target,
        },
        finalizer=partial(
            _finalize_consensus_tree_job,
            context=context,
        ),
        cancel_exceptions=(persistent_worker.JobCancelled,),
    )
    return ref


def _get_tree_service():
    # Lazy local import to avoid the tree_service module being pulled
    # in at callback registration time.
    from ..db.tree_service import get_tree_service
    return get_tree_service()


def register_consensus_tree_compute_callbacks():
    @callback(
        # View stores: only the originating tab changes.
        Output("treespace-view-consensus-tree-store", "data", allow_duplicate=True),
        Output("within-run-view-consensus-tree-store", "data", allow_duplicate=True),
        Output("consensus-tree-registry-store", "data", allow_duplicate=True),
        # Both overlays are dismissed in case the user changed tabs.
        Output("treespace-loading-overlay", "visible", allow_duplicate=True),
        Output("within-run-loading-overlay", "visible", allow_duplicate=True),
        # Only the originating button is re-enabled.
        Output("treespace-view-consensus-tree", "disabled", allow_duplicate=True),
        Output("within-run-view-consensus-tree", "disabled", allow_duplicate=True),
        # The originating selection is cleared on success.
        Output("treespace-selected-trees-store", "data", allow_duplicate=True),
        Output("within-run-selected-trees-store", "data", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
        Output("consensus-tree-poll-interval", "disabled", allow_duplicate=True),
        Output("compute-applied-job-store", "data", allow_duplicate=True),
        Input("consensus-tree-poll-interval", "n_intervals"),
        Input("consensus-job-store", "data"),
        prevent_initial_call=True,
    )
    def poll_consensus_tree_completion(n_intervals, job_data):
        ref = _job_ref_from_store(job_data, expected_kind="consensus")
        if ref is None:
            return (no_update,) * 12

        snapshot = job_manager.snapshot_for_delivery(ref)
        if snapshot is None or snapshot.acknowledged:
            return (no_update,) * 12
        if snapshot.terminal is None:
            return (no_update,) * 12

        terminal = snapshot.terminal
        payload = terminal.payload
        store_target = snapshot.metadata.get("store_target")

        out_treespace_store = no_update
        out_within_store = no_update
        out_registry = no_update
        out_treespace_overlay = False
        out_within_overlay = False
        out_treespace_btn = no_update
        out_within_btn = no_update
        out_treespace_sel = no_update
        out_within_sel = no_update
        if store_target == _TREESPACE_TARGET:
            out_treespace_btn = False
        elif store_target == _WITHIN_RUN_TARGET:
            out_within_btn = False

        notif = no_update
        if terminal.state is JobState.CANCELLED:
            if snapshot.delivery_attempt == 1:
                add_log("Consensus tree computation cancelled by user.", "WARNING")
        elif terminal.state is JobState.FAILED:
            message = str(payload.get("message", "Unknown error"))
            if snapshot.delivery_attempt == 1:
                add_log(f"Consensus tree computation failed: {message}", "ERROR")
            notif = dmc.Notification(
                title="Consensus tree Error",
                message=message,
                color="red",
                action="show",
                autoClose=8000,
                id=f"consensus-terminal-{ref.job_id}",
            )
        else:
            view_payload = {"uuid": payload["uuid"], "name": payload["name"]}
            if store_target == _TREESPACE_TARGET:
                out_treespace_store = view_payload
                out_treespace_sel = []
            elif store_target == _WITHIN_RUN_TARGET:
                out_within_store = view_payload
                out_within_sel = []
            out_registry = _state.get_consensus_tree_registry()
            notif = dmc.Notification(
                title="Consensus Tree Ready",
                message=(
                    f"Consensus tree {payload['name']} computed from "
                    f"{payload['n_trees']} selected trees."
                ),
                color="green",
                action="show",
                autoClose=3000,
                id=f"consensus-terminal-{ref.job_id}",
            )

        return (
            out_treespace_store,
            out_within_store,
            out_registry,
            out_treespace_overlay,
            out_within_overlay,
            out_treespace_btn,
            out_within_btn,
            out_treespace_sel,
            out_within_sel,
            notif,
            True,
            snapshot.terminal_delivery_marker(),
        )
