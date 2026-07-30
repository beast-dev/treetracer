"""Shared consensus tree compute dispatch + polling.

Both the Between-run (``treespace``) and Within-run (``within_run``)
tabs have a "View consensus tree" button. They used to each call
``consensus_tree.assemble_consensus_tree_nexus`` synchronously inside the click callback,
which blocked the Dash request thread for hundreds of ms (or a few
seconds on big presence matrices) with no visual feedback.

This module factors out the shared pieces so both tabs:

* Click → enqueue a consensus tree compute job on the persistent worker, show a
  loading overlay over the active tab, disable the View consensus tree button.
* Wait → ``compute-poll-interval`` ticks at 100 ms; this module's
  ``poll_consensus_tree_completion`` runs once per tick. When the future is done,
  it caches the NEXUS bytes, calls ``state.register_consensus_tree(...)``, fans
  out the result to the right ``*-view-consensus-tree-store`` (which triggers the
  per-tab clientside ``window.open(/peartree/<uid>)`` callback), and
  dismisses the loading overlay.

The per-tab callbacks ``view_consensus_tree`` in ``treespace.py`` and
``within_run.py`` shrink to ~30 lines each — they're only responsible
for validating input, building ``matched_records`` + tab-specific
metadata, and calling ``submit_consensus_tree_job`` here.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any, Dict, Optional

import dash_mantine_components as dmc
from dash import Input, Output, State, callback, no_update

from .. import state as _state
from ..logger import add_log, notif_id
from ..consensus_tree import extract_log_posterior
from . import persistent_worker
from .compute import _get_executor


# ── Module state ───────────────────────────────────────────────────────
# A single in-flight consensus tree job at a time. ``_consensus_tree_meta`` carries the
# tab-specific context the polling callback needs to finalise the
# registry entry and route the result to the right view-consensus-tree-store.

_consensus_tree_future: Optional[Future] = None
_consensus_tree_meta: Dict[str, Any] = {}


def reset() -> None:
    """Interrupt any in-flight consensus tree compute. Called by the sidebar's
    Clear-Data callback so the persistent worker isn't still
    processing a stale request after the DB is wiped.

    ``Future.cancel()`` only drops a not-yet-started future — it can't
    stop a job already running in the worker. ``cancel_current_job()``
    kills the worker, which actually interrupts the compute."""
    global _consensus_tree_future, _consensus_tree_meta
    persistent_worker.cancel_current_job()
    if _consensus_tree_future is not None:
        _consensus_tree_future.cancel()
    _consensus_tree_future = None
    _consensus_tree_meta = {}


def submit_consensus_tree_job(
    *,
    matched_records: list,
    source_distmat: str,
    mode: str,
    selection: list,
    run: Optional[str],
    consensus_tree_coord_by_tree_name: Dict[str, tuple],
    store_target: str,
) -> None:
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
    global _consensus_tree_future, _consensus_tree_meta

    tree_service = _get_tree_service()
    db_manager = tree_service.db_manager

    # ── Build the kwargs the worker needs ──────────────────────────────
    canonical_source = matched_records[0]["file_source"]
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

    _consensus_tree_meta = {
        "mode": mode,
        "run": run,
        "source_distmat": source_distmat,
        "selection": selection,
        "tree_names": [r["name"] for r in matched_records],
        "consensus_tree_coord_by_tree_name": consensus_tree_coord_by_tree_name,
        "store_target": store_target,
    }

    # Lift the rooting flag off the distmat registry. Defaults to True
    # for pre-feature distmats; the worker decides based on this whether
    # to midpoint-root the chosen consensus tree newick before display.
    distmat_is_rooted = _state.get_distmat_is_rooted(source_distmat)

    add_log(
        f"[consensus tree/{mode}] Dispatching to persistent worker "
        f"({len(matched_records)} trees, source {source_distmat}, "
        f"{'rooted' if distmat_is_rooted else 'unrooted+midpoint-root'} mode)..."
    )
    _consensus_tree_future = _get_executor().submit(
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
    )


def _get_tree_service():
    # Lazy local import to avoid the tree_service module being pulled
    # in at callback registration time.
    from ..db.tree_service import get_tree_service
    return get_tree_service()


def register_consensus_tree_compute_callbacks():
    @callback(
        # View-consensus-tree-stores: only one fires per completion (based on mode).
        Output("treespace-view-consensus-tree-store", "data", allow_duplicate=True),
        Output("within-run-view-consensus-tree-store", "data", allow_duplicate=True),
        # Registry — shared by both tabs.
        Output("consensus-tree-registry-store", "data", allow_duplicate=True),
        # Loading overlays — flipped off on both tabs so we don't leave
        # a stale overlay on whichever tab the user might have switched
        # away from mid-compute.
        Output("treespace-loading-overlay", "visible", allow_duplicate=True),
        Output("within-run-loading-overlay", "visible", allow_duplicate=True),
        # View consensus tree buttons — re-enabled on completion.
        Output("treespace-view-consensus-tree", "disabled", allow_duplicate=True),
        Output("within-run-view-consensus-tree", "disabled", allow_duplicate=True),
        # Selection-ring stores — cleared so the orange "selected"
        # marker drops off the MDS view once the compute returns.
        Output("treespace-selected-trees-store", "data", allow_duplicate=True),
        Output("within-run-selected-trees-store", "data", allow_duplicate=True),
        # Notification + the consensus-tree-only poll interval. This callback polls
        # ``consensus-tree-poll-interval`` rather than the shared
        # ``compute-poll-interval`` so it does NOT share an Input — and
        # therefore an allow_duplicate disambiguation hash — with the
        # RF/MDS ``poll_completion`` callback. Sharing the input made
        # both callbacks emit the identical
        # ``compute-poll-interval.disabled`` / ``notifications-container``
        # tokens, which the dash-renderer rejects as duplicates. The hash
        # is derived from the Input signature (see dash/_utils.py).
        Output("notifications-container", "children", allow_duplicate=True),
        Output("consensus-tree-poll-interval", "disabled", allow_duplicate=True),
        Input("consensus-tree-poll-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def poll_consensus_tree_completion(_n):
        global _consensus_tree_future
        if _consensus_tree_future is None or not _consensus_tree_future.done():
            return (no_update,) * 11

        future = _consensus_tree_future
        meta = _consensus_tree_meta
        _consensus_tree_future = None  # consume the future before any further IO

        mode = meta.get("mode", "Between")
        store_target = meta.get("store_target")

        # Idle outputs everywhere except the bits we definitely flip.
        out_treespace_store = no_update
        out_within_store = no_update
        out_registry = no_update
        # Dismiss BOTH overlays on completion (cheap, and the user might
        # have switched tabs mid-compute). The View buttons, though, are
        # scoped to the originating tab in the success path below —
        # enabling both here lights up the OTHER tab's View button for a
        # consensus tree it can't show (the cross-tab leak bug).
        out_treespace_overlay = False
        out_within_overlay = False
        out_treespace_btn = no_update
        out_within_btn = no_update
        out_treespace_sel = no_update
        out_within_sel = no_update

        try:
            result = future.result()
        except persistent_worker.JobCancelled:
            # User Stop — dismiss the overlays + re-enable the buttons
            # (already set above). The cancel callback showed the
            # notification, so don't stack another one here.
            add_log("consensus tree computation cancelled by user.", "WARNING")
            return (out_treespace_store, out_within_store, out_registry,
                    out_treespace_overlay, out_within_overlay,
                    out_treespace_btn, out_within_btn,
                    out_treespace_sel, out_within_sel,
                    no_update, True)
        except Exception as e:
            msg = f"Consensus tree computation failed: {e}"
            add_log(msg, "ERROR")
            notif = dmc.Notification(
                title="Consensus tree Error", message=str(e),
                color="red", action="show", autoClose=6000, id=notif_id(),
            )
            return (out_treespace_store, out_within_store, out_registry,
                    out_treespace_overlay, out_within_overlay,
                    out_treespace_btn, out_within_btn,
                    out_treespace_sel, out_within_sel,
                    notif, True)

        if result.get("missing_taxa"):
            missing = sorted(result["missing_taxa"])
            sample = ", ".join(missing[:5])
            more = "…" if len(missing) > 5 else ""
            notif = dmc.Notification(
                title="Consensus tree Error",
                message=(
                    f"Cannot align translate tables: taxa [{sample}{more}] "
                    "are present in some selected runs but not in the "
                    "canonical Translate block."
                ),
                color="red", action="show", autoClose=8000, id=notif_id(),
            )
            return (out_treespace_store, out_within_store, out_registry,
                    out_treespace_overlay, out_within_overlay,
                    out_treespace_btn, out_within_btn,
                    out_treespace_sel, out_within_sel,
                    notif, True)

        nexus_bytes = result["nexus_bytes"]
        consensus_tree_row = result["consensus_tree_row"]
        consensus_tree_name = consensus_tree_row["name"]
        log_clade_cred = result["log_clade_credibility"]

        uid = _state.cache_consensus_tree(nexus_bytes)

        # consensus tree's (group, treenum) for the green-ring positioning.
        coord = meta.get("consensus_tree_coord_by_tree_name", {}).get(consensus_tree_name)
        if coord is not None:
            consensus_tree_group, consensus_treenum = coord
        else:
            consensus_tree_group, consensus_treenum = None, None

        entry = _state.register_consensus_tree(
            source_distmat=meta["source_distmat"],
            mode=mode,
            run=meta.get("run"),
            uuid=uid,
            consensus_tree={
                "group": consensus_tree_group,
                "treenum": consensus_treenum,
                "tree_name": consensus_tree_name,
            },
            selection=meta["selection"],
            log_clade_credibility=(None if log_clade_cred is None
                                   else float(log_clade_cred)),
            consensus_tree_log_posterior=extract_log_posterior(consensus_tree_row),
            tree_names=meta["tree_names"],
            counts=result["counts"],
            cols_in_consensus_tree=result["cols_in_consensus_tree"],
        )
        registered_name = entry["name"]
        add_log(
            f"Cached consensus tree '{consensus_tree_name}' (from {len(meta['tree_names'])} selected) "
            f"as {uid}; registered as {registered_name}"
        )
        # The rename modal opens next (see ``forward_compute_to_modal``
        # in ``callbacks/rename_consensus_tree.py``); PearTree only opens once the
        # user clicks Save in the modal. Don't promise "opening in
        # PearTree" here — the modal title makes the next step obvious.
        notif = dmc.Notification(
            title="Consensus Tree Ready",
            message=(
                f"Consensus tree {registered_name} computed from "
                f"{len(meta['tree_names'])} selected trees."
            ),
            color="green", action="show", autoClose=3000, id=notif_id(),
        )

        # Route the {uuid, name} payload — and enable the View button —
        # for the originating tab only. The OTHER tab's store and button
        # stay untouched (no_update) so each tab governs its own state.
        payload = {"uuid": uid, "name": registered_name}
        if store_target == "treespace-view-consensus-tree-store":
            out_treespace_store = payload
            out_treespace_sel = []
            out_treespace_btn = False
        else:
            out_within_store = payload
            out_within_sel = []
            out_within_btn = False

        out_registry = _state.get_consensus_tree_registry()

        return (out_treespace_store, out_within_store, out_registry,
                out_treespace_overlay, out_within_overlay,
                out_treespace_btn, out_within_btn,
                out_treespace_sel, out_within_sel,
                notif, True)
