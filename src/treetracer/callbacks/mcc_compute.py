"""Shared MCC compute dispatch + polling.

Both the Between-run (``treespace``) and Within-run (``within_run``)
tabs have a "View MCC" button. They used to each call
``mcc.assemble_mcc_nexus`` synchronously inside the click callback,
which blocked the Dash request thread for hundreds of ms (or a few
seconds on big presence matrices) with no visual feedback.

This module factors out the shared pieces so both tabs:

* Click → enqueue an MCC compute job on the persistent worker, show a
  loading overlay over the active tab, disable the View MCC button.
* Wait → ``compute-poll-interval`` ticks at 100 ms; this module's
  ``poll_mcc_completion`` runs once per tick. When the future is done,
  it caches the NEXUS bytes, calls ``state.register_mcc(...)``, fans
  out the result to the right ``*-view-mcc-store`` (which triggers the
  per-tab clientside ``window.open(/peartree/<uid>)`` callback), and
  dismisses the loading overlay.

The per-tab callbacks ``view_mcc_tree`` in ``treespace.py`` and
``within_run.py`` shrink to ~30 lines each — they're only responsible
for validating input, building ``matched_records`` + tab-specific
metadata, and calling ``submit_mcc_job`` here.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any, Dict, Optional

import dash_mantine_components as dmc
from dash import Input, Output, State, callback, no_update

from .. import state as _state
from ..logger import add_log, notif_id
from ..mcc import extract_log_posterior
from . import persistent_worker
from .compute import _get_executor


# ── Module state ───────────────────────────────────────────────────────
# A single in-flight MCC job at a time. ``_mcc_meta`` carries the
# tab-specific context the polling callback needs to finalise the
# registry entry and route the result to the right view-mcc-store.

_mcc_future: Optional[Future] = None
_mcc_meta: Dict[str, Any] = {}


def reset() -> None:
    """Cancel any in-flight MCC future. Called by the sidebar's
    Clear-Data callback so the persistent worker isn't still
    processing a stale request after the DB is wiped."""
    global _mcc_future, _mcc_meta
    if _mcc_future is not None:
        _mcc_future.cancel()
    _mcc_future = None
    _mcc_meta = {}


def submit_mcc_job(
    *,
    matched_records: list,
    source_distmat: str,
    mode: str,
    selection: list,
    run: Optional[str],
    mcc_coord_by_tree_name: Dict[str, tuple],
    store_target: str,
) -> None:
    """Enqueue an MCC compute job. Called by both tab callbacks.

    Args:
        matched_records: per-tree dicts (``name``, ``file_source``,
            ``line_offset``, ``line_length``, ``metadata``) in DB order.
            The parent extracts these from a vectorised DataFrame slice.
        source_distmat: distmat name the selection was drawn from.
        mode: ``"Between"`` or ``"Within"`` — propagated into the
            registry entry's mode field and the per-tab labelling.
        selection: serialisable representation of the user's selection,
            stored on the registry entry so the MCC list table can
            re-create the orange selection ring.
        run: the run/group name for Within mode, else ``None``.
        mcc_coord_by_tree_name: ``{tree_name: (group, treenum)}`` —
            consulted by the polling callback to populate
            ``mcc_tree.treenum`` (used to put the green ring on the
            MCC's MDS dot).
        store_target: ``"treespace-view-mcc-store"`` or
            ``"within-run-view-mcc-store"`` — tells the polling
            callback which tab's clientside ``window.open`` to fire.
    """
    global _mcc_future, _mcc_meta

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

    _mcc_meta = {
        "mode": mode,
        "run": run,
        "source_distmat": source_distmat,
        "selection": selection,
        "tree_names": [r["name"] for r in matched_records],
        "mcc_coord_by_tree_name": mcc_coord_by_tree_name,
        "store_target": store_target,
    }

    add_log(
        f"[MCC/{mode}] Dispatching to persistent worker "
        f"({len(matched_records)} trees, source {source_distmat})..."
    )
    _mcc_future = _get_executor().submit(
        persistent_worker.submit_job,
        "compute_mcc",
        matched_records=matched_records,
        source_distmat=source_distmat,
        snapshots_path=str(snapshots_path),
        full_distmat_names=list(full_distmat_names),
        translate_maps=translate_maps,
        source_file_paths=source_file_paths,
        source_preambles=source_preambles,
    )


def _get_tree_service():
    # Lazy local import to avoid the tree_service module being pulled
    # in at callback registration time.
    from ..db.tree_service import get_tree_service
    return get_tree_service()


def register_mcc_compute_callbacks():
    @callback(
        # View-mcc-stores: only one fires per completion (based on mode).
        Output("treespace-view-mcc-store", "data", allow_duplicate=True),
        Output("within-run-view-mcc-store", "data", allow_duplicate=True),
        # Registry — shared by both tabs.
        Output("mcc-registry-store", "data", allow_duplicate=True),
        # Loading overlays — flipped off on both tabs so we don't leave
        # a stale overlay on whichever tab the user might have switched
        # away from mid-compute.
        Output("treespace-loading-overlay", "visible", allow_duplicate=True),
        Output("within-run-loading-overlay", "visible", allow_duplicate=True),
        # View MCC buttons — re-enabled on completion.
        Output("treespace-view-mcc", "disabled", allow_duplicate=True),
        Output("within-run-view-mcc", "disabled", allow_duplicate=True),
        # Selection-ring stores — cleared so the orange "selected"
        # marker drops off the MDS view once the compute returns.
        Output("treespace-selected-trees-store", "data", allow_duplicate=True),
        Output("within-run-selected-trees-store", "data", allow_duplicate=True),
        # Notification + the shared poll interval.
        Output("notifications-container", "children", allow_duplicate=True),
        Output("compute-poll-interval", "disabled", allow_duplicate=True),
        Input("compute-poll-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def poll_mcc_completion(_n):
        global _mcc_future
        if _mcc_future is None or not _mcc_future.done():
            return (no_update,) * 11

        future = _mcc_future
        meta = _mcc_meta
        _mcc_future = None  # consume the future before any further IO

        mode = meta.get("mode", "Between")
        store_target = meta.get("store_target")

        # Idle outputs everywhere except the bits we definitely flip.
        out_treespace_store = no_update
        out_within_store = no_update
        out_registry = no_update
        # Always dismiss BOTH overlays + re-enable BOTH buttons on
        # completion. Cheap, and the user might have switched tabs.
        out_treespace_overlay = False
        out_within_overlay = False
        out_treespace_btn = False
        out_within_btn = False
        out_treespace_sel = no_update
        out_within_sel = no_update

        try:
            result = future.result()
        except Exception as e:
            msg = f"MCC computation failed: {e}"
            add_log(msg, "ERROR")
            notif = dmc.Notification(
                title="MCC Error", message=str(e),
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
                title="MCC Error",
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
        mcc_row = result["mcc_row"]
        mcc_tree_name = mcc_row["name"]
        log_clade_cred = result["log_clade_credibility"]

        uid = _state.cache_mcc_tree(nexus_bytes)

        # MCC's (group, treenum) for the green-ring positioning.
        coord = meta.get("mcc_coord_by_tree_name", {}).get(mcc_tree_name)
        if coord is not None:
            mcc_group, mcc_treenum = coord
        else:
            mcc_group, mcc_treenum = None, None

        entry = _state.register_mcc(
            source_distmat=meta["source_distmat"],
            mode=mode,
            run=meta.get("run"),
            uuid=uid,
            mcc_tree={
                "group": mcc_group,
                "treenum": mcc_treenum,
                "tree_name": mcc_tree_name,
            },
            selection=meta["selection"],
            log_clade_credibility=(None if log_clade_cred is None
                                   else float(log_clade_cred)),
            mcc_log_posterior=extract_log_posterior(mcc_row),
            tree_names=meta["tree_names"],
            counts=result["counts"],
            cols_in_mcc=result["cols_in_mcc"],
        )
        registered_name = entry["name"]
        add_log(
            f"Cached MCC tree '{mcc_tree_name}' (from {len(meta['tree_names'])} selected) "
            f"as {uid}; registered as {registered_name}"
        )
        notif = dmc.Notification(
            title="MCC Tree Ready",
            message=(
                f"MCC tree {registered_name} (from {len(meta['tree_names'])} selected) "
                "— opening in PearTree…"
            ),
            color="green", action="show", autoClose=4000, id=notif_id(),
        )

        # Route the {uuid, name} payload to the originating tab's
        # view-mcc-store. The OTHER tab's store stays untouched.
        payload = {"uuid": uid, "name": registered_name}
        if store_target == "treespace-view-mcc-store":
            out_treespace_store = payload
            out_treespace_sel = []
        else:
            out_within_store = payload
            out_within_sel = []

        out_registry = _state.get_mcc_registry()

        return (out_treespace_store, out_within_store, out_registry,
                out_treespace_overlay, out_within_overlay,
                out_treespace_btn, out_within_btn,
                out_treespace_sel, out_within_sel,
                notif, True)
