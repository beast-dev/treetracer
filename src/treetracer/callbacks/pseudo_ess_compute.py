"""Managed Pseudo-ESS dispatch, finalization, and terminal polling.

The Diagnostics tab prepares small per-run index lists and submits one worker
request. ``JobManager`` owns the job identity and terminal state, so polling is
read-only and a completed result remains replayable until the browser confirms
that it applied the matching UI response.
"""

from __future__ import annotations

from typing import Any

import dash_mantine_components as dmc
import numpy as np
from dash import Input, Output, callback, no_update

from ..background_jobs import JobRef, JobState, job_manager
from ..logger import add_log
from .._worker_log import log as _wlog
from . import persistent_worker
from .compute import _get_executor, _job_ref_from_store


def reset() -> None:
    """Invalidate an active Pseudo-ESS job before application state clears."""
    active = job_manager.active_ref()
    if active is None or active.kind != "pseudo_ess":
        return
    persistent_worker.cancel_current_job()
    job_manager.invalidate(active)


def _finalize_pseudo_ess_job(
    _ref: JobRef,
    result: Any,
) -> dict[str, Any]:
    """Validate the worker response and retain only its small table payload."""
    if not isinstance(result, dict):
        raise TypeError("Pseudo-ESS worker returned a non-mapping result")
    rows = result.get("results")
    if not isinstance(rows, list):
        raise TypeError("Pseudo-ESS worker result is missing its results list")
    add_log(f"Pseudo-ESS computed for {len(rows)} row(s).")
    return {"results": rows, "n_rows": len(rows)}


def submit_pseudo_ess_job(
    *,
    distmat_path: str,
    names: list[str],
    requests: list[dict[str, Any]],
    n_refs: int,
    seed: int = 0,
) -> JobRef:
    """Enqueue a Pseudo-ESS job covering all ticked runs (+ optional
    Combined row) in a single subprocess round-trip.

    Args:
        distmat_path: path passed straight to the worker.
        names: row/column labels forwarded to the worker.
        requests: list of ``{"label", "indices", "burnin_label"}``
            dicts the worker iterates over.
        n_refs: forwarded to ``compute_pseudo_ess``.
        seed: forwarded.
    """
    add_log(
        f"[Pseudo-ESS] Dispatching to persistent worker "
        f"({len(requests)} row(s), n_refs={n_refs})..."
    )
    _wlog(
        f"[parent] submit_pseudo_ess_job: {len(requests)} requests, "
        f"n_refs={n_refs}, distmat_path={distmat_path!r}"
    )
    ref = job_manager.submit(
        _get_executor(),
        "pseudo_ess",
        persistent_worker.submit_job,
        "compute_pseudo_ess",
        distmat_path=distmat_path,
        names=names,
        requests=requests,
        n_refs=n_refs,
        seed=seed,
        metadata={
            "display_name": "Pseudo-ESS",
            "source_distmat_path": distmat_path,
            "n_rows": len(requests),
        },
        finalizer=_finalize_pseudo_ess_job,
        cancel_exceptions=(persistent_worker.JobCancelled,),
    )
    _wlog(
        "[parent] submit_pseudo_ess_job: managed job created "
        f"({ref.job_id}/generation-{ref.generation})"
    )
    return ref


def _ess_cell(v: float | None):
    """One stoplight-coloured ESS table cell. Thresholds mirror
    Lanfear's rule of thumb."""
    if v is None or np.isnan(v):
        return dmc.TableTd("—")
    if v < 100:
        color = "red"
    elif v < 200:
        color = "orange"
    else:
        color = "green"
    return dmc.TableTd(
        dmc.Text(f"{v:.1f}", c=color, fw=600, span=True)
    )


def _build_result_table(results: list[dict[str, Any]]):
    """Parent-side render of the per-row Pseudo-ESS table. The worker
    only returns plain dicts; this turns them into Mantine table rows."""
    if not results:
        return dmc.Text(
            "Burn-in leaves fewer than 4 trees per run; nothing to compute.",
            c="dimmed", size="sm",
        )

    rows = []
    for r in results:
        rows.append(dmc.TableTr([
            dmc.TableTd(r["label"]),
            dmc.TableTd(str(r["n_trees"])),
            dmc.TableTd(r["burnin_label"]),
            _ess_cell(r.get("min")),
            _ess_cell(r.get("q2")),
            _ess_cell(r.get("max")),
            dmc.TableTd(str(r.get("n_refs_used", 0))),
        ]))

    return dmc.Table(
        [
            dmc.TableThead(
                dmc.TableTr([
                    dmc.TableTh("Run"),
                    dmc.TableTh("Trees"),
                    dmc.TableTh("Burn-in"),
                    dmc.TableTh("Min"),
                    dmc.TableTh("Median"),
                    dmc.TableTh("Max"),
                    dmc.TableTh("# refs"),
                ])
            ),
            dmc.TableTbody(rows),
        ],
        striped=True,
        withTableBorder=True,
        withColumnBorders=True,
        highlightOnHover=True,
    )


def register_pseudo_ess_compute_callbacks():
    @callback(
        Output("pseudo-ess-output", "children", allow_duplicate=True),
        Output("compute-pseudo-ess-button", "disabled", allow_duplicate=True),
        Output("compute-poll-interval", "disabled", allow_duplicate=True),
        Output("compute-applied-job-store", "data", allow_duplicate=True),
        Input("compute-poll-interval", "n_intervals"),
        Input("pseudo-ess-job-store", "data"),
        prevent_initial_call=True,
    )
    def poll_pseudo_ess_completion(n_intervals, job_data):
        ref = _job_ref_from_store(job_data, expected_kind="pseudo_ess")
        if ref is None:
            return (no_update,) * 4

        snapshot = job_manager.snapshot_for_delivery(ref)
        if snapshot is None or snapshot.acknowledged:
            return (no_update,) * 4
        if snapshot.terminal is None:
            if (n_intervals or 0) % 10 == 0:
                _wlog(
                    f"[parent] poll_pseudo_ess_completion tick={n_intervals}: "
                    f"job={ref.job_id}/generation-{ref.generation}, "
                    f"state={snapshot.state.value}"
                )
            return (no_update,) * 4

        terminal = snapshot.terminal
        applied = snapshot.terminal_delivery_marker()
        if terminal.state is JobState.CANCELLED:
            if snapshot.delivery_attempt == 1:
                add_log("Pseudo-ESS computation cancelled by user.", "WARNING")
            output = dmc.Alert(
                title="Pseudo-ESS computation cancelled",
                children=dmc.Text("Stopped before completion.", size="sm"),
                color="gray",
                variant="light",
            )
        elif terminal.state is JobState.FAILED:
            msg = (
                "Pseudo-ESS computation failed: "
                f"{terminal.payload.get('message', 'Unknown error')}"
            )
            if snapshot.delivery_attempt == 1:
                add_log(msg, "ERROR")
            output = dmc.Text(msg, c="red", size="sm")
        else:
            rows = list(terminal.payload.get("results", []))
            output = _build_result_table(rows)

        # The terminal event remains server-side until the applied marker is
        # processed by the shared acknowledgement callback. If this whole Dash
        # response is lost, the still-enabled interval requests the same event
        # again with a new delivery-attempt marker.
        return output, False, True, applied
