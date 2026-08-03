"""Single-owner reconciliation for every managed background computation.

The persistent worker and :class:`~treetracer.background_jobs.JobManager`
serialize all heavy work, so the browser needs only one polling owner.  This
module is that owner:

* one 250 ms interval callback observes the active managed job;
* the same callback samples RF/MDS progress sidecars;
* terminal state is copied into one small, replayable browser event;
* job-specific presentation callbacks render that event and write a dedicated
  receipt in the same response as their UI;
* the next interval request carries those receipts back as callback state.

The interval remains enabled until a matching receipt has acknowledged the
sticky terminal event. Terminal retries use a capped lease schedule rather
than firing on every poll, so a slow browser gets an uncontested window to
apply UI and return its receipt. A lost terminal-event response, presentation
response, or settling response therefore causes a later delivery/settling
attempt instead of a permanently stuck loading state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from dash import ALL, Input, Output, State, callback, no_update

from .._worker_log import log as _wlog
from ..background_jobs import (
    TERMINAL_STATES,
    JobRef,
    JobSnapshot,
    JobState,
    job_manager,
)


_MANAGED_KINDS = frozenset(
    {
        "rf",
        "mds",
        "pseudo_ess",
        "consensus",
        "rf_trace",
        "clade_compare",
    }
)


def job_ref_from_store(data: Any, *, expected_kind: str | None = None):
    """Parse a browser job reference without trusting browser data."""
    if not isinstance(data, Mapping):
        return None
    try:
        ref = JobRef.from_dict(data)
    except (KeyError, TypeError, ValueError):
        return None
    if ref.kind not in _MANAGED_KINDS:
        return None
    if expected_kind is not None and ref.kind != expected_kind:
        return None
    return ref


def is_compute_busy(data: Any) -> bool:
    """Return the shared browser-side compute gate."""
    return isinstance(data, Mapping) and data.get("busy") is True


def terminal_event_for_job(
    event_data: Any,
    job_data: Any,
    *,
    expected_kind: str,
):
    """Return a terminal event only when it matches the current browser job.

    Both values are callback *Inputs*, not States.  If a newer job reference
    reaches the browser while an older terminal-render request is still in
    flight, Dash schedules a newer execution and the generation mismatch below
    makes the stale event a no-op.
    """
    event_ref = job_ref_from_store(event_data, expected_kind=expected_kind)
    job_ref = job_ref_from_store(job_data, expected_kind=expected_kind)
    if event_ref is None or event_ref != job_ref:
        return None
    if not isinstance(event_data, Mapping):
        return None
    try:
        state = JobState(str(event_data["state"]))
        revision = int(event_data["terminal_revision"])
        delivery_attempt = int(event_data["delivery_attempt"])
    except (KeyError, TypeError, ValueError):
        return None
    if state not in TERMINAL_STATES or revision < 1 or delivery_attempt < 1:
        return None
    if not isinstance(event_data.get("payload"), Mapping):
        return None
    if not isinstance(event_data.get("metadata"), Mapping):
        return None
    return event_data


def terminal_delivery_marker(event_data: Mapping[str, Any]) -> dict[str, Any]:
    """Build the receipt written atomically with feature terminal UI."""
    ref = JobRef.from_dict(event_data)
    return {
        **ref.as_dict(),
        "terminal_revision": int(event_data["terminal_revision"]),
        "delivery_attempt": int(event_data["delivery_attempt"]),
    }


def _terminal_envelope(snapshot: JobSnapshot) -> dict[str, Any]:
    """Serialize one small terminal snapshot for feature presentation."""
    terminal = snapshot.terminal
    if terminal is None:
        raise ValueError("cannot deliver a non-terminal job snapshot")
    return {
        **snapshot.ref.as_dict(),
        "state": terminal.state.value,
        "terminal_revision": terminal.revision,
        "delivery_attempt": snapshot.delivery_attempt,
        "payload": dict(terminal.payload),
        "metadata": dict(snapshot.metadata),
    }


def _busy_payload(ref: JobRef | None) -> dict[str, Any]:
    if ref is None:
        return {"busy": False}
    return {"busy": True, **ref.as_dict()}


def _busy_update(ref: JobRef | None, current: Any):
    desired = _busy_payload(ref)
    return no_update if current == desired else desired


def _terminal_progress(snapshot: JobSnapshot) -> tuple[float, str]:
    progress = snapshot.progress
    fraction = 0.0 if progress is None else progress.fraction
    label = {
        JobState.SUCCEEDED: "complete",
        JobState.FAILED: "failed",
        JobState.CANCELLED: "cancelled",
    }[snapshot.state]
    return fraction * 100.0, label


def _read_rf_progress(ref: JobRef, progress_path: Any):
    if not progress_path:
        return no_update, no_update
    try:
        data = json.loads(Path(progress_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return no_update, no_update

    value = int(data.get("value", 0))
    total = int(data.get("total", 0))
    fraction = max(0.0, min(float(data.get("fraction", 0.0)), 1.0))
    phase = str(data.get("phase", "computing"))
    job_manager.update_progress(ref, fraction, phase)
    if total == 0:
        return 0, "starting…"
    percentage = fraction * 100.0
    if phase == "finalizing":
        label = f"{value:,} / {total:,} pairs — finalizing…"
    else:
        label = f"{value:,} / {total:,} pairs ({percentage:.1f}%)"
    return percentage, label


def _read_mds_progress(ref: JobRef, progress_path: Any):
    if not progress_path:
        return no_update, no_update
    try:
        data = json.loads(Path(progress_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return no_update, no_update

    fraction = max(0.0, min(float(data.get("fraction", 0.0)), 1.0))
    percentage = fraction * 100.0
    phase = str(data.get("phase", "computing"))
    label = str(data.get("label") or phase.replace("_", " "))
    job_manager.update_progress(ref, fraction, phase, label)
    return percentage, f"{label} ({percentage:.0f}%)"


def _progress_component_updates(
    component_ids: Any,
    rf_progress: tuple[Any, Any],
    mds_progress: tuple[Any, Any],
    *,
    position: int,
) -> list[Any]:
    """Map RF/MDS progress to only the dynamic components in the layout.

    Dash rejects a callback response if it names a concrete Output component
    that is not currently mounted. RF and MDS banners are mutually exclusive
    dynamic children, so wildcard Outputs plus their matched IDs are required
    here. Unknown matches receive ``no_update`` defensively.
    """
    updates = []
    for component_id in component_ids or []:
        which = (
            component_id.get("which")
            if isinstance(component_id, Mapping)
            else None
        )
        if which == "rf":
            updates.append(rf_progress[position])
        elif which == "mds":
            updates.append(mds_progress[position])
        else:
            updates.append(no_update)
    return updates


def _dynamic_progress_outputs(
    progress_bar_ids: Any,
    progress_label_ids: Any,
    rf_progress: tuple[Any, Any] = (no_update, no_update),
    mds_progress: tuple[Any, Any] = (no_update, no_update),
) -> tuple[list[Any], list[Any]]:
    return (
        _progress_component_updates(
            progress_bar_ids,
            rf_progress,
            mds_progress,
            position=0,
        ),
        _progress_component_updates(
            progress_label_ids,
            rf_progress,
            mds_progress,
            position=1,
        ),
    )


def _matching_active_receipt(receipts: Any):
    active = job_manager.active_ref()
    if active is None:
        return None, None
    values = receipts if isinstance(receipts, list) else [receipts]
    for value in values:
        ref = job_ref_from_store(value)
        if ref != active or not isinstance(value, Mapping):
            continue
        try:
            revision = int(value["terminal_revision"])
        except (KeyError, TypeError, ValueError):
            continue
        return active, {**dict(value), "terminal_revision": revision}
    return active, None


def register_job_reconciliation_callbacks():
    @callback(
        Output("compute-poll-interval", "disabled"),
        Output("compute-terminal-event-store", "data"),
        Output("compute-busy-store", "data"),
        Output({"type": "compute-progress-bar", "which": ALL}, "value"),
        Output({"type": "compute-progress-label", "which": ALL}, "children"),
        Input("compute-poll-interval", "n_intervals"),
        # Per-workflow stores wake this single owner immediately after submit.
        Input("rf-job-store", "data"),
        Input("mds-job-store", "data"),
        Input("pseudo-ess-job-store", "data"),
        Input("consensus-job-store", "data"),
        Input("rf-trace-job-store", "data"),
        Input("clade-freq-job-store", "data"),
        State("compute-busy-store", "data"),
        State("rf-progress-path", "data"),
        State("mds-progress-path", "data"),
        State({"type": "compute-progress-bar", "which": ALL}, "id"),
        State({"type": "compute-progress-label", "which": ALL}, "id"),
        State(
            {"type": "compute-terminal-receipt", "kind": ALL},
            "data",
        ),
        prevent_initial_call=True,
    )
    def reconcile_compute_job(
        n_intervals,
        _rf_job_data,
        _mds_job_data,
        _pseudo_ess_job_data,
        _consensus_job_data,
        _rf_trace_job_data,
        _clade_job_data,
        current_busy,
        rf_progress_path,
        mds_progress_path,
        progress_bar_ids,
        progress_label_ids,
        receipts,
    ):
        """Own polling, terminal delivery, progress, and the global gate."""
        # A receipt exists in browser state only after its feature callback's
        # terminal UI response was applied. Piggyback acknowledgement on this
        # already-running poll instead of scheduling another callback in the
        # rapidly updating terminal chain. Exact identity/revision matching
        # makes stale receipts harmless.
        receipt_ref, receipt = _matching_active_receipt(receipts)
        if receipt_ref is not None and receipt is not None:
            if job_manager.acknowledge(
                receipt_ref,
                receipt["terminal_revision"],
            ):
                _wlog(
                    "[parent] reconcile_compute_job: acknowledged "
                    f"job={receipt_ref.job_id}/{receipt_ref.kind}/generation-"
                    f"{receipt_ref.generation}, revision="
                    f"{receipt['terminal_revision']}"
                )

        active = job_manager.active_ref()
        busy = _busy_update(active, current_busy)
        if active is None:
            progress_outputs = _dynamic_progress_outputs(
                progress_bar_ids,
                progress_label_ids,
            )
            return (
                True,
                no_update,
                busy,
                *progress_outputs,
            )

        snapshot = job_manager.snapshot(active)
        if snapshot is None:
            progress_outputs = _dynamic_progress_outputs(
                progress_bar_ids,
                progress_label_ids,
            )
            return (
                True,
                no_update,
                _busy_update(None, current_busy),
                *progress_outputs,
            )

        rf_progress = (no_update, no_update)
        mds_progress = (no_update, no_update)
        terminal_event = no_update

        if snapshot.terminal is not None:
            delivery = job_manager.claim_terminal_delivery(active)
            if delivery is None:
                # ``None`` normally means the previous delivery lease is
                # still active. Clear Data can concurrently remove the record,
                # so re-check ownership before deciding to keep polling.
                current_active = job_manager.active_ref()
                if current_active != active:
                    progress_outputs = _dynamic_progress_outputs(
                        progress_bar_ids,
                        progress_label_ids,
                    )
                    return (
                        current_active is None,
                        no_update,
                        _busy_update(current_active, current_busy),
                        *progress_outputs,
                    )
            else:
                terminal_event = _terminal_envelope(delivery)
                _wlog(
                    "[parent] reconcile_compute_job: delivering "
                    f"job={active.job_id}/{active.kind}/generation-"
                    f"{active.generation}, state={delivery.state.value}, "
                    f"attempt={delivery.delivery_attempt}"
                )

            terminal_progress = snapshot if delivery is None else delivery
            if active.kind == "rf":
                rf_progress = _terminal_progress(terminal_progress)
            elif active.kind == "mds":
                mds_progress = _terminal_progress(terminal_progress)
        elif active.kind == "rf":
            rf_progress = _read_rf_progress(active, rf_progress_path)
        elif active.kind == "mds":
            mds_progress = _read_mds_progress(active, mds_progress_path)
        elif (n_intervals or 0) % 20 == 0:
            _wlog(
                "[parent] reconcile_compute_job: "
                f"job={active.job_id}/{active.kind}/generation-"
                f"{active.generation}, state={snapshot.state.value}"
            )

        # Polling remains enabled through terminal presentation. A later tick
        # carries the browser receipt as State, acknowledges it above, and
        # takes the active=None branch in that same response.
        progress_outputs = _dynamic_progress_outputs(
            progress_bar_ids,
            progress_label_ids,
            rf_progress,
            mds_progress,
        )
        return (
            False,
            terminal_event,
            busy,
            *progress_outputs,
        )
