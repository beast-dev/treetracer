"""Pseudo-ESS dispatch + polling.

Same shape as ``mcc_compute.py``: the Diagnostics tab's
"Compute Pseudo-ESS" click handler validates input, slices the
distmat into per-run index lists, and hands the work off to the
persistent worker subprocess via ``persistent_worker.submit_job``.

The click handler returns immediately with a loading spinner in the
``pseudo-ess-output`` slot. ``poll_pseudo_ess_completion`` listens to
``compute-poll-interval`` and, on the worker's response, builds the
result table parent-side and writes it back.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any, Dict, List, Optional

import dash_mantine_components as dmc
import numpy as np
from dash import Input, Output, callback, html, no_update

from ..logger import add_log, notif_id
from .._worker_log import log as _wlog
from . import persistent_worker
from .compute import _get_executor


_pseudo_ess_future: Optional[Future] = None
_pseudo_ess_meta: Dict[str, Any] = {}


def reset() -> None:
    """Interrupt any in-flight Pseudo-ESS compute. Called by sidebar's
    Clear-Data callback so the worker isn't still processing against
    a distmat that no longer exists.

    ``Future.cancel()`` only drops a not-yet-started future — it can't
    stop a job already running in the worker. ``cancel_current_job()``
    kills the worker, which actually interrupts the compute."""
    global _pseudo_ess_future, _pseudo_ess_meta
    persistent_worker.cancel_current_job()
    if _pseudo_ess_future is not None:
        _pseudo_ess_future.cancel()
    _pseudo_ess_future = None
    _pseudo_ess_meta = {}


def submit_pseudo_ess_job(
    *,
    distmat_path: str,
    names: List[str],
    requests: List[Dict[str, Any]],
    n_refs: int,
    seed: int = 0,
) -> None:
    """Enqueue a Pseudo-ESS job covering all ticked runs (+ optional
    Combined row) in a single subprocess round-trip.

    Args:
        distmat_path: path passed straight to the worker.
        names: row/col labels (saved on ``_pseudo_ess_meta`` only for
            potential future cancellation logging; the worker doesn't
            read them).
        requests: list of ``{"label", "indices", "burnin_label"}``
            dicts the worker iterates over.
        n_refs: forwarded to ``compute_pseudo_ess``.
        seed: forwarded.
    """
    global _pseudo_ess_future, _pseudo_ess_meta

    _pseudo_ess_meta = {
        "distmat_path": distmat_path,
        "n_runs": len(requests),
    }
    add_log(
        f"[Pseudo-ESS] Dispatching to persistent worker "
        f"({len(requests)} row(s), n_refs={n_refs})..."
    )
    _wlog(
        f"[parent] submit_pseudo_ess_job: {len(requests)} requests, "
        f"n_refs={n_refs}, distmat_path={distmat_path!r}"
    )
    _pseudo_ess_future = _get_executor().submit(
        persistent_worker.submit_job,
        "compute_pseudo_ess",
        distmat_path=distmat_path,
        names=names,
        requests=requests,
        n_refs=n_refs,
        seed=seed,
    )
    _wlog(f"[parent] submit_pseudo_ess_job: future created (id={id(_pseudo_ess_future):x})")


def _ess_cell(v: Optional[float]):
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


def _build_result_table(results: List[Dict[str, Any]]):
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
            _ess_cell(r.get("q1")),
            _ess_cell(r.get("q2")),
            _ess_cell(r.get("q3")),
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
                    dmc.TableTh("Q1"),
                    dmc.TableTh("Q2 (median)"),
                    dmc.TableTh("Q3"),
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
    # Why only 2 outputs (not 4) — see the giant block comment below.
    @callback(
        Output("pseudo-ess-output", "children", allow_duplicate=True),
        Output("compute-pseudo-ess-button", "disabled", allow_duplicate=True),
        Input("compute-poll-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def poll_pseudo_ess_completion(_n):
        # ────────────────────────────────────────────────────────────
        # Output contention: WHY this callback returns ONLY 2 values
        # (table + button.disabled) instead of also writing
        # ``compute-poll-interval.disabled`` and
        # ``notifications-container.children``.
        #
        # Three poll callbacks share ``Input("compute-poll-interval",
        # "n_intervals")``:
        #
        #   * poll_completion       (compute.py — RF / MDS)
        #   * poll_mcc_completion   (mcc_compute.py)
        #   * poll_pseudo_ess_completion (here)
        #
        # On every 100 ms tick they fire concurrently. When the ESS
        # job finishes, this callback returns a real value for
        # ``compute-poll-interval.disabled``, while the other two
        # return ``no_update`` for everything (no RF/MDS/MCC running).
        # ``compute-poll-interval.disabled``'s primary writer
        # (no ``allow_duplicate``) lives in ``handle_compute_rf`` in
        # compute.py. Dash 4.x has a bug — observed empirically on
        # the Windows bundle and verified by file-log trace — where
        # this secondary-write-during-primary-not-firing pattern
        # causes the *entire callback batch* to be silently dropped:
        # the table never reaches the GUI, the interval stays
        # enabled, the button stays disabled.
        #
        # Dropping ``compute-poll-interval.disabled`` from THIS
        # callback's Outputs removes the conflict. The interval just
        # keeps firing forever after ESS finishes — every 100 ms it
        # re-enters this callback, sees ``_pseudo_ess_future is None``,
        # returns ``(no_update,) * 2``. Cost: a few µs per tick. The
        # other two poll callbacks also no-op on idle ticks today.
        #
        # Dropping ``notifications-container.children`` for the same
        # reason — many callbacks fight over it via allow_duplicate.
        # The user sees the table appear, which is the only signal
        # they need; a green toast on top is redundant.
        # ────────────────────────────────────────────────────────────
        global _pseudo_ess_future
        future_set = _pseudo_ess_future is not None
        future_done = future_set and _pseudo_ess_future.done()
        if future_set:
            _wlog(
                f"[parent] poll_pseudo_ess_completion: "
                f"future_set={future_set}, future_done={future_done}, "
                f"future_id={id(_pseudo_ess_future):x}"
            )
        if not future_set or not future_done:
            return no_update, no_update

        future = _pseudo_ess_future
        _pseudo_ess_future = None
        _wlog("[parent] poll_pseudo_ess_completion: future done, calling .result()")

        try:
            result = future.result()
            _wlog(
                f"[parent] poll_pseudo_ess_completion: .result() returned; "
                f"type={type(result).__name__}, "
                f"keys={sorted(result.keys()) if isinstance(result, dict) else '<not a dict>'}"
            )
        except persistent_worker.JobCancelled:
            _wlog("[parent] poll_pseudo_ess_completion: JobCancelled")
            add_log("Pseudo-ESS computation cancelled by user.", "WARNING")
            return (
                dmc.Alert(
                    title="Pseudo-ESS computation cancelled",
                    children=dmc.Text("Stopped before completion.", size="sm"),
                    color="gray", variant="light",
                ),
                False,        # re-enable button
            )
        except Exception as e:
            _wlog(f"[parent] poll_pseudo_ess_completion: .result() raised {type(e).__name__}: {e}")
            msg = f"Pseudo-ESS computation failed: {e}"
            add_log(msg, "ERROR")
            return (
                dmc.Text(msg, c="red", size="sm"),
                False,                  # re-enable button
            )

        rows = result.get("results", [])
        _wlog(f"[parent] poll_pseudo_ess_completion: building result table for {len(rows)} rows")
        add_log(f"Pseudo-ESS computed for {len(rows)} row(s).")
        try:
            table = _build_result_table(rows)
            _wlog("[parent] poll_pseudo_ess_completion: result table built; returning")
        except Exception as e:
            _wlog(
                f"[parent] poll_pseudo_ess_completion: _build_result_table raised "
                f"{type(e).__name__}: {e}"
            )
            raise
        return (
            table,
            False,                      # re-enable button
        )
