"""Protocol-level response-loss scenarios for Stage 6.

These tests deliberately discard values at each browser-delivery boundary.
They complement the manual real-Dash fault harness in
``tools/stage6_browser_fault_harness.js``.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from treetracer.background_jobs import JobManager
from treetracer.callbacks.job_reconcile import (
    _terminal_envelope,
    terminal_delivery_marker,
    terminal_event_for_job,
)


def _wait_for_terminal(manager, ref, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = manager.snapshot(ref)
        if snapshot is not None and snapshot.terminal is not None:
            return snapshot
        time.sleep(0.002)
    pytest.fail("managed job did not become terminal")


def test_dropped_event_and_ui_responses_both_replay_before_receipt():
    manager = JobManager(id_factory=lambda: "fault-job")
    with ThreadPoolExecutor(max_workers=1) as executor:
        ref = manager.submit(
            executor,
            "rf",
            lambda: "raw",
            finalizer=lambda _ref, _raw: {"result_ref": "RF_001"},
        )
        terminal = _wait_for_terminal(manager, ref)

    # Attempt 1: the coordinator's response is dropped before the generic
    # terminal Store changes. No browser adapter runs and there is no receipt.
    dropped_event = _terminal_envelope(
        manager.snapshot_for_delivery(ref)
    )
    assert manager.active_ref() == ref

    # Attempt 2: the generic event lands, but the feature UI response is
    # dropped. Building a receipt server-side is not acknowledgement; it must
    # reach the browser and trigger the acknowledgement request.
    dropped_ui = _terminal_envelope(manager.snapshot_for_delivery(ref))
    assert terminal_event_for_job(
        dropped_ui,
        ref.as_dict(),
        expected_kind="rf",
    ) == dropped_ui
    _discarded_receipt = terminal_delivery_marker(dropped_ui)
    assert manager.snapshot(ref).acknowledged is False
    assert manager.active_ref() == ref

    # Attempt 3 lands fully. Semantic terminal state/revision stayed stable;
    # only the retry counter advanced.
    delivered = _terminal_envelope(manager.snapshot_for_delivery(ref))
    assert delivered["payload"] == dropped_event["payload"]
    assert delivered["terminal_revision"] == dropped_event["terminal_revision"]
    assert delivered["delivery_attempt"] == 3
    receipt = terminal_delivery_marker(delivered)
    assert manager.acknowledge(ref, receipt["terminal_revision"])
    assert manager.active_ref() is None
    assert terminal.terminal.revision == receipt["terminal_revision"]


def test_dropped_ack_response_is_safe_after_server_acknowledgement():
    manager = JobManager(id_factory=lambda: "ack-fault-job")
    with ThreadPoolExecutor(max_workers=1) as executor:
        ref = manager.submit(executor, "mds", lambda: None)
        _wait_for_terminal(manager, ref)

    event = _terminal_envelope(manager.snapshot_for_delivery(ref))
    receipt = terminal_delivery_marker(event)

    # The browser's acknowledgement HTTP response may be lost after the server
    # mutation. That is safe: terminal UI and receipt already landed together.
    assert manager.acknowledge(ref, receipt["terminal_revision"])
    _dropped_ack_response = True
    assert manager.active_ref() is None
    assert manager.snapshot_for_delivery(ref).delivery_attempt == 1


def test_delayed_old_event_cannot_render_over_a_new_generation():
    ids = iter(("old-job", "new-job"))
    manager = JobManager(id_factory=lambda: next(ids))
    with ThreadPoolExecutor(max_workers=1) as executor:
        old_ref = manager.submit(executor, "rf", lambda: None)
        old_terminal = _wait_for_terminal(manager, old_ref)
        old_event = _terminal_envelope(
            manager.snapshot_for_delivery(old_ref)
        )
        assert manager.acknowledge(
            old_ref,
            old_terminal.terminal.revision,
        )

        new_ref = manager.submit(executor, "rf", lambda: None)
        _wait_for_terminal(manager, new_ref)

    # This models a delayed Dash response from the preceding computation.
    assert terminal_event_for_job(
        old_event,
        new_ref.as_dict(),
        expected_kind="rf",
    ) is None
    assert manager.snapshot(new_ref).acknowledged is False
