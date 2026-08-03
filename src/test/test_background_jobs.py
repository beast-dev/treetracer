"""Correctness tests for sticky background-job lifecycle state."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from treetracer.background_jobs import (
    JobBusyError,
    JobManager,
    JobState,
)


def _wait_for_state(manager, ref, expected, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = manager.snapshot(ref)
        if snapshot is not None and snapshot.state is expected:
            return snapshot
        time.sleep(0.005)
    snapshot = manager.snapshot(ref)
    pytest.fail(
        f"job did not reach {expected.value}; "
        f"last state={None if snapshot is None else snapshot.state.value}"
    )


def test_success_is_sticky_until_matching_acknowledgement():
    now = [10.0]
    manager = JobManager(clock=lambda: now[0])
    with ThreadPoolExecutor(max_workers=1) as executor:
        ref = manager.submit(
            executor,
            "rf",
            lambda: {"matrix": "large-result"},
            metadata={"display_name": "RF_001"},
            finalizer=lambda _ref, result: {
                "result_ref": "RF_001",
                "source": result["matrix"],
            },
        )
        terminal = _wait_for_state(manager, ref, JobState.SUCCEEDED)

    assert terminal.progress.fraction == 1.0
    assert terminal.progress.phase == "done"
    assert terminal.terminal.payload["result_ref"] == "RF_001"
    assert terminal.metadata == {"display_name": "RF_001"}
    assert manager.active_ref() == ref

    first = manager.claim_terminal_delivery(ref)
    assert manager.claim_terminal_delivery(ref) is None
    now[0] += 1.0
    second = manager.claim_terminal_delivery(ref)
    assert first.terminal == second.terminal
    assert first.delivery_attempt == 1
    assert second.delivery_attempt == 2
    assert manager.snapshot(ref).terminal == terminal.terminal

    assert manager.acknowledge(ref, terminal.terminal.revision) is True
    assert manager.snapshot(ref).acknowledged is True
    assert manager.active_ref() is None


def test_stale_generation_cannot_read_update_or_acknowledge_job():
    manager = JobManager()
    with ThreadPoolExecutor(max_workers=1) as executor:
        ref = manager.submit(executor, "mds", lambda: 1)
        terminal = _wait_for_state(manager, ref, JobState.SUCCEEDED)

    stale = replace(ref, generation=ref.generation + 1)
    assert manager.snapshot(stale) is None
    assert manager.update_progress(stale, 0.5, "wrong job") is False
    assert manager.acknowledge(stale, terminal.terminal.revision) is False
    assert manager.snapshot(ref).acknowledged is False


def test_finalizer_runs_once_and_delivery_lease_has_one_concurrent_winner():
    manager = JobManager()
    finalizer_entered = threading.Event()
    release_finalizer = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def finalize(_ref, result):
        nonlocal calls
        with calls_lock:
            calls += 1
        finalizer_entered.set()
        assert release_finalizer.wait(timeout=2)
        return {"result_ref": result}

    with ThreadPoolExecutor(max_workers=1) as executor:
        ref = manager.submit(
            executor,
            "pseudo_ess",
            lambda: "ESS_001",
            finalizer=finalize,
        )
        assert finalizer_entered.wait(timeout=2)
        assert manager.snapshot(ref).state is JobState.FINALIZING
        release_finalizer.set()
        _wait_for_state(manager, ref, JobState.SUCCEEDED)

    barrier = threading.Barrier(9)
    snapshots = []
    snapshots_lock = threading.Lock()

    def read_terminal():
        barrier.wait(timeout=2)
        value = manager.claim_terminal_delivery(ref)
        with snapshots_lock:
            snapshots.append(value)

    threads = [threading.Thread(target=read_terminal) for _ in range(8)]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=2)
    for thread in threads:
        thread.join(timeout=2)

    assert calls == 1
    assert len(snapshots) == 8
    deliveries = [snapshot for snapshot in snapshots if snapshot is not None]
    assert len(deliveries) == 1
    assert deliveries[0].terminal.payload["result_ref"] == "ESS_001"
    assert deliveries[0].delivery_attempt == 1
    assert manager.snapshot(ref).delivery_attempt == 1


def test_terminal_delivery_retries_follow_the_capped_lease_schedule():
    now = [20.0]
    manager = JobManager(
        clock=lambda: now[0],
        terminal_retry_delays=(0.5, 1.0, 2.0),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        ref = manager.submit(executor, "rf", lambda: None)
        _wait_for_state(manager, ref, JobState.SUCCEEDED)

    assert manager.claim_terminal_delivery(ref).delivery_attempt == 1
    now[0] += 0.49
    assert manager.claim_terminal_delivery(ref) is None
    now[0] += 0.01
    assert manager.claim_terminal_delivery(ref).delivery_attempt == 2
    now[0] += 0.99
    assert manager.claim_terminal_delivery(ref) is None
    now[0] += 0.01
    assert manager.claim_terminal_delivery(ref).delivery_attempt == 3
    now[0] += 2.0
    assert manager.claim_terminal_delivery(ref).delivery_attempt == 4
    now[0] += 2.0
    assert manager.claim_terminal_delivery(ref).delivery_attempt == 5


@pytest.mark.parametrize(
    "delays",
    [(), (0.0,), (float("inf"),), (2.0, 1.0)],
)
def test_terminal_delivery_retry_schedule_must_be_valid(delays):
    with pytest.raises(ValueError):
        JobManager(terminal_retry_delays=delays)


def test_compute_exception_becomes_sticky_failure():
    manager = JobManager()

    def fail():
        raise RuntimeError("worker broke")

    with ThreadPoolExecutor(max_workers=1) as executor:
        ref = manager.submit(executor, "mds", fail)
        terminal = _wait_for_state(manager, ref, JobState.FAILED)

    assert terminal.terminal.payload == {
        "message": "worker broke",
        "error_type": "RuntimeError",
        "stage": "compute",
    }
    assert manager.claim_terminal_delivery(ref).state is JobState.FAILED
    assert manager.claim_terminal_delivery(ref) is None


def test_configured_worker_cancellation_exception_is_not_an_error():
    class WorkerCancelled(RuntimeError):
        pass

    manager = JobManager()

    def cancel():
        raise WorkerCancelled("stopped by user")

    with ThreadPoolExecutor(max_workers=1) as executor:
        ref = manager.submit(
            executor,
            "consensus",
            cancel,
            cancel_exceptions=(WorkerCancelled,),
        )
        terminal = _wait_for_state(manager, ref, JobState.CANCELLED)

    assert terminal.terminal.payload["message"] == "stopped by user"
    assert terminal.terminal.payload["error_type"] == "WorkerCancelled"


def test_finalizer_exception_is_terminal_failure_and_not_retried():
    manager = JobManager()
    calls = 0

    def broken_finalizer(_ref, _result):
        nonlocal calls
        calls += 1
        raise ValueError("could not register result")

    with ThreadPoolExecutor(max_workers=1) as executor:
        ref = manager.submit(
            executor,
            "rf",
            lambda: object(),
            finalizer=broken_finalizer,
        )
        terminal = _wait_for_state(manager, ref, JobState.FAILED)

    for _ in range(5):
        manager.snapshot(ref)
    assert calls == 1
    assert terminal.terminal.payload["stage"] == "finalize"
    assert terminal.terminal.payload["error_type"] == "ValueError"


def test_explicit_cancellation_cannot_be_overwritten_by_late_success():
    manager = JobManager()
    task_started = threading.Event()
    release_task = threading.Event()

    def work():
        task_started.set()
        assert release_task.wait(timeout=2)
        return "too late"

    with ThreadPoolExecutor(max_workers=1) as executor:
        ref = manager.submit(executor, "rf", work)
        assert task_started.wait(timeout=2)
        assert manager.mark_cancelled(ref, message="reset by user") is True
        cancelled = manager.snapshot(ref)
        assert cancelled.state is JobState.CANCELLED
        release_task.set()

    after_completion = manager.snapshot(ref)
    assert after_completion.state is JobState.CANCELLED
    assert after_completion.terminal.payload["message"] == "reset by user"


def test_single_active_gate_opens_only_after_terminal_acknowledgement():
    manager = JobManager(single_active=True)
    task_started = threading.Event()
    release_task = threading.Event()

    def work():
        task_started.set()
        assert release_task.wait(timeout=2)

    with ThreadPoolExecutor(max_workers=1) as executor:
        first = manager.submit(executor, "rf", work)
        assert task_started.wait(timeout=2)
        with pytest.raises(JobBusyError) as exc_info:
            manager.submit(executor, "mds", lambda: None)
        assert exc_info.value.active == first

        release_task.set()
        terminal = _wait_for_state(manager, first, JobState.SUCCEEDED)
        with pytest.raises(JobBusyError):
            manager.submit(executor, "mds", lambda: None)

        assert manager.acknowledge(first, terminal.terminal.revision)
        second = manager.submit(executor, "mds", lambda: None)
        _wait_for_state(manager, second, JobState.SUCCEEDED)


def test_progress_is_monotonic_clamped_and_terminal_is_authoritative():
    manager = JobManager()
    task_started = threading.Event()
    release_task = threading.Event()

    def work():
        task_started.set()
        assert release_task.wait(timeout=2)

    with ThreadPoolExecutor(max_workers=1) as executor:
        ref = manager.submit(executor, "mds", work)
        assert task_started.wait(timeout=2)
        assert manager.update_progress(ref, 0.7, "eigensolve", "70%")
        assert manager.update_progress(ref, 0.4, "stale sidecar", "40%")
        assert manager.snapshot(ref).progress.fraction == 0.7
        assert manager.update_progress(ref, 4.0, "finalizing")
        assert manager.snapshot(ref).progress.fraction == 1.0
        with pytest.raises(ValueError):
            manager.update_progress(ref, float("nan"), "invalid")

        release_task.set()
        terminal = _wait_for_state(manager, ref, JobState.SUCCEEDED)

    assert terminal.progress.fraction == 1.0
    assert terminal.progress.phase == "done"
    assert manager.update_progress(ref, 0.5, "late sidecar") is False


def test_terminal_payload_and_metadata_snapshots_are_defensive_copies():
    manager = JobManager()
    metadata = {"nested": {"value": 1}}
    payload = {"nested": {"value": 2}}

    with ThreadPoolExecutor(max_workers=1) as executor:
        ref = manager.submit(
            executor,
            "rf",
            lambda: None,
            metadata=metadata,
            finalizer=lambda _ref, _result: payload,
        )
        _wait_for_state(manager, ref, JobState.SUCCEEDED)

    metadata["nested"]["value"] = 100
    payload["nested"]["value"] = 200
    first = manager.snapshot(ref)
    first.metadata["nested"]["value"] = 300
    first.terminal.payload["nested"]["value"] = 400
    second = manager.snapshot(ref)
    assert second.metadata["nested"]["value"] == 1
    assert second.terminal.payload["nested"]["value"] == 2


def test_acknowledgement_revision_must_match_and_forget_requires_ack():
    manager = JobManager()
    with ThreadPoolExecutor(max_workers=1) as executor:
        ref = manager.submit(executor, "rf", lambda: None)
        terminal = _wait_for_state(manager, ref, JobState.SUCCEEDED)

    revision = terminal.terminal.revision
    assert manager.forget(ref) is False
    assert manager.acknowledge(ref, revision + 1) is False
    assert manager.snapshot(ref).acknowledged is False
    assert manager.acknowledge(ref, revision) is True
    assert manager.forget(ref) is True
    assert manager.snapshot(ref) is None


def test_invalidate_discards_late_result_without_running_finalizer():
    manager = JobManager()
    task_started = threading.Event()
    release_task = threading.Event()
    finalizer_calls = 0

    def work():
        task_started.set()
        assert release_task.wait(timeout=2)
        return "stale result"

    def finalize(_ref, _result):
        nonlocal finalizer_calls
        finalizer_calls += 1
        return {"should_not": "appear"}

    with ThreadPoolExecutor(max_workers=1) as executor:
        ref = manager.submit(executor, "rf", work, finalizer=finalize)
        assert task_started.wait(timeout=2)
        assert manager.invalidate(ref) is True
        assert manager.snapshot(ref) is None
        assert manager.active_ref() is None
        release_task.set()

    assert finalizer_calls == 0
    assert manager.snapshot(ref) is None
