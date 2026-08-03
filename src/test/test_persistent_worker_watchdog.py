"""Watchdog behavior without launching the scientific worker process."""

from __future__ import annotations

import socket
import threading
import time

import pytest

from treetracer.callbacks import persistent_worker
from treetracer.worker_protocol import (
    WorkerProtocolError,
    receive_message,
    send_message,
)


def test_result_wait_accepts_heartbeats_until_success():
    parent, worker = socket.socketpair()

    def respond():
        send_message(worker, {
            "type": "heartbeat",
            "job": "compute_rf",
            "sequence": 1,
            "elapsed_s": 0.0,
        })
        send_message(worker, {
            "type": "heartbeat",
            "job": "compute_rf",
            "sequence": 2,
            "elapsed_s": 0.01,
        })
        send_message(worker, {
            "type": "result",
            "ok": True,
            "result": {"done": True},
        })

    thread = threading.Thread(target=respond)
    thread.start()
    try:
        result = persistent_worker._receive_worker_result(
            parent,
            "compute_rf",
            persistent_worker.WatchdogPolicy(0.2, 1.0),
            started_at=time.monotonic(),
        )
        assert result["result"] == {"done": True}
    finally:
        thread.join(timeout=1)
        parent.close()
        worker.close()


def test_missing_heartbeat_has_a_bounded_wait():
    parent, worker = socket.socketpair()
    try:
        with pytest.raises(
            persistent_worker.WorkerHeartbeatTimeout,
            match="no heartbeat or result",
        ):
            persistent_worker._receive_worker_result(
                parent,
                "compute_mds",
                persistent_worker.WatchdogPolicy(0.03, 1.0),
                started_at=time.monotonic(),
            )
    finally:
        parent.close()
        worker.close()


def test_maximum_runtime_wins_even_while_heartbeats_continue():
    parent, worker = socket.socketpair()
    stop = threading.Event()

    def heartbeat_forever():
        sequence = 0
        while not stop.wait(0.005):
            sequence += 1
            try:
                send_message(worker, {
                    "type": "heartbeat",
                    "job": "compute_pseudo_ess",
                    "sequence": sequence,
                    "elapsed_s": sequence * 0.005,
                })
            except OSError:
                return

    thread = threading.Thread(target=heartbeat_forever)
    thread.start()
    try:
        with pytest.raises(
            persistent_worker.WorkerRuntimeExceeded,
            match="maximum runtime",
        ):
            persistent_worker._receive_worker_result(
                parent,
                "compute_pseudo_ess",
                persistent_worker.WatchdogPolicy(0.05, 0.03),
                started_at=time.monotonic(),
            )
    finally:
        stop.set()
        parent.close()
        worker.close()
        thread.join(timeout=1)


def test_heartbeat_must_match_job_and_advance_sequence():
    parent, worker = socket.socketpair()
    try:
        send_message(worker, {
            "type": "heartbeat",
            "job": "compute_rf_trace",
            "sequence": 1,
        })
        with pytest.raises(WorkerProtocolError, match="job mismatch"):
            persistent_worker._receive_worker_result(
                parent,
                "compute_rf",
                persistent_worker.WatchdogPolicy(0.2, 1.0),
                started_at=time.monotonic(),
            )
    finally:
        parent.close()
        worker.close()


def test_watchdog_policy_supports_safe_environment_overrides(monkeypatch):
    monkeypatch.setenv("TREETRACER_WORKER_HEARTBEAT_INTERVAL_S", "0.01")
    monkeypatch.setenv("TREETRACER_WORKER_HEARTBEAT_TIMEOUT_S", "0.02")
    monkeypatch.setenv("TREETRACER_WORKER_MAX_RUNTIME_S", "100")
    monkeypatch.setenv(
        "TREETRACER_WORKER_MAX_RUNTIME_COMPUTE_RF_S",
        "50",
    )

    policy = persistent_worker.watchdog_policy("compute_rf")
    # Timeout is raised to three heartbeat periods to tolerate jitter.
    assert policy.heartbeat_timeout_s == pytest.approx(0.03)
    assert policy.max_runtime_s == 50

    disabled = persistent_worker.watchdog_policy(
        "compute_rf",
        max_runtime_s=0,
    )
    assert disabled.max_runtime_s is None


class _FakeProcess:
    def __init__(self, pid):
        self.pid = pid
        self.returncode = None
        self.killed = False

    def poll(self):
        return -9 if self.killed else None

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.killed = True
        self.returncode = -9
        return self.returncode


def test_submit_timeout_restarts_worker_before_error_reaches_manager(
    monkeypatch,
):
    parent, worker = socket.socketpair()
    replacement_parent, replacement_worker = socket.socketpair()
    old_proc = _FakeProcess(101)
    new_proc = _FakeProcess(202)

    monkeypatch.setattr(persistent_worker, "_worker_proc", old_proc)
    monkeypatch.setattr(persistent_worker, "_worker_sock", parent)
    monkeypatch.setattr(persistent_worker, "_current_job", None)
    monkeypatch.setattr(persistent_worker, "_cancelled", False)
    monkeypatch.setenv("TREETRACER_WORKER_HEARTBEAT_INTERVAL_S", "0.01")
    monkeypatch.setenv("TREETRACER_WORKER_HEARTBEAT_TIMEOUT_S", "0.03")

    def fake_spawn():
        persistent_worker._worker_sock = replacement_parent
        return new_proc

    monkeypatch.setattr(persistent_worker, "_spawn_worker", fake_spawn)

    # Consume the request but deliberately send no heartbeat or result.
    received = {}

    def consume_request():
        received.update(receive_message(worker))

    thread = threading.Thread(target=consume_request)
    thread.start()
    try:
        with pytest.raises(
            persistent_worker.WorkerHeartbeatTimeout,
            match="worker was restarted",
        ):
            persistent_worker.submit_job(
                "compute_rf_trace",
                max_runtime_s=1,
                matrix_path="ignored.npy",
            )
        thread.join(timeout=1)
        assert received["job"] == "compute_rf_trace"
        assert old_proc.killed is True
        assert persistent_worker._worker_proc is new_proc
        assert persistent_worker._worker_sock is replacement_parent
    finally:
        parent.close()
        worker.close()
        replacement_parent.close()
        replacement_worker.close()
