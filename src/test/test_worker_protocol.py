"""Deterministic tests for framed worker messages and heartbeats."""

from __future__ import annotations

import socket

import pytest

from treetracer.worker_protocol import (
    HeartbeatEmitter,
    WorkerConnectionClosed,
    WorkerProtocolError,
    receive_message,
    send_message,
)


def test_framed_mapping_round_trip():
    parent, worker = socket.socketpair()
    try:
        send_message(parent, {"job": "compute_rf", "kwargs": {"n": 3}})
        assert receive_message(worker) == {
            "job": "compute_rf",
            "kwargs": {"n": 3},
        }
    finally:
        parent.close()
        worker.close()


def test_clean_eof_and_truncated_frame_are_distinct():
    parent, worker = socket.socketpair()
    parent.close()
    with pytest.raises(WorkerConnectionClosed):
        receive_message(worker)
    worker.close()

    parent, worker = socket.socketpair()
    try:
        parent.sendall(b"\x08\x00\x00\x00abc")
        parent.close()
        with pytest.raises(WorkerProtocolError, match="during frame body"):
            receive_message(worker)
    finally:
        worker.close()


def test_heartbeat_frames_stop_before_the_result_frame():
    parent, worker = socket.socketpair()
    worker.settimeout(1.0)
    emitter = HeartbeatEmitter(
        parent,
        "compute_mds",
        interval_s=0.01,
    )
    try:
        emitter.start()
        first = receive_message(worker)
        second = receive_message(worker)
        assert first["type"] == "heartbeat"
        assert first["phase"] == "accepted"
        assert second["type"] == "heartbeat"
        assert second["sequence"] > first["sequence"]

        emitter.stop()
        send_message(
            parent,
            {"type": "result", "ok": True, "result": [1, 2, 3]},
            lock=emitter.send_lock,
        )
        result = receive_message(worker)
        assert result == {
            "type": "result",
            "ok": True,
            "result": [1, 2, 3],
        }
    finally:
        emitter.stop()
        parent.close()
        worker.close()
