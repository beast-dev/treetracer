"""Small framed-message protocol shared by the GUI and compute worker.

The persistent worker uses one localhost TCP socket for requests, heartbeat
messages, and results.  Every message is a length-prefixed pickle frame::

    [4-byte little-endian payload length][pickled mapping]

Only TreeTracer processes connected through the one-shot localhost rendezvous
use this protocol.  The size limit protects both sides from allocating an
unbounded buffer if a connection is corrupt.
"""

from __future__ import annotations

import pickle
import socket
import struct
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any


MAX_FRAME_BYTES = 256 * 1024 * 1024


class WorkerConnectionClosed(EOFError):
    """The peer closed the socket before another complete frame arrived."""


class WorkerProtocolError(RuntimeError):
    """A worker frame is truncated, malformed, or violates the protocol."""


def encode_message(message: Mapping[str, Any]) -> bytes:
    """Serialize one mapping with its four-byte length prefix."""
    payload = pickle.dumps(dict(message), protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > MAX_FRAME_BYTES:
        raise WorkerProtocolError(
            f"worker frame is {len(payload):,} bytes; "
            f"limit is {MAX_FRAME_BYTES:,} bytes"
        )
    return struct.pack("<I", len(payload)) + payload


def send_message(
    sock: socket.socket,
    message: Mapping[str, Any],
    *,
    lock: threading.Lock | None = None,
) -> None:
    """Send one complete frame, optionally serializing concurrent writers."""
    frame = encode_message(message)
    if lock is None:
        sock.sendall(frame)
        return
    with lock:
        sock.sendall(frame)


def receive_message(sock: socket.socket) -> dict[str, Any]:
    """Receive and validate one complete mapping frame."""
    size_bytes = _recv_exactly(sock, 4, field="header")
    (size,) = struct.unpack("<I", size_bytes)
    if size > MAX_FRAME_BYTES:
        raise WorkerProtocolError(
            f"worker announced a {size:,}-byte frame; "
            f"limit is {MAX_FRAME_BYTES:,} bytes"
        )
    payload = _recv_exactly(sock, size, field="body")
    try:
        message = pickle.loads(payload)
    except Exception as exc:
        raise WorkerProtocolError(
            f"could not decode worker frame: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(message, Mapping):
        raise WorkerProtocolError(
            f"worker frame must contain a mapping, got {type(message).__name__}"
        )
    return dict(message)


def _recv_exactly(sock: socket.socket, size: int, *, field: str) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            if not data:
                raise WorkerConnectionClosed(
                    f"worker connection closed before frame {field}"
                )
            raise WorkerProtocolError(
                f"worker connection closed during frame {field} "
                f"({len(data):,}/{size:,} bytes)"
            )
        data.extend(chunk)
    return bytes(data)


class HeartbeatEmitter:
    """Send serialized heartbeat frames while one worker job is running.

    The result writer uses :attr:`send_lock` too, so a heartbeat can never be
    interleaved with a result frame.  ``Event.wait`` makes shutdown immediate
    instead of sleeping for the rest of the heartbeat period.
    """

    def __init__(
        self,
        sock: socket.socket,
        job_name: str,
        *,
        interval_s: float,
        log: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("heartbeat interval must be positive")
        self.send_lock = threading.Lock()
        self._sock = sock
        self._job_name = str(job_name)
        self._interval_s = float(interval_s)
        self._log = log or (lambda _message: None)
        self._clock = clock
        self._started_at = self._clock()
        self._sequence = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Publish acceptance immediately, then start periodic heartbeats."""
        if self._thread is not None:
            raise RuntimeError("heartbeat emitter already started")
        self._emit("accepted")
        self._thread = threading.Thread(
            target=self._run,
            name="treetracer-worker-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop before the caller sends the terminal result frame."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(1.0, self._interval_s + 0.5))

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            try:
                self._emit("running")
            except OSError as exc:
                self._log(
                    "heartbeat send failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                return

    def _emit(self, phase: str) -> None:
        self._sequence += 1
        send_message(
            self._sock,
            {
                "type": "heartbeat",
                "job": self._job_name,
                "sequence": self._sequence,
                "phase": phase,
                "elapsed_s": max(0.0, self._clock() - self._started_at),
            },
            lock=self.send_lock,
        )
