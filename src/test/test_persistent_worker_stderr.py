"""Pin the Windows-pipe-deadlock fix in persistent_worker.

Without ``_StderrDrainer``, the persistent worker would deadlock on a
full ~64 KB Windows stderr pipe buffer once import-time warnings from
numpy / rapidtrees / etc. exceeded that budget under x64-on-ARM
emulation. The drainer reads stderr continuously in a background
thread so the worker never blocks on a stderr write.

These tests pump bytes through a real OS pipe to verify:

* the drainer keeps draining well past 64 KB without the writer
  blocking,
* ``tail()`` returns the most-recent bytes (bounded ring, not
  unbounded growth),
* the drainer exits cleanly on EOF.

No real worker subprocess is involved — we use ``os.pipe()`` directly
to keep the test fast and deterministic.
"""

from __future__ import annotations

import io
import os
import threading
import time

import pytest

from treetracer.callbacks.persistent_worker import _StderrDrainer


def _make_pipe_streams():
    """Return ``(reader_stream, writer_fd)`` backed by an OS pipe.

    The reader is wrapped in a Python file object so it has the same
    ``.read(n)`` interface the drainer expects from ``subprocess.PIPE``.
    The writer side stays as a raw fd so the test can call ``os.write``
    on it without any Python-level buffering getting in the way.
    """
    r_fd, w_fd = os.pipe()
    reader = os.fdopen(r_fd, "rb", buffering=0, closefd=True)
    return reader, w_fd


def test_drainer_unblocks_writer_past_64KB():
    """Without a drainer, writing more than the pipe buffer (~64 KB
    on Windows, often ~64 KB on Linux too) would block. With one,
    the write completes immediately regardless of total volume."""
    reader, w_fd = _make_pipe_streams()
    drainer = _StderrDrainer(reader)

    # 256 KB — well past any reasonable pipe buffer. If the drainer
    # weren't running, the os.write below would block on Linux too
    # (linux pipes default to 64 KB).
    payload = b"x" * (256 * 1024)
    start = time.monotonic()
    n_written = os.write(w_fd, payload)
    elapsed = time.monotonic() - start

    assert n_written == len(payload)
    # If the drainer were absent or stuck, this write would block for
    # seconds. A working drainer makes it return in milliseconds.
    assert elapsed < 1.0, f"write took {elapsed:.2f}s — drainer not draining"

    # Let the drainer catch up before closing.
    os.close(w_fd)
    # Wait briefly for the drainer thread to consume EOF.
    for _ in range(50):
        if reader.closed or len(drainer.tail()) >= len(payload):
            break
        time.sleep(0.02)
    # Drainer captured the payload (modulo the ring cap, which our
    # 256 KB easily fits inside the ~1 MB ring).
    assert len(drainer.tail()) == len(payload)


def test_drainer_tail_is_bounded():
    """``tail()`` keeps only the most recent bytes — the deque is
    capped so a long-running worker can't OOM the parent via stderr
    spam."""
    reader, w_fd = _make_pipe_streams()
    drainer = _StderrDrainer(reader)

    # Cap is _MAX_CHUNKS=128 * _CHUNK_SIZE=8192 = 1 MiB tail. Write
    # 4 MiB and confirm tail is bounded.
    block = b"y" * 8192
    for _ in range(4 * 128):  # 4 MiB total
        os.write(w_fd, block)
    os.close(w_fd)

    # Wait for drainer to finish reading.
    for _ in range(200):
        if len(drainer.tail()) > 0 and reader.closed:
            break
        time.sleep(0.02)

    tail = drainer.tail()
    cap_bytes = _StderrDrainer._MAX_CHUNKS * _StderrDrainer._CHUNK_SIZE
    assert len(tail) <= cap_bytes, (
        f"tail is {len(tail)} bytes, exceeds cap {cap_bytes}"
    )


def test_drainer_exits_on_eof():
    """The drainer thread is a daemon, but it should also exit
    naturally when stderr EOFs — otherwise long-running parents leak
    threads on every worker restart."""
    reader, w_fd = _make_pipe_streams()
    drainer = _StderrDrainer(reader)
    os.write(w_fd, b"hello\n")
    os.close(w_fd)

    # Find the drainer thread by name and wait briefly for it to exit.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        live = [
            t for t in threading.enumerate()
            if t.name == "treetracer-worker-stderr-drain" and t.is_alive()
        ]
        if not live:
            break
        time.sleep(0.05)

    live = [
        t for t in threading.enumerate()
        if t.name == "treetracer-worker-stderr-drain" and t.is_alive()
    ]
    assert not live, "drainer thread didn't exit after EOF"
    assert b"hello\n" in drainer.tail()


def test_drainer_tail_safe_to_call_before_any_writes():
    """Edge case: parent crashes before worker writes anything. ``tail()``
    must return ``b""`` rather than raising."""
    reader, w_fd = _make_pipe_streams()
    drainer = _StderrDrainer(reader)
    assert drainer.tail() == b""
    os.close(w_fd)
