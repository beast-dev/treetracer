"""Parent-side client for the persistent compute subprocess.

The worker is a single subprocess spawned at app launch (see
``app.py:main``). It sits in a read loop blocked on
``sys.stdin.buffer.read(4)``. We talk to it via length-prefixed pickle
frames over its stdin/stdout pipes.

Why this exists (instead of ``ProcessPoolExecutor`` or per-compute
subprocesses):

* ``multiprocessing.spawn`` doesn't work in Briefcase macOS bundles
  (no standalone Python binary; ``sys.executable`` is the launcher
  stub which doesn't accept ``-c``).
* Spawning a fresh subprocess per compute pays ~1.5 s of Python +
  imports overhead every time. Persistent worker pays it once at app
  launch, where the user is already waiting for the GUI.
* Threads can't fix it — rapidtrees holds the GIL during its
  iterator-consumption phase, which triggers the macOS beach-ball
  watchdog on big datasets.

Wire protocol (parent ↔ worker):

    request:  [4-byte LE length N][N bytes pickle.dumps({
                  "job": "compute_rf" | "compute_mcc",
                  "kwargs": {...},
              })]
    response: [4-byte LE length M][M bytes pickle.dumps({
                  "ok": True, "result": ...
              }) | pickle.dumps({
                  "ok": False, "error": str, "traceback": str
              })]

The wire is synchronous: every request gets exactly one response,
in order. ``submit_job`` is serialised via ``_lock`` so concurrent
callbacks don't corrupt the stream.
"""

from __future__ import annotations

import os
import pickle
import struct
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict


_worker_proc: subprocess.Popen | None = None
_lock = threading.Lock()


def _worker_argv() -> list[str]:
    """Pick the right argv based on how this Python is being run.

    Bundle: ``sys.executable`` is ``TreeTracer.app/Contents/MacOS/TreeTracer``
    (or ``TreeTracer.exe`` on Windows) — a launcher stub that's
    hard-wired to invoke ``python -m treetracer``. No extra args
    needed; the stub already does the right thing.

    Dev / ``uv tool install`` / ``uv run``: ``sys.executable`` is a
    regular Python interpreter, so we need ``-m treetracer`` to load
    the package as ``__main__``.
    """
    name = Path(sys.executable).name
    if name in ("TreeTracer", "TreeTracer.exe"):
        return [sys.executable]
    return [sys.executable, "-m", "treetracer"]


def start() -> None:
    """Spawn the persistent worker subprocess if it isn't already
    running. Returns immediately — the subprocess boots in the
    background. The first ``submit_job`` call after this returns will
    block on the worker reading from stdin, which is automatic — OS
    pipe buffering covers any race between Popen returning and the
    worker entering its read loop.
    """
    global _worker_proc
    with _lock:
        if _worker_proc is not None and _worker_proc.poll() is None:
            return  # already running
        env = os.environ.copy()
        env["TREETRACER_WORKER_MODE"] = "persistent"
        _worker_proc = subprocess.Popen(
            _worker_argv(),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Worker errors are surfaced to the parent via the response
            # protocol; stderr is for catastrophic-failure debug only,
            # captured so it doesn't bleed onto the user's terminal in
            # ``uv run`` mode.
            stderr=subprocess.PIPE,
            bufsize=0,
        )


def shutdown() -> None:
    """Close the worker's stdin so its read loop returns cleanly. Safe
    to call even if the worker is already dead or never started. The
    parent's ``_kill_process_tree`` in ``app.py`` will also SIGKILL the
    worker on hard exits — this is just the graceful path.
    """
    global _worker_proc
    with _lock:
        if _worker_proc is None:
            return
        try:
            if _worker_proc.stdin and not _worker_proc.stdin.closed:
                _worker_proc.stdin.close()
        except OSError:
            pass
        # Give the worker a moment to exit on its own; if it doesn't,
        # the process-tree-kill on app shutdown handles it.
        try:
            _worker_proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass
        _worker_proc = None


def _ensure_running() -> subprocess.Popen:
    """Return the worker proc, restarting it if it died. Holds ``_lock``
    around the check + restart so concurrent submitters can't race."""
    global _worker_proc
    if _worker_proc is None or _worker_proc.poll() is not None:
        # Worker dead (or never started). Restart.
        if _worker_proc is not None:
            stderr_tail = b""
            try:
                if _worker_proc.stderr:
                    stderr_tail = _worker_proc.stderr.read()
            except OSError:
                pass
            sys.stderr.write(
                f"persistent worker died (exit {_worker_proc.returncode}); "
                f"restarting. stderr tail: "
                f"{stderr_tail.decode('utf-8', errors='replace')[-500:]}\n"
            )
        env = os.environ.copy()
        env["TREETRACER_WORKER_MODE"] = "persistent"
        _worker_proc = subprocess.Popen(
            _worker_argv(),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    return _worker_proc


def submit_job(job_name: str, **kwargs: Any) -> Dict[str, Any]:
    """Send a job to the persistent worker and block on its response.

    The wait happens inside a ``subprocess`` pipe read, which releases
    the GIL — so the parent's other threads (Dash callbacks, the
    pywebview event loop on its native thread, etc.) stay responsive.

    Args:
        job_name: ``"compute_rf"`` or ``"compute_mcc"`` — must match a
            branch in ``__init__.py:_run_persistent_worker``.
        **kwargs: forwarded to the worker function.

    Returns the worker function's return value (unpickled). Raises
    ``RuntimeError`` if the worker reported an error or crashed.
    """
    with _lock:
        proc = _ensure_running()
        assert proc.stdin is not None and proc.stdout is not None

        request_bytes = pickle.dumps({"job": job_name, "kwargs": kwargs})
        proc.stdin.write(struct.pack("<I", len(request_bytes)))
        proc.stdin.write(request_bytes)
        proc.stdin.flush()

        size_bytes = _read_exactly(proc.stdout, 4)
        if len(size_bytes) != 4:
            # Worker died mid-job — surface stderr so we have something
            # to debug with.
            stderr_tail = b""
            try:
                if proc.stderr:
                    stderr_tail = proc.stderr.read()
            except OSError:
                pass
            raise RuntimeError(
                f"persistent worker died during {job_name!r}: "
                f"{stderr_tail.decode('utf-8', errors='replace')[-500:]}"
            )
        (size,) = struct.unpack("<I", size_bytes)
        response_bytes = _read_exactly(proc.stdout, size)

    response = pickle.loads(response_bytes)
    if not response.get("ok"):
        raise RuntimeError(
            response.get("error", "<unknown worker error>")
            + "\n" + response.get("traceback", "")
        )
    return response["result"]


def _read_exactly(stream, n: int) -> bytes:
    """``stream.read(n)`` can return short reads when the worker writes
    in chunks — keep reading until we have N bytes or hit EOF."""
    data = b""
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            return data
        data += chunk
    return data
