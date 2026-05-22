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

Cancellation:

The compute happens inside opaque native calls (``rapidtrees`` Rust,
scipy LAPACK/ARPACK) with no Python checkpoint to poll a flag, and the
worker isn't reading stdin while it computes — so the only way to
interrupt a running job is to kill the worker process.
``cancel_current_job`` does exactly that. The kill makes ``submit_job``'s
blocking read return EOF; it then raises ``JobCancelled``. A fresh
worker is spawned immediately so the next compute stays warm.
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

# Cancellation state. ``_current_job`` is the name of the job whose
# response ``submit_job`` is currently blocked on (``None`` when the
# worker is idle); ``_cancelled`` is set by ``cancel_current_job`` so
# ``submit_job`` can tell a user Stop apart from a genuine crash.
_current_job: str | None = None
_cancelled: bool = False


class JobCancelled(RuntimeError):
    """Raised by ``submit_job`` when the worker was killed via
    ``cancel_current_job()`` — lets the polling callbacks render a
    user-requested Stop as a neutral "cancelled" state rather than a
    red error."""


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


def _spawn_worker() -> subprocess.Popen:
    """Popen the worker subprocess in its persistent IPC configuration.

    Worker errors are surfaced to the parent via the response protocol;
    stderr is for catastrophic-failure debug only, captured so it
    doesn't bleed onto the user's terminal in ``uv run`` mode.
    """
    env = os.environ.copy()
    env["TREETRACER_WORKER_MODE"] = "persistent"
    return subprocess.Popen(
        _worker_argv(),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )


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
        _worker_proc = _spawn_worker()


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


def cancel_current_job() -> bool:
    """Interrupt the compute currently running in the worker by killing
    the worker subprocess. Returns ``True`` if a running job was killed,
    ``False`` if the worker was idle or not running.

    Safe to call from any thread. It deliberately does **not** acquire
    ``_lock``: the thread that called ``submit_job`` holds that lock for
    the whole job, so acquiring it here would block until the job
    finished on its own — exactly what a Stop button must avoid. We only
    read the ``_worker_proc`` reference (atomic) and signal it, both
    thread-safe.

    The kill makes ``submit_job``'s blocking read return EOF; that call
    then raises ``JobCancelled`` and respawns a fresh worker.
    """
    global _cancelled
    proc = _worker_proc  # atomic snapshot of the module global
    if proc is None or proc.poll() is not None:
        return False  # no live worker
    if _current_job is None:
        return False  # worker idle — nothing to interrupt
    # Order matters: set the flag before the kill so submit_job sees it
    # on the EOF the kill is about to cause.
    _cancelled = True
    try:
        proc.kill()
    except OSError:
        pass  # already exited between the checks above and here
    return True


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
        _worker_proc = _spawn_worker()
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
    ``JobCancelled`` if the user stopped the job via
    ``cancel_current_job``, or ``RuntimeError`` if the worker reported
    an error or crashed.
    """
    global _worker_proc, _current_job, _cancelled
    with _lock:
        _cancelled = False
        proc = _ensure_running()
        assert proc.stdin is not None and proc.stdout is not None

        request_bytes = pickle.dumps({"job": job_name, "kwargs": kwargs})
        # ``_current_job`` is the cancellation window: while it's set,
        # cancel_current_job() may kill this worker.
        _current_job = job_name
        try:
            proc.stdin.write(struct.pack("<I", len(request_bytes)))
            proc.stdin.write(request_bytes)
            proc.stdin.flush()
            size_bytes = _read_exactly(proc.stdout, 4)
        except OSError:
            # Pipe broke mid-request — almost always because
            # cancel_current_job() just killed the worker.
            size_bytes = b""
        finally:
            _current_job = None

        if len(size_bytes) != 4:
            # Worker died mid-job.
            if _cancelled:
                # User Stop. Respawn now, while we still hold _lock, so
                # the next compute doesn't pay interpreter-boot latency.
                try:
                    _worker_proc = _spawn_worker()
                except Exception:  # noqa: BLE001 — _ensure_running retries
                    _worker_proc = None
                raise JobCancelled(f"{job_name} cancelled by user")
            # Genuine crash — surface stderr so we have something to
            # debug with.
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
