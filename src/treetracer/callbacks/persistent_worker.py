"""Parent-side client for the persistent compute subprocess.

The worker is a single subprocess spawned at app launch (see
``app.py:main``). We talk to it via length-prefixed pickle frames
over a **localhost TCP socket**, NOT the worker's stdin/stdout pipes.

Why a socket and not stdio: on Briefcase Windows GUI bundles the
launcher stub rebinds the worker subprocess's CRT ``fd 1`` during
``Py_Initialize`` (likely to a log file or ``NUL``), so any
``WriteFile`` against fd 1 from the worker never reaches the parent's
end of the Popen stdout pipe. The request direction (parent → worker
stdin / fd 0) still worked because the stub doesn't touch fd 0, but
the response direction was permanently broken in the bundle. Sockets
sidestep CRT stdio entirely — the connection is established via
``socket.connect`` / ``socket.accept`` independent of fd 0/1/2 — so
no stub-side rebinding can affect them.

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

Wire protocol (parent ↔ worker, framed identically on the socket):

    request:  [4-byte LE length N][N bytes pickle.dumps({
                  "job": "compute_rf" | "compute_consensus_tree" | ...,
                  "kwargs": {...},
              })]
    heartbeat: [frame containing {"type": "heartbeat", ...}]
    response:  [frame containing {"type": "result", "ok": bool, ...}]

The wire is synchronous: every request gets zero or more heartbeat frames and
then exactly one result frame. ``submit_job`` is serialised via ``_lock`` so
concurrent callbacks don't corrupt the stream.

Bootstrap (parent → worker rendezvous):

1. Parent opens a localhost TCP listener on an OS-picked port.
2. Parent passes the port to the worker via the
   ``TREETRACER_WORKER_PORT`` env var.
3. Parent ``Popen``s the worker with stdin/stdout=``DEVNULL`` (still
   keeps stderr=PIPE for crash diagnostics via the drainer).
4. Worker reads the env var, ``socket.connect``s back.
5. Parent ``accept``s the connection — listener is closed
   immediately after (one-shot rendezvous).

Cancellation:

The compute happens inside opaque native calls (``rapidtrees`` Rust,
scipy LAPACK/ARPACK) with no Python checkpoint to poll a flag, and the
worker isn't reading the socket while it computes — so the only way to
interrupt a running job is to kill the worker process.
``cancel_current_job`` does exactly that. The kill makes ``submit_job``'s
blocking ``recv`` return EOF; it then raises ``JobCancelled``. A fresh
worker is spawned immediately so the next compute stays warm.

Watchdog:

The worker emits a tiny heartbeat every two seconds from a daemon thread. The
parent treats a prolonged heartbeat gap as an unresponsive worker, kills it,
starts a replacement, and raises a normal compute exception. ``JobManager``
then delivers that failure through the same sticky terminal protocol as every
other result. A generous per-operation maximum runtime is a final safety net
for a compute function that remains able to heartbeat but never returns.
"""

from __future__ import annotations

import collections
import math
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

# Shared parent + worker file log (treetracer._worker_log) so both
# sides of the wire write checkpoint lines to the same place. The
# worker log already covers everything inside the subprocess; these
# parent-side entries are tagged ``[parent]`` so an interleaved
# timestamp-sorted view shows the IPC handshake clearly.
from .._worker_log import log as _wlog
from ..worker_protocol import (
    WorkerConnectionClosed,
    WorkerProtocolError,
    receive_message,
    send_message,
)


_worker_proc: subprocess.Popen | None = None
_worker_sock: socket.socket | None = None  # connected to current worker
_lock = threading.Lock()
_job_state_lock = threading.Lock()

# Cancellation state, protected by ``_job_state_lock``. ``_current_job`` is
# the name of the job whose
# response ``submit_job`` is currently blocked on (``None`` when the
# worker is idle); ``_cancelled`` is set by ``cancel_current_job`` so
# ``submit_job`` can tell a user Stop apart from a genuine crash.
_current_job: str | None = None
_cancelled: bool = False

# Stderr drainer for the live worker — see ``_StderrDrainer`` below.
# Refreshed by ``_spawn_worker`` on every worker (re)spawn so the
# tail reflects the currently-live process's diagnostic output, not a
# previous one's.
_stderr_drainer: "_StderrDrainer | None" = None


_DEFAULT_HEARTBEAT_INTERVAL_S = 2.0
_DEFAULT_HEARTBEAT_TIMEOUT_S = 120.0
_DEFAULT_MAX_RUNTIME_BY_JOB_S = {
    # RF, MDS, and Pseudo-ESS can scale quadratically and legitimately run for
    # hours on older machines. The ceilings are intentionally conservative.
    "compute_rf": 12 * 60 * 60,
    "compute_mds": 6 * 60 * 60,
    "compute_pseudo_ess": 6 * 60 * 60,
    # These operations work on a selection, row, or selected clade columns.
    "compute_consensus_tree": 2 * 60 * 60,
    "compute_rf_trace": 60 * 60,
    "compute_clade_frequencies": 60 * 60,
}
_DEFAULT_MAX_RUNTIME_S = 6 * 60 * 60


@dataclass(frozen=True, slots=True)
class WatchdogPolicy:
    """Resolved health limits for one worker request.

    ``None`` disables the corresponding limit. Environment overrides accept
    seconds; setting a value to ``0`` disables that limit deliberately.
    """

    heartbeat_timeout_s: float | None
    max_runtime_s: float | None


class WorkerUnresponsive(RuntimeError):
    """Base class for worker failures detected by the parent watchdog."""


class WorkerHeartbeatTimeout(WorkerUnresponsive):
    """No complete heartbeat or result frame arrived within the health limit."""


class WorkerRuntimeExceeded(WorkerUnresponsive):
    """A job exceeded its configured maximum runtime while still heartbeating."""


def watchdog_policy(
    job_name: str,
    *,
    max_runtime_s: float | None = None,
) -> WatchdogPolicy:
    """Resolve defaults and environment overrides for ``job_name``.

    Override precedence for maximum runtime is: explicit argument, per-job
    environment variable, global environment variable, built-in default.
    For example, RF can be overridden with
    ``TREETRACER_WORKER_MAX_RUNTIME_COMPUTE_RF_S``.
    """

    heartbeat_interval = _heartbeat_interval_s()
    heartbeat_timeout = _duration_from_env(
        "TREETRACER_WORKER_HEARTBEAT_TIMEOUT_S",
        _DEFAULT_HEARTBEAT_TIMEOUT_S,
    )
    if (
        heartbeat_timeout is not None
        and heartbeat_timeout < heartbeat_interval * 3
    ):
        adjusted = heartbeat_interval * 3
        _wlog(
            "[parent] heartbeat timeout is shorter than three worker "
            f"intervals; using {adjusted:g}s"
        )
        heartbeat_timeout = adjusted
    if max_runtime_s is None:
        per_job_name = (
            "TREETRACER_WORKER_MAX_RUNTIME_"
            f"{str(job_name).upper()}_S"
        )
        default_runtime = _DEFAULT_MAX_RUNTIME_BY_JOB_S.get(
            str(job_name),
            _DEFAULT_MAX_RUNTIME_S,
        )
        if per_job_name in os.environ:
            max_runtime = _duration_from_env(per_job_name, default_runtime)
        else:
            max_runtime = _duration_from_env(
                "TREETRACER_WORKER_MAX_RUNTIME_S",
                default_runtime,
            )
    else:
        max_runtime = _normalise_duration(
            max_runtime_s,
            name="max_runtime_s",
        )
    return WatchdogPolicy(heartbeat_timeout, max_runtime)


def _duration_from_env(name: str, default: float) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return _normalise_duration(raw, name=name)
    except (TypeError, ValueError):
        _wlog(f"[parent] ignoring invalid {name}={raw!r}")
        return float(default)


def _heartbeat_interval_s() -> float:
    name = "TREETRACER_WORKER_HEARTBEAT_INTERVAL_S"
    raw = os.environ.get(name)
    if raw is None:
        return _DEFAULT_HEARTBEAT_INTERVAL_S
    try:
        interval = float(raw)
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError
        return interval
    except (TypeError, ValueError):
        _wlog(f"[parent] ignoring invalid {name}={raw!r}")
        return _DEFAULT_HEARTBEAT_INTERVAL_S


def _normalise_duration(value: Any, *, name: str) -> float | None:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return None if seconds == 0 else seconds


class _StderrDrainer:
    """Continuously read the worker's stderr in a background thread so
    the worker never blocks on a full pipe buffer.

    Why this is load-bearing on Windows:

    * Anonymous pipes have a default buffer of ~64 KB on Windows.
    * Once the buffer fills, the *writer's* next ``write()`` blocks
      until a reader drains some bytes.
    * The persistent worker writes intermittently to stderr — numpy /
      scipy / rapidtrees deprecation and CPU-feature warnings on
      import, occasional Python warnings — and previously the parent
      only read stderr in error paths, never during normal operation.
    * Under x64-on-ARM emulation (Parallels on Apple Silicon) the
      library-import noise is much larger than on native x64; it
      reliably exceeds 64 KB before the first compute response is
      written. Worker blocks on stderr write → can't send response →
      parent blocks forever on stdout read → Task Manager shows two
      idle TreeTracer.exe processes (the classic "RF never finishes"
      symptom).

    The drainer reads in 8 KB chunks into a bounded ``deque`` so we
    keep diagnostic value (the last ~1 MB of stderr is preserved for
    surfacing on a crash) without unbounded memory growth.
    """

    _MAX_CHUNKS = 128  # ~1 MB tail kept
    _CHUNK_SIZE = 8192

    def __init__(self, stream):
        self._buffer: "collections.deque[bytes]" = collections.deque(
            maxlen=self._MAX_CHUNKS
        )
        threading.Thread(
            target=self._drain,
            args=(stream,),
            name="treetracer-worker-stderr-drain",
            daemon=True,
        ).start()

    def _drain(self, stream) -> None:
        try:
            while True:
                chunk = stream.read(self._CHUNK_SIZE)
                if not chunk:
                    return  # EOF — worker exited; thread done.
                self._buffer.append(chunk)
        except Exception:
            # Pipe closed mid-read or any other I/O error — give up
            # silently. The thread is daemon so it never blocks shutdown.
            return

    def tail(self) -> bytes:
        """Return the most recent ~1 MB of stderr bytes. Safe to call
        from any thread (``deque`` iteration is atomic for the GIL-
        level ops used here)."""
        return b"".join(self._buffer)


class JobCancelled(RuntimeError):
    """Raised by ``submit_job`` when the worker was killed via
    ``cancel_current_job()`` — lets the managed terminal adapters render a
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


_RENDEZVOUS_TIMEOUT_S = 60.0  # generous for slow Python init under emulation


def _spawn_worker() -> subprocess.Popen:
    """Popen the worker subprocess and establish the socket rendezvous.

    Returns the live ``Popen`` and, as a side effect, sets the module
    globals ``_worker_sock`` (the connected socket used by
    ``submit_job``) and ``_stderr_drainer`` (the background stderr
    reader — see the class docstring for why that's load-bearing).

    Worker errors are surfaced to the parent via the response protocol
    on the socket; stderr is for catastrophic-failure debug only,
    drained continuously so the worker never blocks on a full pipe
    buffer.
    """
    global _stderr_drainer, _worker_sock

    # ── Open a localhost listener for the worker to connect back to.
    # Port 0 lets the OS pick a free port. listen(1) — single worker.
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    _wlog(f"[parent] _spawn_worker: listening on 127.0.0.1:{port}")

    env = os.environ.copy()
    env["TREETRACER_WORKER_MODE"] = "persistent"
    env["TREETRACER_WORKER_PORT"] = str(port)
    env["TREETRACER_WORKER_HEARTBEAT_INTERVAL_S"] = str(
        _heartbeat_interval_s()
    )
    _wlog(f"[parent] _spawn_worker: argv={_worker_argv()!r}")

    # ── Spawn. Stdio is intentionally DEVNULL for stdin/stdout —
    # they're no longer used for IPC. stderr=PIPE stays so the
    # drainer can scoop up crash diagnostics.
    proc = subprocess.Popen(
        _worker_argv(),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    _wlog(f"[parent] _spawn_worker: Popen returned; worker pid={proc.pid}")

    # Drain immediately, including during rendezvous. A noisy import must not
    # fill stderr and prevent the worker from ever reaching socket.connect().
    _stderr_drainer = _StderrDrainer(proc.stderr)
    _wlog("[parent] _spawn_worker: stderr drainer started")

    # ── Wait for the worker to connect back. Generous timeout because
    # Python startup in the bundle (especially on emulated Windows) can
    # take several seconds. If the worker dies before connecting we
    # surface stderr from its drainer so the user knows why.
    listener.settimeout(0.25)
    deadline = time.monotonic() + _RENDEZVOUS_TIMEOUT_S
    try:
        while True:
            try:
                sock, addr = listener.accept()
                break
            except socket.timeout:
                if proc.poll() is not None:
                    raise RuntimeError(
                        "persistent worker exited before connecting "
                        f"(exit {proc.returncode})"
                    )
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "persistent worker did not connect within "
                        f"{_RENDEZVOUS_TIMEOUT_S:.0f}s"
                    )
    except Exception as exc:
        try:
            if proc.poll() is None:
                proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            pass
        stderr_tail = _worker_stderr_tail()
        raise RuntimeError(
            f"{exc}. stderr tail: "
            f"{stderr_tail.decode('utf-8', errors='replace')[-500:]}"
        ) from exc
    finally:
        listener.close()

    sock.settimeout(None)  # blocking I/O on the rest of the connection
    _worker_sock = sock
    _wlog(f"[parent] _spawn_worker: worker connected from {addr}")

    return proc


def _worker_stderr_tail() -> bytes:
    """Return the most recent stderr bytes from the live worker, or
    empty bytes if there isn't a drainer yet. Centralises the access
    pattern so error paths don't have to null-check the global."""
    drainer = _stderr_drainer
    if drainer is None:
        return b""
    return drainer.tail()


def start() -> None:
    """Spawn the persistent worker subprocess if it isn't already
    running. The application invokes this on a daemon startup thread; this
    function returns after the worker completes its socket rendezvous.
    """
    global _worker_proc
    with _lock:
        if _worker_proc is not None and _worker_proc.poll() is None:
            return  # already running
        _worker_proc = _spawn_worker()


def shutdown() -> None:
    """Close the IPC socket so the worker's recv loop returns cleanly.
    Safe to call even if the worker is already dead or never started.
    The parent's ``_kill_process_tree`` in ``app.py`` will also SIGKILL
    the worker on hard exits — this is just the graceful path.
    """
    global _worker_proc, _worker_sock
    with _lock:
        if _worker_sock is not None:
            try:
                _worker_sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                _worker_sock.close()
            except OSError:
                pass
            _worker_sock = None
        if _worker_proc is None:
            return
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
    finished on its own — exactly what a Stop button must avoid. A separate,
    tiny state lock closes the race between completion and cancellation without
    waiting for worker IPC.

    The kill makes ``submit_job``'s blocking read return EOF; that call
    then raises ``JobCancelled`` and respawns a fresh worker.
    """
    global _cancelled
    with _job_state_lock:
        proc = _worker_proc
        if proc is None or proc.poll() is not None:
            return False  # no live worker
        if _current_job is None:
            return False  # worker idle — nothing to interrupt
        # Set the flag before the kill so submit_job sees it on the EOF the
        # kill is about to cause.
        _cancelled = True
    try:
        proc.kill()
    except OSError:
        pass  # already exited between the checks above and here
    return True


def _begin_current_job(job_name: str) -> None:
    global _current_job, _cancelled
    with _job_state_lock:
        _cancelled = False
        _current_job = job_name


def _finish_current_job() -> bool:
    """Close the cancellation window and return whether Stop won it."""
    global _current_job
    with _job_state_lock:
        was_cancelled = _cancelled
        _current_job = None
        return was_cancelled


def _clear_current_job() -> None:
    global _current_job
    with _job_state_lock:
        _current_job = None


def _restart_worker_locked(reason: str) -> bool:
    """Retire the current process/socket and warm a replacement.

    ``submit_job`` and ``_ensure_running`` call this while holding ``_lock``.
    The old socket is closed before the process is killed so no late frame can
    be mistaken for a response from the replacement worker.
    """
    global _worker_proc, _worker_sock

    old_proc = _worker_proc
    old_sock = _worker_sock
    _worker_proc = None
    _worker_sock = None

    if old_sock is not None:
        try:
            old_sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            old_sock.close()
        except OSError:
            pass

    if old_proc is not None and old_proc.poll() is None:
        try:
            old_proc.kill()
        except OSError:
            pass
        try:
            old_proc.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            pass

    _wlog(f"[parent] restarting persistent worker: {reason}")
    try:
        _worker_proc = _spawn_worker()
    except Exception as exc:  # noqa: BLE001 — next submission retries startup
        _worker_proc = None
        _worker_sock = None
        _wlog(
            "[parent] worker restart failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return False
    _wlog(f"[parent] worker restart complete; pid={_worker_proc.pid}")
    return True


def _recovery_message(restarted: bool) -> str:
    if restarted:
        return "The worker was restarted and is ready for another computation."
    return (
        "The worker could not be restarted immediately; TreeTracer will retry "
        "startup on the next computation."
    )


def _ensure_running() -> subprocess.Popen:
    """Return the worker proc, restarting it if it died. Holds ``_lock``
    around the check + restart so concurrent submitters can't race."""
    global _worker_proc, _worker_sock
    if _worker_proc is None or _worker_proc.poll() is not None:
        # Worker dead (or never started). Restart. ``_worker_stderr_tail``
        # reads from the background drainer — a direct
        # ``_worker_proc.stderr.read()`` here would block forever
        # because the drainer thread owns the pipe.
        if _worker_proc is not None:
            stderr_tail = _worker_stderr_tail()
            sys.stderr.write(
                f"persistent worker died (exit {_worker_proc.returncode}); "
                f"restarting. stderr tail: "
                f"{stderr_tail.decode('utf-8', errors='replace')[-500:]}\n"
            )
        if not _restart_worker_locked("worker was not running"):
            raise RuntimeError("persistent worker could not be restarted")
    return _worker_proc


def _receive_worker_result(
    sock: socket.socket,
    job_name: str,
    policy: WatchdogPolicy,
    *,
    started_at: float,
) -> dict[str, Any]:
    """Consume heartbeat frames until the job's result frame arrives."""
    last_heartbeat_at = started_at
    last_sequence = 0

    while True:
        now = time.monotonic()
        heartbeat_remaining = (
            None
            if policy.heartbeat_timeout_s is None
            else policy.heartbeat_timeout_s - (now - last_heartbeat_at)
        )
        runtime_remaining = (
            None
            if policy.max_runtime_s is None
            else policy.max_runtime_s - (now - started_at)
        )

        if runtime_remaining is not None and runtime_remaining <= 0:
            raise WorkerRuntimeExceeded(
                f"{job_name} exceeded its {policy.max_runtime_s:.0f}s "
                "maximum runtime"
            )
        if heartbeat_remaining is not None and heartbeat_remaining <= 0:
            raise WorkerHeartbeatTimeout(
                f"persistent worker sent no heartbeat or result for "
                f"{policy.heartbeat_timeout_s:.0f}s during {job_name}"
            )

        waits = [
            value
            for value in (heartbeat_remaining, runtime_remaining)
            if value is not None
        ]
        sock.settimeout(min(waits) if waits else None)
        try:
            message = receive_message(sock)
        except socket.timeout as exc:
            now = time.monotonic()
            if (
                policy.max_runtime_s is not None
                and now - started_at >= policy.max_runtime_s
            ):
                raise WorkerRuntimeExceeded(
                    f"{job_name} exceeded its {policy.max_runtime_s:.0f}s "
                    "maximum runtime"
                ) from exc
            raise WorkerHeartbeatTimeout(
                f"persistent worker sent no heartbeat or result for "
                f"{policy.heartbeat_timeout_s:.0f}s during {job_name}"
            ) from exc

        message_type = message.get("type")
        if message_type == "heartbeat":
            if message.get("job") != job_name:
                raise WorkerProtocolError(
                    "heartbeat job mismatch: "
                    f"expected {job_name!r}, got {message.get('job')!r}"
                )
            try:
                sequence = int(message["sequence"])
            except (KeyError, TypeError, ValueError) as exc:
                raise WorkerProtocolError(
                    "heartbeat is missing a valid sequence"
                ) from exc
            if sequence < 1:
                raise WorkerProtocolError(
                    f"heartbeat has invalid sequence {sequence}"
                )
            if sequence <= last_sequence:
                raise WorkerProtocolError(
                    "heartbeat sequence did not advance: "
                    f"previous={last_sequence}, current={sequence}"
                )
            last_sequence = sequence
            last_heartbeat_at = time.monotonic()
            if sequence == 1 or sequence % 30 == 0:
                _wlog(
                    "[parent] worker heartbeat: "
                    f"job={job_name!r}, sequence={sequence}, "
                    f"elapsed={message.get('elapsed_s', '?')}s"
                )
            continue

        # Accept an untyped result for one-version rolling compatibility with
        # a worker started just before an application update.
        if message_type in (None, "result"):
            return message
        raise WorkerProtocolError(
            f"unexpected worker message type {message_type!r}"
        )


def submit_job(
    job_name: str,
    *,
    max_runtime_s: float | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Send one job and wait for heartbeats followed by its result.

    Socket waits release the GIL, so Dash and the desktop event loop remain
    responsive. Missing heartbeats, a maximum-runtime breach, a corrupt frame,
    or a dead connection retires the worker and warms a replacement before the
    exception reaches ``JobManager``.

    ``max_runtime_s`` overrides the environment/default ceiling for this call;
    pass ``0`` to disable only the hard runtime ceiling. Heartbeat monitoring
    remains independently configurable through
    ``TREETRACER_WORKER_HEARTBEAT_TIMEOUT_S``.
    """
    global _worker_sock

    policy = watchdog_policy(job_name, max_runtime_s=max_runtime_s)
    _wlog(
        f"[parent] submit_job called: job={job_name!r}, "
        f"kwarg keys={sorted(kwargs.keys())}, policy={policy}"
    )
    with _lock:
        _wlog("[parent] submit_job: _lock acquired")
        proc = _ensure_running()
        assert _worker_sock is not None
        sock = _worker_sock
        _wlog(
            f"[parent] submit_job: worker pid={proc.pid}, "
            f"alive={proc.poll() is None}"
        )

        _begin_current_job(job_name)
        started_at = time.monotonic()
        try:
            _wlog("[parent] submit_job: sending request frame")
            send_message(sock, {"job": job_name, "kwargs": kwargs})
            response = _receive_worker_result(
                sock,
                job_name,
                policy,
                started_at=started_at,
            )
            if not isinstance(response.get("ok"), bool):
                raise WorkerProtocolError(
                    "result is missing a boolean ok field"
                )
            if response["ok"] and "result" not in response:
                raise WorkerProtocolError(
                    "successful result has no result payload"
                )
        except (
            OSError,
            WorkerConnectionClosed,
            WorkerProtocolError,
            WorkerUnresponsive,
        ) as exc:
            was_cancelled = _finish_current_job()
            stderr_tail = _worker_stderr_tail().decode(
                "utf-8",
                errors="replace",
            )[-500:]
            reason = (
                f"user cancelled {job_name}"
                if was_cancelled
                else f"{type(exc).__name__} during {job_name}: {exc}"
            )
            restarted = _restart_worker_locked(reason)
            recovery = _recovery_message(restarted)
            if was_cancelled:
                raise JobCancelled(
                    f"{job_name} cancelled by user. {recovery}"
                ) from exc
            if isinstance(exc, WorkerUnresponsive):
                raise type(exc)(f"{exc}. {recovery}") from exc
            if isinstance(exc, WorkerProtocolError):
                raise WorkerProtocolError(
                    f"persistent worker protocol failed during {job_name}: "
                    f"{exc}. {recovery}"
                ) from exc
            detail = f" stderr tail: {stderr_tail}" if stderr_tail else ""
            raise RuntimeError(
                f"persistent worker connection failed during {job_name}: "
                f"{type(exc).__name__}: {exc}.{detail} {recovery}"
            ) from exc
        finally:
            _clear_current_job()
            try:
                sock.settimeout(None)
            except OSError:
                pass

    _wlog(
        "[parent] submit_job: result received; "
        f"ok={response.get('ok', '?')}"
    )
    if not response["ok"]:
        raise RuntimeError(
            str(response.get("error", "<unknown worker error>"))
            + "\n"
            + str(response.get("traceback", ""))
        )
    _wlog("[parent] submit_job: returning result to caller")
    return response["result"]
