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
                  "job": "compute_rf" | "compute_consensus_tree",
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
"""

from __future__ import annotations

import collections
import os
import pickle
import socket
import struct
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict

# Shared parent + worker file log (treetracer._worker_log) so both
# sides of the wire write checkpoint lines to the same place. The
# worker log already covers everything inside the subprocess; these
# parent-side entries are tagged ``[parent]`` so an interleaved
# timestamp-sorted view shows the IPC handshake clearly.
from .._worker_log import log as _wlog


_worker_proc: subprocess.Popen | None = None
_worker_sock: socket.socket | None = None  # connected to current worker
_lock = threading.Lock()

# Cancellation state. ``_current_job`` is the name of the job whose
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

    # ── Wait for the worker to connect back. Generous timeout because
    # Python startup in the bundle (especially on emulated Windows) can
    # take several seconds. If the worker dies before connecting we
    # surface stderr from its drainer so the user knows why.
    listener.settimeout(_RENDEZVOUS_TIMEOUT_S)
    try:
        sock, addr = listener.accept()
    except socket.timeout:
        # Kill the worker, salvage whatever it printed to stderr.
        try:
            proc.kill()
        except OSError:
            pass
        # Drainer might not even exist yet — read stderr directly,
        # but bounded so we don't block forever.
        stderr_tail = b""
        try:
            if proc.stderr is not None:
                stderr_tail = proc.stderr.read()
        except OSError:
            pass
        raise RuntimeError(
            f"persistent worker did not connect within "
            f"{_RENDEZVOUS_TIMEOUT_S:.0f}s. stderr tail: "
            f"{stderr_tail.decode('utf-8', errors='replace')[-500:]}"
        )
    finally:
        listener.close()

    sock.settimeout(None)  # blocking I/O on the rest of the connection
    _worker_sock = sock
    _wlog(f"[parent] _spawn_worker: worker connected from {addr}")

    # Fresh drainer per worker — the previous drainer (if any) is
    # still draining the dead worker's stderr until that pipe EOFs;
    # its daemon thread will exit on its own.
    _stderr_drainer = _StderrDrainer(proc.stderr)
    _wlog("[parent] _spawn_worker: stderr drainer started")
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
        # Close the dead worker's socket before spawning a fresh one.
        # ``_spawn_worker`` will install a new one alongside the new
        # ``_worker_proc``.
        if _worker_sock is not None:
            try:
                _worker_sock.close()
            except OSError:
                pass
            _worker_sock = None
        _worker_proc = _spawn_worker()
    return _worker_proc


def submit_job(job_name: str, **kwargs: Any) -> Dict[str, Any]:
    """Send a job to the persistent worker and block on its response.

    The wait happens inside a ``subprocess`` pipe read, which releases
    the GIL — so the parent's other threads (Dash callbacks, the
    pywebview event loop on its native thread, etc.) stay responsive.

    Args:
        job_name: ``"compute_rf"`` or ``"compute_consensus_tree"`` — must match a
            branch in ``__init__.py:_run_persistent_worker``.
        **kwargs: forwarded to the worker function.

    Returns the worker function's return value (unpickled). Raises
    ``JobCancelled`` if the user stopped the job via
    ``cancel_current_job``, or ``RuntimeError`` if the worker reported
    an error or crashed.
    """
    global _worker_proc, _worker_sock, _current_job, _cancelled
    _wlog(f"[parent] submit_job called: job={job_name!r}, kwarg keys={sorted(kwargs.keys())}")
    with _lock:
        _wlog("[parent] submit_job: _lock acquired")
        _cancelled = False
        proc = _ensure_running()
        # _ensure_running guarantees _worker_sock is set alongside
        # _worker_proc (both are populated by _spawn_worker).
        assert _worker_sock is not None
        sock = _worker_sock
        _wlog(f"[parent] submit_job: worker pid={proc.pid}, alive={proc.poll() is None}")

        request_bytes = pickle.dumps({"job": job_name, "kwargs": kwargs})
        _wlog(f"[parent] submit_job: pickled request; size={len(request_bytes)}")
        # ``_current_job`` is the cancellation window: while it's set,
        # cancel_current_job() may kill this worker.
        _current_job = job_name
        try:
            _wlog("[parent] submit_job: sending 4-byte size header on socket")
            sock.sendall(struct.pack("<I", len(request_bytes)))
            _wlog(f"[parent] submit_job: sending {len(request_bytes)}-byte request body")
            sock.sendall(request_bytes)
            _wlog("[parent] submit_job: receiving 4-byte response size header")
            size_bytes = _recv_exactly(sock, 4)
            _wlog(f"[parent] submit_job: got response header; bytes={len(size_bytes)}")
        except OSError as e:
            # Socket broke mid-request — almost always because
            # cancel_current_job() just killed the worker.
            _wlog(f"[parent] submit_job: OSError during IPC: {type(e).__name__}: {e}")
            size_bytes = b""
        finally:
            _current_job = None

        if len(size_bytes) != 4:
            # Worker died mid-job.
            _wlog(f"[parent] submit_job: short/empty response header; cancelled={_cancelled}")
            if _cancelled:
                # User Stop. Respawn now, while we still hold _lock, so
                # the next compute doesn't pay interpreter-boot latency.
                try:
                    _worker_proc = _spawn_worker()
                except Exception:  # noqa: BLE001 — _ensure_running retries
                    _worker_proc = None
                raise JobCancelled(f"{job_name} cancelled by user")
            # Genuine crash — surface stderr so we have something to
            # debug with. ``_worker_stderr_tail`` reads from the
            # background drainer; a direct ``proc.stderr.read()``
            # here would block forever because the drainer owns it.
            stderr_tail = _worker_stderr_tail()
            raise RuntimeError(
                f"persistent worker died during {job_name!r}: "
                f"{stderr_tail.decode('utf-8', errors='replace')[-500:]}"
            )
        (size,) = struct.unpack("<I", size_bytes)
        _wlog(f"[parent] submit_job: response body size={size}; receiving body")
        response_bytes = _recv_exactly(sock, size)
        _wlog(f"[parent] submit_job: received response body; got {len(response_bytes)} bytes")

    _wlog("[parent] submit_job: _lock released; unpickling response")
    response = pickle.loads(response_bytes)
    _wlog(f"[parent] submit_job: response unpickled; ok={response.get('ok', '?')}")
    if not response.get("ok"):
        raise RuntimeError(
            response.get("error", "<unknown worker error>")
            + "\n" + response.get("traceback", "")
        )
    _wlog("[parent] submit_job: returning result to caller")
    return response["result"]


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """``sock.recv(n)`` can return short reads (e.g. across TCP MSS
    boundaries) — keep reading until we have N bytes or hit EOF.

    Returns the bytes received (which may be fewer than ``n`` if the
    peer closed the connection mid-frame; callers treat that as EOF).
    """
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return data
        data += chunk
    return data
