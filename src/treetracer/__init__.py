"""TreeTracer package init.

Two execution modes, branching on the env var
``TREETRACER_WORKER_MODE``:

1. **Worker mode** (env var set) — re-entrant subprocess invocation
   from ``callbacks/persistent_worker.py``. Skips all GUI imports and
   enters a request loop reading length-prefixed pickle frames from a
   localhost socket, dispatching to one of the registered worker functions,
   and returning heartbeat/result frames on that socket. Lives for the
   lifetime of the parent app.

   The persistent design avoids paying ~1.5 s of Python interpreter
   boot + import on every Compute RF / View consensus tree click — the boot is
   paid once at app launch, hidden behind the normal startup
   sequence.

2. **Normal mode** (env var absent) — configure thread-pool
   environment variables, then import the Dash app entry point.
"""

import os
import sys


_WORKER_MODE = os.environ.pop("TREETRACER_WORKER_MODE", None)


def _run_persistent_worker() -> int:
    """Read-dispatch-respond loop. Reached when this process was started
    by ``callbacks/persistent_worker.start()`` — see that module for
    the parent-side IPC client and the protocol description.

    IPC happens over a localhost TCP socket whose port number was
    handed to us via the ``TREETRACER_WORKER_PORT`` env var (the parent
    listens, we connect). Stdio is NOT used — the Briefcase Windows
    GUI stub silently rebinds the worker's CRT fd 1 during
    ``Py_Initialize``, so any writes against it never reached the
    parent's pipe and the GUI hung forever. Sockets sidestep CRT
    stdio entirely.

    Every transition point in this function emits a line to the
    shared worker log (``treetracer._worker_log``) so a stuck worker
    can be diagnosed post-mortem by reading one file.
    """
    import math
    import socket
    import traceback
    from collections.abc import Mapping

    from ._worker_log import log as wlog, get_log_path
    from .worker_protocol import (
        HeartbeatEmitter,
        WorkerConnectionClosed,
        WorkerProtocolError,
        receive_message,
        send_message,
    )

    wlog("entered _run_persistent_worker")
    # Surface the log path on stderr too. The parent's stderr drainer
    # captures it, so it shows up in the RuntimeError tail if the
    # worker eventually dies. For a hang, the user opens the file
    # directly.
    try:
        sys.stderr.write(f"treetracer worker log: {get_log_path()}\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 — defensive; sys.stderr can be None
        pass

    # ── Connect back to the parent ─────────────────────────────────
    port_str = os.environ.get("TREETRACER_WORKER_PORT")
    if not port_str:
        wlog("FATAL: TREETRACER_WORKER_PORT not set in env")
        try:
            sys.stderr.write("worker: TREETRACER_WORKER_PORT missing\n")
        except Exception:
            pass
        return 1
    try:
        port = int(port_str)
    except ValueError:
        wlog(f"FATAL: TREETRACER_WORKER_PORT={port_str!r} is not an int")
        return 1

    wlog(f"connecting to parent at 127.0.0.1:{port}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(("127.0.0.1", port))
    except OSError as e:
        wlog(f"FATAL: connect failed: {type(e).__name__}: {e}")
        return 1
    wlog("connected; entering recv loop")

    heartbeat_interval_raw = os.environ.get(
        "TREETRACER_WORKER_HEARTBEAT_INTERVAL_S",
        "2",
    )
    try:
        heartbeat_interval_s = float(heartbeat_interval_raw)
        if not math.isfinite(heartbeat_interval_s) or heartbeat_interval_s <= 0:
            raise ValueError
    except ValueError:
        heartbeat_interval_s = 2.0
        wlog(
            "invalid TREETRACER_WORKER_HEARTBEAT_INTERVAL_S="
            f"{heartbeat_interval_raw!r}; using 2s"
        )

    while True:
        wlog("waiting for next request frame")
        try:
            request = receive_message(sock)
        except WorkerConnectionClosed:
            wlog("EOF on socket; clean shutdown")
            try:
                sock.close()
            except OSError:
                pass
            return 0
        except (OSError, WorkerProtocolError) as exc:
            wlog(
                "FATAL: request receive failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return 1

        job = request.get("job")
        kwargs = request.get("kwargs")
        if not isinstance(job, str) or not isinstance(kwargs, Mapping):
            wlog("FATAL: request is missing string job or mapping kwargs")
            return 1
        kwargs = dict(kwargs)
        wlog(f"received job={job!r}, kwarg keys={sorted(kwargs.keys())}")

        heartbeat = HeartbeatEmitter(
            sock,
            job,
            interval_s=heartbeat_interval_s,
            log=wlog,
        )
        try:
            heartbeat.start()
        except OSError as exc:
            wlog(
                "FATAL: initial heartbeat failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return 1

        try:
            if job == "compute_rf":
                wlog("importing compute_rf_worker_entry")
                from .rf._subprocess_worker import compute_rf_worker_entry
                wlog("calling compute_rf_worker_entry")
                result = compute_rf_worker_entry(**kwargs)
                wlog("compute_rf_worker_entry returned")
            elif job == "compute_consensus_tree":
                wlog("importing compute_consensus_tree_worker_entry")
                from .consensus_tree._subprocess_worker import compute_consensus_tree_worker_entry
                wlog("calling compute_consensus_tree_worker_entry")
                result = compute_consensus_tree_worker_entry(**kwargs)
                wlog("compute_consensus_tree_worker_entry returned")
            elif job == "compute_pseudo_ess":
                wlog("importing compute_pseudo_ess_worker_entry")
                from .ess._subprocess_worker import compute_pseudo_ess_worker_entry
                wlog("calling compute_pseudo_ess_worker_entry")
                result = compute_pseudo_ess_worker_entry(**kwargs)
                wlog("compute_pseudo_ess_worker_entry returned")
            elif job == "compute_rf_trace":
                wlog("importing compute_rf_trace_worker_entry")
                from .ess._rf_trace_worker import compute_rf_trace_worker_entry
                wlog("calling compute_rf_trace_worker_entry")
                result = compute_rf_trace_worker_entry(**kwargs)
                wlog("compute_rf_trace_worker_entry returned")
            elif job == "compute_clade_frequencies":
                wlog("importing compute_clade_frequencies_worker_entry")
                from .clade_freq._subprocess_worker import (
                    compute_clade_frequencies_worker_entry,
                )
                wlog("calling compute_clade_frequencies_worker_entry")
                result = compute_clade_frequencies_worker_entry(**kwargs)
                wlog("compute_clade_frequencies_worker_entry returned")
            elif job == "compute_mds":
                # MDS doesn't need a dedicated worker wrapper — the
                # ``rf._worker.compute_mds_worker`` function is already
                # subprocess-friendly (reads matrix from disk, returns
                # plain Python lists). We just import and call it.
                wlog("importing compute_mds_worker")
                from .rf._worker import compute_mds_worker
                wlog("calling compute_mds_worker")
                result = compute_mds_worker(**kwargs)
                wlog("compute_mds_worker returned")
            else:
                wlog(f"FATAL: unknown job {job!r}")
                raise RuntimeError(f"Unknown job: {job!r}")
            response = {"type": "result", "ok": True, "result": result}
            wlog("job succeeded; serialising response")
        except BaseException as e:  # noqa: BLE001 — defensive: one bad
            # job must not crash the worker.
            wlog(f"job raised: {type(e).__name__}: {e}")
            response = {
                "type": "result",
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
        finally:
            heartbeat.stop()

        wlog("sending result response")
        try:
            send_message(sock, response, lock=heartbeat.send_lock)
        except OSError as e:
            wlog(f"FATAL: sendall failed: {type(e).__name__}: {e}")
            return 1
        wlog("response sent; loop back to next request")


if _WORKER_MODE == "persistent":
    sys.exit(_run_persistent_worker())


# ---------------------------------------------------------------------------
# Normal mode below this line — only reached when not in worker mode.
# ---------------------------------------------------------------------------


def _configure_threads() -> int:
    """Cap compute parallelism at ``cpu_count - 2``, leaving two cores
    free for the OS, the Dash callback thread, and the pywebview / Edge
    WebView2 process. Returns the value applied.

    Honors a user override via ``TREETRACER_NUM_THREADS=<n>`` and never
    clobbers a variable the user has already set in their shell.
    """
    override = os.environ.get("TREETRACER_NUM_THREADS")
    if override and override.strip().isdigit():
        n = max(1, int(override))
    else:
        n = max(1, (os.cpu_count() or 4) - 2)

    # ``setdefault``: respect any value the user already set externally.
    for var in (
        "RAYON_NUM_THREADS",     # rapidtrees (Rust + rayon)
        "OPENBLAS_NUM_THREADS",  # scipy / numpy — PyPI wheels ship OpenBLAS
        "OMP_NUM_THREADS",       # generic OpenMP umbrella; cheap safety net
    ):
        os.environ.setdefault(var, str(n))
    return n


NUM_THREADS = _configure_threads()
del _configure_threads

from .app import main

# Allow ``python -m treetracer``.
if __name__ == "__main__":
    main()
