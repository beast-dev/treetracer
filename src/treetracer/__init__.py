"""TreeTracer package init.

Two execution modes, branching on the env var
``TREETRACER_WORKER_MODE``:

1. **Worker mode** (env var set) — re-entrant subprocess invocation
   from ``callbacks/persistent_worker.py``. Skips all GUI imports and
   enters a request loop reading length-prefixed pickle frames from
   stdin, dispatching to one of the registered worker functions, and
   writing the result back to stdout. Lives for the lifetime of the
   parent app.

   The persistent design avoids paying ~1.5 s of Python interpreter
   boot + import on every Compute RF / View MCC click — the boot is
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
    """
    import io
    import pickle
    import struct
    import traceback

    # ``sys.stdin.buffer`` / ``sys.stdout.buffer`` give us raw bytes
    # streams. We deliberately do NOT touch sys.stdin/sys.stdout
    # otherwise — the text wrappers buffer in ways that confuse
    # length-prefixed binary framing.
    stdin: io.BufferedReader = sys.stdin.buffer
    stdout: io.BufferedWriter = sys.stdout.buffer

    def _read_exactly(n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = stdin.read(n - len(data))
            if not chunk:
                return b""  # EOF mid-frame → parent closed; exit loop
            data += chunk
        return data

    while True:
        size_bytes = _read_exactly(4)
        if not size_bytes:
            return 0  # clean shutdown
        (size,) = struct.unpack("<I", size_bytes)
        request_bytes = _read_exactly(size)
        if len(request_bytes) != size:
            sys.stderr.write("persistent worker: short read on request body\n")
            return 1
        try:
            request = pickle.loads(request_bytes)
            job = request["job"]
            kwargs = request["kwargs"]

            if job == "compute_rf":
                from .rf._subprocess_worker import compute_rf_worker_entry
                result = compute_rf_worker_entry(**kwargs)
            elif job == "compute_mcc":
                from .mcc._subprocess_worker import compute_mcc_worker_entry
                result = compute_mcc_worker_entry(**kwargs)
            elif job == "compute_pseudo_ess":
                from .ess._subprocess_worker import compute_pseudo_ess_worker_entry
                result = compute_pseudo_ess_worker_entry(**kwargs)
            elif job == "compute_mds":
                # MDS doesn't need a dedicated worker wrapper — the
                # ``rf._worker.compute_mds_worker`` function is already
                # subprocess-friendly (reads matrix from disk, returns
                # plain Python lists). We just import and call it.
                from .rf._worker import compute_mds_worker
                result = compute_mds_worker(**kwargs)
            else:
                raise RuntimeError(f"Unknown job: {job!r}")
            response = {"ok": True, "result": result}
        except BaseException as e:  # noqa: BLE001 — defensive: one bad
            # job must not crash the worker.
            response = {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }

        response_bytes = pickle.dumps(response)
        stdout.write(struct.pack("<I", len(response_bytes)))
        stdout.write(response_bytes)
        stdout.flush()


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
