"""Shared file logger for the persistent worker subprocess.

Used from ``__init__.py``'s ``_run_persistent_worker`` dispatcher
AND from the per-job worker entry points
(``rf/_subprocess_worker.py``, ``mcc/_subprocess_worker.py``, etc.)
so a stuck worker can be diagnosed by reading one file post-mortem
instead of poking at process state with a debugger.

Why a file (not stderr): the parent's stderr drainer captures bytes
into a 1 MB ring, but that's only surfaced when the worker DIES.
The whole problem with the current Windows hang is that the worker
DOESN'T die — it just sits there alive but unproductive. A file the
user can open in Notepad while the GUI is still hung is the simplest
way to see what happened.

Log path: ``<TEMP>/treetracer_worker.log`` — a fixed location so
users can find it without us having to surface a UI for it. Appended
to (not truncated), with a session banner on each spawn so a sequence
of Stop-and-retry cycles is readable as one timeline.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time


_LOG_PATH = os.path.join(tempfile.gettempdir(), "treetracer_worker.log")
_FILE_LOCK = threading.Lock()
# Tri-state: ``None`` = not yet opened, ``False`` = open failed and
# we've stopped trying, file object = open and ready. Tri-state
# avoids retrying the open on every log() call after a permissions /
# disk-full failure.
_log_file = None
_session_banner_emitted = False


def get_log_path() -> str:
    """Return the absolute path of the worker log file."""
    return _LOG_PATH


def log(msg: str) -> None:
    """Append a timestamped message to the worker log file.

    Best-effort: any I/O error is swallowed so logging can never
    crash the worker. Thread-safe — the worker is single-threaded
    on the dispatcher path but rapidtrees / numpy may use rayon /
    omp pools internally, and if they ever call back into Python
    logging we want serialised writes.
    """
    global _log_file, _session_banner_emitted
    with _FILE_LOCK:
        if _log_file is None:
            try:
                # Line-buffered (``buffering=1``) so each write is
                # flushed on the trailing newline — important for
                # crash diagnostics: if the worker is killed mid-step,
                # we want the most-recent line on disk.
                _log_file = open(_LOG_PATH, "a", encoding="utf-8", buffering=1)
            except OSError:
                _log_file = False
                return
        if _log_file is False:
            return
        if not _session_banner_emitted:
            _session_banner_emitted = True
            try:
                _log_file.write("=" * 60 + "\n")
                _log_file.write(
                    f"worker session start; pid={os.getpid()}; "
                    f"time={time.time():.3f}; "
                    f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}; "
                    f"platform={sys.platform}; "
                    f"exe={sys.executable!r}\n"
                )
                _log_file.write("=" * 60 + "\n")
            except Exception:
                pass
        try:
            _log_file.write(
                f"[{time.time():.6f}] [pid {os.getpid()}] {msg}\n"
            )
        except Exception:
            pass
