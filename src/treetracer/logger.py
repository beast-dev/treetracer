"""Centralized logging for the TreeTracer UI.

Stores log messages in a module-level deque so any callback can call
add_log() and the polling callback can read them with get_logs().
"""

from collections import deque
from datetime import datetime

MAX_LOG_ENTRIES = 500

_log_queue: deque = deque(maxlen=MAX_LOG_ENTRIES)


def add_log(message: str, level: str = "INFO"):
    """Append a timestamped log entry."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    _log_queue.append({
        "timestamp": timestamp,
        "level": level,
        "message": message,
    })


def get_logs():
    """Return all log entries as a list."""
    return list(_log_queue)


def clear_logs():
    """Clear all log entries."""
    _log_queue.clear()
