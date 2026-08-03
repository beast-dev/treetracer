"""Process-isolated RF-trace row extraction.

An RF trace needs one row of an already-computed square distance matrix. The
old callback called ``np.load`` without memory mapping in the GUI process,
materialising the complete O(n²) matrix to read O(n) values. This worker maps
the file read-only and touches only the selected row.
"""

from __future__ import annotations

from typing import Any


def compute_rf_trace_worker_entry(
    *,
    matrix_path: str,
    reference_index: int,
) -> dict[str, Any]:
    """Return one RF-distance row as plain integers for IPC."""
    import time

    import numpy as np

    started = time.perf_counter()
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"RF matrix must be square; received shape {matrix.shape!r}"
        )

    reference_index = int(reference_index)
    if not 0 <= reference_index < matrix.shape[0]:
        raise IndexError(
            f"reference index {reference_index} is outside a "
            f"{matrix.shape[0]}-row RF matrix"
        )

    # ``np.asarray`` keeps the mmap-backed view; ``tolist`` performs the only
    # materialisation and converts the O(n) row into a compact pickle payload.
    distances = np.asarray(matrix[reference_index]).tolist()
    return {
        "distances": distances,
        "matrix_size": int(matrix.shape[0]),
        "elapsed": time.perf_counter() - started,
    }
