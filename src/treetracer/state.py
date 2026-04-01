"""Server-side state for data too large to send through dcc.Store.

RF distance matrices are stored on disk as uint16 numpy .npy files in a temp
directory. For 4000 trees a matrix is ~32 MB on disk, reads in ~25 ms.
Only lightweight metadata (tree names) goes through dcc.Store / JSON.
"""

import atexit
import os
import shutil
import tempfile

import numpy as np


_tmpdir = None
_distmat_index = {}  # name -> {"names": list[str], "path": str, "file_breakdown": dict}
_distmat_counter = 0  # auto-incrementing ID for unique matrix names


def _ensure_tmpdir():
    global _tmpdir
    if _tmpdir is None:
        _tmpdir = tempfile.mkdtemp(prefix="treetracer_distmat_")
        atexit.register(_cleanup_tmpdir)
    return _tmpdir


def _cleanup_tmpdir():
    global _tmpdir
    if _tmpdir and os.path.isdir(_tmpdir):
        shutil.rmtree(_tmpdir, ignore_errors=True)
        _tmpdir = None


def next_distmat_name():
    """Return a unique matrix name like 'RF_001', 'RF_002', etc."""
    global _distmat_counter
    _distmat_counter += 1
    return f"RF_{_distmat_counter:03d}"


def get_distmat_path(name):
    """Return the .npy file path for a matrix name (creates temp dir if needed)."""
    d = _ensure_tmpdir()
    return os.path.join(d, name.replace("/", "_") + ".npy")


def register_distmat(name, names, path, file_breakdown=None):
    """Register a matrix that was already saved to disk by a subprocess worker."""
    _distmat_index[name] = {
        "names": list(names),
        "path": path,
        "file_breakdown": file_breakdown or {},
    }


def save_distmat(name, names, matrix, file_breakdown=None):
    """Save an RF distance matrix as uint16 .npy and register it.

    Args:
        name: Key for this matrix (e.g. "RF_001").
        names: List of tree name strings (length n).
        matrix: n×n distance values (list-of-lists or numpy array).
        file_breakdown: Optional dict of {filename: n_trees} showing which
            source files contributed and how many trees each.
    """
    d = _ensure_tmpdir()
    arr = np.array(matrix, dtype=np.uint16)
    path = os.path.join(d, name.replace("/", "_") + ".npy")
    np.save(path, arr)
    _distmat_index[name] = {
        "names": list(names),
        "path": path,
        "file_breakdown": file_breakdown or {},
    }


def load_distmat(name):
    """Load a matrix from disk.

    Returns:
        (names, np.ndarray[uint16]) — tree names and n×n matrix.

    Raises:
        KeyError: if *name* is not in the index.
    """
    entry = _distmat_index[name]
    arr = np.load(entry["path"])
    return entry["names"], arr


def load_distmat_as_lists(name):
    """Load a matrix and convert to plain Python lists (for subprocess pickling)."""
    names, arr = load_distmat(name)
    return names, arr.tolist()


def get_distmat_index():
    """Return lightweight metadata dict suitable for dcc.Store (no paths, no matrix).

    The names list is kept server-side only (too large for JSON with long tree names).
    The browser only needs the file_breakdown and tree count.

    Returns:
        {"name1": {"n_trees": int, "file_breakdown": {...}}, ...}
    """
    return {
        k: {"n_trees": len(v["names"]), "file_breakdown": v.get("file_breakdown", {})}
        for k, v in _distmat_index.items()
    }


def has_distmat(name):
    return name in _distmat_index


def clear_all_distmats():
    """Remove all .npy files from disk and reset the index."""
    global _distmat_counter
    for entry in _distmat_index.values():
        try:
            os.remove(entry["path"])
        except OSError:
            pass
    _distmat_index.clear()
    _distmat_counter = 0
