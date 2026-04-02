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
_MAX_DISTMATS = 50   # evict oldest when exceeded


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


def register_distmat(name, names, path, file_breakdown=None, groups_per_file=None):
    """Register a matrix that was already saved to disk by a subprocess worker."""
    # Evict oldest if at capacity
    if len(_distmat_index) >= _MAX_DISTMATS and name not in _distmat_index:
        oldest = next(iter(_distmat_index))
        old_path = _distmat_index[oldest].get("path")
        if old_path:
            try:
                os.remove(old_path)
            except OSError:
                pass
        del _distmat_index[oldest]
    _distmat_index[name] = {
        "names": list(names),
        "path": path,
        "file_breakdown": file_breakdown or {},
        "groups_per_file": groups_per_file or {},
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
        k: {
            "n_trees": len(v["names"]),
            "file_breakdown": v.get("file_breakdown", {}),
            "groups_per_file": v.get("groups_per_file", {}),
        }
        for k, v in _distmat_index.items()
    }


def has_distmat(name):
    return name in _distmat_index


def get_distmat_file_path(name):
    """Return the .npy file path for a stored matrix."""
    return _distmat_index[name]["path"]


def get_distmat_groups_per_file(name):
    """Return the groups_per_file mapping for a stored matrix."""
    return _distmat_index.get(name, {}).get("groups_per_file", {})


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


# ---------------------------------------------------------------------------
# Server-side MDS result storage
# ---------------------------------------------------------------------------
# MDS results (between-run and within-run) are stored server-side to avoid
# sending coordinate data through JSON callback responses. Only lightweight
# metadata (n_trees, groups, dimensions) goes through dcc.Store.

_mds_results = {}       # key -> {"metadata": {...}, "data": list[dict]}
_wr_mds_results = {}    # key -> {"file": str, "source_distmat": str, "dimensions": [...], "n_trees": int, "data": list[dict]}
_MAX_MDS_RESULTS = 50   # evict oldest when exceeded


def store_mds_result(key, result):
    """Store a between-run MDS result server-side."""
    if len(_mds_results) >= _MAX_MDS_RESULTS and key not in _mds_results:
        oldest = next(iter(_mds_results))
        del _mds_results[oldest]
    _mds_results[key] = result


def get_mds_result(key):
    """Get a between-run MDS result by key."""
    return _mds_results.get(key)


def get_mds_results_index():
    """Return lightweight metadata for dcc.Store (no coordinate data)."""
    return {
        k: {
            "filename": v["metadata"]["filename"],
            "source_distmat": v["metadata"].get("source_distmat", "?"),
            "rows": v["metadata"]["rows"],
            "dimensions": v["metadata"]["dimensions"],
            "groups": v["metadata"]["groups"],
            "MIN_TREENUM": v["metadata"]["MIN_TREENUM"],
            "MAX_TREENUM": v["metadata"]["MAX_TREENUM"],
        }
        for k, v in _mds_results.items()
    }


def store_wr_mds_result(key, result):
    """Store a within-run MDS result server-side."""
    if len(_wr_mds_results) >= _MAX_MDS_RESULTS and key not in _wr_mds_results:
        oldest = next(iter(_wr_mds_results))
        del _wr_mds_results[oldest]
    _wr_mds_results[key] = result


def get_wr_mds_result(key):
    """Get a within-run MDS result by key."""
    return _wr_mds_results.get(key)


def get_wr_mds_results_index():
    """Return lightweight metadata for dcc.Store (no coordinate data)."""
    return {
        k: {
            "file": v.get("file", "?"),
            "source_distmat": v.get("source_distmat", "?"),
            "n_trees": v.get("n_trees", 0),
            "dimensions": v.get("dimensions", []),
        }
        for k, v in _wr_mds_results.items()
    }


def clear_all_mds_results():
    """Clear all server-side MDS results."""
    _mds_results.clear()
    _wr_mds_results.clear()
