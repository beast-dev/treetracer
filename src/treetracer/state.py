"""Server-side state for data too large to send through dcc.Store.

RF distance matrices are stored on disk as uint16 numpy .npy files in a temp
directory. For 4000 trees a matrix is ~32 MB on disk, reads in ~25 ms.
Only lightweight metadata (tree names) goes through dcc.Store / JSON.
"""

import atexit
import os
import secrets
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


def get_snapshots_path(name):
    """Return the .npz file path for a matrix's interned-snapshot data.

    The compute_rf worker saves the presence matrix + leaf_names to this
    parallel path whenever it computes an RF matrix. Convergence
    diagnostics (Pseudo-ESS, ASDSF, Fréchet) read it via:

        data = np.load(get_snapshots_path(name), allow_pickle=False)
        presence = data["presence"]          # (n_trees, n_bipartitions) uint8
        leaf_names = data["leaf_names"]      # alphabetical taxa
    """
    d = _ensure_tmpdir()
    return os.path.join(d, name.replace("/", "_") + "_snapshots.npz")


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


def get_distmat_names(name):
    """Return the row/column tree-name ordering of a stored matrix.

    The same ordering is used by the snapshot file at
    ``get_snapshots_path(name)``, so callers can map a tree name to its
    row index in the presence matrix.
    """
    return _distmat_index[name]["names"]


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
# Between-run MDS results are stored server-side to avoid sending coordinate
# data through JSON callback responses. Only lightweight metadata (n_trees,
# groups, dimensions) goes through dcc.Store. The Within-run Analysis tab
# now consumes the same store, filtered to one group per view — there is no
# longer a separate within-run MDS computation.

_mds_results = {}       # key -> {"metadata": {...}, "data": list[dict]}
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


def clear_all_mds_results():
    """Clear all server-side MDS results."""
    _mds_results.clear()


# ---------------------------------------------------------------------------
# In-memory MCC tree cache
# ---------------------------------------------------------------------------
# When the user clicks "View MCC" we compute the MCC NEXUS bytes once on the
# server side and stash them under a random URL-safe token, then open
# /peartree/<token> in a new browser window. The peartree page then fetches
# /peartree/<token>/tree.nex back from this cache. Cache lives in memory
# only — a server restart drops it.

_mcc_cache = {}             # uuid_str -> bytes (NEXUS)
_MAX_MCC_TREES = 50         # evict oldest when exceeded


def cache_mcc_tree(nexus_bytes):
    """Stash NEXUS bytes for a freshly-computed MCC tree and return a
    URL-safe handle. Oldest cached tree is evicted when the cache fills."""
    global _mcc_cache
    if len(_mcc_cache) >= _MAX_MCC_TREES:
        oldest = next(iter(_mcc_cache))
        del _mcc_cache[oldest]
    uid = secrets.token_urlsafe(8)
    _mcc_cache[uid] = nexus_bytes
    return uid


def get_cached_mcc_tree(uid):
    """Return cached NEXUS bytes for *uid*, or None if absent / evicted."""
    return _mcc_cache.get(uid)


def has_cached_mcc_tree(uid):
    return uid in _mcc_cache


def clear_all_mcc_trees():
    """Drop every cached MCC tree. Called from the sidebar's Clear-data
    handler so the cache doesn't outlive the data it summarises."""
    _mcc_cache.clear()
