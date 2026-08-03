"""Server-side state for data too large to send through dcc.Store.

RF distance matrices are stored on disk as uint16 numpy .npy files in a temp
directory. For 4000 trees a matrix is ~32 MB on disk, reads in ~25 ms.
Only lightweight metadata (tree names) goes through dcc.Store / JSON.
"""

import atexit
import glob
import os
import re
import secrets
import shutil
import tempfile
import time

import numpy as np


_tmpdir = None
_distmat_index = {}  # name -> {"names": list[str], "path": str, "file_breakdown": dict}
_distmat_counter = 0  # auto-incrementing ID for unique matrix names
_MAX_DISTMATS = 50   # evict oldest when exceeded

# Per-distmat decode cache for the Clade Frequency Comparison pipeline.
#
# ``bipartition_bits`` from the snapshot is an (n_splits, n_leaves) uint8
# matrix. We turn each row into a sorted ``tuple[int]`` of leaf indices
# (the canonical side, as defined by rapidtrees: "side NOT containing
# leaf 0"). Storing these as int-tuples instead of frozensets-of-strings
# is ~5× faster to decode and ~5× smaller (8.5 MB vs 40 MB for a
# 41k-bipartition / 283-taxon distmat, measured in bench_clade_freq.py).
#
# Lazily populated on first ``get_canonical_keys(name)`` call. Cleared
# in ``clear_all_distmats`` so the cache lifetime is bound to the
# underlying snapshot's lifetime.
_distmat_canonical_keys = {}  # name -> {"tuples": list[tuple[int,...]], "leaf_names": list[str]}


# Tmpdir name format: ``treetracer_distmat_<pid>_<random>``. Embedding
# the PID lets a startup sweep distinguish leaked dirs (owning process
# is dead) from dirs owned by a still-running treetracer instance.
_TMPDIR_PREFIX = "treetracer_distmat_"
_TMPDIR_RE = re.compile(rf"^{re.escape(_TMPDIR_PREFIX)}(\d+)_")


def _sweep_stale_tmpdirs():
    """Remove leaked tmpdirs from previous sessions whose owning PID
    is no longer alive.

    Safe under concurrent treetracer windows: another live session's
    PID will be found alive and its dir spared. PIDs can in theory be
    reused after death, but the chance of a recycled PID running yet
    another treetracer instance is astronomically small in practice.
    """
    pattern = os.path.join(tempfile.gettempdir(), f"{_TMPDIR_PREFIX}*")
    try:
        import psutil
        pid_alive = psutil.pid_exists
    except Exception:
        # psutil should be available (declared in deps), but if anything
        # goes wrong, skip the sweep rather than risk deleting live dirs.
        return

    for path in glob.glob(pattern):
        name = os.path.basename(path)
        m = _TMPDIR_RE.match(name)
        if not m:
            # Old-style ``treetracer_distmat_<random>`` without an
            # embedded PID. Don't touch — can't tell if it's a leak
            # or a live session running an older code path.
            continue
        try:
            owning_pid = int(m.group(1))
        except ValueError:
            continue
        if pid_alive(owning_pid):
            continue
        shutil.rmtree(path, ignore_errors=True)


def _ensure_tmpdir():
    global _tmpdir
    if _tmpdir is None:
        # First call of the session: sweep stale leaks before we add
        # our own dir to the pile. Cheap (a few stat calls per leaked
        # dir) and self-healing for users with previous-session leaks.
        _sweep_stale_tmpdirs()
        _tmpdir = tempfile.mkdtemp(prefix=f"{_TMPDIR_PREFIX}{os.getpid()}_")
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


def register_distmat(name, names, path, file_breakdown=None, groups_per_file=None,
                     is_rooted=True):
    """Register a matrix that was already saved to disk by a subprocess worker.

    ``is_rooted`` reflects the rooting convention of the input ``.trees``
    files (validated to be consistent at compute time). Downstream
    features key off this:
      * consensus tree: midpoint-roots the chosen tree when ``is_rooted=False``.
      * RF dropdown UI: shows a "(rooted)" / "(unrooted)" suffix.
      * Future ASDSF / clade-freq exports: use bipartition semantics
        if ``is_rooted=False``.
    Defaults to True to keep the call-site simple for legacy paths and
    to match the pre-feature default.
    """
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
        "is_rooted": bool(is_rooted),
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
            "is_rooted": v.get("is_rooted", True),
        }
        for k, v in _distmat_index.items()
    }


def has_distmat(name):
    return name in _distmat_index


def get_distmat_file_path(name):
    """Return the .npy file path for a stored matrix."""
    return _distmat_index[name]["path"]


def get_distmat_is_rooted(name):
    """Return the rooting convention recorded for a stored matrix.
    Defaults to True for pre-feature distmats that don't carry the flag."""
    return _distmat_index.get(name, {}).get("is_rooted", True)


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


def get_distmat_groups_with_counts(name):
    """Per-group tree counts for a stored matrix.

    Tree names in the matrix are stored as ``"<group>/<orig>"`` (see
    ``TreeManagerPandas.insert_trees_batch_raw``). We split on ``/`` to
    derive the group. Returns a list of ``(group_name, count)`` tuples
    in a stable order (by first appearance in the matrix's row order,
    which matches how trees were inserted into the chain).
    """
    entry = _distmat_index.get(name)
    if not entry:
        return []
    counts = {}
    for tree_name in entry["names"]:
        group = str(tree_name).split("/", 1)[0]
        counts[group] = counts.get(group, 0) + 1
    return list(counts.items())


def clear_all_distmats():
    """Remove all .npy files from disk and reset the index.

    Also drops the lazy canonical-keys cache used by the Clade
    Frequency Comparison feature — those decoded tuples become stale
    the moment their backing snapshots disappear.
    """
    global _distmat_counter
    for entry in _distmat_index.values():
        try:
            os.remove(entry["path"])
        except OSError:
            pass
    _distmat_index.clear()
    _distmat_canonical_keys.clear()
    _distmat_counter = 0


def get_canonical_keys(source_distmat):
    """Lazily decode and cache the per-bipartition canonical-side tip
    index tuples for *source_distmat*.

    Returns a dict::

        {"tuples": list[tuple[int, ...]],  # one per bipartition column
         "leaf_names": list[str]}          # leaf_names[i] is the taxon
                                           # at index i in each tuple

    The tuple encoding feeds the scatter→tanglegram highlight
    resolution: each rooted clade's tuple holds the leaf indices of
    its descendants, and the click handler resolves those to tip
    names via ``leaf_names[i] for i in tuple``.

    Raises FileNotFoundError if the snapshot is missing on disk.
    """
    cached = _distmat_canonical_keys.get(source_distmat)
    if cached is not None:
        return cached
    snap = np.load(get_snapshots_path(source_distmat), allow_pickle=False)
    if "bipartition_bits" not in snap.files:
        raise KeyError(
            f"snapshot for {source_distmat!r} has no 'bipartition_bits' — "
            "regenerate with rapidtrees ≥ 0.5.0."
        )
    bits = snap["bipartition_bits"]
    tuples = [tuple(np.flatnonzero(row).tolist()) for row in bits]
    # ``parse_nexus`` strips outer single/double quotes from quoted
    # taxon identifiers when reading the Translate block (per NEXUS
    # spec: quotes are syntactic, not part of the name). Rapidtrees
    # passes the translate values through verbatim, so its snapshot
    # ``leaf_names`` keep the quote characters. Without normalising
    # here, the canonical names (e.g. ``"'24P021_..._2024'"``) and the
    # parsed consensus-tree node names (e.g. ``"24P021_..._2024"``) don't
    # match, the descendant-bits walk silently drops 1000+ tips, and
    # the tanglegram highlights scatter across paraphyletic groups.
    leaf_names = [str(n).strip("'\"") for n in snap["leaf_names"]]
    _distmat_canonical_keys[source_distmat] = {
        "tuples":     tuples,
        "leaf_names": leaf_names,
    }
    return _distmat_canonical_keys[source_distmat]


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
# Managed diagnostic/exploration result storage
# ---------------------------------------------------------------------------
# JobManager deliberately retains only small terminal references. These caches
# own the larger render payloads produced by RF Trace and clade comparison so a
# dropped browser response can be replayed without retaining worker results or
# recomputing domain work. They are bounded independently because neither UI
# needs unbounded history.

_rf_trace_results = {}
_clade_frequency_results = {}
_MAX_ANALYSIS_RESULTS = 16


def _store_bounded_result(cache, key, result):
    if len(cache) >= _MAX_ANALYSIS_RESULTS and key not in cache:
        del cache[next(iter(cache))]
    cache[key] = result


def store_rf_trace_result(key, result):
    """Store one full RF-trace render payload under a managed-job key."""
    _store_bounded_result(_rf_trace_results, key, result)


def get_rf_trace_result(key):
    """Return an RF-trace render payload, or ``None`` after eviction/reset."""
    return _rf_trace_results.get(key)


def store_clade_frequency_result(key, result):
    """Store one clade-comparison render and split-resolution payload."""
    _store_bounded_result(_clade_frequency_results, key, result)


def get_clade_frequency_result(key):
    """Return a clade-comparison payload, or ``None`` after eviction/reset."""
    return _clade_frequency_results.get(key)


def clear_all_analysis_results():
    """Drop managed RF-trace and clade-comparison render payloads."""
    _rf_trace_results.clear()
    _clade_frequency_results.clear()


# ---------------------------------------------------------------------------
# In-memory consensus tree cache
# ---------------------------------------------------------------------------
# When the user clicks "View consensus tree" we compute the consensus tree NEXUS bytes once on the
# server side and stash them under a random URL-safe token, then open
# /peartree/<token> in a new browser window. The peartree page then fetches
# /peartree/<token>/tree.nex back from this cache. Cache lives in memory
# only — a server restart drops it.

_consensus_tree_cache = {}             # uuid_str -> bytes (NEXUS)
_MAX_CONSENSUS_TREE_TREES = 50         # evict oldest when exceeded


def cache_consensus_tree(nexus_bytes):
    """Stash NEXUS bytes for a freshly-computed consensus tree and return a
    URL-safe handle. Oldest cached tree is evicted when the cache fills."""
    global _consensus_tree_cache
    if len(_consensus_tree_cache) >= _MAX_CONSENSUS_TREE_TREES:
        oldest = next(iter(_consensus_tree_cache))
        del _consensus_tree_cache[oldest]
    uid = secrets.token_urlsafe(8)
    _consensus_tree_cache[uid] = nexus_bytes
    return uid


def get_cached_consensus_tree(uid):
    """Return cached NEXUS bytes for *uid*, or None if absent / evicted."""
    return _consensus_tree_cache.get(uid)


def has_cached_consensus_tree(uid):
    return uid in _consensus_tree_cache


def clear_all_consensus_trees():
    """Drop every cached consensus tree and its registry entry. Called from the
    sidebar's Clear-data handler so the cache doesn't outlive the data it
    summarises."""
    _consensus_tree_cache.clear()
    clear_all_consensus_tree_registry()


# ---------------------------------------------------------------------------
# consensus tree registry
# ---------------------------------------------------------------------------
# A persistent (within-session) record of every consensus tree computed in the
# Between-runs and Within-run MDS tabs. Each entry pairs a cached consensus tree's
# uuid with metadata about the source distmat / mode / run / selection,
# so the UI can list, re-open, and overlay them on the MDS plots.

_consensus_tree_registry: list = []                # list of registry entry dicts
_consensus_tree_registry_counters: dict = {}       # (distmat, mode, run|None) -> int
_MAX_CONSENSUS_TREE_REGISTRY = _MAX_CONSENSUS_TREE_TREES      # mirror cache cap


def _next_consensus_tree_name(source_distmat, mode, run):
    """Allocate the next sequential consensus tree name for *(distmat, mode, run)*.

    Run is included in the key (and the resulting name) only for Within
    so that consensus trees computed for different runs of the same matrix don't
    collide.
    """
    key = (source_distmat, mode, run)
    n = _consensus_tree_registry_counters.get(key, 0) + 1
    _consensus_tree_registry_counters[key] = n
    if mode == "Within" and run:
        return f"{source_distmat}_Within_{run}_consensus_tree_{n}"
    return f"{source_distmat}_{mode}_consensus_tree_{n}"


def register_consensus_tree(*, source_distmat, mode, run, uuid, consensus_tree,
                 selection, log_clade_credibility,
                 consensus_tree_log_posterior=None, tree_names=None,
                 counts=None, cols_in_consensus_tree=None):
    """Append a new consensus tree registry entry and return it.

    Evicts the oldest entry (and its uuid from the cache) if the
    registry is at cap, keeping list and cache strictly synchronised.

    ``tree_names`` is the flat list of "group/STATE_N" tree names from
    the user's selection — kept for provenance.

    ``counts`` is the pre-computed column-sum of the snapshot's
    presence matrix over the selected rows: a numpy uint32/int32 array
    of length ``n_bipartitions``. Caller computes this from
    ``presence[row_idx].sum(axis=0)`` while it already has
    ``presence_sub`` in scope (in ``consensus_tree.compute_consensus_tree_for_selection``).
    Caching at registration time means Compare clicks don't pay the
    row-sum cost. Optional — registry stays usable without it but
    falls back to the slow recompute path in
    ``clade_freq.compute_clade_frequencies``.

    ``cols_in_consensus_tree`` is an iterable of presence-matrix column indices
    that appear in the chosen consensus tree itself (``np.flatnonzero(
    presence_sub[consensus_tree_local])``). These are the interned bipartition IDs
    of the consensus tree's own clades; the Clade Frequency Comparison filter wraps
    them in a ``set`` once per Compare click for O(1) membership.
    Stored as a sorted list because the registry travels through the
    browser-side ``consensus-tree-registry-store`` and ``frozenset`` is not
    JSON-serialisable.
    """
    global _consensus_tree_registry
    if len(_consensus_tree_registry) >= _MAX_CONSENSUS_TREE_REGISTRY:
        oldest = _consensus_tree_registry.pop(0)
        _consensus_tree_cache.pop(oldest.get("uuid"), None)
    name = _next_consensus_tree_name(source_distmat, mode, run)
    entry = {
        "name": name,
        "uuid": uuid,
        "source_distmat": source_distmat,
        "mode": mode,
        "run": run,
        "consensus_tree": consensus_tree,
        "selection": selection,
        "log_clade_credibility": log_clade_credibility,
        "consensus_tree_log_posterior": consensus_tree_log_posterior,
        "tree_names": list(tree_names) if tree_names is not None else [],
        "n_trees": len(tree_names) if tree_names is not None else 0,
        "counts": (np.asarray(counts, dtype=np.int32)
                   if counts is not None else None),
        # Stored as a plain list (JSON-serialisable) because the
        # registry payload flows through ``consensus-tree-registry-store`` in
        # the browser. Callers that need O(1) membership wrap with
        # ``set(...)`` at use time — cheap (~few hundred ints) and
        # only paid once per Compare click.
        "cols_in_consensus_tree": (sorted(cols_in_consensus_tree)
                        if cols_in_consensus_tree is not None else None),
        # Flipped True by ``rename_consensus_tree`` once the user has confirmed a
        # name (even an unedited save counts — "I looked at it and this
        # is fine"). Drives the View-flow rename modal in
        # ``callbacks/rename_consensus_tree.py``: as long as this is False, the
        # first View click on the entry opens the rename prompt instead
        # of going straight to PearTree.
        "name_user_set": False,
        "created_at": time.time(),
    }
    _consensus_tree_registry.append(entry)
    return entry


def rename_consensus_tree(name_or_uuid, new_name):
    """Rename a consensus tree registry entry.

    Returns ``(entry, error)``. On success ``entry`` is the mutated
    registry dict and ``error`` is ``None``. On failure ``entry`` is
    ``None`` and ``error`` is a short user-facing string suitable for
    inline display in the rename modal or a Notification.

    Lookup accepts either the current name OR the entry's uuid — the
    uuid form lets callers that already hold a stable handle (e.g. the
    pattern-matching button id) skip the name round-trip.

    Validation:
      * non-empty after stripping
      * ≤ 80 characters
      * unique across the registry (case-sensitive)

    A no-op rename (same name) is allowed and still flips
    ``name_user_set`` to True — the user explicitly accepted the
    current name, so the View-flow modal stops prompting.
    """
    if new_name is None:
        return None, "Name cannot be empty."
    new_name = str(new_name).strip()
    if not new_name:
        return None, "Name cannot be empty."
    if len(new_name) > 80:
        return None, "Name is too long (max 80 characters)."

    target = None
    for e in _consensus_tree_registry:
        if e.get("name") == name_or_uuid or e.get("uuid") == name_or_uuid:
            target = e
            break
    if target is None:
        return None, "consensus tree entry not found (may have been cleared)."

    if target["name"] != new_name:
        for e in _consensus_tree_registry:
            if e is target:
                continue
            if e.get("name") == new_name:
                return None, f"Name '{new_name}' is already in use."
        target["name"] = new_name

    target["name_user_set"] = True
    return target, None


def get_consensus_tree_registry():
    """Snapshot the registry for a dcc.Store payload."""
    return list(_consensus_tree_registry)


def get_consensus_tree_registry_entry(uid):
    """Return the registry entry whose uuid matches *uid*, or None."""
    for e in _consensus_tree_registry:
        if e.get("uuid") == uid:
            return e
    return None


def get_consensus_tree_registry_filtered(*, source_distmat=None, mode=None, run=None):
    """Return registry entries matching the given filters."""
    out = []
    for e in _consensus_tree_registry:
        if source_distmat is not None and e["source_distmat"] != source_distmat:
            continue
        if mode is not None and e["mode"] != mode:
            continue
        if run is not None and e["run"] != run:
            continue
        out.append(e)
    return out


def delete_consensus_tree(name):
    """Remove the entry with *name* and its cached NEXUS bytes.

    Returns True if an entry was removed, False otherwise.
    """
    global _consensus_tree_registry
    for i, e in enumerate(_consensus_tree_registry):
        if e["name"] == name:
            _consensus_tree_registry.pop(i)
            _consensus_tree_cache.pop(e.get("uuid"), None)
            return True
    return False


def clear_all_consensus_tree_registry():
    """Drop the registry list and reset all naming counters."""
    _consensus_tree_registry.clear()
    _consensus_tree_registry_counters.clear()
