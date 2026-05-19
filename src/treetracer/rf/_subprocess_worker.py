"""RF compute worker — runs in a subprocess, not a thread.

Why subprocess: ``rapidtrees.pairwise_rf_with_snapshots_from_newick_iter``
holds the GIL during its iterator-consumption phase (reading newick
strings out of a Python iterator one at a time). For thousands of trees
that's a few seconds of contiguous GIL ownership, which is enough for
macOS to show the spinning-beach-ball watchdog on the main GUI thread.

Threads can't fix that — they share the GIL. So we re-invoke the app's
own launcher in "worker mode" via ``TREETRACER_WORKER_MODE=compute_rf``,
which short-circuits in ``__init__.py`` before any GUI imports and
ultimately calls ``compute_rf_worker_entry`` below.

Architecture note: this function INTENTIONALLY does not import the
parent's in-memory tree DB. The parent passes pre-collected metadata
(byte offsets into the original ``.trees`` files, translate maps, etc.)
and we do the disk reads ourselves. That keeps the parent's prep work
to ~50 ms of pandas slicing, well below the macOS beach-ball threshold.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List


def compute_rf_worker_entry(
    *,
    tree_descriptors: List[Dict[str, Any]],
    source_file_paths: Dict[str, str],
    translate_maps: Dict[str, Dict[str, str]],
    save_path: str,
    rf_name: str,
    is_rooted: bool = True,
) -> Dict[str, Any]:
    """Run the full RF pipeline inside a subprocess.

    Args:
        tree_descriptors: per-tree metadata in MCMC iteration order. Each
            entry has keys ``name`` (full ``group_name/tree_name``),
            ``newick_offset``, ``newick_length``, ``file_source``,
            ``group_name``.
        source_file_paths: ``file_source`` → absolute path of the
            original ``.trees`` file on disk. We re-open these and
            seek + read each newick.
        translate_maps: ``file_source`` → NEXUS translate map dict
            (or ``None`` if the file had no Translate block).
        save_path: where ``compute_rf`` writes the uint16 RF matrix
            (a corresponding ``_snapshots.npz`` is written by its side).
        rf_name: human-readable label (only used in return value for
            the caller's bookkeeping).
        is_rooted: whether the input trees are rooted. Passes through
            to ``rapidtrees.pairwise_rf_with_snapshots_from_newick_iter``'s
            ``rooted=`` arg:
              * True  → presence-matrix columns are rooted clades
                (subtree-from-root identity).
              * False → presence-matrix columns are bipartitions
                (split-induced unordered pairs of taxon sets).
            Defaults to True for backward compatibility with the
            previous always-rooted behaviour.

    Returns a dict consumed by ``poll_completion`` in callbacks/compute.py.
    """
    t0 = time.time()

    # ── Read newicks from the original .trees files ────────────────────
    # Mirrors ``TreeManagerPandas._read_newick`` exactly: open binary,
    # seek to offset, read N bytes, decode UTF-8. One handle per source
    # file kept open for the duration so we're not paying the open()
    # cost per tree.
    file_handles: Dict[str, Any] = {}
    try:
        for fs, path in source_file_paths.items():
            file_handles[fs] = open(path, "rb")

        names: List[str] = []
        newicks: List[str] = []
        for desc in tree_descriptors:
            fh = file_handles[desc["file_source"]]
            fh.seek(desc["newick_offset"])
            newick = fh.read(desc["newick_length"]).decode("utf-8")
            names.append(desc["name"])
            newicks.append(newick)
    finally:
        for fh in file_handles.values():
            try:
                fh.close()
            except OSError:
                pass

    # ── Convert per-source translate maps into the (list, indices) shape
    # rapidtrees expects ───────────────────────────────────────────────
    file_to_map_idx: Dict[str, int] = {}
    map_list: List[Dict[str, str]] = []
    for fname, tmap in translate_maps.items():
        if tmap is not None:
            file_to_map_idx[fname] = len(map_list)
            map_list.append(tmap)
    map_indices = [
        file_to_map_idx.get(d["file_source"], 0) for d in tree_descriptors
    ]

    # ── File breakdown / groups-per-file for the parent's registry entry
    file_breakdown: Dict[str, int] = {}
    groups_per_file: Dict[str, set] = {}
    for d in tree_descriptors:
        fs = d["file_source"]
        gn = d["group_name"]
        file_breakdown[fs] = file_breakdown.get(fs, 0) + 1
        groups_per_file.setdefault(fs, set()).add(gn)
    groups_per_file_sorted = {k: sorted(v) for k, v in groups_per_file.items()}

    # ── Run rapidtrees (this is the actual compute that we wanted to
    # isolate from the parent process's GIL) ───────────────────────────
    from ._worker import compute_rf
    result_names, compute_elapsed = compute_rf(
        names, newicks, map_list, map_indices, save_path,
        is_rooted=is_rooted,
    )

    return {
        "result_names": list(result_names),
        "compute_elapsed": compute_elapsed,
        "total_elapsed": time.time() - t0,
        "file_breakdown": file_breakdown,
        "groups_per_file": groups_per_file_sorted,
        "rf_name": rf_name,
    }
