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

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def compute_rf_worker_entry(
    *,
    tree_descriptors: List[Dict[str, Any]],
    source_file_paths: Dict[str, str],
    translate_maps: Dict[str, Dict[str, str]],
    save_path: str,
    rf_name: str,
    is_rooted: bool = True,
    progress_path: Optional[str] = None,
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

    Returns a dict finalized by the managed job wrapper; the central
    reconciler later delivers only its small terminal payload to the browser.
    """
    t0 = time.time()

    # Checkpoint logging into the shared worker log
    # (``treetracer._worker_log``) so a hung RF compute can be
    # diagnosed by reading one file — see that module's docstring.
    from .._worker_log import log as wlog
    wlog(
        f"compute_rf_worker_entry: n_trees={len(tree_descriptors)}, "
        f"n_source_files={len(source_file_paths)}, "
        f"is_rooted={is_rooted}, save_path={save_path!r}"
    )

    # ── Read newicks from the original .trees files ────────────────────
    # Mirrors ``TreeManagerPandas._read_newick`` exactly: open binary,
    # seek to offset, read N bytes, decode UTF-8. One handle per source
    # file kept open for the duration so we're not paying the open()
    # cost per tree.
    wlog(f"opening {len(source_file_paths)} source .trees files")
    file_handles: Dict[str, Any] = {}
    try:
        for fs, path in source_file_paths.items():
            wlog(f"  opening source file: {path!r}")
            file_handles[fs] = open(path, "rb")

        wlog(f"reading {len(tree_descriptors)} newicks from disk")
        names: List[str] = []
        newicks: List[str] = []
        for desc in tree_descriptors:
            fh = file_handles[desc["file_source"]]
            fh.seek(desc["newick_offset"])
            newick = fh.read(desc["newick_length"]).decode("utf-8")
            names.append(desc["name"])
            newicks.append(newick)
        wlog(f"finished reading newicks; total bytes={sum(len(n) for n in newicks)}")
    finally:
        for fh in file_handles.values():
            try:
                fh.close()
            except OSError:
                pass

    # ── Convert per-source translate maps into the (list, indices) shape
    # rapidtrees expects ───────────────────────────────────────────────
    wlog("building translate-map lookup tables")
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
    wlog("importing rapidtrees wrapper (._worker.compute_rf)")
    from ._worker import compute_rf

    # ── Progress reporting (optional) ──────────────────────────────────
    # When ``progress_path`` is set, share a ``ProgressCounter`` with the
    # rayon workers via the new rapidtrees 0.6 API, and run a daemon
    # thread that mirrors the counter state into a small JSON sidecar
    # file. The parent's central reconciler samples it every 250ms to drive a
    # progress bar — no IPC changes needed.
    counter = None
    stop_event: Optional[threading.Event] = None
    writer: Optional[threading.Thread] = None
    if progress_path:
        import rapidtrees
        counter = rapidtrees.ProgressCounter()
        stop_event = threading.Event()

        def _writer_loop():
            pp = Path(progress_path)
            while not stop_event.is_set():
                val = counter.value()
                tot = counter.total()
                frac = (val / tot) if tot > 0 else 0.0
                # Once the pairwise loop hits 100%, rapidtrees is done
                # but the worker still has to write the .npy + snapshot
                # .npz to disk (the .npz can be 100MB+ for big
                # bipartition matrices — measurably ~1 s for 4k trees).
                # Flip the phase so the UI shows "Finalizing…" instead
                # of leaving the bar sitting at a static 100%.
                phase = "finalizing" if (tot > 0 and val >= tot) else "computing"
                try:
                    pp.write_text(json.dumps({
                        "value": val,
                        "total": tot,
                        "fraction": min(max(frac, 0.0), 1.0),
                        "phase": phase,
                    }))
                except OSError:
                    # Transient disk hiccup — try again next tick.
                    pass
                # No early-break when fraction hits 1.0 — keep emitting
                # the "finalizing" phase until the parent sets
                # stop_event (which happens after compute_rf returns
                # from its disk-save work).
                if stop_event.wait(0.1):
                    break

        writer = threading.Thread(
            target=_writer_loop, daemon=True, name="rf-progress-writer",
        )
        writer.start()
        wlog(f"progress writer started for {progress_path!r}")

    wlog(
        f"calling compute_rf: n_trees={len(names)}, "
        f"n_translate_maps={len(map_list)}, is_rooted={is_rooted}, "
        f"progress={'on' if counter is not None else 'off'}"
    )
    compute_t0 = time.time()
    try:
        result_names, compute_elapsed = compute_rf(
            names, newicks, map_list, map_indices, save_path,
            is_rooted=is_rooted, progress=counter,
        )
    finally:
        if stop_event is not None:
            stop_event.set()
        if writer is not None:
            writer.join(timeout=1.0)
        if progress_path:
            try:
                Path(progress_path).unlink()
            except OSError:
                pass
            wlog("progress writer stopped, sidecar file removed")
    wlog(
        f"compute_rf returned in {time.time() - compute_t0:.3f}s "
        f"(rapidtrees-reported elapsed={compute_elapsed:.3f}s)"
    )

    return {
        "result_names": list(result_names),
        "compute_elapsed": compute_elapsed,
        "total_elapsed": time.time() - t0,
        "file_breakdown": file_breakdown,
        "groups_per_file": groups_per_file_sorted,
        "rf_name": rf_name,
    }
