"""MCC compute worker — runs in the persistent worker subprocess.

Mirrors the shape of ``rf/_subprocess_worker.py``: the parent extracts
plain-Python descriptors from its in-memory DB + state, pickles them
to the worker, and the worker does the heavy lifting (loading the
big presence matrix from disk, argmax, newick read, label substitution,
NEXUS assembly).

The parent owns the DB; the worker owns the computation. They share
nothing in memory.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def compute_mcc_worker_entry(
    *,
    matched_records: List[Dict[str, Any]],
    source_distmat: str,
    snapshots_path: str,
    full_distmat_names: List[str],
    translate_maps: Dict[str, Dict[str, str]],
    source_file_paths: Dict[str, str],
    source_preambles: Dict[str, bytes],
    is_rooted: bool = True,
) -> Dict[str, Any]:
    """Compute the MCC tree for a selection and assemble its NEXUS bytes.

    Args:
        matched_records: list of dicts, one per selected tree, in the
            order the parent's matched DataFrame had them. Each dict
            has keys ``name``, ``file_source``, ``line_offset``,
            ``line_length``, ``metadata``.
        source_distmat: distmat name (used for logging only).
        snapshots_path: filesystem path to ``<distmat>_snapshots.npz``
            (parent gets it from ``state.get_snapshots_path``).
        full_distmat_names: ordered list of names for the presence
            matrix's rows (parent gets it from
            ``state.get_distmat_names``).
        translate_maps: per-source NEXUS translate map dicts.
        source_file_paths: per-source absolute path to the original
            ``.trees`` file on disk.
        source_preambles: per-source NEXUS preamble bytes (everything
            up to and including the Translate block).

    Returns a dict the parent's polling callback unpacks. Same shape as
    ``assemble_mcc_nexus`` returned, with ``mcc_row`` flattened to a
    plain dict so it pickles cleanly.
    """
    import numpy as np

    from ._canonical_remap import (
        _build_canonical_remaps, _substitute_newick_labels,
    )
    # Lazy import to keep startup cost out of the path until first use.
    from . import compute_mcc_index, _inject_tree_annotation

    if not matched_records:
        raise ValueError("matched_records is empty")

    # ── Canonical remap (cross-source taxon-int alignment) ────────────
    canonical_source = matched_records[0]["file_source"]
    unique_sources = []
    seen = set()
    for rec in matched_records:
        fs = rec["file_source"]
        if fs not in seen:
            seen.add(fs)
            unique_sources.append(fs)

    remaps, missing_taxa = _build_canonical_remaps(
        unique_sources, lambda s: translate_maps.get(s), canonical_source,
    )
    if missing_taxa:
        return {
            "nexus_bytes": None,
            "mcc_row": None,
            "log_clade_credibility": None,
            "counts": None,
            "cols_in_mcc": None,
            "missing_taxa": missing_taxa,
        }

    # ── Load presence matrix + index by name ───────────────────────────
    snap = np.load(snapshots_path, allow_pickle=False)
    presence = snap["presence"]                  # (N_total, n_splits) uint8
    name_to_idx = {n: i for i, n in enumerate(full_distmat_names)}

    selected_idx = [name_to_idx[rec["name"]] for rec in matched_records]
    presence_sub = presence[selected_idx]         # (n_sel, n_splits)
    mcc_local, log_clade_cred = compute_mcc_index(presence_sub)
    mcc_record = matched_records[mcc_local]

    counts = presence_sub.sum(axis=0).astype(np.int32)
    cols_in_mcc = frozenset(np.flatnonzero(presence_sub[mcc_local]).tolist())

    # ── Read the MCC's newick from disk ────────────────────────────────
    mcc_file_path = source_file_paths[mcc_record["file_source"]]
    with open(mcc_file_path, "rb") as fh:
        fh.seek(int(mcc_record["line_offset"]))
        line = fh.read(int(mcc_record["line_length"])).decode("utf-8")

    # If the source distmat was computed in unrooted mode, the chosen
    # tree's newick has an arbitrary root inherited from whatever the
    # MCMC writer chose. Midpoint-root it so PearTree displays a
    # sensible rooting and the tanglegram code (which assumes rooted
    # trees) sees a consistent root across MCMCs from the same posterior.
    if not is_rooted:
        line = _midpoint_root_tree_line(line)

    line = _substitute_newick_labels(
        line, remaps.get(mcc_record["file_source"], {}),
    )
    line = _inject_tree_annotation(
        line, "lnCladeCred", format(log_clade_cred, ".4f"),
    )

    # ── Assemble final NEXUS bytes ─────────────────────────────────────
    canonical_preamble = (
        source_preambles.get(canonical_source)
        or b"#NEXUS\n\nbegin trees;\n"
    )
    body = line if line.endswith("\n") else line + "\n"
    nexus_bytes = canonical_preamble + body.encode("utf-8") + b"End;\n"

    return {
        "nexus_bytes": nexus_bytes,
        "mcc_row": mcc_record,
        "log_clade_credibility": float(log_clade_cred),
        "counts": counts,
        "cols_in_mcc": cols_in_mcc,
        "missing_taxa": set(),
    }


def _midpoint_root_tree_line(line: str) -> str:
    """Take a NEXUS ``tree NAME [&…] = [&U] (…);`` line, midpoint-root
    the newick body, and emit ``tree NAME [&…] = [&R] (…);``.

    Preserves the tree name + any pre-``=`` annotations. Drops the
    ``[&U]`` flag (the tree is rooted now) and emits ``[&R]`` instead.
    """
    from ._midpoint import midpoint_root_newick

    # Split at the FIRST `=` to separate the name+meta header from the
    # newick body. BEAST inline annotations on the left side use `=`
    # inside their brackets (e.g. ``[&lnP=…]``), so we have to find
    # the `=` that ISN'T inside brackets. The DB stores tree lines with
    # the convention ``<name> = <body>`` separated by space-equals-space,
    # so look for that first.
    eq_idx = line.find(" = ")
    if eq_idx < 0:
        # Fall back: find first `=` not inside `[...]`.
        depth = 0
        for i, ch in enumerate(line):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            elif ch == "=" and depth == 0:
                eq_idx = i
                break
        if eq_idx < 0:
            return line  # malformed; return unchanged
        header = line[:eq_idx].rstrip()
        body = line[eq_idx + 1:].lstrip()
    else:
        header = line[:eq_idx]
        body = line[eq_idx + 3:].lstrip()

    # Strip a leading ``[&R]`` / ``[&U]`` flag from the body. We'll
    # emit ``[&R]`` ourselves on the way out.
    if body.startswith("[&R]"):
        body = body[4:].lstrip()
    elif body.startswith("[&U]"):
        body = body[4:].lstrip()

    # Strip a trailing newline if present so emit doesn't double up.
    trailing_newline = body.endswith("\n")
    body_stripped = body.rstrip("\n").rstrip()

    try:
        rooted_body = midpoint_root_newick(body_stripped)
    except Exception:
        # If midpoint rooting fails for any reason, fall through with
        # the original body — better to display an arbitrarily-rooted
        # tree than to bail on the whole MCC compute.
        rooted_body = body_stripped

    suffix = "\n" if trailing_newline else ""
    return f"{header} = [&R] {rooted_body}{suffix}"
