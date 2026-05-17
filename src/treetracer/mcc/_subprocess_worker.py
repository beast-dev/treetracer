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
