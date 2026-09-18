"""Summary-tree compute worker running in the persistent subprocess.

The parent extracts plain-Python descriptors from its in-memory database and
state. This worker loads the snapshot and dispatches to either the sampled MCC
path or the synthetic MrHIPSTR path without sharing parent memory.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


_SUMMARY_METHODS = frozenset({"mcc", "mrhipstr"})


def _normalise_summary_method(value: object) -> str:
    """Return the internal lower-case summary method or raise clearly."""
    method = "mrhipstr" if value is None else str(value).strip().lower()
    if method not in _SUMMARY_METHODS:
        supported = ", ".join(sorted(_SUMMARY_METHODS))
        raise ValueError(
            f"unsupported summary_method {value!r}; expected one of {supported}"
        )
    return method


def compute_consensus_tree_worker_entry(
    *,
    matched_records: List[Dict[str, Any]],
    source_distmat: str,
    snapshots_path: str,
    full_distmat_names: List[str],
    translate_maps: Dict[str, Dict[str, str]],
    source_file_paths: Dict[str, str],
    source_preambles: Dict[str, bytes],
    is_rooted: bool = True,
    summary_method: str = "mrhipstr",
) -> Dict[str, Any]:
    """Compute an MCC or MrHIPSTR summary and assemble its NEXUS bytes.

    MrHIPSTR records additionally require ``newick_offset`` and
    ``newick_length`` so the selected source Newicks can be streamed once.
    Omitting ``summary_method`` selects MrHIPSTR. MCC remains available by
    passing ``summary_method="mcc"`` explicitly.
    """
    import time

    worker_started_at = time.perf_counter()
    worker_started_wall_time = time.time()

    import numpy as np

    from .._worker_log import log as wlog
    from . import compute_consensus_tree_index, _inject_tree_annotation
    from ._canonical_remap import (
        _build_canonical_remaps,
        _substitute_newick_labels,
    )

    method = _normalise_summary_method(summary_method)
    if not matched_records:
        raise ValueError("matched_records is empty")
    if method == "mrhipstr" and not is_rooted:
        raise ValueError(
            "MrHIPSTR requires a rooted RF snapshot and rooted source trees"
        )

    wlog(
        "compute_consensus_tree_worker_entry: "
        f"method={method}, n_trees={len(matched_records)}, "
        f"source_distmat={source_distmat!r}, is_rooted={is_rooted}"
    )

    mrhipstr_profile = None
    if method == "mrhipstr":
        mrhipstr_profile = {
            "worker_started_wall_time": worker_started_wall_time,
            "worker_setup_seconds": (
                time.perf_counter() - worker_started_at
            ),
        }

    # Canonical remapping validates cross-source taxon alignment for both
    # methods. MCC uses the remap on its sampled line; MrHIPSTR parses each
    # source with its own Translate map and emits the canonical table.
    canonical_source = matched_records[0]["file_source"]
    unique_sources = list(
        dict.fromkeys(record["file_source"] for record in matched_records)
    )
    alignment_started_at = time.perf_counter()
    remaps, missing_taxa = _build_canonical_remaps(
        unique_sources,
        lambda source: translate_maps.get(source),
        canonical_source,
    )
    if mrhipstr_profile is not None:
        mrhipstr_profile["taxon_alignment_seconds"] = (
            time.perf_counter() - alignment_started_at
        )
    if missing_taxa:
        return {
            "nexus_bytes": None,
            "summary_method": method,
            "height_method": "sampled" if method == "mcc" else "mean",
            "consensus_tree_row": None,
            "summary_tree_name": None,
            "log_clade_credibility": None,
            "majority_clade_count": None,
            "counts": None,
            "cols_in_consensus_tree": None,
            "negative_branch_count": 0,
            "minimum_branch_length": None,
            "mrhipstr_statistics": None,
            "mrhipstr_profile": None,
            "missing_taxa": missing_taxa,
        }

    snapshot_started_at = time.perf_counter()
    wlog(f"loading summary-tree snapshot {snapshots_path!r}")
    snap = np.load(snapshots_path, allow_pickle=False)
    mrhipstr_result = None
    try:
        required_snapshot_keys = {"presence"}
        if method == "mrhipstr":
            required_snapshot_keys.update({"bipartition_bits", "leaf_names"})
        missing_snapshot_keys = required_snapshot_keys.difference(snap.files)
        if missing_snapshot_keys:
            missing_list = ", ".join(sorted(missing_snapshot_keys))
            raise ValueError(
                f"{method} snapshot is missing required arrays: {missing_list}"
            )
        presence = snap["presence"]
        bipartition_bits = (
            snap["bipartition_bits"] if method == "mrhipstr" else None
        )
        leaf_names = snap["leaf_names"] if method == "mrhipstr" else None
        name_to_idx = {
            name: index for index, name in enumerate(full_distmat_names)
        }
        try:
            selected_idx = [
                name_to_idx[record["name"]] for record in matched_records
            ]
        except KeyError as exc:
            raise KeyError(
                f"selected tree {exc.args[0]!r} is absent from "
                f"{source_distmat!r}"
            ) from exc

        if mrhipstr_profile is not None:
            mrhipstr_profile["snapshot_selection_seconds"] = (
                time.perf_counter() - snapshot_started_at
            )

        if method == "mrhipstr":
            mrhipstr_result = _compute_mrhipstr_result(
                matched_records=matched_records,
                selected_idx=selected_idx,
                presence=presence,
                bipartition_bits=bipartition_bits,
                leaf_names=leaf_names,
                translate_maps=translate_maps,
                source_file_paths=source_file_paths,
                canonical_source=canonical_source,
                canonical_preamble=source_preambles.get(canonical_source),
                profile=mrhipstr_profile,
                wlog=wlog,
            )
        else:
            # Preserve the original MCC objective and source-line
            # serialization.
            presence_sub = presence[selected_idx]
            consensus_tree_local, log_clade_cred = (
                compute_consensus_tree_index(presence_sub)
            )
            consensus_tree_record = matched_records[consensus_tree_local]
            counts = presence_sub.sum(axis=0).astype(np.int32)
            cols_in_consensus_tree = frozenset(
                np.flatnonzero(presence_sub[consensus_tree_local]).tolist()
            )
    finally:
        snap.close()

    if method == "mrhipstr":
        if mrhipstr_result is None or mrhipstr_profile is None:
            raise RuntimeError("MrHIPSTR worker produced no result")
        worker_total_seconds = time.perf_counter() - worker_started_at
        measured_seconds = sum(
            float(mrhipstr_profile.get(key, 0.0))
            for key in (
                "worker_setup_seconds",
                "taxon_alignment_seconds",
                "snapshot_selection_seconds",
                "clade_collection_seconds",
                "source_tree_ingestion_seconds",
                "topology_search_seconds",
                "nexus_serialization_seconds",
                "statistics_seconds",
            )
        )
        mrhipstr_profile.update(
            {
                "worker_unattributed_seconds": max(
                    0.0,
                    worker_total_seconds - measured_seconds,
                ),
                "worker_total_seconds": worker_total_seconds,
                "worker_finished_wall_time": time.time(),
            }
        )
        mrhipstr_result["mrhipstr_profile"] = mrhipstr_profile
        return mrhipstr_result

    consensus_tree_file_path = source_file_paths[
        consensus_tree_record["file_source"]
    ]
    with open(consensus_tree_file_path, "rb") as handle:
        handle.seek(int(consensus_tree_record["line_offset"]))
        line = handle.read(
            int(consensus_tree_record["line_length"])
        ).decode("utf-8")

    if not is_rooted:
        line = _midpoint_root_tree_line(line)

    line = _substitute_newick_labels(
        line,
        remaps.get(consensus_tree_record["file_source"], {}),
    )
    line = _inject_tree_annotation(
        line,
        "lnCladeCred",
        format(log_clade_cred, ".4f"),
    )

    canonical_preamble = (
        source_preambles.get(canonical_source)
        or b"#NEXUS\n\nbegin trees;\n"
    )
    body = line if line.endswith("\n") else line + "\n"
    nexus_bytes = canonical_preamble + body.encode("utf-8") + b"End;\n"

    return {
        "nexus_bytes": nexus_bytes,
        "summary_method": "mcc",
        "height_method": "sampled",
        "consensus_tree_row": consensus_tree_record,
        "summary_tree_name": str(consensus_tree_record["name"]),
        "log_clade_credibility": float(log_clade_cred),
        "majority_clade_count": None,
        "counts": counts,
        "cols_in_consensus_tree": cols_in_consensus_tree,
        "negative_branch_count": 0,
        "minimum_branch_length": None,
        "mrhipstr_statistics": None,
        "mrhipstr_profile": None,
        "missing_taxa": set(),
    }


def _compute_mrhipstr_result(
    *,
    matched_records: List[Dict[str, Any]],
    selected_idx: List[int],
    presence: Any,
    bipartition_bits: Any,
    leaf_names: Any,
    translate_maps: Dict[str, Dict[str, str]],
    source_file_paths: Dict[str, str],
    canonical_source: str,
    canonical_preamble: Optional[bytes],
    profile: Dict[str, float],
    wlog: Any,
) -> Dict[str, Any]:
    """Run the rooted synthetic MrHIPSTR pipeline for one selection."""
    import statistics
    import time

    from .mrhipstr import (
        SourceTreeRecord,
        compute_mrhipstr_topology,
        count_selected_clades,
        decode_rooted_clade_catalog,
        ingest_source_trees,
        serialize_mrhipstr_tree,
    )

    started = time.perf_counter()
    selected_counts = count_selected_clades(presence, selected_idx)
    catalog = decode_rooted_clade_catalog(
        bipartition_bits=bipartition_bits,
        leaf_names=leaf_names,
        active_columns=selected_counts.active_columns,
    )
    profile["clade_collection_seconds"] = time.perf_counter() - started
    wlog(
        "MrHIPSTR snapshot counts decoded: "
        f"active_clades={len(selected_counts.active_columns)}, "
        f"taxa={catalog.n_taxa}, "
        f"elapsed={profile['clade_collection_seconds']:.3f}s"
    )

    started = time.perf_counter()
    wlog("MrHIPSTR source-tree ingestion started")
    source_records = _iter_source_tree_records(
        matched_records,
        source_file_paths=source_file_paths,
        translate_maps=translate_maps,
        record_type=SourceTreeRecord,
    )
    try:
        source_summary = ingest_source_trees(
            source_records,
            catalog=catalog,
            snapshot_counts=selected_counts,
        )
    finally:
        source_records.close()
    profile["source_tree_ingestion_seconds"] = (
        time.perf_counter() - started
    )
    wlog(
        "MrHIPSTR source-tree ingestion finished: "
        f"observed_split_parents={len(source_summary.observed_splits)}, "
        f"elapsed={profile['source_tree_ingestion_seconds']:.3f}s"
    )

    started = time.perf_counter()
    wlog("MrHIPSTR dynamic programming started")
    topology = compute_mrhipstr_topology(
        catalog=catalog,
        snapshot_counts=selected_counts,
        source_summary=source_summary,
    )
    topology_seconds = time.perf_counter() - started
    profile["topology_search_seconds"] = topology_seconds
    wlog(
        "MrHIPSTR dynamic programming finished: "
        f"selected_clades={len(topology.selected_clades)}, "
        f"majority_clades={topology.majority_clade_count}, "
        f"elapsed={topology_seconds:.3f}s"
    )

    started = time.perf_counter()
    wlog("MrHIPSTR mean-height serialization started")
    exported = serialize_mrhipstr_tree(
        topology=topology,
        catalog=catalog,
        snapshot_counts=selected_counts,
        source_summary=source_summary,
        canonical_translate=translate_maps.get(canonical_source),
        nexus_preamble=canonical_preamble or None,
        tree_name="MrHIPSTR",
    )
    profile["nexus_serialization_seconds"] = (
        time.perf_counter() - started
    )
    profile["nexus_created_wall_time"] = time.time()
    wlog(
        "MrHIPSTR mean-height serialization finished: "
        f"negative_branches={exported.negative_branch_count}, "
        f"minimum_branch_length={exported.minimum_branch_length!r}, "
        f"elapsed={profile['nexus_serialization_seconds']:.3f}s"
    )

    statistics_started_at = time.perf_counter()
    # Match TreeAnnotator's reporting population: internal clades include the
    # implicit all-taxa root and exclude the always-present singleton tips.
    internal_clades = tuple(
        clade_bits
        for clade_bits in catalog.clades
        if clade_bits.bit_count() > 1
    )
    selected_internal_frequencies = [
        exported.clade_frequencies[clade_bits]
        for clade_bits in topology.selected_clades
        if clade_bits.bit_count() > 1
    ]
    if len(selected_internal_frequencies) != catalog.n_taxa - 1:
        raise RuntimeError(
            "MrHIPSTR topology has an unexpected number of internal clades"
        )
    mrhipstr_statistics = {
        "total_trees": int(source_summary.n_trees),
        "n_tips": int(catalog.n_taxa),
        "total_unique_clades": len(internal_clades),
        "clades_in_more_than_one_tree": sum(
            source_summary.observation_counts[clade_bits] > 1
            for clade_bits in internal_clades
        ),
        "topology_seconds": float(topology_seconds),
        "lowest_clade_credibility": float(
            min(selected_internal_frequencies)
        ),
        "mean_clade_credibility": float(
            statistics.fmean(selected_internal_frequencies)
        ),
        "median_clade_credibility": float(
            statistics.median(selected_internal_frequencies)
        ),
        "clades_with_credibility_1": sum(
            frequency == 1.0
            for frequency in selected_internal_frequencies
        ),
        "clades_with_credibility_gt_0_99": sum(
            frequency > 0.99
            for frequency in selected_internal_frequencies
        ),
        "clades_with_credibility_gt_0_95": sum(
            frequency > 0.95
            for frequency in selected_internal_frequencies
        ),
        "clades_with_credibility_gt_0_5": sum(
            frequency > 0.5
            for frequency in selected_internal_frequencies
        ),
    }
    profile["statistics_seconds"] = (
        time.perf_counter() - statistics_started_at
    )

    return {
        "nexus_bytes": exported.nexus_bytes,
        "summary_method": "mrhipstr",
        "height_method": "mean",
        "consensus_tree_row": None,
        "summary_tree_name": "MrHIPSTR",
        "log_clade_credibility": float(topology.log_clade_credibility),
        "majority_clade_count": int(topology.majority_clade_count),
        "counts": selected_counts.counts,
        "cols_in_consensus_tree": topology.cols_in_consensus_tree,
        "negative_branch_count": int(exported.negative_branch_count),
        "minimum_branch_length": (
            None
            if exported.minimum_branch_length is None
            else float(exported.minimum_branch_length)
        ),
        "mrhipstr_statistics": mrhipstr_statistics,
        "missing_taxa": set(),
    }


def _iter_source_tree_records(
    matched_records: List[Dict[str, Any]],
    *,
    source_file_paths: Dict[str, str],
    translate_maps: Dict[str, Dict[str, str]],
    record_type: Any,
):
    """Yield selected Newick bodies while keeping one handle per source."""
    file_handles: Dict[str, Any] = {}
    try:
        for record in matched_records:
            source = record["file_source"]
            handle = file_handles.get(source)
            if handle is None:
                try:
                    path = source_file_paths[source]
                except KeyError as exc:
                    raise KeyError(
                        f"no source file path is available for {source!r}"
                    ) from exc
                handle = open(path, "rb")
                file_handles[source] = handle

            try:
                offset = int(record["newick_offset"])
                length = int(record["newick_length"])
            except KeyError as exc:
                raise KeyError(
                    "MrHIPSTR matched records require newick_offset and "
                    "newick_length"
                ) from exc
            handle.seek(offset)
            newick_bytes = handle.read(length)
            if len(newick_bytes) != length:
                raise ValueError(
                    f"short Newick read for {record['name']!r}: expected "
                    f"{length} bytes, got {len(newick_bytes)}"
                )
            yield record_type(
                name=str(record["name"]),
                newick=newick_bytes.decode("utf-8"),
                translate=translate_maps.get(source),
            )
    finally:
        for handle in file_handles.values():
            try:
                handle.close()
            except OSError:
                pass


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
        # tree than to bail on the whole consensus tree compute.
        rooted_body = body_stripped

    suffix = "\n" if trailing_newline else ""
    return f"{header} = [&R] {rooted_body}{suffix}"
