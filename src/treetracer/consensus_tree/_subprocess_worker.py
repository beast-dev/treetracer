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


def _compute_mcc_statistics(
    *,
    counts: Any,
    cols_in_consensus_tree: frozenset[int],
    n_trees: int,
    best_tree_number: int,
    is_rooted: bool,
    sparse_snapshot: Any,
) -> Dict[str, Any]:
    """Summarize non-trivial clade credibilities for the sampled MCC tree."""
    import statistics

    import numpy as np

    counts_array = np.asarray(counts, dtype=np.int64)
    if counts_array.ndim != 1:
        raise ValueError("MCC clade counts must be one-dimensional")
    if n_trees <= 0:
        raise ValueError("MCC statistics require at least one tree")

    packed_clades = np.asarray(sparse_snapshot.packed_clades)
    n_taxa = int(sparse_snapshot.n_taxa)
    if packed_clades.shape[0] != len(counts_array):
        raise ValueError(
            "sparse MCC clade catalog disagrees with clade counts"
        )

    # Count packed bits in bounded-width blocks. This avoids expanding the
    # complete clade-by-taxon catalog merely to omit trivial tip clades.
    popcount = np.asarray(
        [value.bit_count() for value in range(256)],
        dtype=np.uint8,
    )
    clade_sizes = np.zeros(len(counts_array), dtype=np.uint32)
    for start in range(0, packed_clades.shape[1], 64):
        clade_sizes += popcount[
            packed_clades[:, start : start + 64]
        ].sum(axis=1, dtype=np.uint32)

    if is_rooted:
        reportable = clade_sizes > 1
    else:
        reportable = (clade_sizes > 1) & (clade_sizes < n_taxa - 1)

    winning_columns = np.fromiter(
        sorted(cols_in_consensus_tree),
        dtype=np.intp,
        count=len(cols_in_consensus_tree),
    )
    if len(winning_columns) and (
        winning_columns[0] < 0
        or winning_columns[-1] >= len(counts_array)
    ):
        raise IndexError("MCC tree references a clade outside the catalog")
    reportable_columns = winning_columns[reportable[winning_columns]]
    tree_clade_counts = counts_array[reportable_columns]
    majority_clades_in_all_trees = int(
        np.count_nonzero(reportable & (counts_array * 2 > n_trees))
    )
    if is_rooted:
        # RapidTrees deliberately omits the uninformative all-taxa root.
        # TreeAnnotator includes it in its individual-clade statistics, where
        # its credibility is always 1.0 and its log-score contribution is 0.
        tree_clade_counts = np.append(tree_clade_counts, n_trees)
        majority_clades_in_all_trees += 1
    frequencies = tree_clade_counts.astype(np.float64) / float(n_trees)

    if len(frequencies):
        lowest = float(np.min(frequencies))
        mean = float(statistics.fmean(frequencies))
        median = float(statistics.median(frequencies))
    else:
        lowest = mean = median = None

    return {
        "total_trees": int(n_trees),
        "best_tree_number": int(best_tree_number),
        "number_of_clades": int(len(tree_clade_counts)),
        "lowest_clade_credibility": lowest,
        "mean_clade_credibility": mean,
        "median_clade_credibility": median,
        "clades_with_credibility_1": int(
            np.count_nonzero(tree_clade_counts == n_trees)
        ),
        "clades_with_credibility_gt_0_99": int(
            np.count_nonzero(tree_clade_counts * 100 > n_trees * 99)
        ),
        "clades_with_credibility_gt_0_95": int(
            np.count_nonzero(tree_clade_counts * 20 > n_trees * 19)
        ),
        "clades_with_credibility_gt_0_5": int(
            np.count_nonzero(tree_clade_counts * 2 > n_trees)
        ),
        "majority_clades_in_all_trees": majority_clades_in_all_trees,
    }


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

    MrHIPSTR prefers persisted RapidTrees rooted facts. Generic CSR snapshots
    still use ``newick_offset`` and ``newick_length`` to stream selected source
    Newicks once for heights and observed splits. Omitting ``summary_method``
    selects MrHIPSTR. MCC remains available by passing
    ``summary_method="mcc"`` explicitly.
    """
    import time

    worker_started_at = time.perf_counter()
    worker_started_wall_time = time.time()

    import numpy as np

    from .._worker_log import log as wlog
    from . import (
        _inject_tree_annotation,
        compute_consensus_tree_index_sparse,
    )
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
            "mcc_statistics": None,
            "mrhipstr_statistics": None,
            "mrhipstr_profile": None,
            "snapshot_input_mode": "not_loaded",
            "missing_taxa": missing_taxa,
        }

    snapshot_started_at = time.perf_counter()
    wlog(f"loading summary-tree snapshot {snapshots_path!r}")
    snap = np.load(snapshots_path, allow_pickle=False)
    mrhipstr_result = None
    try:
        rooted_facts = None
        from ..rf.sparse_snapshots import (
            snapshot_has_generic_sparse_snapshot,
            sparse_snapshot_from_npz,
        )

        try:
            sparse_presence = sparse_snapshot_from_npz(snap)
        except KeyError as exc:
            raise ValueError(
                f"{method} requires a sparse RF snapshot; recompute the RF "
                "matrix with RapidTrees 0.9.1 or newer"
            ) from exc
        sparse_presence_mode = (
            "sparse"
            if snapshot_has_generic_sparse_snapshot(snap)
            else "rooted_facts"
        )
        if sparse_presence.tree_names != tuple(full_distmat_names):
            raise ValueError(
                "persisted sparse-snapshot tree names disagree with "
                "the RF registry ordering"
            )
        if sparse_presence.rooted != bool(is_rooted):
            raise ValueError(
                "persisted sparse-snapshot rooting mode disagrees "
                "with the RF registry"
            )

        if method == "mrhipstr":
            from ..rf.rooted_facts import (
                rooted_facts_from_npz,
                snapshot_has_rooted_facts,
            )

            if snapshot_has_rooted_facts(snap):
                rooted_facts = rooted_facts_from_npz(snap)
                if rooted_facts.tree_names != tuple(full_distmat_names):
                    raise ValueError(
                        "persisted rooted-facts tree names disagree with the "
                        "RF registry ordering"
                    )
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
            snapshot_input_mode = (
                "rooted_facts"
                if rooted_facts is not None
                else sparse_presence_mode
            )
            mrhipstr_result = _compute_mrhipstr_result(
                matched_records=matched_records,
                selected_idx=selected_idx,
                translate_maps=translate_maps,
                source_file_paths=source_file_paths,
                canonical_source=canonical_source,
                canonical_preamble=source_preambles.get(canonical_source),
                profile=mrhipstr_profile,
                wlog=wlog,
                rooted_facts=rooted_facts,
                sparse_snapshot=sparse_presence,
            )
        else:
            # Preserve the original MCC objective and source-line
            # serialization.
            from ..rf.sparse_snapshots import count_sparse_columns

            counts = count_sparse_columns(
                sparse_presence,
                selected_idx,
            ).astype(np.int32)
            consensus_tree_local, log_clade_cred = (
                compute_consensus_tree_index_sparse(
                    sparse_presence,
                    selected_idx,
                    counts=counts,
                )
            )
            winning_row = selected_idx[consensus_tree_local]
            cols_in_consensus_tree = frozenset(
                int(column)
                for column in sparse_presence.row_columns(winning_row)
            )
            snapshot_input_mode = sparse_presence_mode
            wlog("MCC scoring used sparse clade-presence rows")
            consensus_tree_record = matched_records[consensus_tree_local]
            mcc_statistics = _compute_mcc_statistics(
                counts=counts,
                cols_in_consensus_tree=cols_in_consensus_tree,
                n_trees=len(selected_idx),
                # Match TreeAnnotator's 1-based source-order ordinal rather
                # than renumbering an arbitrary selected subset from one.
                best_tree_number=selected_idx[consensus_tree_local] + 1,
                is_rooted=is_rooted,
                sparse_snapshot=sparse_presence,
            )
            wlog(
                "MCC credibility statistics calculated: "
                f"tree_number={mcc_statistics['best_tree_number']}, "
                f"clades={mcc_statistics['number_of_clades']}, "
                "majority_clades_in_all_trees="
                f"{mcc_statistics['majority_clades_in_all_trees']}"
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
        mrhipstr_result["snapshot_input_mode"] = snapshot_input_mode
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
        format(log_clade_cred, ".17g"),
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
        "mcc_statistics": mcc_statistics,
        "mrhipstr_statistics": None,
        "mrhipstr_profile": None,
        "snapshot_input_mode": snapshot_input_mode,
        "missing_taxa": set(),
    }


def _compute_mrhipstr_result(
    *,
    matched_records: List[Dict[str, Any]],
    selected_idx: List[int],
    translate_maps: Dict[str, Dict[str, str]],
    source_file_paths: Dict[str, str],
    canonical_source: str,
    canonical_preamble: Optional[bytes],
    profile: Dict[str, Any],
    wlog: Any,
    rooted_facts: Any = None,
    sparse_snapshot: Any = None,
) -> Dict[str, Any]:
    """Run MrHIPSTR from rooted facts, falling back to source Newicks."""
    import statistics
    import time

    from .mrhipstr import (
        SourceTreeRecord,
        compute_mrhipstr_topology,
        ingest_source_trees,
        serialize_mrhipstr_tree,
    )
    from .rooted_facts import (
        count_selected_clades_from_rooted_facts,
        decode_rooted_clade_catalog_from_rooted_facts,
        source_summary_from_rooted_facts,
    )
    from .sparse_inputs import (
        count_selected_clades_from_sparse_snapshot,
        decode_rooted_clade_catalog_from_sparse_snapshot,
    )

    started = time.perf_counter()
    if rooted_facts is not None:
        selected_counts = count_selected_clades_from_rooted_facts(
            rooted_facts,
            selected_idx,
        )
        catalog = decode_rooted_clade_catalog_from_rooted_facts(
            rooted_facts,
            selected_counts,
        )
        input_mode = "rooted_facts"
    else:
        selected_counts = count_selected_clades_from_sparse_snapshot(
            sparse_snapshot,
            selected_idx,
        )
        catalog = decode_rooted_clade_catalog_from_sparse_snapshot(
            sparse_snapshot,
            selected_counts,
        )
        input_mode = "sparse_snapshot"
    profile["input_mode"] = input_mode
    profile["clade_collection_seconds"] = time.perf_counter() - started
    wlog(
        f"MrHIPSTR clades decoded from {input_mode}: "
        f"active_clades={len(selected_counts.active_columns)}, "
        f"taxa={catalog.n_taxa}, "
        f"elapsed={profile['clade_collection_seconds']:.3f}s"
    )

    started = time.perf_counter()
    if rooted_facts is None:
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
        ingestion_label = "source-tree ingestion"
    else:
        wlog("MrHIPSTR rooted-facts aggregation started")
        source_summary = source_summary_from_rooted_facts(
            rooted_facts,
            selected_idx,
            catalog=catalog,
            snapshot_counts=selected_counts,
        )
        ingestion_label = "rooted-facts aggregation"
    profile["source_tree_ingestion_seconds"] = (
        time.perf_counter() - started
    )
    wlog(
        f"MrHIPSTR {ingestion_label} finished: "
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
        "mcc_statistics": None,
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
