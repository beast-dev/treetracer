"""Managed consensus-tree dispatch, publication, and terminal presentation.

Both the Between-run (``treespace``) and Within-run (``within_run``)
tabs have a "View consensus tree" button. They used to each call
``consensus_tree.assemble_consensus_tree_nexus`` synchronously inside the click callback,
which blocked the Dash request thread for hundreds of ms (or a few
seconds on big presence matrices) with no visual feedback.

This module factors out the shared pieces so both tabs:

* Click → enqueue a consensus tree compute job on the persistent worker, show a
  loading overlay over the active tab, disable the View consensus tree button.
* Wait → the shared reconciler publishes a sticky terminal event. The success
  finalizer caches the NEXUS bytes and registers the tree exactly once; this
  module only renders a matching event and routes it to the originating tab.

The per-tab callbacks ``view_consensus_tree`` in ``treespace.py`` and
``within_run.py`` shrink to ~30 lines each — they're only responsible
for validating input, building ``matched_records`` + tab-specific
metadata, and calling ``submit_consensus_tree_job`` here.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from functools import partial
from typing import Any

import dash_mantine_components as dmc
from dash import Input, Output, callback, no_update

from .. import state as _state
from ..background_jobs import JobRef, JobState, job_manager
from ..logger import add_log
from ..consensus_tree import extract_log_posterior
from . import persistent_worker
from .compute import _get_executor
from .job_reconcile import (
    terminal_delivery_marker,
    terminal_event_for_job,
)


_TREESPACE_TARGET = "treespace-view-consensus-tree-store"
_WITHIN_RUN_TARGET = "within-run-view-consensus-tree-store"
_STORE_TARGETS = frozenset({_TREESPACE_TARGET, _WITHIN_RUN_TARGET})


@dataclass(frozen=True, slots=True)
class _ConsensusFinalizationContext:
    """Parent-only state needed for exactly-once consensus publication."""

    source_distmat: str
    mode: str
    run: str | None
    selection: list[Any]
    tree_names: tuple[str, ...]
    coord_by_tree_name: dict[str, tuple[Any, int]]
    summary_method: str = "mcc"
    is_rooted: bool = True
    click_started_at: float | None = None
    click_started_wall_time: float | None = None
    submit_started_at: float | None = None
    dispatch_started_at: float | None = None
    dispatch_wall_time: float | None = None


class ConsensusTaxaAlignmentError(ValueError):
    """Raised when selected source files cannot share one Translate table."""


def _normalise_summary_method(value: object) -> str:
    """Normalize method metadata, treating missing legacy values as MCC."""
    method = "mcc" if value is None else str(value).strip().lower()
    if method not in {"mcc", "mrhipstr"}:
        raise ValueError(
            f"unsupported summary_method {value!r}; "
            "expected 'mcc' or 'mrhipstr'"
        )
    return method


def _summary_method_label(value: object) -> str:
    """Return the user-facing method label used in logs and notifications."""
    try:
        method = _normalise_summary_method(value)
    except ValueError:
        return "Summary tree"
    return "MrHIPSTR" if method == "mrhipstr" else "MCC"


def _format_mrhipstr_statistics(
    statistics: object,
    *,
    log_clade_credibility: float,
) -> str:
    """Format the TreeAnnotator-style MrHIPSTR console report."""
    if not isinstance(statistics, dict):
        raise TypeError("MrHIPSTR worker result is missing its statistics")
    try:
        total_trees = int(statistics["total_trees"])
        n_tips = int(statistics["n_tips"])
        total_unique_clades = int(statistics["total_unique_clades"])
        recurring_clades = int(
            statistics["clades_in_more_than_one_tree"]
        )
        topology_seconds = float(statistics["topology_seconds"])
        lowest = float(statistics["lowest_clade_credibility"])
        mean = float(statistics["mean_clade_credibility"])
        median = float(statistics["median_clade_credibility"])
        credibility_1 = int(statistics["clades_with_credibility_1"])
        credibility_0_99 = int(
            statistics["clades_with_credibility_gt_0_99"]
        )
        credibility_0_95 = int(
            statistics["clades_with_credibility_gt_0_95"]
        )
        credibility_0_5 = int(
            statistics["clades_with_credibility_gt_0_5"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TypeError(
            "MrHIPSTR worker returned malformed statistics"
        ) from exc

    return "\n".join(
        [
            f"Total trees read: {total_trees}",
            f"Size of trees: {n_tips} tips",
            f"Total unique clades: {total_unique_clades}",
            "Total clades in more than one tree: "
            f"{recurring_clades}",
            "",
            (
                "Finding majority rule highest independent posterior "
                "subtree reconstruction (MrHIPSTR) tree..."
            ),
            f"[{topology_seconds:.3f} secs]",
            "",
            "MrHIPSTR tree's log clade credibility: "
            f"{log_clade_credibility:.4f}",
            f"Lowest individual clade credibility: {lowest:.4f}",
            f"Mean individual clade credibility: {mean:.4f}",
            f"Median individual clade credibility: {median:.4f}",
            "Number of clades with credibility 1.0: "
            f"{credibility_1}",
            "Number of clades with credibility > 0.99: "
            f"{credibility_0_99}",
            "Number of clades with credibility > 0.95: "
            f"{credibility_0_95}",
            "Number of clades with credibility > 0.5: "
            f"{credibility_0_5}",
        ]
    )


def _format_mcc_statistics(
    statistics: object,
    *,
    tree_name: str,
    log_clade_credibility: float,
) -> str:
    """Format the selected-tree MCC credibility report for the console."""
    if not isinstance(statistics, dict):
        raise TypeError("MCC worker result is missing its statistics")
    try:
        total_trees = int(statistics["total_trees"])
        best_tree_number = int(statistics["best_tree_number"])
        number_of_clades = int(statistics["number_of_clades"])
        lowest_value = statistics["lowest_clade_credibility"]
        mean_value = statistics["mean_clade_credibility"]
        median_value = statistics["median_clade_credibility"]
        lowest = None if lowest_value is None else float(lowest_value)
        mean = None if mean_value is None else float(mean_value)
        median = None if median_value is None else float(median_value)
        credibility_1 = int(statistics["clades_with_credibility_1"])
        credibility_0_99 = int(
            statistics["clades_with_credibility_gt_0_99"]
        )
        credibility_0_95 = int(
            statistics["clades_with_credibility_gt_0_95"]
        )
        credibility_0_5 = int(
            statistics["clades_with_credibility_gt_0_5"]
        )
        majority_clades = int(
            statistics["majority_clades_in_all_trees"]
        )
        log_score = float(log_clade_credibility)
    except (KeyError, TypeError, ValueError) as exc:
        raise TypeError("MCC worker returned malformed statistics") from exc

    def format_credibility(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    return "\n".join(
        [
            "Finding maximum credibility tree...",
            f"Analyzing {total_trees} trees...",
            "",
            f"Best tree: {tree_name} (tree number {best_tree_number})",
            f"Best tree's log clade credibility: {log_score:.4f}",
            "Lowest individual clade credibility: "
            f"{format_credibility(lowest)}",
            "Mean individual clade credibility: "
            f"{format_credibility(mean)}",
            "Median individual clade credibility: "
            f"{format_credibility(median)}",
            f"Number of clades in tree: {number_of_clades}",
            "Number of clades with credibility 1.0: "
            f"{credibility_1}",
            "Number of clades with credibility > 0.99: "
            f"{credibility_0_99}",
            "Number of clades with credibility > 0.95: "
            f"{credibility_0_95}",
            "Number of clades with credibility > 0.5: "
            f"{credibility_0_5} / {majority_clades} (in all trees)",
        ]
    )


def _format_mrhipstr_timing_profile(profile: object) -> str:
    """Format the end-to-end MrHIPSTR timings for the visible console."""
    if not isinstance(profile, dict):
        raise TypeError("MrHIPSTR timing profile is missing")

    input_mode = profile.get("input_mode")
    if input_mode not in {
        None,
        "rooted_facts",
        "sparse_snapshot",
        "source_newicks",
    }:
        raise TypeError("MrHIPSTR timing profile has an unknown input mode")
    ingestion_label = (
        "Worker — rooted-facts aggregation (splits and heights)"
        if input_mode == "rooted_facts"
        else "Worker — source-tree parsing, splits, and heights"
    )
    fields = (
        (
            "selection_and_database_seconds",
            "Click callback — selection and database lookup",
        ),
        (
            "request_preparation_seconds",
            "Parent — worker request preparation",
        ),
        (
            "dispatch_queue_seconds",
            "Dispatch/queue — worker handoff and wait",
        ),
        ("worker_setup_seconds", "Worker — setup and imports"),
        (
            "taxon_alignment_seconds",
            "Worker — canonical taxon alignment",
        ),
        (
            "snapshot_selection_seconds",
            "Worker — snapshot load and selected-row lookup",
        ),
        (
            "clade_collection_seconds",
            "Worker — clade counting and catalog decoding",
        ),
        (
            "source_tree_ingestion_seconds",
            ingestion_label,
        ),
        (
            "topology_search_seconds",
            "Worker — MrHIPSTR topology search",
        ),
        (
            "nexus_serialization_seconds",
            "Worker — mean-height NEXUS construction",
        ),
        (
            "statistics_seconds",
            "Worker — summary-statistics calculation",
        ),
        (
            "worker_unattributed_seconds",
            "Worker — orchestration and cleanup",
        ),
        ("worker_total_seconds", "Worker subtotal"),
        (
            "result_handoff_seconds",
            "Result transfer and finalizer scheduling",
        ),
        (
            "publication_seconds",
            "Parent — cache and registry publication",
        ),
    )
    try:
        rows = [
            f"  {label}: {float(profile[key]):.4f} secs"
            for key, label in fields
        ]
        click_to_nexus = float(profile["click_to_nexus_seconds"])
        click_to_ready = float(profile["click_to_ready_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TypeError("MrHIPSTR timing profile is malformed") from exc

    mode_rows = {
        "rooted_facts": ["  Input path: RapidTrees rooted facts"],
        "sparse_snapshot": [
            "  Input path: sparse clade snapshot + source-tree parsing"
        ],
        "source_newicks": [
            "  Input path: legacy snapshot + source-tree parsing"
        ],
        None: [],
    }[input_mode]
    return "\n".join(
        [
            "MrHIPSTR timing profile:",
            *mode_rows,
            *rows,
            "",
            "Total View Summary click to MrHIPSTR NEXUS creation: "
            f"{click_to_nexus:.4f} secs",
            "Total View Summary click to cached and registered tree: "
            f"{click_to_ready:.4f} secs",
        ]
    )


def reset() -> None:
    """Invalidate an active consensus job before application state clears."""
    active = job_manager.active_ref()
    if active is None or active.kind != "consensus":
        return
    persistent_worker.cancel_current_job()
    job_manager.invalidate(active)


def _missing_taxa_message(missing_taxa: Any) -> str:
    missing = sorted(str(name) for name in missing_taxa)
    sample = ", ".join(missing[:5])
    more = "…" if len(missing) > 5 else ""
    return (
        f"Cannot align translate tables: taxa [{sample}{more}] are present "
        "in some selected runs but not in the canonical Translate block."
    )


def _finalize_consensus_tree_job(
    _ref: JobRef,
    result: Any,
    *,
    context: _ConsensusFinalizationContext,
) -> dict[str, Any]:
    """Publish one worker result and return a small browser payload."""
    finalization_started_at = time.perf_counter()
    finalization_started_wall_time = time.time()

    if not isinstance(result, dict):
        raise TypeError("consensus-tree worker returned a non-mapping result")
    if result.get("missing_taxa"):
        raise ConsensusTaxaAlignmentError(
            _missing_taxa_message(result["missing_taxa"])
        )

    nexus_bytes = result.get("nexus_bytes")
    if not isinstance(nexus_bytes, (bytes, bytearray)):
        raise TypeError("consensus-tree worker result is missing NEXUS bytes")

    expected_method = _normalise_summary_method(context.summary_method)
    summary_method = _normalise_summary_method(
        result.get("summary_method", expected_method)
    )
    if summary_method != expected_method:
        raise ValueError(
            "consensus-tree worker returned a different summary method "
            f"({summary_method!r}) than requested ({expected_method!r})"
        )

    consensus_tree_row = result.get("consensus_tree_row")
    if summary_method == "mcc":
        if not isinstance(consensus_tree_row, dict):
            raise TypeError("MCC worker result is missing its sampled tree row")
        consensus_tree_name = str(
            result.get("summary_tree_name") or consensus_tree_row["name"]
        )
        coord = context.coord_by_tree_name.get(consensus_tree_name)
        if coord is None:
            consensus_tree_group, consensus_treenum = None, None
        else:
            consensus_tree_group, consensus_treenum = coord
        consensus_tree_log_posterior = extract_log_posterior(
            consensus_tree_row
        )
    else:
        if consensus_tree_row is not None:
            raise TypeError(
                "MrHIPSTR worker result must not identify a sampled tree row"
            )
        consensus_tree_name = str(
            result.get("summary_tree_name") or "MrHIPSTR"
        )
        consensus_tree_group, consensus_treenum = None, None
        consensus_tree_log_posterior = None

    log_clade_cred = result.get("log_clade_credibility")
    log_clade_cred = (
        None if log_clade_cred is None else float(log_clade_cred)
    )
    height_method = str(
        result.get(
            "height_method",
            "sampled" if summary_method == "mcc" else "mean",
        )
    ).strip().lower()
    majority_clade_count = result.get("majority_clade_count")
    negative_branch_count = int(result.get("negative_branch_count") or 0)
    minimum_branch_length = result.get("minimum_branch_length")
    minimum_branch_length = (
        None
        if minimum_branch_length is None
        else float(minimum_branch_length)
    )

    # This finalizer runs behind JobManager's publication barrier and can only
    # be claimed once. Poll retries never execute these state mutations again.
    uid = _state.cache_consensus_tree(bytes(nexus_bytes))
    entry = _state.register_consensus_tree(
        source_distmat=context.source_distmat,
        mode=context.mode,
        run=context.run,
        uuid=uid,
        consensus_tree={
            "group": consensus_tree_group,
            "treenum": consensus_treenum,
            "tree_name": consensus_tree_name,
        },
        selection=context.selection,
        summary_method=summary_method,
        height_method=height_method,
        log_clade_credibility=log_clade_cred,
        majority_clade_count=majority_clade_count,
        negative_branch_count=negative_branch_count,
        minimum_branch_length=minimum_branch_length,
        consensus_tree_log_posterior=consensus_tree_log_posterior,
        tree_names=context.tree_names,
        counts=result.get("counts"),
        cols_in_consensus_tree=result.get("cols_in_consensus_tree"),
    )
    publication_finished_at = time.perf_counter()
    registered_name = entry["name"]
    n_trees = len(context.tree_names)
    method_label = _summary_method_label(summary_method)
    log_prefix = f"[{method_label}/{context.mode}]"
    score_text = (
        "n/a" if log_clade_cred is None else f"{log_clade_cred:.6g}"
    )
    if summary_method == "mrhipstr":
        worker_profile = result.get("mrhipstr_profile")
        input_mode = (
            worker_profile.get("input_mode")
            if isinstance(worker_profile, dict)
            else None
        )
        data_source = {
            "rooted_facts": "RapidTrees rooted facts",
            "sparse_snapshot": (
                "sparse clade rows plus source-tree parsing"
            ),
            "source_newicks": (
                "legacy dense snapshot plus source-tree parsing"
            ),
        }.get(input_mode, "worker-reported source unavailable")
        add_log(
            f"{log_prefix} Summary path: ROOTED clade RF → MrHIPSTR; "
            f"splits/heights from {data_source}; output is a synthetic "
            "mean-height rooted tree."
        )
        mrhipstr_statistics = result.get("mrhipstr_statistics")
        if mrhipstr_statistics is None:
            # Backward-compatible fallback for an in-flight result produced
            # by an older worker during a development hot reload.
            majority_text = (
                "n/a"
                if majority_clade_count is None
                else str(int(majority_clade_count))
            )
            add_log(
                f"{log_prefix} Completed clade-frequency collection, "
                "source-tree split/height ingestion, dynamic-programming "
                "topology construction, and mean-height NEXUS serialization "
                f"for {n_trees} selected trees; log clade credibility="
                f"{score_text}, majority clades={majority_text}."
            )
        else:
            add_log(
                _format_mrhipstr_statistics(
                    mrhipstr_statistics,
                    log_clade_credibility=log_clade_cred,
                )
            )
        if isinstance(worker_profile, dict):
            try:
                required_context_times = (
                    context.click_started_at,
                    context.click_started_wall_time,
                    context.submit_started_at,
                    context.dispatch_started_at,
                    context.dispatch_wall_time,
                )
                if any(value is None for value in required_context_times):
                    raise TypeError(
                        "parent timing context is incomplete"
                    )
                timing_profile = dict(worker_profile)
                timing_profile.update(
                    {
                        "selection_and_database_seconds": max(
                            0.0,
                            float(context.submit_started_at)
                            - float(context.click_started_at),
                        ),
                        "request_preparation_seconds": max(
                            0.0,
                            float(context.dispatch_started_at)
                            - float(context.submit_started_at),
                        ),
                        "dispatch_queue_seconds": max(
                            0.0,
                            float(
                                worker_profile[
                                    "worker_started_wall_time"
                                ]
                            )
                            - float(context.dispatch_wall_time),
                        ),
                        "result_handoff_seconds": max(
                            0.0,
                            finalization_started_wall_time
                            - float(
                                worker_profile[
                                    "worker_finished_wall_time"
                                ]
                            ),
                        ),
                        "publication_seconds": max(
                            0.0,
                            publication_finished_at
                            - finalization_started_at,
                        ),
                        "click_to_nexus_seconds": max(
                            0.0,
                            float(
                                worker_profile[
                                    "nexus_created_wall_time"
                                ]
                            )
                            - float(context.click_started_wall_time),
                        ),
                        "click_to_ready_seconds": max(
                            0.0,
                            publication_finished_at
                            - float(context.click_started_at),
                        ),
                    }
                )
                add_log(_format_mrhipstr_timing_profile(timing_profile))
            except (KeyError, TypeError, ValueError) as exc:
                add_log(
                    f"{log_prefix} Could not format the timing profile: "
                    f"{exc}.",
                    "WARNING",
                )
        branch_level = "WARNING" if negative_branch_count else "INFO"
        add_log(
            f"{log_prefix} Mean-height branch diagnostics: "
            f"negative branches={negative_branch_count}, "
            f"minimum branch length={minimum_branch_length!r}.",
            branch_level,
        )
    else:
        snapshot_input_mode = result.get("snapshot_input_mode")
        snapshot_input_text = {
            "sparse": "sparse CSR clade-presence rows",
            "rooted_facts": "sparse clade rows from RapidTrees rooted facts",
            "dense_legacy": "legacy dense clade-presence rows",
        }.get(snapshot_input_mode, "worker-reported clade-presence rows")
        if context.is_rooted:
            add_log(
                f"{log_prefix} Summary path: ROOTED clade RF → MCC; "
                "output is the highest-scoring sampled source tree with "
                "its rooting retained."
            )
            serialization_text = "source-tree serialization"
        else:
            add_log(
                f"{log_prefix} Summary path: UNROOTED bipartition RF → "
                "MCC; output is the highest-scoring sampled source tree, "
                "midpoint-rooted for export."
            )
            serialization_text = "midpoint-rooted source-tree serialization"
        add_log(
            f"{log_prefix} MCC scoring input: {snapshot_input_text}."
        )
        mcc_statistics = result.get("mcc_statistics")
        if mcc_statistics is not None:
            add_log(
                _format_mcc_statistics(
                    mcc_statistics,
                    tree_name=consensus_tree_name,
                    log_clade_credibility=log_clade_cred,
                )
            )
        add_log(
            f"{log_prefix} Completed clade-frequency scoring and "
            f"{serialization_text} for {n_trees} selected trees; selected "
            f"'{consensus_tree_name}', log clade credibility={score_text}."
        )
    add_log(
        f"{log_prefix} Cached summary tree as {uid}; registered as "
        f"{registered_name}."
    )
    return {
        "uuid": uid,
        "name": registered_name,
        "consensus_tree_name": consensus_tree_name,
        "n_trees": n_trees,
        "mode": context.mode,
        "summary_method": summary_method,
        "negative_branch_count": negative_branch_count,
    }


def submit_consensus_tree_job(
    *,
    matched_records: list,
    source_distmat: str,
    mode: str,
    selection: list,
    run: str | None,
    consensus_tree_coord_by_tree_name: dict[str, tuple],
    store_target: str,
    summary_method: str = "mrhipstr",
    click_started_at: float | None = None,
    click_started_wall_time: float | None = None,
) -> JobRef:
    """Enqueue a consensus tree compute job. Called by both tab callbacks.

    Args:
        matched_records: per-tree dicts (``name``, ``file_source``,
            ``newick_offset``, ``newick_length``, ``line_offset``,
            ``line_length``, ``metadata``) in DB order. The parent extracts
            these from a vectorised DataFrame slice.
        source_distmat: distmat name the selection was drawn from.
        mode: ``"Between"`` or ``"Within"`` — propagated into the
            registry entry's mode field and the per-tab labelling.
        selection: serialisable representation of the user's selection,
            stored on the registry entry so the consensus tree list table can
            re-create the orange selection ring.
        run: the run/group name for Within mode, else ``None``.
        consensus_tree_coord_by_tree_name: ``{tree_name: (group, treenum)}`` —
            consulted by the terminal presentation adapter to populate
            ``consensus_tree.treenum`` (used to put the green ring on the
            consensus tree's MDS dot).
        store_target: ``"treespace-view-consensus-tree-store"`` or
            ``"within-run-view-consensus-tree-store"`` — tells the terminal
            adapter which tab's clientside ``window.open`` to fire.
        summary_method: ``"mrhipstr"`` (default) or explicit ``"mcc"`` for
            compatibility with the retained sampled-tree implementation.
            An unrooted RF matrix automatically resolves MrHIPSTR to MCC;
            its selected sampled tree is midpoint-rooted for export.
    """
    submit_started_at = time.perf_counter()
    if click_started_at is None:
        click_started_at = submit_started_at
    if click_started_wall_time is None:
        click_started_wall_time = time.time()

    if not matched_records:
        raise ValueError("matched_records must not be empty")
    if store_target not in _STORE_TARGETS:
        raise ValueError(f"unsupported consensus-tree store target: {store_target}")
    requested_summary_method = _normalise_summary_method(summary_method)
    distmat_is_rooted = _state.get_distmat_is_rooted(source_distmat)
    use_unrooted_mcc = (
        requested_summary_method == "mrhipstr" and not distmat_is_rooted
    )
    summary_method = "mcc" if use_unrooted_mcc else requested_summary_method

    tree_service = _get_tree_service()
    db_manager = tree_service.db_manager

    # ── Build the kwargs the worker needs ──────────────────────────────
    unique_sources = list(dict.fromkeys(r["file_source"] for r in matched_records))

    snapshots_path = _state.get_snapshots_path(source_distmat)
    full_distmat_names = _state.get_distmat_names(source_distmat)

    source_file_paths = {
        fs: db_manager._source_files[fs]
        for fs in unique_sources
        if fs in db_manager._source_files
    }
    translate_maps = {
        fs: db_manager.get_translate_map(fs) for fs in unique_sources
    }
    source_preambles = {
        fs: db_manager._source_preambles.get(fs)
        for fs in unique_sources
        if fs in getattr(db_manager, "_source_preambles", {})
    }

    tree_names = tuple(str(r["name"]) for r in matched_records)
    context_selection = copy.deepcopy(selection)
    context_coords = {
        str(name): (coord[0], int(coord[1]))
        for name, coord in consensus_tree_coord_by_tree_name.items()
    }

    method_label = _summary_method_label(summary_method)
    log_prefix = f"[{method_label}/{mode}]"
    if use_unrooted_mcc:
        add_log(
            f"{log_prefix} Unrooted RF matrix detected; automatically using "
            "MCC instead of MrHIPSTR. The selected MCC tree will be "
            "midpoint-rooted for export."
        )
    rf_input_text = (
        "ROOTED clade RF"
        if distmat_is_rooted
        else "UNROOTED bipartition RF; MCC output will be midpoint-rooted"
    )
    add_log(
        f"{log_prefix} Starting summary-tree computation for "
        f"{len(matched_records)} selected trees from {source_distmat} "
        f"(input={rf_input_text})."
    )
    if summary_method == "mrhipstr":
        add_log(
            f"{log_prefix} Stages: collect selected clade frequencies "
            "from rooted RF data → obtain observed splits and node heights "
            "from rooted facts (source-tree fallback) → run the MrHIPSTR "
            "dynamic program → serialize mean-height NEXUS."
        )
    else:
        add_log(
            f"{log_prefix} Stages: load selected clade-presence rows → "
            "score source topologies → serialize the highest-scoring "
            "source tree."
        )
    dispatch_started_at = time.perf_counter()
    dispatch_wall_time = time.time()
    context = _ConsensusFinalizationContext(
        mode=str(mode),
        run=None if run is None else str(run),
        source_distmat=str(source_distmat),
        selection=context_selection,
        tree_names=tree_names,
        coord_by_tree_name=context_coords,
        summary_method=summary_method,
        is_rooted=distmat_is_rooted,
        click_started_at=float(click_started_at),
        click_started_wall_time=float(click_started_wall_time),
        submit_started_at=submit_started_at,
        dispatch_started_at=dispatch_started_at,
        dispatch_wall_time=dispatch_wall_time,
    )
    ref = job_manager.submit(
        _get_executor(),
        "consensus",
        persistent_worker.submit_job,
        "compute_consensus_tree",
        matched_records=matched_records,
        source_distmat=source_distmat,
        snapshots_path=str(snapshots_path),
        full_distmat_names=list(full_distmat_names),
        translate_maps=translate_maps,
        source_file_paths=source_file_paths,
        source_preambles=source_preambles,
        is_rooted=distmat_is_rooted,
        summary_method=summary_method,
        metadata={
            "display_name": f"{mode} {summary_method} summary tree",
            "mode": mode,
            "source_distmat": source_distmat,
            "summary_method": summary_method,
            "store_target": store_target,
        },
        finalizer=partial(
            _finalize_consensus_tree_job,
            context=context,
        ),
        cancel_exceptions=(persistent_worker.JobCancelled,),
    )
    return ref


def _get_tree_service():
    # Lazy local import to avoid the tree_service module being pulled
    # in at callback registration time.
    from ..db.tree_service import get_tree_service
    return get_tree_service()


def register_consensus_tree_compute_callbacks():
    @callback(
        # View stores: only the originating tab changes.
        Output("treespace-view-consensus-tree-store", "data", allow_duplicate=True),
        Output("within-run-view-consensus-tree-store", "data", allow_duplicate=True),
        Output("consensus-tree-registry-store", "data", allow_duplicate=True),
        # Both overlays are dismissed in case the user changed tabs.
        Output("treespace-loading-overlay", "visible", allow_duplicate=True),
        Output("within-run-loading-overlay", "visible", allow_duplicate=True),
        # The originating selection is cleared on success.
        Output("treespace-selected-trees-store", "data", allow_duplicate=True),
        Output("within-run-selected-trees-store", "data", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
        Output(
            {"type": "compute-terminal-receipt", "kind": "consensus"},
            "data",
        ),
        Input("compute-terminal-event-store", "data"),
        Input("consensus-job-store", "data"),
        prevent_initial_call=True,
    )
    def render_consensus_terminal_event(terminal_event, job_data):
        event = terminal_event_for_job(
            terminal_event,
            job_data,
            expected_kind="consensus",
        )
        if event is None:
            return (no_update,) * 9

        ref = JobRef.from_dict(event)
        terminal_state = JobState(str(event["state"]))
        payload = event["payload"]
        metadata = event.get("metadata") or {}
        store_target = metadata.get("store_target")
        method_label = _summary_method_label(
            metadata.get("summary_method", payload.get("summary_method"))
        )
        mode = metadata.get("mode") or payload.get("mode") or "Summary"
        log_prefix = f"[{method_label}/{mode}]"
        first_delivery = int(event["delivery_attempt"]) == 1

        out_treespace_store = no_update
        out_within_store = no_update
        out_registry = no_update
        out_treespace_overlay = False
        out_within_overlay = False
        out_treespace_sel = no_update
        out_within_sel = no_update

        notif = no_update
        if terminal_state is JobState.CANCELLED:
            if first_delivery:
                add_log(
                    f"{log_prefix} Summary-tree computation cancelled by user.",
                    "WARNING",
                )
        elif terminal_state is JobState.FAILED:
            message = str(payload.get("message", "Unknown error"))
            if first_delivery:
                add_log(
                    f"{log_prefix} Summary-tree computation failed: {message}",
                    "ERROR",
                )
            notif = dmc.Notification(
                title="Summary Tree Error",
                message=message,
                color="red",
                action="show",
                autoClose=8000,
                id=f"consensus-terminal-{ref.job_id}",
            )
        else:
            view_payload = {"uuid": payload["uuid"], "name": payload["name"]}
            if store_target == _TREESPACE_TARGET:
                out_treespace_store = view_payload
                out_treespace_sel = []
            elif store_target == _WITHIN_RUN_TARGET:
                out_within_store = view_payload
                out_within_sel = []
            out_registry = _state.get_consensus_tree_registry()
            negative_branch_count = int(
                payload.get("negative_branch_count") or 0
            )
            if negative_branch_count:
                title = "Summary Tree Ready with Warning"
                message = (
                    f"{method_label} summary tree {payload['name']} was "
                    "computed from "
                    f"{payload['n_trees']} selected trees, but mean heights "
                    f"produced {negative_branch_count} negative branch"
                    f"{'es' if negative_branch_count != 1 else ''}."
                )
                color = "yellow"
                auto_close = 8000
            else:
                title = "Summary Tree Ready"
                message = (
                    f"{method_label} summary tree {payload['name']} computed "
                    f"from {payload['n_trees']} selected trees."
                )
                color = "green"
                auto_close = 3000
            notif = dmc.Notification(
                title=title,
                message=message,
                color=color,
                action="show",
                autoClose=auto_close,
                id=f"consensus-terminal-{ref.job_id}",
            )

        return (
            out_treespace_store,
            out_within_store,
            out_registry,
            out_treespace_overlay,
            out_within_overlay,
            out_treespace_sel,
            out_within_sel,
            notif,
            terminal_delivery_marker(event),
        )
