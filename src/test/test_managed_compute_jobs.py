"""Lifecycle integration tests for managed Pseudo-ESS and consensus jobs."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from dash import no_update

from treetracer.background_jobs import JobManager, JobState
from treetracer.callbacks import (
    compute,
    consensus_tree_compute,
    job_reconcile,
    pseudo_ess_compute,
    sidebar,
    treespace,
    within_run,
)


def _wait_for_terminal(manager, ref, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = manager.snapshot(ref)
        if snapshot is not None and snapshot.terminal is not None:
            return snapshot
        time.sleep(0.005)
    pytest.fail("background job did not become terminal")


def _registered_callback(name, register):
    from dash import _callback

    def matches():
        found = []
        for callback_data in _callback.GLOBAL_CALLBACK_MAP.values():
            callback_fn = callback_data.get("callback")
            if callback_fn is None:
                continue
            original = getattr(callback_fn, "__wrapped__", callback_fn)
            if original.__name__ == name:
                found.append(original)
        return found

    found = matches()
    if not found:
        register()
        found = matches()
    assert found
    return found[-1]


def test_pseudo_ess_submit_and_terminal_ui_replay_until_ack(monkeypatch):
    now = [10.0]
    manager = JobManager(
        id_factory=lambda: "pseudo-test-job",
        clock=lambda: now[0],
    )
    worker_calls = []

    def worker(job_name, **kwargs):
        worker_calls.append((job_name, kwargs))
        return {
            "results": [
                {
                    "label": "run-a",
                    "n_trees": 20,
                    "burnin_label": "5",
                    "min": 110.0,
                    "q2": 210.0,
                    "max": 310.0,
                    "n_refs_used": 20,
                }
            ]
        }

    monkeypatch.setattr(pseudo_ess_compute, "job_manager", manager)
    monkeypatch.setattr(compute, "job_manager", manager)
    monkeypatch.setattr(pseudo_ess_compute, "add_log", lambda *_a, **_k: None)
    monkeypatch.setattr(pseudo_ess_compute, "_wlog", lambda *_a, **_k: None)
    monkeypatch.setattr(pseudo_ess_compute.persistent_worker, "submit_job", worker)

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(pseudo_ess_compute, "_get_executor", lambda: executor)
        ref = pseudo_ess_compute.submit_pseudo_ess_job(
            distmat_path="/tmp/RF_001.npy",
            names=["run-a/tree-1"],
            requests=[
                {
                    "label": "run-a",
                    "indices": [0, 1, 2, 3],
                    "burnin_label": "5",
                }
            ],
            n_refs=20,
        )
        terminal = _wait_for_terminal(manager, ref)

    assert ref.kind == "pseudo_ess"
    assert worker_calls[0][0] == "compute_pseudo_ess"
    assert terminal.state is JobState.SUCCEEDED
    assert terminal.terminal.payload["n_rows"] == 1

    render = _registered_callback(
        "render_pseudo_ess_terminal_event",
        pseudo_ess_compute.register_pseudo_ess_compute_callbacks,
    )
    first_event = job_reconcile._terminal_envelope(
        manager.claim_terminal_delivery(ref)
    )
    now[0] += 1.0
    second_event = job_reconcile._terminal_envelope(
        manager.claim_terminal_delivery(ref)
    )
    first = render(first_event, ref.as_dict())
    second = render(second_event, ref.as_dict())

    assert len(first) == 2
    assert first[1]["terminal_revision"] == terminal.terminal.revision
    assert first[1]["delivery_attempt"] == 1
    assert second[1]["delivery_attempt"] == 2
    assert manager.snapshot(ref).acknowledged is False

    assert manager.acknowledge(ref, second[1]["terminal_revision"]) is True
    assert manager.snapshot(ref).acknowledged is True
    assert manager.active_ref() is None


def test_consensus_finalizer_publishes_once_and_poll_only_replays(monkeypatch):
    now = [20.0]
    manager = JobManager(
        id_factory=lambda: "consensus-test-job",
        clock=lambda: now[0],
    )
    cache_calls = []
    register_calls = []
    registry = [{"name": "RF_001_Between_consensus_tree_1"}]

    def cache(nexus_bytes):
        cache_calls.append(nexus_bytes)
        return "tree-uuid"

    def register(**kwargs):
        register_calls.append(kwargs)
        return {"name": "RF_001_Between_consensus_tree_1"}

    monkeypatch.setattr(consensus_tree_compute, "job_manager", manager)
    monkeypatch.setattr(compute, "job_manager", manager)
    monkeypatch.setattr(consensus_tree_compute, "add_log", lambda *_a, **_k: None)
    monkeypatch.setattr(consensus_tree_compute._state, "cache_consensus_tree", cache)
    monkeypatch.setattr(
        consensus_tree_compute._state,
        "register_consensus_tree",
        register,
    )
    monkeypatch.setattr(
        consensus_tree_compute._state,
        "get_consensus_tree_registry",
        lambda: registry,
    )

    context = consensus_tree_compute._ConsensusFinalizationContext(
        source_distmat="RF_001",
        mode="Between",
        run=None,
        selection=[["run-a", 1], ["run-b", 2]],
        tree_names=("run-a/tree-1", "run-b/tree-2"),
        coord_by_tree_name={"run-b/tree-2": ("run-b", 2)},
    )
    result = {
        "nexus_bytes": b"#NEXUS\n",
        "consensus_tree_row": {"name": "run-b/tree-2", "metadata": {}},
        "log_clade_credibility": -2.5,
        "counts": np.array([1, 2], dtype=np.int32),
        "cols_in_consensus_tree": frozenset({1}),
        "missing_taxa": set(),
    }
    with ThreadPoolExecutor(max_workers=1) as executor:
        ref = manager.submit(
            executor,
            "consensus",
            lambda: result,
            metadata={
                "store_target": consensus_tree_compute._TREESPACE_TARGET,
            },
            finalizer=partial(
                consensus_tree_compute._finalize_consensus_tree_job,
                context=context,
            ),
        )
        terminal = _wait_for_terminal(manager, ref)

    assert terminal.state is JobState.SUCCEEDED
    assert cache_calls == [b"#NEXUS\n"]
    assert len(register_calls) == 1
    assert register_calls[0]["consensus_tree"] == {
        "group": "run-b",
        "treenum": 2,
        "tree_name": "run-b/tree-2",
    }
    assert "nexus_bytes" not in terminal.terminal.payload
    assert "counts" not in terminal.terminal.payload

    render = _registered_callback(
        "render_consensus_terminal_event",
        consensus_tree_compute.register_consensus_tree_compute_callbacks,
    )
    first_event = job_reconcile._terminal_envelope(
        manager.claim_terminal_delivery(ref)
    )
    now[0] += 1.0
    second_event = job_reconcile._terminal_envelope(
        manager.claim_terminal_delivery(ref)
    )
    first = render(first_event, ref.as_dict())
    second = render(second_event, ref.as_dict())

    assert len(first) == 9
    assert first[0] == {
        "uuid": "tree-uuid",
        "name": "RF_001_Between_consensus_tree_1",
    }
    assert first[1] is no_update
    assert first[2] == registry
    assert first[3:5] == (False, False)
    assert first[5] == []
    assert first[6] is no_update
    assert first[8]["delivery_attempt"] == 1
    assert second[8]["delivery_attempt"] == 2
    assert len(cache_calls) == 1
    assert len(register_calls) == 1

    assert manager.acknowledge(ref, second[8]["terminal_revision"]) is True
    assert manager.snapshot(ref).acknowledged is True


def test_consensus_finalizer_registers_synthetic_mrhipstr_without_mds_row(
    monkeypatch,
):
    register_calls = []
    log_calls = []
    monkeypatch.setattr(
        consensus_tree_compute._state,
        "cache_consensus_tree",
        lambda value: "mrhipstr-uuid",
    )
    monkeypatch.setattr(
        consensus_tree_compute._state,
        "register_consensus_tree",
        lambda **kwargs: (
            register_calls.append(kwargs)
            or {"name": "RF_001_Between_MrHIPSTR_1"}
        ),
    )
    monkeypatch.setattr(
        consensus_tree_compute,
        "add_log",
        lambda message, level="INFO": log_calls.append((message, level)),
    )

    now_perf = time.perf_counter()
    now_wall = time.time()
    context = consensus_tree_compute._ConsensusFinalizationContext(
        source_distmat="RF_001",
        mode="Between",
        run=None,
        selection=[["run-a", 1], ["run-b", 2]],
        tree_names=("run-a/tree-1", "run-b/tree-2"),
        coord_by_tree_name={"run-a/tree-1": ("run-a", 1)},
        summary_method="mrhipstr",
        click_started_at=now_perf - 1.0,
        click_started_wall_time=now_wall - 1.0,
        submit_started_at=now_perf - 0.9,
        dispatch_started_at=now_perf - 0.8,
        dispatch_wall_time=now_wall - 0.8,
    )
    result = {
        "nexus_bytes": b"#NEXUS\n",
        "summary_method": "mrhipstr",
        "height_method": "mean",
        "consensus_tree_row": None,
        "summary_tree_name": "MrHIPSTR",
        "log_clade_credibility": -1.5,
        "majority_clade_count": 4,
        "negative_branch_count": 1,
        "minimum_branch_length": -0.01,
        "mrhipstr_statistics": {
            "total_trees": 2,
            "n_tips": 4,
            "total_unique_clades": 5,
            "clades_in_more_than_one_tree": 3,
            "topology_seconds": 0.215,
            "lowest_clade_credibility": 0.5,
            "mean_clade_credibility": 0.75,
            "median_clade_credibility": 0.75,
            "clades_with_credibility_1": 1,
            "clades_with_credibility_gt_0_99": 1,
            "clades_with_credibility_gt_0_95": 1,
            "clades_with_credibility_gt_0_5": 2,
        },
        "mrhipstr_profile": {
            "input_mode": "rooted_facts",
            "worker_started_wall_time": now_wall - 0.7,
            "worker_setup_seconds": 0.01,
            "taxon_alignment_seconds": 0.01,
            "snapshot_selection_seconds": 0.02,
            "clade_collection_seconds": 0.03,
            "source_tree_ingestion_seconds": 0.20,
            "topology_search_seconds": 0.215,
            "nexus_serialization_seconds": 0.04,
            "nexus_created_wall_time": now_wall - 0.1,
            "statistics_seconds": 0.01,
            "worker_unattributed_seconds": 0.005,
            "worker_total_seconds": 0.54,
            "worker_finished_wall_time": now_wall - 0.05,
        },
        "counts": np.array([2, 1], dtype=np.int32),
        "cols_in_consensus_tree": frozenset({0}),
        "missing_taxa": set(),
    }

    payload = consensus_tree_compute._finalize_consensus_tree_job(
        None,
        result,
        context=context,
    )

    assert len(register_calls) == 1
    registered = register_calls[0]
    assert registered["consensus_tree"] == {
        "group": None,
        "treenum": None,
        "tree_name": "MrHIPSTR",
    }
    assert registered["consensus_tree_log_posterior"] is None
    assert registered["summary_method"] == "mrhipstr"
    assert registered["height_method"] == "mean"
    assert registered["majority_clade_count"] == 4
    assert registered["negative_branch_count"] == 1
    assert registered["minimum_branch_length"] == pytest.approx(-0.01)
    assert payload == {
        "uuid": "mrhipstr-uuid",
        "name": "RF_001_Between_MrHIPSTR_1",
        "consensus_tree_name": "MrHIPSTR",
        "n_trees": 2,
        "mode": "Between",
        "summary_method": "mrhipstr",
        "negative_branch_count": 1,
    }
    assert any(
        "Total trees read: 2" in message
        and "[0.215 secs]" in message
        and "Mean individual clade credibility: 0.7500" in message
        for message, _level in log_calls
    )
    assert any(
        "negative branches=1" in message and level == "WARNING"
        for message, level in log_calls
    )
    assert any(
        "Summary path: ROOTED clade RF → MrHIPSTR" in message
        and "splits/heights from RapidTrees rooted facts" in message
        and "synthetic mean-height rooted tree" in message
        for message, _level in log_calls
    )
    assert any(
        "MrHIPSTR timing profile:" in message
        and "Total View Summary click to MrHIPSTR NEXUS creation:"
        in message
        for message, _level in log_calls
    )
    assert any(
        "registered as RF_001_Between_MrHIPSTR_1" in message
        for message, _level in log_calls
    )


def test_mrhipstr_console_statistics_match_treeannotator_style():
    report = consensus_tree_compute._format_mrhipstr_statistics(
        {
            "total_trees": 912,
            "n_tips": 1383,
            "total_unique_clades": 43311,
            "clades_in_more_than_one_tree": 15392,
            "topology_seconds": 0.2154,
            "lowest_clade_credibility": 0.0011,
            "mean_clade_credibility": 0.7026,
            "median_clade_credibility": 0.9682,
            "clades_with_credibility_1": 2,
            "clades_with_credibility_gt_0_99": 599,
            "clades_with_credibility_gt_0_95": 720,
            "clades_with_credibility_gt_0_5": 915,
        },
        log_clade_credibility=-923.4827,
    )

    assert report == """Total trees read: 912
Size of trees: 1383 tips
Total unique clades: 43311
Total clades in more than one tree: 15392

Finding majority rule highest independent posterior subtree reconstruction (MrHIPSTR) tree...
[0.215 secs]

MrHIPSTR tree's log clade credibility: -923.4827
Lowest individual clade credibility: 0.0011
Mean individual clade credibility: 0.7026
Median individual clade credibility: 0.9682
Number of clades with credibility 1.0: 2
Number of clades with credibility > 0.99: 599
Number of clades with credibility > 0.95: 720
Number of clades with credibility > 0.5: 915"""


def test_mrhipstr_console_timing_profile_lists_every_stage():
    report = consensus_tree_compute._format_mrhipstr_timing_profile(
        {
            "input_mode": "source_newicks",
            "selection_and_database_seconds": 0.01,
            "request_preparation_seconds": 0.02,
            "dispatch_queue_seconds": 0.03,
            "worker_setup_seconds": 0.04,
            "taxon_alignment_seconds": 0.05,
            "snapshot_selection_seconds": 0.06,
            "clade_collection_seconds": 0.07,
            "source_tree_ingestion_seconds": 0.08,
            "topology_search_seconds": 0.09,
            "nexus_serialization_seconds": 0.10,
            "statistics_seconds": 0.11,
            "worker_unattributed_seconds": 0.12,
            "worker_total_seconds": 0.72,
            "result_handoff_seconds": 0.13,
            "publication_seconds": 0.14,
            "click_to_nexus_seconds": 1.23,
            "click_to_ready_seconds": 1.50,
        }
    )

    assert report == """MrHIPSTR timing profile:
  Input path: legacy snapshot + source-tree parsing
  Click callback — selection and database lookup: 0.0100 secs
  Parent — worker request preparation: 0.0200 secs
  Dispatch/queue — worker handoff and wait: 0.0300 secs
  Worker — setup and imports: 0.0400 secs
  Worker — canonical taxon alignment: 0.0500 secs
  Worker — snapshot load and selected-row lookup: 0.0600 secs
  Worker — clade counting and catalog decoding: 0.0700 secs
  Worker — source-tree parsing, splits, and heights: 0.0800 secs
  Worker — MrHIPSTR topology search: 0.0900 secs
  Worker — mean-height NEXUS construction: 0.1000 secs
  Worker — summary-statistics calculation: 0.1100 secs
  Worker — orchestration and cleanup: 0.1200 secs
  Worker subtotal: 0.7200 secs
  Result transfer and finalizer scheduling: 0.1300 secs
  Parent — cache and registry publication: 0.1400 secs

Total View Summary click to MrHIPSTR NEXUS creation: 1.2300 secs
Total View Summary click to cached and registered tree: 1.5000 secs"""


def test_synthetic_mrhipstr_registry_entry_produces_no_mds_overlay():
    frame = pd.DataFrame(
        {
            "group": ["run-a"],
            "treenum": [1],
            "x": [0.0],
            "y": [1.0],
            "z": [2.0],
        }
    )
    synthetic = {
        "name": "RF_001_Between_MrHIPSTR_1",
        "source_distmat": "RF_001",
        "mode": "Between",
        "summary_method": "mrhipstr",
        "consensus_tree": {
            "group": None,
            "treenum": None,
            "tree_name": "MrHIPSTR",
        },
    }

    between_panels = treespace._consensus_tree_overlay_panels_data(
        frame,
        [synthetic],
        "RF_001",
        "x",
        "y",
        "z",
    )
    within_panels = within_run._consensus_tree_panels_data(
        frame,
        [synthetic],
        "x",
        "y",
        "z",
    )

    assert all(panel["x"] == [] for panel in between_panels)
    assert all(panel["x"] == [] for panel in within_panels)


def test_consensus_submit_carries_mrhipstr_method_to_worker_and_context(
    monkeypatch,
):
    submitted = {}
    log_calls = []

    class CapturingManager:
        def submit(self, *args, **kwargs):
            submitted["args"] = args
            submitted["kwargs"] = kwargs
            return "job-ref"

    db = SimpleNamespace(
        _source_files={"run.trees": "/tmp/run.trees"},
        _source_preambles={"run.trees": b"#NEXUS\nBegin trees;\n"},
        get_translate_map=lambda _source: {"1": "A", "2": "B"},
    )
    monkeypatch.setattr(
        consensus_tree_compute,
        "job_manager",
        CapturingManager(),
    )
    monkeypatch.setattr(
        consensus_tree_compute,
        "_get_tree_service",
        lambda: SimpleNamespace(db_manager=db),
    )
    monkeypatch.setattr(
        consensus_tree_compute,
        "_get_executor",
        lambda: "executor",
    )
    monkeypatch.setattr(
        consensus_tree_compute._state,
        "get_snapshots_path",
        lambda _name: "/tmp/RF_001_snapshots.npz",
    )
    monkeypatch.setattr(
        consensus_tree_compute._state,
        "get_distmat_names",
        lambda _name: ["run/tree-1"],
    )
    monkeypatch.setattr(
        consensus_tree_compute._state,
        "get_distmat_is_rooted",
        lambda _name: True,
    )
    monkeypatch.setattr(
        consensus_tree_compute,
        "add_log",
        lambda message, level="INFO": log_calls.append((message, level)),
    )

    ref = consensus_tree_compute.submit_consensus_tree_job(
        matched_records=[
            {
                "name": "run/tree-1",
                "file_source": "run.trees",
                "newick_offset": 10,
                "newick_length": 20,
                "line_offset": 0,
                "line_length": 40,
                "metadata": {},
            }
        ],
        source_distmat="RF_001",
        mode="Between",
        selection=[["run", 1]],
        run=None,
        consensus_tree_coord_by_tree_name={"run/tree-1": ("run", 1)},
        store_target=consensus_tree_compute._TREESPACE_TARGET,
        summary_method="mrhipstr",
    )

    assert ref == "job-ref"
    assert submitted["args"][2] is consensus_tree_compute.persistent_worker.submit_job
    assert submitted["args"][3] == "compute_consensus_tree"
    assert submitted["kwargs"]["summary_method"] == "mrhipstr"
    assert submitted["kwargs"]["metadata"]["summary_method"] == "mrhipstr"
    context = submitted["kwargs"]["finalizer"].keywords["context"]
    assert context.summary_method == "mrhipstr"
    assert context.click_started_at == context.submit_started_at
    assert context.dispatch_started_at >= context.submit_started_at
    assert context.click_started_wall_time <= context.dispatch_wall_time
    assert any(
        "[MrHIPSTR/Between] Starting summary-tree computation" in message
        for message, _level in log_calls
    )
    assert any(
        "collect selected clade frequencies" in message
        and "serialize mean-height NEXUS" in message
        for message, _level in log_calls
    )


def test_consensus_submit_uses_midpoint_rooted_mcc_for_unrooted_matrix(
    monkeypatch,
):
    submitted = {}
    log_calls = []

    class CapturingManager:
        def submit(self, *args, **kwargs):
            submitted["args"] = args
            submitted["kwargs"] = kwargs
            return "job-ref"

    db = SimpleNamespace(
        _source_files={"run.trees": "/tmp/run.trees"},
        _source_preambles={"run.trees": b"#NEXUS\nBegin trees;\n"},
        get_translate_map=lambda _source: {"1": "A", "2": "B"},
    )
    monkeypatch.setattr(
        consensus_tree_compute,
        "job_manager",
        CapturingManager(),
    )
    monkeypatch.setattr(
        consensus_tree_compute._state,
        "get_distmat_is_rooted",
        lambda _name: False,
    )
    monkeypatch.setattr(
        consensus_tree_compute,
        "_get_tree_service",
        lambda: SimpleNamespace(db_manager=db),
    )
    monkeypatch.setattr(
        consensus_tree_compute,
        "_get_executor",
        lambda: "executor",
    )
    monkeypatch.setattr(
        consensus_tree_compute._state,
        "get_snapshots_path",
        lambda _name: "/tmp/RF_unrooted_snapshots.npz",
    )
    monkeypatch.setattr(
        consensus_tree_compute._state,
        "get_distmat_names",
        lambda _name: ["run/tree-1"],
    )
    monkeypatch.setattr(
        consensus_tree_compute,
        "add_log",
        lambda message, level="INFO": log_calls.append((message, level)),
    )

    ref = consensus_tree_compute.submit_consensus_tree_job(
        matched_records=[
            {
                "name": "run/tree-1",
                "file_source": "run.trees",
            }
        ],
        source_distmat="RF_unrooted",
        mode="Between",
        selection=[["run", 1]],
        run=None,
        consensus_tree_coord_by_tree_name={
            "run/tree-1": ("run", 1),
        },
        store_target=consensus_tree_compute._TREESPACE_TARGET,
        summary_method="mrhipstr",
    )

    assert ref == "job-ref"
    assert submitted["kwargs"]["is_rooted"] is False
    assert submitted["kwargs"]["summary_method"] == "mcc"
    assert submitted["kwargs"]["metadata"]["summary_method"] == "mcc"
    context = submitted["kwargs"]["finalizer"].keywords["context"]
    assert context.summary_method == "mcc"
    assert context.is_rooted is False
    assert any(
        "automatically using MCC instead of MrHIPSTR" in message
        and "midpoint-rooted for export" in message
        for message, _level in log_calls
    )
    assert any(
        "[MCC/Between] Starting summary-tree computation" in message
        for message, _level in log_calls
    )


def test_consensus_finalization_failure_reenables_origin_button(monkeypatch):
    manager = JobManager(id_factory=lambda: "consensus-failed-job")
    monkeypatch.setattr(consensus_tree_compute, "job_manager", manager)
    monkeypatch.setattr(consensus_tree_compute, "add_log", lambda *_a, **_k: None)
    monkeypatch.setattr(
        consensus_tree_compute._state,
        "cache_consensus_tree",
        lambda _value: pytest.fail("invalid result must not be cached"),
    )

    context = consensus_tree_compute._ConsensusFinalizationContext(
        source_distmat="RF_001",
        mode="Within",
        run="run-a",
        selection=[["run-a", 1]],
        tree_names=("run-a/tree-1",),
        coord_by_tree_name={},
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        ref = manager.submit(
            executor,
            "consensus",
            lambda: {"missing_taxa": {"Taxon C", "Taxon A"}},
            metadata={
                "store_target": consensus_tree_compute._WITHIN_RUN_TARGET,
            },
            finalizer=partial(
                consensus_tree_compute._finalize_consensus_tree_job,
                context=context,
            ),
        )
        terminal = _wait_for_terminal(manager, ref)

    assert terminal.state is JobState.FAILED
    assert terminal.terminal.payload["stage"] == "finalize"
    assert "Taxon A, Taxon C" in terminal.terminal.payload["message"]

    render = _registered_callback(
        "render_consensus_terminal_event",
        consensus_tree_compute.register_consensus_tree_compute_callbacks,
    )
    event = job_reconcile._terminal_envelope(
        manager.claim_terminal_delivery(ref)
    )
    output = render(event, ref.as_dict())
    assert output[3:5] == (False, False)
    assert output[5] is no_update
    assert output[6] is no_update
    assert output[8]["job_id"] == ref.job_id
    assert output[7].id == f"consensus-terminal-{ref.job_id}"


def test_clear_data_idle_branch_matches_managed_lifecycle_outputs():
    clear_uploads = _registered_callback(
        "clear_uploads",
        sidebar.register_sidebar_callbacks,
    )
    assert len(clear_uploads(None)) == 44
