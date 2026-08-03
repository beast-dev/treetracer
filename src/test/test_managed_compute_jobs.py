"""Lifecycle integration tests for managed Pseudo-ESS and consensus jobs."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import pytest
from dash import no_update

from treetracer.background_jobs import JobManager, JobState
from treetracer.callbacks import (
    compute,
    consensus_tree_compute,
    job_reconcile,
    pseudo_ess_compute,
    sidebar,
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
