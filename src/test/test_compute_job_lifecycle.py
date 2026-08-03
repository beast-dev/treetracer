"""RF/MDS integration tests for the sticky background-job lifecycle."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from treetracer.background_jobs import JobManager, JobState
from treetracer.callbacks import compute, job_reconcile


def _wait_for_terminal(manager, ref, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = manager.snapshot(ref)
        if snapshot is not None and snapshot.terminal is not None:
            return snapshot
        time.sleep(0.005)
    pytest.fail("background job did not become terminal")


def _registered_callback(name, register=compute.register_compute_callbacks):
    from dash import _callback

    matches = []
    for callback_data in _callback.GLOBAL_CALLBACK_MAP.values():
        callback_fn = callback_data.get("callback")
        if callback_fn is None:
            continue
        original = getattr(callback_fn, "__wrapped__", callback_fn)
        if original.__name__ == name:
            matches.append(original)
    if not matches:
        register()
        return _registered_callback(name, register)
    return matches[-1]


def test_job_ref_parser_rejects_invalid_and_unknown_kinds():
    rf = {
        "job_id": "rf-job",
        "generation": 4,
        "kind": "rf",
        "owner_id": None,
    }
    mds = {
        "job_id": "mds-job",
        "generation": 7,
        "kind": "mds",
        "owner_id": None,
    }

    assert job_reconcile.job_ref_from_store(rf).job_id == "rf-job"
    assert job_reconcile.job_ref_from_store(mds).job_id == "mds-job"
    assert job_reconcile.job_ref_from_store({"kind": "rf"}) is None
    assert job_reconcile.job_ref_from_store(
        {**rf, "kind": "unknown"},
    ) is None


def test_terminal_event_must_match_the_current_browser_generation():
    old_ref = {
        "job_id": "old-rf-job",
        "generation": 4,
        "kind": "rf",
        "owner_id": None,
    }
    current_ref = {
        "job_id": "current-rf-job",
        "generation": 5,
        "kind": "rf",
        "owner_id": None,
    }
    event = {
        **old_ref,
        "state": "succeeded",
        "terminal_revision": 1,
        "delivery_attempt": 2,
        "payload": {},
        "metadata": {},
    }

    assert job_reconcile.terminal_event_for_job(
        event,
        current_ref,
        expected_kind="rf",
    ) is None
    assert job_reconcile.terminal_event_for_job(
        event,
        old_ref,
        expected_kind="rf",
    ) == event


def test_rf_terminal_replays_until_applied_marker_is_acknowledged(monkeypatch):
    manager = JobManager(id_factory=lambda: "rf-test-job")
    register_calls = []
    expected_index = {
        "RF_001": {
            "n_trees": 2,
            "file_breakdown": {"run.trees": 2},
            "groups_per_file": {"run.trees": ["run"]},
            "is_rooted": True,
        }
    }
    monkeypatch.setattr(compute, "job_manager", manager)
    monkeypatch.setattr(job_reconcile, "job_manager", manager)
    monkeypatch.setattr(compute, "add_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute, "_wlog", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        compute,
        "register_distmat",
        lambda *args, **kwargs: register_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(compute, "get_distmat_index", lambda: expected_index)

    pipeline = {
        "rf_name": "RF_001",
        "result_names": ["run/tree-1", "run/tree-2"],
        "file_breakdown": {"run.trees": 2},
        "groups_per_file": {"run.trees": ["run"]},
        "total_elapsed": 1.25,
        "compute_elapsed": 1.0,
        "is_rooted": True,
        "save_path": "/tmp/rf-test.npy",
    }
    with ThreadPoolExecutor(max_workers=1) as executor:
        ref = manager.submit(
            executor,
            "rf",
            lambda: pipeline,
            finalizer=compute._finalize_rf_job,
        )
        terminal = _wait_for_terminal(manager, ref)

    assert terminal.state is JobState.SUCCEEDED
    assert len(register_calls) == 1
    assert terminal.terminal.payload["distmat_index"] == expected_index

    reconcile = _registered_callback(
        "reconcile_compute_job",
        job_reconcile.register_job_reconciliation_callbacks,
    )
    render = _registered_callback("render_rf_mds_terminal_event")
    acknowledge = _registered_callback(
        "acknowledge_terminal_receipt",
        job_reconcile.register_job_reconciliation_callbacks,
    )

    first_reconcile = reconcile(
        10,
        ref.as_dict(),
        None,
        None,
        None,
        None,
        None,
        {"busy": False},
        None,
        None,
    )
    first = render(first_reconcile[1], ref.as_dict(), None, False)
    second_reconcile = reconcile(
        11,
        ref.as_dict(),
        None,
        None,
        None,
        None,
        None,
        first_reconcile[2],
        None,
        None,
    )
    second = render(second_reconcile[1], ref.as_dict(), None, False)

    assert len(first_reconcile) == 7
    assert first_reconcile[0] is False
    assert first_reconcile[2]["busy"] is True
    assert first_reconcile[3:5] == (100.0, "complete")
    assert len(first) == 12
    assert first[1] == expected_index
    assert first[2] is False
    assert first[8]["terminal_revision"] == terminal.terminal.revision
    assert first[8]["delivery_attempt"] == 1
    assert second[8]["delivery_attempt"] == 2
    assert len(register_calls) == 1
    assert manager.snapshot(ref).acknowledged is False

    ack_store = acknowledge([second[8], None, None, None, None])
    assert ack_store["acknowledged"] is True
    assert manager.snapshot(ref).acknowledged is True
    assert manager.active_ref() is None

    settled = reconcile(
        12,
        ref.as_dict(),
        None,
        None,
        None,
        None,
        None,
        first_reconcile[2],
        None,
        None,
    )
    assert settled[0] is True
    assert settled[2] == {"busy": False}


def test_mds_finalization_stores_full_result_once_and_replays_small_index(
    monkeypatch,
):
    manager = JobManager(id_factory=lambda: "mds-test-job")
    stored = {}
    expected_index = {
        "RF_002_MDS.tsv": {
            "filename": "RF_002_MDS.tsv",
            "source_distmat": "RF_002",
            "rows": 4,
            "dimensions": ["MDS1", "MDS2"],
            "groups": ["run-a", "run-b"],
            "MIN_TREENUM": 1,
            "MAX_TREENUM": 2,
        }
    }
    monkeypatch.setattr(compute, "job_manager", manager)
    monkeypatch.setattr(compute, "add_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        compute,
        "store_mds_result",
        lambda key, value: stored.__setitem__(key, value),
    )
    monkeypatch.setattr(compute, "get_mds_results_index", lambda: expected_index)

    result = {
        "embedding": [
            [0.0, 0.1],
            [0.2, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
        ],
        "elapsed": 2.5,
        "tree_names": [
            "run-a/tree-1",
            "run-a/tree-2",
            "run-b/tree-1",
            "run-b/tree-2",
        ],
        "selected_distmat": "RF_002",
        "n_components": 2,
    }
    with ThreadPoolExecutor(max_workers=1) as executor:
        ref = manager.submit(
            executor,
            "mds",
            lambda: result,
            finalizer=compute._finalize_mds_job,
        )
        terminal = _wait_for_terminal(manager, ref)

    assert list(stored) == ["RF_002_MDS.tsv"]
    assert len(stored["RF_002_MDS.tsv"]["data"]) == 4
    assert terminal.terminal.payload["mds_results_index"] == expected_index
    assert "embedding" not in terminal.terminal.payload
    assert "data" not in terminal.terminal.payload

    manager.snapshot_for_delivery(ref)
    manager.snapshot_for_delivery(ref)
    assert list(stored) == ["RF_002_MDS.tsv"]


def test_reset_waits_for_an_inflight_result_publication(monkeypatch):
    manager = JobManager(id_factory=lambda: "reset-test-job")
    publication_started = threading.Event()
    release_publication = threading.Event()
    reset_finished = threading.Event()
    register_calls = []

    def blocking_register(*args, **kwargs):
        publication_started.set()
        assert release_publication.wait(timeout=2)
        register_calls.append((args, kwargs))

    monkeypatch.setattr(compute, "job_manager", manager)
    monkeypatch.setattr(compute, "add_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compute, "register_distmat", blocking_register)
    monkeypatch.setattr(compute, "get_distmat_index", lambda: {})
    monkeypatch.setattr(
        compute.persistent_worker,
        "cancel_current_job",
        lambda: False,
    )

    pipeline = {
        "rf_name": "RF_003",
        "result_names": ["run/tree-1", "run/tree-2"],
        "file_breakdown": {"run.trees": 2},
        "groups_per_file": {"run.trees": ["run"]},
        "total_elapsed": 1.0,
        "compute_elapsed": 0.8,
        "is_rooted": True,
        "save_path": "/tmp/rf-reset-test.npy",
    }
    executor = ThreadPoolExecutor(max_workers=1)
    ref = manager.submit(
        executor,
        "rf",
        lambda: pipeline,
        finalizer=compute._finalize_rf_job,
    )
    assert publication_started.wait(timeout=2)

    def run_reset():
        compute.reset()
        reset_finished.set()

    reset_thread = threading.Thread(target=run_reset)
    reset_thread.start()
    deadline = time.monotonic() + 2
    while manager.snapshot(ref) is not None and time.monotonic() < deadline:
        time.sleep(0.005)

    assert manager.snapshot(ref) is None
    assert reset_finished.wait(timeout=0.05) is False
    release_publication.set()
    assert reset_finished.wait(timeout=2)
    reset_thread.join(timeout=2)
    executor.shutdown(wait=True)

    assert len(register_calls) == 1
