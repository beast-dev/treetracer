"""Managed lifecycle and worker tests for Stage 4 analysis jobs."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from treetracer import state
from treetracer.background_jobs import JobManager, JobState
from treetracer.callbacks import clade_explore, diagnostics, job_reconcile
from treetracer.clade_freq._subprocess_worker import (
    compute_clade_frequencies_worker_entry,
)
from treetracer.ess._rf_trace_worker import compute_rf_trace_worker_entry


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


@pytest.fixture(autouse=True)
def _clean_analysis_results():
    state.clear_all_analysis_results()
    yield
    state.clear_all_analysis_results()


def test_rf_trace_worker_memory_maps_and_returns_only_requested_row(tmp_path):
    matrix = np.arange(25, dtype=np.uint16).reshape(5, 5)
    path = tmp_path / "matrix.npy"
    np.save(path, matrix)

    result = compute_rf_trace_worker_entry(
        matrix_path=str(path),
        reference_index=3,
    )

    assert result["matrix_size"] == 5
    assert result["distances"] == matrix[3].tolist()
    assert "matrix" not in result


def test_clade_worker_decodes_only_requested_columns(tmp_path):
    path = tmp_path / "snapshot.npz"
    presence = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 1],
        ],
        dtype=np.uint8,
    )
    bits = np.array(
        [
            [1, 0, 0],
            [0, 1, 1],
            [1, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.uint8,
    )
    np.savez(
        path,
        presence=presence,
        bipartition_bits=bits,
        leaf_names=np.array(["'A'", "B", "C"]),
    )

    result = compute_clade_frequencies_worker_entry(
        snapshots_path=str(path),
        columns=[1, 3],
        counts_1=[1, 1],
        counts_2=[2, 1],
        n_trees_1=2,
        n_trees_2=2,
    )

    assert result["leaf_names"] == ["A", "B", "C"]
    assert {row["column_j"] for row in result["rows"]} == {1, 3}
    by_column = {row["column_j"]: row for row in result["rows"]}
    assert by_column[1]["split_key"] == (1, 2)
    assert by_column[1]["freq_1"] == pytest.approx(0.5)
    assert by_column[1]["freq_2"] == pytest.approx(1.0)
    assert by_column[3]["split_key"] == (2,)


def test_rf_trace_terminal_replays_cached_render_until_ack(monkeypatch):
    now = [30.0]
    manager = JobManager(
        id_factory=lambda: "rf-trace-test-job",
        clock=lambda: now[0],
    )
    db = SimpleNamespace(
        _trees=pd.DataFrame(
            {
                "name": ["run-a/tree-1", "run-a/tree-2", "run-b/tree-1"],
                "file_source": ["a.trees", "a.trees", "b.trees"],
            }
        ),
        flush=lambda: None,
    )
    fake_figure = SimpleNamespace(to_dict=lambda: {"data": [], "layout": {}})
    monkeypatch.setattr(diagnostics, "job_manager", manager)
    monkeypatch.setattr(diagnostics, "add_log", lambda *_a, **_k: None)
    monkeypatch.setattr(
        diagnostics,
        "get_tree_service",
        lambda: SimpleNamespace(db_manager=db),
    )
    monkeypatch.setattr(
        diagnostics,
        "_build_rf_trace_fig",
        lambda *_a, **_k: fake_figure,
    )

    context = diagnostics._RfTraceFinalizationContext(
        selected_matrix="RF_001",
        names=("run-a/tree-1", "run-a/tree-2", "run-b/tree-1"),
        reference_index=0,
        reference_name="run-a/tree-1",
        reference_group="run-a",
        reference_position="first",
        burnin=0,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        ref = manager.submit(
            executor,
            "rf_trace",
            lambda: {
                "distances": [0, 4, 8],
                "matrix_size": 3,
                "elapsed": 0.01,
            },
            finalizer=partial(
                diagnostics._finalize_rf_trace_job,
                context=context,
            ),
        )
        terminal = _wait_for_terminal(manager, ref)

    assert terminal.state is JobState.SUCCEEDED
    assert terminal.terminal.payload["result_key"] == ref.job_id
    assert "distances" not in terminal.terminal.payload
    cached = state.get_rf_trace_result(ref.job_id)
    assert [row["rf_distance"] for row in cached["records"]] == [4, 8]

    render = _registered_callback(
        "render_rf_trace_terminal_event",
        diagnostics.register_diagnostics_callbacks,
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
    assert len(first) == 5
    assert len(first[1]) == 2
    assert first[3] is False
    assert first[4]["delivery_attempt"] == 1
    assert second[4]["delivery_attempt"] == 2

    assert manager.acknowledge(ref, second[4]["terminal_revision"]) is True
    assert manager.snapshot(ref).acknowledged is True


def test_stage_four_submit_callbacks_have_matching_idle_output_shapes():
    rf_submit = _registered_callback(
        "compute_rf_trace",
        diagnostics.register_diagnostics_callbacks,
    )
    clade_submit = _registered_callback(
        "compute_and_plot_clade_frequencies",
        clade_explore.register_clade_explore_callbacks,
    )

    assert len(rf_submit(None, None, None, None, None, None)) == 5
    assert len(clade_submit(None, None, None, None)) == 7


def test_clade_terminal_keeps_rows_server_side_and_keys_click_resolution(
    monkeypatch,
):
    now = [40.0]
    manager = JobManager(
        id_factory=lambda: "clade-test-job",
        clock=lambda: now[0],
    )
    fake_figure = SimpleNamespace(to_dict=lambda: {"data": [], "layout": {}})
    monkeypatch.setattr(clade_explore, "job_manager", manager)
    monkeypatch.setattr(clade_explore, "add_log", lambda *_a, **_k: None)
    monkeypatch.setattr(
        clade_explore,
        "_build_scatter_fig",
        lambda *_a, **_k: fake_figure,
    )

    context = clade_explore._CladeComparisonFinalizationContext(
        source_distmat="RF_001",
        uid_1="uid-a",
        uid_2="uid-b",
        label_1="Consensus A",
        label_2="Consensus B",
        consensus_columns_1=frozenset({2}),
        consensus_columns_2=frozenset({2, 5}),
        min_clade_size=2,
    )
    worker_result = {
        "rows": [
            {
                "split_key": (0, 2),
                "column_j": 2,
                "freq_1": 0.8,
                "freq_2": 0.6,
                "clade_size": 2,
            },
            {
                "split_key": (1,),
                "column_j": 5,
                "freq_1": 0.0,
                "freq_2": 0.4,
                "clade_size": 1,
            },
        ],
        "leaf_names": ["A", "B", "C"],
        "elapsed": 0.02,
    }
    with ThreadPoolExecutor(max_workers=1) as executor:
        ref = manager.submit(
            executor,
            "clade_compare",
            lambda: worker_result,
            finalizer=partial(
                clade_explore._finalize_clade_comparison_job,
                context=context,
            ),
        )
        terminal = _wait_for_terminal(manager, ref)

    assert terminal.state is JobState.SUCCEEDED
    assert "rows" not in terminal.terminal.payload
    assert "leaf_names" not in terminal.terminal.payload
    resolved = clade_explore._resolved_clade(
        {"result_key": ref.job_id, "split_id": 0},
        expected_pair=("uid-a", "uid-b"),
    )
    assert resolved == {
        "source_distmat": "RF_001",
        "column_j": 2,
        "tip_names": ("A", "C"),
    }
    assert clade_explore._resolved_clade(
        {"result_key": ref.job_id, "split_id": 0},
        expected_pair=("uid-b", "uid-a"),
    ) is None

    render = _registered_callback(
        "render_clade_frequency_terminal_event",
        clade_explore.register_clade_explore_callbacks,
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
    assert len(first) == 6
    assert len(first[1]) == 2
    assert first[2] == {}
    assert first[3] == ref.job_id
    assert first[4] is None
    assert first[5]["delivery_attempt"] == 1
    assert second[5]["delivery_attempt"] == 2

    assert manager.acknowledge(ref, second[5]["terminal_revision"]) is True
    assert manager.snapshot(ref).acknowledged is True
