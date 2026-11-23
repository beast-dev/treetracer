"""Integration tests for sister-clade instability in the worker and Compare Consensus callbacks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import pytest

from treetracer import state
from treetracer.background_jobs import JobManager, JobState
from treetracer.callbacks import clade_explore
from treetracer.clade_freq._subprocess_worker import (
    compute_clade_frequencies_worker_entry,
)
from treetracer.rf import rf_distance_with_rooted_facts_from_newick_iter
from treetracer.rf.rooted_facts import rooted_facts_npz_payload
from treetracer.rf.sparse_snapshots import (
    SparseSnapshot,
    sparse_snapshot_npz_payload,
)


@pytest.fixture(autouse=True)
def _clean_analysis_results():
    state.clear_all_analysis_results()
    yield
    state.clear_all_analysis_results()


def _write_6taxon_rooted_npz(path):
    newicks = [
        "((((A:1,B:1):1,R:1):1,C:1):1,(D:1,E:1):1);",
        "((((A:1,B:1):1,R:1):1,C:1):1,(D:1,E:1):1);",
        "(((A:1,B:1):1,C:1):1,((D:1,E:1):1,R:1):1);",
        "(((A:1,B:1):1,C:1):1,(D:1,(E:1,R:1):1):1);",
    ]
    names = [f"tree-{i}" for i in range(len(newicks))]
    _, _, facts = rf_distance_with_rooted_facts_from_newick_iter(
        names, iter(newicks), [{}], [0] * len(newicks)
    )
    np.savez(path, **rooted_facts_npz_payload(facts))
    return facts


def test_worker_rooted_vs_unrooted(tmp_path):
    """Rooted facts returns has_instability=True + sorted tip_rows; generic CSR degrades cleanly."""
    rooted_path = tmp_path / "rooted.npz"
    facts = _write_6taxon_rooted_npz(rooted_path)
    all_cols = list(range(facts.n_clades))

    rooted = compute_clade_frequencies_worker_entry(
        snapshots_path=str(rooted_path),
        columns=all_cols,
        counts_1=None,
        counts_2=None,
        n_trees_1=0,
        n_trees_2=0,
        tree_names_1=["tree-0", "tree-1"],
        tree_names_2=["tree-2", "tree-3"],
        full_distmat_names=list(facts.tree_names),
        row_indices_1=[0, 1],
        row_indices_2=[2, 3],
    )
    assert rooted["has_instability"] is True
    assert len(rooted["tip_rows"]) == 6
    assert rooted["tip_rows"][0]["taxon"] == "R"
    assert rooted["tip_rows"][0]["between_shift"] == pytest.approx(0.875, abs=1e-12)
    assert all(row["clade_instability"] is not None for row in rooted["rows"])

    # Omitting row_indices preserves backward compatibility and disables instability cleanly.
    no_rows = compute_clade_frequencies_worker_entry(
        snapshots_path=str(rooted_path),
        columns=all_cols,
        counts_1=[1] * len(all_cols),
        counts_2=[1] * len(all_cols),
        n_trees_1=2,
        n_trees_2=2,
    )
    assert no_rows["has_instability"] is False
    assert no_rows["tip_rows"] == []
    assert all(row["clade_instability"] is None for row in no_rows["rows"])

    # Generic unrooted CSR snapshot degrades cleanly to has_instability=False.
    csr_path = tmp_path / "unrooted.npz"
    bits = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    sparse = SparseSnapshot(
        tree_names=("t0", "t1"),
        leaf_names=("A", "B", "C", "D"),
        n_clades=2,
        packed_clades=np.packbits(bits, axis=1, bitorder="little"),
        row_offsets=np.array([0, 2, 4], dtype=np.uint64),
        column_indices=np.array([0, 1, 0, 1], dtype=np.uint32),
        rooted=False,
    )
    np.savez(csr_path, **sparse_snapshot_npz_payload(sparse))
    unrooted = compute_clade_frequencies_worker_entry(
        snapshots_path=str(csr_path),
        columns=[0, 1],
        counts_1=[1, 1],
        counts_2=[1, 1],
        n_trees_1=1,
        n_trees_2=1,
        row_indices_1=[0],
        row_indices_2=[1],
    )
    assert unrooted["has_instability"] is False
    assert unrooted["tip_rows"] == []
    assert all(row["clade_instability"] is None for row in unrooted["rows"])


def test_finalizer_excludes_singletons_and_caches_instability(monkeypatch):
    """Finalizer drops clade_size==1 from scatter records and caches tip_rows server-side."""
    manager = JobManager(id_factory=lambda: "instab-job", clock=lambda: 10.0)
    monkeypatch.setattr(clade_explore, "job_manager", manager)
    monkeypatch.setattr(clade_explore, "add_log", lambda *_a, **_k: None)

    context = clade_explore._CladeComparisonFinalizationContext(
        source_distmat="RF_001",
        uid_1="uid-1",
        uid_2="uid-2",
        label_1="G1",
        label_2="G2",
        consensus_columns_1=frozenset({0, 1, 2}),
        consensus_columns_2=frozenset({0, 1, 3}),
        min_clade_size=2,
        exclusive_only=False,
    )
    worker_result = {
        "rows": [
            {
                "split_key": (0,),
                "column_j": 0,
                "freq_1": 1.0,
                "freq_2": 1.0,
                "clade_size": 1,
                "clade_instability": 0.5625,
                "instab_1": 0.0,
                "instab_2": 0.25,
                "between_shift": 0.875,
            },
            {
                "split_key": (0, 1),
                "column_j": 1,
                "freq_1": 1.0,
                "freq_2": 1.0,
                "clade_size": 2,
                "clade_instability": 0.5,
                "instab_1": 0.0,
                "instab_2": 0.0,
                "between_shift": 1.0,
            },
            {
                "split_key": (0, 1, 2),
                "column_j": 2,
                "freq_1": 1.0,
                "freq_2": 0.0,
                "clade_size": 3,
                "clade_instability": 0.0,
                "instab_1": 0.0,
                "instab_2": None,
                "between_shift": None,
            },
        ],
        "leaf_names": ["A", "B", "C"],
        "has_instability": True,
        "tip_rows": [
            {
                "taxon_index": 0,
                "taxon": "A",
                "column_j": 0,
                "between_shift": 0.875,
                "instab_1": 0.0,
                "instab_2": 0.25,
                "combined_instability": 0.5625,
                "n_sisters_1": 1,
                "n_sisters_2": 2,
                "n_sisters_comb": 3,
                "p_cherry_1": 0.0,
                "p_cherry_2": 0.5,
                "top_sister_1": "{B, C} 100%",
                "top_sister_2": "B (cherry) 50%, C (cherry) 50%",
            }
        ],
        "snapshot_input_mode": "rooted_facts",
        "elapsed": 0.01,
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
        while manager.snapshot(ref).terminal is None:
            pass
        terminal = manager.snapshot(ref)

    assert terminal.state is JobState.SUCCEEDED
    assert "tip_rows" not in terminal.terminal.payload

    cached = state.get_clade_frequency_result(ref.job_id)
    assert cached["has_instability"] is True
    assert len(cached["tip_rows"]) == 1
    assert cached["label_1"] == "G1"
    assert cached["label_2"] == "G2"
    # Singleton column 0 was excluded globally; only clade_size >= 2 remain.
    assert [r["clade_size"] for r in cached["records"]] == [2, 3]
    assert cached["records"][0]["clade_instability"] == pytest.approx(0.5)
    assert "(Shift: 1.000)" in cached["records"][0]["instability_hover"]
    assert "Shift:" not in cached["records"][1]["instability_hover"]
    assert "Sister instability: 0.000" in cached["records"][1]["instability_hover"]
    # Explicit dict title ensures downstream Patch on colorbar.title.text works.
    assert cached["figure"]["data"][0]["marker"]["colorbar"]["title"] == {"text": "Clade size"}
