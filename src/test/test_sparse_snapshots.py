"""Parity and persistence tests for sparse-only RF snapshots."""

from __future__ import annotations

import numpy as np
import pytest
import rapidtrees

from treetracer.rf import (
    rf_distance_with_rooted_facts_from_newick_iter,
    rf_distance_with_sparse_snapshots_from_newick_iter,
    rf_distance_with_snapshots_from_newick_iter,
)
from treetracer.rf.sparse_snapshots import (
    count_sparse_columns,
    dense_clade_bits_from_sparse,
    dense_presence_from_sparse,
    sparse_clade_tip_indices,
    sparse_snapshot_from_npz,
    sparse_snapshot_from_rooted_facts,
    sparse_snapshot_npz_payload,
)
from treetracer.rf._worker import compute_rf
from treetracer.rf.rooted_facts import rooted_facts_npz_payload


pytestmark = pytest.mark.skipif(
    not hasattr(rapidtrees, "pairwise_rf_with_sparse_snapshots_from_newick_iter"),
    reason="installed RapidTrees does not provide sparse snapshots",
)

TREES = (
    "((A:1,B:1):1,(C:1,D:1):1);",
    "((A:1,C:1):1,(B:1,D:1):1);",
    "((A:1,D:1):1,(B:1,C:1):1);",
)
NAMES = tuple(f"tree-{index}" for index in range(len(TREES)))


@pytest.mark.parametrize("rooted", [False, True])
def test_sparse_wrapper_reconstructs_established_dense_snapshot(rooted):
    dense = rf_distance_with_snapshots_from_newick_iter(
        list(NAMES),
        iter(TREES),
        [{}],
        [0] * len(TREES),
        rooted=rooted,
    )
    sparse_names, sparse_rf, sparse = (
        rf_distance_with_sparse_snapshots_from_newick_iter(
            list(NAMES),
            iter(TREES),
            [{}],
            [0] * len(TREES),
            rooted=rooted,
        )
    )

    dense_names, dense_rf, presence, leaves, n_clades, clade_bits = dense
    assert sparse_names == dense_names
    np.testing.assert_array_equal(sparse_rf, dense_rf)
    assert sparse.tree_names == NAMES
    assert sparse.leaf_names == tuple(leaves)
    assert sparse.n_clades == n_clades
    assert sparse.rooted is rooted
    np.testing.assert_array_equal(
        dense_presence_from_sparse(sparse),
        presence,
    )
    np.testing.assert_array_equal(
        dense_clade_bits_from_sparse(sparse),
        clade_bits,
    )
    np.testing.assert_array_equal(
        count_sparse_columns(sparse, [0, 2]),
        presence[[0, 2]].sum(axis=0),
    )
    assert sparse_clade_tip_indices(sparse) == [
        tuple(np.flatnonzero(row)) for row in clade_bits
    ]


def test_sparse_snapshot_npz_round_trip(tmp_path):
    _, _, sparse = rf_distance_with_sparse_snapshots_from_newick_iter(
        list(NAMES),
        iter(TREES),
        [{}],
        [0] * len(TREES),
        rooted=False,
    )
    path = tmp_path / "sparse_snapshot.npz"
    np.savez(path, **sparse_snapshot_npz_payload(sparse))

    with np.load(path, allow_pickle=False) as persisted:
        loaded = sparse_snapshot_from_npz(persisted)

    assert loaded.tree_names == sparse.tree_names
    assert loaded.leaf_names == sparse.leaf_names
    assert loaded.n_clades == sparse.n_clades
    assert loaded.rooted is False
    np.testing.assert_array_equal(loaded.packed_clades, sparse.packed_clades)
    np.testing.assert_array_equal(loaded.row_offsets, sparse.row_offsets)
    np.testing.assert_array_equal(
        loaded.column_indices,
        sparse.column_indices,
    )


def test_rooted_facts_are_a_sparse_presence_snapshot():
    _, _, rooted_facts = rf_distance_with_rooted_facts_from_newick_iter(
        list(NAMES),
        iter(TREES),
        [{}],
        [0] * len(TREES),
    )
    sparse = sparse_snapshot_from_rooted_facts(rooted_facts)

    assert sparse.rooted is True
    assert sparse.n_entries == rooted_facts.clade_columns.size
    np.testing.assert_array_equal(
        sparse.column_indices,
        rooted_facts.clade_columns.reshape(-1),
    )
    assert all(
        len(sparse.row_columns(row)) == rooted_facts.nodes_per_tree
        for row in range(sparse.n_trees)
    )


def test_rooted_facts_sparse_loader_skips_heights_and_splits():
    _, _, rooted_facts = rf_distance_with_rooted_facts_from_newick_iter(
        list(NAMES),
        iter(TREES),
        [{}],
        [0] * len(TREES),
    )
    payload = rooted_facts_npz_payload(rooted_facts)
    forbidden = {
        "rooted_facts_node_heights",
        "rooted_facts_root_heights",
        "rooted_facts_split_ids",
        "rooted_facts_split_table",
    }

    class TrackingSnapshot:
        files = list(payload)

        def __init__(self):
            self.accessed = set()

        def __getitem__(self, key):
            self.accessed.add(key)
            if key in forbidden:
                raise AssertionError(f"unexpected heavy facts load: {key}")
            return payload[key]

    persisted = TrackingSnapshot()
    sparse = sparse_snapshot_from_npz(persisted)

    assert sparse.tree_names == rooted_facts.tree_names
    assert sparse.n_entries == rooted_facts.clade_columns.size
    assert persisted.accessed.isdisjoint(forbidden)


def test_count_sparse_columns_can_limit_work_to_requested_columns():
    _, _, sparse = rf_distance_with_sparse_snapshots_from_newick_iter(
        list(NAMES),
        iter(TREES),
        [{}],
        [0] * len(TREES),
        rooted=True,
    )
    presence = dense_presence_from_sparse(sparse)
    requested = [1, sparse.n_clades - 1]

    counts = count_sparse_columns(
        sparse,
        [0, 1],
        columns=requested,
    )

    np.testing.assert_array_equal(
        counts,
        presence[[0, 1]][:, requested].sum(axis=0),
    )


def test_rf_worker_writes_only_sparse_unrooted_snapshot(tmp_path):
    matrix_path = tmp_path / "RF_SPARSE.npy"

    result_names, _, details = compute_rf(
        list(NAMES),
        list(TREES),
        [{}],
        [0] * len(TREES),
        str(matrix_path),
        is_rooted=False,
    )

    assert result_names == list(NAMES)
    assert details["sparse_snapshot_used"] is True
    assert details["sparse_snapshot_source"] == "sparse_endpoint"
    snapshot_path = tmp_path / "RF_SPARSE_snapshots.npz"
    with np.load(snapshot_path, allow_pickle=False) as persisted:
        sparse = sparse_snapshot_from_npz(persisted)
        assert sparse.tree_names == NAMES
        assert sparse.rooted is False
        assert "presence" not in persisted.files
        assert "bipartition_bits" not in persisted.files
        assert "leaf_names" not in persisted.files


def test_rf_worker_does_not_fall_back_to_dense_endpoint(monkeypatch, tmp_path):
    monkeypatch.delattr(
        rapidtrees,
        "pairwise_rf_with_sparse_snapshots_from_newick_iter",
    )

    with pytest.raises(RuntimeError, match="RapidTrees 0.9.1 or newer"):
        compute_rf(
            list(NAMES),
            list(TREES),
            [{}],
            [0] * len(TREES),
            str(tmp_path / "RF_NO_SPARSE.npy"),
            is_rooted=False,
        )

    assert not (tmp_path / "RF_NO_SPARSE.npy").exists()
    assert not (tmp_path / "RF_NO_SPARSE_snapshots.npz").exists()
