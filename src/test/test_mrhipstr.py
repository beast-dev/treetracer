"""Focused tests for the MrHIPSTR implementation."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from treetracer.consensus_tree.mrhipstr import (
    MAJORITY_RULE_REWARD,
    SourceTreeRecord,
    compute_hipstr_topology,
    compute_mrhipstr_topology,
    count_selected_clades,
    decode_rooted_clade_catalog,
    ingest_source_trees,
)
from treetracer.rf import rf_distance_with_snapshots_from_newick_iter


def _bits(n_taxa: int, *taxon_indices: int) -> np.ndarray:
    row = np.zeros(n_taxa, dtype=np.uint8)
    row[list(taxon_indices)] = 1
    return row


def _snapshot_inputs(leaf_names, clades, presence):
    bipartition_bits = np.stack(
        [_bits(len(leaf_names), *clade) for clade in clades]
    )
    counts = count_selected_clades(
        np.asarray(presence, dtype=np.uint8),
        selected_rows=np.arange(len(presence)),
        chunk_rows=1,
    )
    catalog = decode_rooted_clade_catalog(
        bipartition_bits=bipartition_bits,
        leaf_names=leaf_names,
        active_columns=counts.active_columns,
    )
    return catalog, counts


def _topology_inputs(newicks, *, column_order=None, record_order=None):
    newicks = list(newicks)
    names = [f"tree-{index}" for index in range(len(newicks))]
    _, _, presence, leaf_names, _, bipartition_bits = (
        rf_distance_with_snapshots_from_newick_iter(
            names,
            iter(newicks),
            [{}],
            [0] * len(newicks),
            rooted=True,
        )
    )
    if column_order is not None:
        column_order = np.asarray(column_order, dtype=np.intp)
        presence = presence[:, column_order]
        bipartition_bits = bipartition_bits[column_order]

    counts = count_selected_clades(
        presence,
        selected_rows=np.arange(len(newicks)),
    )
    catalog = decode_rooted_clade_catalog(
        bipartition_bits=bipartition_bits,
        leaf_names=leaf_names,
        active_columns=counts.active_columns,
    )
    if record_order is None:
        record_order = range(len(newicks))
    records = [
        SourceTreeRecord(name=names[index], newick=newicks[index])
        for index in record_order
    ]
    summary = ingest_source_trees(
        records,
        catalog=catalog,
        snapshot_counts=counts,
    )
    return catalog, counts, summary


def _named_clade(catalog, *names):
    indices = {name: index for index, name in enumerate(catalog.leaf_names)}
    return sum(1 << indices[name] for name in names)


def test_count_selected_clades_sums_in_chunks_and_finds_active_columns():
    presence = np.array(
        [
            [1, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [1, 1, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 1, 0],
        ],
        dtype=np.uint8,
    )

    result = count_selected_clades(
        presence,
        selected_rows=[4, 0, 2],
        chunk_rows=2,
    )

    np.testing.assert_array_equal(result.counts, [3, 1, 2, 2, 0])
    np.testing.assert_array_equal(result.active_columns, [0, 1, 2, 3])
    assert result.n_trees == 3
    assert not result.counts.flags.writeable
    assert not result.active_columns.flags.writeable


def test_count_selected_clades_checks_only_selected_data_blocks():
    presence = np.array(
        [
            [1, 0],
            [2, 0],  # invalid, but deliberately outside this selection
        ],
        dtype=np.uint8,
    )

    result = count_selected_clades(presence, selected_rows=[0], chunk_rows=1)
    np.testing.assert_array_equal(result.counts, [1, 0])

    with pytest.raises(ValueError, match="only 0 or 1"):
        count_selected_clades(presence, selected_rows=[1], chunk_rows=1)


@pytest.mark.parametrize(
    ("presence", "selected_rows", "chunk_rows", "error", "message"),
    [
        (
            np.array([1, 0], dtype=np.uint8),
            [0],
            1,
            ValueError,
            "two-dimensional",
        ),
        (
            np.eye(2, dtype=np.float64),
            [0],
            1,
            TypeError,
            "uint8 or bool",
        ),
        (
            np.eye(2, dtype=np.uint8),
            [],
            1,
            ValueError,
            "empty",
        ),
        (
            np.eye(2, dtype=np.uint8),
            [0.0],
            1,
            TypeError,
            "integer row IDs",
        ),
        (
            np.eye(2, dtype=np.uint8),
            [0, 0],
            1,
            ValueError,
            "duplicate row",
        ),
        (
            np.eye(2, dtype=np.uint8),
            [2],
            1,
            IndexError,
            "outside",
        ),
        (
            np.eye(2, dtype=np.uint8),
            [0],
            0,
            ValueError,
            "positive integer",
        ),
    ],
)
def test_count_selected_clades_rejects_invalid_inputs(
    presence,
    selected_rows,
    chunk_rows,
    error,
    message,
):
    with pytest.raises(error, match=message):
        count_selected_clades(
            presence,
            selected_rows,
            chunk_rows=chunk_rows,
        )


def test_ingest_source_trees_collects_splits_and_exact_clade_mean_heights():
    catalog, counts = _snapshot_inputs(
        ["A", "B", "C"],
        [(0,), (1,), (2,), (0, 1)],
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
    )
    records = [
        SourceTreeRecord(
            name="tree-one",
            newick=(
                "[&R] ((1[&rate=1]:1,2:1)[&posterior=.9]:2,3:3):0;"
            ),
            translate={"1": "'A'", "2": "'B'", "3": "'C'"},
        ),
        SourceTreeRecord(
            name="tree-two",
            newick="[&R] ((2:2,3:2):2,1:4):0;",
            translate={"1": "C", "2": "A", "3": "B"},
        ),
    ]

    summary = ingest_source_trees(
        iter(records),
        catalog=catalog,
        snapshot_counts=counts,
    )

    assert summary.n_trees == 2
    assert summary.observed_splits == {
        0b011: ((0b001, 0b010),),
        0b111: ((0b011, 0b100),),
    }
    assert summary.observation_counts == {
        0b001: 2,
        0b010: 2,
        0b100: 2,
        0b011: 2,
        0b111: 2,
    }
    assert summary.height_sums[0b001] == pytest.approx(0.0)
    assert summary.height_sums[0b010] == pytest.approx(0.0)
    assert summary.height_sums[0b100] == pytest.approx(0.0)
    assert summary.height_sums[0b011] == pytest.approx(3.0)
    assert summary.height_sums[0b111] == pytest.approx(7.0)
    assert summary.mean_height(0b011) == pytest.approx(1.5)
    assert summary.mean_height(0b111) == pytest.approx(3.5)


def test_ingest_source_trees_keeps_heterochronous_tip_heights():
    catalog, counts = _snapshot_inputs(
        ["A", "B"],
        [(0,), (1,)],
        [[1, 1]],
    )

    summary = ingest_source_trees(
        [SourceTreeRecord(newick="(A:1,B:2):0;")],
        catalog=catalog,
        snapshot_counts=counts,
    )

    assert summary.mean_height(0b01) == pytest.approx(1.0)
    assert summary.mean_height(0b10) == pytest.approx(0.0)
    assert summary.mean_height(0b11) == pytest.approx(2.0)
    assert summary.observed_splits[0b11] == ((0b01, 0b10),)


def test_ingest_source_trees_records_only_observed_parent_child_splits():
    catalog, counts = _snapshot_inputs(
        ["A", "B", "C"],
        [(0,), (1,), (2,), (0, 1), (0, 2)],
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 1],
        ],
    )

    summary = ingest_source_trees(
        [
            SourceTreeRecord(newick="((A:1,B:1):1,C:2):0;"),
            SourceTreeRecord(newick="((A:1,C:1):1,B:2):0;"),
        ],
        catalog=catalog,
        snapshot_counts=counts,
    )

    assert summary.observed_splits == {
        0b011: ((0b001, 0b010),),
        0b101: ((0b001, 0b100),),
        0b111: (
            (0b010, 0b101),
            (0b011, 0b100),
        ),
    }


@pytest.mark.parametrize(
    ("newick", "message"),
    [
        ("(A,B:1):0;", "explicit branch length required"),
        ("(A:1,A:1):0;", "duplicate taxa"),
        ("(A:1,X:1):0;", "unexpected taxa"),
        ("(A:1,B:1,C:1):0;", "exactly two children"),
    ],
)
def test_ingest_source_trees_rejects_invalid_source_tree(newick, message):
    leaf_names = ["A", "B", "C"] if ",C:" in newick else ["A", "B"]
    catalog, counts = _snapshot_inputs(
        leaf_names,
        [(index,) for index in range(len(leaf_names))],
        [[1] * len(leaf_names)],
    )

    with pytest.raises(ValueError, match=message):
        ingest_source_trees(
            [SourceTreeRecord(name="bad-tree", newick=newick)],
            catalog=catalog,
            snapshot_counts=counts,
        )


def test_ingest_source_trees_rejects_clade_absent_from_snapshot_catalog():
    catalog, counts = _snapshot_inputs(
        ["A", "B", "C"],
        [(0,), (1,), (2,), (0, 2)],
        [[1, 1, 1, 1]],
    )

    with pytest.raises(ValueError, match="has no active RapidTrees"):
        ingest_source_trees(
            [SourceTreeRecord(newick="((A:1,B:1):1,C:2):0;")],
            catalog=catalog,
            snapshot_counts=counts,
        )


def test_ingest_source_trees_cross_checks_all_snapshot_clade_counts():
    catalog, counts = _snapshot_inputs(
        ["A", "B", "C"],
        [(0,), (1,), (2,), (0, 1)],
        [
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ],
    )

    with pytest.raises(ValueError, match=r"parsed=2, snapshot=1"):
        ingest_source_trees(
            [
                SourceTreeRecord(newick="((A:1,B:1):1,C:2):0;"),
                SourceTreeRecord(newick="((A:1,B:1):1,C:2):0;"),
            ],
            catalog=catalog,
            snapshot_counts=counts,
        )


def test_ingest_source_trees_requires_snapshot_selection_tree_count():
    catalog, counts = _snapshot_inputs(
        ["A", "B"],
        [(0,), (1,)],
        [[1, 1], [1, 1]],
    )

    with pytest.raises(ValueError, match="yielded 1 trees.*contains 2"):
        ingest_source_trees(
            [SourceTreeRecord(newick="(A:1,B:1):0;")],
            catalog=catalog,
            snapshot_counts=counts,
        )


def test_ingest_source_trees_is_iterative_for_deep_caterpillar():
    n_taxa = 1_101
    leaf_names = [f"T{index}" for index in range(n_taxa)]
    n_snapshot_clades = 2 * n_taxa - 2
    bipartition_bits = np.zeros(
        (n_snapshot_clades, n_taxa),
        dtype=np.uint8,
    )
    bipartition_bits[np.arange(n_taxa), np.arange(n_taxa)] = 1
    for clade_size in range(2, n_taxa):
        column = n_taxa + clade_size - 2
        bipartition_bits[column, :clade_size] = 1

    presence = np.ones((1, n_snapshot_clades), dtype=np.uint8)
    counts = count_selected_clades(presence, selected_rows=[0])
    catalog = decode_rooted_clade_catalog(
        bipartition_bits=bipartition_bits,
        leaf_names=leaf_names,
        active_columns=counts.active_columns,
    )

    newick = "(T0:1,T1:1):1"
    for index in range(2, n_taxa):
        newick = f"({newick},T{index}:1):1"
    summary = ingest_source_trees(
        [SourceTreeRecord(newick=newick + ";")],
        catalog=catalog,
        snapshot_counts=counts,
    )

    assert summary.n_trees == 1
    assert len(summary.observation_counts) == n_snapshot_clades + 1
    assert len(summary.observed_splits) == n_taxa - 1

    topology = compute_mrhipstr_topology(
        catalog=catalog,
        snapshot_counts=counts,
        source_summary=summary,
    )
    assert len(topology.selected_clades) == 2 * n_taxa - 1
    assert len(topology.selected_splits) == n_taxa - 1
    assert len(topology.cols_in_consensus_tree) == n_snapshot_clades


def test_hipstr_derivation_example_returns_unsampled_amalgamation():
    tree_1 = "(((A:1,B:1):1,C:2):1,(D:1,E:1):2):0;"
    tree_2 = "((((B:1,C:1):1,A:2):1,D:3):1,E:4):0;"
    tree_3 = "((((B:1,C:1):1,A:2):1,E:3):1,D:4):0;"
    catalog, counts, summary = _topology_inputs(
        [tree_1] * 3 + [tree_2] * 2 + [tree_3] * 2
    )

    result = compute_mrhipstr_topology(
        catalog=catalog,
        snapshot_counts=counts,
        source_summary=summary,
    )

    a = _named_clade(catalog, "A")
    b = _named_clade(catalog, "B")
    c = _named_clade(catalog, "C")
    d = _named_clade(catalog, "D")
    e = _named_clade(catalog, "E")
    bc = b | c
    de = d | e
    abc = a | b | c
    root = a | b | c | d | e
    assert result.selected_splits == {
        bc: tuple(sorted((b, c))),
        de: tuple(sorted((d, e))),
        abc: tuple(sorted((a, bc))),
        root: tuple(sorted((abc, de))),
    }
    assert result.log_clade_credibility == pytest.approx(math.log(12 / 49))
    assert result.majority_clade_count == 2
    assert result.objective_score == pytest.approx(
        math.log(12 / 49) + 3 * MAJORITY_RULE_REWARD
    )
    assert result.cols_in_consensus_tree == frozenset(
        catalog.column_by_clade[clade_bits]
        for clade_bits in result.selected_clades
        if clade_bits != catalog.root_bits
    )


def test_java_majority_reward_can_change_the_hipstr_topology():
    # AC occurs in three of five trees. Ordinary HIPSTR prefers A|(B,(C,D))
    # by clade product, while the Java 1E10 reward makes MrHIPSTR retain AC.
    tree_1 = "(A:1,(B:1,(C:1,D:1):1):1):0;"
    tree_2 = "((A:1,C:1):1,(B:1,D:1):1):0;"
    tree_3 = "(((A:1,C:1):1,B:1):1,D:1):0;"
    tree_4 = "(((A:1,C:1):1,D:1):1,B:1):0;"
    catalog, counts, summary = _topology_inputs(
        [tree_1, tree_1, tree_2, tree_3, tree_4]
    )

    hipstr = compute_hipstr_topology(
        catalog=catalog,
        snapshot_counts=counts,
        source_summary=summary,
        majority_rule=False,
    )
    mrhipstr = compute_mrhipstr_topology(
        catalog=catalog,
        snapshot_counts=counts,
        source_summary=summary,
    )

    ac = _named_clade(catalog, "A", "C")
    assert ac not in hipstr.selected_clades
    assert ac in mrhipstr.selected_clades
    assert hipstr.log_clade_credibility > mrhipstr.log_clade_credibility
    assert hipstr.majority_clade_count == 0
    assert mrhipstr.majority_clade_count == 1
    assert hipstr.objective_score == pytest.approx(
        hipstr.log_clade_credibility
    )
    assert mrhipstr.objective_score == pytest.approx(
        mrhipstr.log_clade_credibility + 2 * MAJORITY_RULE_REWARD
    )


def test_mrhipstr_uses_strict_majority_threshold_and_stable_ties():
    newicks = [
        "((A:1,B:1):1,C:2):0;",
        "((A:1,C:1):1,B:2):0;",
    ]
    catalog, counts, summary = _topology_inputs(newicks)
    reversed_summary = replace(
        summary,
        observed_splits={
            parent: tuple(reversed(splits))
            for parent, splits in reversed(tuple(summary.observed_splits.items()))
        },
    )

    result = compute_mrhipstr_topology(
        catalog=catalog,
        snapshot_counts=counts,
        source_summary=summary,
    )
    reordered = compute_mrhipstr_topology(
        catalog=catalog,
        snapshot_counts=counts,
        source_summary=reversed_summary,
    )

    assert result.selected_splits == reordered.selected_splits
    assert result.selected_splits[catalog.root_bits] == min(
        summary.observed_splits[catalog.root_bits]
    )
    assert result.majority_clade_count == 0
    assert result.log_clade_credibility == pytest.approx(math.log(0.5))
    # Only the always-present root receives the reward: a clade at exactly
    # 0.5 must not receive it.
    assert result.objective_score == pytest.approx(
        MAJORITY_RULE_REWARD + math.log(0.5)
    )


def test_mrhipstr_never_invents_an_unobserved_parent_child_split():
    catalog, counts, summary = _topology_inputs(
        [
            "(((A:1,B:1):1,C:1):1,D:1):0;",
            "(A:1,(B:1,(C:1,D:1):1):1):0;",
        ]
    )
    result = compute_mrhipstr_topology(
        catalog=catalog,
        snapshot_counts=counts,
        source_summary=summary,
    )

    ab = _named_clade(catalog, "A", "B")
    cd = _named_clade(catalog, "C", "D")
    hypothetical_unobserved_split = tuple(sorted((ab, cd)))
    assert hypothetical_unobserved_split not in summary.observed_splits[
        catalog.root_bits
    ]
    assert result.selected_splits[catalog.root_bits] != hypothetical_unobserved_split
    assert all(
        child_pair in summary.observed_splits[parent_bits]
        for parent_bits, child_pair in result.selected_splits.items()
    )


def test_mrhipstr_is_stable_under_source_and_snapshot_column_permutations():
    tree_1 = "(((A:1,B:1):1,C:2):1,(D:1,E:1):2):0;"
    tree_2 = "((((B:1,C:1):1,A:2):1,D:3):1,E:4):0;"
    tree_3 = "((((B:1,C:1):1,A:2):1,E:3):1,D:4):0;"
    newicks = [tree_1] * 3 + [tree_2] * 2 + [tree_3] * 2
    catalog, counts, summary = _topology_inputs(newicks)

    _, _, raw_presence, _, _, _ = rf_distance_with_snapshots_from_newick_iter(
        [f"tree-{index}" for index in range(len(newicks))],
        iter(newicks),
        [{}],
        [0] * len(newicks),
        rooted=True,
    )
    reverse_columns = np.arange(raw_presence.shape[1] - 1, -1, -1)
    permuted_catalog, permuted_counts, permuted_summary = _topology_inputs(
        newicks,
        column_order=reverse_columns,
        record_order=range(len(newicks) - 1, -1, -1),
    )

    result = compute_mrhipstr_topology(
        catalog=catalog,
        snapshot_counts=counts,
        source_summary=summary,
    )
    permuted = compute_mrhipstr_topology(
        catalog=permuted_catalog,
        snapshot_counts=permuted_counts,
        source_summary=permuted_summary,
    )

    assert result.selected_clades == permuted.selected_clades
    assert result.selected_splits == permuted.selected_splits
    assert result.objective_score == pytest.approx(permuted.objective_score)
    assert result.log_clade_credibility == pytest.approx(
        permuted.log_clade_credibility
    )
    assert result.majority_clade_count == permuted.majority_clade_count


def test_mrhipstr_rejects_missing_or_malformed_observed_splits():
    catalog, counts, summary = _topology_inputs(
        ["((A:1,B:1):1,C:2):0;"] * 2
    )
    without_root = replace(
        summary,
        observed_splits={
            parent: splits
            for parent, splits in summary.observed_splits.items()
            if parent != catalog.root_bits
        },
    )
    with pytest.raises(ValueError, match="has no observed child split"):
        compute_mrhipstr_topology(
            catalog=catalog,
            snapshot_counts=counts,
            source_summary=without_root,
        )

    a = _named_clade(catalog, "A")
    ab = _named_clade(catalog, "A", "B")
    malformed = replace(
        summary,
        observed_splits={
            **summary.observed_splits,
            catalog.root_bits: ((a, ab),),
        },
    )
    with pytest.raises(ValueError, match="overlap"):
        compute_mrhipstr_topology(
            catalog=catalog,
            snapshot_counts=counts,
            source_summary=malformed,
        )


def test_decode_rooted_clade_catalog_maps_columns_and_adds_implicit_root():
    rows = np.stack(
        [
            _bits(4, 0),
            _bits(4, 1),
            _bits(4, 2),
            _bits(4, 3),
            _bits(4, 0, 1),
            _bits(4, 2, 3),
            _bits(4, 0, 1, 2),
            _bits(4, 0, 2),  # inactive in the selected posterior sample
        ]
    )

    catalog = decode_rooted_clade_catalog(
        bipartition_bits=rows,
        leaf_names=np.array(["'A'", '"B"', "C", "D"]),
        active_columns=[6, 4, 0, 5, 2, 3, 1],
    )

    assert catalog.leaf_names == ("A", "B", "C", "D")
    assert catalog.n_taxa == 4
    assert catalog.root_bits == 0b1111
    assert catalog.clade_by_column == {
        0: 0b0001,
        1: 0b0010,
        2: 0b0100,
        3: 0b1000,
        4: 0b0011,
        5: 0b1100,
        6: 0b0111,
    }
    assert catalog.column_by_clade[0b0011] == 4
    assert catalog.root_bits not in catalog.column_by_clade
    assert 7 not in catalog.clade_by_column
    assert catalog.clades == (
        0b0001,
        0b0010,
        0b0100,
        0b1000,
        0b0011,
        0b1100,
        0b0111,
        0b1111,
    )


def test_decode_rooted_clade_catalog_supports_more_than_64_taxa():
    n_taxa = 70
    rows = [_bits(n_taxa, index) for index in range(n_taxa)]
    rows.append(_bits(n_taxa, 0, 64, 69))

    catalog = decode_rooted_clade_catalog(
        bipartition_bits=np.stack(rows),
        leaf_names=[f"taxon-{index}" for index in range(n_taxa)],
        active_columns=np.arange(len(rows)),
    )

    wide_clade = (1 << 0) | (1 << 64) | (1 << 69)
    assert catalog.clade_by_column[70] == wide_clade
    assert catalog.column_by_clade[wide_clade] == 70
    assert catalog.root_bits == (1 << 70) - 1


def test_decode_rooted_clade_catalog_matches_rapidtrees_rooted_snapshot():
    newicks = [
        "(A:1,(B:1,(C:1,D:1):1):1):0;",
        "((A:1,B:1):1,(C:1,D:1):1):0;",
    ]
    _, _, presence, leaf_names, _, bipartition_bits = (
        rf_distance_with_snapshots_from_newick_iter(
            ["tree-1", "tree-2"],
            iter(newicks),
            [{}],
            [0, 0],
            rooted=True,
        )
    )
    selection = count_selected_clades(
        presence,
        selected_rows=[1, 0],
        chunk_rows=1,
    )

    catalog = decode_rooted_clade_catalog(
        bipartition_bits=bipartition_bits,
        leaf_names=leaf_names,
        active_columns=selection.active_columns,
    )

    np.testing.assert_array_equal(selection.counts, presence.sum(axis=0))
    assert selection.n_trees == 2
    assert catalog.leaf_names == ("A", "B", "C", "D")
    assert catalog.root_bits == 0b1111
    assert catalog.root_bits not in catalog.column_by_clade
    assert {1 << index for index in range(4)} <= set(catalog.clades)
    assert presence.sum(axis=1).tolist() == [6, 6]

    summary = ingest_source_trees(
        [SourceTreeRecord(newick=newick) for newick in newicks],
        catalog=catalog,
        snapshot_counts=selection,
    )
    assert summary.n_trees == 2
    for clade_bits, column in catalog.column_by_clade.items():
        assert summary.observation_counts[clade_bits] == selection.counts[column]
    assert summary.observation_counts[catalog.root_bits] == 2


def test_decode_rooted_clade_catalog_rejects_missing_singletons():
    rows = np.stack(
        [
            _bits(3, 0),
            _bits(3, 1),
            _bits(3, 0, 1),
        ]
    )

    with pytest.raises(ValueError, match="missing active singleton.*'C'"):
        decode_rooted_clade_catalog(
            bipartition_bits=rows,
            leaf_names=["A", "B", "C"],
            active_columns=[0, 1, 2],
        )


def test_decode_rooted_clade_catalog_rejects_explicit_root():
    rows = np.stack(
        [
            _bits(2, 0),
            _bits(2, 1),
            _bits(2, 0, 1),
        ]
    )

    with pytest.raises(ValueError, match="all-taxa root"):
        decode_rooted_clade_catalog(
            bipartition_bits=rows,
            leaf_names=["A", "B"],
            active_columns=[0, 1, 2],
        )


@pytest.mark.parametrize(
    ("rows", "leaf_names", "active_columns", "error", "message"),
    [
        (
            np.array([1, 0], dtype=np.uint8),
            ["A", "B"],
            [0],
            ValueError,
            "two-dimensional",
        ),
        (
            np.eye(2, dtype=np.uint8),
            ["A"],
            [0, 1],
            ValueError,
            "width does not match",
        ),
        (
            np.eye(2, dtype=np.float64),
            ["A", "B"],
            [0, 1],
            TypeError,
            "uint8 or bool",
        ),
        (
            np.eye(2, dtype=np.uint8),
            ["A", "B"],
            [0, 2],
            IndexError,
            "outside",
        ),
        (
            np.eye(2, dtype=np.uint8),
            ["A", "B"],
            [0, 0, 1],
            ValueError,
            "duplicate column",
        ),
        (
            np.array([[1, 0], [1, 0], [0, 1]], dtype=np.uint8),
            ["A", "B"],
            [0, 1, 2],
            ValueError,
            "same clade",
        ),
        (
            np.eye(2, dtype=np.uint8),
            ["'A'", "A"],
            [0, 1],
            ValueError,
            "duplicate normalized",
        ),
    ],
)
def test_decode_rooted_clade_catalog_rejects_incompatible_snapshots(
    rows,
    leaf_names,
    active_columns,
    error,
    message,
):
    with pytest.raises(error, match=message):
        decode_rooted_clade_catalog(
            bipartition_bits=rows,
            leaf_names=leaf_names,
            active_columns=active_columns,
        )
