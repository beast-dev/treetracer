"""Parity tests for the additive RapidTrees rooted-facts path."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import rapidtrees

from treetracer.consensus_tree.mrhipstr import (
    SourceTreeRecord,
    compute_mrhipstr_topology,
    count_selected_clades,
    decode_rooted_clade_catalog,
    ingest_source_trees,
    serialize_mrhipstr_tree,
)
from treetracer.consensus_tree.rooted_facts import (
    count_selected_clades_from_rooted_facts,
    prepare_mrhipstr_inputs_from_rooted_facts,
    source_summary_from_rooted_facts,
)
from treetracer.rf import (
    rf_distance_with_rooted_facts_from_newick_iter,
    rf_distance_with_snapshots_from_newick_iter,
)


pytestmark = pytest.mark.skipif(
    not hasattr(rapidtrees, "pairwise_rf_with_rooted_facts_from_newick_iter"),
    reason="installed RapidTrees does not provide rooted facts",
)

SOURCE_TREES = (
    "((A:1,B:2):3,(C:4,D:5):6);",
    "((A:2,B:1):5,(C:3,D:4):2);",
    "((A:3,C:2):4,(B:6,D:1):5);",
    "((A:4,D:2):1,(B:3,C:5):6);",
    "((A:1.5,B:2.5):3.5,(C:4.5,D:5.5):6.5);",
)
SOURCE_NAMES = tuple(f"tree-{index}" for index in range(len(SOURCE_TREES)))


def _legacy_inputs(selected_rows):
    _, _, presence, leaf_names, _, bipartition_bits = (
        rf_distance_with_snapshots_from_newick_iter(
            list(SOURCE_NAMES),
            iter(SOURCE_TREES),
            [{}],
            [0] * len(SOURCE_TREES),
            rooted=True,
        )
    )
    counts = count_selected_clades(presence, selected_rows)
    catalog = decode_rooted_clade_catalog(
        bipartition_bits=bipartition_bits,
        leaf_names=leaf_names,
        active_columns=counts.active_columns,
    )
    summary = ingest_source_trees(
        [
            SourceTreeRecord(
                name=SOURCE_NAMES[row],
                newick=SOURCE_TREES[row],
            )
            for row in selected_rows
        ],
        catalog=catalog,
        snapshot_counts=counts,
    )
    return catalog, counts, summary


def _rooted_facts():
    return rf_distance_with_rooted_facts_from_newick_iter(
        list(SOURCE_NAMES),
        iter(SOURCE_TREES),
        [{}],
        [0] * len(SOURCE_TREES),
    )


def test_rooted_facts_wrapper_matches_established_rooted_rf_snapshot():
    old_names, old_rf, old_presence, old_leaves, old_n_clades, old_bits = (
        rf_distance_with_snapshots_from_newick_iter(
            list(SOURCE_NAMES),
            iter(SOURCE_TREES),
            [{}],
            [0] * len(SOURCE_TREES),
            rooted=True,
        )
    )
    names, rf_matrix, facts = _rooted_facts()

    assert names == old_names
    assert facts.tree_names == tuple(old_names)
    assert facts.leaf_names == tuple(old_leaves)
    assert facts.n_clades == old_n_clades
    np.testing.assert_array_equal(rf_matrix, old_rf)

    reconstructed_presence = np.zeros_like(old_presence)
    np.put_along_axis(
        reconstructed_presence,
        facts.clade_columns.astype(np.intp, copy=False),
        1,
        axis=1,
    )
    np.testing.assert_array_equal(reconstructed_presence, old_presence)

    unpacked_clades = np.unpackbits(
        facts.packed_clades,
        axis=1,
        bitorder="little",
    )[:, : facts.n_taxa]
    np.testing.assert_array_equal(unpacked_clades, old_bits)
    assert facts.clade_columns.dtype == np.dtype(np.uint32)
    assert facts.node_heights.dtype == np.dtype(np.float64)
    assert facts.root_heights.dtype == np.dtype(np.float64)
    assert facts.split_ids.dtype == np.dtype(np.uint32)
    assert facts.split_table.dtype == np.dtype(np.uint32)
    assert not facts.clade_columns.flags.writeable
    assert not facts.packed_clades.flags.writeable


def test_rooted_facts_reproduce_existing_mrhipstr_inputs_and_nexus_exactly():
    # Deliberately use a non-monotonic subset so height accumulation follows
    # the same order as the existing selected source-record stream.
    selected_rows = [4, 1, 2, 0]
    legacy_catalog, legacy_counts, legacy_summary = _legacy_inputs(selected_rows)
    _, _, facts = _rooted_facts()

    prepared = prepare_mrhipstr_inputs_from_rooted_facts(
        facts,
        selected_rows,
        chunk_rows=2,
    )

    np.testing.assert_array_equal(
        prepared.snapshot_counts.counts,
        legacy_counts.counts,
    )
    np.testing.assert_array_equal(
        prepared.snapshot_counts.active_columns,
        legacy_counts.active_columns,
    )
    assert prepared.catalog == legacy_catalog
    assert prepared.source_summary.observed_splits == legacy_summary.observed_splits
    assert (
        prepared.source_summary.observation_counts
        == legacy_summary.observation_counts
    )
    assert prepared.source_summary.height_sums == legacy_summary.height_sums

    legacy_topology = compute_mrhipstr_topology(
        catalog=legacy_catalog,
        snapshot_counts=legacy_counts,
        source_summary=legacy_summary,
    )
    facts_topology = compute_mrhipstr_topology(
        catalog=prepared.catalog,
        snapshot_counts=prepared.snapshot_counts,
        source_summary=prepared.source_summary,
    )
    assert facts_topology == legacy_topology

    legacy_tree = serialize_mrhipstr_tree(
        topology=legacy_topology,
        catalog=legacy_catalog,
        snapshot_counts=legacy_counts,
        source_summary=legacy_summary,
    )
    facts_tree = serialize_mrhipstr_tree(
        topology=facts_topology,
        catalog=prepared.catalog,
        snapshot_counts=prepared.snapshot_counts,
        source_summary=prepared.source_summary,
    )
    assert facts_tree.nexus_bytes == legacy_tree.nexus_bytes


@pytest.mark.parametrize(
    ("selected_rows", "error", "message"),
    [
        ([], ValueError, "empty"),
        ([0.0], TypeError, "integer"),
        ([0, 0], ValueError, "duplicate"),
        ([-1], IndexError, "non-negative"),
        ([len(SOURCE_TREES)], IndexError, "outside"),
    ],
)
def test_rooted_facts_selection_rejects_invalid_rows(
    selected_rows,
    error,
    message,
):
    _, _, facts = _rooted_facts()
    with pytest.raises(error, match=message):
        count_selected_clades_from_rooted_facts(facts, selected_rows)


def test_rooted_facts_summary_cross_checks_selected_counts():
    _, _, facts = _rooted_facts()
    prepared = prepare_mrhipstr_inputs_from_rooted_facts(facts, [0, 1, 2])
    wrong_counts = replace(
        prepared.snapshot_counts,
        n_trees=prepared.snapshot_counts.n_trees + 1,
    )

    with pytest.raises(ValueError, match="different tree counts"):
        source_summary_from_rooted_facts(
            facts,
            [0, 1, 2],
            catalog=prepared.catalog,
            snapshot_counts=wrong_counts,
        )
