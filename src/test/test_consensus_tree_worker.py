"""Worker-level coverage for MCC and synthetic MrHIPSTR summaries."""

from __future__ import annotations

import dendropy
import numpy as np
import pytest

from treetracer.consensus_tree._subprocess_worker import (
    compute_consensus_tree_worker_entry,
)
from treetracer.db.process_trees import process_nexus_trees_streaming
from treetracer.db.tree_manager import TreeManagerPandas
from treetracer.rf import rf_distance_with_snapshots_from_newick_iter


_RESULT_KEYS = {
    "nexus_bytes",
    "summary_method",
    "height_method",
    "consensus_tree_row",
    "summary_tree_name",
    "log_clade_credibility",
    "majority_clade_count",
    "counts",
    "cols_in_consensus_tree",
    "negative_branch_count",
    "minimum_branch_length",
    "mrhipstr_statistics",
    "mrhipstr_profile",
    "missing_taxa",
}


def _build_worker_inputs(tmp_path):
    source = "posterior.trees"
    translate = {
        "1": "Taxon A",
        "2": "Taxon B",
        "3": "Taxon C",
        "4": "Taxon D",
    }
    preamble = (
        b"#NEXUS\n\nBegin trees;\n"
        b"    Translate\n"
        b"        1 'Taxon A',\n"
        b"        2 'Taxon B',\n"
        b"        3 'Taxon C',\n"
        b"        4 'Taxon D'\n"
        b"    ;\n"
    )
    newicks = [
        "((1:1,2:1):1,(3:1,4:1):1):0;",
        "((1:1,2:1):1,(3:1,4:1):1):0;",
        "((1:1,3:1):1,(2:1,4:1):1):0;",
    ]
    names = [f"posterior/STATE_{index}" for index in range(len(newicks))]

    payload = bytearray(preamble)
    records = []
    for index, (name, newick) in enumerate(zip(names, newicks, strict=True)):
        local_name = name.split("/", 1)[1]
        prefix = f"tree {local_name} [&lnP={-10.0 - index}] = ".encode()
        body = f"[&R] {newick}".encode()
        line = prefix + body + b"\n"
        line_offset = len(payload)
        payload.extend(line)
        records.append(
            {
                "name": name,
                "file_source": source,
                "newick_offset": line_offset + len(prefix),
                "newick_length": len(body),
                "line_offset": line_offset,
                "line_length": len(line),
                "metadata": {"lnP": -10.0 - index},
            }
        )
    payload.extend(b"End;\n")

    source_path = tmp_path / source
    source_path.write_bytes(payload)

    _, _, presence, leaf_names, _, bipartition_bits = (
        rf_distance_with_snapshots_from_newick_iter(
            names,
            iter(newicks),
            [translate],
            [0] * len(newicks),
            rooted=True,
        )
    )
    snapshot_path = tmp_path / "RF_TEST_snapshots.npz"
    np.savez(
        snapshot_path,
        presence=presence,
        leaf_names=np.asarray(leaf_names),
        bipartition_bits=bipartition_bits,
    )

    return {
        "matched_records": records,
        "source_distmat": "RF_TEST",
        "snapshots_path": str(snapshot_path),
        "full_distmat_names": names,
        "translate_maps": {source: translate},
        "source_file_paths": {source: str(source_path)},
        "source_preambles": {source: preamble},
        "is_rooted": True,
    }, presence


def _parse_nexus_tree(nexus_bytes):
    return dendropy.Tree.get(
        data=nexus_bytes.decode("utf-8"),
        schema="nexus",
        preserve_underscores=True,
    )


def test_worker_explicit_method_preserves_mcc_path(tmp_path):
    kwargs, presence = _build_worker_inputs(tmp_path)

    result = compute_consensus_tree_worker_entry(
        **kwargs,
        summary_method="mcc",
    )

    assert set(result) == _RESULT_KEYS
    assert result["summary_method"] == "mcc"
    assert result["height_method"] == "sampled"
    assert result["consensus_tree_row"] in kwargs["matched_records"]
    assert result["summary_tree_name"] == result["consensus_tree_row"]["name"]
    assert result["majority_clade_count"] is None
    assert result["negative_branch_count"] == 0
    assert result["minimum_branch_length"] is None
    assert result["mrhipstr_statistics"] is None
    assert result["mrhipstr_profile"] is None
    np.testing.assert_array_equal(result["counts"], presence.sum(axis=0))
    tree = _parse_nexus_tree(result["nexus_bytes"])
    assert {taxon.label for taxon in tree.taxon_namespace} == set(
        kwargs["translate_maps"]["posterior.trees"].values()
    )


def test_worker_midpoint_roots_unrooted_mcc_output(tmp_path):
    kwargs, _ = _build_worker_inputs(tmp_path)
    kwargs["is_rooted"] = False

    result = compute_consensus_tree_worker_entry(
        **kwargs,
        summary_method="mcc",
    )

    assert result["summary_method"] == "mcc"
    assert b"= [&R]" in result["nexus_bytes"]
    assert _parse_nexus_tree(result["nexus_bytes"]).is_rooted


def test_worker_defaults_to_synthetic_mean_height_mrhipstr_tree(tmp_path):
    kwargs, presence = _build_worker_inputs(tmp_path)

    result = compute_consensus_tree_worker_entry(**kwargs)

    assert set(result) == _RESULT_KEYS
    assert result["summary_method"] == "mrhipstr"
    assert result["height_method"] == "mean"
    assert result["consensus_tree_row"] is None
    assert result["summary_tree_name"] == "MrHIPSTR"
    assert result["majority_clade_count"] == 2
    assert result["negative_branch_count"] == 0
    assert result["minimum_branch_length"] == pytest.approx(1.0)
    mrhipstr_statistics = result["mrhipstr_statistics"]
    assert mrhipstr_statistics["total_trees"] == 3
    assert mrhipstr_statistics["n_tips"] == 4
    assert mrhipstr_statistics["total_unique_clades"] == 5
    assert mrhipstr_statistics["clades_in_more_than_one_tree"] == 3
    assert mrhipstr_statistics["topology_seconds"] >= 0.0
    assert mrhipstr_statistics["lowest_clade_credibility"] == pytest.approx(
        2 / 3
    )
    assert mrhipstr_statistics["mean_clade_credibility"] == pytest.approx(
        7 / 9
    )
    assert mrhipstr_statistics["median_clade_credibility"] == pytest.approx(
        2 / 3
    )
    assert mrhipstr_statistics["clades_with_credibility_1"] == 1
    assert mrhipstr_statistics["clades_with_credibility_gt_0_99"] == 1
    assert mrhipstr_statistics["clades_with_credibility_gt_0_95"] == 1
    assert mrhipstr_statistics["clades_with_credibility_gt_0_5"] == 3
    profile = result["mrhipstr_profile"]
    assert set(profile) == {
        "worker_started_wall_time",
        "worker_setup_seconds",
        "taxon_alignment_seconds",
        "snapshot_selection_seconds",
        "clade_collection_seconds",
        "source_tree_ingestion_seconds",
        "topology_search_seconds",
        "nexus_serialization_seconds",
        "nexus_created_wall_time",
        "statistics_seconds",
        "worker_unattributed_seconds",
        "worker_total_seconds",
        "worker_finished_wall_time",
    }
    duration_keys = {
        key for key in profile if key.endswith("_seconds")
    }
    assert all(profile[key] >= 0.0 for key in duration_keys)
    assert profile["topology_search_seconds"] == pytest.approx(
        mrhipstr_statistics["topology_seconds"]
    )
    assert profile["worker_finished_wall_time"] >= profile[
        "worker_started_wall_time"
    ]
    assert profile["worker_started_wall_time"] <= profile[
        "nexus_created_wall_time"
    ] <= profile["worker_finished_wall_time"]
    measured_worker_total = sum(
        profile[key]
        for key in duration_keys
        if key != "worker_total_seconds"
    )
    assert measured_worker_total == pytest.approx(
        profile["worker_total_seconds"]
    )
    np.testing.assert_array_equal(result["counts"], presence.sum(axis=0))
    assert result["cols_in_consensus_tree"]
    assert b"summaryMethod=MrHIPSTR" in result["nexus_bytes"]
    tree = _parse_nexus_tree(result["nexus_bytes"])
    assert tree.is_rooted
    assert {taxon.label for taxon in tree.taxon_namespace} == set(
        kwargs["translate_maps"]["posterior.trees"].values()
    )


def test_worker_rejects_mrhipstr_for_unrooted_snapshot(tmp_path):
    kwargs, _ = _build_worker_inputs(tmp_path)
    kwargs["is_rooted"] = False

    with pytest.raises(ValueError, match="requires a rooted RF snapshot"):
        compute_consensus_tree_worker_entry(
            **kwargs,
            summary_method="mrhipstr",
        )


def test_worker_reports_missing_mrhipstr_snapshot_arrays(tmp_path):
    kwargs, presence = _build_worker_inputs(tmp_path)
    incomplete_path = tmp_path / "incomplete_snapshots.npz"
    np.savez(incomplete_path, presence=presence)
    kwargs["snapshots_path"] = str(incomplete_path)

    with pytest.raises(
        ValueError,
        match="missing required arrays: bipartition_bits, leaf_names",
    ):
        compute_consensus_tree_worker_entry(
            **kwargs,
            summary_method="mrhipstr",
        )


@pytest.mark.integration
def test_mrhipstr_worker_round_trips_repository_fixture(
    tmp_path,
    trees_path,
    parsed_full,
):
    translate, _, newicks = parsed_full
    manager = TreeManagerPandas()
    process_nexus_trees_streaming(
        str(trees_path),
        manager,
        file_source="test.trees",
    )
    manager.flush()
    matched_records = manager._trees[[
        "name",
        "file_source",
        "newick_offset",
        "newick_length",
        "line_offset",
        "line_length",
        "metadata",
    ]].to_dict("records")
    for record in matched_records:
        for field in (
            "newick_offset",
            "newick_length",
            "line_offset",
            "line_length",
        ):
            record[field] = int(record[field])

    full_names = [record["name"] for record in matched_records]
    _, _, presence, leaf_names, _, bipartition_bits = (
        rf_distance_with_snapshots_from_newick_iter(
            full_names,
            iter(newicks),
            [translate],
            [0] * len(newicks),
            rooted=True,
        )
    )
    snapshot_path = tmp_path / "RF_FIXTURE_snapshots.npz"
    np.savez(
        snapshot_path,
        presence=presence,
        leaf_names=np.asarray(leaf_names),
        bipartition_bits=bipartition_bits,
    )

    result = compute_consensus_tree_worker_entry(
        matched_records=matched_records,
        source_distmat="RF_FIXTURE",
        snapshots_path=str(snapshot_path),
        full_distmat_names=full_names,
        translate_maps={"test.trees": manager.get_translate_map("test.trees")},
        source_file_paths={"test.trees": str(trees_path)},
        source_preambles={
            "test.trees": manager._source_preambles["test.trees"]
        },
        is_rooted=True,
        summary_method="mrhipstr",
    )

    tree = _parse_nexus_tree(result["nexus_bytes"])
    assert result["consensus_tree_row"] is None
    assert result["summary_method"] == "mrhipstr"
    assert len(tree.taxon_namespace) == len(leaf_names)
    assert len(result["cols_in_consensus_tree"]) == 2 * len(leaf_names) - 2
