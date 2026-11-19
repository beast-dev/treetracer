"""End-to-end tests for ``treetracer.clade_freq.compute_clade_frequencies``.

Backed by the 100-tree ``test.trees`` CI fixture: parsed via
``conftest.py``'s session fixtures, rapidtrees-encoded (rooted clades
since the production switch to ``rooted=True``), then fed two
synthetic consensus tree registry entries that split the fixture in half.

What's covered:

* Output DataFrame schema and value ranges.
* Per-row column-basis invariant — every row's ``column_j`` is a valid
  index into the snapshot's canonical-keys ``tuples`` list, and the
  ``split_key`` matches ``tuples[column_j]``.
* The defensive ``ValueError`` when called across distmats (the UI
  filters this out, but the module-level invariant is real).
* Frequencies sum to the expected counts when compared with an independent
  dense reference generated only inside the test.
"""

from __future__ import annotations

import numpy as np
import pytest

from treetracer import state
from treetracer.clade_freq import compute_clade_frequencies


@pytest.fixture(autouse=True)
def _clean_state():
    state.clear_all_distmats()
    yield
    state.clear_all_distmats()


def _register_fixture_and_build_entries(
    tmp_path,
    parsed_full,
    rapidtrees_full,
    *,
    cached_counts=True,
):
    """Register the fixture's RF matrix + snapshot under
    ``state.RF_CF`` and return two consensus tree registry entries that
    split the 100 trees into the first 50 and the last 50.

    The snapshot file is the only one rapidtrees writes on the
    production path; we synthesise it from ``rapidtrees_full``'s
    return values since the conftest fixture doesn't persist it.
    """
    names, rf_matrix, presence, _leaf_names, _n_bip = rapidtrees_full
    # Persist the RF matrix as .npy + the snapshot as .npz to a tmp
    # directory the ``state`` module's lazy decode will read from.
    distmat_path = tmp_path / "RF_CF.npy"
    np.save(distmat_path, np.asarray(rf_matrix, dtype=np.uint16))

    # Mirror the sparse-only snapshot written by ``rf._worker.compute_rf``.
    from treetracer.rf import rf_distance_with_sparse_snapshots_from_newick_iter
    from treetracer.rf.sparse_snapshots import sparse_snapshot_npz_payload
    tmap, names_in, newicks = parsed_full
    _, _, sparse = rf_distance_with_sparse_snapshots_from_newick_iter(
        names_in,
        iter(newicks),
        [tmap],
        [0] * len(names_in),
        rooted=True,
    )

    snap_path = tmp_path / "RF_CF_snapshots.npz"
    np.savez(snap_path, **sparse_snapshot_npz_payload(sparse))

    # ``state.get_canonical_keys`` will look the snapshot up at
    # ``state.get_snapshots_path(name)`` which derives from
    # ``state.get_distmat_path(name)``; route both to our tmp dir.
    monkey_dir = tmp_path
    state._tmpdir = str(monkey_dir)  # make get_snapshots_path use our dir

    state.register_distmat(
        "RF_CF", names, str(distmat_path),
        file_breakdown={"test.trees": len(names)},
    )

    n_total = len(names)
    half = n_total // 2

    counts_1 = presence[:half].sum(axis=0).astype(np.int32)
    counts_2 = presence[half:].sum(axis=0).astype(np.int32)
    cols_in_consensus_tree_1 = sorted(np.flatnonzero(counts_1).tolist())
    cols_in_consensus_tree_2 = sorted(np.flatnonzero(counts_2).tolist())

    entry1 = {
        "source_distmat": "RF_CF",
        "tree_names": list(names[:half]),
        "n_trees": half,
        "counts": counts_1 if cached_counts else None,
        "cols_in_consensus_tree": cols_in_consensus_tree_1,
    }
    entry2 = {
        "source_distmat": "RF_CF",
        "tree_names": list(names[half:]),
        "n_trees": n_total - half,
        "counts": counts_2 if cached_counts else None,
        "cols_in_consensus_tree": cols_in_consensus_tree_2,
    }
    return entry1, entry2, presence


def test_compute_clade_frequencies_schema_and_value_ranges(
    tmp_path, parsed_full, rapidtrees_full,
):
    entry1, entry2, presence = _register_fixture_and_build_entries(
        tmp_path, parsed_full, rapidtrees_full,
    )
    df = compute_clade_frequencies(entry1, entry2)

    # Schema.
    assert set(df.columns) == {
        "split_key", "column_j", "freq_1", "freq_2", "clade_size",
    }
    assert len(df) > 0
    assert df["freq_1"].dtype == np.dtype(np.float64)
    assert df["freq_2"].dtype == np.dtype(np.float64)

    # Value ranges.
    assert (df["freq_1"] >= 0).all() and (df["freq_1"] <= 1).all()
    assert (df["freq_2"] >= 0).all() and (df["freq_2"] <= 1).all()
    assert (df["clade_size"] >= 1).all()
    # At least one row should have nonzero frequency in BOTH groups
    # (the fixture's two halves share many clades).
    assert ((df["freq_1"] > 0) & (df["freq_2"] > 0)).any()


def test_column_j_indexes_canonical_tuples(
    tmp_path, parsed_full, rapidtrees_full,
):
    """``column_j`` must address the snapshot's canonical-keys table.
    ``tuples[column_j]`` must equal the row's ``split_key`` (and
    therefore decode to the same leaf-name set the tanglegram will
    highlight on click)."""
    entry1, entry2, _ = _register_fixture_and_build_entries(
        tmp_path, parsed_full, rapidtrees_full,
    )
    df = compute_clade_frequencies(entry1, entry2)
    canonical = state.get_canonical_keys("RF_CF")
    tuples = canonical["tuples"]
    n_cols = len(tuples)

    # Every column_j is an int in range; split_key matches tuples[j].
    assert df["column_j"].between(0, n_cols - 1).all()
    sample = df.sample(min(20, len(df)), random_state=0)
    for _, row in sample.iterrows():
        assert tuple(row["split_key"]) == tuples[int(row["column_j"])]


def test_frequencies_round_trip_against_presence(
    tmp_path, parsed_full, rapidtrees_full,
):
    """For each row, ``freq_1 * n_trees_1`` must equal the column-sum
    over the corresponding rows of the presence matrix. Catches any
    off-by-one or denominator drift in ``compute_clade_frequencies``."""
    entry1, entry2, presence = _register_fixture_and_build_entries(
        tmp_path, parsed_full, rapidtrees_full,
    )
    df = compute_clade_frequencies(entry1, entry2)
    n1 = entry1["n_trees"]
    n2 = entry2["n_trees"]
    # Slice rows and verify a sample.
    sample = df.sample(min(20, len(df)), random_state=0)
    for _, row in sample.iterrows():
        j = int(row["column_j"])
        expected_1 = int(presence[:n1, j].sum()) / n1
        expected_2 = int(presence[n1:, j].sum()) / n2
        assert row["freq_1"] == pytest.approx(expected_1)
        assert row["freq_2"] == pytest.approx(expected_2)


def test_sparse_only_snapshot_supports_canonical_keys_and_count_fallback(
    tmp_path,
    parsed_full,
    rapidtrees_full,
):
    entry1, entry2, presence = _register_fixture_and_build_entries(
        tmp_path,
        parsed_full,
        rapidtrees_full,
        cached_counts=False,
    )

    result = compute_clade_frequencies(entry1, entry2)
    canonical = state.get_canonical_keys("RF_CF")
    n1 = entry1["n_trees"]
    sample = result.sample(min(20, len(result)), random_state=0)

    assert len(canonical["tuples"]) == presence.shape[1]
    for _, row in sample.iterrows():
        column = int(row["column_j"])
        assert tuple(row["split_key"]) == canonical["tuples"][column]
        assert row["freq_1"] == pytest.approx(
            int(presence[:n1, column].sum()) / n1
        )
        assert row["freq_2"] == pytest.approx(
            int(presence[n1:, column].sum()) / (len(presence) - n1)
        )


def test_synthetic_mrhipstr_entry_uses_existing_clade_frequency_path(
    tmp_path,
    parsed_full,
    rapidtrees_full,
):
    entry1, entry2, _ = _register_fixture_and_build_entries(
        tmp_path,
        parsed_full,
        rapidtrees_full,
    )
    entry1.update(
        {
            "summary_method": "mrhipstr",
            "height_method": "mean",
            "consensus_tree": {
                "group": None,
                "treenum": None,
                "tree_name": "MrHIPSTR",
            },
        }
    )

    result = compute_clade_frequencies(entry1, entry2)

    assert len(result) > 0
    assert set(entry1["cols_in_consensus_tree"]) <= set(result["column_j"])

    entry2.update(
        {
            "summary_method": "mrhipstr",
            "height_method": "mean",
            "consensus_tree": {
                "group": None,
                "treenum": None,
                "tree_name": "MrHIPSTR",
            },
        }
    )
    both_synthetic = compute_clade_frequencies(entry1, entry2)
    assert both_synthetic.equals(result)


def test_raises_on_cross_distmat():
    """The UI filters consensus tree dropdowns to the active distmat so this
    can't happen via the app — but the function-level invariant
    is what stops a programmer-error from silently producing
    nonsense by merging columns from two different bases."""
    entry1 = {"source_distmat": "RF_A", "tree_names": [], "n_trees": 1,
              "counts": np.zeros(1, dtype=np.int32), "cols_in_consensus_tree": []}
    entry2 = {"source_distmat": "RF_B", "tree_names": [], "n_trees": 1,
              "counts": np.zeros(1, dtype=np.int32), "cols_in_consensus_tree": []}
    with pytest.raises(ValueError, match="different RF matrices"):
        compute_clade_frequencies(entry1, entry2)
