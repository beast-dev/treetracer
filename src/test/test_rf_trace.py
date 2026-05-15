"""``ess.rf_trace.compute_rf_trace_data`` builds the RF-distance trace
shown in the Diagnostics tab.

Validation is self-consistency: every value in the returned DataFrame
must equal the corresponding entry in the underlying distance matrix
read directly. Also pins the ``first`` / ``last`` reference-position
contract, which is easy to silently flip.

Backing data is the real 100-tree ``test.trees`` fixture via
``rapidtrees_full`` (session-scoped from ``conftest.py``), saved as a
small .npy to a tmp path and registered as a distmat. This exercises
the same path the production worker writes through, on real BEAST-style
tree names — so ``compute_rf_trace_data``'s ``<group>/<name>`` prefix
parsing (used to filter ``ref_trees_in_group``) is covered for free.
"""

from __future__ import annotations

import numpy as np
import pytest

from treetracer import state
from treetracer.ess.rf_trace import compute_rf_trace_data


@pytest.fixture(autouse=True)
def _clean_state():
    state.clear_all_distmats()
    yield
    state.clear_all_distmats()


def _register_real_fixture(tmp_path, rapidtrees_full):
    """Persist the real RF matrix from the fixture to a .npy and
    register it under ``state``. Returns ``(group_name, names, M)``.

    Production tree names look like ``<group>/<tree>`` (see
    ``db/process_trees.py:213`` where ``group_name = base_filename``).
    The fixture's raw newick-line names lack that prefix, so we
    rename in place to ``test/<orig>`` and register the renamed list.
    ``compute_rf_trace_data`` parses the group out of the prefix to
    filter ``ref_trees_in_group``; without the prefix every tree
    would land in a singleton group of its own.
    """
    raw_names, rf_matrix, _presence, _leaf_names, _n_bip = rapidtrees_full
    group = "test"
    names = [f"{group}/{n}" for n in raw_names]
    M = np.asarray(rf_matrix, dtype=np.uint16)
    path = tmp_path / "fixture.npy"
    np.save(path, M)
    state.register_distmat(
        "RF_FIXTURE", names, str(path),
        groups_per_file={"test.trees": [group]},
    )
    return group, names, M


def test_first_reference_returns_row_zero_excluding_self(
    tmp_path, rapidtrees_full,
):
    group, names, M = _register_real_fixture(tmp_path, rapidtrees_full)
    trace_df, ref_name = compute_rf_trace_data("RF_FIXTURE", group, "first")
    assert ref_name == names[0]
    # Trace excludes the reference tree itself.
    assert len(trace_df) == len(names) - 1
    # Every row's rf_distance equals M[0, i] for tree i.
    name_to_idx = {n: i for i, n in enumerate(names)}
    for _, row in trace_df.iterrows():
        i = name_to_idx[row["name"]]
        assert row["rf_distance"] == int(M[0, i]), row.to_dict()


def test_last_reference_returns_last_row(tmp_path, rapidtrees_full):
    group, names, M = _register_real_fixture(tmp_path, rapidtrees_full)
    trace_df, ref_name = compute_rf_trace_data("RF_FIXTURE", group, "last")
    assert ref_name == names[-1]
    name_to_idx = {n: i for i, n in enumerate(names)}
    last = len(names) - 1
    for _, row in trace_df.iterrows():
        i = name_to_idx[row["name"]]
        assert row["rf_distance"] == int(M[last, i])


def test_unknown_group_returns_error_string(tmp_path, rapidtrees_full):
    _register_real_fixture(tmp_path, rapidtrees_full)
    msg, ref_name = compute_rf_trace_data(
        "RF_FIXTURE", "doesnotexist", "last",
    )
    assert ref_name is None
    assert "doesnotexist" in msg


def test_unknown_matrix_returns_error_string():
    msg, ref_name = compute_rf_trace_data("RF_DOES_NOT_EXIST", "test", "first")
    assert ref_name is None
    assert "Distance matrix" in msg
