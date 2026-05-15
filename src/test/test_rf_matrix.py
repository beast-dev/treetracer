"""Element-wise validation of rapidtrees' pairwise RF matrix against
a DendroPy reference, in **rooted-clade mode** (matches production —
``rf/_worker.py`` calls rapidtrees with ``rooted=True``).

DendroPy's ``treecompare.symmetric_difference`` is unrooted-by-default
and ignores rooting even when ``is_rooted=True`` on the tree object.
The rooted reference is instead the manual symmetric difference of
each tree's per-internal-node descendant tip-label set — same thing
rapidtrees computes when ``rooted=True``. The helper that builds
those clade sets lives in ``conftest.py`` (`rooted_clade_set`).

The RF matrix is the upstream input to PCoA, pseudo-ESS, and MCC, so
any rapidtrees / DendroPy disagreement here ripples through everything
downstream.

This test uses the first 50 trees of the 100-tree fixture; the
``parsed_50`` + ``rapidtrees_50`` + ``dendropy_trees_50`` session
fixtures in ``conftest.py`` keep parsing cost out of every test run.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from .conftest import rooted_clade_set


@pytest.mark.integration
def test_rf_matrix_symmetric_zero_diagonal(rapidtrees_full):
    _names, rf, _presence, _leaf_names, _ = rapidtrees_full
    M = np.asarray(rf, dtype=np.int64)
    assert (M.diagonal() == 0).all()
    assert (M == M.T).all()


@pytest.mark.integration
def test_rapidtrees_matches_dendropy_rooted_50(rapidtrees_50, dendropy_trees_50):
    """Compare rapidtrees rooted-RF against DendroPy's rooted-clade
    symmetric difference, element-wise on 50 trees / 1225 pairs."""
    _rt_names, rf_rt, _, _, _ = rapidtrees_50
    rf_rt = np.asarray(rf_rt, dtype=np.int64)
    n = rf_rt.shape[0]
    assert n == 50

    clade_sets = [rooted_clade_set(t) for t in dendropy_trees_50]
    rf_dp = np.zeros((n, n), dtype=np.int64)
    for i, j in itertools.combinations(range(n), 2):
        d = len(clade_sets[i] ^ clade_sets[j])
        rf_dp[i, j] = d
        rf_dp[j, i] = d

    # Locate the first few disagreements if any — much more useful than
    # a bare ``assert (rf_rt == rf_dp).all()``.
    diff = rf_rt - rf_dp
    n_bad = int((diff != 0).sum() // 2)
    if n_bad:
        iu = np.triu_indices(n, k=1)
        bad = np.where(rf_rt[iu] != rf_dp[iu])[0][:5]
        rows = []
        for k in bad:
            i, j = iu[0][k], iu[1][k]
            rows.append(f"  ({i},{j}): rapidtrees={rf_rt[i,j]}, dendropy={rf_dp[i,j]}")
        raise AssertionError(
            f"{n_bad}/{n*(n-1)//2} pairs disagree; "
            f"max abs diff={int(np.abs(diff).max())}\n" + "\n".join(rows)
        )
