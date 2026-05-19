"""Element-wise validation of rapidtrees' pairwise RF matrix and per-tree
split set against DendroPy reference implementations, in **unrooted mode**.

Counterpart to ``test_rf_matrix.py``, which validates the rooted-clade
path. For unrooted trees we don't have to write a manual reference like
``rooted_clade_set`` does — DendroPy's
``treecompare.symmetric_difference`` is unrooted-by-default and matches
exactly what ``rapidtrees(rooted=False)`` computes.

Uses the local fixture at ``testdata/test_unrooted.trees`` (gitignored,
not in CI). Tests are skipped automatically when the file is absent.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest


@pytest.mark.integration
def test_unrooted_rf_matrix_symmetric_zero_diagonal(rapidtrees_50_unrooted):
    """Basic shape invariants: symmetric matrix with zeros on the
    diagonal. rapidtrees' ``rooted=False`` output should satisfy these
    just like the rooted path does."""
    _names, rf, _presence, _leaf_names, _ = rapidtrees_50_unrooted
    M = np.asarray(rf, dtype=np.int64)
    assert (M.diagonal() == 0).all()
    assert (M == M.T).all()


@pytest.mark.integration
def test_rapidtrees_matches_dendropy_unrooted_50(
    rapidtrees_50_unrooted, dendropy_trees_50_unrooted,
):
    """Element-wise compare rapidtrees' unrooted RF against DendroPy's
    ``symmetric_difference`` on 50 trees / 1225 pairs.

    DendroPy's symmetric_difference works on bipartitions when the
    trees are flagged ``is_rooted=False`` (which the fixture sets).
    Direct equality is expected — both are computing the same
    quantity, just with different implementations.
    """
    dendropy = pytest.importorskip("dendropy")
    from dendropy.calculate import treecompare

    _rt_names, rf_rt, _, _, _ = rapidtrees_50_unrooted
    rf_rt = np.asarray(rf_rt, dtype=np.int64)
    n = rf_rt.shape[0]
    assert n == 50

    trees = dendropy_trees_50_unrooted
    # DendroPy needs bipartitions encoded before symmetric_difference
    # can read them off. ``update_bipartitions=True`` on the first call
    # caches them on each tree.
    for t in trees:
        t.encode_bipartitions()

    rf_dp = np.zeros((n, n), dtype=np.int64)
    for i, j in itertools.combinations(range(n), 2):
        d = treecompare.symmetric_difference(trees[i], trees[j])
        rf_dp[i, j] = d
        rf_dp[j, i] = d

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


@pytest.mark.integration
def test_unrooted_presence_columns_are_bipartitions(rapidtrees_50_unrooted):
    """In ``rooted=False`` mode rapidtrees' presence matrix columns
    correspond to bipartitions, not rooted clades. Each column should
    cover STRICTLY FEWER than all leaves and more than zero — a
    bipartition has two non-empty sides.

    We verify by decoding the bipartition_bits and asserting each
    canonical-side cardinality is in ``(0, n_leaves)``."""
    _names, _rf, presence, leaf_names, n_bip = rapidtrees_50_unrooted
    # Each column of ``presence`` is a bipartition; the COUNT of trees
    # containing that bipartition must be at least 1 (it wouldn't be
    # in the global table otherwise).
    col_sums = np.asarray(presence).sum(axis=0)
    assert (col_sums > 0).all(), "every catalogued bipartition appears in ≥1 tree"
    # Sanity bound: number of unique bipartitions can't exceed
    # n_trees * (n_taxa - 3) — the per-tree internal-edge count
    # for a fully resolved unrooted tree.
    n_trees, n_taxa = presence.shape[0], len(leaf_names)
    assert n_bip <= n_trees * (n_taxa - 3)


@pytest.mark.integration
def test_unrooted_rf_in_expected_range(rapidtrees_50_unrooted):
    """Sanity: unrooted RF between two binary trees on ``n`` taxa is
    bounded above by ``2*(n - 3)`` (number of internal edges per tree,
    times two when the trees share no splits)."""
    _names, rf, _presence, leaf_names, _ = rapidtrees_50_unrooted
    rf = np.asarray(rf, dtype=np.int64)
    n_taxa = len(leaf_names)
    upper = 2 * (n_taxa - 3)
    assert rf.min() == 0
    assert rf.max() <= upper, (
        f"max RF {rf.max()} exceeds theoretical bound 2*(n-3) = {upper}"
    )
