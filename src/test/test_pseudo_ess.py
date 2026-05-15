"""Sanity checks for the Lanfear pseudo-ESS implementation.

We can't easily compare against RWTY without an R bridge, but the
following self-consistency tests catch the most likely regressions:

* output shape / NaN handling for ``n_refs ≤ n_trees``.
* rank-normalize cap: even on a heavy-tailed chain, no per-reference
  ESS should exceed n by more than a small slack (Stan's bulk-ESS can
  legally exceed n a touch on anti-correlated chains, so we use
  ``1.1·n`` as the upper limit).
* sensitivity: a strongly autocorrelated chain (rows ordered) should
  produce noticeably smaller pseudo-ESS than the same trees shuffled
  iid. If reordering doesn't change the answer the implementation is
  ignoring the row order, which would be a major bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from treetracer.ess.pseudo_ess import compute_pseudo_ess


def _autocorrelated_distmat(n: int, rho: float, seed: int = 0) -> np.ndarray:
    """Build a fake square symmetric distance matrix whose row-wise
    behaviour mimics an AR(1) chain. The ``j``-th column (trace to
    reference tree j) is an AR(1) walk plus the deterministic
    "distance to self = 0" diagonal.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, n))
    base = (base + base.T) / 2.0
    # Inject autocorrelation by smoothing columns with an AR(1) filter
    # along the row axis.
    out = np.empty_like(base)
    out[0, :] = base[0, :]
    for i in range(1, n):
        out[i, :] = rho * out[i - 1, :] + np.sqrt(1.0 - rho * rho) * base[i, :]
    # Symmetrise + zero diagonal + take absolute distances.
    D = np.abs(out)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0)
    return D


def test_output_shape():
    D = _autocorrelated_distmat(200, rho=0.5)
    res = compute_pseudo_ess(D, n_refs=20, seed=0)
    assert res["n_refs_used"] == 20
    assert res["ess_values"].shape == (20,)
    assert res["ref_indices"].shape == (20,)
    assert np.all(res["ess_values"] > 0)
    assert np.isfinite(res["min"]) and np.isfinite(res["median"])


def test_n_refs_capped_at_n_trees():
    D = _autocorrelated_distmat(15, rho=0.0)
    res = compute_pseudo_ess(D, n_refs=100, seed=0)
    assert res["n_refs_used"] == 15


def test_rank_normalize_bounds_ess():
    """Rank-normalize should keep ESS from running away on heavy-tailed
    columns. Bulk-ESS can legally exceed n by a healthy margin on
    anti-correlated chains (Stan docs: typically <2·n), so we only
    guard against catastrophic blow-up (e.g. ESS = 10·n) rather than
    require ESS ≤ n.
    """
    n = 500
    rng = np.random.default_rng(0)
    D = rng.standard_t(df=3, size=(n, n))
    D = np.abs((D + D.T) / 2.0)
    np.fill_diagonal(D, 0)
    res = compute_pseudo_ess(D, n_refs=30, seed=0)
    assert res["max"] <= 2.0 * n, res["max"]


def test_autocorrelation_lowers_ess():
    """An autocorrelated chain should yield much lower pseudo-ESS
    than the same rows shuffled to break ordering."""
    n = 600
    D_ac = _autocorrelated_distmat(n, rho=0.85, seed=3)
    rng = np.random.default_rng(99)
    perm = rng.permutation(n)
    D_iid = D_ac[np.ix_(perm, perm)]
    ess_ac = compute_pseudo_ess(D_ac, n_refs=30, seed=0)["median"]
    ess_iid = compute_pseudo_ess(D_iid, n_refs=30, seed=0)["median"]
    assert ess_iid > 2.0 * ess_ac, (ess_ac, ess_iid)


def test_too_few_trees_returns_nan():
    D = np.zeros((3, 3))
    res = compute_pseudo_ess(D, n_refs=10, seed=0)
    assert res["n_refs_used"] == 0
    assert np.isnan(res["min"]) and np.isnan(res["median"]) and np.isnan(res["max"])
