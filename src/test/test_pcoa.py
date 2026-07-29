"""TreeTracer PCoA must agree with scipy's classical-MDS up to
Procrustes alignment (PCoA's only freedom is per-axis sign and
orthogonal rotation within degenerate-eigenvalue subspaces).

Two angles:

* **Synthetic Euclidean** — points truly live in ℝᵏ, so PCoA must
  recover them exactly. Disparity at machine precision proves the
  eigendecomp path is correct.
* **Real RF matrix** — generated on the fly from the 100-tree CI
  fixture via rapidtrees. Catches anything specific to non-Euclidean
  distances (negative eigenvalues, the clamp-to-zero step) that the
  synthetic path can't reach.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform

from treetracer.rf.mds import compute_mds


def _scipy_pcoa(D: np.ndarray, k: int) -> np.ndarray:
    """Classical metric MDS via double-centering + scipy eigendecomp."""
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * (J @ (D ** 2) @ J)
    w, V = eigh(B)
    order = np.argsort(w)[::-1]
    top = order[:k]
    return V[:, top] * np.sqrt(np.maximum(w[top], 0))


# Both algorithms exposed by ``treetracer.rf.mds.compute_mds`` get
# the same correctness treatment. ``pcoa`` is the textbook reference
# (full ``eigh`` + materialised centering matrix); ``pcoa_fast`` is
# the production default (vectorised centering + ARPACK eigsh, ~100×
# faster at n=5000). Both must match scipy's classical PCoA up to
# Procrustes alignment.
ALGORITHMS = ["pcoa", "pcoa_fast"]


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_pcoa_synthetic_euclidean_round_trip(algorithm):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4))
    D = squareform(pdist(X))
    Y_tt = compute_mds(D, n_components=4, algorithm=algorithm)
    Y_sp = _scipy_pcoa(D, 4)
    _, _, disp = procrustes(Y_sp, Y_tt)
    # PCoA on truly-Euclidean input collapses to PCA — should recover
    # original coords up to numerical noise.
    assert disp < 1e-12, (algorithm, disp)


def test_pcoa_fast_reports_phase_progress():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 4))
    D = squareform(pdist(X))
    events = []

    compute_mds(
        D,
        n_components=3,
        algorithm="pcoa_fast",
        progress=lambda phase, fraction, label: events.append(
            (phase, fraction, label)
        ),
    )

    phases = [phase for phase, _fraction, _label in events]
    assert "centering" in phases
    assert "eigensolve" in phases
    assert "finalizing" in phases
    assert all(0 <= fraction <= 1 for _phase, fraction, _label in events)


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_pcoa_real_rf_matches_scipy(rapidtrees_full, algorithm):
    """100-tree real RF matrix from rapidtrees → PCoA matches scipy.

    Non-Euclidean: RF tree-space has negative eigenvalues that TT
    clamps to zero. Procrustes disparity collapses sign / rotation /
    scale, so a tiny disparity proves the two eigendecomp paths
    agree on the dominant subspace. Run for both ``pcoa`` (textbook)
    and ``pcoa_fast`` (production default).
    """
    _names, rf_matrix, _presence, _leaf_names, _n_bip = rapidtrees_full
    D = np.asarray(rf_matrix, dtype=float)
    k = 6
    Y_tt = compute_mds(D, n_components=k, algorithm=algorithm)
    Y_sp = _scipy_pcoa(D, k)
    _, _, disp = procrustes(Y_sp, Y_tt)
    assert disp < 1e-10, (algorithm, disp)

    # Per-axis variance ratio (basis-sign-independent). Both
    # implementations diagonalize the same matrix, so the variance
    # shares must agree to ~6 sig figs.
    var_tt = (Y_tt ** 2).sum(axis=0) / (Y_tt ** 2).sum()
    var_sp = (Y_sp ** 2).sum(axis=0) / (Y_sp ** 2).sum()
    np.testing.assert_allclose(var_tt, var_sp, atol=1e-6)


def test_pcoa_fast_matches_pcoa_reference(rapidtrees_full):
    """The two MDS algorithms registered in ``compute_mds`` should be
    bit-equivalent (up to Procrustes alignment) on the same input.
    Catches drift if anyone changes ``pcoa_fast`` without checking
    against the textbook reference."""
    _names, rf_matrix, _presence, _leaf_names, _n_bip = rapidtrees_full
    D = np.asarray(rf_matrix, dtype=float)
    k = 6
    Y_ref = compute_mds(D, n_components=k, algorithm="pcoa")
    Y_fast = compute_mds(D, n_components=k, algorithm="pcoa_fast")
    _, _, disp = procrustes(Y_ref, Y_fast)
    assert disp < 1e-20, disp
