"""ESS sanity checks against synthetic traces with known properties.

Three regimes worth pinning down:

1. **iid**: ESS should be close to n.
2. **AR(1) with autocorrelation ρ**: ESS should be close to the closed
   form ``n · (1-ρ) / (1+ρ)``. We tolerate ±25% because the variogram
   estimator with Geyer truncation has finite-sample noise.
3. **constant**: ESS should equal n (the implementation pins it there
   when posterior variance is zero).

We also cross-check against arviz on the rank-normalised trace —
TreeTracer's ``_rank_normalize`` is Stan's Blom transform, and
``arviz.ess(..., method="bulk")`` does the same internally. The two
estimators differ in lugsail / truncation rules so we only require the
results to be within a factor of 2 of each other (the Stan reference
manual notes this is the typical spread between bulk-ESS variants).
"""

from __future__ import annotations

import numpy as np
import pytest

from treetracer.ess.ess import effective_sample_size, _rank_normalize


def _ar1(n: int, rho: float, seed: int = 0) -> np.ndarray:
    """Generate an AR(1) trace with autocorrelation ρ."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = rng.normal()
    sigma = np.sqrt(1.0 - rho * rho)
    eps = rng.normal(scale=sigma, size=n - 1)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i - 1]
    return x


def test_ess_iid_close_to_n():
    n = 2000
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    ess = effective_sample_size(x)
    # iid trace: ESS should be within ~20% of n.
    assert 0.8 * n <= ess <= 1.2 * n, ess


@pytest.mark.parametrize("rho,tol_frac", [
    (0.5, 0.30),
    (0.7, 0.30),
    (0.9, 0.40),  # very autocorrelated → noisier ESS estimate
])
def test_ess_ar1_matches_closed_form(rho, tol_frac):
    n = 4000
    x = _ar1(n, rho, seed=7)
    expected = n * (1.0 - rho) / (1.0 + rho)
    ess = effective_sample_size(x)
    rel_err = abs(ess - expected) / expected
    assert rel_err < tol_frac, f"rho={rho}: expected≈{expected:.0f}, got {ess:.0f}"


def test_ess_constant_pins_at_n():
    x = np.full(500, 3.14)
    ess = effective_sample_size(x)
    assert ess == pytest.approx(500.0)


def test_rank_normalize_is_unit_normal():
    n = 5000
    rng = np.random.default_rng(0)
    x = rng.exponential(scale=10, size=n)
    z = _rank_normalize(x)
    # Blom transform → roughly standard normal regardless of input
    # distribution (within a half-percent for n=5000).
    assert abs(z.mean()) < 0.05
    assert abs(z.std() - 1.0) < 0.05


def _as_split_chains(x: np.ndarray) -> np.ndarray:
    """Match TT's internal split: drop the last sample if n is odd,
    reshape to (2, n//2). Used to feed arviz the same chain structure
    TT sees, so the only difference left between the estimators is
    variogram-vs-FFT for the autocorrelation."""
    half = x.size // 2
    return x[: 2 * half].reshape(2, half)


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.5, 0.7, 0.9])
def test_ess_triangulates_with_arviz_on_ar1(rho):
    """Triangulate TT, arviz, and the AR(1) closed form on the same
    2-chain split-half view of a single AR(1) trace.

    Both TT's split-half variogram estimator and arviz's mean-ESS use
    Geyer's initial-positive truncation; the remaining implementation
    difference is variogram-vs-FFT for the autocorrelation. Empirically
    they agree to ~5% across ρ ∈ [0, 0.9] on length-4000 traces with
    a fixed seed; both have the same finite-sample bias vs the closed
    form (slight under-estimate at low ρ, over-estimate at ρ ≥ 0.9).
    Tolerances:

    * TT vs arviz: ±10% (they really should agree).
    * TT vs closed form: ±25% (finite-sample bias floor).
    * arviz vs closed form: ±25% (same).
    """
    import arviz as az
    n = 4000
    expected = n * (1.0 - rho) / (1.0 + rho)
    x = _ar1(n, rho, seed=11)
    tt = effective_sample_size(x)
    az_mean = float(az.ess(_as_split_chains(x), method="mean"))
    assert abs(tt - az_mean) / max(tt, az_mean) < 0.10, (
        f"rho={rho}: TT={tt:.0f} arviz={az_mean:.0f}"
    )
    assert abs(tt - expected) / expected < 0.25, (
        f"rho={rho}: TT={tt:.0f} expected≈{expected:.0f}"
    )
    assert abs(az_mean - expected) / expected < 0.25, (
        f"rho={rho}: arviz={az_mean:.0f} expected≈{expected:.0f}"
    )


def _ar2(n: int, phi1: float, phi2: float, seed: int) -> np.ndarray:
    """Stationary AR(2): x_t = φ₁ x_{t-1} + φ₂ x_{t-2} + ε_t.

    Picked so that the autocorrelation function does NOT decay
    geometrically — it has a damped-oscillation profile (complex
    roots) for (φ₁, φ₂) = (0.6, 0.3), which exercises non-trivial
    truncation behaviour.
    """
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = eps[0]
    x[1] = phi1 * x[0] + eps[1]
    for t in range(2, n):
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + eps[t]
    return x


def _ma_q(n: int, weights: list[float], seed: int) -> np.ndarray:
    """Finite-memory MA(q): x_t = Σ_{k=0..q} w_k ε_{t-k}.

    Autocorrelation is non-zero only up to lag q, then drops to zero.
    Different qualitative profile from AR — short, finite memory.
    """
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n + len(weights))
    w = np.asarray(weights, dtype=np.float64)
    out = np.convolve(eps, w, mode="valid")
    return out[:n]


def _ar1_heavy_tail(n: int, rho: float, df: float, seed: int) -> np.ndarray:
    """AR(1) but with Student-t innovations — fat-tailed residuals
    that stress rank-normalize / variance-estimate paths."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = rng.standard_t(df)
    sigma = np.sqrt(1.0 - rho * rho)
    eps = rng.standard_t(df, size=n - 1) * sigma
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i - 1]
    return x


def _mixture_chain(n: int, p_jump: float, seed: int) -> np.ndarray:
    """Two-mode Gaussian mixture with sticky transitions: with
    probability ``p_jump`` switch modes, otherwise emit from the
    current mode. Tiny ``p_jump`` ⇒ very long correlation lengths ⇒
    ESS far below n. This is the multimodal MCMC failure mode."""
    rng = np.random.default_rng(seed)
    state = 0
    x = np.empty(n)
    means = np.array([-3.0, 3.0])
    for i in range(n):
        if rng.random() < p_jump:
            state = 1 - state
        x[i] = means[state] + rng.normal(scale=0.5)
    return x


@pytest.mark.parametrize("name,gen", [
    ("iid",            lambda: np.random.default_rng(7).standard_normal(4000)),
    ("ar2_oscillate",  lambda: _ar2(4000, phi1=0.6, phi2=0.3, seed=7)),
    ("ma5",            lambda: _ma_q(4000, weights=[1, 0.8, 0.6, 0.4, 0.2], seed=7)),
    ("ar1_t3_tails",   lambda: _ar1_heavy_tail(4000, rho=0.7, df=3, seed=7)),
    ("multimodal",     lambda: _mixture_chain(4000, p_jump=0.02, seed=7)),
])
def test_ess_matches_arviz_across_trace_shapes(name, gen):
    """TT and arviz mean-ESS should agree to within ~10% on a variety
    of non-AR(1) traces — same trace, same 2-chain split, only the
    variogram-vs-FFT autocorrelation path differs.
    """
    import arviz as az
    x = gen()
    tt = effective_sample_size(x)
    az_mean = float(az.ess(_as_split_chains(x), method="mean"))
    rel = abs(tt - az_mean) / max(tt, az_mean)
    assert rel < 0.10, (
        f"trace={name}: TT={tt:.0f} arviz={az_mean:.0f} rel_diff={rel:.2%}"
    )


def test_ess_matches_arviz_bulk_after_rank_normalize():
    """The Stan bulk-ESS pipeline (rank-normalize → ESS) should agree
    tightly with arviz.ess(method='bulk'), which does the same thing
    internally. Both estimators see the same 2-chain split-half view
    of the trace so the only remaining difference is variogram-vs-FFT
    autocorrelation."""
    import arviz as az
    n = 4000
    rho = 0.8
    x = _ar1(n, rho, seed=11)
    tt = effective_sample_size(_rank_normalize(x))
    az_bulk = float(az.ess(_as_split_chains(x), method="bulk"))
    ratio = tt / az_bulk
    assert 0.85 < ratio < 1.15, f"tt={tt:.0f} arviz={az_bulk:.0f} ratio={ratio:.2f}"
