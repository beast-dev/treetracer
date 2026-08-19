"""Effective Sample Size — Gelman variogram estimator on a split chain.

A single MCMC trace is split in half and the two halves are treated as
two pseudo-chains. That gives us a Gelman-Rubin-style marginal posterior
variance (within + between) which catches drift between the early and
late portions of the chain, then the autocorrelation is estimated via
the variogram form (BDA3 §11.5):

    V_t = ⟨(x_{j,i+t} − x_{j,i})²⟩      averaged over j and i
    ρ_t = 1 − V_t / (2·σ²)
    ESS = m·n / τ

The lag correlations are passed to the same paired initial-positive and
initial-monotone sequence reducer used by Fréchet-correlation ESS. It forms
``(ρ_0 + ρ_1), (ρ_2 + ρ_3), ...`` and computes

    τ = -1 + 2·Σ_k P'_k

where ``P'_k`` are the retained, monotonically smoothed pair sums. No FFT.

Reference implementation that this is modelled on:
    https://gist.github.com/… (see Gelman BDA3 §11.5; reproduced in
    Stan and PyMC). The split-half pseudo-chain trick mirrors Stan's
    split-Rhat / split-ESS.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm, rankdata


LagCorrelationFunction = Callable[[int], float]


def distance_correlation(
    variance_left: float,
    variance_right: float,
    mean_squared_distance: float,
    *,
    zero_variance_correlation: float = 1.0,
) -> float:
    """Convert two variances and a squared displacement to correlation.

    The metric-space covariance identity is

    ``covariance = (variance_left + variance_right - distance_sq) / 2``.

    Dividing by the geometric mean of the two variances gives the lag
    correlation.  When the variances are the same scalar value ``s2``, this
    reduces exactly to the usual variogram formula
    ``1 - distance_sq / (2 * s2)``.

    The R implementation assigns correlation one when either distance
    variance is zero.  ``zero_variance_correlation`` exposes that convention
    explicitly while retaining one as the default.
    """

    variance_left = float(variance_left)
    variance_right = float(variance_right)
    mean_squared_distance = float(mean_squared_distance)
    inputs = np.asarray(
        (variance_left, variance_right, mean_squared_distance),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(inputs)):
        raise ValueError("variance and distance inputs must be finite")
    if variance_left < 0.0 or variance_right < 0.0:
        raise ValueError("variances must be nonnegative")
    if mean_squared_distance < 0.0:
        raise ValueError("mean_squared_distance must be nonnegative")

    if variance_left == 0.0 or variance_right == 0.0:
        return float(zero_variance_correlation)

    covariance = 0.5 * (variance_left + variance_right - mean_squared_distance)
    return float(covariance / np.sqrt(variance_left * variance_right))


def ess_from_correlations(
    correlations: ArrayLike,
    sample_size: int,
    *,
    cap_at_n: bool = False,
) -> float:
    """Reduce lag correlations using a paired initial-monotone sequence.

    ``correlations`` must start with ``rho_0 = 1``.  Complete adjacent pairs
    are formed as ``(0, 1), (2, 3), ...``.  Evaluation stops at the first
    negative raw pair.  Retained pairs are made non-increasing with a
    cumulative minimum, after which

    ``tau = -1 + 2 * sum(monotone_pairs)``.

    The terminal nonpositive pair is excluded from ``tau``.  Matching
    ``treeess``, a negative ``tau`` falls back to one, while an exactly zero
    ``tau`` yields infinite ESS unless ``cap_at_n`` is enabled.
    """

    values = np.asarray(correlations, dtype=np.float64).ravel()
    if values.size < 2:
        raise ValueError("correlations must contain rho_0 and at least rho_1")
    if not np.all(np.isfinite(values)):
        raise ValueError("correlations must contain only finite values")
    if not np.isclose(values[0], 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("correlations must begin with rho_0 = 1")
    if isinstance(sample_size, bool) or not isinstance(sample_size, (int, np.integer)):
        raise TypeError("sample_size must be an integer")
    if sample_size < 1:
        raise ValueError("sample_size must be positive")

    complete_pair_count = values.size // 2
    paired = values[: 2 * complete_pair_count].reshape(-1, 2).sum(axis=1)

    negative = np.flatnonzero(paired < 0.0)
    if negative.size:
        paired = paired[: int(negative[0]) + 1]

    monotone = np.minimum.accumulate(paired)
    if monotone[-1] <= 0.0:
        monotone = monotone[:-1]

    tau = float(-1.0 + 2.0 * monotone.sum(dtype=np.float64))
    if tau < 0.0:
        tau = 1.0

    ess = float("inf") if tau == 0.0 else float(sample_size / tau)
    if cap_at_n:
        ess = min(float(sample_size), ess)

    return ess


def ess_from_lag_function(
    correlation_at_lag: LagCorrelationFunction,
    *,
    sample_size: int,
    max_lag_exclusive: int,
    cap_at_n: bool = False,
) -> float:
    """Evaluate lag correlations lazily and apply the shared ESS reducer."""

    if max_lag_exclusive <= 1:
        raise ValueError("at least one positive lag must be available")

    correlations = [1.0]
    for lag in range(1, max_lag_exclusive):
        correlation = float(correlation_at_lag(lag))
        if not np.isfinite(correlation):
            raise ValueError(f"correlation at lag {lag} is not finite")
        correlations.append(correlation)

        # Standard pairs complete at odd lags: (0, 1), (2, 3), ... .
        if lag % 2 == 1 and correlations[-2] + correlations[-1] < 0.0:
            break

    return ess_from_correlations(
        correlations,
        sample_size,
        cap_at_n=cap_at_n,
    )


def _split_chain_variance(chains: np.ndarray) -> float:
    """Marginal posterior variance for an (m, n) array of m chains × n
    samples. Combines between-chain variance ``B/n`` and within-chain
    variance ``W`` into the Gelman-Rubin over-estimate

        s² = ((n − 1)/n)·W + B/n
    """
    m, n = chains.shape
    chain_means = chains.mean(axis=1)
    grand_mean = chains.mean()
    B_over_n = ((chain_means - grand_mean) ** 2).sum() / (m - 1)
    W = ((chains - chain_means[:, None]) ** 2).sum() / (m * (n - 1))
    return W * (n - 1) / n + B_over_n


def effective_sample_size(x: np.ndarray) -> float:
    """Estimate the ESS of a 1-D MCMC trace.

    The trace is split in half (drop the last sample if N is odd so the
    halves are equal length), the two halves are treated as a pair of
    pseudo-chains for the Gelman-Rubin between/within variance, and the
    autocorrelation is integrated via the variogram form with the
    standard pair-sum truncation.

    Args:
        x: 1-D numpy array of MCMC samples in iteration order.
            Anything with ``.ravel()`` works. Length ≥ 4.

    Returns:
        ESS as a float. Special cases:
          * N < 4 → NaN (need at least two samples per half).
          * Non-finite trace → NaN.
          * Constant trace (zero variance) → retained N.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n_total = x.size
    if n_total < 4 or not np.all(np.isfinite(x)):
        return float("nan")

    # Even-length split. For odd N we drop the last sample so both halves
    # are length N // 2 — the standard split-Rhat convention.
    half = n_total // 2
    chains = x[: 2 * half].reshape(2, half)
    m, n = chains.shape

    post_var = _split_chain_variance(chains)
    if post_var == 0.0:
        return float(m * n)

    def correlation_at_lag(lag: int) -> float:
        diff = chains[:, lag:] - chains[:, : n - lag]
        variogram = np.mean(diff * diff, dtype=np.float64)
        return distance_correlation(post_var, post_var, float(variogram))

    return ess_from_lag_function(
        correlation_at_lag,
        sample_size=int(m * n),
        max_lag_exclusive=n,
    )


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """rank normalize parameter values before ESS computation like Stan.

    https://mc-stan.org/docs/reference-manual/analysis.html#effective-sample-size.section
    """
    r = rankdata(x, method="average")
    n = r.size
    return norm.ppf((r - 3.0 / 8.0) / (n + 1.0 / 4.0))
