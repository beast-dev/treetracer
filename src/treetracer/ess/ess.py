"""Effective Sample Size — Gelman variogram estimator on a split chain.

A single MCMC trace is split in half and the two halves are treated as
two pseudo-chains. That gives us a Gelman-Rubin-style marginal posterior
variance (within + between) which catches drift between the early and
late portions of the chain, then the autocorrelation is estimated via
the variogram form (BDA3 §11.5):

    V_t = ⟨(x_{j,i+t} − x_{j,i})²⟩      averaged over j and i
    ρ_t = 1 − V_t / (2·σ²)
    ESS = m·n / (1 + 2·Σ_t ρ_t)

The sum over ρ_t is truncated at the first **even** lag whose
consecutive pair (ρ_{t-1} + ρ_t) goes negative — Geyer's initial-positive
sequence rule rewritten on consecutive lags. No FFT.

Reference implementation that this is modelled on:
    https://gist.github.com/… (see Gelman BDA3 §11.5; reproduced in
    Stan and PyMC). The split-half pseudo-chain trick mirrors Stan's
    split-Rhat / split-ESS.
"""

from __future__ import annotations

import numpy as np


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
          * Constant trace (zero variance) → N.
          * Truncation lands at lag 0 → N (chain looks uncorrelated).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n_total = x.size
    if n_total < 4:
        return float("nan")

    # Even-length split. For odd N we drop the last sample so both halves
    # are length N // 2 — the standard split-Rhat convention.
    half = n_total // 2
    chains = x[: 2 * half].reshape(2, half)
    m, n = chains.shape

    post_var = _split_chain_variance(chains)
    if post_var == 0.0:
        return float(m * n)

    # Walk lags 1, 2, 3, … computing ρ_t from the variogram. Stop at the
    # first even t where (ρ_{t-1} + ρ_t) < 0; this is Geyer's
    # initial-positive truncation on consecutive lags.
    rho = np.ones(n)
    t = 1
    negative_autocorr = False
    while not negative_autocorr and t < n:
        diff = chains[:, t:] - chains[:, : n - t]
        v_t = (diff * diff).sum() / (m * (n - t))
        rho[t] = 1.0 - v_t / (2.0 * post_var)
        if t % 2 == 0 and rho[t - 1] + rho[t] < 0.0:
            negative_autocorr = True
        t += 1

    tau = 1.0 + 2.0 * float(rho[1:t].sum())
    if tau <= 0.0:
        return float(m * n)
    return float(m * n) / tau

def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """rank normalize parameter values before ESS computation like Stan.
    
    https://mc-stan.org/docs/reference-manual/analysis.html#effective-sample-size.section
    """
    r = rankdata(x, method="average")
    n = r.size
    return norm.ppf((r - 3.0 / 8.0) / (n + 1.0 / 4.0))

