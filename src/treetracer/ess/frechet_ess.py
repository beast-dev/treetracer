"""Fréchet-correlation effective sample size from pairwise distances.

This module implements the supported ``lower.bound = TRUE`` path from the R
``treeess`` package.  Its input is an ordered, unsquared pairwise distance
matrix for an MCMC chain.  Rooted and unrooted Robinson--Foulds distances are
both valid inputs; the estimator depends only on the resulting metric-space
distances and their sampling order, not on split vectors or tree encodings.

The shared distance-to-correlation conversion and paired initial-positive/
initial-monotone ESS reduction live in :mod:`treetracer.ess.ess`. This
module only validates the distance matrix and constructs the Fréchet-specific
front/back distance variances and lag displacements before delegating to
those shared functions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .ess import distance_correlation, ess_from_lag_function


FloatArray = NDArray[np.float64]


def _validated_distance_matrix(
    distances: ArrayLike,
    *,
    validate: bool,
) -> FloatArray:
    raw = np.asarray(distances)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distances must be a square two-dimensional matrix")
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise ValueError("distances must be real-valued")

    matrix = np.asarray(raw, dtype=np.float64)
    if validate:
        if not np.all(np.isfinite(matrix)):
            raise ValueError("distances must contain only finite values")
        if np.any(matrix < 0.0):
            raise ValueError("distances must be nonnegative")
        if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("the distance-matrix diagonal must be zero")
        if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
            raise ValueError("the distance matrix must be symmetric")
    return matrix


def _validate_min_samples(min_samples: int, n: int) -> None:
    if isinstance(min_samples, bool) or not isinstance(min_samples, (int, np.integer)):
        raise TypeError("min_samples must be an integer")
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")
    if n < min_samples + 2:
        raise ValueError(
            "chain is too short: at least min_samples + 2 observations are "
            f"required (got n={n}, min_samples={min_samples})"
        )


def frechet_correlation_ess(
    distances: ArrayLike,
    *,
    min_samples: int = 5,
    cap_at_n: bool = False,
    validate: bool = True,
) -> float:
    """Estimate Fréchet-correlation ESS from ordered pairwise distances.

    Args:
        distances: Square, symmetric matrix of unsquared pairwise distances in
            MCMC sampling order.  Rooted or unrooted RF distances may be used.
        min_samples: R ``treeess``'s ``min.nsamples`` value.  Correlations are
            evaluated only while at least this many trailing samples remain.
        cap_at_n: Enforce the optional paper-style ``ESS <= n`` cap.  It is
            disabled by default because the supported R implementation has no
            cap.
        validate: Check finiteness, nonnegativity, symmetry, and zero diagonal.
            Disable only if the caller has already established these invariants.

    Returns:
        Fréchet-correlation effective sample size.

    Notes:
        The function squares distances internally.  Supplying an already
        squared matrix therefore produces the wrong geometry.

        The optimized endpoint sums evaluate all requested lags in
        ``O(n**2)`` time after distance construction, rather than repeatedly
        slicing progressively smaller square matrices as the R source does.
        The caller's matrix is never mutated.
    """

    matrix = _validated_distance_matrix(distances, validate=validate)
    n = int(matrix.shape[0])

    # Preserve the R convention before applying the usual length check.
    if np.all(matrix == 0.0):
        return 1.0

    _validate_min_samples(min_samples, n)

    # Always copy before squaring so writable float64 inputs are not changed.
    squared = np.array(matrix, dtype=np.float64, order="C", copy=True)
    np.square(squared, out=squared)

    # front_pairs[j] is the unordered-pair sum in squared[:j+1, :j+1].
    right_endpoint_sums = np.fromiter(
        (squared[:j, j].sum(dtype=np.float64) for j in range(n)),
        dtype=np.float64,
        count=n,
    )
    front_pairs = np.cumsum(right_endpoint_sums, dtype=np.float64)

    # back_pairs[i] is the unordered-pair sum in squared[i:, i:].
    left_endpoint_sums = np.fromiter(
        (squared[i, i + 1 :].sum(dtype=np.float64) for i in range(n)),
        dtype=np.float64,
        count=n,
    )
    back_pairs = np.cumsum(left_endpoint_sums[::-1], dtype=np.float64)[::-1]

    def correlation_at_lag(lag: int) -> float:
        retained = n - lag
        denominator = float(retained * (retained - 1))
        variance_front = front_pairs[retained - 1] / denominator
        variance_back = back_pairs[lag] / denominator
        lagged_squared_distance = np.diagonal(squared, offset=lag).mean(
            dtype=np.float64
        )
        return distance_correlation(
            variance_front,
            variance_back,
            float(lagged_squared_distance),
        )

    return ess_from_lag_function(
        correlation_at_lag,
        sample_size=n,
        max_lag_exclusive=n - min_samples,
        cap_at_n=cap_at_n,
    )


__all__ = ["frechet_correlation_ess"]
