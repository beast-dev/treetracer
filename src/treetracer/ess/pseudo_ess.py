"""Pseudo-ESS for phylogenetic MCMC chains (Lanfear et al. 2016).

The trick: tree topology is too high-dimensional to run univariate ESS
on directly, but the **RF distance from every tree in the chain to a
reference tree** is a perfectly good 1-D scalar trace. Different
references give different traces (and hence different ESS values), so
we draw a sample of ~100 random reference trees, compute the trace per
reference, run :func:`treetracer.ess.ess.effective_sample_size` on
each, and return the resulting distribution.

The cached RF distance matrix that ``rf._worker.compute_rf`` writes to
disk (``uint16``, ``(n_trees, n_trees)``) makes step 1 cheap: column
``j`` is exactly the trace ``RF(tree_i, tree_j)`` over ``i`` — no
recomputation, no newick parsing.
"""

from __future__ import annotations

import numpy as np

from .ess import effective_sample_size, _rank_normalize


def compute_pseudo_ess(
    distmat: np.ndarray,
    n_refs: int = 100,
    seed: int | None = None,
) -> dict:
    """Pseudo-ESS over a random sample of reference trees.

    Args:
        distmat: ``(n_trees, n_trees)`` pairwise RF distance matrix
            (any integer / float dtype is accepted; converted to
            float64 internally for ESS).
        n_refs: number of reference trees to sample without
            replacement. Capped at ``n_trees``. Default 100.
        seed: optional RNG seed so the reference set is reproducible.

    Returns:
        ``dict`` with keys:

        =================  =============================================
        ``ess_values``     ``np.ndarray`` of length ``n_refs_used``;
                           one ESS estimate per reference tree, in the
                           order references were drawn (NaN entries are
                           possible if a trace is too short / constant).
        ``ref_indices``    ``np.ndarray[int]`` of the reference tree row
                           indices used.
        ``min`` / ``median`` / ``max``
                           scalar summaries of ``ess_values``,
                           excluding NaN entries.
        ``n_refs_used``    effective number of references actually
                           used (= ``min(n_refs, n_trees)``).
        =================  =============================================

        The conservative interpretation of "the chain's ESS" is the
        ``min`` (or some low quantile) of ``ess_values`` — that's the
        tightest bound across reference choices.

    Notes:
        For each reference ``j``, the trace ``distmat[:, j]`` includes
        the diagonal element ``RF(tree_j, tree_j) == 0``. Lanfear
        et al. leave this single-zero dip in the trace and so do we;
        the univariate ESS estimator is robust to one outlier.
    """
    n_trees = distmat.shape[0]
    if n_trees < 4:
        return {
            "ess_values": np.array([], dtype=float),
            "ref_indices": np.array([], dtype=int),
            "min": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
            "n_refs_used": 0,
        }

    rng = np.random.default_rng(seed)
    k = min(int(n_refs), n_trees)
    ref_indices = rng.choice(n_trees, size=k, replace=False)

    ess_values = np.empty(k, dtype=np.float64)
    for i, ref in enumerate(ref_indices):
        trace = distmat[:, ref].astype(np.float64, copy=False)
        ess_values[i] = effective_sample_size(_rank_normalize(trace))

    valid = ess_values[~np.isnan(ess_values)]
    return {
        "ess_values": ess_values,
        "ref_indices": ref_indices,
        "min": float(valid.min()) if valid.size else float("nan"),
        "median": float(np.median(valid)) if valid.size else float("nan"),
        "max": float(valid.max()) if valid.size else float("nan"),
        "n_refs_used": k,
    }
