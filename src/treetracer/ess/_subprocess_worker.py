"""Tree-ESS compute worker — runs in the persistent worker subprocess.

Diagnostics-tab "Compute Tree-ESS" can tick multiple runs at once; we send a
single job to the worker that does all ticked runs (+ optional Combined row)
and returns both the Pseudo-ESS summary and Fréchet-correlation ESS for each
row. This keeps the IPC round-trip cost paid once per click, not N times.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def compute_pseudo_ess_worker_entry(
    *,
    distmat_path: str,
    names: List[str],
    requests: List[Dict[str, Any]],
    n_refs: int,
    seed: int,
) -> Dict[str, Any]:
    """Compute Pseudo-ESS and Fréchet ESS for slices of an RF distmat.

    Args:
        distmat_path: path to the saved RF matrix .npy on disk
            (parent gets it from ``state.get_distmat_file_path``).
        names: ordered list of tree names — row/col labels of distmat.
        requests: one dict per row the parent wants:
            ``{"label": str, "indices": list[int], "burnin_label": str}``.
            ``indices`` are post-burnin row indices into the full
            distmat; the parent already applied burn-in per chain so
            the worker just slices.
        n_refs: passed through to ``compute_pseudo_ess``.
        seed: passed through.

    Returns a dict:
        ``{"results": [per-request dict, …]}``

    Each per-request dict has::

        {
            "label": str,
            "n_trees": int,
            "burnin_label": str,
            "min": float | None,    # None if no valid ess_values
            "q1": float | None,
            "q2": float | None,
            "q3": float | None,
            "max": float | None,
            "n_refs_used": int,
            "frechet": float | None,  # None when fewer than 7 trees
        }

    The Dash table is built parent-side from this — keeps the worker
    free of dmc / dash imports.
    """
    import numpy as np

    from .frechet_ess import frechet_correlation_ess
    from .pseudo_ess import compute_pseudo_ess

    distmat = np.load(distmat_path)

    out: List[Dict[str, Any]] = []
    for req in requests:
        idx = req["indices"]
        if len(idx) < 4:
            # Parent's caller already filters these out, but be defensive.
            out.append({
                "label": req["label"],
                "n_trees": len(idx),
                "burnin_label": req["burnin_label"],
                "min": None, "q1": None, "q2": None, "q3": None, "max": None,
                "n_refs_used": 0,
                "frechet": None,
            })
            continue

        sub = distmat[np.ix_(idx, idx)]
        res = compute_pseudo_ess(sub, n_refs=n_refs, seed=seed)
        frechet = (
            float(frechet_correlation_ess(sub))
            if len(idx) >= 7
            else None
        )
        valid = res["ess_values"][~np.isnan(res["ess_values"])]
        if valid.size:
            q1, q2, q3 = np.quantile(valid, [0.25, 0.5, 0.75])
            row = {
                "min": float(valid.min()),
                "q1": float(q1),
                "q2": float(q2),
                "q3": float(q3),
                "max": float(valid.max()),
            }
        else:
            row = {"min": None, "q1": None, "q2": None, "q3": None, "max": None}

        out.append({
            "label": req["label"],
            "n_trees": len(idx),
            "burnin_label": req["burnin_label"],
            "n_refs_used": int(res["n_refs_used"]),
            "frechet": frechet,
            **row,
        })

    return {"results": out}
