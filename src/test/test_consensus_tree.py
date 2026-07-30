"""``compute_consensus_tree_index`` and the full consensus tree pipeline must agree with
DendroPy's ``TreeArray.maximum_product_of_split_support_tree`` — they
maximise the same objective (Π P(split), expressed in log-space).

Three layers:

1. **Algorithm only** on 50 trees, presence matrix built from
   DendroPy bipartitions. Validates the argmax math in isolation.
2. **Full pipeline** on all 100 trees: presence matrix comes from
   ``rapidtrees`` (the production path). Validates the
   rapidtrees ↔ DendroPy split-encoding handshake.
3. **Algorithmic discrimination**: confirm the picked tree's score
   is strictly greater than the runner-up's AND the topologies
   differ — so the test isn't passing on a many-way tie.

Plus a few module-level tests for ``extract_log_posterior`` priority.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from treetracer.consensus_tree import compute_consensus_tree_index, extract_log_posterior


# ----- algorithm only (DendroPy presence) ----------------------------------


def _presence_from_dendropy(tlist):
    """Build TT-style (n_trees, n_splits) uint8 presence matrix from
    DendroPy bipartitions, mapping every distinct ``split_bitmask`` to
    a column."""
    split_to_col: dict[int, int] = {}
    rows: list[set[int]] = []
    for t in tlist:
        t.encode_bipartitions()
        my_splits: set[int] = set()
        for edge in t.preorder_edge_iter():
            b = edge.bipartition
            if b is None or edge.head_node is t.seed_node:
                continue
            split = int(b.split_bitmask)
            if split not in split_to_col:
                split_to_col[split] = len(split_to_col)
            my_splits.add(split_to_col[split])
        rows.append(my_splits)
    n, k = len(rows), len(split_to_col)
    presence = np.zeros((n, k), dtype=np.uint8)
    for i, cols in enumerate(rows):
        for c in cols:
            presence[i, c] = 1
    return presence


@pytest.mark.integration
def test_consensus_tree_algorithm_matches_dendropy_50(dendropy_trees_50):
    dendropy = pytest.importorskip("dendropy")
    presence = _presence_from_dendropy(dendropy_trees_50)
    tt_idx, _score = compute_consensus_tree_index(presence)

    array = dendropy.TreeArray(taxon_namespace=dendropy_trees_50.taxon_namespace)
    for t in dendropy_trees_50:
        array.add_tree(t)
    dp_tree = array.maximum_product_of_split_support_tree()

    rf = dendropy.calculate.treecompare.symmetric_difference(
        dendropy_trees_50[tt_idx], dp_tree,
    )
    assert rf == 0


# ----- full pipeline (rapidtrees presence) ---------------------------------


@pytest.mark.integration
def test_consensus_tree_full_pipeline_matches_dendropy(rapidtrees_full, dendropy_trees_100):
    """Run the production consensus tree path end-to-end: rapidtrees produces
    the presence matrix, ``compute_consensus_tree_index`` selects on it,
    DendroPy independently selects from the same trees and the picks
    must agree (RF=0)."""
    dendropy = pytest.importorskip("dendropy")
    _names, _rf, presence, _leaf_names, _n_bip = rapidtrees_full
    presence = np.asarray(presence, dtype=np.uint8)
    n = presence.shape[0]
    assert n == 100

    tt_idx, _score = compute_consensus_tree_index(presence)

    array = dendropy.TreeArray(taxon_namespace=dendropy_trees_100.taxon_namespace)
    for t in dendropy_trees_100:
        array.add_tree(t)
    dp_tree = array.maximum_product_of_split_support_tree()

    rf = dendropy.calculate.treecompare.symmetric_difference(
        dendropy_trees_100[tt_idx], dp_tree,
    )
    assert rf == 0


@pytest.mark.integration
def test_consensus_tree_runner_up_has_real_gap(rapidtrees_full, dendropy_trees_100):
    """Discrimination check: runner-up score must be strictly below
    the winner's AND the two trees must be topologically different —
    otherwise the previous tests could pass on a many-way tie."""
    dendropy = pytest.importorskip("dendropy")
    _names, _rf, presence, _, _ = rapidtrees_full
    presence = np.asarray(presence, dtype=np.uint8)
    n = presence.shape[0]
    counts = presence.astype(np.int64).sum(axis=0)
    freq = counts / n
    log_freq = np.log(np.where(counts > 0, freq, 1.0))
    scores = (presence.astype(np.int64) * log_freq).sum(axis=1)
    order = np.argsort(scores)[::-1]
    best, runner_up = int(order[0]), int(order[1])
    assert scores[best] > scores[runner_up] + 1e-9
    rf_gap = dendropy.calculate.treecompare.symmetric_difference(
        dendropy_trees_100[best], dendropy_trees_100[runner_up],
    )
    assert rf_gap > 0


# ----- extract_log_posterior priority --------------------------------------


def _row(meta):
    """A pandas.Series shaped like a row from ``db_manager._trees``."""
    return pd.Series({"metadata": meta})


def test_extract_log_posterior_prefers_lnP():
    r = _row({"lnP": -1234.5, "posterior": -1000.0, "lnL": -2000.0})
    assert extract_log_posterior(r) == -1234.5


def test_extract_log_posterior_falls_back_to_posterior_then_joint():
    assert extract_log_posterior(_row({"posterior": -42.0})) == -42.0
    assert extract_log_posterior(_row({"joint": -7.0})) == -7.0


def test_extract_log_posterior_ignores_lnL_only():
    # Pure log-likelihood is a different quantity; we deliberately
    # do not fall back to it.
    assert extract_log_posterior(_row({"lnL": -123.4})) is None


def test_extract_log_posterior_handles_json_string():
    r = _row(json.dumps({"lnP": -55.5}))
    assert extract_log_posterior(r) == -55.5


def test_extract_log_posterior_returns_none_on_missing_or_garbage():
    assert extract_log_posterior(_row(None)) is None
    assert extract_log_posterior(_row("not valid json {")) is None
    assert extract_log_posterior(_row({})) is None
