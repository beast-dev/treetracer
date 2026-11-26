"""Unit tests for the pure-NumPy Sister-Clade Jaccard Instability engine."""

from __future__ import annotations

import numpy as np
import pytest

from treetracer.clade_freq.sister_instability import (
    _pairwise_jaccard,
    compute_sister_instability_from_arrays,
    compute_sister_instability_from_npz,
    format_tip_instability_tsv,
)
from treetracer.rf import rf_distance_with_rooted_facts_from_newick_iter
from treetracer.rf.rooted_facts import rooted_facts_npz_payload


def _rooted_facts_from_newicks(newicks, *, translate_maps=None):
    names = [f"tree-{i}" for i in range(len(newicks))]
    tmaps = translate_maps if translate_maps is not None else [{}]
    map_indices = [0] * len(newicks)
    _, _, facts = rf_distance_with_rooted_facts_from_newick_iter(
        names,
        iter(newicks),
        tmaps,
        map_indices,
    )
    return facts


def _clade_col_by_taxa(facts):
    """Map frozenset of quote-stripped taxon names -> clade column index."""
    leaves = [str(name).strip("'\"") for name in facts.leaf_names]
    bits = np.unpackbits(
        facts.packed_clades, axis=1, bitorder="little"
    )[:, : facts.n_taxa]
    mapping = {}
    for col in range(facts.n_clades):
        taxa = frozenset(leaves[idx] for idx in np.flatnonzero(bits[col]))
        mapping[taxa] = col
    return mapping


def _pack_clades(bit_rows):
    return np.packbits(np.asarray(bit_rows, dtype=np.uint8), axis=1, bitorder="little")


def test_6taxon_toy_exact(tmp_path):
    """Matches every tip and clade value in Table 1 of the implementation guide."""
    t1 = "((((A:1,B:1):1,R:1):1,C:1):1,(D:1,E:1):1);"
    t2 = "((((A:1,B:1):1,R:1):1,C:1):1,(D:1,E:1):1);"
    t3 = "(((A:1,B:1):1,C:1):1,((D:1,E:1):1,R:1):1);"
    t4 = "(((A:1,B:1):1,C:1):1,(D:1,(E:1,R:1):1):1);"
    facts = _rooted_facts_from_newicks([t1, t2, t3, t4])

    path = tmp_path / "toy6.npz"
    np.savez(path, **rooted_facts_npz_payload(facts))
    col_by_taxa = _clade_col_by_taxa(facts)
    internal_clades = [
        frozenset({"A", "B"}),
        frozenset({"D", "E"}),
        frozenset({"E", "R"}),
        frozenset({"A", "B", "C"}),
        frozenset({"A", "B", "R"}),
        frozenset({"D", "E", "R"}),
        frozenset({"A", "B", "C", "R"}),
    ]
    clade_cols = [col_by_taxa[c] for c in internal_clades]

    with np.load(path, allow_pickle=False) as snap:
        clade_metrics, tip_rows = compute_sister_instability_from_npz(
            snap,
            row_indices_1=[0, 1],
            row_indices_2=[2, 3],
            clade_columns=clade_cols,
        )

    assert len(tip_rows) == 6
    # R must rank #1 under default (D_sis desc, Comb desc, Taxon asc) ordering.
    assert tip_rows[0]["taxon"] == "R"

    by_tip = {row["taxon"]: row for row in tip_rows}
    expected_tips = {
        "A": (0.0, 0.0, 0.0, 0.0),
        "B": (0.0, 0.0, 0.0, 0.0),
        "C": (0.0, 0.0, 1.0 / 6.0, 1.0 / 3.0),
        "D": (0.0, 0.25, 3.0 / 16.0, 0.125),
        "E": (0.0, 0.5, 0.375, 0.25),
        "R": (0.0, 0.25, 0.5625, 0.875),
    }
    for taxon, (e_i1, e_i2, e_icomb, e_d) in expected_tips.items():
        row = by_tip[taxon]
        assert row["instab_1"] == pytest.approx(e_i1, abs=1e-12)
        assert row["instab_2"] == pytest.approx(e_i2, abs=1e-12)
        assert row["combined_instability"] == pytest.approx(e_icomb, abs=1e-12)
        assert row["d_sis"] == pytest.approx(e_d, abs=1e-12)
        # Exact variance-like decomposition for equal disjoint groups (w1=w2=0.5).
        assert row["combined_instability"] == pytest.approx(
            0.5 * row["instab_1"] + 0.5 * row["instab_2"] + 0.5 * row["d_sis"],
            abs=1e-12,
        )

    assert by_tip["R"]["n_sisters_1"] == 1
    assert by_tip["R"]["n_sisters_2"] == 2
    assert by_tip["R"]["n_sisters_comb"] == 3
    assert by_tip["R"]["p_cherry_1"] == pytest.approx(0.0, abs=1e-12)
    assert by_tip["R"]["p_cherry_2"] == pytest.approx(0.5, abs=1e-12)
    assert by_tip["R"]["top_sister_1"] == "2-tip sister (100%)"
    assert "1-tip cherry (50%)" in by_tip["R"]["top_sister_2"]
    assert "2-tip sister (50%)" in by_tip["R"]["top_sister_2"]

    expected_clades = {
        frozenset({"A", "B"}): (0.0, 0.0, 0.5, 1.0),
        frozenset({"D", "E"}): (0.0, 0.0, 1.0 / 3.0, 0.75),
        frozenset({"E", "R"}): (None, 0.0, 0.0, None),
        frozenset({"A", "B", "C"}): (None, 0.0, 0.0, None),
        frozenset({"A", "B", "R"}): (0.0, None, 0.0, None),
        frozenset({"D", "E", "R"}): (None, 0.0, 0.0, None),
        frozenset({"A", "B", "C", "R"}): (0.0, None, 0.0, None),
    }
    for clade, (e_i1, e_i2, e_icomb, e_d) in expected_clades.items():
        m = clade_metrics[col_by_taxa[clade]]
        assert m.clade_instability == pytest.approx(e_icomb, abs=1e-12)
        if e_i1 is None:
            assert m.instab_1 is None
        else:
            assert m.instab_1 == pytest.approx(e_i1, abs=1e-12)
        if e_i2 is None:
            assert m.instab_2 is None
        else:
            assert m.instab_2 == pytest.approx(e_i2, abs=1e-12)
        if e_d is None:
            assert m.d_sis is None
        else:
            assert m.d_sis == pytest.approx(e_d, abs=1e-12)


def test_7taxon_jumping_cherry():
    """Tips R1/R2 score 0.0 while their cherry {R1, R2} scores I_comb=0.5625, D_sis=0.875."""
    t1 = "((((A:1,B:1):1,(R1:1,R2:1):1):1,C:1):1,(D:1,E:1):1);"
    t2 = "((((A:1,B:1):1,(R1:1,R2:1):1):1,C:1):1,(D:1,E:1):1);"
    t3 = "(((A:1,B:1):1,C:1):1,((D:1,E:1):1,(R1:1,R2:1):1):1);"
    t4 = "(((A:1,B:1):1,C:1):1,(D:1,(E:1,(R1:1,R2:1):1):1):1);"
    facts = _rooted_facts_from_newicks([t1, t2, t3, t4])
    col_by_taxa = _clade_col_by_taxa(facts)

    targets = [
        frozenset({"R1", "R2"}),
        frozenset({"A", "B"}),
        frozenset({"D", "E"}),
    ]
    clade_metrics, tip_rows = compute_sister_instability_from_arrays(
        packed_clades=facts.packed_clades,
        split_table=facts.split_table,
        split_ids=facts.split_ids,
        leaf_names=facts.leaf_names,
        row_indices_1=[0, 1],
        row_indices_2=[2, 3],
        clade_columns=[col_by_taxa[t] for t in targets],
    )

    by_tip = {row["taxon"]: row for row in tip_rows}
    for tip in ("R1", "R2"):
        assert by_tip[tip]["instab_1"] == 0.0
        assert by_tip[tip]["instab_2"] == 0.0
        assert by_tip[tip]["combined_instability"] == 0.0
        assert by_tip[tip]["d_sis"] == 0.0
        assert by_tip[tip]["p_cherry_1"] == pytest.approx(1.0, abs=1e-12)
        assert by_tip[tip]["p_cherry_2"] == pytest.approx(1.0, abs=1e-12)

    cherry = clade_metrics[col_by_taxa[frozenset({"R1", "R2"})]]
    assert cherry.instab_1 == pytest.approx(0.0, abs=1e-12)
    assert cherry.instab_2 == pytest.approx(0.25, abs=1e-12)
    assert cherry.clade_instability == pytest.approx(0.5625, abs=1e-12)
    assert cherry.d_sis == pytest.approx(0.875, abs=1e-12)

    ab = clade_metrics[col_by_taxa[frozenset({"A", "B"})]]
    assert ab.clade_instability == pytest.approx(0.5, abs=1e-12)
    assert ab.d_sis == pytest.approx(1.0, abs=1e-12)

    de = clade_metrics[col_by_taxa[frozenset({"D", "E"})]]
    assert de.clade_instability == pytest.approx(4.0 / 15.0, abs=1e-12)
    assert de.d_sis == pytest.approx(0.6, abs=1e-12)


def test_polytomy_vs_switcher_ranking():
    """Shared 4-way uncertainty scores Comb.=0.75, D_sis=0.0; clean switcher ranks #1 by D_sis=1.0."""
    # Polytomy subtrees where X pairs uniformly with P1, P2, P3, P4 (4 disjoint singletons).
    poly = [
        "(((X:1,P1:1):1,P2:1):1,(P3:1,P4:1):1)",
        "(((X:1,P2:1):1,P3:1):1,(P4:1,P1:1):1)",
        "(((X:1,P3:1):1,P4:1):1,(P1:1,P2:1):1)",
        "(((X:1,P4:1):1,P1:1):1,(P2:1,P3:1):1)",
    ]
    # Group 1: R stably sister to {A,B}; Group 2: R stably sister to {C,D}.
    g1 = [f"((((A:1,B:1):1,R:1):1,(C:1,D:1):1):1,{p}:1);" for p in poly]
    g2 = [f"(((A:1,B:1):1,((C:1,D:1):1,R:1):1):1,{p}:1);" for p in poly]
    facts = _rooted_facts_from_newicks(g1 + g2)

    _, tip_rows = compute_sister_instability_from_arrays(
        packed_clades=facts.packed_clades,
        split_table=facts.split_table,
        split_ids=facts.split_ids,
        leaf_names=facts.leaf_names,
        row_indices_1=[0, 1, 2, 3],
        row_indices_2=[4, 5, 6, 7],
    )

    by_tip = {row["taxon"]: row for row in tip_rows}
    assert by_tip["X"]["instab_1"] == pytest.approx(0.75, abs=1e-12)
    assert by_tip["X"]["instab_2"] == pytest.approx(0.75, abs=1e-12)
    assert by_tip["X"]["combined_instability"] == pytest.approx(0.75, abs=1e-12)
    assert by_tip["X"]["d_sis"] == pytest.approx(0.0, abs=1e-12)

    assert by_tip["R"]["instab_1"] == pytest.approx(0.0, abs=1e-12)
    assert by_tip["R"]["instab_2"] == pytest.approx(0.0, abs=1e-12)
    assert by_tip["R"]["combined_instability"] == pytest.approx(0.5, abs=1e-12)
    assert by_tip["R"]["d_sis"] == pytest.approx(1.0, abs=1e-12)
    assert tip_rows[0]["taxon"] == "R"


def test_rare_clade_no_stray_spike():
    """Clade at (0.99, 0.001) gets I_comb ~= 0.002 under tree-weighted union pooling."""
    # Construct a minimal synthetic split table with 6 taxa:
    # Target clade C = {0, 1} (col 6), disjoint sisters A = {2, 3} (col 7) and B = {4, 5} (col 8).
    leaf_names = ["T0", "T1", "T2", "T3", "T4", "T5"]
    bit_rows = [
        [1, 0, 0, 0, 0, 0],  # 0: T0
        [0, 1, 0, 0, 0, 0],  # 1: T1
        [0, 0, 1, 0, 0, 0],  # 2: T2
        [0, 0, 0, 1, 0, 0],  # 3: T3
        [0, 0, 0, 0, 1, 0],  # 4: T4
        [0, 0, 0, 0, 0, 1],  # 5: T5
        [1, 1, 0, 0, 0, 0],  # 6: C = {T0, T1}
        [0, 0, 1, 1, 0, 0],  # 7: A = {T2, T3}
        [0, 0, 0, 0, 1, 1],  # 8: B = {T4, T5}
        [1, 1, 1, 1, 0, 0],  # 9: C U A
        [1, 1, 0, 0, 1, 1],  # 10: C U B
        [0, 1, 1, 0, 0, 0],  # 11: alt clade {T1, T2} (breaks C)
        [1, 1, 1, 0, 0, 0],  # 12: {T0, T1, T2}
        [0, 0, 0, 1, 1, 1],  # 13: {T3, T4, T5}
    ]
    packed_clades = _pack_clades(bit_rows)
    root_col = len(bit_rows)
    # Tree type 0 (C present, sister A): splits 0,1,2,3,4
    # Tree type 1 (C present, sister B): splits 0,1,2,5,6
    # Tree type 2 (C absent):            splits 7,8,2,9,10
    split_table = np.array(
        [
            [6, 0, 1],         # 0: C -> (T0, T1)
            [7, 2, 3],         # 1: A -> (T2, T3)
            [8, 4, 5],         # 2: B -> (T4, T5)
            [9, 6, 7],         # 3: C U A -> (C, A)
            [root_col, 8, 9],  # 4: root -> (B, C U A)
            [10, 6, 8],        # 5: C U B -> (C, B)
            [root_col, 7, 10], # 6: root -> (A, C U B)
            [11, 1, 2],        # 7: {T1,T2} -> (T1, T2)
            [12, 0, 11],       # 8: {T0,T1,T2} -> (T0, {T1,T2})
            [13, 3, 8],        # 9: {T3,T4,T5} -> (T3, B)
            [root_col, 12, 13],# 10: root -> ({T0,T1,T2}, {T3,T4,T5})
        ],
        dtype=np.uint32,
    )
    tree_a = np.array([0, 1, 2, 3, 4], dtype=np.uint32)
    tree_b = np.array([0, 1, 2, 5, 6], dtype=np.uint32)
    tree_no_c = np.array([2, 7, 8, 9, 10], dtype=np.uint32)

    # G1: 990 trees with (C, A) + 10 trees without C (f1 = 0.99).
    # G2: 1 stray tree with (C, B) + 999 trees without C (f2 = 0.001).
    split_ids = np.vstack(
        [
            np.repeat(tree_a[None, :], 990, axis=0),
            np.repeat(tree_no_c[None, :], 10, axis=0),
            tree_b[None, :],
            np.repeat(tree_no_c[None, :], 999, axis=0),
        ]
    )

    clade_metrics, _ = compute_sister_instability_from_arrays(
        packed_clades=packed_clades,
        split_table=split_table,
        split_ids=split_ids,
        leaf_names=leaf_names,
        row_indices_1=range(0, 1000),
        row_indices_2=range(1000, 2000),
        clade_columns=[6],
    )

    m = clade_metrics[6]
    expected_icomb = 2.0 * (990.0 / 991.0) * (1.0 / 991.0)
    assert m.instab_1 == pytest.approx(0.0, abs=1e-12)
    assert m.instab_2 == pytest.approx(0.0, abs=1e-12)
    assert m.d_sis == pytest.approx(1.0, abs=1e-12)
    assert m.clade_instability == pytest.approx(expected_icomb, abs=1e-12)
    assert m.clade_instability < 0.003


def test_overlapping_rows_deduplicated():
    """Subset R2 <= R1 gives I_comb == I_1 with zero double-counting; R1 == R2 gives D_sis == 0."""
    t1 = "((((A:1,B:1):1,R:1):1,C:1):1,(D:1,E:1):1);"
    t2 = "(((A:1,B:1):1,C:1):1,((D:1,E:1):1,R:1):1);"
    t3 = "(((A:1,B:1):1,C:1):1,(D:1,(E:1,R:1):1):1);"
    facts = _rooted_facts_from_newicks([t1, t2, t3])
    col_by_taxa = _clade_col_by_taxa(facts)
    ab_col = col_by_taxa[frozenset({"A", "B"})]

    # R2 = {1, 2} is a strict subset of R1 = {0, 1, 2}, plus duplicate index inside R1.
    clade_metrics, tip_rows = compute_sister_instability_from_arrays(
        packed_clades=facts.packed_clades,
        split_table=facts.split_table,
        split_ids=facts.split_ids,
        leaf_names=facts.leaf_names,
        row_indices_1=[0, 1, 2, 2],
        row_indices_2=[1, 2],
        clade_columns=[ab_col],
    )
    for row in tip_rows:
        assert row["combined_instability"] == pytest.approx(row["instab_1"], abs=1e-12)
    assert clade_metrics[ab_col].clade_instability == pytest.approx(
        clade_metrics[ab_col].instab_1, abs=1e-12
    )

    # Identical selections R1 == R2 -> D_sis == 0 everywhere, I_1 == I_2 == I_comb.
    clade_same, tips_same = compute_sister_instability_from_arrays(
        packed_clades=facts.packed_clades,
        split_table=facts.split_table,
        split_ids=facts.split_ids,
        leaf_names=facts.leaf_names,
        row_indices_1=[0, 1, 2],
        row_indices_2=[0, 1, 2],
        clade_columns=[ab_col],
    )
    for row in tips_same:
        assert row["d_sis"] == pytest.approx(0.0, abs=1e-12)
        assert row["instab_1"] == pytest.approx(row["combined_instability"], abs=1e-12)
        assert row["instab_2"] == pytest.approx(row["combined_instability"], abs=1e-12)
    assert clade_same[ab_col].d_sis == pytest.approx(0.0, abs=1e-12)


def test_packed_vs_matmul_parity():
    """k_c = 300 sister sets match between bitwise AND+popcount and FP32 U @ U.T."""
    rng = np.random.default_rng(20261123)
    n_taxa = 64
    k_sisters = 300

    # Generate 300 distinct non-empty sister clades (never touching taxon 0, our target tip).
    seen = set()
    sister_bits = []
    while len(sister_bits) < k_sisters:
        row = np.zeros(n_taxa, dtype=np.uint8)
        size = int(rng.integers(1, 20))
        chosen = rng.choice(np.arange(1, n_taxa), size=size, replace=False)
        row[chosen] = 1
        key = row.tobytes()
        if key not in seen:
            seen.add(key)
            sister_bits.append(row)

    # Singletons 0..n_taxa-1 first, then the 300 sister clades, then 300 parent clades (target U sister).
    singleton_bits = np.eye(n_taxa, dtype=np.uint8)
    sister_arr = np.vstack(sister_bits)
    parent_arr = sister_arr.copy()
    parent_arr[:, 0] = 1
    all_bits = np.vstack([singleton_bits, sister_arr, parent_arr])
    packed_clades = _pack_clades(all_bits)
    sizes = np.bitwise_count(packed_clades).sum(axis=1, dtype=np.int32)
    sisters = np.arange(n_taxa, n_taxa + k_sisters, dtype=np.intp)

    j_packed = _pairwise_jaccard(
        packed_clades, sizes, sisters, n_taxa, packed_threshold=1000
    )
    j_matmul = _pairwise_jaccard(
        packed_clades, sizes, sisters, n_taxa, packed_threshold=0
    )
    np.testing.assert_allclose(j_packed, j_matmul, atol=1e-12, rtol=0.0)

    # End-to-end check through compute_sister_instability_from_arrays crossing k_c=256 threshold.
    # Build 300 one-split "rows" exercising target tip 0 paired with each of the 300 sisters.
    split_table = np.column_stack(
        [
            np.arange(n_taxa + k_sisters, n_taxa + 2 * k_sisters, dtype=np.uint32),
            np.zeros(k_sisters, dtype=np.uint32),
            sisters.astype(np.uint32),
        ]
    )
    split_ids = np.arange(k_sisters, dtype=np.uint32)[:, None]
    leaf_names = [f"T{i}" for i in range(n_taxa)]

    _, tips_packed = compute_sister_instability_from_arrays(
        packed_clades=packed_clades,
        split_table=split_table,
        split_ids=split_ids,
        leaf_names=leaf_names,
        row_indices_1=range(0, 150),
        row_indices_2=range(150, 300),
        packed_threshold=1000,
    )
    _, tips_matmul = compute_sister_instability_from_arrays(
        packed_clades=packed_clades,
        split_table=split_table,
        split_ids=split_ids,
        leaf_names=leaf_names,
        row_indices_1=range(0, 150),
        row_indices_2=range(150, 300),
        packed_threshold=256,
    )

    t0_packed = next(row for row in tips_packed if row["taxon"] == "T0")
    t0_matmul = next(row for row in tips_matmul if row["taxon"] == "T0")
    for field in ("instab_1", "instab_2", "combined_instability", "d_sis"):
        assert t0_packed[field] == pytest.approx(t0_matmul[field], abs=1e-12)
    assert t0_matmul["n_sisters_comb"] == k_sisters
    # Every sister in Group 1 has p = 1/150 (~0.67%), so top 1 formats compactly at (1%).
    assert t0_matmul["top_sister_1"].endswith("(1%)")


def test_root_stem_and_quoted_names():
    """Root-stem splits (parent == root_column) and quoted NEXUS taxon names score cleanly."""
    t1 = "('T_root':1,(\"T_a\":1,(B:1,C:1):1):1);"
    t2 = "((\"T_a\":1,'T_root':1):1,(B:1,C:1):1);"
    facts = _rooted_facts_from_newicks([t1, t2])

    # Confirm root-stem split really uses parent == root_column == n_clades.
    assert np.any(facts.split_table[:, 0] == facts.root_column)

    _, tip_rows = compute_sister_instability_from_arrays(
        packed_clades=facts.packed_clades,
        split_table=facts.split_table,
        split_ids=facts.split_ids,
        leaf_names=facts.leaf_names,
        row_indices_1=[0],
        row_indices_2=[1],
    )
    by_tip = {row["taxon"]: row for row in tip_rows}
    assert set(by_tip.keys()) == {"T_root", "T_a", "B", "C"}

    # T_root switches from sister {T_a, B, C} (size 3) in G1 to {T_a} (size 1) in G2.
    # J = 1/3 -> D_sis = 1 - 1/3 = 2/3, I_comb = 2 * 0.5 * 0.5 * (2/3) = 1/3.
    assert by_tip["T_root"]["d_sis"] == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert by_tip["T_root"]["combined_instability"] == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert by_tip["T_root"]["p_cherry_1"] == pytest.approx(0.0, abs=1e-12)
    assert by_tip["T_root"]["p_cherry_2"] == pytest.approx(1.0, abs=1e-12)
    assert by_tip["T_root"]["top_sister_1"] == "3-tip sister (100%)"
    assert by_tip["T_root"]["top_sister_2"] == "1-tip cherry (100%)"

    from treetracer.clade_freq.sister_instability import _format_pct

    assert _format_pct(1.0) == "100%"
    assert _format_pct(0.999) == ">99%"
    assert _format_pct(0.002) == "<1%"


def test_npz_selective_reads_preloaded_reuse_and_tsv():
    """Loader skips heavy arrays, supports preloaded packed_clades/leaf_names, and formats TSV."""
    t1 = "((((A:1,B:1):1,R:1):1,C:1):1,(D:1,E:1):1);"
    t2 = "(((A:1,B:1):1,C:1):1,((D:1,E:1):1,R:1):1);"
    facts = _rooted_facts_from_newicks([t1, t2])
    payload = rooted_facts_npz_payload(facts)

    class _SpySnapshot:
        def __init__(self, data):
            self._data = data
            self.accessed = set()

        @property
        def files(self):
            return list(self._data.keys())

        def __getitem__(self, key):
            self.accessed.add(key)
            return self._data[key]

    spy_default = _SpySnapshot(payload)
    _, tip_rows_default = compute_sister_instability_from_npz(
        spy_default,
        row_indices_1=[0],
        row_indices_2=[1],
    )
    assert spy_default.accessed.isdisjoint(
        {
            "rooted_facts_clade_columns",
            "rooted_facts_node_heights",
            "rooted_facts_root_heights",
        }
    )

    spy_reused = _SpySnapshot(payload)
    _, tip_rows_reused = compute_sister_instability_from_npz(
        spy_reused,
        row_indices_1=[0],
        row_indices_2=[1],
        packed_clades=facts.packed_clades,
        leaf_names=facts.leaf_names,
    )
    assert "rooted_facts_packed_clades" not in spy_reused.accessed
    assert "rooted_facts_leaf_names" not in spy_reused.accessed
    assert tip_rows_reused == tip_rows_default

    # Non-rooted / empty snapshot raises ValueError cleanly.
    with pytest.raises(ValueError, match="rooted facts"):
        compute_sister_instability_from_npz(
            {"sparse_snapshot_format_version": np.asarray(1, dtype=np.uint8)},
            row_indices_1=[0],
            row_indices_2=[1],
        )

    tsv = format_tip_instability_tsv(tip_rows_default)
    lines = tsv.strip().splitlines()
    assert len(lines) == 1 + len(tip_rows_default)
    assert lines[0].startswith("taxon\td_sis\tinstab_1\tinstab_2\tcombined_instability\t")
    assert lines[1].startswith("R\t1.000000\t")
