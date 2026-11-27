"""Integration tests for sister-clade instability in the worker and Compare Consensus callbacks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import pytest

from treetracer import state
from treetracer.background_jobs import JobManager, JobState
from treetracer.callbacks import clade_explore
from treetracer.clade_freq._subprocess_worker import (
    compute_clade_frequencies_worker_entry,
)
from treetracer.rf import rf_distance_with_rooted_facts_from_newick_iter
from treetracer.rf.rooted_facts import rooted_facts_npz_payload
from treetracer.rf.sparse_snapshots import (
    SparseSnapshot,
    sparse_snapshot_npz_payload,
)


@pytest.fixture(autouse=True)
def _clean_analysis_results():
    state.clear_all_analysis_results()
    yield
    state.clear_all_analysis_results()


def _write_6taxon_rooted_npz(path):
    newicks = [
        "((((A:1,B:1):1,R:1):1,C:1):1,(D:1,E:1):1);",
        "((((A:1,B:1):1,R:1):1,C:1):1,(D:1,E:1):1);",
        "(((A:1,B:1):1,C:1):1,((D:1,E:1):1,R:1):1);",
        "(((A:1,B:1):1,C:1):1,(D:1,(E:1,R:1):1):1);",
    ]
    names = [f"tree-{i}" for i in range(len(newicks))]
    _, _, facts = rf_distance_with_rooted_facts_from_newick_iter(
        names, iter(newicks), [{}], [0] * len(newicks)
    )
    np.savez(path, **rooted_facts_npz_payload(facts))
    return facts


def test_worker_rooted_vs_unrooted(tmp_path):
    """Rooted facts returns has_instability=True + sorted tip_rows; generic CSR degrades cleanly."""
    rooted_path = tmp_path / "rooted.npz"
    facts = _write_6taxon_rooted_npz(rooted_path)
    all_cols = list(range(facts.n_clades))

    rooted = compute_clade_frequencies_worker_entry(
        snapshots_path=str(rooted_path),
        columns=all_cols,
        counts_1=None,
        counts_2=None,
        n_trees_1=0,
        n_trees_2=0,
        tree_names_1=["tree-0", "tree-1"],
        tree_names_2=["tree-2", "tree-3"],
        full_distmat_names=list(facts.tree_names),
        row_indices_1=[0, 1],
        row_indices_2=[2, 3],
    )
    assert rooted["has_instability"] is True
    assert len(rooted["tip_rows"]) == 6
    assert rooted["tip_rows"][0]["taxon"] == "R"
    assert rooted["tip_rows"][0]["d_sis"] == pytest.approx(0.875, abs=1e-12)
    assert all(row["clade_instability"] is not None for row in rooted["rows"])

    # Omitting row_indices preserves backward compatibility and disables instability cleanly.
    no_rows = compute_clade_frequencies_worker_entry(
        snapshots_path=str(rooted_path),
        columns=all_cols,
        counts_1=[1] * len(all_cols),
        counts_2=[1] * len(all_cols),
        n_trees_1=2,
        n_trees_2=2,
    )
    assert no_rows["has_instability"] is False
    assert no_rows["tip_rows"] == []
    assert all(row["clade_instability"] is None for row in no_rows["rows"])

    # Generic unrooted CSR snapshot degrades cleanly to has_instability=False.
    csr_path = tmp_path / "unrooted.npz"
    bits = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    sparse = SparseSnapshot(
        tree_names=("t0", "t1"),
        leaf_names=("A", "B", "C", "D"),
        n_clades=2,
        packed_clades=np.packbits(bits, axis=1, bitorder="little"),
        row_offsets=np.array([0, 2, 4], dtype=np.uint64),
        column_indices=np.array([0, 1, 0, 1], dtype=np.uint32),
        rooted=False,
    )
    np.savez(csr_path, **sparse_snapshot_npz_payload(sparse))
    unrooted = compute_clade_frequencies_worker_entry(
        snapshots_path=str(csr_path),
        columns=[0, 1],
        counts_1=[1, 1],
        counts_2=[1, 1],
        n_trees_1=1,
        n_trees_2=1,
        row_indices_1=[0],
        row_indices_2=[1],
    )
    assert unrooted["has_instability"] is False
    assert unrooted["tip_rows"] == []
    assert all(row["clade_instability"] is None for row in unrooted["rows"])


def test_finalizer_excludes_singletons_and_caches_instability(monkeypatch):
    """Finalizer drops clade_size==1 from scatter records and caches tip_rows server-side."""
    manager = JobManager(id_factory=lambda: "instab-job", clock=lambda: 10.0)
    monkeypatch.setattr(clade_explore, "job_manager", manager)
    monkeypatch.setattr(clade_explore, "add_log", lambda *_a, **_k: None)

    context = clade_explore._CladeComparisonFinalizationContext(
        source_distmat="RF_001",
        uid_1="uid-1",
        uid_2="uid-2",
        label_1="G1",
        label_2="G2",
        consensus_columns_1=frozenset({0, 1, 2}),
        consensus_columns_2=frozenset({0, 1, 3}),
        min_clade_size=2,
        exclusive_only=False,
    )
    worker_result = {
        "rows": [
            {
                "split_key": (0,),
                "column_j": 0,
                "freq_1": 1.0,
                "freq_2": 1.0,
                "clade_size": 1,
                "clade_instability": 0.5625,
                "instab_1": 0.0,
                "instab_2": 0.25,
                "d_sis": 0.875,
            },
            {
                "split_key": (0, 1),
                "column_j": 1,
                "freq_1": 1.0,
                "freq_2": 1.0,
                "clade_size": 2,
                "clade_instability": 0.5,
                "instab_1": 0.0,
                "instab_2": 0.0,
                "d_sis": 1.0,
            },
            {
                "split_key": (0, 1, 2),
                "column_j": 2,
                "freq_1": 1.0,
                "freq_2": 0.0,
                "clade_size": 3,
                "clade_instability": 0.0,
                "instab_1": 0.0,
                "instab_2": None,
                "d_sis": None,
            },
        ],
        "leaf_names": ["A", "B", "C"],
        "has_instability": True,
        "tip_rows": [
            {
                "taxon_index": 0,
                "taxon": "A",
                "column_j": 0,
                "d_sis": 0.875,
                "instab_1": 0.0,
                "instab_2": 0.25,
                "combined_instability": 0.5625,
                "n_sisters_1": 1,
                "n_sisters_2": 2,
                "n_sisters_comb": 3,
                "p_cherry_1": 0.0,
                "p_cherry_2": 0.5,
                "top_sister_1": "{B, C} 100%",
                "top_sister_2": "B (cherry) 50%, C (cherry) 50%",
            }
        ],
        "snapshot_input_mode": "rooted_facts",
        "elapsed": 0.01,
    }

    with ThreadPoolExecutor(max_workers=1) as executor:
        ref = manager.submit(
            executor,
            "clade_compare",
            lambda: worker_result,
            finalizer=partial(
                clade_explore._finalize_clade_comparison_job,
                context=context,
            ),
        )
        while manager.snapshot(ref).terminal is None:
            pass
        terminal = manager.snapshot(ref)

    assert terminal.state is JobState.SUCCEEDED
    assert "tip_rows" not in terminal.terminal.payload

    cached = state.get_clade_frequency_result(ref.job_id)
    assert cached["has_instability"] is True
    assert len(cached["tip_rows"]) == 1
    assert cached["label_1"] == "G1"
    assert cached["label_2"] == "G2"
    # Singleton column 0 was excluded globally; only clade_size >= 2 remain.
    assert [r["clade_size"] for r in cached["records"]] == [2, 3]
    assert cached["records"][0]["clade_instability"] == pytest.approx(0.5)
    assert cached["records"][0]["n_eff_comb"] == pytest.approx(2.0)
    assert cached["records"][0]["instability_hover"] == (
        "<br>Instab: 0.50 (N<sub>eff</sub>: 2.00; G1 1.00, G2 1.00)<br>D<sub>sis</sub>: 1"
    )
    assert cached["records"][1]["instability_hover"] == "<br>Instab: 0.00 (N<sub>eff</sub>: 1.00)"
    assert clade_explore._format_instability_hover(0.75, 0.0023456, 0.74, 0.76) == (
        "<br>Instab: 0.75 (N<sub>eff</sub>: 4.00; G1 3.85, G2 4.17)<br>D<sub>sis</sub>: 0.002346"
    )
    assert clade_explore._format_instability_hover(0.75, 1e-16, 0.75, 0.75) == (
        "<br>Instab: 0.75 (N<sub>eff</sub>: 4.00; G1 4.00, G2 4.00)<br>D<sub>sis</sub>: 0"
    )
    assert clade_explore._format_neff(1.0) == "1.00"
    assert clade_explore._format_neff(12.34) == "12.3"
    assert clade_explore._format_neff(124.4) == "124"
    # Explicit dict title ensures downstream Patch on colorbar.title.text works.
    assert cached["figure"]["data"][0]["marker"]["colorbar"]["title"] == {"text": "Clade size"}


def _registered_callback(name):
    from dash import _callback

    def matches():
        found = []
        for cb in _callback.GLOBAL_CALLBACK_MAP.values():
            fn = cb.get("callback")
            if fn is None:
                continue
            orig = getattr(fn, "__wrapped__", fn)
            if orig.__name__ == name:
                found.append(orig)
        return found

    found = matches()
    if not found:
        clade_explore.register_clade_explore_callbacks()
        found = matches()
    assert found, f"callback {name!r} not registered"
    return found[-1]


def _patch_ops(patch_obj):
    return {
        tuple(op["location"]): op["params"]["value"]
        for op in patch_obj.to_plotly_json()["operations"]
    }


def test_filter_patch_color_and_slider(monkeypatch):
    """Switching Color-by toggles sliders, patches cmin=0/cmax=1, lifts size-5 floor, and survives retheme."""
    toggle = _registered_callback("toggle_clade_freq_slider_wrappers")
    assert toggle("clade_size") == ({}, {"display": "none"})
    assert toggle("clade_instability") == ({"display": "none"}, {})

    records = [
        {
            "split_id": 0,
            "freq_1": 1.0,
            "freq_2": 1.0,
            "clade_size": 2,
            "in_consensus_tree_1": True,
            "in_consensus_tree_2": True,
            "consensus_tree_membership": "both consensus trees",
            "clade_instability": 0.5625,
            "instability_hover": "<br>Sister instability: 0.563 (D<sub>sis</sub>: 0.875)",
        },
        {
            "split_id": 1,
            "freq_1": 0.9,
            "freq_2": 0.1,
            "clade_size": 12,
            "in_consensus_tree_1": True,
            "in_consensus_tree_2": False,
            "consensus_tree_membership": "G1 only",
            "clade_instability": 0.05,
            "instability_hover": "<br>Sister instability: 0.050",
        },
    ]

    filter_cb = _registered_callback("filter_clade_freq_plot")
    # Clade size mode with [5, 50] hides the size-2 jumping cherry.
    patch_size, banner_style, banner_text = filter_cb("clade_size", [5, 50], [0.0, 1.0], False, None, records, None)
    assert banner_style == {"display": "none"}
    assert banner_text == ""
    ops_size = _patch_ops(patch_size)
    assert ops_size[("data", 0, "x")] == [0.9]
    assert ops_size[("data", 0, "marker", "color")] == [12]
    assert ops_size[("data", 0, "marker", "cauto")] is True
    assert ops_size[("data", 0, "marker", "cmin")] is None
    assert ops_size[("data", 0, "marker", "colorbar", "title", "text")] == "Clade size"

    # Clade size upper bound isolates cherries ([2, 2]) and hides the size-12 clade.
    patch_size_hi, _, _ = filter_cb("clade_size", [2, 2], [0.0, 1.0], False, None, records, None)
    ops_size_hi = _patch_ops(patch_size_hi)
    assert ops_size_hi[("data", 0, "x")] == [1.0]
    assert ops_size_hi[("data", 0, "marker", "color")] == [2]

    # Clade instability mode lifts the size-5 floor, applies [lo, hi], and fixes colorbar to [0.0, 1.0].
    patch_instab, banner_style_i, _ = filter_cb("clade_instability", [5, 50], [0.25, 1.0], False, None, records, None)
    assert banner_style_i == {"display": "none"}
    ops_instab = _patch_ops(patch_instab)
    assert ops_instab[("data", 0, "x")] == [1.0]
    assert ops_instab[("data", 0, "marker", "color")] == [0.5625]
    assert ops_instab[("data", 0, "marker", "cauto")] is False
    assert ops_instab[("data", 0, "marker", "cmin")] == 0.0
    assert ops_instab[("data", 0, "marker", "cmax")] == 1.0
    assert ops_instab[("data", 0, "marker", "colorbar", "title", "text")] == "Instability"

    # Instability upper bound ([0.0, 0.25]) keeps only the low-instability clade.
    patch_instab_hi, _, _ = filter_cb("clade_instability", [5, 50], [0.0, 0.25], False, None, records, None)
    ops_instab_hi = _patch_ops(patch_instab_hi)
    assert ops_instab_hi[("data", 0, "x")] == [0.9]
    assert ops_instab_hi[("data", 0, "marker", "color")] == [0.05]

    # Theme toggle (Viridis -> Turbo) preserves cauto=False, cmin=0.0, cmax=1.0 and uirevision.
    import pandas as pd

    fig = clade_explore._build_scatter_fig(pd.DataFrame(records), "G1", "G2")
    assert fig.layout.uirevision == "clade-freq-scatter"
    fig.data[0].marker.cauto = False
    fig.data[0].marker.cmin = 0.0
    fig.data[0].marker.cmax = 1.0
    monkeypatch.setattr(clade_explore, "get_template", lambda: clade_explore.DARK_TEMPLATE)
    retheme_cb = _registered_callback("retheme_clade_scatter")
    rethemed = retheme_cb(clade_explore.DARK_TEMPLATE, fig.to_dict())
    assert rethemed.data[0].marker.cauto is False
    assert rethemed.data[0].marker.cmin == 0.0
    assert rethemed.data[0].marker.cmax == 1.0


def test_range_sliders_layout_and_dynamic_size_bounds():
    """Both scatter filters are RangeSliders with minRange=0 and clade-size ceiling syncs to data."""
    import dash_mantine_components as dmc
    from dash import no_update
    from treetracer.ui.panels.clade_freq import _add_clade_freq_panel

    def _walk(node):
        if node is None or isinstance(node, (str, int, float, bool)):
            return
        yield node
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                yield from _walk(child)
        elif children is not None:
            yield from _walk(children)

    panel = _add_clade_freq_panel()
    by_id = {getattr(c, "id", None): c for c in _walk(panel) if getattr(c, "id", None)}
    size_slider = by_id["clade-freq-min-clade-size"]
    instab_slider = by_id["clade-freq-min-instability"]
    assert isinstance(size_slider, dmc.RangeSlider)
    assert size_slider.min == 2
    assert size_slider.max == 50
    assert size_slider.minRange == 0
    assert list(size_slider.value) == [5, 50]
    assert isinstance(instab_slider, dmc.RangeSlider)
    assert instab_slider.min == 0.0
    assert instab_slider.max == 1.0
    assert instab_slider.minRange == 0.0
    assert list(instab_slider.value) == [0.0, 1.0]

    sync_bounds = _registered_callback("sync_clade_size_slider_bounds")

    # Larger tree (max clade size 120): expands ceiling and lifts top handle parked at old max.
    state.store_clade_frequency_result(
        "k-big",
        {"records": [{"clade_size": 2}, {"clade_size": 120}]},
    )
    new_max, marks, new_val = sync_bounds("k-big", [5, 50], 50)
    assert new_max == 120
    assert marks[0]["value"] == 2 and marks[-1]["value"] == 120
    assert new_val == [5, 120]

    # Explicit user upper cap (< old_max) is preserved across same-size comparisons.
    out_max, _, out_val = sync_bounds("k-big", [5, 20], 120)
    assert out_max is no_update
    assert out_val is no_update

    # Switching to a tiny 3-tip tree (data_max == 2) resets lo to 2 and keeps max >= 3.
    state.store_clade_frequency_result("k-tiny", {"records": [{"clade_size": 2}]})
    tiny_max, tiny_marks, tiny_val = sync_bounds("k-tiny", [5, 120], 120)
    assert tiny_max == 3
    assert [m["value"] for m in tiny_marks] == [2, 3]
    assert tiny_val == [2, 3]

    # Missing/empty records returns no_update cleanly.
    assert sync_bounds(None, [5, 50], 50) == (no_update, no_update, no_update)


def _make_tip_rows(n, *, max_dsis=0.8):
    rows = []
    for i in range(n):
        d_sis = max_dsis if i == 0 else (0.2 if i == 1 and max_dsis > 0 else 0.0)
        comb = 0.75 if i == 2 else (0.5 if d_sis > 0 else 0.0)
        rows.append(
            {
                "taxon_index": i,
                "taxon": f"TAXON_{i:03d}",
                "column_j": i,
                "d_sis": d_sis,
                "instab_1": 0.75 if i == 2 else 0.0,
                "instab_2": 0.75 if i == 2 else 0.25 * (d_sis > 0),
                "combined_instability": comb,
                "n_sisters_1": 1,
                "n_sisters_2": 2,
                "n_sisters_comb": 3,
                "p_cherry_1": 0.0,
                "p_cherry_2": 0.5,
                "top_sister_1": "{A, B} 100%",
                "top_sister_2": "C (cherry) 50%, {D, E} 50%",
            }
        )
    return rows


def test_sync_controls_auto_sort_and_unrooted_fallback():
    """Reactive sync enables/disables instability, resets to clade_size, and auto-sorts when D_sis==0."""
    sync_cb = _registered_callback("sync_instability_controls_on_result")

    state.store_clade_frequency_result(
        "k-rooted",
        {"has_instability": True, "tip_rows": _make_tip_rows(5, max_dsis=0.8), "label_1": "G1", "label_2": "G2"},
    )
    opts, val, tip_off, sort_by, search, limit, tip_flt, exp_dis, ctrl_style = sync_cb(
        "k-rooted", "clade_instability"
    )
    assert opts[1]["disabled"] is False
    assert val == "clade_size"
    assert tip_off is True
    assert sort_by == "d_sis"
    assert search == ""
    assert limit == 50
    assert tip_flt is None
    assert exp_dis is False
    assert ctrl_style == {}

    # Identical selections (max D_sis == 0) auto-switch default sort to combined_instability.
    state.store_clade_frequency_result(
        "k-same",
        {"has_instability": True, "tip_rows": _make_tip_rows(5, max_dsis=0.0), "label_1": "G1", "label_2": "G1"},
    )
    _, _, _, sort_same, _, _, _, _, _ = sync_cb("k-same", "clade_size")
    assert sort_same == "combined_instability"

    # Unrooted snapshot disables instability option, turns on tooltip, and hides table controls.
    state.store_clade_frequency_result(
        "k-unrooted",
        {"has_instability": False, "tip_rows": [], "label_1": "U1", "label_2": "U2"},
    )
    opts_u, val_u, tip_off_u, _, _, _, tip_flt_u, exp_dis_u, ctrl_u = sync_cb(
        "k-unrooted", "clade_instability"
    )
    assert opts_u[1]["disabled"] is True
    assert val_u == "clade_size"
    assert tip_off_u is False
    assert tip_flt_u is None
    assert exp_dis_u is True
    assert ctrl_u == {"display": "none"}


def test_tip_table_search_sort_pagination_unrooted_and_tsv(monkeypatch, tmp_path):
    """Tip Table paginates at 50, searches across all L tips, sorts, renders unrooted fallback, and exports TSV."""
    from dash import no_update
    render_cb = _registered_callback("render_tip_instability_table")
    bump_cb = _registered_callback("bump_tip_instability_limit")
    export_cb = _registered_callback("export_tip_instability_tsv")

    tip_rows = _make_tip_rows(60, max_dsis=0.9)
    state.store_clade_frequency_result(
        "k-60",
        {"has_instability": True, "tip_rows": tip_rows, "label_1": "Run 1", "label_2": "Run 2"},
    )

    table, badge, badge_style, summary, more_style, _footer = render_cb(
        "k-60", "", "d_sis", 50, None
    )
    assert badge == "2 D_sis > 0.05"
    assert badge_style == {}
    assert "Showing 50 of 60 tips" in summary
    assert "2 with D_sis > 0.05" in summary
    assert "N_comb > 1.05" in summary
    assert more_style == {}
    # Column headers render N_G1, N_G2, D_sis, Group % with explanatory tooltips.
    header_cells = table.children[0].children.children
    assert header_cells[1].children.children[0] == "N"
    assert header_cells[1].children.children[1].children == "G1"
    assert header_cells[2].children.children[0] == "N"
    assert header_cells[2].children.children[1].children == "G2"
    dsis_th = header_cells[3].children
    assert dsis_th.title == "Net Sister Dissimilarity Coefficient (0–1)"
    assert dsis_th.children[0] == "D"
    assert dsis_th.children[1].children == "sis"
    r2_th = header_cells[4].children
    assert r2_th.children == "Group %"
    assert "ANOVA R²" in r2_th.title
    # tbody holds 50 rows; TAXON_000 (I1=0, I2=0.25, D_sis=0.90) -> N_G1=1.00, N_G2=1.33, D_sis=0.90, R^2=78%.
    tbody_rows = table.children[1].children
    assert len(tbody_rows) == 50
    assert tbody_rows[0].children[0].children.children == "TAXON_000"
    assert tbody_rows[0].children[1].children == "1.00"
    assert tbody_rows[0].children[2].children == "1.33"
    assert tbody_rows[0].children[3].children.children == "0.90"
    assert tbody_rows[0].children[4].children == "78%"
    assert tbody_rows[0].children[4].className == "tt-tip-col-r2"
    # TAXON_001 (I1=0, I2=0.25, D_sis=0.20) -> R^2=44% (unmuted).
    assert tbody_rows[1].children[4].children == "44%"
    assert tbody_rows[1].children[4].className == "tt-tip-col-r2"
    # TAXON_002 (shared rogue: I1=I2=0.75, D_sis=0) -> R^2=0% (muted).
    assert tbody_rows[2].children[0].children.children == "TAXON_002"
    assert tbody_rows[2].children[4].children == "0%"
    assert tbody_rows[2].children[4].className == "tt-tip-col-r2 tt-r2-muted"
    # TAXON_003 (completely stable: I1=I2=D_sis=0) -> R^2="—" (muted).
    assert tbody_rows[3].children[0].children.children == "TAXON_003"
    assert tbody_rows[3].children[4].children == "—"
    assert tbody_rows[3].children[4].className == "tt-tip-col-r2 tt-r2-muted"

    # Pagination bump -> 100 hides Show more.
    assert bump_cb(1, 50) == 100
    table_all, _, _, summary_all, more_all, _ = render_cb("k-60", "", "d_sis", 100, None)
    assert len(table_all.children[1].children) == 60
    assert "Showing 60 of 60 tips" in summary_all
    assert more_all == {"display": "none"}

    # Search finds taxon #59 beyond initial 50-row cap.
    table_q, _, _, _, _, _ = render_cb("k-60", "059", "d_sis", 50, None)
    assert len(table_q.children[1].children) == 1
    assert table_q.children[1].children[0].children[0].children.children == "TAXON_059"

    # Sort by combined_instability puts TAXON_002 (Comb.=0.75) first; active tip filter highlights TAXON_000.
    table_comb, _, _, _, _, footer_sel = render_cb(
        "k-60", "", "combined_instability", 50, {"taxon": "TAXON_000", "result_key": "k-60"}
    )
    assert table_comb.children[1].children[0].children[0].children.children == "TAXON_002"
    selected_rows = [tr for tr in table_comb.children[1].children if tr.className == "tt-tip-row-selected"]
    assert len(selected_rows) == 1
    assert "TAXON_000" in str(footer_sel)
    assert "(2.00 eff.)" in str(footer_sel)

    # TSV export opens native save dialog and writes header + all 60 tips.
    seen_default = []
    out_file = tmp_path / "out.tsv"

    def _fake_save(default_filename="output.tsv", **_kwargs):
        seen_default.append(default_filename)
        return str(out_file)

    monkeypatch.setattr(clade_explore, "_save_file_dialog", _fake_save)
    notif = export_cb(1, "k-60")
    assert seen_default == ["Run_1_vs_Run_2_tip_instability.tsv"]
    assert len(out_file.read_text(encoding="utf-8").strip().splitlines()) == 61
    assert notif.color == "green"

    # Cancelled dialog returns no_update.
    monkeypatch.setattr(clade_explore, "_save_file_dialog", lambda **_kwargs: None)
    assert export_cb(2, "k-60") is no_update

    # Unrooted empty-state card.
    state.store_clade_frequency_result(
        "k-unr",
        {"has_instability": False, "tip_rows": [], "label_1": "U1", "label_2": "U2"},
    )
    empty_body, _, empty_badge_style, _, _, _ = render_cb("k-unr", "", "d_sis", 50, None)
    assert "Tip Instability Unavailable" in str(empty_body)
    assert empty_badge_style == {"display": "none"}


def test_tip_click_filters_scatter_banner_and_dismiss():
    """Tip filter stacks with scatter controls and clade dot clicks without resetting data[0]."""
    from dash import no_update

    store_click = _registered_callback("store_scatter_click")
    dismiss_cb = _registered_callback("dismiss_tip_selection_banner")
    filter_cb = _registered_callback("filter_clade_freq_plot")
    marker_cb = _registered_callback("update_click_marker")

    # Scatter click stamps kind="clade" and clears clickData; banner dismiss clears click store only in tip mode.
    clade_payload, cleared_click = store_click(
        {"points": [{"x": 1.0, "y": 0.0, "customdata": [0, 3, "G1 only", ""]}]},
        "k-rings",
        None,
        None,
    )
    assert cleared_click is None
    assert clade_payload["kind"] == "clade"
    assert clade_payload["split_id"] == 0
    assert dismiss_cb(1, {"kind": "tip", "taxon": "R", "result_key": "k-rings"}) == (None, None)
    assert dismiss_cb(1, clade_payload) == (no_update, None)

    records = [
        {
            "split_id": 0,
            "freq_1": 1.0,
            "freq_2": 0.0,
            "clade_size": 3,
            "in_consensus_tree_1": True,
            "in_consensus_tree_2": False,
            "consensus_tree_membership": "G1 only",
            "clade_instability": 0.0,
        },
        {
            "split_id": 1,
            "freq_1": 0.0,
            "freq_2": 0.5,
            "clade_size": 2,
            "in_consensus_tree_1": False,
            "in_consensus_tree_2": True,
            "consensus_tree_membership": "G2 only",
            "clade_instability": 0.0,
        },
        {
            "split_id": 2,
            "freq_1": 1.0,
            "freq_2": 1.0,
            "clade_size": 2,
            "in_consensus_tree_1": True,
            "in_consensus_tree_2": True,
            "consensus_tree_membership": "both consensus trees",
            "clade_instability": 0.5,
        },
    ]
    state.store_clade_frequency_result(
        "k-rings",
        {
            "records": records,
            "resolution": {
                0: {"source_distmat": "RF_001", "column_j": 10, "tip_names": ("A", "B", "R")},
                1: {"source_distmat": "RF_001", "column_j": 11, "tip_names": ("E", "R")},
                2: {"source_distmat": "RF_001", "column_j": 12, "tip_names": ("A", "B")},
            },
            "pair": ("u1", "u2"),
            "has_instability": True,
            "tip_rows": [],
        },
    )
    tip_filter = {"taxon": "R", "result_key": "k-rings"}

    # Active tip filter R in clade_instability mode filters data[0] to splits 0 and 1.
    patch_tip, banner_style, banner_text = filter_cb(
        "clade_instability", [3, 50], [0.0, 1.0], False, tip_filter, records, {"kind": "tip", "taxon": "R"}
    )
    ops = _patch_ops(patch_tip)
    assert ops[("data", 0, "x")] == [1.0, 0.0]
    assert ops[("data", 0, "y")] == [0.0, 0.5]
    assert ops[("data", 1, "x")] == []
    assert banner_style == {}
    assert banner_text == "Showing 2 clades containing R"

    # Changing scatter controls (Clade size in [3, 50]) stacks with tip filter R AND keeps visible clade ring.
    patch_tip_sz, banner_sz_style, banner_sz = filter_cb(
        "clade_size", [3, 50], [0.0, 1.0], False, tip_filter, records, clade_payload
    )
    ops_tip_sz = _patch_ops(patch_tip_sz)
    assert ops_tip_sz[("data", 0, "x")] == [1.0]
    assert ops_tip_sz[("data", 1, "x")] == [1.0]
    assert ops_tip_sz[("data", 1, "y")] == [0.0]
    assert banner_sz_style == {}
    assert banner_sz == "Showing 1 clade containing R"

    # Filtering out the currently ringed clade (Clade size in [4, 50]) clears the ring cleanly.
    patch_filtered_out, _, _ = filter_cb(
        "clade_size", [4, 50], [0.0, 1.0], False, tip_filter, records, clade_payload
    )
    assert _patch_ops(patch_filtered_out)[("data", 1, "x")] == []

    # Clicking a clade dot inside the subselection only patches data[1] (never touches data[0]).
    patch_clade = marker_cb(clade_payload, records)
    clade_ops = _patch_ops(patch_clade)
    assert ("data", 0, "x") not in clade_ops
    assert clade_ops[("data", 1, "x")] == [1.0]
    assert clade_ops[("data", 1, "y")] == [0.0]

    # Clicking a tip clears the single-clade ring on data[1] without touching data[0].
    patch_tip_ring = marker_cb({"kind": "tip", "taxon": "R", "result_key": "k-rings"}, records)
    tip_ring_ops = _patch_ops(patch_tip_ring)
    assert ("data", 0, "x") not in tip_ring_ops
    assert tip_ring_ops[("data", 1, "x")] == []
    assert marker_cb(None, None) is no_update

    # Dismissing the tip filter (tip_filter=None) restores all clades in data[0], hides banner,
    # and keeps the ring if a clade is currently focused.
    patch_clear, clear_style, clear_text = filter_cb(
        "clade_size", [2, 50], [0.0, 1.0], False, None, records, clade_payload
    )
    clear_ops = _patch_ops(patch_clear)
    assert clear_ops[("data", 0, "x")] == [1.0, 0.0, 1.0]
    assert clear_ops[("data", 1, "x")] == [1.0]
    assert clear_style == {"display": "none"}
    assert clear_text == ""


def test_click_selected_clade_toggles_off():
    """Re-clicking the selected red circle clears selection or restores the active tip view."""
    from dash import no_update

    store_click = _registered_callback("store_scatter_click")
    marker_cb = _registered_callback("update_click_marker")
    click_evt = {"points": [{"x": 1.0, "y": 0.0, "customdata": [0, 3, "G1 only", ""]}]}
    other_evt = {"points": [{"x": 0.0, "y": 0.5, "customdata": [1, 2, "G2 only", ""]}]}
    tip_filter = {"taxon": "R", "result_key": "k-tog"}

    # First click selects clade 0 and resets clickData to None so Dash fires on re-click.
    selected, reset_1 = store_click(click_evt, "k-tog", None, None)
    assert selected["kind"] == "clade" and selected["split_id"] == 0
    assert reset_1 is None
    assert store_click(None, "k-tog", selected, None) == (no_update, no_update)

    # Clicking clade 0 again with no tip filter clears to None and resets clickData.
    undone, reset_2 = store_click(click_evt, "k-tog", selected, None)
    assert undone is None and reset_2 is None
    assert _patch_ops(marker_cb(undone, [{"split_id": 0}]))[("data", 1, "x")] == []

    # Clicking a different dot switches directly to clade 1.
    sel_other, reset_3 = store_click(other_evt, "k-tog", selected, tip_filter)
    assert sel_other["kind"] == "clade" and sel_other["split_id"] == 1
    assert reset_3 is None

    # Clicking the selected clade 1 again WITH an active tip filter returns to tip view.
    back_to_tip, reset_4 = store_click(other_evt, "k-tog", sel_other, tip_filter)
    assert back_to_tip["kind"] == "tip"
    assert back_to_tip["taxon"] == "R"
    assert back_to_tip["result_key"] == "k-tog"
    assert reset_4 is None
    assert _patch_ops(marker_cb(back_to_tip, [{"split_id": 1}]))[("data", 1, "x")] == []


def test_tip_mode_tanglegram_consensus_sisters_complement_and_clear():
    """Tip tanglegram highlights G1/G2 sisters; complement toggle connects sisters with landing tips under red."""
    from dash import no_update

    def _nexus(newick):
        return (
            "#NEXUS\n"
            "Begin trees;\n"
            f"tree t1 = [&R] {newick}\n"
            "End;\n"
        ).encode("utf-8")

    state.clear_all_consensus_trees()
    clade_explore.clear_clade_freq_caches()
    try:
        uid1 = state.cache_consensus_tree(_nexus("((((A:1,B:1):1,R:1):1,C:1):1,(D:1,E:1):1);"))
        uid2 = state.cache_consensus_tree(_nexus("(((A:1,B:1):1,C:1):1,(D:1,(E:1,R:1):1):1);"))
        state.register_consensus_tree(
            source_distmat="RF_001",
            mode="Between",
            run=None,
            uuid=uid1,
            consensus_tree={"tree_name": "G1"},
            selection=[],
            log_clade_credibility=0.0,
            cols_in_consensus_tree=[10],
        )
        state.register_consensus_tree(
            source_distmat="RF_001",
            mode="Between",
            run=None,
            uuid=uid2,
            consensus_tree={"tree_name": "G2"},
            selection=[],
            log_clade_credibility=0.0,
            cols_in_consensus_tree=[10],
        )
        state.store_clade_frequency_result(
            "k-tang",
            {
                "records": [],
                "resolution": {
                    0: {"source_distmat": "RF_001", "column_j": 10, "tip_names": ("A", "B")},
                    1: {"source_distmat": "RF_001", "column_j": 11, "tip_names": ("A", "B", "R")},
                },
                "pair": (uid1, uid2),
                "has_instability": True,
                "tip_rows": [
                    {
                        "taxon": "R",
                        "d_sis": 0.875,
                        "top_sister_1": "2-tip sister (100%)",
                        "top_sister_2": "1-tip cherry (50%), 2-tip sister (50%)",
                    }
                ],
            },
        )

        draw_cb = _registered_callback("draw_tanglegram")
        comp_cb = _registered_callback("toggle_complement_highlights")
        sister_cb = _registered_callback("toggle_sister_connectors")

        tip_click = {"kind": "tip", "taxon": "R", "result_key": "k-tang", "_t": 1}
        # Tip mode with Connect sister OFF: origin sister branches + tips + MRCA diamonds, connectors empty, complement disabled.
        fig, pair, title, comp_disabled = draw_cb(tip_click, uid1, uid2, 1, None, False, False)
        assert pair == [uid1, uid2]
        assert comp_disabled is True
        assert len(fig.data) == 19
        assert (
            clade_explore._TANGLEGRAM_TIP_SISTER_CONN_LEFT
            < clade_explore._TANGLEGRAM_TIP_SISTER_CONN_RIGHT
            < clade_explore._TANGLEGRAM_CONNECTORS
        )
        assert fig.data[clade_explore._TANGLEGRAM_TIP_SISTER_CONN_LEFT].line.width == 1.0
        assert fig.data[clade_explore._TANGLEGRAM_TIP_SISTER_CONN_RIGHT].line.width == 1.0
        assert len(fig.data[clade_explore._TANGLEGRAM_TIP_SISTER_CONN_LEFT].x) == 0
        assert len(fig.data[clade_explore._TANGLEGRAM_TIP_SISTER_CONN_RIGHT].x) == 0
        assert list(fig.data[clade_explore._TANGLEGRAM_HIGHLIGHT_LEFT].text) == ["R"]
        assert list(fig.data[clade_explore._TANGLEGRAM_HIGHLIGHT_RIGHT].text) == ["R"]
        assert len(fig.data[clade_explore._TANGLEGRAM_CONNECTORS].x) == 3
        assert len(fig.data[clade_explore._TANGLEGRAM_MRCA_SUBTREE].x) == 0
        assert len(fig.data[clade_explore._TANGLEGRAM_SISTER_BRANCH_LEFT].x) > 0
        assert len(fig.data[clade_explore._TANGLEGRAM_SISTER_BRANCH_RIGHT].x) > 0
        assert list(fig.data[clade_explore._TANGLEGRAM_TIP_SISTER_TIPS_LEFT].text) == ["A", "B"]
        assert list(fig.data[clade_explore._TANGLEGRAM_TIP_SISTER_TIPS_RIGHT].text) == ["E"]
        assert list(fig.data[clade_explore._TANGLEGRAM_SISTER_MRCA_LEFT].text) == ["G1 sister MRCA — 2 tips"]
        assert list(fig.data[clade_explore._TANGLEGRAM_SISTER_MRCA_RIGHT].text) == ["G2 sister MRCA — 1 tip"]
        assert "0.88" in str(title)

        # Clade complement no-ops in tip mode.
        assert comp_cb(True, tip_click, uid1, uid2) is no_update

        # Connect sister ON in tip mode: connects G1 sister (A, B) in blue and G2 sister (E) in green,
        # colors intact landing branches, and places origin + landing MRCA diamonds on both trees.
        patch_on = sister_cb(True, tip_click, uid1, uid2)
        on_ops = _patch_ops(patch_on)
        assert len(on_ops[("data", clade_explore._TANGLEGRAM_TIP_SISTER_CONN_LEFT, "x")]) == 6
        assert len(on_ops[("data", clade_explore._TANGLEGRAM_TIP_SISTER_CONN_RIGHT, "x")]) == 3
        assert len(on_ops[("data", clade_explore._TANGLEGRAM_TIP_SISTER_TIPS_LEFT, "x")]) == 4
        assert len(on_ops[("data", clade_explore._TANGLEGRAM_TIP_SISTER_TIPS_RIGHT, "x")]) == 2
        assert len(on_ops[("data", clade_explore._TANGLEGRAM_SISTER_BRANCH_LEFT, "x")]) > len(
            fig.data[clade_explore._TANGLEGRAM_SISTER_BRANCH_LEFT].x
        )
        assert len(on_ops[("data", clade_explore._TANGLEGRAM_SISTER_BRANCH_RIGHT, "x")]) > len(
            fig.data[clade_explore._TANGLEGRAM_SISTER_BRANCH_RIGHT].x
        )
        assert on_ops[("data", clade_explore._TANGLEGRAM_SISTER_MRCA_LEFT, "text")] == [
            "G1 sister MRCA — 2 tips",
            "G1 sister MRCA — 2 tips",
        ]
        assert on_ops[("data", clade_explore._TANGLEGRAM_SISTER_MRCA_RIGHT, "text")] == [
            "G2 sister MRCA — 1 tip",
            "G2 sister MRCA — 1 tip",
        ]

        # Connect sister OFF in tip mode hides sister connectors and keeps origin-side sister branches/tips/MRCA.
        patch_off = sister_cb(False, tip_click, uid1, uid2)
        off_ops = _patch_ops(patch_off)
        assert off_ops[("data", clade_explore._TANGLEGRAM_TIP_SISTER_CONN_LEFT, "x")] == []
        assert off_ops[("data", clade_explore._TANGLEGRAM_TIP_SISTER_CONN_RIGHT, "x")] == []
        assert len(off_ops[("data", clade_explore._TANGLEGRAM_TIP_SISTER_TIPS_LEFT, "x")]) == 2
        assert len(off_ops[("data", clade_explore._TANGLEGRAM_TIP_SISTER_TIPS_RIGHT, "x")]) == 1
        assert off_ops[("data", clade_explore._TANGLEGRAM_SISTER_MRCA_LEFT, "text")] == [
            "G1 sister MRCA — 2 tips"
        ]

        # Clade mode ({A, B}, monophyletic in BOTH trees): Connect sister OFF clears sister traces & re-enables complement.
        clade_both = {"kind": "clade", "split_id": 0, "result_key": "k-tang", "_t": 2}
        patch_clade, pair2, _, comp_disabled_clade = draw_cb(
            clade_both, uid1, uid2, 1, [uid1, uid2], False, False
        )
        assert pair2 is no_update
        assert comp_disabled_clade is False
        clade_ops = _patch_ops(patch_clade)
        for idx in (
            clade_explore._TANGLEGRAM_SISTER_BRANCH_LEFT,
            clade_explore._TANGLEGRAM_SISTER_BRANCH_RIGHT,
            clade_explore._TANGLEGRAM_TIP_SISTER_CONN_LEFT,
            clade_explore._TANGLEGRAM_TIP_SISTER_CONN_RIGHT,
            clade_explore._TANGLEGRAM_TIP_SISTER_TIPS_LEFT,
            clade_explore._TANGLEGRAM_TIP_SISTER_TIPS_RIGHT,
            clade_explore._TANGLEGRAM_SISTER_MRCA_LEFT,
            clade_explore._TANGLEGRAM_SISTER_MRCA_RIGHT,
        ):
            assert clade_ops[("data", idx, "x")] == []
        assert len(clade_ops[("data", clade_explore._TANGLEGRAM_HIGHLIGHT_LEFT, "x")]) == 2

        # Connect sister ON for {A, B}: colors G1 sister (R, blue) & G2 sister (C, green) branches + MRCA diamonds + connectors.
        patch_clade_sis = sister_cb(True, clade_both, uid1, uid2)
        cs_ops = _patch_ops(patch_clade_sis)
        assert cs_ops[("data", clade_explore._TANGLEGRAM_TIP_SISTER_TIPS_LEFT, "text")] == ["R", "R"]
        assert cs_ops[("data", clade_explore._TANGLEGRAM_TIP_SISTER_TIPS_RIGHT, "text")] == ["C", "C"]
        assert len(cs_ops[("data", clade_explore._TANGLEGRAM_SISTER_BRANCH_LEFT, "x")]) > 0
        assert len(cs_ops[("data", clade_explore._TANGLEGRAM_SISTER_BRANCH_RIGHT, "x")]) > 0
        assert cs_ops[("data", clade_explore._TANGLEGRAM_SISTER_MRCA_LEFT, "text")] == [
            "G1 sister MRCA — 1 tip",
            "G1 sister MRCA — 1 tip",
        ]
        assert cs_ops[("data", clade_explore._TANGLEGRAM_SISTER_MRCA_RIGHT, "text")] == [
            "G2 sister MRCA — 1 tip",
            "G2 sister MRCA — 1 tip",
        ]
        assert len(cs_ops[("data", clade_explore._TANGLEGRAM_TIP_SISTER_CONN_LEFT, "x")]) == 3
        assert len(cs_ops[("data", clade_explore._TANGLEGRAM_TIP_SISTER_CONN_RIGHT, "x")]) == 3

        # Clade mode ({A, B, R}, monophyletic ONLY in G1, broken in G2): only blue G1 sister (C) branches/MRCA/connectors drawn.
        clade_g1_only = {"kind": "clade", "split_id": 1, "result_key": "k-tang", "_t": 3}
        patch_g1_sis = sister_cb(True, clade_g1_only, uid1, uid2)
        g1_ops = _patch_ops(patch_g1_sis)
        assert g1_ops[("data", clade_explore._TANGLEGRAM_TIP_SISTER_TIPS_LEFT, "text")] == ["C", "C"]
        assert len(g1_ops[("data", clade_explore._TANGLEGRAM_SISTER_BRANCH_LEFT, "x")]) > 0
        assert len(g1_ops[("data", clade_explore._TANGLEGRAM_SISTER_MRCA_LEFT, "x")]) == 2
        assert len(g1_ops[("data", clade_explore._TANGLEGRAM_TIP_SISTER_CONN_LEFT, "x")]) == 3
        assert g1_ops[("data", clade_explore._TANGLEGRAM_SISTER_BRANCH_RIGHT, "x")] == []
        assert g1_ops[("data", clade_explore._TANGLEGRAM_SISTER_MRCA_RIGHT, "x")] == []
        assert g1_ops[("data", clade_explore._TANGLEGRAM_TIP_SISTER_TIPS_RIGHT, "x")] == []
        assert g1_ops[("data", clade_explore._TANGLEGRAM_TIP_SISTER_CONN_RIGHT, "x")] == []

        # Dismissing selection (click_data=None) Patches all dynamic overlays back to [].
        patch_clear, pair3, title_clear, comp_disabled_clear = draw_cb(
            None, uid1, uid2, 1, [uid1, uid2], False, True
        )
        assert pair3 is no_update
        assert title_clear == ""
        assert comp_disabled_clear is False
        clear_ops = _patch_ops(patch_clear)
        for idx in (
            clade_explore._TANGLEGRAM_MRCA_SUBTREE,
            clade_explore._TANGLEGRAM_SISTER_BRANCH_LEFT,
            clade_explore._TANGLEGRAM_SISTER_BRANCH_RIGHT,
            clade_explore._TANGLEGRAM_TIP_SISTER_CONN_LEFT,
            clade_explore._TANGLEGRAM_TIP_SISTER_CONN_RIGHT,
            clade_explore._TANGLEGRAM_TIP_SISTER_TIPS_LEFT,
            clade_explore._TANGLEGRAM_TIP_SISTER_TIPS_RIGHT,
            clade_explore._TANGLEGRAM_HIGHLIGHT_LEFT,
            clade_explore._TANGLEGRAM_HIGHLIGHT_RIGHT,
            clade_explore._TANGLEGRAM_CONNECTORS,
            clade_explore._TANGLEGRAM_MRCA_MARKER,
            clade_explore._TANGLEGRAM_SISTER_MRCA_LEFT,
            clade_explore._TANGLEGRAM_SISTER_MRCA_RIGHT,
            clade_explore._TANGLEGRAM_COMPLEMENT_TIPS,
            clade_explore._TANGLEGRAM_COMPLEMENT_CONN,
        ):
            assert clear_ops[("data", idx, "x")] == []
    finally:
        state.clear_all_consensus_trees()
        clade_explore.clear_clade_freq_caches()

