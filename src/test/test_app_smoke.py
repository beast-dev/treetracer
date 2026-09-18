"""App-import smoke + figure-layout invariants.

Goal: catch regressions where a callback signature drift or a missing
import breaks the Dash app at startup, even if no individual unit test
hits that code path. Also pin the trace-count constants the patch
callbacks rely on (``N_TRACES``, ``N_TRAILING_OVERLAYS``) so renaming
or restructuring traces can't silently shift their offsets.
"""

from __future__ import annotations

import json

import pandas as pd


def _walk_components(component):
    """Yield every Dash component below *component*, including itself."""
    if isinstance(component, (list, tuple)):
        for child in component:
            yield from _walk_components(child)
        return
    if component is None or isinstance(component, (str, int, float)):
        return
    yield component
    yield from _walk_components(getattr(component, "children", None))


def _component_by_id(component, component_id):
    return next(
        item
        for item in _walk_components(component)
        if getattr(item, "id", None) == component_id
    )


def test_app_imports_and_registers_callbacks():
    """Construct a real Dash app the way ``app.py`` does and register
    every callback. If any callback decorator throws on import (e.g.
    duplicate Output without ``allow_duplicate=True``, missing
    component id, etc.), this raises."""
    from dash import Dash
    import dash_mantine_components as dmc
    import treetracer.ui as ui
    from treetracer.callbacks import register_callbacks

    app = Dash(
        __name__,
        external_stylesheets=dmc.styles.ALL,
        suppress_callback_exceptions=True,
        assets_ignore=r"peartree\.bundle\.min\.js",
    )
    app.layout = ui.add_main_body()
    register_callbacks(app)


def test_summary_method_dropdowns_are_absent_from_both_analysis_tabs():
    from treetracer.ui.panels.treespace import _add_treespace_panel
    from treetracer.ui.panels.within_run import _add_within_run_panel

    component_ids = {
        getattr(component, "id", None)
        for panel in (_add_treespace_panel(), _add_within_run_panel())
        for component in _walk_components(panel)
    }
    assert {
        "treespace-summary-method",
        "treespace-summary-method-tooltip",
        "within-run-summary-method",
        "within-run-summary-method-tooltip",
    }.isdisjoint(component_ids)


def test_summary_tree_callbacks_use_mrhipstr_without_method_selectors():
    """Both View actions use the default without hidden selector state."""
    import inspect

    from dash import Dash, _callback
    import dash_mantine_components as dmc
    import treetracer.ui as ui
    from treetracer.callbacks import register_callbacks
    from treetracer.callbacks import consensus_tree_compute
    from treetracer.consensus_tree._subprocess_worker import (
        compute_consensus_tree_worker_entry,
    )

    app = Dash(
        __name__,
        external_stylesheets=dmc.styles.ALL,
        suppress_callback_exceptions=True,
        assets_ignore=r"peartree\.bundle\.min\.js",
    )
    app.layout = ui.add_main_body()
    register_callbacks(app)

    state_ids = set()
    for callback_data in _callback.GLOBAL_CALLBACK_MAP.values():
        callback_fn = callback_data.get("callback")
        callback_fn = getattr(callback_fn, "__wrapped__", callback_fn)
        if getattr(callback_fn, "__name__", "") != "view_consensus_tree":
            continue
        state_ids.update(item.get("id") for item in callback_data.get("state", []))

    assert {
        "treespace-summary-method",
        "within-run-summary-method",
    }.isdisjoint(state_ids)
    assert inspect.signature(
        consensus_tree_compute.submit_consensus_tree_job
    ).parameters["summary_method"].default == "mrhipstr"
    assert inspect.signature(
        compute_consensus_tree_worker_entry
    ).parameters["summary_method"].default == "mrhipstr"


def test_summary_tree_registry_has_method_column_and_synthetic_dashes(
    monkeypatch,
):
    from treetracer.callbacks import consensus_tree_list

    monkeypatch.setattr(
        consensus_tree_list.state,
        "has_cached_consensus_tree",
        lambda _uuid: True,
    )
    row = consensus_tree_list._entry_summary_row(
        {
            "name": "RF_001_Between_MrHIPSTR_1",
            "uuid": "synthetic-uuid",
            "mode": "Between",
            "selection": [["run-a", 1], ["run-b", 2]],
            "summary_method": "mrhipstr",
            "consensus_tree": {
                "group": None,
                "treenum": None,
                "tree_name": "MrHIPSTR",
            },
            "consensus_tree_log_posterior": None,
        },
        source="treespace",
    )
    cells = list(row.children)
    method_cell = next(
        cell
        for cell in cells
        if getattr(cell, "className", None) == "tt-consensus-tree-method-col"
    )
    assert method_cell.children.children == "MrHIPSTR"
    assert next(
        cell.children.children
        for cell in cells
        if getattr(cell, "className", None) == "tt-consensus-tree-run-col"
    ) == "—"
    assert next(
        cell.children
        for cell in cells
        if getattr(cell, "className", None) == "tt-consensus-tree-col"
    ) == "—"
    assert next(
        cell.children
        for cell in cells
        if getattr(cell, "className", None) == "tt-consensus-tree-lnp-col"
    ) == "—"

    table = consensus_tree_list._table_for(
        [
            {
                "name": "legacy-mcc",
                "uuid": "legacy-uuid",
                "selection": [],
                "consensus_tree": {},
            }
        ],
        source="treespace",
    )
    assert any(
        getattr(item, "children", None) == "Method"
        for item in _walk_components(table)
    )


def test_compute_interval_has_one_reconciliation_owner():
    """No feature callback may independently poll or stop the shared timer."""
    from dash import _callback

    owners = set()
    for callback_data in _callback.GLOBAL_CALLBACK_MAP.values():
        inputs = callback_data.get("inputs", [])
        output = callback_data.get("output")
        outputs = output if isinstance(output, list) else [output]
        reads_interval = any(
            item.get("id") == "compute-poll-interval" for item in inputs
        )
        writes_interval = any(
            getattr(item, "component_id", None) == "compute-poll-interval"
            for item in outputs
        )
        if not (reads_interval or writes_interval):
            continue
        callback_fn = callback_data.get("callback")
        callback_fn = getattr(callback_fn, "__wrapped__", callback_fn)
        owners.add(getattr(callback_fn, "__name__", ""))

    assert owners == {"reconcile_compute_job"}


def test_reconciler_piggybacks_receipts_as_state_without_an_ack_callback():
    """Browser receipts settle through the one polling owner.

    Keeping receipts as State avoids a receipt-triggered callback cycle while
    removing the independently schedulable acknowledgement request that could
    be starved by rapid terminal replays.
    """
    from dash import _callback

    reconciler = None
    callback_names = set()
    for callback_data in _callback.GLOBAL_CALLBACK_MAP.values():
        callback_fn = callback_data.get("callback")
        callback_fn = getattr(callback_fn, "__wrapped__", callback_fn)
        name = getattr(callback_fn, "__name__", "")
        callback_names.add(name)
        if name == "reconcile_compute_job":
            reconciler = callback_data

    assert reconciler is not None
    receipt_states = []
    for state in reconciler.get("state", []):
        component_id = state.get("id")
        if not isinstance(component_id, str) or not component_id.startswith("{"):
            continue
        parsed = json.loads(component_id)
        if parsed.get("type") == "compute-terminal-receipt":
            receipt_states.append((parsed, state.get("property")))

    assert receipt_states == [
        (
            {"kind": ["ALL"], "type": "compute-terminal-receipt"},
            "data",
        )
    ]
    assert "acknowledge_terminal_receipt" not in callback_names


def test_reconciler_uses_wildcards_for_dynamic_progress_banners():
    """RF and MDS banners never coexist, so concrete Outputs are unsafe.

    Dash rejects the whole reconciler response when a concrete output names
    the progress component belonging to the other, currently-unmounted banner.
    """
    from dash import _callback

    reconciler_outputs = None
    for callback_data in _callback.GLOBAL_CALLBACK_MAP.values():
        callback_fn = callback_data.get("callback")
        callback_fn = getattr(callback_fn, "__wrapped__", callback_fn)
        if getattr(callback_fn, "__name__", "") != "reconcile_compute_job":
            continue
        output = callback_data.get("output")
        reconciler_outputs = output if isinstance(output, list) else [output]
        break

    assert reconciler_outputs is not None
    component_ids = [item.component_id for item in reconciler_outputs]
    concrete_progress_ids = {
        "rf-progress-bar",
        "rf-progress-label",
        "mds-progress-bar",
        "mds-progress-label",
    }
    assert not any(
        isinstance(component_id, str)
        and component_id in concrete_progress_ids
        for component_id in component_ids
    )
    pattern_types = {
        component_id.get("type")
        for component_id in component_ids
        if isinstance(component_id, dict)
    }
    assert {
        "compute-progress-bar",
        "compute-progress-label",
    } <= pattern_types


def test_every_compute_action_reads_the_shared_busy_gate():
    """All entry points must become unavailable while the worker is owned."""
    from dash import _callback

    gated_outputs = set()
    for callback_data in _callback.GLOBAL_CALLBACK_MAP.values():
        inputs = callback_data.get("inputs", [])
        if not any(item.get("id") == "compute-busy-store" for item in inputs):
            continue
        output = callback_data.get("output")
        outputs = output if isinstance(output, list) else [output]
        gated_outputs.update(
            getattr(item, "component_id", None) for item in outputs
        )

    assert {
        "compute-rf-button",
        "compute-mds-button",
        "compute-rf-trace-button",
        "compute-pseudo-ess-button",
        "clade-freq-compare-button",
        "treespace-view-consensus-tree",
        "within-run-view-consensus-tree",
    } <= gated_outputs


def test_between_run_trailing_overlay_invariant():
    """Between-run figure has exactly 8 trailing overlays
    (4 selection + 4 consensus tree). The patch callbacks address them by fixed
    negative offsets, so changing this count without updating the
    callbacks silently breaks selection/consensus tree rendering."""
    from treetracer.plot_utils import (
        N_SELECTION_OVERLAYS, N_CONSENSUS_TREE_OVERLAYS, N_TRAILING_OVERLAYS,
        add_trace_multiplot_interleaved, make_plot_grid,
    )
    assert N_SELECTION_OVERLAYS == 4
    assert N_CONSENSUS_TREE_OVERLAYS == 4
    assert N_TRAILING_OVERLAYS == 8

    # Build a tiny figure and assert the last 8 traces are the overlay
    # bundles named "selection" and "consensus tree".
    df = pd.DataFrame({
        "group": ["a", "a", "b", "b"],
        "treenum": [1, 2, 1, 2],
        "tree": ["a/1", "a/2", "b/1", "b/2"],
        "MDS1": [0.1, 0.2, 0.3, 0.4],
        "MDS2": [0.5, 0.6, 0.7, 0.8],
        "MDS3": [0.9, 1.0, 1.1, 1.2],
    })
    fig = make_plot_grid()
    add_trace_multiplot_interleaved(
        fig, df, "MDS1", "MDS2", "MDS3",
        GROUPS=["a", "b"],
        COLOR_DICT={"a": "red", "b": "blue"},
        show_lines=False,
    )
    trailing = [tr.name for tr in fig.data[-N_TRAILING_OVERLAYS:]]
    assert trailing == ["selection"] * 4 + ["consensus tree"] * 4, trailing


def test_within_run_trace_count_invariant():
    """Within-run figure has exactly 12 traces:
    3 panels × {out-of-range, in-range} + 3 selection + 3 consensus tree overlays.
    The patch callbacks bail if ``len(fig.data) != N_TRACES``."""
    from treetracer.callbacks.within_run import (
        N_TRACES, SELECTION_OVERLAY_OFFSETS, CONSENSUS_TREE_OVERLAY_OFFSETS,
        _make_within_run_figure,
    )
    assert N_TRACES == 12
    assert SELECTION_OVERLAY_OFFSETS == (-6, -5, -4)
    assert CONSENSUS_TREE_OVERLAY_OFFSETS == (-3, -2, -1)

    df = pd.DataFrame({
        "group": ["runA"] * 5,
        "treenum": [1, 2, 3, 4, 5],
        "tree": [f"runA/STATE_{i}" for i in range(5)],
        "MDS1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "MDS2": [0.5, 0.4, 0.3, 0.2, 0.1],
        "MDS3": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    fig = _make_within_run_figure(df, "MDS1", "MDS2", "MDS3")
    assert len(fig.data) == N_TRACES, (
        f"within-run figure has {len(fig.data)} traces, expected {N_TRACES}"
    )


def test_theme_get_template_returns_valid_template():
    """The theme module is read every figure build; a non-string here
    would break Plotly's ``template=`` kwarg."""
    from treetracer.theme import get_template
    t = get_template()
    assert isinstance(t, str) and t, t
