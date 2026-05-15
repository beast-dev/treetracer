"""App-import smoke + figure-layout invariants.

Goal: catch regressions where a callback signature drift or a missing
import breaks the Dash app at startup, even if no individual unit test
hits that code path. Also pin the trace-count constants the patch
callbacks rely on (``N_TRACES``, ``N_TRAILING_OVERLAYS``) so renaming
or restructuring traces can't silently shift their offsets.
"""

from __future__ import annotations

import pandas as pd
import pytest


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


def test_between_run_trailing_overlay_invariant():
    """Between-run figure has exactly 8 trailing overlays
    (4 selection + 4 MCC). The patch callbacks address them by fixed
    negative offsets, so changing this count without updating the
    callbacks silently breaks selection/MCC rendering."""
    from treetracer.plot_utils import (
        N_SELECTION_OVERLAYS, N_MCC_OVERLAYS, N_TRAILING_OVERLAYS,
        add_trace_multiplot_interleaved, make_plot_grid,
    )
    assert N_SELECTION_OVERLAYS == 4
    assert N_MCC_OVERLAYS == 4
    assert N_TRAILING_OVERLAYS == 8

    # Build a tiny figure and assert the last 8 traces are the overlay
    # bundles named "selection" and "mcc".
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
    assert trailing == ["selection"] * 4 + ["mcc"] * 4, trailing


def test_within_run_trace_count_invariant():
    """Within-run figure has exactly 12 traces:
    3 panels × {out-of-range, in-range} + 3 selection + 3 MCC overlays.
    The patch callbacks bail if ``len(fig.data) != N_TRACES``."""
    from treetracer.callbacks.within_run import (
        N_TRACES, SELECTION_OVERLAY_OFFSETS, MCC_OVERLAY_OFFSETS,
        _make_within_run_figure,
    )
    assert N_TRACES == 12
    assert SELECTION_OVERLAY_OFFSETS == (-6, -5, -4)
    assert MCC_OVERLAY_OFFSETS == (-3, -2, -1)

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
