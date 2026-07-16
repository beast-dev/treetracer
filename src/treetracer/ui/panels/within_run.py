"""Within-run Analysis tab panel: per-run scatter with playback, gradient
coloring, range slider, and the within-run MCC registry table."""

import dash_mantine_components as dmc
from dash import dcc, html

from ...icons import icon
from ...plot_utils import placeholder_fig
from ..widgets import stop_button


def _add_within_run_panel():
    """Build the Within-run Analysis tab panel content (visualization only)."""
    return html.Div([
        # MCC-compute loading overlay — see treespace panel for the
        # full pattern. Toggled by ``view_mcc_tree`` (on) and
        # ``mcc_compute.poll_mcc_completion`` (off).
        dmc.LoadingOverlay(
            id="within-run-loading-overlay",
            visible=False,
            zIndex=1000,
            overlayProps={"radius": "sm", "blur": 2},
            loaderProps={"size": "lg", "type": "dots",
                         "children": dmc.Stack(
                             [dmc.Text("Computing MCC tree…",
                                       size="sm", c="dimmed"),
                              stop_button("mcc-within")],
                             align="center", gap="xs", mt="sm")},
        ),
        dmc.Stack([
            # Row 1: Result + Run selectors + axis selectors + info (boxed)
            dmc.Paper(
                dmc.Group([
                    html.Div(
                        dmc.Select(
                            label="MDS Result",
                            id="within-run-result-select",
                            placeholder="No MDS results yet",
                            data=[], value=None, size="xs",
                            style={"width": "100%"},
                        ),
                        style={"width": "300px", "flexShrink": 0, "flexGrow": 0},
                    ),
                    html.Div(
                        dmc.Select(
                            label="Run",
                            id="within-run-run-select",
                            placeholder="Run",
                            data=[], value=None, size="xs",
                            style={"width": "100%"},
                        ),
                        style={"width": "180px", "flexShrink": 0, "flexGrow": 0},
                    ),
                    html.Div(
                        dmc.Select(label="X", id="within-run-dim-x",
                                   data=[], value=None, size="xs",
                                   style={"width": "100%"}),
                        style={"width": "100px", "flexShrink": 0, "flexGrow": 0},
                    ),
                    html.Div(
                        dmc.Select(label="Y", id="within-run-dim-y",
                                   data=[], value=None, size="xs",
                                   style={"width": "100%"}),
                        style={"width": "100px", "flexShrink": 0, "flexGrow": 0},
                    ),
                    html.Div(
                        dmc.Select(label="Z", id="within-run-dim-z",
                                   data=[], value=None, size="xs",
                                   style={"width": "100%"}),
                        style={"width": "100px", "flexShrink": 0, "flexGrow": 0},
                    ),
                    html.Div(id="within-run-info"),
                    html.Div(id="within-run-selection-info"),
                ], align="flex-end", gap="md", wrap="nowrap"),
                withBorder=True, p="sm", radius="sm", shadow="xs", mt="sm",
                style={"width": "100%"},
            ),

            # Row 2: All controls (hidden until a result is selected, no border)
            dmc.Group([
                # Range slider + playback
                dmc.Text("Trees:", size="xs", fw=500, style={"alignSelf": "center"}),
                html.Div([
                    dmc.RangeSlider(
                        id="within-run-treenum-slider",
                        min=1, max=100, value=[1, 100],
                        minRange=1, step=1,
                        size="xs",
                        styles={"markLabel": {"fontSize": "10px"}},
                    ),
                ], style={"width": "420px", "alignSelf": "center"}),
                dmc.Text("Window:", size="xs", fw=500, style={"alignSelf": "center"}),
                dmc.NumberInput(
                    id="within-run-window-size",
                    value=100, min=10, step=10,
                    size="xs", w=70,
                    styles={"input": {"height": "28px"}},
                ),
                dmc.ActionIcon(
                    icon("tabler:player-play-filled", size=20),
                    id="within-run-play-button",
                    variant="filled", color="blue", size="md",
                ),
                dmc.Stack([
                    dmc.Checkbox(label="Lines", id="within-run-show-lines",
                                 checked=True, size="xs"),
                    dmc.Checkbox(label="Gradient", id="within-run-color-gradient",
                                 checked=True, size="xs"),
                ], gap=2),
                dmc.Divider(orientation="vertical", style={"height": "24px", "alignSelf": "center"}),
                # Selection tools
                dmc.SegmentedControl(
                    id="within-run-dragmode",
                    data=[
                        {"value": "zoom", "label": "Zoom"},
                        {"value": "select", "label": "Select"},
                    ],
                    value="select",
                    size="xs",
                    style={"flexShrink": 0},
                ),
                dmc.Button(
                    [
                        html.Span("Clear Selected", className="tt-analysis-full-label"),
                        html.Span("Clear", className="tt-analysis-short-label"),
                    ],
                    id="within-run-clear-selection",
                    variant="outline", color="gray", size="xs",
                    className="tt-analysis-shortenable-button",
                    **{"aria-label": "Clear selected trees"},
                ),
                dmc.Button(
                    [
                        html.Span("Reset Zoom", className="tt-analysis-full-label"),
                        html.Span("Reset", className="tt-analysis-short-label"),
                    ],
                    id="within-run-reset-button",
                    variant="outline", color="gray", size="xs",
                    className="tt-analysis-shortenable-button",
                    **{"aria-label": "Reset zoom"},
                ),
                dmc.Tooltip(
                    dmc.Button(
                        html.Span("Export .trees", className="tt-analysis-action-text"),
                        id="within-run-export-trees",
                        variant="filled", color="green", size="xs",
                        disabled=True,
                        leftSection=icon("tabler:download", size=14),
                        className="tt-analysis-collapse-button",
                        classNames={
                            "inner": "tt-analysis-action-inner",
                            "section": "tt-analysis-action-section",
                        },
                        **{"aria-label": "Export selected trees"},
                    ),
                    label="Export selected trees",
                ),
                dmc.Tooltip(
                    dmc.Button(
                        html.Span("View MCC", className="tt-analysis-action-text"),
                        id="within-run-view-mcc",
                        variant="filled", color="violet", size="xs",
                        disabled=True,
                        leftSection=icon("tabler:tree", size=14),
                        className="tt-analysis-collapse-button",
                        classNames={
                            "inner": "tt-analysis-action-inner",
                            "section": "tt-analysis-action-section",
                        },
                        **{"aria-label": "View MCC tree"},
                    ),
                    label="View MCC tree",
                ),
                dmc.Tooltip(
                    dmc.Button(
                        html.Span("Export", className="tt-analysis-action-text"),
                        id="within-run-export-pdf",
                        variant="light", size="xs",
                        leftSection=icon("tabler:pdf", size=20),
                        className="tt-analysis-collapse-button",
                        classNames={
                            "inner": "tt-analysis-action-inner",
                            "section": "tt-analysis-action-section",
                        },
                        **{"aria-label": "Export PDF"},
                    ),
                    label="Export PDF",
                ),
            ], align="center", gap="sm", wrap="nowrap",
               id="within-run-controls-paper",
               className="tt-analysis-controls",
               style={"display": "none"}),

            # Hidden stores
            dcc.Store(id="within-run-selected-trees-store", storage_type="memory"),
            # Drives the clientside ``window.open(/peartree/<uid>)`` callback;
            # populated by the View-MCC handler with {"uuid", "name"}.
            dcc.Store(id="within-run-view-mcc-store", storage_type="memory"),
            dcc.Interval(id="within-run-anim-interval", interval=300, disabled=True),

            # Per-tab MCC registry list (hidden when nothing to show).
            # Rendered by ``render_mcc_list`` in callbacks/within_run.py.
            dmc.Paper(
                html.Div(id="within-run-mcc-list"),
                withBorder=True, p="xs", radius="sm",
                id="within-run-mcc-list-paper",
                style={"display": "none"},
            ),

            # Graph — starts with the same placeholder message as between-run
            # so the empty state is consistent across the two tabs.
            dcc.Graph(
                id="within-run-graph",
                figure=placeholder_fig(
                    "No MDS result selected. Compute an MDS in the Compute tab."
                ),
                style={"height": "calc(100vh - 280px)"},
                config={"doubleClick": False},
            ),
        ], gap="xs"),
    ], style={"padding": "10px", "position": "relative"})
