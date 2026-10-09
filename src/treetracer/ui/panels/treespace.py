"""Between-run Analysis tab panel: MDS result selector, dim selectors,
selection tools, the 3D scatter + three 2D projections, plus the
per-tab consensus tree registry table."""

import dash_mantine_components as dmc
from dash import dcc, html

from ...icons import icon
from ...plot_utils import placeholder_fig
from ..widgets import stop_button


def _add_treespace_panel():
    """Build the Between-run Analysis tab panel content."""
    return html.Div([
        # consensus-tree-compute loading overlay. Visible=True flipped on by
        # ``view_consensus_tree`` (click handler), back to False by
        # consensus terminal presentation adapter. Position relative on the
        # wrapping Div lets the overlay sit on top.
        dmc.LoadingOverlay(
            id="treespace-loading-overlay",
            visible=False,
            zIndex=1000,
            overlayProps={"radius": "sm", "blur": 2},
            loaderProps={"size": "lg", "type": "dots",
                         "children": dmc.Stack(
                             [dmc.Text("Computing summary tree…",
                                       size="sm", c="dimmed"),
                              stop_button("consensus-treespace")],
                             align="center", gap="xs", mt="sm")},
        ),
        dmc.Stack([
            # Row 1: MDS-result selector + dim selectors + info + selection-info
            # badge (boxed). Same shape as the within-run panel.
            # (Note: outer padding + Stack gap="xs" below match _add_within_run_panel
            # so both tabs share identical spacing around the controls.)
            dmc.Paper(
                dmc.Stack(
                    [
                        dmc.Group([
                            html.Div(
                                dmc.Select(
                                    label="MDS Result",
                                    id="treespace-result-select",
                                    placeholder="No MDS results yet",
                                    data=[], value=None, size="xs",
                                    style={"width": "100%"},
                                ),
                                style={"width": "300px", "flexShrink": 0, "flexGrow": 0},
                            ),
                            html.Div(
                                dmc.Select(label="X", id="dim-x-select",
                                           data=[], value=None, size="xs",
                                           style={"width": "100%"}),
                                style={"width": "100px", "flexShrink": 0, "flexGrow": 0},
                            ),
                            html.Div(
                                dmc.Select(label="Y", id="dim-y-select",
                                           data=[], value=None, size="xs",
                                           style={"width": "100%"}),
                                style={"width": "100px", "flexShrink": 0, "flexGrow": 0},
                            ),
                            html.Div(
                                dmc.Select(label="Z", id="dim-z-select",
                                           data=[], value=None, size="xs",
                                           style={"width": "100%"}),
                                style={"width": "100px", "flexShrink": 0, "flexGrow": 0},
                            ),
                        ], align="flex-end", gap="md", wrap="nowrap"),
                        dmc.Group(
                            [
                                html.Div(id="treespace-info"),
                                html.Div(id="treespace-selection-info"),
                            ],
                            align="center",
                            gap="sm",
                        ),
                    ],
                    gap="xs",
                ),
                withBorder=True, p="sm", radius="sm", shadow="xs",
                style={"width": "100%"},
            ),

            # Row 2: all controls in a single horizontal row (hidden until a
            # result is selected, no border) — mirrors the within-run layout.
            dmc.Group([
                dmc.Text("Trees:", size="xs", fw=500, style={"alignSelf": "center"}),
                html.Div([
                    dmc.RangeSlider(
                        id="treenum-slider",
                        min=1, max=100, value=[1, 100],
                        marks=[
                            {"value": 1, "label": "1"},
                            {"value": 100, "label": "100"},
                        ],
                        step=1, size="xs",
                        styles={"markLabel": {"fontSize": "10px"}},
                    ),
                ], style={"width": "420px", "alignSelf": "center"}),
                dmc.Checkbox(label="Lines", id="show-lines-checkbox",
                             checked=True, size="xs"),
                # No "Plot" button — the multiplot auto-rebuilds on
                # any dim/slider/lines change via ``auto_update_graph``
                # in callbacks/treespace.py. Matches the within-run tab's
                # always-live UX.
                dmc.Divider(orientation="vertical",
                            style={"height": "24px", "alignSelf": "center"}),
                # Selection tools
                dmc.SegmentedControl(
                    id="treespace-dragmode",
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
                    id="treespace-clear-selection",
                    variant="outline", color="gray", size="xs",
                    className="tt-analysis-shortenable-button",
                    **{"aria-label": "Clear selected trees"},
                ),
                dmc.Button(
                    [
                        html.Span("Reset Zoom", className="tt-analysis-full-label"),
                        html.Span("Reset", className="tt-analysis-short-label"),
                    ],
                    id="treespace-reset-button",
                    variant="outline", color="gray", size="xs",
                    className="tt-analysis-shortenable-button",
                    **{"aria-label": "Reset zoom"},
                ),
                dmc.Tooltip(
                    dmc.Button(
                        html.Span("Export .trees", className="tt-analysis-action-text"),
                        id="treespace-export-trees",
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
                        html.Span("View summary tree", className="tt-analysis-action-text"),
                        id="treespace-view-consensus-tree",
                        variant="filled", color="violet", size="xs",
                        disabled=True,
                        leftSection=icon("tabler:tree", size=14),
                        className="tt-analysis-collapse-button",
                        classNames={
                            "inner": "tt-analysis-action-inner",
                            "section": "tt-analysis-action-section",
                        },
                        **{"aria-label": "View summary tree"},
                    ),
                    label="View summary tree",
                ),
                dmc.Tooltip(
                    dmc.Button(
                        html.Span("Export", className="tt-analysis-action-text"),
                        id="treespace-export-pdf",
                        variant="light", size="xs",
                        leftSection=icon("tabler:download", size=20),
                        className="tt-analysis-collapse-button",
                        classNames={
                            "inner": "tt-analysis-action-inner",
                            "section": "tt-analysis-action-section",
                        },
                        **{"aria-label": "Export SVG"},
                    ),
                    label="Export SVG",
                ),
            ], align="center", gap="sm", wrap="nowrap",
                id="treespace-controls-paper",
                className="tt-analysis-controls",
                style={"display": "none"}),

            dcc.Store(id="treespace-selected-trees-store",
                      storage_type="memory"),
            # Drives the clientside ``window.open(/peartree/<uid>)`` callback;
            # populated by the View-consensus-tree handler with {"uuid", "name"}.
            dcc.Store(id="treespace-view-consensus-tree-store", storage_type="memory"),

            # Per-tab consensus tree registry list. Hidden when no consensus trees match the
            # currently-selected MDS result. Rendered by
            # ``render_consensus_tree_list`` in callbacks/treespace.py.
            dmc.Paper(
                html.Div(id="treespace-consensus-tree-list"),
                withBorder=True, p="xs", radius="sm",
                id="treespace-consensus-tree-list-paper",
                style={"display": "none"},
            ),

            # Plot canvas — statically defined so the selection callbacks can
            # always target it. Starts empty with a centered placeholder
            # message; configure_for_selected_result and the Plot button
            # handler drive the figure content from there.
            dcc.Graph(
                id="graph",
                figure=placeholder_fig(
                    "No MDS result selected. Compute an MDS in the Compute tab."
                ),
                style={"height": "calc(100vh - 320px)"},
                # Hide Plotly's modebar — its toolbar overlaps the legend
                # when many files are loaded, and the zoom/pan/export
                # interactions are already exposed via the tab's buttons.
                # ``responsive`` relayouts once the container reaches its real
                # size — without it, the first paint lays out the figure at a
                # transitional size and the top legend (wrapped wide by long
                # file names) spills over the plot until the user resizes.
                config={"doubleClick": False, "displayModeBar": False,
                        "responsive": True},
            ),
        ], gap="xs"),
    ], style={"padding": "10px", "position": "relative"})
