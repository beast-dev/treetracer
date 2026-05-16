"""Between-run Analysis tab panel: MDS result selector, dim selectors,
selection tools, the 3D scatter + three 2D projections, plus the
per-tab MCC registry table."""

import dash_mantine_components as dmc
from dash import dcc, html

from ...icons import icon
from ...plot_utils import placeholder_fig


def _add_treespace_panel():
    """Build the Between-run Analysis tab panel content."""
    return html.Div([
        dmc.Stack([
            # Row 1: MDS-result selector + dim selectors + info + selection-info
            # badge (boxed). Same shape as the within-run panel.
            # (Note: outer padding + Stack gap="xs" below match _add_within_run_panel
            # so both tabs share identical spacing around the controls.)
            dmc.Paper(
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
                    html.Div(id="treespace-info"),
                    html.Div(id="treespace-selection-info"),
                ], align="flex-end", gap="md", wrap="nowrap"),
                withBorder=True, p="sm", radius="sm", shadow="xs", mt="sm",
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
                ),
                dmc.Button("Clear Selected", id="treespace-clear-selection",
                           variant="outline", color="gray", size="xs"),
                dmc.Button("Reset Zoom", id="treespace-reset-button",
                           variant="outline", color="gray", size="xs"),
                dmc.Button("Export .trees", id="treespace-export-trees",
                           variant="filled", color="green", size="xs",
                           disabled=True,
                           leftSection=icon("tabler:download", size=14)),
                dmc.Button("View MCC", id="treespace-view-mcc",
                           variant="filled", color="violet", size="xs",
                           disabled=True,
                           leftSection=icon("tabler:tree", size=14)),
                dmc.Button("Export PDF", id="treespace-export-pdf",
                           variant="light", size="xs"),
            ], align="center", gap="sm", wrap="nowrap",
                id="treespace-controls-paper",
                style={"display": "none"}),

            dcc.Store(id="treespace-selected-trees-store",
                      storage_type="memory"),
            # Drives the clientside ``window.open(/peartree/<uid>)`` callback;
            # populated by the View-MCC handler with {"uuid", "name"}.
            dcc.Store(id="treespace-view-mcc-store", storage_type="memory"),

            # Per-tab MCC registry list. Hidden when no MCCs match the
            # currently-selected MDS result. Rendered by
            # ``render_mcc_list`` in callbacks/treespace.py.
            dmc.Paper(
                html.Div(id="treespace-mcc-list"),
                withBorder=True, p="xs", radius="sm",
                id="treespace-mcc-list-paper",
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
                style={"height": "calc(100vh - 280px)"},
                config={"doubleClick": False},
            ),
        ], gap="xs"),
    ], style={"padding": "10px"})
