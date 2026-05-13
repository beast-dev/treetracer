import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify
from .theme import get_template

from .plot_utils import placeholder_fig

# Header
def add_header():
    logo_path = "assets/treetracer-icon.png"
    return dmc.AppShellHeader(
        dmc.Group(
            [
                dmc.Tooltip(
                    dmc.ActionIcon(
                        DashIconify(icon="tabler:layout-sidebar-left-collapse", width=28),
                        id="sidebar-toggle",
                        variant="subtle",
                        size="xl",
                    ),
                    label="Toggle Sidebar",
                ),
                dmc.Image(src=logo_path, w=50, fit="contain"),
                dmc.Title("TreeTracer", c="blue"),
                dmc.Space(style={"flex": 1}),
                dmc.Tooltip(
                    dmc.ActionIcon(
                        DashIconify(icon="tabler:info-circle", width=20),
                        id="about-modal-button",
                        variant="subtle",
                        size="lg",
                    ),
                    label="About",
                ),
                dmc.Tooltip(
                    dmc.ActionIcon(
                        DashIconify(icon="tabler:terminal-2", width=20),
                        id="log-toggle-button",
                        variant="subtle",
                        size="lg",
                    ),
                    label="Toggle Log",
                ),
                dmc.Tooltip(
                    dmc.ActionIcon(
                        DashIconify(icon="tabler:moon", width=20, id="dark-mode-icon"),
                        id="dark-mode-toggle",
                        variant="subtle",
                        size="lg",
                    ),
                    label="Toggle dark mode",
                ),
            ],
            h="100%",
            px="md",
            gap="xs",
            wrap="nowrap",
        )
    )


# Main Body


def _add_about_modal():
    return dmc.Modal(
        id="about-modal",
        title="About TreeTracer",
        centered=True,
        children=[
            dmc.Stack(
                [
                    dmc.Title("TreeTracer v0.0-DEMO", order=3),
                    dmc.Text(
                        "TreeTracer is a diagnostic tool used to visualize convergence of tree topologies"
                    ),
                    dmc.Title("Citation", order=4),
                    dmc.Text("To cite TreeTracer, please use the following citation:"),
                    dmc.Blockquote(
                        children=[dmc.Text("Treetracer Full citation", fs="italic")]
                    ),
                ],
            )
        ],
    )

def _add_clade_freq_panel():
    """Build the Clade Frequency Comparison section for the Diagnostics tab.
 
    This is a self-contained helper so other branches can import or reuse it
    without touching _add_diagnostics_panel directly.
 
    Layout
    ------
    A single dmc.Paper containing:
      - A title row with a badge explaining the prerequisite.
      - Two Select dropdowns side by side (Group 1 / Group 2), each listing
        every MCC tree computed this session via the mcc-registry-store.
      - A "Compare" button, enabled only when both dropdowns have a value.
      - A hint line explaining where MCC trees come from.
      - Two Div placeholders:
          "clade-freq-plot"    — receives the frequency scatter plot
          "clade-freq-tanglegram" — receives the tanglegram
    """
    return dmc.Paper([
        # ── Header row ──────────────────────────────────────────────────────
        dmc.Group([
            dmc.Title("Clade Frequency Comparison", order=5),
            dmc.Badge(
                "requires two saved MCC trees",
                variant="light", size="sm",
            ),
        ], gap="sm", align="center"),
 
        dmc.Space(h=10),
 
        # ── Controls row ────────────────────────────────────────────────────
        dmc.Group([
            dmc.Select(
                id="clade-freq-mcc-select-1",
                label="Group 1 (MCC tree)",
                placeholder="No MCC trees saved yet",
                data=[],
                value=None,
                disabled=True,
                w=280,
                size="sm",
            ),
            dmc.Select(
                id="clade-freq-mcc-select-2",
                label="Group 2 (MCC tree)",
                placeholder="No MCC trees saved yet",
                data=[],
                value=None,
                disabled=True,
                w=280,
                size="sm",
            ),
            dmc.Button(
                "Compare Clade Frequencies",
                id="clade-freq-compare-button",
                variant="filled",
                color="green",
                size="sm",
                disabled=True,
                style={"alignSelf": "flex-end"},
            ),
        ], gap="md", align="flex-end"),
 
        dmc.Text(
            "Save MCC trees by selecting trees in the Within-run or "
            "Between-run Analysis tabs and clicking 'View MCC'.",
            size="xs", c="dimmed", mt=4,
        ),
 
        dmc.Space(h=10),
 
        # ── Output placeholders ─────────────────────────────────────────────
        dcc.Loading(
            html.Div(id="clade-freq-plot"),
            type="circle",
            parent_style={"minHeight": "200px"},
        ),
        dmc.Stack([
            dmc.Text("Expand tree", size="xs", c="dimmed"),
            dmc.Slider(
                id="tanglegram-yscale-slider",
                min=1, max=30, step=1, value=1,
                w=300,
                marks=[
                    #{"value": 4,  "label": "4"},
                    #{"value": 10, "label": "10"},
                    #{"value": 20, "label": "20"},
                    #{"value": 30, "label": "30"},
                ],
            ),
        ], gap=4, mt=8, mb=4),
        dmc.Stack([
            dmc.Text("Minimum clade size", size="xs", c="dimmed"),
            dmc.Slider(
                id="clade-freq-min-clade-size",
                min=2, max=50, step=1, value=5,
                w=300,
                marks=[
                    {"value": 2,  "label": "2"},
                    {"value": 10, "label": "10"},
                    {"value": 25, "label": "25"},
                    {"value": 50, "label": "50"},
                ],
            ),
        ], gap=4, mt=8, mb=4),
        dcc.Loading(
            html.Div(id="clade-freq-tanglegram"),
            type="circle",
            parent_style={"minHeight": "200px"},
        ),
 
    ], p="md", withBorder=True, radius="sm")
 
 
def _add_diagnostics_panel():
    """Build the Diagnostics tab panel content."""
    return html.Div([
        dmc.Stack([
            # Header: shared RF Matrix selector that conditions every
            # downstream section (LnL trace, RF-to-reference, Pseudo-ESS).
            dmc.Paper([
                dmc.Group([
                    dmc.Title("Diagnostics", order=4),
                    dmc.Select(
                        id="diagnostics-distmat-select",
                        label="RF Matrix",
                        placeholder="No distance matrix available",
                        data=[],
                        value=None,
                        size="xs",
                        w=300,
                    ),
                    html.Div(id="diagnostics-distmat-info"),
                ], align="flex-end", gap="md"),
            ], p="md", withBorder=True, radius="sm"),

            # Section 1: Log-Likelihood Trace
            dmc.Paper([
                dmc.Group([
                    dmc.Title("Log-Posterior Trace", order=5),
                    dmc.Badge("from tree metadata", variant="light", size="sm"),
                    dmc.NumberInput(
                        id="lnl-burnin-input",
                        label="Burnin",
                        value=0,
                        min=0,
                        step=100,
                        size="xs",
                        w=120,
                    ),
                    dmc.Button("Export PDF", id="export-lnl-trace-button", variant="light",
                               size="xs", disabled=True),
                ], gap="sm", align="center"),
                dmc.Space(h=10),
                dcc.Loading(
                    html.Div(id="lnl-trace-plot"),
                    type="circle",
                    parent_style={"minHeight": "200px"},
                ),
            ], p="md", withBorder=True, radius="sm"),
 
            # Section 2: RF Distance to Reference  (unchanged)
            dmc.Paper([
                dmc.Group([
                    dmc.Title("RF Distance to Reference", order=5),
                    dmc.Select(
                        id="rf-reference-group-select",
                        placeholder="Reference group",
                        data=[],
                        value=None,
                        w=200,
                    ),
                    dmc.Select(
                        id="rf-reference-position-select",
                        placeholder="Position",
                        data=[
                            {"value": "last", "label": "Last tree"},
                            {"value": "first", "label": "First tree"},
                        ],
                        value="last",
                        w=150,
                    ),
                    dmc.Button(
                        "Compute RF Trace",
                        id="compute-rf-trace-button",
                        variant="filled",
                        color="green",
                        size="sm",
                        disabled=True,
                    ),
                    dmc.NumberInput(
                        id="rf-burnin-input",
                        label="Burnin",
                        value=0,
                        min=0,
                        step=100,
                        size="xs",
                        w=120,
                    ),
                    dmc.Button("Export PDF", id="export-rf-trace-button", variant="light",
                               size="xs", disabled=True),
                ], align="center", gap="md"),
                dmc.Space(h=10),
                dcc.Loading(
                    html.Div(id="rf-trace-plot"),
                    type="circle",
                    parent_style={"minHeight": "200px"},
                ),
            ], p="md", withBorder=True, radius="sm"),

            # Section 3: Pseudo-ESS — picks the runs within the RF
            # matrix selected at the top of the tab and (eventually)
            # computes Lanfear-style pseudo-ESS for each.
            dmc.Paper([
                dmc.Group([
                    dmc.Title("Pseudo-ESS", order=5),
                    dmc.NumberInput(
                        id="ess-burnin-input",
                        label="Burn-in (trees)",
                        value=0,
                        min=0,
                        step=100,
                        size="xs",
                        w=140,
                    ),
                    dmc.NumberInput(
                        id="ess-n-refs-input",
                        label="# Reference trees",
                        value=100,
                        min=10,
                        step=10,
                        size="xs",
                        w=140,
                    ),
                    dmc.Button(
                        "Compute Pseudo-ESS",
                        id="compute-pseudo-ess-button",
                        variant="filled",
                        color="green",
                        size="sm",
                        disabled=True,
                    ),
                ], align="flex-end", gap="md"),
                dmc.Space(h=10),
                # Per-run table: one row per group inside the selected
                # matrix, each row checkable. Populated by the
                # ``render_ess_runs_table`` callback.
                html.Div(id="ess-runs-table"),
                dmc.Space(h=10),
                # Result area — filled in by the compute callback once
                # we wire the math up. Empty for now.
                html.Div(id="pseudo-ess-output"),
            ], p="md", withBorder=True, radius="sm"),

            # Per-matrix MCC registry summary. Hidden when no MCCs have
            # been computed for the currently-selected matrix; otherwise
            # split into Between-runs and Within-run tables. Driven by
            # ``render_diagnostics_mcc_panel`` in callbacks/diagnostics.py.
            dmc.Paper(
                html.Div(id="diagnostics-mcc-list"),
                p="md", withBorder=True, radius="sm",
                id="diagnostics-mcc-paper",
                style={"display": "none"},
            ),

            # Clade frequency comparison panel (local feature).
            _add_clade_freq_panel(),
        ], gap="md"),
    ], style={"padding": "10px"})

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
                dmc.Button("Plot", id="plot-button",
                           variant="filled", color="blue", size="xs"),
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
                           leftSection=DashIconify(icon="tabler:download", width=14)),
                dmc.Button("View MCC", id="treespace-view-mcc",
                           variant="filled", color="violet", size="xs",
                           disabled=True,
                           leftSection=DashIconify(icon="tabler:tree", width=14)),
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


def _add_within_run_panel():
    """Build the Within-run Analysis tab panel content (visualization only)."""
    return html.Div([
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
                    DashIconify(icon="tabler:player-play-filled", width=20),
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
                ),
                dmc.Button("Clear Selected", id="within-run-clear-selection",
                           variant="outline", color="gray", size="xs"),
                dmc.Button("Reset Zoom", id="within-run-reset-button",
                           variant="outline", color="gray", size="xs"),
                dmc.Button("Export .trees", id="within-run-export-trees",
                           variant="filled", color="green", size="xs",
                           disabled=True,
                           leftSection=DashIconify(icon="tabler:download", width=14)),
                dmc.Button("View MCC", id="within-run-view-mcc",
                           variant="filled", color="violet", size="xs",
                           disabled=True,
                           leftSection=DashIconify(icon="tabler:tree", width=14)),
                dmc.Button("Export PDF", id="within-run-export-pdf",
                           variant="light", size="xs"),
            ], align="center", gap="sm", wrap="nowrap",
               id="within-run-controls-paper",
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
    ], style={"padding": "10px"})


def add_main_body():
    tabs = dmc.Tabs(
        [
            dmc.TabsList(
                [
                    dmc.TabsTab("Compute Distances", value="compute"),
                    dmc.TabsTab("Between-run Analysis", value="treespace"),
                    dmc.TabsTab("Within-run Analysis", value="within-run"),
                    dmc.TabsTab("Diagnostics", value="diagnostics"),
                ],
                grow=True,
                bd="1px solid var(--mantine-color-default-border)",
            ),
            dmc.TabsPanel(
                html.Div([
                    # RF Distances section — two columns
                    dmc.Title("RF Distances", order=4),
                    dmc.Grid([
                        # Left column: file selection + compute
                        dmc.GridCol([
                            html.Div(id="compute-trees-table"),
                            dmc.Space(h=10),
                            dmc.Button(
                                "Compute RF Distances",
                                id="compute-rf-button",
                                variant="filled",
                                color="green",
                                size="sm",
                                disabled=True,
                            ),
                        ], span=6),
                        # Right column: computed matrices
                        dmc.GridCol([
                            dmc.Group([
                                dmc.Text("Computed Matrices", fw=600, size="sm"),
                                dmc.Badge("0", id="rf-matrix-count", variant="light",
                                          color="gray", size="sm"),
                            ], gap="xs", mb="xs"),
                            dmc.Select(
                                id="rf-matrix-select",
                                placeholder="No matrices computed yet",
                                data=[],
                                value=None,
                                size="sm",
                            ),
                            html.Div(id="rf-matrix-info", style={"marginTop": "6px"}),
                            dmc.Space(h=10),
                            dmc.Button(
                                "Export Selected",
                                id="export-rf-button",
                                variant="outline",
                                color="blue",
                                size="sm",
                                disabled=True,
                                leftSection=DashIconify(icon="tabler:download", width=14),
                            ),
                        ], span=6),
                    ], gutter="lg"),
                    dmc.Space(h=10),
                    html.Div(id="compute-rf-output"),
                    # Tree-Space MDS section
                    dmc.Divider(my="lg"),
                    dmc.Title("Tree-Space MDS", order=4),
                    dmc.Grid([
                        # Left: compute controls
                        dmc.GridCol([
                            dmc.Select(
                                id="mds-distmat-select",
                                label="RF Matrix",
                                placeholder="No distance matrix available",
                                data=[],
                                value=None,
                                size="sm",
                            ),
                            html.Div(id="mds-distmat-info", style={"marginTop": "6px"}),
                            html.Div(id="mds-status-text"),
                            dmc.Space(h=10),
                            dmc.Button(
                                "Compute MDS",
                                id="compute-mds-button",
                                variant="filled",
                                color="blue",
                                size="sm",
                                disabled=True,
                            ),
                        ], span=6),
                        # Right: computed MDS results
                        dmc.GridCol([
                            dmc.Group([
                                dmc.Text("Computed MDS", fw=600, size="sm"),
                                dmc.Badge("0", id="mds-result-count", variant="light",
                                          color="gray", size="sm"),
                            ], gap="xs", mb="xs"),
                            dmc.Select(
                                id="mds-result-select",
                                placeholder="No MDS results yet",
                                data=[],
                                value=None,
                                size="sm",
                            ),
                            html.Div(id="mds-result-info", style={"marginTop": "6px"}),
                            dmc.Space(h=10),
                            dmc.Button(
                                "Export MDS",
                                id="export-mds-button",
                                variant="outline",
                                color="blue",
                                size="sm",
                                disabled=True,
                                leftSection=DashIconify(icon="tabler:download", width=14),
                            ),
                        ], span=6),
                    ], gutter="lg"),
                    dmc.Space(h=10),
                    html.Div(id="compute-mds-output"),
                ], style={"padding": "10px"}),
                value="compute",
            ),
            dmc.TabsPanel(_add_treespace_panel(), value="treespace"),
            dmc.TabsPanel(_add_within_run_panel(), value="within-run"),
            dmc.TabsPanel(_add_diagnostics_panel(), value="diagnostics"),
        ],
        id="main-tabs",
        color="blue.2",
        orientation="horizontal",
        variant="pills",
        value="compute",
        autoContrast=True,
    )
    return dmc.AppShellMain([
        html.Div(id="notifications-container"),
        tabs,
        _add_about_modal(),
    ])


# Sidebar


def add_navbar():
    return dmc.AppShellNavbar(
        id="navbar",
        children=[
            dmc.Stack(
                [
                    dmc.Group([
                        dmc.Text("Loaded Trees", fw=600, size="sm"),
                        dmc.Group([
                            dmc.Tooltip(
                                dmc.ActionIcon(
                                    DashIconify(icon="tabler:file-upload", width=18),
                                    id="load-trees-button",
                                    variant="light",
                                    color="green",
                                    size=40,
                                ),
                                label="Load Trees",
                            ),
                            dmc.Tooltip(
                                dmc.ActionIcon(
                                    DashIconify(icon="tabler:trash", width=18),
                                    id="clear-data-button",
                                    variant="light",
                                    color="orange",
                                    size=40,
                                ),
                                label="Clear Data",
                            ),
                        ], gap=4),
                    ], justify="space-between"),
                    dmc.Divider(),
                    html.Div([
                        dmc.LoadingOverlay(
                            id="sidebar-loading-overlay",
                            visible=False,
                            overlayProps={"radius": "sm", "blur": 2},
                        ),
                        dmc.ScrollArea(
                            html.Div(
                                id="sidebar-trees-display",
                                children=[
                                    dmc.Text(
                                        "No trees loaded. Click the upload button above to load a .trees file.",
                                        c="dimmed",
                                        size="sm",
                                        style={"padding": "10px"},
                                    ),
                                ],
                            ),
                            style={"height": "calc(100vh - 160px)"},
                            offsetScrollbars=True,
                        ),
                    ], style={"position": "relative"}),
                    # dcc.Store components for state management
                    dcc.Store(id="distmat-store", storage_type="memory"),
                    dcc.Store(id="plot-config-store", storage_type="memory"),
                    dcc.Store(id="tree-offset-store", storage_type="memory"),
                    dcc.Store(id="mds-result-store", storage_type="memory"),
                    dcc.Store(id="rf-trace-store", storage_type="memory"),
                    dcc.Store(id="within-run-treenum-range-store", storage_type="memory"),
                    # Persistent registry of computed MCC trees
                    # (RF_001_Between_MCC_1 etc.) and a one-shot action
                    # signal driven by the per-row View / Delete buttons
                    # in the MCC list panels.
                    dcc.Store(id="mcc-registry-store", storage_type="memory", data=[]),
                    dcc.Store(id="mcc-registry-action-store", storage_type="memory"),
                    # Current plotly template name (light/dark). Each
                    # plot-rendering callback reads it via
                    # ``theme.get_template()`` at fig build time; the
                    # store is wired as a re-render trigger.
                    dcc.Store(id="plotly-template-store", storage_type="memory", data=get_template()),
                    # Clade frequency comparison — intermediate results and
                    # scatter-click state, kept server-side-friendly.
                    dcc.Store(id="clade-freq-data-store", storage_type="memory"),
                    dcc.Store(id="clade-freq-click-store", storage_type="memory"),
                    # Background computation polling
                    dcc.Interval(id="compute-poll-interval", interval=100, disabled=True),
                    # Log panel state
                    dcc.Store(id="log-panel-visible", storage_type="memory", data=False),
                    dcc.Store(id="sidebar-visible", storage_type="memory", data=True),
                    dcc.Interval(id="log-poll-interval", interval=2000, n_intervals=0),
                    dmc.Alert(
                        id="trees-validation-alert",
                        title="",
                        color="red",
                        withCloseButton=True,
                        style={"display": "none"},
                    ),
                ],
                gap="xs",
            ),
        ],
        p="md",
    )


# Footer (log panel)


def add_footer():
    """Create a toggleable log/terminal output panel."""
    return dmc.AppShellFooter(
        id="log-footer",
        style={"display": "none"},
        children=[
            dmc.Stack(
                [
                    dmc.Group(
                        [
                            dmc.Text("Log Output", fw=600, size="sm"),
                            dmc.Button(
                                "Clear",
                                id="log-clear-button",
                                variant="subtle",
                                size="compact-sm",
                            ),
                        ],
                        justify="space-between",
                        px="sm",
                        py="4px",
                        style={
                            "borderBottom": "1px solid var(--mantine-color-default-border)",
                        },
                    ),
                    dmc.ScrollArea(
                        id="log-scroll-area",
                        h=200,
                        children=[
                            html.Div(
                                id="log-content",
                                style={
                                    "fontFamily": "monospace",
                                    "fontSize": "12px",
                                    "padding": "8px",
                                    "backgroundColor": "#1a1b1e",
                                    "color": "#c9d1d9",
                                    "minHeight": "200px",
                                },
                            )
                        ],
                        offsetScrollbars=True,
                    ),
                ],
                gap=0,
            )
        ],
    )
