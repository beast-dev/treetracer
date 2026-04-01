import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify

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
            ],
            h="100%",
            px="md",
            gap="xs",
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


def _add_diagnostics_panel():
    """Build the Diagnostics tab panel content."""
    return html.Div([
        dmc.Stack([
            # Section 1: Log-Likelihood Trace
            dmc.Paper([
                dmc.Group([
                    dmc.Title("Log-Likelihood Trace", order=5),
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

            # Section 2: RF Distance to Reference
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
        ], gap="md"),
    ], style={"padding": "10px"})


def _add_within_run_panel():
    """Build the Within-run Analysis tab panel content (visualization only)."""
    return html.Div([
        dmc.Stack([
            # Row 1: Result selector + axis selectors + info (boxed)
            dmc.Paper(
                dmc.Group([
                    dmc.Select(
                        id="within-run-result-select",
                        placeholder="No results yet",
                        data=[],
                        value=None,
                        size="xs",
                        w=350,
                    ),
                    dmc.Select(label="X", id="within-run-dim-x",
                               data=[], value=None, size="xs", w=100),
                    dmc.Select(label="Y", id="within-run-dim-y",
                               data=[], value=None, size="xs", w=100),
                    dmc.Select(label="Z", id="within-run-dim-z",
                               data=[], value=None, size="xs", w=100),
                    html.Div(id="within-run-info"),
                ], align="flex-end", gap="md"),
                withBorder=True, p="xs", radius="sm",
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
                        {"value": "select", "label": "Box"},
                        {"value": "lasso", "label": "Lasso"},
                    ],
                    value="zoom",
                    size="xs",
                ),
                dmc.Button("Clear", id="within-run-clear-selection",
                           variant="outline", color="gray", size="xs"),
                dmc.Button("Reset Axes", id="within-run-reset-button",
                           variant="outline", color="gray", size="xs"),
                html.Div(id="within-run-selection-info"),
                dmc.Button("Export .trees", id="within-run-export-trees",
                           variant="filled", color="green", size="xs",
                           disabled=True,
                           leftSection=DashIconify(icon="tabler:download", width=14)),
                dmc.Button("Export PDF", id="within-run-export-pdf",
                           variant="light", size="xs"),
            ], align="center", gap="sm", wrap="nowrap",
               id="within-run-controls-paper",
               style={"display": "none"}),

            # Hidden stores
            dcc.Store(id="within-run-selected-trees-store", storage_type="memory"),
            dcc.Interval(id="within-run-anim-interval", interval=500, disabled=True),

            # Graph
            dcc.Graph(figure={}, id="within-run-graph",
                      style={"height": "calc(100vh - 280px)"},
                      config={"doubleClick": False}),
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
                    # RF Distances section
                    dmc.Title("RF Distances", order=4),
                    html.Div(id="compute-trees-table"),
                    dmc.Space(h=10),
                    dmc.Group([
                        dmc.Button(
                            "Compute RF Distances",
                            id="compute-rf-button",
                            variant="filled",
                            color="green",
                            size="md",
                            disabled=True,
                        ),
                        dmc.Group([
                            dmc.Button(
                                "Load RF Matrix",
                                id="load-rf-button",
                                variant="outline",
                                color="green",
                                size="sm",
                            ),
                            dmc.Button(
                                "Export RF Matrix",
                                id="export-rf-button",
                                variant="outline",
                                color="blue",
                                size="sm",
                                disabled=True,
                            ),
                        ], gap="xs"),
                    ], justify="space-between"),
                    dmc.Space(h=10),
                    html.Div(id="compute-rf-output"),
                    # MDS Embedding section
                    dmc.Divider(my="lg"),
                    dmc.Title("MDS Embedding", order=4),
                    dmc.Select(
                        id="mds-distmat-select",
                        label="Distance Matrix",
                        placeholder="No distance matrix available",
                        data=[],
                        value=None,
                        w=400,
                    ),
                    html.Div(id="mds-distmat-info"),
                    html.Div(id="mds-status-text"),
                    dmc.Space(h=10),
                    dmc.Group([
                        dmc.Button(
                            "Compute MDS",
                            id="compute-mds-button",
                            variant="filled",
                            color="blue",
                            size="md",
                            disabled=True,
                        ),
                        dmc.Group([
                            dmc.Button(
                                "Load MDS",
                                id="load-mds-button",
                                variant="outline",
                                color="green",
                                size="sm",
                            ),
                            dmc.Button(
                                "Export MDS",
                                id="export-mds-button",
                                variant="outline",
                                color="blue",
                                size="sm",
                                disabled=True,
                            ),
                        ], gap="xs"),
                    ], justify="space-between"),
                    dmc.Space(h=10),
                    html.Div(id="compute-mds-output"),
                    # Within-run MDS section
                    dmc.Divider(my="lg"),
                    dmc.Title("Within-run MDS", order=4),
                    dmc.Group([
                        dmc.Select(
                            id="wr-mds-distmat-select",
                            label="RF Matrix",
                            placeholder="No distance matrix available",
                            data=[],
                            value=None,
                            w=300,
                        ),
                        dmc.Select(
                            id="wr-mds-run-select",
                            label="Run",
                            placeholder="Select a run",
                            data=[],
                            value=None,
                            w=300,
                        ),
                        dmc.Button(
                            "Compute Within-run MDS",
                            id="compute-wr-mds-button",
                            variant="filled",
                            color="violet",
                            size="md",
                            disabled=True,
                            style={"alignSelf": "flex-end"},
                        ),
                    ], align="flex-end", gap="lg"),
                    html.Div(id="wr-mds-info"),
                    dmc.Space(h=10),
                    html.Div(id="compute-wr-mds-output"),
                ], style={"padding": "10px"}),
                value="compute",
            ),
            dmc.TabsPanel(html.Div(id="plot-display"), value="treespace"),
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
                    dcc.Store(id="within-run-mds-results-store", storage_type="memory"),
                    dcc.Store(id="within-run-treenum-range-store", storage_type="memory"),
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
