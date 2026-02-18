import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify

# Header


def add_header():
    logo_path = "assets/treetracer-icon.png"
    return dmc.AppShellHeader(
        dmc.Group(
            [
                dmc.Burger(
                    id="burger",
                    size="sm",
                    hiddenFrom="sm",
                    opened=False,
                ),
                dmc.Image(src=logo_path, w=50, fit="contain"),
                dmc.Title("TreeTracer", c="blue"),
                dmc.Space(style={"flex": 1}),
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


def add_about():
    return dmc.Paper(
        children=[
            dmc.Stack(
                [
                    dmc.Text(""),
                    dmc.Title("TreeTracer v0.0-DEMO", order=1),
                    dmc.Title("About TreeTracer", order=4),
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
        radius="sm",
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
                    dmc.Button("Export PDF", id="export-lnl-trace-button", variant="light",
                               size="xs", disabled=True),
                ], gap="sm"),
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


def add_main_body():
    tabs = dmc.Tabs(
        [
            dmc.TabsList(
                [
                    dmc.TabsTab("Compute Distances", value="compute"),
                    dmc.TabsTab("Tree Space", value="treespace"),
                    dmc.TabsTab("Diagnostics", value="diagnostics"),
                    dmc.TabsTab("About", value="about"),
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
                    dcc.Loading(
                        html.Div(id="compute-rf-output"),
                        type="circle",
                        parent_style={"minHeight": "50px"},
                    ),
                    # MDS Embedding section
                    dmc.Divider(my="lg"),
                    dmc.Title("MDS Embedding", order=4),
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
                    dcc.Loading(
                        html.Div(id="compute-mds-output"),
                        type="circle",
                        parent_style={"minHeight": "50px"},
                    ),
                ], style={"padding": "10px"}),
                value="compute",
            ),
            dmc.TabsPanel(html.Div(id="plot-display"), value="treespace"),
            dmc.TabsPanel(_add_diagnostics_panel(), value="diagnostics"),
            dmc.TabsPanel(add_about(), value="about"),
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
                    # Log panel state
                    dcc.Store(id="log-panel-visible", storage_type="memory", data=False),
                    dcc.Interval(id="log-poll-interval", interval=500, n_intervals=0),
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
