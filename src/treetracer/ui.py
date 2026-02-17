import dash_mantine_components as dmc
from dash import dcc, html

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
                dmc.ActionIcon(
                    dmc.Text(">_", ff="monospace", fw=700, size="sm"),
                    id="log-toggle-button",
                    variant="subtle",
                    size="lg",
                ),
            ],
            h="100%",
            px="md",
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
        radius="sm",  # or p=10 for border-radius of 10px
    )


def add_main_body():
    tabs = dmc.Tabs(
        [
            dmc.TabsList(
                [
                    dmc.TabsTab("Trees", value="trees"),
                    dmc.TabsTab("Compute", value="compute"),
                    dmc.TabsTab("Visualize", value="visualize"),
                    dmc.TabsTab("About", value="about"),
                ],
                grow=True,
                bd="1px solid var(--mantine-color-default-border)",
            ),
            dmc.TabsPanel(
                html.Div([
                    dcc.Loading(
                        html.Div(id="trees-info-display"),
                        type="circle",
                        parent_style={"minHeight": "200px"},
                    ),
                ]),
                value="trees",
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
                                color="green",
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
            dmc.TabsPanel(html.Div(id="plot-display"), value="visualize"),
            dmc.TabsPanel(add_about(), value="about"),
        ],
        id="main-tabs",
        color="blue.2",  # default is blue
        orientation="horizontal",  # or "vertical"
        variant="pills",  # or "outline" or "pills"
        value="about",
        autoContrast=True,
    )
    return dmc.AppShellMain([
        html.Div(id="notifications-container"),
        tabs,
    ])


# Sidebar


clear_data_button = dmc.Button(
    "Clear Data",
    justify="center",
    fullWidth=True,
    variant="filled",
    color="orange",
    id="clear-data-button",
)


def add_navbar():
    return dmc.AppShellNavbar(
        id="navbar",
        children=[
            dmc.Stack(
                [
                    dmc.Button(
                        "Load Trees",
                        id="load-trees-button",
                        justify="center",
                        fullWidth=True,
                        variant="filled",
                        color="green",
                    ),
                    clear_data_button,
                    # dcc.Store components for state management
                    dcc.Store(id="distmat-store", storage_type="memory"),
                    dcc.Store(id="plot-config-store", storage_type="memory"),
                    dcc.Store(id="tree-offset-store", storage_type="memory"),
                    dcc.Store(id="mds-result-store", storage_type="memory"),
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
                ]
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
