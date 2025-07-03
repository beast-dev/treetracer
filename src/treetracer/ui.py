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
                    dmc.TabsTab("Data", value="data"),
                    dmc.TabsTab("Traces", value="traces"),
                    dmc.TabsTab("About", value="about"),
                ],
                grow=True,
                bd="1px solid var(--mantine-color-default-border)",
            ),
            dmc.TabsPanel(html.Div(id="data-info-display"), value="data"),
            dmc.TabsPanel(html.Div(id="plot-display"), value="traces"),
            dmc.TabsPanel(add_about(), value="about"),
        ],
        color="blue.2",  # default is blue
        orientation="horizontal",  # or "vertical"
        variant="pills",  # or "outline" or "pills"
        value="about",
        autoContrast=True,
    )
    return dmc.AppShellMain(tabs)


# Sidebar


upload_button = dcc.Upload(
    id="upload-data-button",
    children=html.Div(
        [
            dmc.Button(
                "Upload Tree Traces",
                justify="center",
                fullWidth=True,
            )
        ]
    ),
    multiple=True,
)


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
                    upload_button,
                    clear_data_button,
                    html.Div(id="upload-placeholder"),
                    # dcc.Store components for state management
                    dcc.Store(id="uploaded-files-store", storage_type="memory"),
                    dcc.Store(id="dataframes-store", storage_type="memory"),
                    dcc.Store(id="plot-config-store", storage_type="memory"),
                    # Add a div to display validation messages
                    dmc.Alert(
                        id="validation-alert",
                        title="",
                        color="red",
                        withCloseButton=True,
                        style={"display": "none"},
                    ),
                    # Display selected file info
                    html.Div(id="file-info-display"),
                ]
            ),
        ],
        p="md",
    )
