"""Main body assembly: tab strip + the Compute Distances tab (small enough
to live inline) + delegates to the per-panel modules for the other three
tabs. Also defines the About modal that lives at the main-body level."""

import dash_mantine_components as dmc
from dash import dcc, html

from ..icons import icon
from .panels.clade_explore import _add_clade_explore_panel
from .panels.diagnostics import _add_diagnostics_panel
from .panels.treespace import _add_treespace_panel
from .panels.within_run import _add_within_run_panel
from .rename_modal import _add_rename_modal


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


def _save_action(id_, label):
    return dmc.Tooltip(
        dmc.Button(
            "Save",
            id=id_,
            color="blue",
            variant="light",
            size="xs",
            leftSection=icon("tabler:download", size=14),
        ),
        label=label,
    )


def _add_export_drawer():
    return dmc.Drawer(
        id="compute-export-drawer",
        title="Save distances",
        position="right",
        size="md",
        padding="md",
        opened=False,
        closeOnEscape=True,
        closeOnClickOutside=True,
        withCloseButton=True,
        children=[
            dmc.Stack(
                [
                    dmc.Stack(
                        [
                            dmc.Group(
                                [
                                    dmc.Text("RF Distance Matrices", fw=600),
                                    dmc.Badge(
                                        "0",
                                        id="rf-matrix-count",
                                        variant="light",
                                        color="gray",
                                        size="sm",
                                    ),
                                ],
                                gap="xs",
                            ),
                            dmc.Select(
                                id="rf-matrix-select",
                                label="Matrix",
                                placeholder="No matrices computed yet",
                                data=[],
                                value=None,
                                size="sm",
                            ),
                            html.Div(id="rf-matrix-info"),
                            dmc.Button(
                                "Export Selected",
                                id="export-rf-button",
                                variant="outline",
                                color="blue",
                                size="sm",
                                disabled=True,
                                leftSection=icon("tabler:download", size=14),
                            ),
                        ],
                        gap="xs",
                    ),
                    dmc.Divider(),
                    dmc.Stack(
                        [
                            dmc.Group(
                                [
                                    dmc.Text("MDS Results", fw=600),
                                    dmc.Badge(
                                        "0",
                                        id="mds-result-count",
                                        variant="light",
                                        color="gray",
                                        size="sm",
                                    ),
                                ],
                                gap="xs",
                            ),
                            dmc.Select(
                                id="mds-result-select",
                                label="Result",
                                placeholder="No MDS results yet",
                                data=[],
                                value=None,
                                size="sm",
                            ),
                            html.Div(id="mds-result-info"),
                            dmc.Button(
                                "Export MDS",
                                id="export-mds-button",
                                variant="outline",
                                color="blue",
                                size="sm",
                                disabled=True,
                                leftSection=icon("tabler:download", size=14),
                            ),
                        ],
                        gap="xs",
                    ),
                ],
                gap="md",
            )
        ],
    )


def add_main_body():
    tabs = dmc.Tabs(
        [
            dmc.TabsList(
                [
                    dmc.TabsTab("Compute Distances", value="compute"),
                    dmc.TabsTab("Between-run Analysis", value="treespace"),
                    dmc.TabsTab("Within-run Analysis", value="within-run"),
                    dmc.TabsTab("Diagnostics", value="diagnostics"),
                    dmc.TabsTab("Compare Consensus", value="clade-explore"),
                ],
                grow=True,
                bd="1px solid var(--mantine-color-default-border)",
                # Pin the tab strip just below the fixed app header so it
                # stays in view while the active panel scrolls under it.
                # ``--app-shell-header-offset`` is the header height
                # Mantine already uses to offset the main content (60px
                # fallback if the var is absent). The opaque body
                # background keeps scrolling content from showing through.
                style={
                    "position": "sticky",
                    "top": "var(--app-shell-header-offset, 60px)",
                    "zIndex": 2,
                    "background": "var(--mantine-color-body)",
                },
            ),
            dmc.TabsPanel(
                html.Div([
                    dmc.Group(
                        [
                            dmc.Group(
                                [
                                    dmc.Title("RF Distances", order=4),
                                    dmc.Badge(
                                        "0",
                                        id="rf-matrix-count-main",
                                        variant="light",
                                        color="gray",
                                        size="sm",
                                    ),
                                ],
                                gap="xs",
                                align="center",
                            ),
                            _save_action("open-rf-export-drawer", "Save RF matrix"),
                        ],
                        justify="space-between",
                        align="center",
                        mb="sm",
                    ),
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
                    dmc.Space(h=10),
                    html.Div(id="compute-rf-output"),
                    # Tree-Space MDS section
                    dmc.Divider(my="lg"),
                    dmc.Group(
                        [
                            dmc.Group(
                                [
                                    dmc.Title("Tree-Space MDS", order=4),
                                    dmc.Badge(
                                        "0",
                                        id="mds-result-count-main",
                                        variant="light",
                                        color="gray",
                                        size="sm",
                                    ),
                                ],
                                gap="xs",
                                align="center",
                            ),
                            _save_action("open-mds-export-drawer", "Save MDS result"),
                        ],
                        justify="space-between",
                        align="center",
                        mb="sm",
                    ),
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
                    dmc.Space(h=10),
                    html.Div(id="compute-mds-output"),
                ], style={"padding": "10px"}),
                value="compute",
            ),
            dmc.TabsPanel(_add_treespace_panel(), value="treespace"),
            dmc.TabsPanel(_add_within_run_panel(), value="within-run"),
            dmc.TabsPanel(_add_diagnostics_panel(), value="diagnostics"),
            dmc.TabsPanel(_add_clade_explore_panel(), value="clade-explore"),
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
        _add_export_drawer(),
        # Shared rename modal — opens on first View, on the pencil
        # icon in consensus tree tables, and right after View consensus tree → compute
        # completes. See ui/rename_modal.py for layout and
        # callbacks/rename_consensus_tree.py for the lifecycle.
        _add_rename_modal(),
    ])
