"""Left-hand navbar (sidebar): loaded-trees panel + all of the dcc.Stores
and Intervals that act as the app's client-side state bus."""

import dash_mantine_components as dmc
from dash import dcc, html

from ..icons import icon
from ..theme import get_template


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
                                    icon("tabler:file-upload", size=18),
                                    id="load-trees-button",
                                    variant="light",
                                    color="green",
                                    size=40,
                                ),
                                label="Load Trees (max 10 files)",
                            ),
                            dmc.Tooltip(
                                dmc.ActionIcon(
                                    icon("tabler:trash", size=18),
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
                    # Persistent registry of computed consensus trees
                    # (RF_001_Between_consensus_tree_1 etc.) and a one-shot action
                    # signal driven by the per-row View / Delete buttons
                    # in the consensus tree list panels.
                    dcc.Store(id="consensus-tree-registry-store", storage_type="memory", data=[]),
                    dcc.Store(id="consensus-tree-registry-action-store", storage_type="memory"),
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
                    # Per-workflow job identities and shared two-phase terminal
                    # delivery. The poll that renders a terminal result writes
                    # an applied marker; only then does the acknowledgement
                    # callback release the server-side sticky event.
                    dcc.Store(id="rf-job-store", storage_type="memory"),
                    dcc.Store(id="mds-job-store", storage_type="memory"),
                    dcc.Store(id="pseudo-ess-job-store", storage_type="memory"),
                    dcc.Store(id="consensus-job-store", storage_type="memory"),
                    dcc.Store(id="rf-trace-job-store", storage_type="memory"),
                    dcc.Store(id="clade-freq-job-store", storage_type="memory"),
                    dcc.Store(id="compute-applied-job-store", storage_type="memory"),
                    dcc.Store(id="compute-job-ack-store", storage_type="memory"),
                    # Resolves scatter split IDs through the matching server-side
                    # managed comparison result; avoids shipping tip sets through
                    # browser JSON or decoding the full snapshot on click.
                    dcc.Store(id="clade-freq-result-key-store", storage_type="memory"),
                    # Consensus trees keep a dedicated cadence because their
                    # overlay and button lifecycle can stop independently of
                    # the shared RF/MDS/Pseudo-ESS interval.
                    dcc.Interval(id="consensus-tree-poll-interval", interval=100, disabled=True),
                    # Path to the RF worker's sidecar progress file
                    # (``<save_path>.progress``). Set by
                    # ``handle_compute_rf`` when an RF compute starts;
                    # consumed by ``update_rf_progress`` to drive the
                    # progress bar inside the computing banner.
                    dcc.Store(id="rf-progress-path", storage_type="memory"),
                    # Same sidecar-progress pattern for MDS/PCoA. This
                    # is phase progress: centering/eigensolve/finalizing,
                    # not ARPACK iteration-level progress.
                    dcc.Store(id="mds-progress-path", storage_type="memory"),
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
