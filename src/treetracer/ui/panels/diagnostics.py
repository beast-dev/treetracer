"""Diagnostics tab panel: log-posterior trace, RF-to-reference trace,
and Pseudo-ESS — the MCMC-convergence diagnostics.

The MCC clade-frequency comparison that used to live here now has its
own tab — see ui/panels/clade_explore.py."""

import dash_mantine_components as dmc
from dash import dcc, html

from ...icons import icon


def _add_diagnostics_panel():
    """Build the Diagnostics tab panel content."""
    return html.Div([
        dmc.Stack([
            # Header: shared RF Matrix selector that conditions every
            # downstream section (LnL trace, RF-to-reference, Pseudo-ESS).
            dmc.Paper([
                dmc.Group([
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

            # Placeholder shown until an RF matrix is selected. The
            # diagnostic sections below (Log-Posterior Trace, RF Distance
            # to Reference, Pseudo-ESS) all depend on having a distmat
            # to operate against, so until one exists they're hidden
            # behind ``diagnostics-rf-sections`` and this placeholder
            # takes their place. Visibility flips in
            # ``toggle_diagnostics_rf_sections`` (callbacks/diagnostics.py).
            dmc.Paper(
                id="diagnostics-no-rf-placeholder",
                children=dmc.Stack([
                    dmc.Text(
                        "No RF distance matrix selected.",
                        size="md", fw=500, c="dimmed",
                    ),
                    dmc.Text(
                        "Compute an RF matrix in the Compute tab, then "
                        "pick one above to enable the diagnostics.",
                        size="sm", c="dimmed",
                    ),
                ], gap="xs", align="center"),
                p="md", withBorder=True, radius="sm",
                style={"textAlign": "center"},
            ),

            # ── RF-gated diagnostic sections ────────────────────────────
            # Wrapped in a single div so the visibility toggle is one
            # callback writing one style. Individual sections inside
            # keep their existing IDs / callbacks.
            html.Div(id="diagnostics-rf-sections", style={"display": "none"}, children=[
            dmc.Stack([
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
                    dmc.Button("Export", id="export-lnl-trace-button", variant="light",
                               size="xs", disabled=True,
                               leftSection=icon("tabler:pdf", size=20)),
                    # ``flex-end`` lines every control on the row's
                    # bottom edge. The Burnin ``NumberInput`` has a
                    # stacked label and is therefore taller than the
                    # title/badge/button, so ``align="center"`` used
                    # to center its midpoint and shove the label
                    # above the rest of the row.
                ], gap="sm", align="flex-end"),
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
                    dmc.NumberInput(
                        id="rf-burnin-input",
                        label="Burnin",
                        value=0,
                        min=0,
                        step=100,
                        size="xs",
                        w=120,
                    ),
                    dmc.Button(
                        "Compute RF Trace",
                        id="compute-rf-trace-button",
                        variant="filled",
                        color="green",
                        size="sm",
                        disabled=True,
                    ),
                    dmc.Button("Export", id="export-rf-trace-button", variant="light",
                               size="xs", disabled=True,
                               leftSection=icon("tabler:pdf", size=20)),
                    # Bottom-align (see Log-Posterior Trace) so the
                    # labeled Burnin input doesn't drift above the row.
                ], align="flex-end", gap="md"),
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
            ], gap="md"),
            ]),  # end ``diagnostics-rf-sections``
        ], gap="md"),
    ], style={"padding": "10px"})
