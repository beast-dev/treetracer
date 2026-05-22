"""Clade Exploration tab panel: per-matrix MCC registry table + the
two-MCC Compare controls, and the Clade Frequency Comparison output
(scatter + tanglegram).

Split out of the Diagnostics tab — clade-frequency comparison is
exploratory phylogenetics, not a convergence diagnostic. The tab has
its own "RF Matrix" selector (fed by the shared ``distmat-store``)
because comparison only makes sense between MCCs of the same matrix.
"""

import dash_mantine_components as dmc
from dash import html

from .clade_freq import _add_clade_freq_panel


def _add_clade_explore_panel():
    """Build the Clade Exploration tab panel content."""
    return html.Div([
        dmc.Stack([
            # Header: the tab's own RF Matrix selector. Comparison is
            # matrix-scoped (MCCs only compare within one distmat), so
            # the MCC table + Compare dropdowns below condition on it.
            dmc.Paper([
                dmc.Group([
                    dmc.Select(
                        id="clade-distmat-select",
                        label="RF Matrix",
                        placeholder="No distance matrix available",
                        data=[],
                        value=None,
                        size="xs",
                        w=300,
                    ),
                    html.Div(id="clade-distmat-info"),
                ], align="flex-end", gap="md"),
            ], p="md", withBorder=True, radius="sm"),

            # Per-matrix MCC registry summary + Clade-Frequency
            # comparison controls (two MCC dropdowns + Compare button).
            # The dropdowns and button sit at the top of this paper,
            # with the MCC table immediately below them, so picking
            # and comparing happen in the same visual unit.
            #
            # Output (scatter + tanglegram) renders into the separate
            # ``_add_clade_freq_panel`` below — those plots get heavy
            # and benefit from owning the page width without the table
            # crammed above them.
            #
            # Whole paper is hidden when no MCCs match the active
            # matrix; driven by ``render_clade_mcc_panel`` in
            # callbacks/clade_explore.py.
            dmc.Paper([
                # Layout order: "MCC trees for <matrix>" title, then
                # the Compare-clade dropdowns + button, then the
                # registered-MCC table itself. Title and table are
                # two separate slots so the dropdowns can sit between
                # them without being re-rendered (and losing state)
                # every time the registry updates.
                html.Div(id="clade-mcc-title"),
                dmc.Space(h=10),
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
                dmc.Space(h=10),
                html.Div(id="clade-mcc-list"),
            ], p="md", withBorder=True, radius="sm",
               id="clade-mcc-paper",
               style={"display": "none"}),

            # Clade frequency comparison panel (scatter + tanglegram).
            _add_clade_freq_panel(),
        ], gap="md"),
    ], style={"padding": "10px"})
