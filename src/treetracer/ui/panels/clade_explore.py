"""Clade Exploration tab panel: per-matrix consensus tree registry table + the
two-consensus-tree Compare controls, and the Clade Frequency Comparison output
(scatter + tanglegram).

Split out of the Diagnostics tab — clade-frequency comparison is
exploratory phylogenetics, not a convergence diagnostic. The tab has
its own "RF Matrix" selector (fed by the shared ``distmat-store``)
because comparison only makes sense between consensus trees of the same matrix.
"""

import dash_mantine_components as dmc
from dash import html

from .clade_freq import _add_clade_freq_panel


def _add_clade_explore_panel():
    """Build the Clade Exploration tab panel content."""
    return html.Div([
        dmc.Stack([
            # Header: the tab's own RF Matrix selector. Comparison is
            # matrix-scoped (consensus trees only compare within one distmat), so
            # the consensus tree table + Compare dropdowns below condition on it.
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

            # Empty-state notice shown while the consensus tree registry has no
            # entries at all. Hidden the moment any consensus tree is registered;
            # driven by ``toggle_clade_consensus_tree_empty_state`` in
            # callbacks/clade_explore.py. Visible by default because the
            # registry starts empty on load.
            dmc.Paper(
                dmc.Stack([
                    dmc.Text(
                        "No consensus trees available",
                        size="md", fw=500, c="dimmed",
                    ),
                    dmc.Text(
                        "Compute at least two consensus trees in the Between- or "
                        "Within-analysis tabs for the comparison.",
                        size="sm", c="dimmed",
                    ),
                ], gap="xs", align="center"),
                id="clade-consensus-tree-empty-paper",
                p="md", withBorder=True, radius="sm",
                style={"textAlign": "center"},
            ),

            # Per-matrix consensus tree registry summary + Clade-Frequency
            # comparison controls (two consensus tree dropdowns + Compare button).
            # The dropdowns and button sit at the top of this paper,
            # with the consensus tree table immediately below them, so picking
            # and comparing happen in the same visual unit.
            #
            # Output (scatter + tanglegram) renders into the separate
            # ``_add_clade_freq_panel`` below — those plots get heavy
            # and benefit from owning the page width without the table
            # crammed above them.
            #
            # Whole paper is hidden when no consensus trees match the active
            # matrix; driven by ``render_clade_consensus_tree_panel`` in
            # callbacks/clade_explore.py.
            dmc.Paper([
                # Layout order: "consensus trees for <matrix>" title, then
                # the Compare-clade dropdowns + button, then the
                # registered-consensus-tree table itself. Title and table are
                # two separate slots so the dropdowns can sit between
                # them without being re-rendered (and losing state)
                # every time the registry updates.
                html.Div(id="clade-consensus-tree-title"),
                dmc.Space(h=10),
                dmc.Group([
                    dmc.Select(
                        id="clade-freq-consensus-tree-select-1",
                        label="Group 1 (consensus tree)",
                        placeholder="No consensus trees saved yet",
                        data=[],
                        value=None,
                        disabled=True,
                        w=280,
                        size="sm",
                    ),
                    dmc.Select(
                        id="clade-freq-consensus-tree-select-2",
                        label="Group 2 (consensus tree)",
                        placeholder="No consensus trees saved yet",
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
                html.Div(id="clade-consensus-tree-list"),
            ], p="md", withBorder=True, radius="sm",
               id="clade-consensus-tree-paper",
               style={"display": "none"}),

            # Clade frequency comparison panel (scatter + tanglegram).
            _add_clade_freq_panel(),
        ], gap="md"),
    ], style={"padding": "10px"})
