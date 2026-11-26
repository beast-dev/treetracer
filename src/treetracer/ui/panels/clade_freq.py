"""Clade Frequency Comparison panel — the scatter + tanglegram pair that
sits inside the Clade Exploration tab once two consensus trees are picked.

The controls (two consensus tree dropdowns, Compare button) live in the
``clade-consensus-tree-paper`` block in the Clade Exploration panel; this module
defines only the output surface.
"""

import dash_mantine_components as dmc
from dash import dcc, html

from ...icons import icon


def _add_clade_freq_panel():
    """Output paper for the Clade Frequency Comparison feature.

    The CONTROLS — two consensus-tree dropdowns and the Compare button —
    live in ``clade-consensus-tree-paper``. This paper holds the output
    surface: scatter controls, side-by-side scatter + Tip Instability Table,
    and the tanglegram below.
    """
    return dmc.Paper([
        dmc.Group([
            dmc.Title("Clade Frequency & Tip Instability Comparison", order=5),
            dmc.Badge(
                "select two consensus trees above and click Compare",
                variant="light", size="sm",
            ),
        ], gap="sm", align="center"),

        dmc.Space(h=10),

        # Controls strip above the scatter. Keep ``clade-freq-exclusive-only``
        # as the direct last child of this space-between Group (pinned by
        # ``test_app_smoke``). Two dedicated slider wrappers toggle CSS display
        # on mode switch so Dash never evaluates an integer size threshold
        # against [0, 1] instability.
        dmc.Group(
            [
                dmc.Group(
                    [
                        dmc.Tooltip(
                            id="clade-freq-color-by-tooltip",
                            label=(
                                "Clade instability requires strictly binary "
                                "rooted snapshots (RootedFactsSnapshot)"
                            ),
                            disabled=True,
                            withArrow=True,
                            children=dmc.Select(
                                id="clade-freq-color-by",
                                label="Color & filter by",
                                data=[
                                    {"value": "clade_size", "label": "Clade size"},
                                    {
                                        "value": "clade_instability",
                                        "label": "Clade instability",
                                    },
                                ],
                                value="clade_size",
                                allowDeselect=False,
                                size="sm",
                                w=185,
                            ),
                        ),
                        html.Div(
                            dmc.Stack(
                                [
                                    dmc.Text("Minimum clade size", size="sm", fw=500),
                                    dmc.Slider(
                                        id="clade-freq-min-clade-size",
                                        min=2,
                                        max=50,
                                        step=1,
                                        value=5,
                                        w=260,
                                        size="md",
                                        marks=[
                                            {"value": 2, "label": "2"},
                                            {"value": 10, "label": "10"},
                                            {"value": 25, "label": "25"},
                                            {"value": 50, "label": "50"},
                                        ],
                                        styles={"markLabel": {"fontSize": "12px"}},
                                    ),
                                ],
                                gap=6,
                            ),
                            id="clade-freq-size-slider-wrap",
                        ),
                        html.Div(
                            dmc.Stack(
                                [
                                    dmc.Text(
                                        "Minimum clade instability",
                                        size="sm",
                                        fw=500,
                                    ),
                                    dmc.Slider(
                                        id="clade-freq-min-instability",
                                        min=0.0,
                                        max=1.0,
                                        step=0.05,
                                        value=0.0,
                                        w=260,
                                        size="md",
                                        marks=[
                                            {"value": 0.0, "label": "0.0"},
                                            {"value": 0.25, "label": "0.25"},
                                            {"value": 0.5, "label": "0.50"},
                                            {"value": 0.75, "label": "0.75"},
                                            {"value": 1.0, "label": "1.0"},
                                        ],
                                        styles={"markLabel": {"fontSize": "12px"}},
                                    ),
                                ],
                                gap=6,
                            ),
                            id="clade-freq-instability-slider-wrap",
                            style={"display": "none"},
                        ),
                    ],
                    gap="lg",
                    align="flex-end",
                    wrap="wrap",
                ),
                dmc.Checkbox(
                    id="clade-freq-exclusive-only",
                    label=(
                        "Show only clades present in a single consensus tree"
                    ),
                    checked=False,
                    size="sm",
                ),
            ],
            justify="space-between",
            align="center",
            w="100%",
            mt=4,
            mb=0,
        ),

        # Slider marks overhang ~18px below the track; separate them cleanly
        # from the scatter banner and Tip Instability card below.
        dmc.Divider(mt=24, mb="sm"),

        dmc.Grid(
            [
                dmc.GridCol(
                    [
                        dmc.Group(
                            [
                                dmc.Text(
                                    "",
                                    id="clade-freq-tip-filter-banner-text",
                                    className="tt-tip-filter-banner-text",
                                    size="xs",
                                    fw=600,
                                ),
                                dmc.ActionIcon(
                                    "×",
                                    id="clade-freq-tip-banner-dismiss",
                                    size="xs",
                                    variant="subtle",
                                    color="red",
                                    **{"aria-label": "Clear selected tip"},
                                ),
                            ],
                            id="clade-freq-tip-filter-banner",
                            className="tt-tip-filter-banner",
                            gap=6,
                            align="center",
                            wrap="nowrap",
                            style={"display": "none"},
                        ),
                        dcc.Loading(
                            html.Div(id="clade-freq-plot"),
                            type="circle",
                            parent_style={"minHeight": "450px"},
                        ),
                    ],
                    span={"base": 12, "lg": 7},
                ),
                dmc.GridCol(
                    dmc.Paper(
                        [
                            dmc.Group(
                                [
                                    dmc.Group(
                                        [
                                            dmc.Text("Tip Instability", fw=600, size="sm"),
                                            dmc.Badge(
                                                "",
                                                id="tip-instability-badge",
                                                variant="light",
                                                color="red",
                                                size="sm",
                                                style={"display": "none"},
                                            ),
                                        ],
                                        gap="xs",
                                    ),
                                    dmc.Button(
                                        "Export TSV",
                                        id="tip-instability-export-btn",
                                        leftSection=icon("tabler:download", size=14),
                                        variant="light",
                                        size="xs",
                                        disabled=True,
                                    ),
                                ],
                                justify="space-between",
                                align="center",
                            ),
                            dmc.Group(
                                [
                                    dmc.TextInput(
                                        id="tip-instability-search",
                                        placeholder="Search taxon...",
                                        size="xs",
                                        style={"flex": "1"},
                                    ),
                                    dmc.Select(
                                        id="tip-instability-sort",
                                        size="xs",
                                        w=150,
                                        allowDeselect=False,
                                        value="d_sis",
                                        data=[
                                            {
                                                "value": "d_sis",
                                                "label": "D_sis (desc)",
                                            },
                                            {
                                                "value": "combined_instability",
                                                "label": "N_comb (desc)",
                                            },
                                            {
                                                "value": "instab_1",
                                                "label": "N_G1 (desc)",
                                            },
                                            {
                                                "value": "instab_2",
                                                "label": "N_G2 (desc)",
                                            },
                                            {
                                                "value": "taxon",
                                                "label": "Taxon (A–Z)",
                                            },
                                        ],
                                    ),
                                ],
                                id="tip-instability-controls",
                                gap="xs",
                                mt="xs",
                                mb="xs",
                                wrap="nowrap",
                            ),
                            html.Div(
                                id="tip-instability-table-body",
                                className="tt-tip-instability-scroll",
                            ),
                            dmc.Group(
                                [
                                    dmc.Text(
                                        "",
                                        id="tip-instability-summary",
                                        size="xs",
                                        c="dimmed",
                                    ),
                                    dmc.Button(
                                        "Show more",
                                        id="tip-instability-show-more",
                                        variant="subtle",
                                        size="compact-xs",
                                        style={"display": "none"},
                                    ),
                                ],
                                id="tip-instability-pagination-bar",
                                justify="space-between",
                                align="center",
                                mt=6,
                            ),
                            html.Div(
                                dmc.Text(
                                    "Click ⊙ on any taxon to highlight its jump in "
                                    "the tanglegram and filter the scatter to clades "
                                    "containing it.",
                                    size="xs",
                                    c="dimmed",
                                ),
                                id="tip-instability-detail-footer",
                                className="tt-tip-instability-footer",
                            ),
                        ],
                        withBorder=True,
                        radius="sm",
                        p="sm",
                        className="tt-tip-instability-card",
                    ),
                    span={"base": 12, "lg": 5},
                ),
            ],
            gutter="md",
            align="stretch",
            mb="sm",
        ),
        # Tanglegram + its vertical Expand-tree slider. The slider
        # scales the figure's y-axis so it reads naturally as a
        # vertical control on the right edge of the chart. Static
        # dcc.Graph so ``dash.Patch`` can update only the dynamic
        # traces (highlight markers + connectors) on each click —
        # the branches and grey-tips skeleton stays put. The
        # tanglegram-pair-store tracks which consensus tree pair is currently
        # rendered so the callback knows when a full rebuild is
        # required (different uids) vs a Patch-only update.
        # Title lives in its own ``position: sticky`` div above the
        # tanglegram graph so it stays visible when the user expands
        # the tree (the figure can grow to many thousands of pixels
        # tall; an in-figure annotation at paper-y=1.02 scrolls off
        # the top of the viewport along with the plot's upper edge).
        html.Div(
            id="clade-freq-tanglegram-title",
            style={
                "position": "sticky",
                # Stick BELOW the pinned tab strip (header offset + the
                # tab strip's height) instead of at the viewport top, so
                # the title doesn't ride up over the tabs while the tall
                # tanglegram scrolls. zIndex stays under the tabs (2) so
                # the tabs always win any overlap, but above the
                # scrolling graph (0) so the title stays readable.
                "top": "calc(var(--app-shell-header-offset, 60px) + 42px)",
                "zIndex": 1,
                # Follow the app theme (white in light, dark surface in
                # dark) so the sticky title blends instead of showing a
                # white box over the dark tanglegram.
                "background": "var(--mantine-color-body)",
                "padding": "6px 0",
                "textAlign": "center",
                "fontSize": "18px",
                "minHeight": "30px",
            },
        ),
        dmc.Group([
            dcc.Loading(
                dcc.Graph(
                    id="clade-freq-tanglegram",
                    figure={
                        "data": [],
                        "layout": {
                            "height": 200,
                            "xaxis": {"visible": False},
                            "yaxis": {"visible": False},
                            "plot_bgcolor": "rgba(0,0,0,0)",
                            "paper_bgcolor": "rgba(0,0,0,0)",
                            "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
                            "annotations": [{
                                "text": "Select two consensus trees and click "
                                        "<b>Compare Clade Frequencies</b>,"
                                        " then click a dot in the scatter "
                                        "above to draw the tanglegram.",
                                "xref": "paper", "yref": "paper",
                                "x": 0.5, "y": 0.5,
                                "showarrow": False,
                                "font": {"size": 13, "color": "#888"},
                                "align": "center",
                            }],
                        },
                    },
                    config={"displayModeBar": False},
                    style={"width": "100%"},
                ),
                type="circle",
                parent_style={"minHeight": "200px", "flex": "1"},
                style={"flex": "1"},
            ),
            dmc.Stack([
                # ── Expand-tree slider: label centred BESIDE the vertical
                # track (horizontal Group), not stacked above it. The
                # control sits in a fixed-width cell and the label uses
                # the SAME font size as the complement label below; since
                # vertical text has a constant horizontal width (= line
                # height, independent of string length), equal cells +
                # equal font size keep the two labels in one column. ──
                dmc.Group([
                    dmc.Box(
                        # dcc.Slider has a native vertical orientation; dmc
                        # currently does not, so we drop down to dcc here.
                        # ``verticalHeight`` is in px and is independent of
                        # the tanglegram's dynamic height — 240 keeps it
                        # reachable for short trees and not overwhelming for
                        # tall ones.
                        # ``reverse=True`` puts the slider's max at the BOTTOM
                        # so dragging the thumb downwards expands the tree —
                        # parallels how the tanglegram itself grows downward
                        # as its height increases.
                        dcc.Slider(
                            id="tanglegram-yscale-slider",
                            min=1, max=5, step=1, value=1,
                            vertical=True,
                            verticalHeight=240,
                            reverse=True,
                            # Empty-LABEL marks (not an omitted/empty ``marks``,
                            # which makes dcc auto-generate the numeric 1–5
                            # labels). Two jobs: they hide the numbers, AND they
                            # keep the slider element narrow so the rail centres
                            # in its 40px cell — lining up with the centre-
                            # rotated switch below. Omitting marks widens the
                            # element with the number column and shoves the rail
                            # left, breaking that alignment. Verified by headless
                            # render.
                            marks={1: "", 3: "", 5: ""},
                            tooltip={"placement": "left", "always_visible": False},
                        ),
                        style={"width": 40, "display": "flex",
                               "justifyContent": "center"},
                    ),
                    dmc.Text("Expand tree", size="md", fw=500,
                             style={"writingMode": "vertical-rl",
                                    "transform": "rotate(180deg)"}),
                ], gap=4, align="center", wrap="nowrap"),
                # ── Complement toggle: the Switch is rotated 90° so it
                # runs vertically like the slider, with its label centred
                # beside it. Same fixed-width control cell + same label
                # font size as the slider above so the two vertical labels
                # line up. The Box also reserves vertical space for the
                # rotated switch (CSS transforms don't reflow layout, so
                # without it the rotated track would clip/overlap). ──
                dmc.Group([
                    dmc.Box(
                        dmc.Tooltip(
                            label="Highlight tips inside the MRCA that are not part of the selected clade",
                            position="left",
                            withArrow=True,
                            children=dmc.Switch(
                                id="tanglegram-complement-toggle",
                                label="",
                                checked=False,
                                size="sm",
                                color="green",
                                # The Switch's track sits ~10px right of the
                                # element centre (the root reserves a label
                                # slot), so once centred in its cell the pill
                                # lands right of the slider's thumb. translateX
                                # (applied after the rotate) nudges the pill
                                # back onto the thumb's vertical axis. The 10px
                                # is for size="sm"; measured via headless render
                                # (thumb-centre == track-centre, Δ=0).
                                style={"transform": "translateX(-10px) rotate(90deg)"},
                            ),
                        ),
                        # Box reserves room for the rotated track; width
                        # stays 40 to keep the label aligned with the
                        # slider's label above.
                        style={"width": 40, "height": 48, "display": "flex",
                               "alignItems": "center", "justifyContent": "center"},
                    ),
                    dmc.Text("Clade complement", size="md", fw=500,
                             style={"writingMode": "vertical-rl",
                                    "transform": "rotate(180deg)"}),
                ], gap=4, align="center", wrap="nowrap"),
                dmc.Group([
                    dmc.Box(
                        dmc.Tooltip(
                            label=(
                                "Connect G1 (blue) and G2 (green) immediate sister "
                                "clades across both trees (monophyletic sides only)"
                            ),
                            position="left",
                            withArrow=True,
                            children=dmc.Switch(
                                id="tanglegram-sister-toggle",
                                label="",
                                checked=False,
                                size="sm",
                                color="blue",
                                style={"transform": "translateX(-10px) rotate(90deg)"},
                            ),
                        ),
                        style={"width": 40, "height": 48, "display": "flex",
                               "alignItems": "center", "justifyContent": "center"},
                    ),
                    dmc.Text("Connect sister", size="md", fw=500,
                             style={"writingMode": "vertical-rl",
                                    "transform": "rotate(180deg)"}),
                ], gap=4, align="center", wrap="nowrap"),
            ], gap=6, align="center", pt=8,
               # Pin the Expand-tree slider + Clade-complement toggle so
               # they stay reachable while a tall tanglegram scrolls past.
               # Sticky just below the pinned tab strip; the parent
               # Group's ``align="flex-start"`` keeps this Stack at its
               # natural height (not stretched to the graph's height),
               # which position:sticky needs to have room to move.
               style={
                   "position": "sticky",
                   "top": "calc(var(--app-shell-header-offset, 60px) + 42px)",
                   "zIndex": 2,
                   "alignSelf": "flex-start",
               }),
        ], gap="md", align="flex-start", wrap="nowrap"),
        dcc.Store(id="clade-freq-tanglegram-pair-store"),
        dcc.Store(id="tip-instability-limit-store", storage_type="memory", data=50),
        dcc.Store(id="tip-instability-filter-store", storage_type="memory"),
    ], p="md", withBorder=True, radius="sm",
       # Hidden until ``compute_and_plot_clade_frequencies`` succeeds —
       # nothing useful to show before Compare has run. Cleared back to
       # hidden by the sidebar's Clear-data flow.
       id="clade-freq-output-paper",
       style={"display": "none"})
