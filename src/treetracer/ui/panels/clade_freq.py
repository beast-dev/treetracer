"""Clade Frequency Comparison panel — the scatter + tanglegram pair that
sits inside the Clade Exploration tab once two MCC trees are picked.

The controls (two MCC dropdowns, Compare button) live in the
``clade-mcc-paper`` block in the Clade Exploration panel; this module
defines only the output surface.
"""

import dash_mantine_components as dmc
from dash import dcc, html


def _add_clade_freq_panel():
    """Output paper for the Clade Frequency Comparison feature.

    The CONTROLS — two MCC-tree dropdowns and the Compare button —
    live in ``clade-mcc-paper`` (immediately under the MCC table
    they pull from). This paper is just the output surface: the
    scatter, the two sliders that re-shape it, and the tanglegram
    below.

    Components built here:
        clade-freq-min-clade-size   horizontal slider above the scatter
        clade-freq-plot             scatter (freq_1 vs freq_2)
        clade-freq-tanglegram       pair of trees with red-highlighted
                                    clicked clade
        tanglegram-yscale-slider    vertical slider beside the tanglegram
    """
    return dmc.Paper([
        dmc.Group([
            dmc.Title("Clade Frequency Comparison", order=5),
            dmc.Badge(
                "select two MCC trees above and click Compare",
                variant="light", size="sm",
            ),
        ], gap="sm", align="center"),

        dmc.Space(h=10),

        # Min-clade-size sits directly above the scatter it filters —
        # placing it between scatter and tanglegram (as previously)
        # left it visually orphaned from either chart.
        dmc.Stack([
            dmc.Text("Minimum clade size", size="md", fw=500),
            dmc.Slider(
                id="clade-freq-min-clade-size",
                min=2, max=50, step=1, value=5,
                w=300,
                size="md",
                marks=[
                    {"value": 2,  "label": "2"},
                    {"value": 10, "label": "10"},
                    {"value": 25, "label": "25"},
                    {"value": 50, "label": "50"},
                ],
                styles={"markLabel": {"fontSize": "13px"}},
            ),
        ], gap=6, mt=4, mb=12),

        dcc.Loading(
            html.Div(id="clade-freq-plot"),
            type="circle",
            parent_style={"minHeight": "200px"},
        ),
        # Tanglegram + its vertical Expand-tree slider. The slider
        # scales the figure's y-axis so it reads naturally as a
        # vertical control on the right edge of the chart. Static
        # dcc.Graph so ``dash.Patch`` can update only the dynamic
        # traces (highlight markers + connectors) on each click —
        # the branches and grey-tips skeleton stays put. The
        # tanglegram-pair-store tracks which MCC pair is currently
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
                                "text": "Select two MCC trees and click "
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
    ], p="md", withBorder=True, radius="sm",
       # Hidden until ``compute_and_plot_clade_frequencies`` succeeds —
       # nothing useful to show before Compare has run. Cleared back to
       # hidden by the sidebar's Clear-data flow.
       id="clade-freq-output-paper",
       style={"display": "none"})
