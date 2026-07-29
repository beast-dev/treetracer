import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .theme import DARK_TEMPLATE, get_template


def retheme_figure(fig_dict, *, skip_invalid=False):
    """Rebuild a figure dict with the active Plotly template.

    Updating ``layout.template`` in-place via a Dash Patch doesn't reliably
    cause Plotly to recompute template-derived defaults for an already-
    rendered figure. Reconstructing the figure and then applying the current
    template does.
    """
    fig = go.Figure(fig_dict, skip_invalid=skip_invalid)
    fig.update_layout(template=get_template())
    return fig


def placeholder_fig(text):
    """Empty figure with centered placeholder text. Used so the between-run
    Graph component can exist statically (giving selection callbacks a stable
    target) even before the user has plotted anything."""
    dark = get_template() == DARK_TEMPLATE
    return {
        "data": [],
        "layout": {
            "template": get_template(),
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "annotations": [{
                "text": text,
                "xref": "paper", "yref": "paper",
                "x": 0.5, "y": 0.5,
                "showarrow": False,
                "font": {"size": 18, "color": "#868e96" if dark else "#666"},
            }],
            "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
            "plot_bgcolor": "rgba(0,0,0,0)",
            "paper_bgcolor": "rgba(0,0,0,0)",
        },
    }


# Number of slices each group's points are split into before being added to
# the 2D panels. With N groups and K chunks, the marker traces are added in
# chunk-index order — chunk 0 of all groups, then chunk 1 of all groups, etc.
# At every z-depth in the resulting plot each group has roughly equal
# representation, so no single run sits entirely on top of the others.
#
# Bigger K → finer-grained interleaving (less chance a run dominates any
# zoom level) at the cost of more ``fig.add_trace`` calls (G·K·3 in total).
# 15 lands at ~135 trace adds for a 3-run plot, which Plotly handles
# comfortably in WebGL. ``_chunked`` caps at ``min(K, n)`` so small runs
# aren't penalised.
N_CHUNKS_2D = 15


# Trailing overlay bundles appended to the between-runs figure: one
# bundle for the user's current SELECTION (red), one for every
# REGISTERED MCC (green). Each bundle has 4 traces (1 × Scatter3d for
# the 3D panel, 3 × Scatter for the 2D panels).
N_SELECTION_OVERLAYS = 4
N_MCC_OVERLAYS = 4
N_TRAILING_OVERLAYS = N_SELECTION_OVERLAYS + N_MCC_OVERLAYS

_OVERLAY_STYLES = {
    "selection": {
        "color": "red",
        "size3d": 8,
        "size2d": 12,
        "line2d_width": 1.5,
        "name": "selection",
        "hover": "Tree #%{customdata[0]}: %{customdata[1]}<extra>selected</extra>",
    },
    "mcc": {
        "color": "#39ff14",   # neon green
        # 3D ring needs to be substantially bigger than the selection
        # ring — Scatter3d ``circle-open`` strokes scale with size, not
        # with line.width, so size IS the visual weight.
        "size3d": 12,
        "size2d": 14,
        "line2d_width": 3.0,
        "name": "mcc",
        "hover": "Tree #%{customdata[0]}: %{customdata[1]}<extra>MCC</extra>",
    },
}


def _add_overlay_bundle(fig, panels_2d, *, kind):
    """Append a 4-trace overlay bundle (1 × 3D + 3 × 2D) to *fig*.

    *kind* is ``"selection"`` (red) or ``"mcc"`` (green). Both bundles
    share the same Scatter3d / Scatter shape so the patching callbacks
    can address them by fixed negative offsets.
    """
    style = _OVERLAY_STYLES[kind]
    # Scatter3d ignores ``marker.line.width`` for visible thickness, so
    # both 3D overlays use the ``circle-open`` symbol (the marker colour
    # *is* the ring) and lean on size for prominence. The MCC ring is
    # noticeably larger than the selection ring so a tree that is both
    # selected and a registered MCC reads as two concentric circles.
    fig.add_trace(
        go.Scatter3d(
            x=[], y=[], z=[],
            mode="markers",
            marker=dict(size=style["size3d"], color=style["color"],
                        symbol="circle-open"),
            customdata=[],
            hovertemplate=style["hover"],
            showlegend=False,
            name=style["name"],
        ),
        row=1, col=1,
    )
    # ``Scattergl`` overlays so they live on the SAME WebGL canvas as
    # the data traces. With both on one canvas, trace insertion order
    # determines draw order — and since this bundle is appended AFTER
    # all data + selection bundles, the MCC ring lands on top of
    # everything. Trying ``zorder`` on a Scatter overlay didn't work:
    # Plotly's WebGL canvas paints above the SVG layer in subplots, so
    # the SVG ring was hidden behind data. (Scattergl rejects zorder.)
    for xcol, ycol, row, col in panels_2d:
        fig.add_trace(
            go.Scattergl(
                x=[], y=[],
                mode="markers",
                marker=dict(
                    size=style["size2d"],
                    color="rgba(0,0,0,0)",
                    line=dict(color=style["color"],
                              width=style["line2d_width"]),
                ),
                customdata=[],
                hovertemplate=style["hover"],
                showlegend=False,
                # Pin both states to opacity 1 so Plotly's box-select
                # selectedpoints stamping doesn't fade overlay circles.
                selected=dict(marker=dict(opacity=1)),
                unselected=dict(marker=dict(opacity=1)),
                name=style["name"],
            ),
            row=row, col=col,
        )


def _chunked(group_data, n_chunks, seed=42):
    """Partition a DataFrame into ``n_chunks`` chunks of randomly-sampled rows
    (without replacement). Every row appears in exactly one chunk; chunks are
    roughly equal in size. The shuffle is seeded for stable rendering — the
    same input produces the same partition on every re-render.

    Yields ``(chunk_idx, sub_df)``. When the group has fewer rows than
    ``n_chunks``, only ``len(group_data)`` chunks of size 1 are emitted.
    """
    n = len(group_data)
    if n == 0:
        return
    n_chunks = min(n_chunks, n)
    base, extra = divmod(n, n_chunks)
    shuffled = group_data.sample(frac=1, random_state=seed)
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < extra else 0)
        yield i, shuffled.iloc[start:start + size]
        start += size


def make_plot_grid():
    f = make_subplots(
        rows=3,
        cols=2,
        column_widths=[0.75, 0.25],
        horizontal_spacing=0.093,
        specs=[
            [{"type": "scatter3d", "rowspan": 3}, {"type": "scatter"}],
            [None, {"type": "scatter"}],
            [None, {"type": "scatter"}],
        ],
    )
    return f


def add_trace_multiplot(fig, df, x, y, z, GROUPS, COLOR_DICT, show_lines=True):
    mode = "lines+markers" if show_lines else "markers"

    # 2D panels: (x_col, y_col, row, col)
    panels_2d = [
        (x, y, 1, 2),
        (x, z, 2, 2),
        (y, z, 3, 2),
    ]

    for i, gr in enumerate(GROUPS):
        group_data = df[df["group"] == gr]
        color = COLOR_DICT[gr]
        tree_short = group_data["tree"].str.split("/").str[-1].str.strip()
        customdata = list(zip(group_data["treenum"], tree_short))
        hover = f"{gr}<br>Tree #%{{customdata[0]}}: %{{customdata[1]}}<extra></extra>"

        # Invisible legend-only trace
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                name=gr,
                showlegend=True,
                marker=dict(color=color, size=10),
                legendgroup=gr,
                legendrank=i,
            ),
            row=1, col=2,
        )

        # 3D scatter
        fig.add_trace(
            go.Scatter3d(
                x=group_data[x], y=group_data[y], z=group_data[z],
                mode=mode,
                name=gr,
                showlegend=False,
                marker=dict(color=color, size=4),
                line=dict(color=color, width=1),
                legendgroup=gr,
                hovertemplate=hover,
                customdata=customdata,
            ),
            row=1, col=1,
        )

        # Three 2D projections. ``Scattergl`` (WebGL) handles thousands
        # of points per group without freezing the browser; SVG starts
        # to feel sluggish around ~2k. The data traces share their
        # ``legendgroup`` with the invisible legend driver above, so
        # toggling the legend hides both the WebGL data and the
        # (separate, SVG) overlay rings tied to that group.
        for xcol, ycol, row, col in panels_2d:
            fig.add_trace(
                go.Scattergl(
                    x=group_data[xcol], y=group_data[ycol],
                    mode=mode,
                    name=gr,
                    showlegend=False,
                    marker=dict(color=color),
                    line=dict(color=color, width=1),
                    legendgroup=gr,
                    hovertemplate=hover,
                    customdata=customdata,
                ),
                row=row, col=col,
            )

    fig.update_layout(
        template=get_template(),
        scene=dict(
            xaxis=dict(title=dict(text=x)),
            yaxis=dict(title=dict(text=y)),
            zaxis=dict(title=dict(text=z)),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=14),
            itemsizing="constant",
        ),
        legend_itemwidth=40,
        uirevision="constant",
        # ``t=50`` gives the horizontal legend (anchored just above the
        # plot area at y=1.02) room to render without clipping.
        margin=dict(l=2, r=20, t=50, b=10),
    )

    for xcol, ycol, row, col in panels_2d:
        fig.update_xaxes(title_text=xcol, row=row, col=col)
        fig.update_yaxes(title_text=ycol, row=row, col=col)


def add_trace_multiplot_interleaved(fig, df, x, y, z, GROUPS, COLOR_DICT, show_lines=True):
    """Variant of ``add_trace_multiplot`` that interleaves z-order in the 2D panels.

    Plotly draws traces in the order they're added — so when each group becomes
    one trace per panel, the last group's points sit on top of every earlier
    group's. With overlapping clouds in MDS space that produces a stacked
    "covering" effect that hides whatever is drawn first.

    This variant splits each group's points into ``N_CHUNKS_2D`` chunks and
    adds the chunks layered by chunk-index — chunk 0 of every group, then
    chunk 1 of every group, and so on — so every z-depth contains a roughly
    equal sample from each group. Lines (when enabled) are kept as continuous
    per-group traces drawn underneath the markers; splitting lines into chunks
    would break the trajectory. Legend interactivity is preserved via
    ``legendgroup`` on every chunk.

    The 3D panel and the legend traces match ``add_trace_multiplot`` exactly —
    3D z-order is rotation-dependent and doesn't suffer from the per-group
    covering problem.
    """
    mode_3d = "lines+markers" if show_lines else "markers"

    panels_2d = [
        (x, y, 1, 2),
        (x, z, 2, 2),
        (y, z, 3, 2),
    ]

    # 1. Invisible legend-only markers — one per group, drives the clickable legend.
    for i, gr in enumerate(GROUPS):
        color = COLOR_DICT[gr]
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                name=gr,
                showlegend=True,
                marker=dict(color=color, size=10),
                legendgroup=gr,
                legendrank=i,
            ),
            row=1, col=2,
        )

    # 2. 3D scatter — one trace per group.
    for gr in GROUPS:
        group_data = df[df["group"] == gr]
        color = COLOR_DICT[gr]
        tree_short = group_data["tree"].str.split("/").str[-1].str.strip()
        # customdata carries (treenum, basename, group) so click handlers can
        # disambiguate points across runs — `treenum` resets per group.
        customdata = list(zip(group_data["treenum"], tree_short,
                              [gr] * len(group_data)))
        hover = f"{gr}<br>Tree #%{{customdata[0]}}: %{{customdata[1]}}<extra></extra>"

        fig.add_trace(
            go.Scatter3d(
                x=group_data[x], y=group_data[y], z=group_data[z],
                mode=mode_3d,
                name=gr,
                showlegend=False,
                marker=dict(color=color, size=4),
                line=dict(color=color, width=1),
                legendgroup=gr,
                hovertemplate=hover,
                customdata=customdata,
            ),
            row=1, col=1,
        )

    # 3a. 2D continuous lines per group, drawn underneath the markers.
    # WebGL-backed ``Scattergl`` so 8k-tree trajectories don't choke
    # the SVG renderer.
    if show_lines:
        for gr in GROUPS:
            group_data = df[df["group"] == gr]
            color = COLOR_DICT[gr]
            for xcol, ycol, row, col in panels_2d:
                fig.add_trace(
                    go.Scattergl(
                        x=group_data[xcol], y=group_data[ycol],
                        mode="lines",
                        line=dict(color=color, width=1),
                        legendgroup=gr,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row, col=col,
                )

    # 3b. Interleaved markers. Build chunks for every group, then sort by
    # chunk-index so chunk 0 of all groups is added before chunk 1 of any.
    all_chunks = []
    for gr in GROUPS:
        group_data = df[df["group"] == gr]
        for chunk_idx, chunk_df in _chunked(group_data, N_CHUNKS_2D):
            all_chunks.append((chunk_idx, gr, chunk_df))
    all_chunks.sort(key=lambda t: t[0])

    # WebGL marker traces. ``marker.line`` (the thin grey outline) is
    # silently ignored by Scattergl on most Plotly versions, so the
    # outline drops away and points read as solid disks — acceptable
    # trade-off for handling thousands of trees per group without lag.
    for _chunk_idx, gr, chunk_df in all_chunks:
        color = COLOR_DICT[gr]
        tree_short = chunk_df["tree"].str.split("/").str[-1].str.strip()
        customdata = list(zip(chunk_df["treenum"], tree_short,
                              [gr] * len(chunk_df)))
        hover = f"{gr}<br>Tree #%{{customdata[0]}}: %{{customdata[1]}}<extra></extra>"
        for xcol, ycol, row, col in panels_2d:
            fig.add_trace(
                go.Scattergl(
                    x=chunk_df[xcol], y=chunk_df[ycol],
                    mode="markers",
                    marker=dict(color=color),
                    opacity=0.9,
                    legendgroup=gr,
                    showlegend=False,
                    hovertemplate=hover,
                    customdata=customdata,
                    selected=dict(marker=dict(opacity=0.9)),
                    unselected=dict(marker=dict(opacity=0.9)),
                ),
                row=row, col=col,
            )

    # 4. Trailing overlay traces, in two bundles of 4 (1×3D + 3×2D each):
    #
    #     -8 → selection 3D (red, hollow)
    #     -7 → selection 2D x-y
    #     -6 → selection 2D x-z
    #     -5 → selection 2D y-z
    #     -4 → MCC      3D (green, hollow, slightly larger so a tree
    #                       that's both selected and a registered MCC
    #                       reads as two concentric rings)
    #     -3 → MCC      2D x-y
    #     -2 → MCC      2D x-z
    #     -1 → MCC      2D y-z
    #
    # The two callbacks (update_selection_overlay / update_mcc_overlay)
    # patch their bundle by these fixed negative offsets without
    # rebuilding the figure.
    _add_overlay_bundle(fig, panels_2d, kind="selection")
    _add_overlay_bundle(fig, panels_2d, kind="mcc")

    fig.update_layout(
        template=get_template(),
        scene=dict(
            xaxis=dict(title=dict(text=x)),
            yaxis=dict(title=dict(text=y)),
            zaxis=dict(title=dict(text=z)),
        ),
        # Sync the 2D panels: panel 2 (xz) shares x with panel 1 (xy);
        # panel 3 (yz) shares its x with panel 1's y; panel 3's y shares
        # with panel 2's y. Same logic as within-run's 1×3 layout.
        xaxis2=dict(matches='x'),
        xaxis3=dict(matches='y'),
        yaxis3=dict(matches='y2'),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=14),
            itemsizing="constant",
        ),
        legend_itemwidth=40,
        uirevision="treespace",
        # ``t=50`` gives the horizontal legend (anchored just above the
        # plot area at y=1.02) room to render without clipping.
        margin=dict(l=2, r=20, t=50, b=10),
    )

    for xcol, ycol, row, col in panels_2d:
        fig.update_xaxes(title_text=xcol, row=row, col=col)
        fig.update_yaxes(title_text=ycol, row=row, col=col)
