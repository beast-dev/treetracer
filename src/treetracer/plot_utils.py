import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Number of slices each group's points are split into before being added to
# the 2D panels. With N groups and K chunks, the marker traces are added in
# chunk-index order — chunk 0 of all groups, then chunk 1 of all groups, etc.
# At every z-depth in the resulting plot each group has roughly equal
# representation, so no single run sits entirely on top of the others.
N_CHUNKS_2D = 10


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

        # Three 2D projections
        for xcol, ycol, row, col in panels_2d:
            fig.add_trace(
                go.Scatter(
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
        template="simple_white",
        scene=dict(
            xaxis=dict(title=dict(text=x)),
            yaxis=dict(title=dict(text=y)),
            zaxis=dict(title=dict(text=z)),
        ),
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.15,
            xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=14),
            itemsizing="constant",
        ),
        legend_itemwidth=40,
        uirevision="constant",
        margin=dict(l=2, r=20, t=25, b=10),
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
        customdata = list(zip(group_data["treenum"], tree_short))
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
    if show_lines:
        for gr in GROUPS:
            group_data = df[df["group"] == gr]
            color = COLOR_DICT[gr]
            for xcol, ycol, row, col in panels_2d:
                fig.add_trace(
                    go.Scatter(
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

    for _chunk_idx, gr, chunk_df in all_chunks:
        color = COLOR_DICT[gr]
        tree_short = chunk_df["tree"].str.split("/").str[-1].str.strip()
        customdata = list(zip(chunk_df["treenum"], tree_short))
        hover = f"{gr}<br>Tree #%{{customdata[0]}}: %{{customdata[1]}}<extra></extra>"
        for xcol, ycol, row, col in panels_2d:
            fig.add_trace(
                go.Scatter(
                    x=chunk_df[xcol], y=chunk_df[ycol],
                    mode="markers",
                    marker=dict(
                        color=color,
                        line=dict(width=0.5, color="rgba(0,0,0,0.5)"),
                    ),
                    opacity=0.9,
                    legendgroup=gr,
                    showlegend=False,
                    hovertemplate=hover,
                    customdata=customdata,
                ),
                row=row, col=col,
            )

    fig.update_layout(
        template="simple_white",
        scene=dict(
            xaxis=dict(title=dict(text=x)),
            yaxis=dict(title=dict(text=y)),
            zaxis=dict(title=dict(text=z)),
        ),
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.15,
            xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=14),
            itemsizing="constant",
        ),
        legend_itemwidth=40,
        uirevision="constant",
        margin=dict(l=2, r=20, t=25, b=10),
    )

    for xcol, ycol, row, col in panels_2d:
        fig.update_xaxes(title_text=xcol, row=row, col=col)
        fig.update_yaxes(title_text=ycol, row=row, col=col)
