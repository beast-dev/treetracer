import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
