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
    mode_2d = "lines+markers" if show_lines else "markers"
    mode_3d = "lines+markers" if show_lines else "markers"
    for i, gr in enumerate(GROUPS):
        group_data = df[df["group"] == gr]
        fig.add_trace(
            go.Scatter3d(
                x=group_data[x],
                y=group_data[y],
                z=group_data[z],
                mode=mode_3d,
                name=gr,
                showlegend=True,
                marker=dict(color=COLOR_DICT[gr], size=4),
                line=dict(color=COLOR_DICT[gr], width=1),
                legendgroup=gr,
                hovertemplate=f"{gr}<br>Tree: %{{customdata[0]}}<extra></extra>",
                customdata=group_data[["treenum"]].values,
            ),
            row=1,
            col=1,
        )  # scatter 3D
        fig.add_trace(
            go.Scatter(
                x=group_data[x],
                y=group_data[y],
                mode=mode_2d,
                name=gr,
                showlegend=False,  # Show legend for proper sync
                marker=dict(color=COLOR_DICT[gr]),
                line=dict(color=COLOR_DICT[gr], width=1),
                legendgroup=gr,  # Add legend group for synchronization
                hovertemplate=f"{gr}<br>Tree: %{{customdata[0]}}<extra></extra>",
                customdata=group_data[["treenum"]].values,
            ),
            row=1,
            col=2,
        )  # scatter 2D - 1
        fig.add_trace(
            go.Scatter(
                x=group_data[x],
                y=group_data[z],
                mode=mode_2d,
                name=gr,
                showlegend=False,  # Show legend for proper sync
                marker=dict(color=COLOR_DICT[gr]),
                line=dict(color=COLOR_DICT[gr], width=1),
                legendgroup=gr,  # Add legend group for synchronization
                hovertemplate=f"{gr}<br>Tree: %{{customdata[0]}}<extra></extra>",
                customdata=group_data[["treenum"]].values,
            ),
            row=2,
            col=2,
        )  # scatter 2D - 2
        fig.add_trace(
            go.Scatter(
                x=group_data[y],
                y=group_data[z],
                mode=mode_2d,
                name=gr,
                showlegend=False,  # Hide duplicate legends
                marker=dict(color=COLOR_DICT[gr]),
                line=dict(color=COLOR_DICT[gr], width=1),
                legendgroup=gr,  # Add legend group
                hovertemplate=f"{gr}<br>Tree: %{{customdata[0]}}<extra></extra>",
                customdata=group_data[["treenum"]].values,
            ),
            row=3,
            col=2,
        )

    fig.update_layout(
        template="simple_white",
        scene=dict(
            xaxis=dict(title=dict(text=x)),
            yaxis=dict(title=dict(text=y)),
            zaxis=dict(title=dict(text=z)),
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            bgcolor="rgba(0,0,0,0)",
            xanchor="left",
            x=1.02,
        ),
        uirevision="constant",
        margin=dict(l=2, r=20, t=25, b=10),
    )

    fig.update_xaxes(title_text=x, row=1, col=2)
    fig.update_yaxes(title_text=y, row=1, col=2)

    fig.update_xaxes(title_text=x, row=2, col=2)
    fig.update_yaxes(title_text=z, row=2, col=2)

    fig.update_xaxes(title_text=y, row=3, col=2)
    fig.update_yaxes(title_text=z, row=3, col=2)
