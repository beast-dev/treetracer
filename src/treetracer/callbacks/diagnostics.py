from dash import dcc, callback, Input, Output, State, no_update, ctx
import dash_mantine_components as dmc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd

from ..logger import add_log
from ..db.tree_service import get_tree_service
from ..state import load_distmat
from ._helpers import _save_file_dialog


def _build_rf_trace_fig(trace_df, ref_group, ref_position, burnin=0):
    """Build the RF trace figure with optional burnin zoom."""
    all_groups = sorted(trace_df['group'].unique().tolist())
    colors = px.colors.qualitative.Dark24[:len(all_groups)]
    color_map = dict(zip(all_groups, colors))

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        column_widths=[0.8, 0.2],
        horizontal_spacing=0.02,
    )

    value_col = 'rf_distance'

    for group in all_groups:
        gdf = trace_df[trace_df['group'] == group]
        vals = gdf[value_col].values.astype(float)
        fig.add_trace(go.Scatter(
            x=gdf['treenum'], y=vals,
            mode='lines', name=group,
            line=dict(color=color_map[group], width=1),
            legendgroup=group,
        ), row=1, col=1)

        # KDE: use post-burnin values if burnin is set
        kde_vals = vals
        if burnin > 0:
            post = gdf[gdf['treenum'] > burnin][value_col].values.astype(float)
            if len(post) > 1:
                kde_vals = post

        if len(kde_vals) > 1 and np.std(kde_vals) > 0:
            kde = gaussian_kde(kde_vals)
            y_grid = np.linspace(kde_vals.min(), kde_vals.max(), 200)
            density = kde(y_grid)
            fig.add_trace(go.Scatter(
                x=density, y=y_grid, mode='lines',
                line=dict(color=color_map[group], width=1),
                fill='tozerox', opacity=0.3,
                legendgroup=group, showlegend=False,
            ), row=1, col=2)

    # Apply burnin zoom
    if burnin > 0:
        post_burnin = trace_df[trace_df['treenum'] > burnin]
        if len(post_burnin) > 0:
            max_treenum = int(trace_df['treenum'].max())
            fig.update_xaxes(range=[burnin, max_treenum], row=1, col=1)
            all_post = post_burnin[value_col].values.astype(float)
            ymin, ymax = all_post.min(), all_post.max()
            ypad = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
            fig.update_yaxes(range=[ymin - ypad, ymax + ypad], row=1, col=1)

    ref_label = f"{ref_position} tree of {ref_group}"
    fig.update_layout(
        template="simple_white",
        xaxis_title="Tree number",
        yaxis_title=f"RF distance to {ref_label}",
        xaxis2_title="Density",
        margin=dict(l=60, r=20, t=30, b=40),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def register_diagnostics_callbacks():
    @callback(
        Output("lnl-trace-plot", "children"),
        Output("export-lnl-trace-button", "disabled", allow_duplicate=True),
        Output("lnl-burnin-input", "value"),
        Output("rf-burnin-input", "value"),
        Input("tree-offset-store", "data"),
        Input("lnl-burnin-input", "value"),
        prevent_initial_call=True,
    )
    def update_lnl_trace(stored_summaries, burnin):
        """Render log-likelihood trace plot when trees are loaded/changed."""
        if not stored_summaries:
            return dmc.Text(
                "No trees loaded yet.",
                c="dimmed", size="sm", style={"padding": "20px"},
            ), True, 0, 0

        # When trees are loaded/changed, set burnin to 10% of max per-group tree count
        triggered = ctx.triggered_id
        if triggered == "tree-offset-store":
            max_trees = max(
                s.get("total_trees", 0) for s in stored_summaries.values()
            )
            default_burnin = max(0, int(max_trees * 0.1))
            burnin = default_burnin
            burnin_out = (default_burnin, default_burnin)
        else:
            burnin_out = (no_update, no_update)

        tree_service = get_tree_service()
        file_sources = list(stored_summaries.keys())
        traces = tree_service.get_metadata_traces(file_sources)

        if not traces:
            return (dmc.Text(
                "No log-likelihood data found in tree annotations.",
                c="dimmed", size="sm", style={"padding": "20px"},
            ), True, *burnin_out)

        # Pick first available field (prefer lnP > lnL > posterior > joint > loglikelihood)
        preferred = ['lnP', 'lnL', 'posterior', 'joint', 'loglikelihood']
        field_name = None
        for f in preferred:
            if f in traces:
                field_name = f
                break
        if field_name is None:
            field_name = next(iter(traces))

        trace_df = traces[field_name]
        groups = sorted(trace_df['group'].unique().tolist())
        colors = px.colors.qualitative.Dark24[:len(groups)]
        color_map = dict(zip(groups, colors))

        fig = make_subplots(
            rows=1, cols=2, shared_yaxes=True,
            column_widths=[0.8, 0.2],
            horizontal_spacing=0.02,
        )
        for group in groups:
            gdf = trace_df[trace_df['group'] == group]
            vals = gdf['value'].values
            fig.add_trace(go.Scatter(
                x=gdf['treenum'],
                y=vals,
                mode='lines',
                name=group,
                line=dict(color=color_map[group], width=1),
                legendgroup=group,
            ), row=1, col=1)
            if len(vals) > 1 and np.std(vals) > 0:
                kde = gaussian_kde(vals)
                y_grid = np.linspace(vals.min(), vals.max(), 200)
                density = kde(y_grid)
                fig.add_trace(go.Scatter(
                    x=density, y=y_grid,
                    mode='lines',
                    line=dict(color=color_map[group], width=1),
                    fill='tozerox',
                    opacity=0.3,
                    legendgroup=group,
                    showlegend=False,
                ), row=1, col=2)

        # Apply burnin: filter data for KDE and set x-axis range
        try:
            burnin = int(burnin) if burnin else 0
        except (ValueError, TypeError):
            burnin = 0
        if burnin > 0:
            post_burnin = trace_df[trace_df['treenum'] > burnin]
            if len(post_burnin) > 0:
                # Recompute KDE using only post-burnin values
                fig.data = []  # clear traces, rebuild with filtered KDE
                for group in groups:
                    gdf = trace_df[trace_df['group'] == group]
                    vals = gdf['value'].values
                    # Trace line: show all data (full range)
                    fig.add_trace(go.Scatter(
                        x=gdf['treenum'], y=vals,
                        mode='lines', name=group,
                        line=dict(color=color_map[group], width=1),
                        legendgroup=group,
                    ), row=1, col=1)
                    # KDE: only post-burnin
                    post_vals = gdf[gdf['treenum'] > burnin]['value'].values
                    if len(post_vals) > 1 and np.std(post_vals) > 0:
                        kde = gaussian_kde(post_vals)
                        y_grid = np.linspace(post_vals.min(), post_vals.max(), 200)
                        density = kde(y_grid)
                        fig.add_trace(go.Scatter(
                            x=density, y=y_grid, mode='lines',
                            line=dict(color=color_map[group], width=1),
                            fill='tozerox', opacity=0.3,
                            legendgroup=group, showlegend=False,
                        ), row=1, col=2)

                # Zoom x-axis to post-burnin range
                max_treenum = int(trace_df['treenum'].max())
                fig.update_xaxes(range=[burnin, max_treenum], row=1, col=1)

                # Zoom y-axis to post-burnin value range with 5% padding
                all_post = post_burnin['value'].values
                ymin, ymax = all_post.min(), all_post.max()
                ypad = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
                fig.update_yaxes(range=[ymin - ypad, ymax + ypad], row=1, col=1)

        fig.update_layout(
            template="simple_white",
            xaxis_title="Tree number",
            yaxis_title=field_name,
            xaxis2_title="Density",
            margin=dict(l=60, r=20, t=30, b=40),
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return (dcc.Graph(id="lnl-trace-graph", figure=fig, config={"displayModeBar": False}),
                False, *burnin_out)

    @callback(
        Output("rf-reference-group-select", "data", allow_duplicate=True),
        Output("rf-reference-group-select", "value", allow_duplicate=True),
        Input("tree-offset-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_rf_trace_controls(stored_summaries):
        """Populate group dropdown when trees are loaded."""
        if not stored_summaries:
            return [], None

        # Collect all groups across all loaded files
        all_groups = []
        for summary in stored_summaries.values():
            all_groups.extend(summary.get("groups", []))
        all_groups = sorted(set(all_groups))

        group_options = [{"value": g, "label": g} for g in all_groups]
        # Default to last group alphabetically
        default_group = all_groups[-1] if all_groups else None
        return group_options, default_group

    @callback(
        Output("rf-reference-group-select", "data", allow_duplicate=True),
        Output("rf-reference-group-select", "value", allow_duplicate=True),
        Input("distmat-store", "data"),
        prevent_initial_call=True,
    )
    def populate_groups_from_distmat(stored_distmats):
        """Populate group dropdown from distance matrix tree names when no tree files loaded."""
        if not stored_distmats:
            return no_update, no_update

        # Extract group names from file_breakdown (groups are the keys)
        compact = next(iter(stored_distmats.values()), None)
        if not compact:
            return no_update, no_update

        breakdown = compact.get("file_breakdown", {})
        all_groups = sorted(breakdown.keys())
        if not all_groups:
            return no_update, no_update

        group_options = [{"value": g, "label": g} for g in all_groups]
        default_group = all_groups[-1] if all_groups else None
        return group_options, default_group

    @callback(
        Output("rf-trace-plot", "children"),
        Output("rf-trace-store", "data"),
        Output("notifications-container", "children", allow_duplicate=True),
        Output("export-rf-trace-button", "disabled", allow_duplicate=True),
        Input("compute-rf-trace-button", "n_clicks"),
        State("tree-offset-store", "data"),
        State("rf-reference-group-select", "value"),
        State("rf-reference-position-select", "value"),
        State("distmat-store", "data"),
        prevent_initial_call=True,
    )
    def compute_rf_trace(n_clicks, stored_summaries, ref_group, ref_position, stored_distmats):
        """Compute RF distance of every tree to a single shared reference tree using pre-computed distance matrix."""
        if not n_clicks:
            return no_update, no_update, no_update, no_update

        if not ref_group:
            return (
                dmc.Text("Please select a reference group.", c="red"),
                no_update,
                no_update,
                no_update,
            )

        if not stored_distmats:
            return (
                dmc.Text("Please compute RF distances first (Distances tab).", c="red"),
                no_update,
                no_update,
                no_update,
            )

        # Load matrix from disk (stored as uint16 .npy)
        distmat_key = next(iter(stored_distmats))
        try:
            distmat_names, distmat_matrix = load_distmat(distmat_key)
        except KeyError:
            return (
                dmc.Text("Distance matrix not available. Please recompute RF distances.", c="red"),
                no_update, no_update, no_update,
            )
        # Build name→index lookup for O(1) distance access
        name_to_idx = {n: i for i, n in enumerate(distmat_names)}

        add_log(f"Computing RF trace to {ref_position} tree of group '{ref_group}' (using pre-computed matrix)...")

        # Build ordered tree list: prefer DB if trees are loaded, otherwise derive from distmat names
        tree_service = get_tree_service()
        tree_service.db_manager.flush()
        all_df = tree_service.db_manager._trees

        if len(all_df) > 0:
            # Trees loaded in DB — use DB ordering
            all_df = all_df.sort_values('id')
            tree_names = all_df['name'].tolist()
            tree_groups = all_df['group_name'].tolist()
            tree_file_sources = all_df['file_source'].tolist()
        else:
            # No tree files loaded — derive from distance matrix names
            tree_names = distmat_names
            tree_groups = [
                name.rsplit("/", 1)[0] if "/" in name else name
                for name in tree_names
            ]
            tree_file_sources = ["(from distance matrix)"] * len(tree_names)

        # Filter to reference group and pick first/last
        ref_trees_in_group = [
            name for name, grp in zip(tree_names, tree_groups) if grp == ref_group
        ]

        if not ref_trees_in_group:
            msg = f"No trees found in group '{ref_group}'."
            add_log(msg, "ERROR")
            return dmc.Text(msg, c="red"), no_update, no_update, no_update

        ref_name = ref_trees_in_group[0] if ref_position == "first" else ref_trees_in_group[-1]
        add_log(f"Reference tree: name='{ref_name}' ({ref_position} of group '{ref_group}')")

        if ref_name not in name_to_idx:
            msg = f"Reference tree '{ref_name}' not found in distance matrix."
            add_log(msg, "ERROR")
            return dmc.Text(msg, c="red"), no_update, no_update, no_update

        ref_idx = name_to_idx[ref_name]

        # --- Look up RF distance for every tree from the pre-computed matrix ---
        # Exclude the reference tree itself (RF=0 skews the axes)
        all_records = []
        for tree_name, group, file_source in zip(tree_names, tree_groups, tree_file_sources):
            if tree_name == ref_name:
                continue
            if tree_name not in name_to_idx:
                add_log(f"Tree '{tree_name}' not found in distance matrix, skipping.", "WARNING")
                continue
            tree_idx = name_to_idx[tree_name]
            all_records.append({
                'rf_distance': int(distmat_matrix[ref_idx, tree_idx]),
                'group': group,
                'name': tree_name,
                'file_source': file_source,
            })

        if not all_records:
            return dmc.Text("No trees available for RF trace.", c="dimmed"), no_update, no_update, no_update

        trace_df = pd.DataFrame(all_records)
        trace_df['treenum'] = trace_df.groupby('group').cumcount() + 1

        fig = _build_rf_trace_fig(trace_df, ref_group, ref_position, burnin=0)

        notification = dmc.Notification(
            title="RF Trace Computed",
            message=f"Computed RF distances for {len(all_records)} trees to {ref_position} tree of {ref_group}.",
            color="green",
            action="show",
            autoClose=3000,
            id="rf-trace-notification",
        )

        store_data = trace_df.to_dict("records")

        return (
            dcc.Graph(id="rf-trace-graph", figure=fig, config={"displayModeBar": False}),
            store_data,
            notification,
            False,
        )

    # Re-render RF trace plot when burnin changes
    @callback(
        Output("rf-trace-plot", "children", allow_duplicate=True),
        Input("rf-burnin-input", "value"),
        State("rf-trace-store", "data"),
        State("rf-reference-group-select", "value"),
        State("rf-reference-position-select", "value"),
        prevent_initial_call=True,
    )
    def update_rf_trace_burnin(burnin, store_data, ref_group, ref_position):
        if not store_data:
            return no_update
        try:
            burnin = int(burnin) if burnin else 0
        except (ValueError, TypeError):
            burnin = 0

        trace_df = pd.DataFrame(store_data)
        fig = _build_rf_trace_fig(trace_df, ref_group or "", ref_position or "last", burnin)
        return dcc.Graph(id="rf-trace-graph", figure=fig, config={"displayModeBar": False})

    # ------ EXPORT DIAGNOSTICS PLOTS AS PDF ------

    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("export-lnl-trace-button", "n_clicks"),
        State("lnl-trace-graph", "figure"),
        prevent_initial_call=True,
    )
    def export_lnl_trace_pdf(n_clicks, fig_dict):
        if not n_clicks or not fig_dict:
            return no_update
        path = _save_file_dialog(default_filename="lnl_trace.pdf")
        if not path:
            return no_update
        fig = go.Figure(fig_dict)
        fig.update_layout(template="simple_white")
        fig.write_image(path, width=1200, height=400, scale=2)
        add_log(f"Exported LnL trace plot to {path}")
        return dmc.Notification(
            title="LnL Trace Exported",
            message=f"Saved to {path}",
            color="green",
            action="show",
            autoClose=3000,
            id="export-lnl-trace-notification",
        )

    @callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("export-rf-trace-button", "n_clicks"),
        State("rf-trace-graph", "figure"),
        prevent_initial_call=True,
    )
    def export_rf_trace_pdf(n_clicks, fig_dict):
        if not n_clicks or not fig_dict:
            return no_update
        path = _save_file_dialog(default_filename="rf_trace.pdf")
        if not path:
            return no_update
        fig = go.Figure(fig_dict)
        fig.update_layout(template="simple_white")
        fig.write_image(path, width=1200, height=400, scale=2)
        add_log(f"Exported RF trace plot to {path}")
        return dmc.Notification(
            title="RF Trace Exported",
            message=f"Saved to {path}",
            color="green",
            action="show",
            autoClose=3000,
            id="export-rf-trace-notification",
        )
