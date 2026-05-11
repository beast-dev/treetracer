from dash import dcc, html, callback, Input, Output, State, no_update, ctx, ALL
import dash_mantine_components as dmc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd

from ..logger import add_log, notif_id
from ..db.tree_service import get_tree_service
from ..ess.rf_trace import compute_rf_trace_data
from ..ess import compute_pseudo_ess
from ..theme import get_template
from .. import state
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
        template=get_template(),
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
        Input("diagnostics-distmat-select", "value"),
        Input("plotly-template-store", "data"),
        State("distmat-store", "data"),
        prevent_initial_call=True,
    )
    def update_lnl_trace(stored_summaries, burnin, selected_matrix,
                         _template, distmat_data):
        """Render log-posterior trace plot for the SELECTED RF matrix.

        The Diagnostics tab is unified around the header RF-matrix
        selector — every section, including this one, conditions on
        that pick. With no matrix selected the plot shows a hint and
        nothing else; once a matrix is chosen, traces are restricted
        to the *exact* set of trees that went into it (by tree-name
        match against the matrix's stored ``names`` list — see
        ``state.get_distmat_names``). This keeps the lnP plot in lock-
        step with the matrix's downsample.

        The ``plotly-template-store`` Input is unused inside the
        function body — it's only there so dark-mode toggles trigger a
        re-render, which then re-reads ``get_template()`` at figure
        build time.
        """
        if not stored_summaries:
            return dmc.Text(
                "No trees loaded yet.",
                c="dimmed", size="sm", style={"padding": "20px"},
            ), True, 0, 0

        if not selected_matrix or not distmat_data or selected_matrix not in distmat_data:
            return (
                dmc.Text(
                    "Pick an RF matrix above to see the log-posterior "
                    "trace for the trees that went into it.",
                    c="dimmed", size="sm", style={"padding": "20px"},
                ),
                True, no_update, no_update,
            )

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
        all_file_sources = list(stored_summaries.keys())
        try:
            tree_names = list(state.get_distmat_names(selected_matrix))
        except KeyError:
            return (
                dmc.Text(
                    f"Matrix {selected_matrix!r} is no longer available.",
                    c="red", size="sm", style={"padding": "20px"},
                ),
                True, no_update, no_update,
            )
        traces = tree_service.get_metadata_traces(
            file_sources=all_file_sources,
            tree_names=tree_names,
        )

        if not traces:
            return (dmc.Text(
                "No log-posterior data found in tree annotations.",
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
            template=get_template(),
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
        Input("diagnostics-distmat-select", "value"),
        State("distmat-store", "data"),
        prevent_initial_call=True,
    )
    def populate_groups_from_distmat(selected_matrix, stored_distmats):
        """Populate the reference-group dropdown from the SELECTED RF matrix.

        Driven by the shared header selector so the RF-trace section
        only offers groups that actually live in the matrix the user is
        currently inspecting.
        """
        if not stored_distmats or not selected_matrix:
            return no_update, no_update

        compact = stored_distmats.get(selected_matrix)
        if not compact:
            return no_update, no_update

        groups_per_file = compact.get("groups_per_file", {})
        all_groups = sorted(set(g for groups in groups_per_file.values() for g in groups))
        if not all_groups:
            # Fallback: use file_breakdown keys
            all_groups = sorted(compact.get("file_breakdown", {}).keys())
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
        State("diagnostics-distmat-select", "value"),
        State("rf-burnin-input", "value"),
        prevent_initial_call=True,
    )
    def compute_rf_trace(n_clicks, stored_summaries, ref_group, ref_position,
                         stored_distmats, selected_matrix, burnin):
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
                no_update, no_update, no_update,
            )

        if not selected_matrix or selected_matrix not in stored_distmats:
            return (
                dmc.Text("Please pick an RF matrix at the top of the page.", c="red"),
                no_update, no_update, no_update,
            )

        result, ref_name = compute_rf_trace_data(selected_matrix, ref_group, ref_position)

        # If result is a string, it's an error message
        if isinstance(result, str):
            return dmc.Text(result, c="red"), no_update, no_update, no_update

        trace_df = result

        try:
            burnin = int(burnin) if burnin else 0
        except (ValueError, TypeError):
            burnin = 0
        fig = _build_rf_trace_fig(trace_df, ref_group, ref_position, burnin=burnin)

        notification = dmc.Notification(
            title="RF Trace Computed",
            message=f"Computed RF distances for {len(trace_df)} trees to {ref_position} tree of {ref_group}.",
            color="green",
            action="show",
            autoClose=3000,
            id=notif_id(),
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
        Input("plotly-template-store", "data"),
        State("rf-trace-store", "data"),
        State("rf-reference-group-select", "value"),
        State("rf-reference-position-select", "value"),
        prevent_initial_call=True,
    )
    def update_rf_trace_burnin(burnin, _, store_data, ref_group, ref_position):
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
        path = _save_file_dialog(default_filename="lnP_trace.pdf")
        if not path:
            return no_update
        fig = go.Figure(fig_dict)
        fig.update_layout(template=get_template())
        fig.write_image(path, width=1200, height=400, scale=2)
        add_log(f"Exported lnP trace plot to {path}")
        return dmc.Notification(
            title="lnP Trace Exported",
            message=f"Saved to {path}",
            color="green",
            action="show",
            autoClose=3000,
            id=notif_id(),
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
        fig.update_layout(template=get_template())
        fig.write_image(path, width=1200, height=400, scale=2)
        add_log(f"Exported RF trace plot to {path}")
        return dmc.Notification(
            title="RF Trace Exported",
            message=f"Saved to {path}",
            color="green",
            action="show",
            autoClose=3000,
            id=notif_id(),
        )

    # ─── Shared Diagnostics RF Matrix selector ─────────────────────────
    # One dropdown at the top of the tab feeds every section below.
    # `update_lnl_trace`, the RF-trace callbacks, and the Pseudo-ESS
    # callbacks all read this id.
    @callback(
        Output("diagnostics-distmat-select", "data"),
        Output("diagnostics-distmat-select", "value"),
        Output("diagnostics-distmat-info", "children"),
        Input("distmat-store", "data"),
        State("diagnostics-distmat-select", "value"),
    )
    def populate_diagnostics_distmat_select(distmat_data, current_value):
        if not distmat_data:
            return [], None, ""
        options = [
            {"value": k,
             "label": f"{k} ({v.get('n_trees', '?')} trees)"}
            for k, v in distmat_data.items()
        ]
        # Keep the user's pick if it's still around; otherwise default to
        # the most recently registered matrix.
        new_value = (
            current_value
            if current_value and current_value in distmat_data
            else list(distmat_data.keys())[-1]
        )
        meta = distmat_data.get(new_value, {})
        groups_per_file = meta.get("groups_per_file", {})
        n_runs = len({g for groups in groups_per_file.values() for g in groups})
        info = dmc.Group([
            dmc.Badge(f"{meta.get('n_trees', '?')} trees",
                      variant="light", color="grape", size="sm"),
            dmc.Badge(f"{n_runs} runs",
                      variant="light", color="teal", size="sm"),
        ], gap="xs")
        return options, new_value, info

    # ─── MCC trees registered for the selected matrix ──────────────────
    # Renders a panel at the bottom of the Diagnostics tab listing the
    # MCC trees registered against the currently-selected RF matrix,
    # split into "Between-runs" and "Within-run" sub-tables. Hidden
    # when no MCCs match the active matrix. The Tab itself decides
    # which subset is interesting; here we surface both since the user
    # is viewing the matrix as a whole. Follow-up PRs can wire
    # post-processing actions onto a clicked row.
    @callback(
        Output("diagnostics-mcc-list", "children"),
        Output("diagnostics-mcc-paper", "style"),
        Input("mcc-registry-store", "data"),
        Input("diagnostics-distmat-select", "value"),
    )
    def render_diagnostics_mcc_panel(registry, selected_matrix):
        from .mcc_list import _table_for
        if not registry or not selected_matrix:
            return html.Div(), {"display": "none"}
        matched = [e for e in registry
                   if e.get("source_distmat") == selected_matrix]
        if not matched:
            return html.Div(), {"display": "none"}
        return dmc.Stack([
            dmc.Title(f"MCC trees for {selected_matrix}", order=5),
            _table_for(matched, show_mode=True),
        ], gap="sm"), {}

    # ─── Pseudo-ESS section ────────────────────────────────────────────
    # Two callbacks own the per-run table + Compute button. Both react
    # to ``diagnostics-distmat-select`` (the shared header dropdown).

    @callback(
        Output("ess-runs-table", "children"),
        Input("diagnostics-distmat-select", "value"),
    )
    def render_ess_runs_table(selected_matrix):
        if not selected_matrix:
            return html.Div(
                dmc.Text(
                    "Pick an RF matrix above to list its runs.",
                    c="dimmed", size="sm",
                ),
                style={"padding": "10px"},
            )

        # Server-side helper walks the matrix's tree-name list and
        # buckets by group prefix. Returns [(group, count), ...].
        groups = state.get_distmat_groups_with_counts(selected_matrix)
        if not groups:
            return dmc.Text(
                f"Matrix {selected_matrix!r} has no recognisable runs.",
                c="dimmed", size="sm",
            )

        rows = []
        for group, count in groups:
            rows.append(
                dmc.TableTr([
                    dmc.TableTd(group),
                    dmc.TableTd(str(count)),
                    dmc.TableTd(
                        dmc.Checkbox(
                            id={"type": "ess-run-checkbox", "index": group},
                            checked=True,
                        )
                    ),
                ])
            )

        return dmc.Table(
            [
                dmc.TableThead(
                    dmc.TableTr([
                        dmc.TableTh("Run"),
                        dmc.TableTh("Trees"),
                        dmc.TableTh("Select"),
                    ])
                ),
                dmc.TableTbody(rows),
            ],
            striped=True,
            highlightOnHover=True,
            withTableBorder=True,
            withColumnBorders=True,
        )

    @callback(
        Output("compute-pseudo-ess-button", "disabled"),
        Input("diagnostics-distmat-select", "value"),
        Input({"type": "ess-run-checkbox", "index": ALL}, "checked"),
    )
    def toggle_compute_pseudo_ess_button(selected_matrix, checks):
        # Disabled until a matrix is selected AND at least one run is checked.
        if not selected_matrix:
            return True
        if not checks or not any(checks):
            return True
        return False

    @callback(
        Output("pseudo-ess-output", "children"),
        Input("compute-pseudo-ess-button", "n_clicks"),
        State("diagnostics-distmat-select", "value"),
        State("ess-n-refs-input", "value"),
        State("ess-burnin-input", "value"),
        State({"type": "ess-run-checkbox", "index": ALL}, "checked"),
        State({"type": "ess-run-checkbox", "index": ALL}, "id"),
        prevent_initial_call=True,
    )
    def compute_pseudo_ess_for_runs(n_clicks, selected_matrix, n_refs, burnin, checks, ids):
        """Compute Pseudo-ESS per ticked run + an optional Combined row.

        For each ticked run we slice the cached RF distance matrix to the
        rows/cols of trees whose name has the ``"<run>/"`` prefix
        (dropping the first ``burnin`` of those rows), then feed that
        submatrix to :func:`treetracer.ess.compute_pseudo_ess`. Burn-in
        is applied *per run* so chains of different lengths don't get
        clipped against a global tree-index threshold. If more than one
        run is ticked we also compute the same diagnostic on the union
        of their (post-burn-in) tree indices — the "Combined" row.
        """
        if not n_clicks or not selected_matrix:
            return no_update

        ticked = [i["index"] for i, c in zip(ids, checks) if c]
        if not ticked:
            return dmc.Text(
                "No runs selected.", c="dimmed", size="sm",
            )

        try:
            names, distmat = state.load_distmat(selected_matrix)
        except KeyError:
            return dmc.Text(
                f"Matrix {selected_matrix!r} is no longer available.",
                c="red", size="sm",
            )

        # Bucket row indices by group prefix once (matrix-row order
        # matches MCMC iteration order within each chain).
        group_to_indices = {}
        for i, tree_name in enumerate(names):
            grp = str(tree_name).split("/", 1)[0]
            group_to_indices.setdefault(grp, []).append(i)

        try:
            n_refs_int = int(n_refs) if n_refs else 100
        except (ValueError, TypeError):
            n_refs_int = 100

        try:
            burnin_int = max(0, int(burnin)) if burnin else 0
        except (ValueError, TypeError):
            burnin_int = 0

        # Stoplight thresholds match the Lanfear paper's rough rule of
        # thumb: <100 is unreliable, <200 is borderline, ≥200 is the
        # "you can trust this" zone.
        def _ess_cell(v):
            if np.isnan(v):
                return dmc.TableTd("—")
            if v < 100:
                color = "red"
            elif v < 200:
                color = "orange"
            else:
                color = "green"
            return dmc.TableTd(
                dmc.Text(f"{v:.1f}", c=color, fw=600, span=True)
            )

        def _row_for(label, indices, burnin_label):
            sub = distmat[np.ix_(indices, indices)]
            res = compute_pseudo_ess(sub, n_refs=n_refs_int, seed=0)
            valid = res["ess_values"][~np.isnan(res["ess_values"])]
            if valid.size:
                mn = float(valid.min())
                q1, q2, q3 = np.quantile(valid, [0.25, 0.5, 0.75])
                mx = float(valid.max())
            else:
                mn = q1 = q2 = q3 = mx = float("nan")

            return dmc.TableTr([
                dmc.TableTd(label),
                dmc.TableTd(str(len(indices))),
                dmc.TableTd(burnin_label),
                _ess_cell(mn),
                _ess_cell(q1),
                _ess_cell(q2),
                _ess_cell(q3),
                _ess_cell(mx),
                dmc.TableTd(str(res["n_refs_used"])),
            ])

        rows = []
        all_indices = []
        skipped = []
        for grp in ticked:
            idx = group_to_indices.get(grp, [])
            # Burn-in is per chain — first ``burnin_int`` trees of THIS run.
            idx = idx[burnin_int:]
            if len(idx) < 4:
                skipped.append(grp)
                continue
            rows.append(_row_for(grp, idx, str(burnin_int)))
            all_indices.extend(idx)

        if len(ticked) - len(skipped) > 1 and all_indices:
            # Sort to preserve MCMC order across the union — important so
            # the autocorrelation in each reference's RF trace is meaningful.
            combined_idx = sorted(set(all_indices))
            # Per-run burn-in was already applied before union, so the
            # Combined label reads "Nx<burnin>" to make clear it isn't a
            # single global cut.
            rows.append(_row_for(
                "Combined", combined_idx,
                f"{len(ticked) - len(skipped)}×{burnin_int}",
            ))

        if not rows:
            msg = "Burn-in leaves fewer than 4 trees per run; nothing to compute."
            return dmc.Text(msg, c="dimmed", size="sm")

        return dmc.Table(
            [
                dmc.TableThead(
                    dmc.TableTr([
                        dmc.TableTh("Run"),
                        dmc.TableTh("Trees"),
                        dmc.TableTh("Burn-in"),
                        dmc.TableTh("Min"),
                        dmc.TableTh("Q1"),
                        dmc.TableTh("Q2 (median)"),
                        dmc.TableTh("Q3"),
                        dmc.TableTh("Max"),
                        dmc.TableTh("# refs"),
                    ])
                ),
                dmc.TableTbody(rows),
            ],
            striped=True,
            highlightOnHover=True,
            withTableBorder=True,
            withColumnBorders=True,
        )
