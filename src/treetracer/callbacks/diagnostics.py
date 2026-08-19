from dataclasses import dataclass
from functools import partial
from typing import Any

from dash import dcc, html, callback, clientside_callback, Input, Output, State, no_update, ctx, ALL
import dash_mantine_components as dmc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd

from ..background_jobs import JobBusyError, JobRef, JobState, job_manager
from ..logger import add_log
from ..db.tree_service import get_tree_service
from ..ess.rf_trace import find_reference_index
from .. import state
from ..theme import get_template
from ..ui.widgets import stop_button
from . import persistent_worker
from .compute import _get_executor
from .job_reconcile import (
    is_compute_busy,
    terminal_delivery_marker,
    terminal_event_for_job,
)


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

    # Per-group trace + KDE density. ``Scattergl`` so an 8k-tree run
    # doesn't churn out an SVG path with 8k points; ``fill='tozerox'``
    # is supported on Scattergl so the density polygon still renders.
    for group in all_groups:
        gdf = trace_df[trace_df['group'] == group]
        vals = gdf[value_col].values.astype(float)
        fig.add_trace(go.Scattergl(
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
            fig.add_trace(go.Scattergl(
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


@dataclass(frozen=True, slots=True)
class _RfTraceFinalizationContext:
    selected_matrix: str
    names: tuple[str, ...]
    reference_index: int
    reference_name: str
    reference_group: str
    reference_position: str
    burnin: int


def _finalize_rf_trace_job(
    ref: JobRef,
    result: Any,
    *,
    context: _RfTraceFinalizationContext,
) -> dict[str, Any]:
    """Build and store one RF trace while retaining a small terminal event."""
    if not isinstance(result, dict) or not isinstance(
        result.get("distances"), list
    ):
        raise TypeError("RF-trace worker returned an invalid distance row")
    distances = result["distances"]
    if len(distances) != len(context.names):
        raise ValueError(
            "RF-trace worker row length does not match the matrix name index"
        )

    tree_service = get_tree_service()
    tree_service.db_manager.flush()
    all_trees = tree_service.db_manager._trees
    if len(all_trees) > 0:
        name_to_file_source = dict(
            zip(
                all_trees["name"].astype(str).tolist(),
                all_trees["file_source"].astype(str).tolist(),
            )
        )
    else:
        name_to_file_source = {}

    records = []
    group_counts: dict[str, int] = {}
    for index, (tree_name, distance) in enumerate(
        zip(context.names, distances)
    ):
        if index == context.reference_index:
            continue
        group = (
            tree_name.rsplit("/", 1)[0]
            if "/" in tree_name
            else tree_name
        )
        group_counts[group] = group_counts.get(group, 0) + 1
        records.append(
            {
                "rf_distance": int(distance),
                "group": group,
                "name": tree_name,
                "file_source": name_to_file_source.get(
                    tree_name,
                    "(from distance matrix)",
                ),
                "treenum": group_counts[group],
            }
        )

    if not records:
        raise ValueError("No non-reference trees are available for RF Trace")
    trace_df = pd.DataFrame(records)
    figure = _build_rf_trace_fig(
        trace_df,
        context.reference_group,
        context.reference_position,
        burnin=context.burnin,
    )
    state.store_rf_trace_result(
        ref.job_id,
        {
            "records": records,
            "figure": figure.to_dict(),
        },
    )
    elapsed = float(result.get("elapsed", 0.0))
    add_log(
        f"RF Trace computed for {len(records)} trees against "
        f"{context.reference_name!r} in {elapsed:.3f}s."
    )
    return {
        "result_key": ref.job_id,
        "selected_matrix": context.selected_matrix,
        "reference_name": context.reference_name,
        "reference_group": context.reference_group,
        "reference_position": context.reference_position,
        "n_trees": len(records),
        "elapsed": elapsed,
    }

def register_diagnostics_callbacks():

    # ------ Show/hide the RF-dependent diagnostic sections ------
    # Log-Posterior Trace, RF Distance to Reference, and Pseudo-ESS
    # all need a distmat to operate against. Until the user has
    # selected one (or none exists yet), hide them behind a single
    # wrapper div and surface a "no RF distance" placeholder in their
    # place. Toggles are pure ``style.display`` swaps so callbacks
    # inside each section don't have to know about visibility.

    @callback(
        Output("diagnostics-rf-sections", "style"),
        Output("diagnostics-no-rf-placeholder", "style"),
        Input("diagnostics-distmat-select", "value"),
    )
    def toggle_diagnostics_rf_sections(selected_matrix):
        if selected_matrix:
            return {}, {"display": "none"}
        return {"display": "none"}, {"textAlign": "center"}

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
        # Per-group line + KDE on Scattergl so long traces don't drag
        # the SVG layer. The post-burnin rebuild below uses the same
        # treatment.
        for group in groups:
            gdf = trace_df[trace_df['group'] == group]
            vals = gdf['value'].values
            fig.add_trace(go.Scattergl(
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
                fig.add_trace(go.Scattergl(
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
                    fig.add_trace(go.Scattergl(
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
                        fig.add_trace(go.Scattergl(
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
        Output("compute-rf-trace-button", "disabled"),
        Input("diagnostics-distmat-select", "value"),
        Input("rf-reference-group-select", "value"),
        Input("compute-busy-store", "data"),
    )
    def toggle_compute_rf_trace_button(
        selected_matrix,
        reference_group,
        compute_busy,
    ):
        return bool(
            is_compute_busy(compute_busy)
            or not selected_matrix
            or not reference_group
        )

    @callback(
        Output("rf-trace-plot", "children", allow_duplicate=True),
        Output("rf-trace-store", "data", allow_duplicate=True),
        Output("compute-rf-trace-button", "disabled", allow_duplicate=True),
        Output("export-rf-trace-button", "disabled", allow_duplicate=True),
        Output("rf-trace-job-store", "data"),
        Input("compute-rf-trace-button", "n_clicks"),
        State("rf-reference-group-select", "value"),
        State("rf-reference-position-select", "value"),
        State("distmat-store", "data"),
        State("diagnostics-distmat-select", "value"),
        State("rf-burnin-input", "value"),
        prevent_initial_call=True,
    )
    def compute_rf_trace(
        n_clicks,
        ref_group,
        ref_position,
        stored_distmats,
        selected_matrix,
        burnin,
    ):
        """Validate and submit a memory-mapped RF-trace row extraction."""
        if not n_clicks:
            return (no_update,) * 5

        if not ref_group:
            return (
                dmc.Text("Please select a reference group.", c="red"),
                no_update,
                False,
                no_update,
                no_update,
            )

        if not stored_distmats:
            return (
                dmc.Text("Please compute RF distances first (Distances tab).", c="red"),
                no_update, False, no_update, no_update,
            )

        if not selected_matrix or selected_matrix not in stored_distmats:
            return (
                dmc.Text("Please pick an RF matrix at the top of the page.", c="red"),
                no_update, False, no_update, no_update,
            )

        try:
            names = tuple(str(name) for name in state.get_distmat_names(selected_matrix))
            matrix_path = str(state.get_distmat_file_path(selected_matrix))
            reference_index, reference_name = find_reference_index(
                names,
                ref_group,
                ref_position,
            )
        except (KeyError, ValueError) as exc:
            return (
                dmc.Text(str(exc), c="red"),
                no_update,
                False,
                no_update,
                no_update,
            )

        try:
            burnin_int = max(0, int(burnin)) if burnin else 0
        except (ValueError, TypeError):
            burnin_int = 0
        context = _RfTraceFinalizationContext(
            selected_matrix=selected_matrix,
            names=names,
            reference_index=reference_index,
            reference_name=reference_name,
            reference_group=str(ref_group),
            reference_position=str(ref_position),
            burnin=burnin_int,
        )

        try:
            job_ref = job_manager.submit(
                _get_executor(),
                "rf_trace",
                persistent_worker.submit_job,
                "compute_rf_trace",
                matrix_path=matrix_path,
                reference_index=reference_index,
                metadata={
                    "display_name": "RF Trace",
                    "source_distmat": selected_matrix,
                },
                finalizer=partial(
                    _finalize_rf_trace_job,
                    context=context,
                ),
                cancel_exceptions=(persistent_worker.JobCancelled,),
            )
        except JobBusyError as exc:
            msg = (
                f"Another computation ({exc.active.kind.replace('_', ' ').upper()}) "
                "is still finishing. Please wait for it to complete."
            )
            add_log(msg, "WARNING")
            return (
                dmc.Alert(
                    title="Computation already running",
                    children=dmc.Text(msg, size="sm"),
                    color="yellow",
                    variant="light",
                ),
                no_update,
                False,
                no_update,
                no_update,
            )

        spinner = dmc.Group(
            [
                dmc.Loader(size="sm", type="dots"),
                dmc.Text(
                    f"Reading the RF row for {reference_name}…",
                    size="sm",
                    c="dimmed",
                ),
                stop_button("rf-trace"),
            ],
            gap="sm",
        )
        return (
            spinner,
            None,
            True,
            True,
            job_ref.as_dict(),
        )

    @callback(
        Output("rf-trace-plot", "children", allow_duplicate=True),
        Output("rf-trace-store", "data", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
        Output("export-rf-trace-button", "disabled", allow_duplicate=True),
        Output(
            {"type": "compute-terminal-receipt", "kind": "rf-trace"},
            "data",
        ),
        Input("compute-terminal-event-store", "data"),
        Input("rf-trace-job-store", "data"),
        prevent_initial_call=True,
    )
    def render_rf_trace_terminal_event(terminal_event, job_data):
        event = terminal_event_for_job(
            terminal_event,
            job_data,
            expected_kind="rf_trace",
        )
        if event is None:
            return (no_update,) * 5

        ref = JobRef.from_dict(event)
        terminal_state = JobState(str(event["state"]))
        payload = event["payload"]
        first_delivery = int(event["delivery_attempt"]) == 1
        notification = no_update
        store_data = no_update
        export_disabled = True

        if terminal_state is JobState.CANCELLED:
            if first_delivery:
                add_log("RF Trace computation cancelled by user.", "WARNING")
            output = dmc.Alert(
                title="RF Trace computation cancelled",
                children=dmc.Text("Stopped before completion.", size="sm"),
                color="gray",
                variant="light",
            )
        elif terminal_state is JobState.FAILED:
            message = str(payload.get("message", "Unknown error"))
            if first_delivery:
                add_log(f"RF Trace computation failed: {message}", "ERROR")
            output = dmc.Text(
                f"RF Trace computation failed: {message}",
                c="red",
            )
            notification = dmc.Notification(
                title="RF Trace Error",
                message=message,
                color="red",
                action="show",
                autoClose=6000,
                id=f"rf-trace-terminal-{ref.job_id}",
            )
        else:
            cached = state.get_rf_trace_result(payload.get("result_key"))
            if cached is None:
                output = dmc.Text(
                    "RF Trace result is no longer available. Please recompute it.",
                    c="red",
                )
            else:
                store_data = cached["records"]
                output = dcc.Graph(
                    id="rf-trace-graph",
                    figure=cached["figure"],
                    config={"displayModeBar": False},
                )
                export_disabled = False
                notification = dmc.Notification(
                    title="RF Trace Computed",
                    message=(
                        f"Computed RF distances for {payload['n_trees']} trees "
                        f"to the {payload['reference_position']} tree of "
                        f"{payload['reference_group']}."
                    ),
                    color="green",
                    action="show",
                    autoClose=3000,
                    id=f"rf-trace-terminal-{ref.job_id}",
                )

        return (
            output,
            store_data,
            notification,
            export_disabled,
            terminal_delivery_marker(event),
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

    # ------ EXPORT DIAGNOSTICS PLOTS AS SVG ------
    # Client-side render (window.ttExportSvg in assets/export_svg.js) saved via
    # the pywebview ``save_download`` bridge — no Kaleido/Chrome subprocess,
    # which re-launched the app and produced no file inside the Briefcase
    # bundle. These traces are 2D, so the exported SVG is fully vector.
    clientside_callback(
        "function(n){return (n && window.ttExportSvg)"
        " ? window.ttExportSvg('lnl-trace-graph', 'lnP_trace.svg')"
        " : window.dash_clientside.no_update;}",
        Output("notifications-container", "children", allow_duplicate=True),
        Input("export-lnl-trace-button", "n_clicks"),
        prevent_initial_call=True,
    )

    clientside_callback(
        "function(n){return (n && window.ttExportSvg)"
        " ? window.ttExportSvg('rf-trace-graph', 'rf_trace.svg')"
        " : window.dash_clientside.no_update;}",
        Output("notifications-container", "children", allow_duplicate=True),
        Input("export-rf-trace-button", "n_clicks"),
        prevent_initial_call=True,
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
             "label": (
                 f"{k} ({v.get('n_trees', '?')} trees, "
                 f"{'rooted' if v.get('is_rooted', True) else 'unrooted'})"
             )}
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

    # ─── Tree-ESS section ──────────────────────────────────────────────
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
        Input("compute-busy-store", "data"),
    )
    def toggle_compute_pseudo_ess_button(
        selected_matrix,
        checks,
        compute_busy,
    ):
        # Disabled until a matrix is selected AND at least one run is checked.
        if is_compute_busy(compute_busy) or not selected_matrix:
            return True
        if not checks or not any(checks):
            return True
        return False

    @callback(
        Output("pseudo-ess-output", "children", allow_duplicate=True),
        Output("compute-pseudo-ess-button", "disabled", allow_duplicate=True),
        Output("pseudo-ess-job-store", "data"),
        Input("compute-pseudo-ess-button", "n_clicks"),
        State("diagnostics-distmat-select", "value"),
        # Reference-count selection is temporarily deactivated. The submit
        # helper uses 50 reference trees by default.
        # State("ess-n-refs-input", "value"),
        State("ess-burnin-input", "value"),
        State({"type": "ess-run-checkbox", "index": ALL}, "checked"),
        State({"type": "ess-run-checkbox", "index": ALL}, "id"),
        prevent_initial_call=True,
    )
    def compute_pseudo_ess_for_runs(
        n_clicks,
        selected_matrix,
        # n_refs,
        burnin,
        checks,
        ids,
    ):
        """Submit a combined Pseudo-ESS and Fréchet ESS worker job.

        Parent-side: validates input, bins trees per run, applies
        per-chain burn-in, builds the list of slice descriptors the
        worker needs. The actual eigendecomp-heavy ESS compute lives
        in the worker subprocess — see ``ess._subprocess_worker``.

        Returns immediately with a spinner, a disabled originating button,
        and an immutable job identity that wakes the central reconciler.
        """
        from . import pseudo_ess_compute

        if not n_clicks or not selected_matrix:
            return (no_update,) * 3

        ticked = [i["index"] for i, c in zip(ids, checks) if c]
        if not ticked:
            return (dmc.Text("No runs selected.", c="dimmed", size="sm"),
                    no_update, no_update)

        try:
            # Only labels are needed to form per-run row indices. Avoid loading
            # the full n×n matrix into the GUI process; the worker reads it once.
            names = list(state.get_distmat_names(selected_matrix))
        except KeyError:
            return (dmc.Text(f"Matrix {selected_matrix!r} is no longer available.",
                             c="red", size="sm"),
                    no_update, no_update)

        # Bucket row indices by group prefix once (matrix-row order
        # matches MCMC iteration order within each chain).
        group_to_indices = {}
        for i, tree_name in enumerate(names):
            grp = str(tree_name).split("/", 1)[0]
            group_to_indices.setdefault(grp, []).append(i)

        # Reference-count selection is temporarily deactivated. These lines
        # can be restored along with the State/argument above.
        # try:
        #     n_refs_int = int(n_refs) if n_refs else 50
        # except (ValueError, TypeError):
        #     n_refs_int = 50

        try:
            burnin_int = max(0, int(burnin)) if burnin else 0
        except (ValueError, TypeError):
            burnin_int = 0

        requests = []
        all_indices: list[int] = []
        skipped = []
        for grp in ticked:
            idx = group_to_indices.get(grp, [])
            # Burn-in is per chain — first ``burnin_int`` trees of THIS run.
            idx = idx[burnin_int:]
            if len(idx) < 4:
                skipped.append(grp)
                continue
            requests.append({
                "label": grp,
                "indices": idx,
                "burnin_label": str(burnin_int),
            })
            all_indices.extend(idx)

        if len(ticked) - len(skipped) > 1 and all_indices:
            combined_idx = sorted(set(all_indices))
            requests.append({
                "label": "Combined",
                "indices": combined_idx,
                "burnin_label": f"{len(ticked) - len(skipped)}×{burnin_int}",
            })

        if not requests:
            return (dmc.Text(
                "Burn-in leaves fewer than 4 trees per run; nothing to compute.",
                c="dimmed", size="sm",
            ), no_update, no_update)

        # Hand off to the subprocess. The central reconciler publishes the
        # terminal event; pseudo_ess_compute.py replaces the spinner with the
        # matching result table.
        try:
            job_ref = pseudo_ess_compute.submit_pseudo_ess_job(
                distmat_path=str(state.get_distmat_file_path(selected_matrix)),
                names=names,
                requests=requests,
                # Omitted while the control is deactivated; the submit helper
                # supplies its default of 50 reference trees.
                # n_refs=n_refs_int,
                seed=0,
            )
        except JobBusyError as exc:
            msg = (
                f"Another computation ({exc.active.kind.replace('_', ' ').upper()}) "
                "is still finishing. Please wait for it to complete."
            )
            add_log(msg, "WARNING")
            return (
                dmc.Alert(
                    title="Computation already running",
                    children=dmc.Text(msg, size="sm"),
                    color="yellow",
                    variant="light",
                ),
                False,
                no_update,
            )

        spinner = dmc.Group([
            dmc.Loader(size="sm", type="dots"),
            dmc.Text(
                f"Computing Tree-ESS for {len(requests)} row(s)…",
                size="sm", c="dimmed",
            ),
            stop_button("ess"),
        ], gap="sm")

        return spinner, True, job_ref.as_dict()
