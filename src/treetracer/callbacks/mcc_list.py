"""Per-tab MCC list rendering plus the shared View / Delete actions.

The Between-runs and Within-run tabs each show a small ``dmc.Table`` of
the MCC trees registered for the currently-visible MDS view. The rows
have View and Delete buttons; both feed into a single
``mcc-registry-action-store`` so the downstream behaviour
(open peartree, drop the entry) lives in one place per action — not
duplicated across tabs.
"""

from __future__ import annotations

from dash import (
    ALL,
    Input,
    Output,
    State,
    callback,
    callback_context,
    clientside_callback,
    html,
    no_update,
)
import dash_mantine_components as dmc
from dash_iconify import DashIconify

from .. import state


_VIEW_BTN = {"type": "mcc-row-view"}
_DELETE_BTN = {"type": "mcc-row-delete"}


def _row_action_button(*, kind, name, source, color, icon,
                       disabled=False, title=""):
    # ``source`` disambiguates buttons that share a ``name`` across
    # different MCC tables (e.g. the same Between-mode MCC appears in
    # the treespace tab AND the diagnostics tab). Without it, Dash
    # treats the two ActionIcons as duplicate components and silently
    # routes clicks to only one of them, leaving the other table's
    # buttons unresponsive.
    btn = dmc.ActionIcon(
        DashIconify(icon=icon, width=14),
        id={"type": kind, "name": name, "source": source},
        color=color,
        variant="subtle",
        size="sm",
        disabled=disabled,
    )
    if not title:
        return btn
    return dmc.Tooltip(btn, label=title, withArrow=True, position="top")


def _entry_summary_row(entry, *, show_mode=False, source):
    """Render one registry entry as a ``dmc.TableTr`` row.

    By default the mode column is suppressed because the Between /
    Within tab the user is on already disambiguates. Pass
    ``show_mode=True`` for contexts (like the Diagnostics tab) where
    both kinds share one table.

    The log-clade-credibility is intentionally dropped: it's not
    comparable across rows because each MCC is computed against a
    different denominator (the size of the user's selection).
    """
    name = entry.get("name", "")
    n_sel = len(entry.get("selection") or [])
    mt = entry.get("mcc_tree") or {}
    mcc_run = mt.get("group") or "—"
    treenum = mt.get("treenum")
    treenum_text = "—" if treenum is None else str(int(treenum))
    lnp = entry.get("mcc_log_posterior")
    try:
        lnp_text = "—" if lnp is None else f"{float(lnp):.3f}"
    except (TypeError, ValueError):
        lnp_text = "—"
    cached = state.has_cached_mcc_tree(entry.get("uuid", ""))
    cells = [
        dmc.TableTd(name, style={"fontFamily": "monospace"}),
    ]
    if show_mode:
        cells.append(dmc.TableTd(entry.get("mode") or "—"))
    cells.extend([
        dmc.TableTd(mcc_run),
        dmc.TableTd(treenum_text),
        dmc.TableTd(lnp_text),
        dmc.TableTd(f"{n_sel}"),
        dmc.TableTd(
            dmc.Group([
                _row_action_button(
                    kind="mcc-row-view",
                    name=name,
                    source=source,
                    color="violet",
                    icon="tabler:tree",
                    disabled=not cached,
                    title=("View in PearTree" if cached
                           else "MCC was evicted; recompute to view"),
                ),
                _row_action_button(
                    kind="mcc-row-delete",
                    name=name,
                    source=source,
                    color="red",
                    icon="tabler:trash",
                    title="Remove from registry",
                ),
            ], gap=4),
        ),
    ])
    return dmc.TableTr(cells)


def _table_for(entries, *, show_mode=False, source):
    if not entries:
        return None
    rows = [_entry_summary_row(e, show_mode=show_mode, source=source)
            for e in entries]
    headers = [dmc.TableTh("Name")]
    if show_mode:
        headers.append(dmc.TableTh("Mode"))
    headers.extend([
        dmc.TableTh("Run"),
        dmc.TableTh("Tree #"),
        dmc.TableTh("lnP"),
        dmc.TableTh("Selected"),
        dmc.TableTh(""),
    ])
    return dmc.Table(
        [
            dmc.TableThead(dmc.TableTr(headers)),
            dmc.TableTbody(rows),
        ],
        striped=True, highlightOnHover=True, withTableBorder=False,
        verticalSpacing=2, horizontalSpacing=8,
    )


def _filter_for_treespace(registry, selected_key, results):
    """Between tab list: MCCs matching the active MDS result's
    ``source_distmat`` AND ``mode == 'Between'``."""
    if not registry or not selected_key or not results or selected_key not in results:
        return []
    source_distmat = (results[selected_key] or {}).get("source_distmat")
    if not source_distmat:
        return []
    return [e for e in registry
            if e.get("source_distmat") == source_distmat
            and e.get("mode") == "Between"]


def _filter_for_within(registry, selected_key, selected_run, results):
    """Within tab list: matches active matrix + the currently-selected
    run."""
    if (not registry or not selected_key or not selected_run
            or not results or selected_key not in results):
        return []
    source_distmat = (results[selected_key] or {}).get("source_distmat")
    if not source_distmat:
        return []
    return [e for e in registry
            if e.get("source_distmat") == source_distmat
            and e.get("mode") == "Within"
            and e.get("run") == selected_run]


def register_mcc_list_callbacks():
    # Between-runs tab list
    @callback(
        Output("treespace-mcc-list", "children"),
        Output("treespace-mcc-list-paper", "style"),
        Input("mcc-registry-store", "data"),
        Input("treespace-result-select", "value"),
        State("mds-result-store", "data"),
    )
    def render_treespace_mcc_list(registry, selected_key, results):
        entries = _filter_for_treespace(registry, selected_key, results)
        if not entries:
            return html.Div(), {"display": "none"}
        return _table_for(entries, source="treespace"), {}

    # Within-run tab list
    @callback(
        Output("within-run-mcc-list", "children"),
        Output("within-run-mcc-list-paper", "style"),
        Input("mcc-registry-store", "data"),
        Input("within-run-result-select", "value"),
        Input("within-run-run-select", "value"),
        State("mds-result-store", "data"),
    )
    def render_within_run_mcc_list(registry, selected_key, selected_run,
                                   results):
        entries = _filter_for_within(registry, selected_key, selected_run,
                                     results)
        if not entries:
            return html.Div(), {"display": "none"}
        return _table_for(entries, source="within"), {}

    # Pattern-matching: any row View / Delete click → action store
    @callback(
        Output("mcc-registry-action-store", "data", allow_duplicate=True),
        Input({"type": "mcc-row-view",   "name": ALL, "source": ALL}, "n_clicks"),
        Input({"type": "mcc-row-delete", "name": ALL, "source": ALL}, "n_clicks"),
        State("mcc-registry-store", "data"),
        prevent_initial_call=True,
    )
    def emit_row_action(view_clicks, delete_clicks, registry):
        triggered = callback_context.triggered_id
        if not triggered or not isinstance(triggered, dict):
            return no_update
        # The pattern-matching component is restamped on every render,
        # so n_clicks comes back as None for fresh buttons. Bail in
        # that case to avoid acting on the initial render.
        triggered_prop = (callback_context.triggered or [{}])[0].get("value")
        if not triggered_prop:
            return no_update
        action = "view" if triggered.get("type") == "mcc-row-view" else "delete"
        name = triggered.get("name")
        uuid = ""
        for e in registry or []:
            if e.get("name") == name:
                uuid = e.get("uuid", "")
                break
        if not name:
            return no_update
        # Bump nonce so identical click sequences still re-trigger.
        return {
            "action": action,
            "name": name,
            "uuid": uuid,
            "n": (callback_context.triggered or [{}])[0].get("value"),
        }

    # Server-side delete handler — runs whenever the action store says
    # so. View actions are handled clientside (next callback below).
    @callback(
        Output("mcc-registry-store", "data", allow_duplicate=True),
        Input("mcc-registry-action-store", "data"),
        prevent_initial_call=True,
    )
    def handle_delete_action(payload):
        if not payload or payload.get("action") != "delete":
            return no_update
        name = payload.get("name")
        if not name:
            return no_update
        state.delete_mcc(name)
        return state.get_mcc_registry()

    # Clientside View action: open /peartree/<uuid> via the same
    # pywebview / window.open dance the Between-run "View MCC" button
    # already uses.
    clientside_callback(
        """
        function(payload) {
            if (payload && payload.action === 'view' && payload.uuid) {
                const name = payload.name || '';
                if (window.pywebview && window.pywebview.api
                    && window.pywebview.api.open_peartree) {
                    window.pywebview.api.open_peartree(payload.uuid, name);
                } else {
                    const url = '/peartree/' + payload.uuid
                              + '?name=' + encodeURIComponent(name);
                    const features = 'width=1200,height=800,resizable=yes,scrollbars=yes';
                    window.open(url, 'peartree-' + payload.uuid, features);
                }
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("mcc-registry-action-store", "data", allow_duplicate=True),
        Input("mcc-registry-action-store", "data"),
        prevent_initial_call=True,
    )
