"""Per-tab consensus tree list rendering plus the shared View / Delete actions.

The Between-runs and Within-run tabs each show a small ``dmc.Table`` of
the consensus trees registered for the currently-visible MDS view. The rows
have View and Delete buttons; both feed into a single
``consensus-tree-registry-action-store`` so the downstream behaviour
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
from ..icons import icon

from .. import state


_VIEW_BTN = {"type": "consensus-tree-row-view"}
_DELETE_BTN = {"type": "consensus-tree-row-delete"}
# Rename pencil — clicks are handled in callbacks/rename_consensus_tree.py
# (``open_modal_from_pencil``), not by the View/Delete action store.
_RENAME_BTN = {"type": "consensus-tree-row-rename"}


def _uses_compact_registry_table(*, show_mode, source):
    return source in {"treespace", "within", "clade"}


def _row_action_button(*, kind, name, source, color, icon_name,
                       disabled=False, title=""):
    # ``source`` disambiguates buttons that share a ``name`` across
    # different consensus tree tables (e.g. the same Between-mode consensus tree appears in
    # the treespace tab AND the diagnostics tab). Without it, Dash
    # treats the two ActionIcons as duplicate components and silently
    # routes clicks to only one of them, leaving the other table's
    # buttons unresponsive.
    btn = dmc.ActionIcon(
        icon(icon_name, size=14),
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
    comparable across rows because each consensus tree is computed against a
    different denominator (the size of the user's selection).
    """
    name = entry.get("name", "")
    n_sel = len(entry.get("selection") or [])
    mt = entry.get("consensus_tree") or {}
    consensus_tree_run = mt.get("group") or "—"
    treenum = mt.get("treenum")
    treenum_text = "—" if treenum is None else str(int(treenum))
    lnp = entry.get("consensus_tree_log_posterior")
    try:
        lnp_text = "—" if lnp is None else f"{float(lnp):.3f}"
    except (TypeError, ValueError):
        lnp_text = "—"
    cached = state.has_cached_consensus_tree(entry.get("uuid", ""))
    compact_table = _uses_compact_registry_table(
        show_mode=show_mode,
        source=source,
    )
    if compact_table:
        name_cell = dmc.TableTd(
            html.Div(
                name,
                className="tt-consensus-tree-cell-ellipsis tt-consensus-tree-name-text",
                title=name,
            ),
            className="tt-consensus-tree-name-col",
        )
    else:
        name_cell = dmc.TableTd(name, style={"fontFamily": "monospace"})

    cells = [name_cell]
    if show_mode:
        mode_cell_props = {"className": "tt-consensus-tree-mode-col"} if compact_table else {}
        cells.append(dmc.TableTd(entry.get("mode") or "—", **mode_cell_props))

    action_group = dmc.Group([
        _row_action_button(
            kind="consensus-tree-row-rename",
            name=name,
            source=source,
            color="blue",
            icon_name="tabler:pencil",
            title="Rename",
        ),
        _row_action_button(
            kind="consensus-tree-row-view",
            name=name,
            source=source,
            color="violet",
            icon_name="tabler:tree",
            disabled=not cached,
            title=("View in PearTree" if cached
                   else "consensus tree was evicted; recompute to view"),
        ),
        _row_action_button(
            kind="consensus-tree-row-delete",
            name=name,
            source=source,
            color="red",
            icon_name="tabler:trash",
            title="Remove from registry",
        ),
    ], gap=4)

    if compact_table:
        cells.extend([
            dmc.TableTd(
                html.Div(consensus_tree_run, className="tt-consensus-tree-cell-ellipsis", title=consensus_tree_run),
                className="tt-consensus-tree-run-col",
            ),
            dmc.TableTd(treenum_text, className="tt-consensus-tree-col"),
            dmc.TableTd(lnp_text, className="tt-consensus-tree-lnp-col"),
            dmc.TableTd(f"{n_sel}", className="tt-consensus-tree-selected-col"),
            dmc.TableTd(action_group, className="tt-consensus-tree-actions-col"),
        ])
    else:
        cells.extend([
            dmc.TableTd(consensus_tree_run),
            dmc.TableTd(treenum_text),
            dmc.TableTd(lnp_text),
            dmc.TableTd(f"{n_sel}"),
            dmc.TableTd(action_group),
        ])
    return dmc.TableTr(cells)


def _table_for(entries, *, show_mode=False, source):
    if not entries:
        return None
    rows = [_entry_summary_row(e, show_mode=show_mode, source=source)
            for e in entries]
    compact_table = _uses_compact_registry_table(
        show_mode=show_mode,
        source=source,
    )
    if compact_table:
        headers = [dmc.TableTh("Name", className="tt-consensus-tree-name-col")]
        if show_mode:
            headers.append(dmc.TableTh("Mode", className="tt-consensus-tree-mode-col"))
        headers.extend([
            dmc.TableTh("Run", className="tt-consensus-tree-run-col"),
            dmc.TableTh("Tree #", className="tt-consensus-tree-col"),
            dmc.TableTh("lnP", className="tt-consensus-tree-lnp-col"),
            dmc.TableTh("Selected", className="tt-consensus-tree-selected-col"),
            dmc.TableTh("", className="tt-consensus-tree-actions-col"),
        ])
    else:
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
    table_props = {}
    if compact_table:
        table_class = "tt-consensus-tree-registry-table"
        if show_mode:
            table_class += " tt-consensus-tree-registry-table-with-mode"
        table_props = {
            "layout": "fixed",
            "className": table_class,
        }
    return dmc.Table(
        [
            dmc.TableThead(dmc.TableTr(headers)),
            dmc.TableTbody(rows),
        ],
        striped=True, highlightOnHover=True, withTableBorder=False,
        verticalSpacing=2, horizontalSpacing=8,
        **table_props,
    )


def _filter_for_treespace(registry, selected_key, results):
    """Between tab list: consensus trees matching the active MDS result's
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


def register_consensus_tree_list_callbacks():
    # Between-runs tab list
    @callback(
        Output("treespace-consensus-tree-list", "children"),
        Output("treespace-consensus-tree-list-paper", "style"),
        Input("consensus-tree-registry-store", "data"),
        Input("treespace-result-select", "value"),
        State("mds-result-store", "data"),
    )
    def render_treespace_consensus_tree_list(registry, selected_key, results):
        entries = _filter_for_treespace(registry, selected_key, results)
        if not entries:
            return html.Div(), {"display": "none"}
        return _table_for(entries, source="treespace"), {}

    # Within-run tab list
    @callback(
        Output("within-run-consensus-tree-list", "children"),
        Output("within-run-consensus-tree-list-paper", "style"),
        Input("consensus-tree-registry-store", "data"),
        Input("within-run-result-select", "value"),
        Input("within-run-run-select", "value"),
        State("mds-result-store", "data"),
    )
    def render_within_run_consensus_tree_list(registry, selected_key, selected_run,
                                   results):
        entries = _filter_for_within(registry, selected_key, selected_run,
                                     results)
        if not entries:
            return html.Div(), {"display": "none"}
        return _table_for(entries, source="within"), {}

    # Pattern-matching: any row View / Delete click → action store
    @callback(
        Output("consensus-tree-registry-action-store", "data", allow_duplicate=True),
        Input({"type": "consensus-tree-row-view",   "name": ALL, "source": ALL}, "n_clicks"),
        Input({"type": "consensus-tree-row-delete", "name": ALL, "source": ALL}, "n_clicks"),
        State("consensus-tree-registry-store", "data"),
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
        action = "view" if triggered.get("type") == "consensus-tree-row-view" else "delete"
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
        Output("consensus-tree-registry-store", "data", allow_duplicate=True),
        Input("consensus-tree-registry-action-store", "data"),
        prevent_initial_call=True,
    )
    def handle_delete_action(payload):
        if not payload or payload.get("action") != "delete":
            return no_update
        name = payload.get("name")
        if not name:
            return no_update
        state.delete_consensus_tree(name)
        return state.get_consensus_tree_registry()

    # Server-side View dispatcher: if the entry has been renamed by
    # the user (``name_user_set`` is True), open PearTree directly via
    # the shared ``consensus-tree-peartree-open-store`` sink. Otherwise open the
    # rename modal first via ``consensus-tree-rename-state``; the modal's Save
    # handler chains the open after rename. Single clientside
    # ``window.open`` lives in ``callbacks/rename_consensus_tree.py``.
    @callback(
        Output("consensus-tree-peartree-open-store", "data", allow_duplicate=True),
        Output("consensus-tree-rename-state", "data", allow_duplicate=True),
        Input("consensus-tree-registry-action-store", "data"),
        State("consensus-tree-registry-store", "data"),
        prevent_initial_call=True,
    )
    def dispatch_view_action(payload, registry):
        if not payload or payload.get("action") != "view":
            return no_update, no_update
        uuid = payload.get("uuid")
        if not uuid:
            return no_update, no_update
        entry = None
        for e in registry or []:
            if e.get("uuid") == uuid:
                entry = e
                break
        if entry is None:
            return no_update, no_update
        if entry.get("name_user_set"):
            # Direct open — name has been confirmed at least once.
            return (
                {"uuid": uuid, "name": entry.get("name", ""),
                 "n": payload.get("n")},
                no_update,
            )
        # First view: open rename modal with ``after='view'`` so Save
        # chains the peartree open.
        return (
            no_update,
            {"uuid": uuid, "name": entry.get("name", ""),
             "after": "view", "n": payload.get("n")},
        )
