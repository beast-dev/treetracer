"""Rename-modal lifecycle callbacks + the single shared peartree opener.

This module is the *only* place that opens the PearTree viewer
(formerly each of mcc_list.py / treespace.py / within_run.py had its
own near-identical clientside ``window.open`` callback). All three
flows now drop a ``{uuid, name}`` payload into ``mcc-peartree-open-store``
and the lone clientside callback here fans it out to either
``pywebview.api.open_peartree`` (desktop) or ``window.open`` (browser).

The rename modal itself sits in ``ui/rename_modal.py``; the
state-shape contract lives there too.

Callback graph (server unless noted):

    Triggers that should rename / view
    ──────────────────────────────────
    * pencil click on a row             →   mcc-rename-state (after=null)
    * row View click on unrenamed entry →   mcc-rename-state (after='view')
    * row View click on renamed entry   →   mcc-peartree-open-store
    * fresh View MCC → compute done     →   mcc-rename-state (after='view')
        (forwarded from ``*-view-mcc-store`` by ``forward_compute_to_modal``)

    Modal lifecycle
    ───────────────
    mcc-rename-state    →  open_modal   →  opens modal, prefills input,
                                           rewrites Save button label
    Cancel click        →  cancel       →  closes modal
    Save click          →  save         →  state.rename_mcc(...) →
                                             refreshes mcc-registry-store,
                                             closes modal,
                                             writes peartree-open-store
                                             (if after=='view'),
                                             emits success toast
    Modal opened=True   →  focus helper (clientside) — drops cursor in
                                                      input + selects
                                                      text + wires Enter
                                                      to Save button.
    mcc-peartree-open  →  window.open / pywebview (clientside)

The two-pass nature (state-store → open_modal) is deliberate: it lets
us reuse the same modal for compute-driven, view-driven, and pencil-
driven flows by varying only the payload, with no per-flow branches
inside the modal's own callbacks.
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
    no_update,
)
import dash_mantine_components as dmc

from .. import state
from ..logger import notif_id


def register_rename_mcc_callbacks():
    # ── Modal open: prefill input, reset error, set Save button label
    # based on whether peartree should open after Save.
    @callback(
        Output("mcc-rename-modal", "opened"),
        Output("mcc-rename-input", "value"),
        Output("mcc-rename-error", "children"),
        Output("mcc-rename-save", "children"),
        Input("mcc-rename-state", "data"),
        prevent_initial_call=True,
    )
    def open_modal(payload):
        if not payload or not payload.get("uuid"):
            return no_update, no_update, no_update, no_update
        after = payload.get("after")
        return (
            True,
            payload.get("name", ""),
            "",
            "Save & View" if after == "view" else "Save",
        )

    # ── After ``View MCC`` → compute completes, the per-tab view-mcc
    # stores fire ``{uuid, name}``. Forward into the rename modal with
    # ``after='view'`` so Save chains the peartree open.
    @callback(
        Output("mcc-rename-state", "data", allow_duplicate=True),
        Input("treespace-view-mcc-store", "data"),
        Input("within-run-view-mcc-store", "data"),
        prevent_initial_call=True,
    )
    def forward_compute_to_modal(treespace_payload, within_payload):
        triggered = (callback_context.triggered or [{}])[0]
        prop_id = triggered.get("prop_id", "")
        payload = (treespace_payload if prop_id.startswith("treespace-")
                   else within_payload)
        if not payload or not payload.get("uuid"):
            return no_update
        return {
            "uuid": payload["uuid"],
            "name": payload.get("name", ""),
            "after": "view",
            # Use the uuid as a stable signature so identical re-fires
            # of the same store payload still register as a change for
            # Dash. (The poll callback writes a single value per
            # compute, so this isn't load-bearing — defensive only.)
            "n": payload["uuid"],
        }

    # ── Pencil-click on any MCC row: open modal in rename-only mode.
    @callback(
        Output("mcc-rename-state", "data", allow_duplicate=True),
        Input({"type": "mcc-row-rename", "name": ALL, "source": ALL}, "n_clicks"),
        State("mcc-registry-store", "data"),
        prevent_initial_call=True,
    )
    def open_modal_from_pencil(_clicks, registry):
        triggered = callback_context.triggered_id
        if not triggered or not isinstance(triggered, dict):
            return no_update
        # Pattern-matched buttons are restamped on every render and
        # initially report n_clicks=None — bail so we don't act on
        # the render itself.
        triggered_prop = (callback_context.triggered or [{}])[0].get("value")
        if not triggered_prop:
            return no_update
        name = triggered.get("name")
        uuid = ""
        for e in registry or []:
            if e.get("name") == name:
                uuid = e.get("uuid", "")
                break
        if not uuid:
            return no_update
        return {
            "uuid": uuid,
            "name": name,
            "after": None,
            "n": triggered_prop,
        }

    # ── Cancel: just close the modal. No state changes.
    @callback(
        Output("mcc-rename-modal", "opened", allow_duplicate=True),
        Input("mcc-rename-cancel", "n_clicks"),
        prevent_initial_call=True,
    )
    def cancel(n_clicks):
        if not n_clicks:
            return no_update
        return False

    # ── Defensive: close the modal if the registry empties under us
    # (the user hit Clear Data with the modal open). Otherwise Save
    # would error with "MCC entry not found" — harmless but ugly.
    # Renames themselves leave the registry non-empty, so this only
    # fires on the genuine clear / last-delete paths.
    @callback(
        Output("mcc-rename-modal", "opened", allow_duplicate=True),
        Input("mcc-registry-store", "data"),
        State("mcc-rename-modal", "opened"),
        prevent_initial_call=True,
    )
    def close_modal_on_registry_clear(registry, opened):
        if opened and not registry:
            return False
        return no_update

    # ── Save: validate, rename, refresh registry, close modal, and
    # optionally trigger the peartree open. Errors stay in-modal (no
    # toast for validation failures — the error text below the input
    # is enough context). Enter on the input fires the same handler
    # via the TextInput's ``n_submit`` counter — keeps "type a name
    # and hit Enter" working without a separate clientside listener.
    @callback(
        Output("mcc-rename-modal", "opened", allow_duplicate=True),
        Output("mcc-rename-error", "children", allow_duplicate=True),
        Output("mcc-registry-store", "data", allow_duplicate=True),
        Output("mcc-peartree-open-store", "data", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
        Input("mcc-rename-save", "n_clicks"),
        Input("mcc-rename-input", "n_submit"),
        State("mcc-rename-input", "value"),
        State("mcc-rename-state", "data"),
        prevent_initial_call=True,
    )
    def save(n_clicks, n_submit, new_name, payload):
        if not (n_clicks or n_submit) or not payload:
            return no_update, no_update, no_update, no_update, no_update
        uuid = payload.get("uuid")
        if not uuid:
            return no_update, no_update, no_update, no_update, no_update

        entry, error = state.rename_mcc(uuid, new_name)
        if entry is None:
            # Keep modal open, surface the error inline so the user can
            # fix the name (most commonly a collision or empty input).
            return no_update, error or "Rename failed.", no_update, no_update, no_update

        registry_payload = state.get_mcc_registry()

        # Pencil-flow gets a small success toast — the modal closing
        # is the only other feedback, and the renamed entry is the
        # whole point. View-flow doesn't need it (PearTree opening
        # is feedback enough).
        notif = no_update
        peartree_payload = no_update
        if payload.get("after") == "view":
            peartree_payload = {
                "uuid": uuid,
                "name": entry["name"],
                # Signature so repeat opens of the same entry register
                # as data changes (otherwise the clientside callback
                # is a no-op on repeat clicks of View → Save). Combine
                # both counters so Enter and click are equally good.
                "n": (n_clicks or 0) + (n_submit or 0),
            }
        else:
            notif = dmc.Notification(
                title="MCC renamed",
                message=f"Now: {entry['name']}",
                color="green",
                action="show",
                autoClose=2500,
                id=notif_id(),
            )

        return False, "", registry_payload, peartree_payload, notif

    # ── Single clientside ``window.open`` for the PearTree viewer.
    # Replaces the three duplicates that used to live in
    # mcc_list.py / treespace.py / within_run.py.
    clientside_callback(
        """
        function(payload) {
            if (payload && payload.uuid) {
                const name = payload.name || '';
                // Match peartree's theme to TreeTracer's current scheme.
                // Mantine sets data-mantine-color-scheme on its root; read
                // it wherever it lives and default to light.
                const schemeEl = document.querySelector('[data-mantine-color-scheme]');
                const theme = (schemeEl
                    && schemeEl.getAttribute('data-mantine-color-scheme')) || 'light';
                if (window.pywebview && window.pywebview.api
                    && window.pywebview.api.open_peartree) {
                    window.pywebview.api.open_peartree(payload.uuid, name, theme);
                } else {
                    const url = '/peartree/' + payload.uuid
                              + '?name=' + encodeURIComponent(name)
                              + '&theme=' + encodeURIComponent(theme);
                    const features = 'width=1200,height=800,resizable=yes,scrollbars=yes';
                    window.open(url, 'peartree-' + payload.uuid, features);
                }
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("mcc-peartree-open-store", "data", allow_duplicate=True),
        Input("mcc-peartree-open-store", "data"),
        prevent_initial_call=True,
    )

    # ── Open-focus helper: every time the modal opens, drop the
    # cursor in the input and pre-select the prefilled text so the
    # user can immediately type a new name or hit Enter to accept the
    # auto-name. Enter itself is wired through dmc.TextInput's
    # ``n_submit`` prop, fed into the Save callback above — no DOM
    # listener needed here.
    clientside_callback(
        """
        function(opened) {
            if (opened) {
                setTimeout(function() {
                    const inp = document.getElementById('mcc-rename-input');
                    if (inp) { inp.focus(); inp.select(); }
                }, 100);
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("mcc-rename-focus-sink", "children"),
        Input("mcc-rename-modal", "opened"),
        prevent_initial_call=True,
    )
