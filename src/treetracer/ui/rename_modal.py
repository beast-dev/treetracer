"""The consensus tree rename modal — single dialog driven from three places.

The user can rename a consensus tree from:

* The first View click on an entry whose ``name_user_set`` is still
  False — the click opens this modal prefilled with the auto-name
  instead of going straight to PearTree.
* The pencil icon on each consensus tree list row, for any-time edits.
* Right after a fresh ``View consensus tree`` → compute completes (the new entry
  is by definition name-default, so the modal opens automatically with
  ``after = 'view'`` to chain the open after Save).

Three small stores live alongside the modal:

* ``consensus-tree-rename-state`` — drives WHEN the modal opens, what to prefill,
  and what to do on Save. Shape::

      {uuid: str, name: str, after: "view" | None, n: int}

  ``after == "view"`` means "after Save, open PearTree". The ``n``
  key is just a nonce so repeat triggers (same uuid, same after) still
  count as data changes for Dash.

* ``consensus-tree-peartree-open-store`` — the single sink that opens the
  PearTree viewer. One clientside callback (in
  ``callbacks/rename_consensus_tree.py``) reads this and does the
  ``window.open`` (or ``pywebview.api.open_peartree`` on desktop).
  Replaces the three duplicate clientside callbacks that previously
  lived in ``consensus_tree_list.py``, ``treespace.py``, ``within_run.py``.

* ``consensus-tree-rename-focus-sink`` — a hidden div that absorbs the
  no-update return of the open-focus helper clientside callback (we
  don't want to overwrite any real component's prop just to fire JS).
"""

import dash_mantine_components as dmc
from dash import dcc, html


def _add_rename_modal():
    return html.Div([
        dmc.Modal(
            id="consensus-tree-rename-modal",
            title="Name this consensus tree",
            centered=True,
            size="sm",
            # Above any other dmc.Modal in the app (only one exists
            # today — the About modal — but be explicit).
            zIndex=1100,
            opened=False,
            closeOnEscape=True,
            # The rename is a critical workflow step (peartree won't
            # open until Save fires); a stray click outside the modal
            # shouldn't lose context.
            closeOnClickOutside=False,
            withCloseButton=True,
            children=[
                dmc.Stack([
                    dmc.TextInput(
                        id="consensus-tree-rename-input",
                        label="Name",
                        description=(
                            "Used in the PearTree window title, the "
                            "consensus tree list, and Compare dropdowns."
                        ),
                        placeholder="e.g. run3_burn10",
                        value="",
                        size="sm",
                        # DMC 2.4's TextInput doesn't expose
                        # ``maxLength`` or ``autoFocus`` as props.
                        # Length is enforced server-side by
                        # ``state.rename_consensus_tree`` (≤ 80 chars), and focus
                        # is dropped into the input on every open by
                        # the clientside helper in
                        # ``callbacks/rename_consensus_tree.py``.
                    ),
                    html.Div(
                        id="consensus-tree-rename-error",
                        style={
                            "color": "#c92a2a",
                            "fontSize": "12px",
                            # Reserve vertical space so the dialog
                            # doesn't jump when an error appears.
                            "minHeight": "16px",
                        },
                    ),
                    dmc.Group([
                        dmc.Button(
                            "Cancel",
                            id="consensus-tree-rename-cancel",
                            variant="default",
                            size="sm",
                        ),
                        dmc.Button(
                            # Label is rewritten at open time —
                            # "Save & View" when ``after == 'view'``,
                            # plain "Save" for pencil-rename.
                            "Save",
                            id="consensus-tree-rename-save",
                            variant="filled",
                            color="blue",
                            size="sm",
                        ),
                    ], justify="flex-end", gap="xs"),
                ], gap="xs"),
            ],
        ),
        # ── State + dispatch stores ──────────────────────────────────
        dcc.Store(id="consensus-tree-rename-state"),
        dcc.Store(id="consensus-tree-peartree-open-store"),
        # Hidden no-update sink for the focus + Enter-wiring helper.
        html.Div(id="consensus-tree-rename-focus-sink", style={"display": "none"}),
    ])
