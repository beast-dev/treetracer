"""Shared UI widget builders used by both the panels and the callbacks.

Kept in ``ui/`` (not ``callbacks/``) so the layout panels can import
from it without reaching into the callback layer.
"""

import dash_mantine_components as dmc


def stop_button(which: str):
    """A red "Stop" button for a compute banner / loading overlay.

    Every Stop button across the app carries the pattern-matching id
    ``{"type": "compute-stop", "which": <which>}`` so a single callback
    (``handle_compute_stop`` in ``callbacks/compute.py``) serves all of
    them. They are interchangeable: each kills the persistent worker
    subprocess, interrupting whatever compute is in flight.

    ``which`` only has to be unique per button — ``"rf"``, ``"mds"``,
    ``"consensus-treespace"``, ``"consensus-tree-within"``, ``"ess"``.
    """
    return dmc.Button(
        "Stop",
        id={"type": "compute-stop", "which": which},
        color="red",
        variant="filled",
        size="md",
        # Never shrink — keeps the full "Stop" label visible when it
        # sits next to a long message in a no-wrap flex row.
        style={"flexShrink": 0},
    )


def computing_banner(title: str, message: str, which: str,
                     show_progress: bool = False):
    """A blue "computing…" Alert with an embedded Stop button — the
    shared banner used by the RF and MDS compute callbacks.

    When ``show_progress=True`` the banner stacks an additional
    ``dmc.Progress`` bar and a small status label under the
    message/stop row. The bar's id is ``f"{which}-progress-bar"`` and
    the label's id is ``f"{which}-progress-label"`` so a separate
    polling callback can drive them off a sidecar progress file (see
    ``update_rf_progress`` in ``callbacks/compute.py``).
    """
    # flex:1 + minWidth:0 lets the (often long) message shrink and
    # wrap instead of shoving the Stop button past the Alert's right
    # edge, where it gets clipped.
    message_row = dmc.Group(
        [
            dmc.Text(message, size="sm",
                     style={"flex": 1, "minWidth": 0}),
            stop_button(which),
        ],
        align="center",
        wrap="nowrap",
        gap="md",
    )

    if show_progress:
        body = dmc.Stack(
            [
                message_row,
                dmc.Progress(
                    id=f"{which}-progress-bar",
                    value=0,
                    size="md",
                    color="blue",
                ),
                dmc.Text("starting…", size="xs", c="dimmed",
                         id=f"{which}-progress-label"),
            ],
            gap="xs",
        )
    else:
        body = message_row

    return dmc.Alert(
        body,
        title=title,
        color="blue",
        variant="light",
    )
