"""Shared UI widget builders used by both the panels and the callbacks.

Kept in ``ui/`` (not ``callbacks/``) so the layout panels can import
from it without reaching into the callback layer.
"""

import dash_mantine_components as dmc


SUMMARY_METHOD_MCC = "mcc"
SUMMARY_METHOD_MRHIPSTR = "mrhipstr"


def summary_method_options(*, is_rooted: bool = True) -> list[dict]:
    """Return the shared summary-tree method options for an analysis tab.

    MrHIPSTR is defined only for rooted clades.  Keeping the option visible
    but disabled for an unrooted RF matrix makes that constraint discoverable
    without silently removing the user's previous choice.
    """
    return [
        {"value": SUMMARY_METHOD_MCC, "label": "MCC"},
        {
            "value": SUMMARY_METHOD_MRHIPSTR,
            "label": "MrHIPSTR (mean heights)",
            "disabled": not is_rooted,
        },
    ]


def summary_method_ui_state(
    current_value: object,
    *,
    is_rooted: bool,
) -> tuple[list[dict], str, bool]:
    """Build ``(options, value, tooltip_disabled)`` for a method select."""
    method = str(current_value or SUMMARY_METHOD_MRHIPSTR).strip().lower()
    if method not in {SUMMARY_METHOD_MCC, SUMMARY_METHOD_MRHIPSTR}:
        method = SUMMARY_METHOD_MRHIPSTR
    if not is_rooted and method == SUMMARY_METHOD_MRHIPSTR:
        method = SUMMARY_METHOD_MCC
    return summary_method_options(is_rooted=is_rooted), method, is_rooted


def summary_method_control(select_id: str, tooltip_id: str):
    """Build the compact summary-method control shared by both MDS tabs."""
    selector = dmc.Select(
        id=select_id,
        data=summary_method_options(),
        value=SUMMARY_METHOD_MRHIPSTR,
        allowDeselect=False,
        size="xs",
        w=190,
        className="tt-summary-method-select",
        **{"aria-label": "Summary tree method"},
    )
    return dmc.Group(
        [
            dmc.Text(
                "Method:",
                size="xs",
                fw=500,
                className="tt-summary-method-label",
            ),
            dmc.Tooltip(
                selector,
                id=tooltip_id,
                label=(
                    "MrHIPSTR requires a rooted RF matrix. "
                    "MCC remains available for unrooted matrices."
                ),
                disabled=True,
                withArrow=True,
                position="top",
            ),
        ],
        gap="xs",
        wrap="nowrap",
        className="tt-summary-method-control",
    )


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
    message/stop row. The bar and label use pattern-matching IDs so the
    central reconciler can update whichever banner currently exists without
    naming the absent RF or MDS banner as a Dash callback output.
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
                    id={"type": "compute-progress-bar", "which": which},
                    value=0,
                    size="md",
                    color="blue",
                ),
                dmc.Text(
                    "starting…",
                    size="xs",
                    c="dimmed",
                    id={"type": "compute-progress-label", "which": which},
                ),
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
