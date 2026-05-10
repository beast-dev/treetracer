"""
Global Plotly template state.

Change LIGHT_TEMPLATE / DARK_TEMPLATE here to try different chart themes.

Built-in Plotly templates:
  simple_white, plotly_white, plotly, plotly_dark, ggplot2, seaborn

Usage in callbacks:
    from .theme import set_template
    set_template(template_from_store)
    fig = build_some_figure()
"""

import plotly.io as pio

LIGHT_TEMPLATE = "simple_white"
DARK_TEMPLATE = "plotly_dark"

_current = LIGHT_TEMPLATE


def get_template() -> str:
    return _current


def set_template(t: str | None) -> None:
    global _current
    _current = t or LIGHT_TEMPLATE
    pio.templates.default = _current
