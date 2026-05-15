"""Offline-bundled tabler icons.

The previous setup used ``dash-iconify``, which fetches every icon's
SVG from ``api.iconify.design`` at runtime. That breaks offline use
(no internet, corporate firewall, iconify down) and is racy on first
launch in pywebview's WKWebView, which gave us empty icons on a fair
fraction of launches.

We ship the ~11 unique icons we use as SVGs under
``assets/icons/`` and render them via a CSS ``mask-image`` so that
``currentColor`` keeps theming them (sun/moon dark-mode toggle,
ActionIcon ``color=`` prop, etc.).

Add a new icon:

1. ``curl -o src/treetracer/assets/icons/tabler-<name>.svg \\
        https://api.iconify.design/tabler/<name>.svg``
2. Reference it as ``icon("tabler:<name>", size=...)``.
"""

from __future__ import annotations

from dash import html


def _asset_url(name: str) -> str:
    """e.g. ``"tabler:moon"`` -> ``"/assets/icons/tabler-moon.svg"``.

    Dash mounts assets at ``/assets/`` by default. We don't run with a
    URL prefix so the hardcoded path is fine; if that ever changes,
    swap to ``dash.get_app().get_asset_url(...)``.
    """
    set_name, icon_name = name.split(":", 1)
    return f"/assets/icons/{set_name}-{icon_name}.svg"


def icon(name: str, size: int = 18, **kwargs):
    """Render a bundled tabler icon as an inline ``<span>`` whose
    foreground is masked to the SVG shape.

    ``background-color: currentColor`` lets the icon pick up its
    parent's text colour, so Mantine's ``color=`` prop and the
    dark-mode toggle still recolour the icon without any extra wiring.
    """
    extra_style = kwargs.pop("style", {}) or {}
    style = {
        "display": "inline-block",
        "width": f"{size}px",
        "height": f"{size}px",
        "backgroundColor": "currentColor",
        "WebkitMaskImage": f"url({_asset_url(name)})",
        "maskImage": f"url({_asset_url(name)})",
        "WebkitMaskRepeat": "no-repeat",
        "maskRepeat": "no-repeat",
        "WebkitMaskSize": "contain",
        "maskSize": "contain",
        "WebkitMaskPosition": "center",
        "maskPosition": "center",
        **extra_style,
    }
    return html.Span(style=style, **kwargs)
