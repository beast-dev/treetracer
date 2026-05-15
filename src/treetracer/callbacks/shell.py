from dash import html, callback, Input, Output, State, no_update
import dash_mantine_components as dmc

from ..logger import get_logs, clear_logs
from ..theme import LIGHT_TEMPLATE, DARK_TEMPLATE, set_template

_TEMPLATES = {"light": LIGHT_TEMPLATE, "dark": DARK_TEMPLATE}


def register_shell_callbacks():

    # ------ DARK MODE TOGGLE ------
    # Flip color scheme, swap icon, update Plotly template store.
    # To change the dark/light chart theme, edit LIGHT_TEMPLATE / DARK_TEMPLATE in theme.py.

    @callback(
        Output("mantine-provider", "forceColorScheme"),
        Output("dark-mode-icon", "style"),
        Output("plotly-template-store", "data"),
        Input("dark-mode-toggle", "n_clicks"),
        State("mantine-provider", "forceColorScheme"),
        State("dark-mode-icon", "style"),
        prevent_initial_call=True,
    )
    def toggle_dark_mode(n_clicks, current_scheme, current_style):
        new_scheme = "light" if current_scheme == "dark" else "dark"
        icon_name = "tabler:sun" if new_scheme == "dark" else "tabler:moon"
        url = f"/assets/icons/{icon_name.replace(':', '-')}.svg"
        new_style = {
            **(current_style or {}),
            "WebkitMaskImage": f"url({url})",
            "maskImage": f"url({url})",
        }
        template = _TEMPLATES[new_scheme]
        set_template(template)
        return new_scheme, new_style, template

    # ------ SIDEBAR TOGGLE ------

    @callback(
        Output("navbar", "style"),
        Output("sidebar-visible", "data"),
        Output("appshell", "navbar"),
        Input("sidebar-toggle", "n_clicks"),
        State("sidebar-visible", "data"),
        prevent_initial_call=True,
    )
    def toggle_sidebar(n_clicks, is_visible):
        if n_clicks:
            new_state = not is_visible
            if new_state:
                style = {}
                navbar = {"width": 300, "breakpoint": "sm", "collapsed": {"mobile": True}}
            else:
                style = {"display": "none"}
                navbar = {"width": 0, "breakpoint": "sm", "collapsed": {"mobile": True}}
            return style, new_state, navbar
        return no_update, no_update, no_update

    # ------ AUTO-COLLAPSE SIDEBAR AFTER RF COMPLETES ------
    # Once an RF distance matrix is registered (``distmat-store`` gains
    # at least one entry), reclaim the sidebar's 300 px for the main
    # panel — the user is done with uploads/compute and now needs room
    # to read plots. No-op if the sidebar is already collapsed, or if
    # ``distmat-store`` is empty (e.g. just after Clear Data).

    @callback(
        Output("navbar", "style", allow_duplicate=True),
        Output("sidebar-visible", "data", allow_duplicate=True),
        Output("appshell", "navbar", allow_duplicate=True),
        Input("distmat-store", "data"),
        State("sidebar-visible", "data"),
        prevent_initial_call=True,
    )
    def auto_collapse_after_rf(distmat_data, is_visible):
        if not distmat_data or not is_visible:
            return no_update, no_update, no_update
        return (
            {"display": "none"},
            False,
            {"width": 0, "breakpoint": "sm", "collapsed": {"mobile": True}},
        )

    # ------ ABOUT MODAL CALLBACK ------

    @callback(
        Output("about-modal", "opened"),
        Input("about-modal-button", "n_clicks"),
        State("about-modal", "opened"),
        prevent_initial_call=True,
    )
    def toggle_about_modal(n_clicks, opened):
        if n_clicks:
            return not opened
        return no_update

    # ------ LOG PANEL CALLBACKS ------

    @callback(
        Output("log-footer", "style"),
        Output("log-panel-visible", "data"),
        Output("appshell", "footer"),
        Input("log-toggle-button", "n_clicks"),
        State("log-panel-visible", "data"),
        prevent_initial_call=True,
    )
    def toggle_log_panel(n_clicks, is_visible):
        if n_clicks:
            new_state = not is_visible
            style = {"display": "block"} if new_state else {"display": "none"}
            footer = {"height": 250} if new_state else {"height": 0}
            return style, new_state, footer
        return no_update, no_update, no_update

    @callback(
        Output("log-content", "children"),
        Input("log-poll-interval", "n_intervals"),
        State("log-panel-visible", "data"),
    )
    def update_log_display(n_intervals, is_visible):
        if not is_visible:
            return no_update
        logs = get_logs()
        if not logs:
            return dmc.Text("No log entries yet.", c="dimmed", size="sm",
                            style={"padding": "8px"})
        level_colors = {
            "INFO": "#58a6ff",
            "WARNING": "#d29922",
            "ERROR": "#f85149",
        }
        elements = []
        for entry in logs:
            color = level_colors.get(entry["level"], "#8b949e")
            elements.append(
                html.Div([
                    html.Span(f"[{entry['timestamp']}] ",
                              style={"color": "#8b949e"}),
                    html.Span(f"{entry['level']}: ",
                              style={"color": color, "fontWeight": "bold"}),
                    html.Span(entry["message"],
                              style={"color": "#c9d1d9"}),
                ], style={"marginBottom": "2px"})
            )
        return elements

    @callback(
        Output("log-content", "children", allow_duplicate=True),
        Input("log-clear-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_log_display(n_clicks):
        if n_clicks:
            clear_logs()
            return dmc.Text("No log entries yet.", c="dimmed", size="sm",
                            style={"padding": "8px"})
        return no_update
