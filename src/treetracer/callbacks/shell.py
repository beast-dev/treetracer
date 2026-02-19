from dash import html, callback, Input, Output, State, no_update
import dash_mantine_components as dmc

from ..logger import add_log, get_logs, clear_logs


def register_shell_callbacks():
    # Sidebar collapse callback
    @callback(
        Output("appshell", "navbar"),
        Input("burger", "opened"),
        State("appshell", "navbar"),
    )
    def toggle_navbar(opened, navbar):
        navbar["collapsed"] = {"mobile": not opened}
        return navbar

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
