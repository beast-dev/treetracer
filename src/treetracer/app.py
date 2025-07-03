import dash_mantine_components as dmc
from dash import Dash, _dash_renderer
from .ui import add_header, add_navbar, add_main_body
from .callbacks import register_callbacks
import sys
import webbrowser
import threading


_dash_renderer._set_react_version("18.2.0")


def create_dash_app():
    app = Dash(
        __name__, external_stylesheets=dmc.styles.ALL, suppress_callback_exceptions=True
    )

    layout = dmc.AppShell(
        [
            add_header(),
            add_navbar(),
            add_main_body(),
        ],
        header={"height": 60},
        # footer={"height": 60},
        navbar={
            "width": 300,
            "breakpoint": "sm",
            "collapsed": {"mobile": True},
        },
        padding="md",
        id="appshell",
    )
    app.layout = dmc.MantineProvider(layout)
    return app


def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")


def main():
    """
    Main entry point for the TreeTracer application.
    """
    try:
        app = create_dash_app()
        register_callbacks(app)
        threading.Timer(1.0, open_browser).start()

        # VERY IMPORTANT: This call starts the server and keeps the process running.
        # Use debug=True for development. Set host='0.0.0.0' if you need to access
        # it from other devices on your network (e.g., Docker, WSL).
        app.run(debug=False, host="127.0.0.1", port=8050)

    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)  # Prints the full traceback
    return 0


if __name__ == "__main__":
    sys.exit(main())
