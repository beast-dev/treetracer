import dash_mantine_components as dmc
from dash import Dash
from .ui import add_header, add_navbar, add_main_body, add_footer
from .callbacks import register_callbacks
import sys
import logging
import threading


def create_dash_app():
    app = Dash(
        __name__, external_stylesheets=dmc.styles.ALL, suppress_callback_exceptions=True
    )

    layout = dmc.AppShell(
        [
            add_header(),
            add_navbar(),
            add_main_body(),
            add_footer(),
        ],
        header={"height": 60},
        footer={"height": 0},
        navbar={
            "width": 300,
            "breakpoint": "sm",
            "collapsed": {"mobile": True},
        },
        padding="md",
        id="appshell",
    )
    app.layout = dmc.MantineProvider(
        [
            dmc.NotificationProvider(position="top-right"),
            layout,
        ],
    )
    return app


def main():
    """Main entry point for the TreeTracer application."""
    import webview

    try:
        app = create_dash_app()
        register_callbacks(app)

        # Suppress Flask's per-request logging (noisy with polling intervals)
        logging.getLogger("werkzeug").setLevel(logging.WARNING)

        # Start Dash server in a background thread (no debug/reloader)
        server_thread = threading.Thread(
            target=lambda: app.run(host="127.0.0.1", port=8050, debug=False),
            daemon=True,
        )
        server_thread.start()

        # Wait for server to be ready
        import time
        import urllib.request
        for _ in range(30):
            try:
                urllib.request.urlopen("http://127.0.0.1:8050/", timeout=1)
                break
            except Exception:
                time.sleep(0.5)

        # Open native desktop window
        webview.create_window("TreeTracer", "http://127.0.0.1:8050/", width=1600, height=900)
        webview.start()

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
