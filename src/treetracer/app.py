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
    """Main entry point for the TreeTracer application.

    Usage:
        uv run treetracer             # native desktop window (default)
        uv run treetracer --browser   # browser mode with debug for development
    """
    browser_mode = "--browser" in sys.argv

    try:
        app = create_dash_app()
        register_callbacks(app)

        if browser_mode:
            import webbrowser
            threading.Timer(1.0, lambda: webbrowser.open_new("http://127.0.0.1:8050/")).start()
            app.run(debug=True, host="127.0.0.1", port=8050)
        else:
            import webview

            # Suppress Flask's per-request logging
            logging.getLogger("werkzeug").setLevel(logging.WARNING)

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

            def _kill_process_tree():
                """Kill all child processes (subprocess workers, resource trackers) then exit."""
                import os
                import psutil
                try:
                    parent = psutil.Process(os.getpid())
                    for child in parent.children(recursive=True):
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                except Exception:
                    pass
                os._exit(0)

            def _on_closed():
                _kill_process_tree()

            window = webview.create_window("TreeTracer", "http://127.0.0.1:8050/",
                                           width=1600, height=900)
            window.events.closed += _on_closed
            webview.start()

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    # Fallback
    import os
    import psutil
    try:
        parent = psutil.Process(os.getpid())
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
