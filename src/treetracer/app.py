import dash_mantine_components as dmc
from dash import Dash
from .ui import add_header, add_navbar, add_main_body, add_footer
from .callbacks import register_callbacks
from . import peartree_view
import sys
import logging
import threading


def create_dash_app():
    app = Dash(
        __name__,
        external_stylesheets=dmc.styles.ALL,
        suppress_callback_exceptions=True,
        # peartree.bundle.min.js lives in assets/ so Dash auto-serves it
        # at /assets/peartree.bundle.min.js, but we don't want it injected
        # into the main page's <head> — it's only needed inside the
        # /peartree/<uid> windows opened by "View MCC". This regex stops
        # the auto-injection while leaving the file accessible.
        assets_ignore=r"peartree\.bundle\.min\.js",
        title="TreeTracer",
    )

    # Use the SVG logo as the browser tab favicon (PNG fallback for older browsers).
    # Dash's default {%favicon%} looks for assets/favicon.ico; we override it here.
    app.index_string = app.index_string.replace(
        "{%favicon%}",
        '<link rel="icon" type="image/svg+xml" href="/assets/treetracer-icon.svg">'
        '<link rel="alternate icon" type="image/png" href="/assets/treetracer-icon.png">',
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
        id="mantine-provider",
        forceColorScheme="light",
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
        peartree_view.register_routes(app.server)

        if browser_mode:
            import webbrowser
            threading.Timer(1.0, lambda: webbrowser.open_new("http://127.0.0.1:8050/")).start()
            app.run(debug=True, host="127.0.0.1", port=8050)
        else:
            import webview

            # macOS: the system menu bar reads its app name from the
            # running bundle's ``CFBundleName``, which for a plain
            # ``python -m treetracer`` launch is just "python3". Patch
            # the in-process info dict before pywebview spins up Cocoa
            # so the menu shows "TreeTracer" in dev. A properly bundled
            # .app overrides this via its own Info.plist.
            if sys.platform == "darwin":
                try:
                    from Foundation import NSBundle
                    bundle = NSBundle.mainBundle()
                    info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
                    info["CFBundleName"] = "TreeTracer"
                except Exception:
                    pass

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
                """Kill all child processes (subprocess workers, resource trackers) then exit.

                ``os._exit(0)`` below is a HARD exit that bypasses every
                Python finalizer — including ``atexit`` handlers. Anything
                that needs cleanup on close (the tmp distmat directory, in
                particular) must be torn down explicitly here.
                """
                import os
                import psutil
                # Wipe the on-disk distmat tmpdir before the hard exit;
                # the atexit handler registered in state._ensure_tmpdir
                # would otherwise never run.
                try:
                    from . import state
                    state._cleanup_tmpdir()
                except Exception:
                    pass
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

            # Catch SIGTERM / SIGINT so the tmpdir is wiped on:
            #   * macOS "Force Quit" from the dock (sends SIGTERM)
            #   * Ctrl-C in dev mode (sends SIGINT)
            # The window-close event already routes through _on_closed
            # → _kill_process_tree above, but those signal paths bypass
            # pywebview's closed-event entirely. Without explicit
            # handlers they'd hard-exit and leak the tmpdir again.
            import signal
            def _on_signal(signum, _frame):
                _kill_process_tree()
            try:
                signal.signal(signal.SIGTERM, _on_signal)
                signal.signal(signal.SIGINT, _on_signal)
            except (ValueError, AttributeError):
                # signal.signal only works on the main thread; on some
                # platforms SIGTERM may not be supported. In either
                # case fall back to the existing _on_closed path.
                pass

            # Expose the peartree JS API on this window so the View-MCC
            # clientside callback can spawn sibling pywebview windows via
            # ``window.pywebview.api.open_peartree(uid, name)`` instead of
            # bouncing out to the system browser.
            window = webview.create_window(
                "TreeTracer", "http://127.0.0.1:8050/",
                width=1600, height=900,
                js_api=peartree_view.peartree_api,
            )
            # The api needs a handle on this window so it can re-focus
            # it when a peartree sibling window closes (Cocoa otherwise
            # leaves the app with no key window).
            peartree_view.peartree_api.set_main_window(window)
            window.events.closed += _on_closed
            webview.start()

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    # Fallback (reached when the try block above falls through, e.g.
    # an unexpected exception during pywebview startup). Same tmpdir
    # cleanup rationale as ``_kill_process_tree`` — ``os._exit`` skips
    # atexit handlers.
    import os
    import psutil
    try:
        from . import state
        state._cleanup_tmpdir()
    except Exception:
        pass
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
