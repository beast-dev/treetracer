"""Flask routes serving an in-browser peartree view of an MCC tree.

When the user clicks "View MCC" we:
  1. compute the MCC tree server-side (mcc.assemble_mcc_nexus) and stash
     the resulting NEXUS bytes in ``state._mcc_cache`` under a random
     URL-safe handle ``uid``;
  2. open a new browser window pointing at /peartree/<uid> via a Dash
     clientside callback (window.open).

This module owns the two routes that window depends on:

  GET /peartree/<uid>            HTML page that boots peartree.
  GET /peartree/<uid>/tree.nex   Raw NEXUS bytes the page fetches.

peartree itself is loaded from /assets/peartree.bundle.min.js; the
vendored bundle sits in ``src/treetracer/assets/`` and Dash auto-serves
it at that URL. Auto-injection of the bundle into the main TreeTracer
page is suppressed via ``assets_ignore`` in app.py.
"""

from __future__ import annotations

from flask import Response, abort, render_template_string, request

from . import state


# Minimal HTML that fills the viewport with peartree. ``treeUrl`` is a
# same-origin fetch — peartree pulls /peartree/<uid>/tree.nex itself.
_PEARTREE_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MCC tree — {{ name }}</title>
  <style>
    html, body { margin: 0; padding: 0; height: 100%; width: 100%; }
    #tree { height: 100vh; width: 100vw; }
  </style>
</head>
<body>
  <div id="tree"></div>
  <script src="/assets/peartree.bundle.min.js"></script>
  <script>
    // Embed the tree, then apply descending clade-size sort.
    //
    // We synthesise a click on the toolbar's "Sort descending" button
    // (``btn-order-desc``) rather than calling ``controller.sort('desc')``
    // from ``onTreeLoad``. Both run the same internal ``gs(true)`` sort,
    // but the bundle only un-disables ``btn-order-desc`` once its tree
    // state is fully hydrated, so polling the button's ``disabled`` flag
    // gives us a robust "tree is ready" signal that ``onTreeLoad`` alone
    // didn't reliably provide.
    //
    // ``introAnimation: 'none'`` is load-bearing: the bundle starts the
    // intro animation BEFORE dispatching ``peartree-tree-loaded``, so a
    // sort applied mid-animation gets clobbered when peartree finishes
    // the animation by snapping back to the pre-sort layout ("appears
    // sorted, then unsorts"). Disabling the animation means the layout
    // is final at sort time.
    PearTreeEmbed.embed({
      container: "tree",
      treeUrl:   "/peartree/{{ uid }}/tree.nex",
      filename:  "mcc.nex",
      height:    "100vh",
      settings: { introAnimation: "none" },
    });
    (() => {
      const iv = setInterval(() => {
        const btn = document.getElementById("btn-order-desc");
        if (!btn || btn.disabled) return;
        btn.click();
        clearInterval(iv);
      }, 100);
      // Hard cap so the interval doesn't leak if the tree never loads.
      setTimeout(() => clearInterval(iv), 10000);
    })();
  </script>
  <script>
    // Pywebview-only download interceptor.
    //
    // In a real browser, peartree's "Download tree" anchor click
    // (`<a href="blob:..." download="…">`) triggers the OS save
    // dialog. In pywebview's embedded webview the same anchor click
    // is treated as a navigation to the blob URL and the page just
    // renders the newick text. We intercept the click and round-trip
    // the bytes to Python via ``window.pywebview.api.save_download``,
    // which opens a native save dialog.
    //
    // Two subtleties:
    //   1. Peartree creates the blob, programmatically clicks an
    //      anchor, then immediately calls URL.revokeObjectURL().
    //      If we ``fetch(blobUrl)`` async after preventDefault, the
    //      URL has already been revoked and the fetch fails silently.
    //      We avoid that by shadowing createObjectURL/revokeObjectURL
    //      to keep the blob alive in a small in-memory registry.
    //   2. We attach the click listener unconditionally; in browser
    //      mode (no pywebview) the listener bails on the first check
    //      and the native download path is left intact.
    (function () {
      const _blobs = new Map();        // url -> Blob
      const _origCreate = URL.createObjectURL.bind(URL);
      const _origRevoke = URL.revokeObjectURL.bind(URL);
      URL.createObjectURL = function (obj) {
        const url = _origCreate(obj);
        if (obj instanceof Blob) _blobs.set(url, obj);
        return url;
      };
      URL.revokeObjectURL = function (url) {
        // Defer real revocation so our async handler can still use
        // the blob; we also keep our own reference in _blobs so the
        // Blob isn't garbage-collected after the page revokes.
        setTimeout(() => {
          try { _origRevoke(url); } catch (e) { /* ignore */ }
          _blobs.delete(url);
        }, 5000);
      };

      function _blobToBase64(blob) {
        return new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onloadend = () => {
            const r = reader.result || "";
            const i = r.indexOf(",");
            resolve(i >= 0 ? r.slice(i + 1) : "");
          };
          reader.onerror = () => reject(reader.error);
          reader.readAsDataURL(blob);
        });
      }

      async function _intercept(e) {
        const path = (e.composedPath && e.composedPath()) || [];
        const a = path.find(el => el && el.tagName === "A");
        if (!a) return;
        const href = a.getAttribute("href") || "";
        const isDownload =
          a.hasAttribute("download")
          || href.startsWith("blob:")
          || href.startsWith("data:");
        if (!isDownload) return;
        // Browser / dev mode: no bridge → leave native flow intact.
        if (!(window.pywebview && window.pywebview.api
              && window.pywebview.api.save_download)) return;
        e.preventDefault();
        e.stopPropagation();
        try {
          const blob = _blobs.get(a.href)
                       || await fetch(a.href).then(r => r.blob());
          const b64 = await _blobToBase64(blob);
          const filename = a.getAttribute("download") || "download";
          const ack = await window.pywebview.api.save_download(filename, b64);
          if (ack && ack.error) {
            alert("Download failed: " + ack.error);
          }
        } catch (err) {
          alert("Download failed: " + (err && err.message ? err.message : err));
        }
      }

      // Attach immediately; the bail-out check inside handles the
      // case where pywebview's API hasn't been bridged yet at click
      // time (in practice the user clicks long after page load).
      document.addEventListener("click", _intercept, true);
    })();
  </script>
</body>
</html>
"""


class PeartreeJSApi:
    """JS API exposed to the main TreeTracer pywebview window.

    Methods on this object are reachable from JS as
    ``window.pywebview.api.<method>(...)``. The View-MCC clientside
    callback calls ``open_peartree(uid, name)`` to spawn a sibling
    native window pointing at this server's ``/peartree/<uid>`` route,
    keeping the desktop experience inside pywebview rather than handing
    off to the system browser.
    """

    def __init__(self):
        # Reference to the main TreeTracer pywebview window — used to
        # re-focus it when a peartree sibling window closes (see the
        # ``closed`` handler in ``open_peartree``). Set from app.py
        # after the main window is created.
        self._main_window = None

    def set_main_window(self, window):
        """Tell the API which pywebview window is the TreeTracer main
        window. Called once from app.py."""
        self._main_window = window

    def open_peartree(self, uid, name=""):
        """Open a sibling pywebview window for ``/peartree/<uid>``.

        Returns a small ack dict so the JS-side promise has something
        to resolve to. ``webview`` is imported lazily so that importing
        this module in a non-pywebview context (browser mode, tests)
        doesn't drag pywebview in unnecessarily.

        ``js_api=self`` is passed to the sibling window so the page's
        download-interceptor JS can call ``save_download`` on the same
        API object — pywebview attaches the api per-window, not
        globally.

        On the ``closing`` event we navigate the window to a tiny
        placeholder page. macOS WKWebView's teardown of peartree's
        ~1.5 MB JS bundle + rendered DOM otherwise blocks the main
        Cocoa runloop for a few seconds, which freezes the main
        TreeTracer window. Swapping the content out first means the
        eventual destroy has nothing heavy to unload.
        """
        from urllib.parse import quote
        import webview

        title = f"MCC tree — {name}" if name else f"MCC tree — {uid}"
        url = f"http://127.0.0.1:8050/peartree/{uid}"
        if name:
            url += "?name=" + quote(name)
        win = webview.create_window(title, url, width=1200, height=800,
                                    resizable=True, js_api=self)

        def _on_closing(*_):
            try:
                win.load_html(
                    "<!doctype html><title>Closing…</title>"
                    "<body style=\"background:#fafafa\"></body>"
                )
            except Exception:
                pass
            return True  # let the close proceed

        def _on_closed(*_):
            # macOS Cocoa quirk: when a non-key window closes, focus
            # doesn't always return to another existing window — the
            # app sits with no key window and the user has to click
            # the main window's title bar before it accepts input.
            # Re-asserting the main window's foreground state (and
            # then activating the app process itself) restores normal
            # click behaviour without that intervening title-bar click.
            main = self._main_window
            if main is not None:
                try:
                    main.show()
                except Exception:
                    pass
            try:
                from AppKit import NSApp
                NSApp.activateIgnoringOtherApps_(True)
            except Exception:
                # Non-macOS or PyObjC unavailable — main.show() above
                # is sufficient on Windows / Linux backends anyway.
                pass

        win.events.closing += _on_closing
        win.events.closed += _on_closed
        return {"ok": True, "uid": uid}

    def save_download(self, filename, content_b64):
        """Receive a base64 payload from the embedded peartree window's
        click-interceptor JS and write it via a native save dialog.

        Pywebview's WKWebView/WebView2/QtWebEngine wrappers don't fire
        the OS download dialog for ``<a href="blob:..." download>`` the
        way a real browser does, so we round-trip the bytes through
        Python, where ``_save_file_dialog`` shells out to the native
        save picker.
        """
        import base64
        from .callbacks._helpers import _save_file_dialog
        try:
            data = base64.b64decode(content_b64 or "")
        except Exception as exc:
            return {"ok": False, "error": f"decode failed: {exc}"}
        path = _save_file_dialog(default_filename=filename or "download")
        if not path:
            return {"ok": False, "cancelled": True}
        try:
            with open(path, "wb") as out:
                out.write(data)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "path": path}


# Module-level singleton — passed as ``js_api=`` when the main
# TreeTracer pywebview window is created in app.py.
peartree_api = PeartreeJSApi()


def register_routes(server):
    """Attach the two routes onto a Flask ``server`` instance.

    Idempotent — registering twice would crash Flask, so we tag the
    server with a sentinel attribute and bail on re-registration.
    """
    if getattr(server, "_treetracer_peartree_routes_registered", False):
        return
    server._treetracer_peartree_routes_registered = True

    @server.route("/peartree/<uid>")
    def peartree_page(uid):
        if not state.has_cached_mcc_tree(uid):
            abort(404)
        # Allow callers to label the window's <title> via ?name=...
        name = request.args.get("name") or "MCC tree"
        return render_template_string(_PEARTREE_PAGE, uid=uid, name=name)

    @server.route("/peartree/<uid>/tree.nex")
    def peartree_data(uid):
        data = state.get_cached_mcc_tree(uid)
        if data is None:
            abort(404)
        # Cache-Control no-store — UUIDs are short-lived and may collide
        # with a re-export under a server restart in pathological cases;
        # we always want a fresh fetch.
        return Response(
            data,
            mimetype="text/plain",
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": 'inline; filename="mcc.nex"',
            },
        )
