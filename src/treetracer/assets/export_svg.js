// Client-side SVG export for TreeTracer plots.
//
// Replaces the former server-side ``fig.write_image()`` (Kaleido) path.
// Kaleido v1 drives an external Chrome as a subprocess, and on macOS/Linux
// choreographer launches it via ``sys.executable`` + a pipe wrapper. Inside
// the Briefcase bundle ``sys.executable`` is the TreeTracer app stub, not a
// Python interpreter, so that subprocess re-launched the whole app (a second
// window with an empty grey webview) and Chrome never connected — no file was
// ever written. Same trap the file-picker already documents in ``_helpers.py``
// / ``peartree_view.py``.
//
// Instead we render in the webview that already holds the figure (Plotly.js is
// present) and save the bytes through the existing ``save_download`` pywebview
// bridge — or a normal browser download when running in ``--browser`` dev mode.
window.dash_clientside = window.dash_clientside || {};

(function () {
  function _findGraphDiv(graphId) {
    var root = document.getElementById(graphId);
    if (!root) return null;
    // dcc.Graph renders the live Plotly node as ``.js-plotly-plot``; that is
    // the element Plotly.toImage expects (it carries ``_fullLayout``).
    if (root.classList && root.classList.contains("js-plotly-plot")) return root;
    return root.querySelector(".js-plotly-plot");
  }

  function _svgFromDataUri(uri) {
    // Plotly's ``format:'svg'`` returns "data:image/svg+xml,<encodeURIComponent>".
    var comma = uri.indexOf(",");
    return decodeURIComponent(uri.slice(comma + 1));
  }

  function _utf8ToBase64(str) {
    // save_download base64-decodes then writes raw bytes, so encode UTF-8
    // first (taxon/run labels may contain non-latin1 characters).
    var bytes = new TextEncoder().encode(str);
    var bin = "";
    for (var i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    return btoa(bin);
  }

  function _notif(title, message, color) {
    // Mirrors the dmc.Notification(action="show") the server callbacks used
    // to return into #notifications-container.
    return {
      namespace: "dash_mantine_components",
      type: "Notification",
      props: {
        title: title,
        message: message,
        color: color,
        action: "show",
        autoClose: color === "green" ? 3000 : 6000,
        id: "notif-svg-" + Date.now() + "-" + Math.floor(Math.random() * 1e9),
      },
    };
  }

  // Render the given Dash graph to SVG and save it. Returns a Notification
  // component (for #notifications-container) or dash_clientside.no_update.
  window.ttExportSvg = async function (graphId, filename) {
    var noUpdate = window.dash_clientside.no_update;
    var gd = _findGraphDiv(graphId);
    if (!gd || !window.Plotly) {
      return _notif("Export failed", "Plot is not ready to export yet.", "red");
    }

    var svgText;
    try {
      // Render at the graph's CURRENT on-screen size so the SVG matches
      // exactly what's displayed (aspect ratio, legend reflow, 3D camera,
      // current zoom). ``_fullLayout.width/height`` is the pixel size Plotly
      // actually drew with; we must pass it explicitly, because toImage
      // otherwise clones the graph and the clone falls back to Plotly's
      // 700x450 default — not the live size. Fixed dimensions were the old
      // bug (a different aspect than the screen); vector SVG needs no size.
      var fl = gd._fullLayout || {};
      var opts = { format: "svg" };
      var w = fl.width || gd.offsetWidth;
      var h = fl.height || gd.offsetHeight;
      if (w) opts.width = w;
      if (h) opts.height = h;
      var uri = await window.Plotly.toImage(gd, opts);
      svgText = _svgFromDataUri(uri);
    } catch (err) {
      return _notif("Export failed", String((err && err.message) || err), "red");
    }

    // Desktop (pywebview) path: hand the bytes to Python, which opens a native
    // save dialog and writes the file.
    var api = window.pywebview && window.pywebview.api;
    if (api && api.save_download) {
      var ack;
      try {
        ack = await api.save_download(filename, _utf8ToBase64(svgText));
      } catch (err) {
        return _notif("Export failed", String((err && err.message) || err), "red");
      }
      if (ack && ack.cancelled) return noUpdate; // user dismissed the dialog
      if (ack && ack.error) return _notif("Export failed", ack.error, "red");
      var where = ack && ack.path ? "Saved to " + ack.path : "Saved.";
      return _notif("SVG Exported", where, "green");
    }

    // Dev / browser mode: no bridge, so use a normal anchor download.
    var blob = new Blob([svgText], { type: "image/svg+xml" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 5000);
    return _notif("SVG Exported", filename, "green");
  };
})();
