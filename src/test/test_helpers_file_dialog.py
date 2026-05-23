"""Pin the Briefcase-Windows fix in callbacks/_helpers.py.

Before this fix, ``_open_file_dialog`` / ``_save_file_dialog`` always
walked the ``subprocess.run([sys.executable, "-c", "import tkinter ..."])``
path on Windows. In a Briefcase bundle that re-launched the whole app
(``sys.executable`` is the launcher stub) instead of opening a picker.

The fix routes the dialog through pywebview's ``create_file_dialog``
when a live pywebview window is available, and only falls back to the
subprocess shim in browser / dev mode. These tests assert that
behaviour by mocking ``_get_pywebview_window`` and the ``webview``
module — no real pywebview window required.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from treetracer.callbacks import _helpers


def _install_fake_webview(monkeypatch, file_dialog_return):
    """Drop a stub ``webview`` module into ``sys.modules`` so the
    inline ``import webview`` inside the dialog helpers picks it up.

    Returns the MagicMock standing in for ``window.create_file_dialog``
    so callers can assert it was invoked with the expected kwargs.
    """
    fake = types.ModuleType("webview")
    fake.FileDialog = types.SimpleNamespace(OPEN=10, SAVE=20)
    monkeypatch.setitem(sys.modules, "webview", fake)
    window = MagicMock()
    window.create_file_dialog.return_value = file_dialog_return
    return window


def test_open_file_dialog_uses_pywebview_when_window_available(monkeypatch):
    """Bundle / desktop path: a live window means we call
    ``window.create_file_dialog`` and never spawn a subprocess."""
    window = _install_fake_webview(monkeypatch, ("/tmp/run1.trees", "/tmp/run2.t"))
    monkeypatch.setattr(_helpers, "_get_pywebview_window", lambda: window)

    with patch.object(_helpers, "subprocess") as fake_subprocess:
        paths = _helpers._open_file_dialog()

    assert paths == ["/tmp/run1.trees", "/tmp/run2.t"]
    window.create_file_dialog.assert_called_once()
    # Critical Windows-bundle invariant: NO subprocess.run on the
    # desktop path. That call is what re-launched the app.
    fake_subprocess.run.assert_not_called()


def test_open_file_dialog_returns_none_on_pywebview_cancel(monkeypatch):
    """User cancels the pywebview dialog → return None without falling
    through to a second (subprocess) dialog."""
    window = _install_fake_webview(monkeypatch, None)
    monkeypatch.setattr(_helpers, "_get_pywebview_window", lambda: window)

    with patch.object(_helpers, "subprocess") as fake_subprocess:
        paths = _helpers._open_file_dialog()

    assert paths is None
    fake_subprocess.run.assert_not_called()


def test_open_file_dialog_accepts_bare_string_result(monkeypatch):
    """Some pywebview backends return a single string instead of a
    tuple for OPEN; normalise to a 1-element list."""
    window = _install_fake_webview(monkeypatch, "/tmp/single.trees")
    monkeypatch.setattr(_helpers, "_get_pywebview_window", lambda: window)

    paths = _helpers._open_file_dialog()
    assert paths == ["/tmp/single.trees"]


def test_save_file_dialog_uses_pywebview_when_window_available(monkeypatch):
    window = _install_fake_webview(monkeypatch, "/tmp/out.tsv")
    monkeypatch.setattr(_helpers, "_get_pywebview_window", lambda: window)

    with patch.object(_helpers, "subprocess") as fake_subprocess:
        path = _helpers._save_file_dialog("out.tsv")

    assert path == "/tmp/out.tsv"
    window.create_file_dialog.assert_called_once()
    fake_subprocess.run.assert_not_called()


def test_save_file_dialog_normalises_tuple_result(monkeypatch):
    """SAVE returns string on Win/macOS but a 1-element tuple on some
    Linux GTK backends — normalise."""
    window = _install_fake_webview(monkeypatch, ("/tmp/out.tsv",))
    monkeypatch.setattr(_helpers, "_get_pywebview_window", lambda: window)

    path = _helpers._save_file_dialog("out.tsv")
    assert path == "/tmp/out.tsv"


def test_open_file_dialog_falls_back_to_subprocess_in_browser_mode(monkeypatch):
    """Browser / dev mode: no pywebview window → use the subprocess
    shim. ``sys.executable`` is a real interpreter here, so it works.
    """
    monkeypatch.setattr(_helpers, "_get_pywebview_window", lambda: None)
    fake_completed = MagicMock()
    fake_completed.returncode = 0
    fake_completed.stdout = "/tmp/a.trees\n/tmp/b.t\n"
    with patch.object(_helpers.subprocess, "run", return_value=fake_completed) as run:
        paths = _helpers._open_file_dialog()

    assert paths == ["/tmp/a.trees", "/tmp/b.t"]
    run.assert_called_once()


def test_save_file_dialog_falls_back_to_subprocess_in_browser_mode(monkeypatch):
    monkeypatch.setattr(_helpers, "_get_pywebview_window", lambda: None)
    fake_completed = MagicMock()
    fake_completed.returncode = 0
    fake_completed.stdout = "/tmp/out.tsv\n"
    with patch.object(_helpers.subprocess, "run", return_value=fake_completed) as run:
        path = _helpers._save_file_dialog("out.tsv")

    assert path == "/tmp/out.tsv"
    run.assert_called_once()


def test_open_file_dialog_filter_advertises_both_extensions(monkeypatch):
    """The Trees-files filter must list BOTH .trees and .t, matching
    the loader's ``endswith((".trees", ".t"))`` gate. If they drift,
    users can't pick .t files from the bundle dialog."""
    window = _install_fake_webview(monkeypatch, ())
    monkeypatch.setattr(_helpers, "_get_pywebview_window", lambda: window)

    _helpers._open_file_dialog()
    _, kwargs = window.create_file_dialog.call_args
    filters = kwargs["file_types"]
    trees_filter = next(f for f in filters if f.lower().startswith("trees"))
    assert ".trees" in trees_filter and ".t" in trees_filter, trees_filter


def test_get_pywebview_window_returns_none_when_peartree_api_unavailable(monkeypatch):
    """If the peartree_view module fails to import (e.g. before app
    boot during tests), ``_get_pywebview_window`` returns None instead
    of raising — callers can safely fall through to the subprocess
    path."""
    import treetracer.peartree_view as pv
    monkeypatch.setattr(pv.peartree_api, "_main_window", None, raising=False)
    assert _helpers._get_pywebview_window() is None
