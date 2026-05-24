import shutil
import subprocess
import sys


def _escape_for_applescript(s):
    """Escape a string for use inside AppleScript double quotes."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _escape_for_python_string(s):
    """Escape a string for use inside Python single-quoted string."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


def extract_group(tree_name):
    """Extract the group prefix from a tree name (everything before the first /)."""
    return str(tree_name).split("/")[0].strip()


def extract_tree_label(tree_name):
    """Extract the tree label from a tree name (everything after the last /)."""
    return str(tree_name).split("/")[-1].strip()


def _has_zenity():
    return shutil.which("zenity") is not None


def _get_pywebview_window():
    """Return the live pywebview main window if we're running inside the
    desktop bundle, else ``None``.

    The window handle is stashed on ``peartree_api`` by ``app.main`` once
    pywebview has created the window (``peartree_api.set_main_window(window)``).
    In ``--browser`` / dev mode that handle stays ``None`` because there
    is no pywebview involved.

    Why this matters on Windows: the Briefcase bundle's ``sys.executable``
    is the launcher stub ``TreeTracer.exe``, not a real Python
    interpreter. The stub ignores ``-c "..."`` and just re-enters
    ``__main__`` (which calls ``treetracer.app.main``) — so any
    ``subprocess.run([sys.executable, "-c", "import tkinter ..."])``
    fallback we used to use for the file picker would open a *second*
    TreeTracer window instead of a dialog. Asking pywebview directly
    keeps the dialog in-process and triggers a real native Win32 /
    Cocoa / GTK file picker without spawning anything.
    """
    try:
        from ..peartree_view import peartree_api
    except Exception:
        return None
    return getattr(peartree_api, "_main_window", None)


# Tuple of accepted tree-file glob patterns. The picker shows both
# .trees and .t (BEAST writes the latter for shorter chain names);
# downstream parsing is extension-agnostic. Kept here so the four
# dialog paths below stay in sync.
_TREE_FILE_GLOBS = ("*.trees", "*.t")


def _save_file_dialog(default_filename="output.tsv"):
    """Open a native save-file dialog and return the chosen path, or
    ``None`` if the user cancelled / no dialog could be shown."""
    # Desktop path: ask pywebview directly. Stays in-process; pywebview
    # marshals the call onto the GUI thread internally and triggers the
    # OS-native dialog. This is the ONLY path that runs in a Briefcase
    # bundle on Windows — the subprocess fallback below would re-launch
    # the app (see ``_get_pywebview_window`` for the gory detail).
    window = _get_pywebview_window()
    if window is not None:
        try:
            import webview
            result = window.create_file_dialog(
                webview.FileDialog.SAVE,
                save_filename=default_filename,
                file_types=("TSV files (*.tsv)", "All files (*.*)"),
            )
        except Exception:
            result = None
        if result:
            # SAVE returns a string on most platforms but a 1-element
            # sequence on some pywebview backends — normalise.
            if isinstance(result, (list, tuple)):
                return result[0] if result else None
            return result
        # User cancelled the pywebview dialog — don't silently fall
        # through to a second dialog implementation.
        return None

    # Dev / browser-mode fallback. ``sys.executable`` here is a real
    # interpreter (uv-run / system Python), so the Tkinter subprocess
    # actually runs Tk and prints the chosen path.
    if sys.platform == "darwin":
        cmd = [
            "osascript", "-e",
            'POSIX path of (choose file name with prompt '
            '"Save file as" default name "' + _escape_for_applescript(default_filename) + '")',
        ]
    elif _has_zenity():
        cmd = [
            "zenity", "--file-selection", "--save", "--confirm-overwrite",
            "--title=Save file as",
            "--filename=" + default_filename,
            "--file-filter=TSV files | *.tsv",
            "--file-filter=All files | *",
        ]
    else:
        cmd = [
            sys.executable, "-c",
            "import tkinter as tk; from tkinter import filedialog; "
            "root = tk.Tk(); root.withdraw(); "
            "print(filedialog.asksaveasfilename("
            "title='Save file as', "
            "initialfile='" + _escape_for_python_string(default_filename) + "', "
            "defaultextension='.tsv', "
            "filetypes=[('TSV files', '*.tsv'), ('All files', '*.*')])); "
            "root.destroy()",
        ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _open_file_dialog():
    """Open a native file picker with multi-select and return a list of
    paths, or ``None`` if the user cancelled / no dialog could be
    shown."""
    # Desktop path — see ``_save_file_dialog`` for the rationale; same
    # bundle-vs-subprocess story.
    window = _get_pywebview_window()
    if window is not None:
        try:
            import webview
            # pywebview's ``file_types`` strings use semicolon-separated
            # globs inside one parenthesised filter (Win32 + Cocoa parse
            # this consistently). Listing ``*.trees;*.t`` together gives
            # us one "Trees files" entry that matches both extensions.
            result = window.create_file_dialog(
                webview.FileDialog.OPEN,
                allow_multiple=True,
                file_types=(
                    "Trees files (*.trees;*.t)",
                    "All files (*.*)",
                ),
            )
        except Exception:
            result = None
        if result:
            # OPEN with allow_multiple returns a tuple of strings; a
            # bare string is possible if a backend disagrees. Normalise.
            if isinstance(result, str):
                return [result]
            paths = [str(p) for p in result]
            return paths if paths else None
        return None

    # Dev / browser-mode fallback.
    if sys.platform == "darwin":
        cmd = [
            "osascript", "-e",
            'set theFiles to (choose file of type {"trees", "t"} '
            'with prompt "Select .trees file(s)" with multiple selections allowed)\n'
            'set output to ""\n'
            'repeat with f in theFiles\n'
            '  set output to output & POSIX path of f & "\n"\n'
            'end repeat\n'
            'return output',
        ]
    elif _has_zenity():
        cmd = [
            "zenity", "--file-selection", "--multiple", "--separator=\n",
            "--title=Select .trees file(s)",
            "--file-filter=Trees files | *.trees *.t",
            "--file-filter=All files | *",
        ]
    else:
        cmd = [
            sys.executable, "-c",
            "import tkinter as tk; from tkinter import filedialog; "
            "root = tk.Tk(); root.withdraw(); "
            "paths = filedialog.askopenfilenames("
            "title='Select .trees file(s)', "
            "filetypes=[('Trees files', ('*.trees', '*.t')), ('All files', '*.*')]); "
            "print('\\n'.join(paths)); "
            "root.destroy()",
        ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and result.stdout.strip():
            paths = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
            return paths if paths else None
    except Exception:
        pass
    return None
