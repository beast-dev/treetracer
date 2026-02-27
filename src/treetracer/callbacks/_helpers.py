import shutil
import subprocess
import sys


def _has_zenity():
    return shutil.which("zenity") is not None


def _save_file_dialog(default_filename="output.tsv"):
    """Open a native save-file dialog and return the chosen path."""
    if sys.platform == "darwin":
        cmd = [
            "osascript", "-e",
            'POSIX path of (choose file name with prompt '
            '"Save file as" default name "' + default_filename + '")',
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
            "initialfile='" + default_filename + "', "
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
    """Open a native file picker and return the selected path."""
    if sys.platform == "darwin":
        cmd = [
            "osascript", "-e",
            'POSIX path of (choose file of type {"trees"} '
            'with prompt "Select a .trees file")',
        ]
    elif _has_zenity():
        cmd = [
            "zenity", "--file-selection",
            "--title=Select a .trees file",
            "--file-filter=Trees files | *.trees",
            "--file-filter=All files | *",
        ]
    else:
        cmd = [
            sys.executable, "-c",
            "import tkinter as tk; from tkinter import filedialog; "
            "root = tk.Tk(); root.withdraw(); "
            "print(filedialog.askopenfilename("
            "title='Select a .trees file', "
            "filetypes=[('Trees files', '*.trees'), ('All files', '*.*')])); "
            "root.destroy()",
        ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _open_tsv_dialog():
    """Open a native file picker filtered to .tsv files and return the selected path."""
    if sys.platform == "darwin":
        cmd = [
            "osascript", "-e",
            'POSIX path of (choose file of type {"tsv","tab"} '
            'with prompt "Select a .tsv file")',
        ]
    elif _has_zenity():
        cmd = [
            "zenity", "--file-selection",
            "--title=Select a .tsv file",
            "--file-filter=TSV files | *.tsv",
            "--file-filter=All files | *",
        ]
    else:
        cmd = [
            sys.executable, "-c",
            "import tkinter as tk; from tkinter import filedialog; "
            "root = tk.Tk(); root.withdraw(); "
            "print(filedialog.askopenfilename("
            "title='Select a .tsv file', "
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


def _validate_group_names(names):
    """Check that every tree name contains at least one '/' as a group delimiter.

    Returns (True, None) if valid, (False, error_message) if any name lacks a group.
    """
    bad_names = [n for n in names if "/" not in str(n)]
    if bad_names:
        preview = ", ".join(str(n) for n in bad_names[:5])
        if len(bad_names) > 5:
            preview += f" ... ({len(bad_names)} total)"
        return False, (
            "Tree names must include a group prefix (e.g., 'group1/tree_name'). "
            f"Found names without '/': {preview}"
        )
    return True, None
