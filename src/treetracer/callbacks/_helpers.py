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


def _save_file_dialog(default_filename="output.tsv"):
    """Open a native save-file dialog and return the chosen path."""
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
    """Open a native file picker with multi-select and return a list of paths."""
    if sys.platform == "darwin":
        cmd = [
            "osascript", "-e",
            'set theFiles to (choose file of type {"trees"} '
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
            "--file-filter=Trees files | *.trees",
            "--file-filter=All files | *",
        ]
    else:
        cmd = [
            sys.executable, "-c",
            "import tkinter as tk; from tkinter import filedialog; "
            "root = tk.Tk(); root.withdraw(); "
            "paths = filedialog.askopenfilenames("
            "title='Select .trees file(s)', "
            "filetypes=[('Trees files', '*.trees'), ('All files', '*.*')]); "
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


