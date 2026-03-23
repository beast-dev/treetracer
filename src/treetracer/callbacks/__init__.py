from .shell import register_shell_callbacks
from .sidebar import register_sidebar_callbacks
from .compute import register_compute_callbacks
from .treespace import register_treespace_callbacks
from .diagnostics import register_diagnostics_callbacks
from .within_run import register_within_run_callbacks


def register_callbacks(app):
    register_shell_callbacks()
    register_sidebar_callbacks()
    register_compute_callbacks()
    register_treespace_callbacks()
    register_diagnostics_callbacks()
    register_within_run_callbacks()
