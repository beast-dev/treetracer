from .shell import register_shell_callbacks
from .sidebar import register_sidebar_callbacks
from .compute import register_compute_callbacks
from .treespace import register_treespace_callbacks
from .diagnostics import register_diagnostics_callbacks
from .within_run import register_within_run_callbacks
from .mcc_list import register_mcc_list_callbacks
from .mcc_compute import register_mcc_compute_callbacks
from .pseudo_ess_compute import register_pseudo_ess_compute_callbacks
from .clade_explore import register_clade_explore_callbacks


def register_callbacks(app):
    register_shell_callbacks()
    register_sidebar_callbacks()
    register_compute_callbacks()
    register_treespace_callbacks()
    register_diagnostics_callbacks()
    register_within_run_callbacks()
    register_mcc_list_callbacks()
    register_mcc_compute_callbacks()
    register_pseudo_ess_compute_callbacks()
    register_clade_explore_callbacks()
