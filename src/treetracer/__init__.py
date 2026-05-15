"""TreeTracer package init.

Configures parallel-compute thread limits *before* importing any
compute-related submodule so the limits are picked up by rapidtrees'
rayon pool and scipy/numpy's BLAS pool on first use.
"""

import os


def _configure_threads() -> int:
    """Cap compute parallelism at ``cpu_count - 2``, leaving two cores
    free for the OS, the Dash callback thread, and the pywebview / Edge
    WebView2 process. Returns the value applied.

    Honors a user override via ``TREETRACER_NUM_THREADS=<n>`` and never
    clobbers a variable the user has already set in their shell.
    """
    override = os.environ.get("TREETRACER_NUM_THREADS")
    if override and override.strip().isdigit():
        n = max(1, int(override))
    else:
        n = max(1, (os.cpu_count() or 4) - 2)

    # ``setdefault``: respect any value the user already set externally.
    for var in (
        "RAYON_NUM_THREADS",     # rapidtrees (Rust + rayon)
        "OPENBLAS_NUM_THREADS",  # scipy / numpy — PyPI wheels ship OpenBLAS
        "OMP_NUM_THREADS",       # generic OpenMP umbrella; cheap safety net
    ):
        os.environ.setdefault(var, str(n))
    return n


NUM_THREADS = _configure_threads()
del _configure_threads

from .app import main

# Allow ``python -m treetracer``.
if __name__ == "__main__":
    main()
