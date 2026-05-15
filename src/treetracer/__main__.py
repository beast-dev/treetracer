"""Allow running as ``python -m treetracer``."""
from __future__ import annotations


if __name__ == "__main__":
    import sys

    from .app import main

    sys.exit(main())
