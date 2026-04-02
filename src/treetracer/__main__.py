"""Allow running as `python -m treetracer`."""
from .app import main
import sys

sys.exit(main())
