"""Dash layout for TreeTracer.

Split into per-panel modules so each tab can be edited without
scrolling through the others. External callers use the same imports
as before — ``from treetracer.ui import add_header, ...`` resolves
to the symbols re-exported here.
"""

from .header import add_header
from .navbar import add_navbar
from .footer import add_footer
from .main_body import add_main_body

__all__ = ["add_header", "add_navbar", "add_footer", "add_main_body"]
