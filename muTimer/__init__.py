"""
Hierarchical timing utility with nested context manager support.
"""

# MIT License. See LICENSE file for details.

from .Timer import Timer

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

__all__ = ["Timer", "__version__"]
