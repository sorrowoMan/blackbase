"""Plugin subsystem for the blackbase shared substrate.

Exports the shared lifecycle base, manager, and soft-error reporting surface.
"""

from __future__ import annotations

from .base import PluginBase, PluginManager
from ._soft_error import report_soft_error

__all__ = [
    "PluginBase",
    "PluginManager",
    "report_soft_error",
]
