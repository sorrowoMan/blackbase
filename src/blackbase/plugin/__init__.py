"""Plugin subsystem for the blackbase shared substrate.

Exports:
- :class:`PluginBase` — abstract base class for all plugins/capabilities.
- :class:`PluginManager` — plugin registration, lifecycle, and dispatch (complete implementation).
- :func:`report_soft_error` — lightweight soft-error reporting.

Compatibility alias:
- ``Plugin`` = ``PluginBase`` (for backward compatibility).
"""

from __future__ import annotations

from .base import PluginBase, PluginManager
from ._soft_error import report_soft_error

# Backward compatibility alias
Plugin = PluginBase

__all__ = [
    "PluginBase",
    "Plugin",
    "PluginManager",
    "report_soft_error",
]
