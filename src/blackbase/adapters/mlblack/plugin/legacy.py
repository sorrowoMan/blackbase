"""
Legacy capability module compatibility layer for MLBlack.

Provides import paths for the old mlblack.core.capability module structure,
redirecting to blackbase implementations.
"""

from __future__ import annotations

import warnings

# Issue deprecation warning
warnings.warn(
    "mlblack.core.capability legacy path is deprecated. "
    "Use blackbase.plugin instead. "
    "Migration guide: See blackbase/MIGRATION.md",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the main compatibility layer
from blackbase.adapters.mlblack.plugin import *  # noqa: F401, F403


__all__ = [
    "Plugin",
    "PluginManager",
    "report_soft_error",
    "CapabilityPluginAdapter",
]
