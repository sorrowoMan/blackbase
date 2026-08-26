"""Plugin subsystem for the blackbase shared substrate.

Exports the shared lifecycle base, manager, and soft-error reporting surface.
"""

from __future__ import annotations

from .base import PluginBase, PluginManager
from ._soft_error import report_soft_error
from .lifecycle import (
    ATTEMPT_END,
    ATTEMPT_START,
    GENERATION_COMMITTED,
    GENERATION_END,
    GENERATION_START,
    PluginLifecycleCleanupError,
    PluginLifecycleDispatchError,
    PluginLifecycleReceipt,
    RUNTIME_LIFECYCLE_SLOTS,
    normalize_lifecycle_slot,
)

__all__ = [
    "PluginBase",
    "PluginManager",
    "ATTEMPT_END",
    "ATTEMPT_START",
    "GENERATION_COMMITTED",
    "GENERATION_END",
    "GENERATION_START",
    "PluginLifecycleCleanupError",
    "PluginLifecycleDispatchError",
    "PluginLifecycleReceipt",
    "RUNTIME_LIFECYCLE_SLOTS",
    "normalize_lifecycle_slot",
    "report_soft_error",
]
