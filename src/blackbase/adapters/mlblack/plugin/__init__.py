"""
MLBlack Plugin Compatibility Layer.

This module provides backward compatibility for MLBlack's capability/plugin modules,
wrapping blackbase implementations while maintaining the original API.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

# Re-export from blackbase
from blackbase.plugin import PluginBase, PluginManager, report_soft_error


def _warn_deprecated(module: str = "capability", removal_version: str = "1.0.0") -> None:
    """Issue deprecation warning for MLBlack-specific plugin modules."""
    warnings.warn(
        f"mlblack.core.{module} is deprecated. "
        f"Use blackbase.plugin instead. "
        f"This will be removed in {removal_version}.",
        DeprecationWarning,
        stacklevel=3,
    )


class Capability(PluginBase):
    """MLBlack Capability base class.

    This is a Plugin subclass that provides the mlblack-style lifecycle hooks
    (on_fit_start, on_step_start, on_step_end, on_fit_end, on_evaluate_start,
    on_evaluate_end, on_error) as aliases for the unified Plugin lifecycle.

    Subclass this just like the old mlblack Capability:
        class MyCap(Capability):
            def on_step_end(self, trainer, context, row):
                ...
    """

    def __init__(self, name: Optional[str] = None, priority: int = 0, **kwargs):
        super().__init__(name=name or "capability", priority=priority, **kwargs)

    # --- mlblack-style hooks (override these in subclasses) ---
    # These use _ml_ prefix internally to avoid name collision with Plugin hooks.

    def on_fit_start(self, trainer, context):
        """Called when training begins."""
        pass

    def on_step_start(self, trainer, context, row):
        """Called before each training step."""
        pass

    def on_evaluate_start_ml(self, trainer, candidate, context):
        """Called before candidate evaluation (mlblack signature)."""
        pass

    def on_evaluate_end_ml(self, trainer, candidate, feedback, context):
        """Called after candidate evaluation (mlblack signature)."""
        pass

    def on_step_end(self, trainer, context, row):
        """Called after each training step."""
        pass

    def on_fit_end(self, trainer, context, result):
        """Called when training ends."""
        pass

    def on_error_ml(self, trainer, error, context):
        """Called on error (mlblack signature)."""
        pass

    # --- Plugin lifecycle mapping ---

    def on_solver_init(self, solver):
        self.on_fit_start(solver, {})

    def on_population_init(self, solver):
        pass

    def on_generation_start(self, generation):
        self.on_step_start(None, {}, {})

    def on_evaluate_start(self, candidate, context=None):
        self.on_evaluate_start_ml(None, candidate, context or {})

    def on_evaluate_end(self, candidate, feedback, context=None):
        self.on_evaluate_end_ml(None, candidate, feedback, context or {})

    def on_generation_end(self, generation):
        self.on_step_end(None, {}, {})

    def on_solver_finish(self, result):
        self.on_fit_end(None, {}, result)

    def on_error(self, error, context=None):
        self.on_error_ml(None, error, context or {})


class CapabilityPluginAdapter(PluginBase):
    """Adapts an existing mlblack Capability instance to the Plugin lifecycle.

    Use this when you have a Capability *instance* (not subclass) that needs
    to be registered with a PluginManager.
    """

    def __init__(self, capability, name: Optional[str] = None, priority: int = 0):
        super().__init__(
            name=name or getattr(capability, "name", "capability"),
            priority=priority,
        )
        self.capability = capability

    def on_solver_init(self, solver):
        handler = getattr(self.capability, "on_fit_start", None)
        if callable(handler):
            handler(solver, {})

    def on_generation_start(self, generation: int):
        handler = getattr(self.capability, "on_step_start", None)
        if callable(handler):
            handler(None, {}, {})

    def on_evaluate_start(self, candidate, context: Optional[Any] = None):
        handler = getattr(self.capability, "on_evaluate_start_ml", None)
        if callable(handler):
            handler(None, candidate, context or {})
        else:
            handler = getattr(self.capability, "on_evaluate_start", None)
            if callable(handler):
                handler(None, candidate, context or {})

    def on_evaluate_end(self, candidate, feedback, context: Optional[Any] = None):
        handler = getattr(self.capability, "on_evaluate_end_ml", None)
        if callable(handler):
            handler(None, candidate, feedback, context or {})
        else:
            handler = getattr(self.capability, "on_evaluate_end", None)
            if callable(handler):
                handler(None, candidate, feedback, context or {})

    def on_generation_end(self, generation: int):
        handler = getattr(self.capability, "on_step_end", None)
        if callable(handler):
            handler(None, {}, {})

    def on_solver_finish(self, result):
        handler = getattr(self.capability, "on_fit_end", None)
        if callable(handler):
            handler(None, {}, result)

    def on_error(self, error: BaseException, context: Optional[Any] = None):
        handler = getattr(self.capability, "on_error_ml", None)
        if callable(handler):
            handler(None, error, context or {})
        else:
            handler = getattr(self.capability, "on_error", None)
            if callable(handler):
                handler(None, error, context or {})


__all__ = [
    # Re-exported from blackbase.plugin
    "PluginBase",
    "PluginManager",
    "report_soft_error",
    # MLBlack-specific
    "Capability",
    "CapabilityPluginAdapter",
]
