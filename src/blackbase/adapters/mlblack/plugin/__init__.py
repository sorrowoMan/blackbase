"""
MLBlack Plugin Compatibility Layer.

This module provides backward compatibility for MLBlack's capability/plugin modules,
wrapping blackbase implementations while maintaining the original API.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Optional

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


def _trainer_context(trainer: Any) -> dict[str, Any]:
    """Return a detached context projection for legacy Capability hooks."""

    if trainer is None:
        return {}
    builder = getattr(trainer, "build_context", None)
    if callable(builder):
        value = builder()
        if isinstance(value, Mapping):
            return dict(value)
    store = getattr(trainer, "context_store", None)
    snapshot = getattr(store, "snapshot", None)
    if callable(snapshot):
        value = snapshot()
        if isinstance(value, Mapping):
            return dict(value)
    if isinstance(store, Mapping):
        return dict(store)
    return {}


def _step_row(trainer: Any, generation: int) -> dict[str, Any]:
    history = getattr(trainer, "history", None) if trainer is not None else None
    if history:
        latest = history[-1]
        if isinstance(latest, Mapping):
            return dict(latest)
    return {"step": int(generation)}


def _legacy_evaluation_handler(capability: Any, name: str, mapped_name: str):
    """Resolve old mlblack hook names without selecting Capability's Plugin shim."""

    normal = getattr(capability, name, None)
    if not isinstance(capability, Capability):
        return normal if callable(normal) else None
    implementation = getattr(type(capability), name, None)
    base_implementation = getattr(Capability, name, None)
    if callable(normal) and implementation is not base_implementation:
        return normal
    mapped = getattr(capability, mapped_name, None)
    return mapped if callable(mapped) else None


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
        return None

    def on_step_start(self, trainer, context, row):
        """Called before each training step."""
        return None

    def on_evaluate_start_ml(self, trainer, candidate, context):
        """Called before candidate evaluation (mlblack signature)."""
        return None

    def on_evaluate_end_ml(self, trainer, candidate, feedback, context):
        """Called after candidate evaluation (mlblack signature)."""
        return None

    def on_step_end(self, trainer, context, row):
        """Called after each training step."""
        return None

    def on_fit_end(self, trainer, context, result):
        """Called when training ends."""
        return None

    def on_error_ml(self, trainer, error, context):
        """Called on error (mlblack signature)."""
        return None

    # --- Plugin lifecycle mapping ---

    def on_solver_init(self, solver):
        self.on_fit_start(solver, _trainer_context(solver))

    def on_population_init(self, population, objectives, violations):
        return None

    def on_generation_start(self, generation):
        self.on_step_start(
            self.solver,
            _trainer_context(self.solver),
            {"step": int(generation)},
        )

    def on_evaluate_start(self, candidate, context=None):
        self.on_evaluate_start_ml(self.solver, candidate, dict(context or {}))

    def on_evaluate_end(self, candidate, feedback, context=None):
        self.on_evaluate_end_ml(self.solver, candidate, feedback, dict(context or {}))

    def on_generation_end(self, generation):
        self.on_step_end(
            self.solver,
            _trainer_context(self.solver),
            _step_row(self.solver, generation),
        )

    def on_solver_finish(self, result):
        report = result.get("report", {}) if isinstance(result, Mapping) else result
        if not isinstance(report, Mapping):
            report = {"result": report}
        self.on_fit_end(self.solver, _trainer_context(self.solver), dict(report))

    def on_error(self, error, context=None):
        self.on_error_ml(self.solver, error, dict(context or {}))


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
        self._capability = capability

    def get_context_contract(self):
        getter = getattr(self.capability, "get_context_contract", None)
        if callable(getter):
            contract = getter()
            if isinstance(contract, Mapping):
                return dict(contract)
        return super().get_context_contract()

    def on_solver_init(self, solver):
        handler = getattr(self.capability, "on_fit_start", None)
        if callable(handler):
            handler(solver, _trainer_context(solver))

    def on_generation_start(self, generation: int):
        handler = getattr(self.capability, "on_step_start", None)
        if callable(handler):
            handler(
                self.solver,
                _trainer_context(self.solver),
                {"step": int(generation)},
            )

    def on_evaluate_start(self, candidate, context: Optional[Any] = None):
        handler = _legacy_evaluation_handler(
            self.capability,
            "on_evaluate_start",
            "on_evaluate_start_ml",
        )
        if callable(handler):
            handler(self.solver, candidate, dict(context or {}))

    def on_evaluate_end(self, candidate, feedback, context: Optional[Any] = None):
        handler = _legacy_evaluation_handler(
            self.capability,
            "on_evaluate_end",
            "on_evaluate_end_ml",
        )
        if callable(handler):
            handler(self.solver, candidate, feedback, dict(context or {}))

    def on_generation_end(self, generation: int):
        handler = getattr(self.capability, "on_step_end", None)
        if callable(handler):
            handler(self.solver, _trainer_context(self.solver), _step_row(self.solver, generation))

    def on_solver_finish(self, result):
        handler = getattr(self.capability, "on_fit_end", None)
        if callable(handler):
            report = result.get("report", {}) if isinstance(result, Mapping) else result
            if not isinstance(report, Mapping):
                report = {"result": report}
            handler(self.solver, _trainer_context(self.solver), dict(report))

    def on_error(self, error: BaseException, context: Optional[Any] = None):
        handler = _legacy_evaluation_handler(
            self.capability,
            "on_error",
            "on_error_ml",
        )
        if callable(handler):
            handler(self.solver, error, dict(context or {}))


__all__ = [
    # Re-exported from blackbase.plugin
    "PluginBase",
    "PluginManager",
    "report_soft_error",
    # MLBlack-specific
    "Capability",
    "CapabilityPluginAdapter",
]
