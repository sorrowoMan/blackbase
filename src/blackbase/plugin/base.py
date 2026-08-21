"""Plugin base classes and plugin manager.

Design:
- ``PluginBase`` is the abstract base class with lifecycle hooks and metadata.
  Frameworks inherit from it and add their own specialized methods.
- ``PluginManager`` is the complete implementation for plugin registration,
  lifecycle dispatch, and short-circuit evaluation. It is shared infrastructure
  because both nsgablack and mlblack use the same dispatch semantics.
- Framework-specific helpers (e.g. ``get_population_snapshot``) live in
  the downstream semantic framework itself.
"""

from __future__ import annotations

from abc import ABC
import copy
import logging
import time
import traceback
import warnings
from typing import Any, Dict, Mapping, Optional

from ._soft_error import report_soft_error

logger = logging.getLogger(__name__)


class PluginBase(ABC):
    """Abstract base class for all plugins / capabilities.

    Subclasses must set ``name`` and may override any lifecycle hook.
    Framework-specific plugins can add their own specialized methods.
    """

    # Optional context contract metadata (class-level defaults)
    context_requires: tuple = ()
    context_provides: tuple = ()
    context_mutates: tuple = ()
    context_cache: tuple = ()
    artifact_requires: tuple = ()
    artifact_provides: tuple = ()
    phase_in: tuple = ()
    phase_out: tuple = ()
    context_notes: str | None = None

    def __init__(self, name: str, solver=None, priority: int = 0):
        self.name = name
        self.solver = solver
        self.enabled = True
        self.config: Dict[str, Any] = {}
        self.priority = priority

        self.is_algorithmic = False
        self._profile = {"total_s": 0.0, "events": {}}

    def get_context_contract(self) -> Dict[str, Any]:
        """Return context contract metadata for this plugin."""
        return {
            "requires": getattr(self, "context_requires", ()),
            "provides": getattr(self, "context_provides", ()),
            "mutates": getattr(self, "context_mutates", ()),
            "cache": getattr(self, "context_cache", ()),
            "artifact_requires": getattr(self, "artifact_requires", ()),
            "artifact_provides": getattr(self, "artifact_provides", ()),
            "phase_in": getattr(self, "phase_in", ()),
            "phase_out": getattr(self, "phase_out", ()),
            "notes": getattr(self, "context_notes", None),
        }

    def attach(self, solver):
        """Attach this plugin to a solver/trainer."""
        self.solver = solver

    def detach(self):
        """Detach this plugin from its solver/trainer."""
        self.solver = None

    def enable(self):
        """Enable this plugin."""
        self.enabled = True

    def disable(self):
        """Disable this plugin."""
        self.enabled = False

    def configure(self, **kwargs):
        """Update plugin config."""
        self.config.update(kwargs)

    def get_config(self, key: str, default=None):
        """Get one config value."""
        return self.config.get(key, default)

    # --- Lifecycle hooks (all optional) ---

    def prepare_restore(self, solver):
        """Resolve and queue a restore envelope after setup, before init hooks.

        Implementations must not mutate live Solver state directly.  They may
        validate/load external state and queue exactly one restore transaction
        through the Solver's restore-envelope surface.
        """
        return None

    def on_solver_init(self, solver):
        """Called when the solver/trainer starts. (nsgablack: on_solver_init, mlblack: on_fit_start)"""
        return None

    def on_population_init(self, population, objectives, violations):
        """Called after initial population is created. (nsgablack-specific)"""
        return None

    def on_generation_start(self, generation: int):
        """Called at the start of each generation/step. (nsgablack: on_generation_start, mlblack: on_step_start)"""
        return None

    def on_generation_end(self, generation: int):
        """Called at the end of each generation/step. (nsgablack: on_generation_end, mlblack: on_step_end)"""
        return None

    def on_step(self, solver, generation: int):
        """Called after the generation step. (nsgablack-specific)"""
        return None

    def on_solver_finish(self, result):
        """Called when the solver/trainer finishes. (nsgablack: on_solver_finish, mlblack: on_fit_end)"""
        return None

    def on_context_build(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Inject keys into the context dict. Return the (possibly modified) context."""
        return context

    def on_evaluate_start(self, candidate, context: Optional[Dict[str, Any]] = None):
        """Called before evaluating a single candidate."""
        return None

    def on_evaluate_end(self, candidate, feedback, context: Optional[Dict[str, Any]] = None):
        """Called after evaluating a single candidate."""
        return None

    def on_error(self, error: BaseException, context: Optional[Dict[str, Any]] = None):
        """Called when an exception is raised during the run loop."""
        return None

    def get_report(self) -> Optional[Dict[str, Any]]:
        """Return a small algorithmic report; tool-only plugins should return None."""
        if not bool(getattr(self, "is_algorithmic", False)):
            return None
        try:
            return {"config": dict(self.config)}
        except Exception as exc:
            report_soft_error(
                component="PluginBase",
                event="get_report_config_copy",
                exc=exc,
                logger=logger,
                strict=False,
                level="debug",
            )
            return {"config": {}}

    def __repr__(self):
        status = "enabled" if self.enabled else "disabled"
        return f"<{self.__class__.__name__}({self.name}, {status})>"


class PluginManager:
    """Manage plugin registration, lifecycle callbacks, and dispatch.

    This is shared infrastructure — both nsgablack and mlblack use the
    same dispatch semantics (priority ordering, short-circuit, profiling,
    soft-error handling).
    """

    def __init__(
        self,
        short_circuit: bool = False,
        short_circuit_events: Optional[list] = None,
        *,
        strict: bool = False,
    ):
        self.plugins: list[PluginBase] = []
        self.plugin_map: Dict[str, PluginBase] = {}
        self.short_circuit = short_circuit
        self.short_circuit_events = set(short_circuit_events or [])
        self.strict = bool(strict)
        self._solver = None
        self._context_build_writers: Dict[str, str] = {}
        self.event_hook = None

    def set_event_hook(self, hook) -> None:
        """Register a lightweight instrumentation hook for plugin events."""
        self.event_hook = hook

    def _emit_event_hook(self, payload: Dict[str, Any]) -> None:
        hook = self.event_hook
        if callable(hook):
            try:
                hook(payload)
            except Exception as exc:
                report_soft_error(
                    component="PluginManager",
                    event="event_hook",
                    exc=exc,
                    logger=logger,
                    strict=False,
                    level="debug",
                )

    @staticmethod
    def _safe_values_differ(a: Any, b: Any) -> bool:
        if a is b:
            return False
        try:
            neq = a != b
            if isinstance(neq, bool):
                return bool(neq)
            return True
        except Exception as exc:
            report_soft_error(
                component="PluginManager",
                event="safe_values_differ",
                exc=exc,
                logger=logger,
                strict=False,
                level="debug",
            )
            return True

    def _collect_changed_keys(self, before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
        keys = set(before.keys()) | set(after.keys())
        changed: list[str] = []
        for key in keys:
            if key not in before or key not in after:
                changed.append(str(key))
                continue
            if self._safe_values_differ(before[key], after[key]):
                changed.append(str(key))
        return changed

    def register(self, plugin: PluginBase):
        """Register a plugin."""
        if plugin.name in self.plugin_map:
            raise ValueError(f"Plugin '{plugin.name}' already registered")
        self.plugins.append(plugin)
        self.plugins.sort(key=lambda p: p.priority)
        self.plugin_map[plugin.name] = plugin
        return plugin

    def unregister(self, plugin_name: str):
        """Unregister one plugin by name."""
        if plugin_name not in self.plugin_map:
            raise ValueError(f"Plugin '{plugin_name}' not found")
        plugin = self.plugin_map[plugin_name]
        self.plugins.remove(plugin)
        del self.plugin_map[plugin_name]

    def get(self, plugin_name: str) -> Optional[PluginBase]:
        """Return plugin instance by name, or None."""
        return self.plugin_map.get(plugin_name)

    def enable(self, plugin_name: str):
        plugin = self.get(plugin_name)
        if plugin:
            plugin.enable()

    def disable(self, plugin_name: str):
        plugin = self.get(plugin_name)
        if plugin:
            plugin.disable()

    def set_execution_order(self, ordered_plugin_names) -> None:
        """Reorder registered plugins by explicit name list."""
        by_name = {str(p.name): p for p in self.plugins}
        seen = set()
        reordered = []
        for name in (ordered_plugin_names or ()):
            key = str(name)
            plugin = by_name.get(key)
            if plugin is None or key in seen:
                continue
            reordered.append(plugin)
            seen.add(key)
        for plugin in self.plugins:
            key = str(plugin.name)
            if key in seen:
                continue
            reordered.append(plugin)
        self.plugins = reordered

    def trigger(self, event_name: str, *args, **kwargs):
        """Trigger an event on plugins.

        If short-circuit is enabled for this event, returns the first non-None
        handler result.
        """
        should_short_circuit = self.short_circuit and event_name in self.short_circuit_events

        for plugin in self.plugins:
            if not plugin.enabled:
                continue
            if bool(getattr(plugin, "_attach_failed", False)):
                continue

            handler = getattr(plugin, event_name, None)
            if handler and callable(handler):
                try:
                    t0 = time.time()
                    result = handler(*args, **kwargs)
                    dt = max(0.0, float(time.time() - t0))
                    prof = getattr(plugin, "_profile", None)
                    if isinstance(prof, dict):
                        prof["total_s"] = float(prof.get("total_s", 0.0) or 0.0) + dt
                        events = prof.get("events")
                        if not isinstance(events, dict):
                            events = {}
                            prof["events"] = events
                        events[event_name] = float(events.get(event_name, 0.0) or 0.0) + dt

                    self._emit_event_hook(
                        {
                            "mode": "trigger",
                            "event_name": str(event_name),
                            "plugin_name": str(plugin.name),
                            "plugin_class": plugin.__class__.__name__,
                            "plugin": plugin,
                            "status": "ok",
                            "has_result": result is not None,
                        }
                    )

                    if should_short_circuit and result is not None:
                        return result
                    if (not should_short_circuit) and result is not None:
                        warnings.warn(
                            (
                                f"Plugin '{plugin.name}' returned a non-None result for event "
                                f"'{event_name}' (ignored). Enable short_circuit and add the event "
                                "to short_circuit_events to allow returning values."
                            ),
                            RuntimeWarning,
                            stacklevel=2,
                        )
                except Exception as e:
                    self._emit_event_hook(
                        {
                            "mode": "trigger",
                            "event_name": str(event_name),
                            "plugin_name": str(plugin.name),
                            "plugin_class": plugin.__class__.__name__,
                            "plugin": plugin,
                            "status": "error",
                            "error_type": e.__class__.__name__,
                        }
                    )
                    if self.strict:
                        raise RuntimeError(
                            f"Plugin '{plugin.name}' failed in event '{event_name}': {e}"
                        ) from e
                    print(
                        f"[WARNING] Plugin {plugin.name} failed to handle {event_name}: {e}\n"
                        f"{traceback.format_exc()}"
                    )

    def dispatch(self, event_name: str, *args, **kwargs):
        """Dispatch an event and return the last non-None result."""
        out = None
        context_writers: Dict[str, str] = {}
        is_context_build = (
            str(event_name) == "on_context_build"
            and len(args) >= 1
            and isinstance(args[0], dict)
        )
        current_context = args[0] if is_context_build else None
        for plugin in self.plugins:
            if not plugin.enabled:
                continue
            if bool(getattr(plugin, "_attach_failed", False)):
                continue
            handler = getattr(plugin, event_name, None)
            if handler and callable(handler):
                before_ctx = None
                if is_context_build:
                    try:
                        before_ctx = copy.deepcopy(current_context)
                    except Exception as exc:
                        report_soft_error(
                            component="PluginManager",
                            event="dispatch.deepcopy_context",
                            exc=exc,
                            logger=logger,
                            strict=False,
                            level="debug",
                        )
                        before_ctx = dict(current_context or {})
                try:
                    call_args = (
                        (current_context, *args[1:])
                        if is_context_build
                        else args
                    )
                    result = handler(*call_args, **kwargs)
                except Exception as exc:
                    self._emit_event_hook(
                        {
                            "mode": "dispatch",
                            "event_name": str(event_name),
                            "plugin_name": str(plugin.name),
                            "plugin_class": plugin.__class__.__name__,
                            "plugin": plugin,
                            "status": "error",
                            "error_type": exc.__class__.__name__,
                        }
                    )
                    if self.strict:
                        raise RuntimeError(
                            f"Plugin '{plugin.name}' dispatch failed on event '{event_name}': {exc}"
                        ) from exc
                    warnings.warn(
                        (
                            f"Plugin '{plugin.name}' dispatch failed on event '{event_name}': {exc}\n"
                            f"{traceback.format_exc()}"
                        ),
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                self._emit_event_hook(
                    {
                        "mode": "dispatch",
                        "event_name": str(event_name),
                        "plugin_name": str(plugin.name),
                        "plugin_class": plugin.__class__.__name__,
                        "plugin": plugin,
                        "status": "ok",
                        "has_result": result is not None,
                    }
                )

                if is_context_build and before_ctx is not None:
                    after_ctx = result if isinstance(result, dict) else current_context
                    if isinstance(after_ctx, dict):
                        changed = self._collect_changed_keys(before_ctx, after_ctx)
                        source = f"plugin.{plugin.name}"
                        for key in changed:
                            context_writers[str(key)] = source
                        current_context = after_ctx

                if result is not None:
                    out = result

        if is_context_build:
            self._context_build_writers = context_writers
            if self._solver is not None:
                try:
                    setattr(self._solver, "_context_build_writers", dict(context_writers))
                except Exception as exc:
                    report_soft_error(
                        component="PluginManager",
                        event="dispatch.set_context_build_writers",
                        exc=exc,
                        logger=logger,
                        strict=False,
                        level="debug",
                    )
        if is_context_build:
            return current_context
        return out

    def prepare_restore(self, solver):
        """Collect restore requests before any ordinary init hook observes state."""
        self._solver = solver
        for plugin in self.plugins:
            if not plugin.enabled or bool(getattr(plugin, "_attach_failed", False)):
                continue
            try:
                if getattr(plugin, "solver", None) is not solver:
                    plugin.attach(solver)
            except Exception as exc:
                plugin._attach_failed = True
                plugin._attach_error = str(exc)
                if bool(getattr(solver, "plugin_strict", False)):
                    raise
                report_soft_error(
                    component="PluginManager",
                    event="prepare_restore.attach",
                    exc=exc,
                    logger=logger,
                    strict=False,
                    level="warning",
                )
                continue
            try:
                plugin.prepare_restore(solver)
            except Exception as exc:
                strict_restore = bool(
                    getattr(plugin, "raise_on_restore_error", False)
                ) or bool(getattr(plugin, "raise_on_init_error", False))
                if strict_restore or bool(getattr(solver, "plugin_strict", False)):
                    raise
                warnings.warn(
                    f"Plugin '{plugin.name}' restore preparation failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                report_soft_error(
                    component="PluginManager",
                    event="prepare_restore.hook",
                    exc=exc,
                    logger=logger,
                    strict=False,
                    level="warning",
                )

    def on_solver_init(self, solver):
        """Notify all plugins that solver init has started."""
        self._solver = solver
        self._context_build_writers = {}
        try:
            setattr(solver, "_context_build_writers", {})
        except Exception as exc:
            report_soft_error(
                component="PluginManager",
                event="on_solver_init.set_context_build_writers",
                exc=exc,
                logger=logger,
                strict=False,
                level="debug",
            )
        for plugin in self.plugins:
            if plugin.enabled:
                if bool(getattr(plugin, "_attach_failed", False)):
                    continue
                try:
                    if getattr(plugin, "solver", None) is not solver:
                        plugin.attach(solver)
                except Exception as exc:
                    plugin._attach_failed = True
                    plugin._attach_error = str(exc)
                    if bool(getattr(solver, "plugin_strict", False)):
                        raise
                    report_soft_error(
                        component="PluginManager",
                        event="on_solver_init.attach",
                        exc=exc,
                        logger=logger,
                        strict=False,
                        level="warning",
                    )
                    continue
                try:
                    plugin.on_solver_init(solver)
                except Exception as exc:
                    strict_init = bool(getattr(plugin, "raise_on_init_error", False)) or bool(
                        getattr(plugin, "strict_init", False)
                    )
                    if strict_init:
                        raise
                    warnings.warn(
                        f"Plugin '{plugin.name}' init failed: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    report_soft_error(
                        component="PluginManager",
                        event="on_solver_init.hook",
                        exc=exc,
                        logger=logger,
                        strict=False,
                        level="warning",
                    )

    def on_population_init(self, population, objectives, violations):
        self.trigger("on_population_init", population, objectives, violations)

    def on_generation_start(self, generation: int):
        self.trigger("on_generation_start", generation)

    def on_generation_end(self, generation: int):
        self.trigger("on_generation_end", generation)

    def on_step(self, solver, generation: int):
        self.trigger("on_step", solver, generation)

    def on_solver_finish(self, result):
        self.trigger("on_solver_finish", result)

    def on_context_build(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch context build to all plugins; returns the last non-None result."""
        return self.dispatch("on_context_build", context)

    def on_evaluate_start(self, candidate, context: Optional[Dict[str, Any]] = None):
        self.trigger("on_evaluate_start", candidate, context)

    def on_evaluate_end(self, candidate, feedback, context: Optional[Dict[str, Any]] = None):
        self.trigger("on_evaluate_end", candidate, feedback, context)

    def on_error(self, error: BaseException, context: Optional[Dict[str, Any]] = None):
        self.trigger("on_error", error, context)

    def list_plugins(self, enabled_only: bool = False) -> list:
        """List plugins, optionally only enabled ones."""
        if enabled_only:
            return [p for p in self.plugins if p.enabled]
        return self.plugins.copy()

    def clear(self):
        """Clear all plugins."""
        self.plugins.clear()
        self.plugin_map.clear()

    def __len__(self):
        return len(self.plugins)

    def __repr__(self):
        enabled_count = sum(1 for p in self.plugins if p.enabled)
        return f"<PluginManager({len(self.plugins)} plugins, {enabled_count} enabled)>"
