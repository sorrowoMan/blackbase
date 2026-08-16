"""
Context contract system for component interaction.

This module provides a unified interface for declaring context requirements
and capabilities across both NSGABlack and MLBlack components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .context_keys import (
    METRIC_FALLBACKS,
    METRIC_KEYS,
    normalize_context_key,
    normalize_context_keys,
    unknown_context_keys,
)

# Attribute name aliases for backward compatibility
_REQUIRES_ATTRS = ("context_requires", "requires_context_keys", "runtime_requires", "requires")
_PROVIDES_ATTRS = ("context_provides", "provides_context_keys", "runtime_provides", "provides")
_MUTATES_ATTRS = ("context_mutates", "mutates_context_keys", "runtime_mutates", "mutates")
_CACHE_ATTRS = ("context_cache", "cache_context_keys", "runtime_cache", "cache")
_ARTIFACT_REQUIRES_ATTRS = ("artifact_requires",)
_ARTIFACT_PROVIDES_ATTRS = ("artifact_provides",)
_PHASE_IN_ATTRS = ("phase_in",)
_PHASE_OUT_ATTRS = ("phase_out",)
_NOTES_ATTRS = (
    "context_notes",
    "recommended_mutators",
    "recommended_plugins",
    "companions",
    "recommended_suite",
)


def _normalize_fields(items: Optional[Iterable[Any]]) -> List[str]:
    """Normalize context field values to a sorted list of strings."""
    if not items:
        return []
    out: List[str] = []
    for item in items:
        if item is None:
            continue
        text = normalize_context_key(str(item))
        if text:
            out.append(text)
    return sorted(set(out))


def _normalize_notes(value: Any) -> Optional[str]:
    """Normalize notes field."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, Mapping):
        parts = []
        for k, v in value.items():
            key = str(k).strip()
            val = _normalize_notes(v)
            if not key or not val:
                continue
            parts.append(f"{key}: {val}")
        return "; ".join(parts) or None
    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            text = _normalize_notes(item)
            if text:
                parts.append(text)
        return "; ".join(parts) or None
    text = str(value).strip()
    return text or None


def _collect_attrs(obj: Any, names: Sequence[str]) -> List[Any]:
    """Collect attribute values from an object."""
    out: List[Any] = []
    for name in names:
        value = getattr(obj, name, None)
        if value is None:
            continue
        out.append(value)
    return out


def _merge_notes(*values: Any) -> Optional[str]:
    """Merge multiple notes into one."""
    parts: List[str] = []
    for value in values:
        text = _normalize_notes(value)
        if not text:
            continue
        parts.append(text)
    if not parts:
        return None
    uniq: List[str] = []
    seen = set()
    for item in parts:
        if item in seen:
            continue
        seen.add(item)
        uniq.append(item)
    return " | ".join(uniq)


def _flatten_values(values: Sequence[Any]) -> List[Any]:
    """Flatten nested sequences into a single list."""
    out: List[Any] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            out.extend(list(value))
        else:
            out.append(value)
    return out


@dataclass(frozen=True)
class ContextContract:
    """
    Unified context contract declaration for components.
    
    Supports both NSGABlack-style (requires/provides/mutates/cache) and
    MLBlack-style (context_requires/context_provides/etc.) attribute names.
    """
    
    name: str = ""

    # Primary fields (unified naming)
    requires: Sequence[str] = field(default_factory=tuple)
    optional: Sequence[str] = field(default_factory=tuple)
    provides: Sequence[str] = field(default_factory=tuple)
    mutates: Sequence[str] = field(default_factory=tuple)
    cache: Sequence[str] = field(default_factory=tuple)
    
    # Extended fields
    artifact_requires: Sequence[str] = field(default_factory=tuple)
    artifact_provides: Sequence[str] = field(default_factory=tuple)
    phase_in: Sequence[str] = field(default_factory=tuple)
    phase_out: Sequence[str] = field(default_factory=tuple)
    notes: Optional[str] = None
    requires_metrics: Sequence[str] = field(default_factory=tuple)
    metrics_fallback: str = "strict"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    
    def __init__(
        self,
        *,
        name: str = "",
        requires: Sequence[str] = (),
        optional: Sequence[str] = (),
        provides: Sequence[str] = (),
        mutates: Sequence[str] = (),
        cache: Sequence[str] = (),
        artifact_requires: Sequence[str] = (),
        artifact_provides: Sequence[str] = (),
        phase_in: Sequence[str] = (),
        phase_out: Sequence[str] = (),
        notes: Optional[str] = None,
        requires_metrics: Sequence[str] = (),
        metrics_fallback: str = "strict",
        metadata: Mapping[str, Any] | None = None,
        # MLBlack-style aliases
        context_requires: Sequence[str] = (),
        context_optional: Sequence[str] = (),
        context_provides: Sequence[str] = (),
        context_mutates: Sequence[str] = (),
        context_cache: Sequence[str] = (),
        requires_context_keys: Sequence[str] = (),
        provides_context_keys: Sequence[str] = (),
        mutates_context_keys: Sequence[str] = (),
        cache_context_keys: Sequence[str] = (),
        context_notes: Optional[str] = None,
    ):
        """
        Initialize a context contract.
        
        Supports both NSGABlack-style and MLBlack-style parameter names.
        When both styles are provided, they are merged.
        """
        # Merge both naming styles
        object.__setattr__(self, "name", str(name or ""))
        object.__setattr__(self, "requires", tuple(requires) + tuple(context_requires) + tuple(requires_context_keys))
        object.__setattr__(self, "optional", tuple(optional) + tuple(context_optional))
        object.__setattr__(self, "provides", tuple(provides) + tuple(context_provides) + tuple(provides_context_keys))
        object.__setattr__(self, "mutates", tuple(mutates) + tuple(context_mutates) + tuple(mutates_context_keys))
        object.__setattr__(self, "cache", tuple(cache) + tuple(context_cache) + tuple(cache_context_keys))
        object.__setattr__(self, "artifact_requires", tuple(artifact_requires))
        object.__setattr__(self, "artifact_provides", tuple(artifact_provides))
        object.__setattr__(self, "phase_in", tuple(phase_in))
        object.__setattr__(self, "phase_out", tuple(phase_out))
        object.__setattr__(self, "notes", _merge_notes(notes, context_notes))
        object.__setattr__(self, "requires_metrics", tuple(str(item) for item in requires_metrics))
        object.__setattr__(self, "metrics_fallback", str(metrics_fallback or "strict"))
        object.__setattr__(self, "metadata", dict(metadata or {}))
    
    def normalized(self) -> "ContextContract":
        """Return a normalized copy of this contract."""
        return ContextContract(
            name=self.name,
            requires=_normalize_fields(self.requires),
            optional=_normalize_fields(self.optional),
            provides=_normalize_fields(self.provides),
            mutates=_normalize_fields(self.mutates),
            cache=_normalize_fields(self.cache),
            artifact_requires=_normalize_fields(self.artifact_requires),
            artifact_provides=_normalize_fields(self.artifact_provides),
            phase_in=_normalize_fields(self.phase_in),
            phase_out=_normalize_fields(self.phase_out),
            notes=self.notes,
            requires_metrics=_normalize_fields(self.requires_metrics),
            metrics_fallback=self.metrics_fallback,
            metadata=self.metadata,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation."""
        norm = self.normalized()
        return {
            "name": norm.name,
            "requires": list(norm.requires),
            "optional": list(norm.optional),
            "provides": list(norm.provides),
            "mutates": list(norm.mutates),
            "cache": list(norm.cache),
            "artifact_requires": list(norm.artifact_requires),
            "artifact_provides": list(norm.artifact_provides),
            "phase_in": list(norm.phase_in),
            "phase_out": list(norm.phase_out),
            "notes": norm.notes,
            "requires_metrics": list(norm.requires_metrics),
            "metrics_fallback": norm.metrics_fallback,
            "metadata": dict(norm.metadata),
        }

    def as_dict(self) -> Dict[str, Any]:
        """Return the legacy mlblack-shaped dictionary representation."""

        norm = self.normalized()
        return {
            "name": norm.name,
            "context_requires": list(norm.requires),
            "context_optional": list(norm.optional),
            "context_provides": list(norm.provides),
            "context_mutates": list(norm.mutates),
            "context_cache": list(norm.cache),
            "requires_metrics": list(norm.requires_metrics),
            "metrics_fallback": norm.metrics_fallback,
            "context_notes": norm.notes or "",
            "metadata": dict(norm.metadata),
        }
    
    def merge(self, other: "ContextContract") -> "ContextContract":
        """Merge this contract with another."""
        a = self.normalized()
        b = other.normalized()
        return ContextContract(
            name=b.name or a.name,
            requires=sorted(set(a.requires) | set(b.requires)),
            optional=sorted(set(a.optional) | set(b.optional)),
            provides=sorted(set(a.provides) | set(b.provides)),
            mutates=sorted(set(a.mutates) | set(b.mutates)),
            cache=sorted(set(a.cache) | set(b.cache)),
            artifact_requires=sorted(set(a.artifact_requires) | set(b.artifact_requires)),
            artifact_provides=sorted(set(a.artifact_provides) | set(b.artifact_provides)),
            phase_in=sorted(set(a.phase_in) | set(b.phase_in)),
            phase_out=sorted(set(a.phase_out) | set(b.phase_out)),
            notes=_merge_notes(a.notes, b.notes),
            requires_metrics=sorted(set(a.requires_metrics) | set(b.requires_metrics)),
            metrics_fallback=(
                b.metrics_fallback
                if b.metrics_fallback != "strict"
                else a.metrics_fallback
            ),
            metadata={**dict(a.metadata), **dict(b.metadata)},
        )
    
    def all_context_keys(self) -> tuple[str, ...]:
        """Return all context keys declared in this contract."""
        return normalize_context_keys((
            *self.requires,
            *self.optional,
            *self.provides,
            *self.mutates,
            *self.cache,
        ))

    def unknown_keys(self) -> tuple[str, ...]:
        """Return declared context keys that are absent from the shared registry."""

        return unknown_context_keys(self.all_context_keys())

    def unknown_metric_keys(self) -> tuple[str, ...]:
        """Return required metric names absent from the shared metric registry."""

        known = {str(key) for key in METRIC_KEYS}
        return tuple(
            str(metric)
            for metric in self.requires_metrics
            if str(metric) not in known
        )

    def validate(self, *, strict: bool = False) -> tuple[str, ...]:
        """Validate declared context keys, metrics, and fallback policy."""

        unknown = self.unknown_keys()
        unknown_metrics = self.unknown_metric_keys()
        invalid_fallback = self.metrics_fallback not in METRIC_FALLBACKS
        if strict and unknown:
            raise ValueError(f"{self.name} declares unknown context keys: {unknown}")
        if strict and unknown_metrics:
            raise ValueError(f"{self.name} declares unknown metric keys: {unknown_metrics}")
        if strict and invalid_fallback:
            raise ValueError(
                f"{self.name} declares invalid metrics_fallback: {self.metrics_fallback}"
            )
        return unknown

    @property
    def context_requires(self) -> tuple[str, ...]:
        return tuple(self.requires)

    @property
    def context_optional(self) -> tuple[str, ...]:
        return tuple(self.optional)

    @property
    def context_provides(self) -> tuple[str, ...]:
        return tuple(self.provides)

    @property
    def context_mutates(self) -> tuple[str, ...]:
        return tuple(self.mutates)

    @property
    def context_cache(self) -> tuple[str, ...]:
        return tuple(self.cache)

    @property
    def context_notes(self) -> str:
        return self.notes or ""
    
    @classmethod
    def from_component(cls, component: Any, *, fallback_contract: Any | None = None) -> "ContextContract":
        """
        Create a context contract from a component instance or class.
        
        Supports both NSGABlack and MLBlack naming conventions.
        """
        source = component if isinstance(component, type) else type(component)
        fallback = fallback_contract
        
        # Get name
        name = str(getattr(component, "name", getattr(source, "name", getattr(fallback, "name", source.__name__))))
        
        return cls(
            name=name,
            requires=_read_keys(component, source, "requires", getattr(fallback, "requires", ())),
            optional=_read_keys(component, source, "optional", getattr(fallback, "optional", ())),
            provides=_read_keys(component, source, "provides", getattr(fallback, "provides", ())),
            mutates=_read_keys(component, source, "mutates", getattr(fallback, "mutates", ())),
            cache=_read_keys(component, source, "cache", getattr(fallback, "cache", ())),
            artifact_requires=_read_keys(component, source, "artifact_requires", ()),
            artifact_provides=_read_keys(component, source, "artifact_provides", ()),
            phase_in=_read_keys(component, source, "phase_in", ()),
            phase_out=_read_keys(component, source, "phase_out", ()),
            notes=_merge_notes(
                *_collect_attrs(component, _NOTES_ATTRS),
                getattr(fallback, "context_notes", getattr(fallback, "notes", None)),
            ),
            requires_metrics=_read_keys(
                component,
                source,
                "requires_metrics",
                getattr(fallback, "requires_metrics", ()),
            ),
            metrics_fallback=str(
                getattr(
                    component,
                    "metrics_fallback",
                    getattr(source, "metrics_fallback", getattr(fallback, "metrics_fallback", "strict")),
                )
            ),
            metadata=dict(getattr(fallback, "metadata", {}) or {}),
        )


def _read_keys(component: Any, source: Any, attr: str, fallback: Iterable[str]) -> tuple[str, ...]:
    """Read context keys from component with fallback support."""
    # Try both naming conventions
    unified_attr = attr
    legacy_attr = f"context_{attr}"
    
    if hasattr(component, unified_attr):
        return normalize_context_keys(getattr(component, unified_attr))
    if hasattr(source, unified_attr):
        return normalize_context_keys(getattr(source, unified_attr))
    if hasattr(component, legacy_attr):
        return normalize_context_keys(getattr(component, legacy_attr))
    if hasattr(source, legacy_attr):
        return normalize_context_keys(getattr(source, legacy_attr))
    return normalize_context_keys(fallback)


def get_component_contract(obj: Any) -> Optional[ContextContract]:
    """Extract context contract from any object."""
    if obj is None:
        return None
    
    # Try explicit contract method first
    getter = getattr(obj, "get_context_contract", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:
            value = None
        if isinstance(value, ContextContract):
            return value.normalized()
        if isinstance(value, Mapping):
            notes = _merge_notes(
                value.get("notes"),
                value.get("context_notes"),
                value.get("recommended_mutators"),
                value.get("recommended_plugins"),
                value.get("companions"),
                value.get("recommended_suite"),
            )
            return ContextContract(
                requires=list(value.get("requires", ()) or ())
                + list(value.get("context_requires", ()) or ())
                + list(value.get("requires_context_keys", ()) or ())
                + list(value.get("runtime_requires", ()) or ()),
                provides=list(value.get("provides", ()) or ())
                + list(value.get("context_provides", ()) or ())
                + list(value.get("provides_context_keys", ()) or ())
                + list(value.get("runtime_provides", ()) or ()),
                mutates=list(value.get("mutates", ()) or ())
                + list(value.get("context_mutates", ()) or ())
                + list(value.get("mutates_context_keys", ()) or ())
                + list(value.get("runtime_mutates", ()) or ()),
                cache=list(value.get("cache", ()) or ())
                + list(value.get("context_cache", ()) or ())
                + list(value.get("cache_context_keys", ()) or ())
                + list(value.get("runtime_cache", ()) or ()),
                artifact_requires=list(value.get("artifact_requires", ()) or ()),
                artifact_provides=list(value.get("artifact_provides", ()) or ()),
                phase_in=list(value.get("phase_in", ()) or ()),
                phase_out=list(value.get("phase_out", ()) or ()),
                notes=notes,
            ).normalized()
    
    # Collect from class/instance attributes
    requires_values = _collect_attrs(obj, _REQUIRES_ATTRS)
    provides_values = _collect_attrs(obj, _PROVIDES_ATTRS)
    mutates_values = _collect_attrs(obj, _MUTATES_ATTRS)
    cache_values = _collect_attrs(obj, _CACHE_ATTRS)
    artifact_requires_values = _collect_attrs(obj, _ARTIFACT_REQUIRES_ATTRS)
    artifact_provides_values = _collect_attrs(obj, _ARTIFACT_PROVIDES_ATTRS)
    phase_in_values = _collect_attrs(obj, _PHASE_IN_ATTRS)
    phase_out_values = _collect_attrs(obj, _PHASE_OUT_ATTRS)
    notes = _merge_notes(*_collect_attrs(obj, _NOTES_ATTRS))
    
    if any([
        requires_values,
        provides_values,
        mutates_values,
        cache_values,
        artifact_requires_values,
        artifact_provides_values,
        phase_in_values,
        phase_out_values,
        notes,
    ]):
        return ContextContract(
            requires=_flatten_values(requires_values),
            provides=_flatten_values(provides_values),
            mutates=_flatten_values(mutates_values),
            cache=_flatten_values(cache_values),
            artifact_requires=_flatten_values(artifact_requires_values),
            artifact_provides=_flatten_values(artifact_provides_values),
            phase_in=_flatten_values(phase_in_values),
            phase_out=_flatten_values(phase_out_values),
            notes=notes,
        ).normalized()
    
    return None


def collect_solver_contracts(solver: Any) -> List[Tuple[str, ContextContract]]:
    """Collect all context contracts from a solver and its components."""
    contracts: List[Tuple[str, ContextContract]] = []
    seen: set[Tuple[str, int]] = set()
    
    def _add(name: str, obj: Any) -> None:
        if obj is None:
            return
        marker = (str(name), id(obj))
        if marker in seen:
            return
        seen.add(marker)
        contract = get_component_contract(obj)
        if contract is not None:
            contracts.append((name, contract))
    
    _add("representation_pipeline", getattr(solver, "representation_pipeline", None))
    _add("bias_module", getattr(solver, "bias_module", None))
    
    adapter = getattr(solver, "adapter", None)
    _add("adapter", adapter)
    if adapter is not None:
        for idx, spec in enumerate(getattr(adapter, "strategies", ()) or ()):
            sub = getattr(spec, "adapter", None)
            name = str(getattr(spec, "name", f"strategy_{idx}"))
            _add(f"adapter.strategy.{name}", sub)
        for idx, role in enumerate(getattr(adapter, "roles", ()) or ()):
            role_name = str(getattr(role, "name", f"role_{idx}"))
            role_adapter = getattr(role, "adapter", None)
            _add(f"adapter.role.{role_name}", role_adapter if not callable(role_adapter) else None)
        for unit in getattr(adapter, "units", ()) or ():
            role_name = str(getattr(unit, "role", "role"))
            unit_id = int(getattr(unit, "unit_id", 0))
            _add(f"adapter.unit.{role_name}#{unit_id}", getattr(unit, "adapter", None))
    
    plugin_manager = getattr(solver, "plugin_manager", None)
    if plugin_manager is not None:
        plugins = getattr(plugin_manager, "plugins", None) or []
        for plugin in plugins:
            name = getattr(plugin, "name", plugin.__class__.__name__)
            _add(f"plugin.{name}", plugin)
    
    return contracts


def validate_context_contracts(
    contracts: Sequence[Tuple[str, ContextContract]],
    context: Mapping[str, Any],
) -> List[str]:
    """Validate contracts against available context keys."""
    warnings: List[str] = []
    ctx_keys = set(context.keys())
    for name, contract in contracts:
        missing = [k for k in contract.requires if k not in ctx_keys]
        if missing:
            warnings.append(f"{name} missing required context keys: {', '.join(missing)}")
    return warnings


def detect_context_conflicts(
    contracts: Sequence[Tuple[str, ContextContract]],
) -> List[str]:
    """Detect potential multi-writer risks on the same context key."""
    writers_by_key: Dict[str, List[str]] = {}
    for name, contract in contracts:
        keys = set(contract.provides) | set(contract.mutates)
        for key in keys:
            k = str(key).strip()
            if not k:
                continue
            writers_by_key.setdefault(k, []).append(str(name))
    
    issues: List[str] = []
    for key, writers in sorted(writers_by_key.items(), key=lambda x: x[0]):
        unique = sorted(set(writers))
        if len(unique) <= 1:
            continue
        issues.append(f"{key}: " + ", ".join(unique))
    return issues
