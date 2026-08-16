"""
Component contract and mixin for declaring composition contracts.

ComponentContract is a frozen dataclass that serves as the serializable bridge
used by reports, catalog, and doctor. It extends the context-level contract
with component-level metadata (name, gradient/batch/resume support, metrics
requirements, notes).

ContractMixin provides a declarative way to attach contracts to components
via class attributes (context_requires, context_provides, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from blackbase.context import ContextContract, normalize_context_keys


@dataclass(frozen=True)
class ComponentContract:
    """Explicit composition contract for a framework component.

    Canonical component declarations are class attributes:
    context_requires/context_provides/context_mutates/context_cache,
    requires_metrics, metrics_fallback and context_notes. This dataclass remains
    the serializable bridge used by existing reports, catalog and doctor.
    """

    name: str = ""
    requires: tuple[str, ...] = tuple()
    optional: tuple[str, ...] = tuple()
    provides: tuple[str, ...] = tuple()
    mutates: tuple[str, ...] = tuple()
    cache: tuple[str, ...] = tuple()
    supports_gradient: bool | None = None
    supports_batch: bool | None = None
    supports_resume: bool | None = None
    requires_metrics: tuple[str, ...] = tuple()
    metrics_fallback: str = "strict"
    context_notes: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_context_contract(
        cls,
        contract: ContextContract,
        *,
        supports_gradient: bool | None = None,
        supports_batch: bool | None = None,
        supports_resume: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ComponentContract":
        """Create a ComponentContract from a ContextContract."""
        return cls(
            name=contract.name,
            requires=tuple(contract.requires),
            optional=tuple(contract.optional),
            provides=tuple(contract.provides),
            mutates=tuple(contract.mutates),
            cache=tuple(contract.cache),
            supports_gradient=supports_gradient,
            supports_batch=supports_batch,
            supports_resume=supports_resume,
            requires_metrics=tuple(contract.requires_metrics),
            metrics_fallback=contract.metrics_fallback,
            context_notes=contract.notes or "",
            metadata={**dict(contract.metadata), **dict(metadata or {})},
        )

    def to_context_contract(self) -> ContextContract:
        """Convert to a ContextContract."""
        return ContextContract(
            name=self.name,
            requires=self.requires,
            optional=self.optional,
            provides=self.provides,
            mutates=self.mutates,
            cache=self.cache,
            notes=self.context_notes or None,
            requires_metrics=self.requires_metrics,
            metrics_fallback=self.metrics_fallback,
            metadata=self.metadata,
        )

    def with_name(self, name: str) -> "ComponentContract":
        """Return a copy of this contract with a different name."""
        return ComponentContract(
            name=str(name),
            requires=self.requires,
            optional=self.optional,
            provides=self.provides,
            mutates=self.mutates,
            cache=self.cache,
            supports_gradient=self.supports_gradient,
            supports_batch=self.supports_batch,
            supports_resume=self.supports_resume,
            requires_metrics=self.requires_metrics,
            metrics_fallback=self.metrics_fallback,
            context_notes=self.context_notes,
            metadata=dict(self.metadata),
        )

    def merged(self, other: "ComponentContract", *, name: str | None = None) -> "ComponentContract":
        """Merge this contract with another, combining all fields."""
        return ComponentContract(
            name=str(name if name is not None else (self.name or other.name)),
            requires=_dedupe((*self.requires, *other.requires)),
            optional=_dedupe((*self.optional, *other.optional)),
            provides=_dedupe((*self.provides, *other.provides)),
            mutates=_dedupe((*self.mutates, *other.mutates)),
            cache=_dedupe((*self.cache, *other.cache)),
            supports_gradient=_merge_bool(self.supports_gradient, other.supports_gradient),
            supports_batch=_merge_bool(self.supports_batch, other.supports_batch),
            supports_resume=_merge_bool(self.supports_resume, other.supports_resume),
            requires_metrics=_dedupe((*self.requires_metrics, *other.requires_metrics)),
            metrics_fallback=other.metrics_fallback if other.metrics_fallback != "strict" else self.metrics_fallback,
            context_notes="; ".join(part for part in (self.context_notes, other.context_notes) if part),
            metadata={**dict(self.metadata), **dict(other.metadata)},
        )

    def describe(self) -> dict[str, Any]:
        """Return a full dictionary description of this contract."""
        context = self.to_context_contract().to_dict()
        return {
            "name": self.name,
            "requires": list(self.requires),
            "optional": list(self.optional),
            "provides": list(self.provides),
            "mutates": list(self.mutates),
            "cache": list(self.cache),
            "context_requires": list(self.requires),
            "context_optional": list(self.optional),
            "context_provides": list(self.provides),
            "context_mutates": list(self.mutates),
            "context_cache": list(self.cache),
            "requires_metrics": list(self.requires_metrics),
            "metrics_fallback": self.metrics_fallback,
            "context_notes": self.context_notes,
            "supports_gradient": self.supports_gradient,
            "supports_batch": self.supports_batch,
            "supports_resume": self.supports_resume,
            "metadata": dict(self.metadata),
            "context_contract": context,
        }


class ContractMixin:
    """Mixin for objects that expose a stable component contract.

    Subclasses declare their contract via class attributes:
    - context_requires, context_optional, context_provides, context_mutates, context_cache
    - requires_metrics, metrics_fallback, context_notes
    - contract (optional ComponentContract override)
    """

    context_requires: tuple[str, ...] = tuple()
    context_optional: tuple[str, ...] = tuple()
    context_provides: tuple[str, ...] = tuple()
    context_mutates: tuple[str, ...] = tuple()
    context_cache: tuple[str, ...] = tuple()
    requires_metrics: tuple[str, ...] = tuple()
    metrics_fallback: str = "strict"
    context_notes: str = ""
    contract: ComponentContract = ComponentContract()

    def get_context_contract(self) -> ContextContract:
        """Return the ContextContract for this component."""
        raw = getattr(self, "contract", ComponentContract())
        fallback = _coerce_contract(raw)
        return ContextContract.from_component(self, fallback_contract=fallback)

    def get_contract(self) -> ComponentContract:
        """Return the ComponentContract for this component."""
        raw = getattr(self, "contract", ComponentContract())
        fallback = _coerce_contract(raw)
        context_contract = ContextContract.from_component(self, fallback_contract=fallback)
        return ComponentContract.from_context_contract(
            context_contract,
            supports_gradient=fallback.supports_gradient,
            supports_batch=fallback.supports_batch,
            supports_resume=fallback.supports_resume,
            metadata=fallback.metadata,
        )


def combine_contracts(name: str, *contracts: ComponentContract) -> ComponentContract:
    """Combine multiple ComponentContracts into one with the given name."""
    merged = ComponentContract(name=name)
    for contract in contracts:
        merged = merged.merged(contract, name=name)
    return merged


def _coerce_contract(raw: Any) -> ComponentContract:
    """Coerce a raw value into a ComponentContract."""
    if isinstance(raw, ComponentContract):
        return raw
    if isinstance(raw, ContextContract):
        return ComponentContract.from_context_contract(raw)
    if isinstance(raw, Mapping):
        payload = dict(raw)
        for source, target in (
            ("context_requires", "requires"),
            ("context_optional", "optional"),
            ("context_provides", "provides"),
            ("context_mutates", "mutates"),
            ("context_cache", "cache"),
        ):
            if source in payload and target not in payload:
                payload[target] = payload[source]
        # Remove source aliases after transfer so they don't hit __init__
        for source in ("context_requires", "context_optional", "context_provides", "context_mutates", "context_cache"):
            payload.pop(source, None)
        for key in ("requires", "optional", "provides", "mutates", "cache"):
            if key in payload:
                payload[key] = normalize_context_keys(payload[key])
        return ComponentContract(**payload)
    raise TypeError("contract must be ComponentContract, ContextContract or mapping")


def _dedupe(values: tuple[str, ...]) -> tuple[str, ...]:
    """Remove duplicates while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = str(value)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return tuple(out)


def _merge_bool(left: bool | None, right: bool | None) -> bool | None:
    """Merge two optional booleans with AND semantics."""
    if left is None:
        return right
    if right is None:
        return left
    return bool(left and right)
