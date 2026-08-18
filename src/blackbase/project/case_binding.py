"""Shared post-build binding contract for standard Case objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from blackbase.resources import ResourceContext

from .execution import ProjectConfigurationError


CASE_RESOURCE_BINDING_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CaseResourceBindingAudit:
    """Evidence that a built Case exposes the Project-authorized grant."""

    status: str
    method: str
    expected: Mapping[str, Any]
    effective: Mapping[str, Any]
    schema_version: int = CASE_RESOURCE_BINDING_SCHEMA_VERSION

    @property
    def current(self) -> bool:
        return self.status == "bound" and dict(self.expected) == dict(self.effective)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "status": str(self.status),
            "method": str(self.method),
            "current": bool(self.current),
            "expected": dict(self.expected),
            "effective": dict(self.effective),
        }


def bind_case_resource_context(
    case_obj: Any,
    resource_context: Mapping[str, Any] | ResourceContext | None,
) -> CaseResourceBindingAudit | None:
    """Bind the authoritative Project grant to a freshly built Case.

    Builders still receive ``resource_context`` for assembly-time decisions.
    This post-build step closes the runtime side of the contract so a builder
    cannot accidentally accept and then discard the grant.
    """

    if resource_context is None:
        return None
    if case_obj is None:
        raise ProjectConfigurationError("canonical Case builder returned None")

    payload = _as_mapping(resource_context)
    expected = ResourceContext.from_mapping(payload).as_dict()
    setter = getattr(case_obj, "set_resource_context", None)
    if callable(setter):
        setter(payload)
        method = "set_resource_context"
    else:
        try:
            setattr(case_obj, "resource_context", payload)
        except (AttributeError, TypeError) as exc:
            raise ProjectConfigurationError(
                "built Case must implement set_resource_context(context) or allow "
                "the shared substrate to bind resource_context"
            ) from exc
        method = "resource_context_attribute"

    getter = getattr(case_obj, "get_resource_context", None)
    effective_value = getter() if callable(getter) else getattr(case_obj, "resource_context", None)
    if effective_value is None:
        raise ProjectConfigurationError(
            "built Case did not expose an effective resource_context after binding"
        )
    effective = ResourceContext.from_mapping(effective_value).as_dict()
    audit = CaseResourceBindingAudit(
        status="bound" if effective == expected else "mismatch",
        method=method,
        expected=expected,
        effective=effective,
    )
    if not audit.current:
        raise ProjectConfigurationError(
            "built Case changed the Project-authorized ResourceContext during binding"
        )
    try:
        setattr(case_obj, "resource_binding_audit", audit.as_dict())
    except (AttributeError, TypeError) as exc:
        raise ProjectConfigurationError(
            "built Case must expose resource_binding_audit for execution evidence"
        ) from exc
    return audit


def case_resource_binding_audit(case_obj: Any) -> dict[str, Any]:
    value = getattr(case_obj, "resource_binding_audit", None)
    if isinstance(value, CaseResourceBindingAudit):
        return value.as_dict()
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _as_mapping(value: Mapping[str, Any] | ResourceContext) -> dict[str, Any]:
    if isinstance(value, ResourceContext):
        return value.as_dict()
    return dict(value)


__all__ = [
    "CASE_RESOURCE_BINDING_SCHEMA_VERSION",
    "CaseResourceBindingAudit",
    "bind_case_resource_context",
    "case_resource_binding_audit",
]
