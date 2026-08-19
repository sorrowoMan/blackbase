"""Auditable build-check output shared by standard framework Cases."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .case_binding import case_resource_binding_audit


_RESOURCE_KEYS = (
    "device",
    "threads",
    "workers",
    "worker_count",
    "namespace",
    "backend",
)


def build_case_check_payload(
    case: Any,
    *,
    resource_context: Any = None,
) -> dict[str, Any]:
    """Describe components actually attached to a built Solver or Trainer."""

    pipeline = (
        getattr(case, "representation_pipeline", None)
        or getattr(case, "pipeline", None)
        or getattr(case, "representation", None)
        or getattr(case, "feature_builder", None)
    )
    describe_pipeline = getattr(pipeline, "describe", None)
    pipeline_description = describe_pipeline() if callable(describe_pipeline) else {}
    if not isinstance(pipeline_description, Mapping):
        pipeline_description = {}

    manager = getattr(case, "plugin_manager", None)
    plugins = tuple(
        getattr(manager, "plugins", ())
        or getattr(case, "plugins", ())
        or ()
    )
    mediator = getattr(case, "evaluation_mediator", None)
    list_providers = getattr(mediator, "list_providers", None)
    mediator_providers = tuple(list_providers()) if callable(list_providers) else ()
    providers = _attached_providers(case, mediator_providers)

    effective_resource_context = resource_context
    if effective_resource_context is None:
        effective_resource_context = getattr(case, "resource_context", None)

    return {
        "status": "assembly ok",
        "assembly": type(case).__name__,
        "problem": _component_name(getattr(case, "problem", None)),
        "pipeline": str(pipeline_description.get("class") or _component_name(pipeline)),
        "pipeline_variant": _pipeline_variant(pipeline_description),
        "initializer": _pipeline_component_name(pipeline, pipeline_description, "initializer"),
        "mutator": _pipeline_component_name(pipeline, pipeline_description, "mutator"),
        "repair": _pipeline_component_name(pipeline, pipeline_description, "repair"),
        "adapter": _component_name(getattr(case, "adapter", None)),
        "providers": [_provider_name(provider) for provider in providers],
        "provider_details": [_provider_details(provider) for provider in providers],
        "plugins": [_component_name(plugin) for plugin in plugins],
        "resource_context": _safe_resource_context(effective_resource_context),
        "resource_binding": case_resource_binding_audit(case),
    }


def format_case_check(case: Any, *, resource_context: Any = None) -> str:
    payload = build_case_check_payload(case, resource_context=resource_context)
    return "[check] " + json.dumps(payload, ensure_ascii=False, sort_keys=True)


def print_case_check(case: Any, *, resource_context: Any = None) -> dict[str, Any]:
    payload = build_case_check_payload(case, resource_context=resource_context)
    print("[check] " + json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def format_resource_context_summary(resource_context: Any) -> str:
    """Format the effective Project grant for a normal Case run."""

    return "[resource-context] " + json.dumps(
        _safe_resource_context(resource_context),
        ensure_ascii=False,
        sort_keys=True,
    )


def print_resource_context_summary(resource_context: Any) -> dict[str, Any]:
    payload = _safe_resource_context(resource_context)
    print("[resource-context] " + json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def _component_name(component: Any) -> str:
    if component is None:
        return "None"
    if isinstance(component, str):
        return component
    name = getattr(component, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return type(component).__name__


def _attached_providers(case: Any, mediator_providers: tuple[Any, ...]) -> tuple[Any, ...]:
    """Discover Providers from formal and compatibility attachment surfaces."""

    values: list[Any] = list(mediator_providers)
    direct = getattr(case, "evaluation_provider", None)
    if direct is not None:
        values.append(direct)
    collection = getattr(case, "evaluation_providers", None)
    if isinstance(collection, Mapping):
        values.extend(collection.values())
    elif isinstance(collection, (list, tuple, set, frozenset)):
        values.extend(collection)
    problem_provider = getattr(getattr(case, "problem", None), "provider", None)
    if problem_provider is not None:
        values.append(problem_provider)

    unique: list[Any] = []
    seen: set[int] = set()
    for provider in values:
        if provider is None or id(provider) in seen:
            continue
        seen.add(id(provider))
        unique.append(provider)
    return tuple(unique)


def _provider_name(provider: Any) -> str:
    spec = getattr(provider, "spec", None)
    provider_id = getattr(spec, "provider_id", None)
    if isinstance(provider_id, str) and provider_id.strip():
        return provider_id.strip()
    return _component_name(provider)


def _provider_details(provider: Any) -> dict[str, Any]:
    spec = getattr(provider, "spec", None)
    transition_ids = getattr(spec, "transition_method_ids", ()) if spec is not None else ()
    return {
        "name": _component_name(provider),
        "provider_id": _provider_name(provider),
        "compute_backend": str(getattr(spec, "compute_backend", "") or ""),
        "problem_ids": [str(item) for item in (getattr(spec, "problem_ids", ()) or ())],
        "transition_methods": [str(item) for item in (transition_ids or ())],
        "materialization_targets": [
            str(item)
            for item in (getattr(spec, "materialization_targets", ()) or ())
        ],
    }


def _pipeline_component_name(
    pipeline: Any,
    description: Mapping[str, Any],
    key: str,
) -> str:
    described = description.get(key)
    if described is not None:
        return str(described)
    component = getattr(pipeline, key, None)
    if callable(component):
        component = None
    if component is None:
        component = getattr(pipeline, f"_{key}", None)
    return _component_name(component)


def _pipeline_variant(description: Mapping[str, Any]) -> str:
    for key in ("route", "variant", "key", "family"):
        value = description.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for container_key in ("codec", "config", "graph_spec"):
        nested = description.get(container_key)
        if not isinstance(nested, Mapping):
            continue
        for key in ("route", "variant", "key", "family"):
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        metadata = nested.get("metadata")
        if isinstance(metadata, Mapping):
            for key in ("route", "variant", "family"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return "None"


def _safe_resource_context(resource_context: Any) -> dict[str, Any]:
    if resource_context is None:
        return {}
    payload: Any = resource_context
    if not isinstance(payload, Mapping):
        for method_name in ("as_dict", "to_dict"):
            method = getattr(payload, method_name, None)
            if callable(method):
                payload = method()
                break
    if not isinstance(payload, Mapping):
        return {"type": type(resource_context).__name__}
    safe = {key: payload[key] for key in _RESOURCE_KEYS if key in payload}
    grant = payload.get("grant")
    if isinstance(grant, Mapping):
        safe["grant"] = {key: grant[key] for key in _RESOURCE_KEYS if key in grant}
    return safe


__all__ = [
    "build_case_check_payload",
    "format_case_check",
    "format_resource_context_summary",
    "print_case_check",
    "print_resource_context_summary",
]
