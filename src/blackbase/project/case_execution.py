"""Shared isolated Case build/run boundary used by process and external workers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from blackbase.resources import DataRef

from .execution import CaseRunRequest, ProjectConfigurationError
from .case_binding import case_resource_binding_audit
from .check_output import build_case_check_payload


def execute_case_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one versioned Case request from a transport-safe payload."""

    project_root = Path(str(payload["project_root"])).resolve()
    extra_python_paths = tuple(Path(path) for path in payload.get("extra_python_paths", ()) or ())
    request_payload = payload.get("request")
    if not isinstance(request_payload, Mapping):
        raise ProjectConfigurationError("Case execution payload omitted versioned request")
    request = CaseRunRequest.from_dict(request_payload)
    from .invocation import CaseExecutor

    result = CaseExecutor(
        project_root,
        extra_python_paths=extra_python_paths,
        supervision_enabled=not bool(payload.get("_blackbase_isolated_worker", False)),
    ).execute(request)
    return result.as_dict()


def make_transport_safe(value: Any, *, path: str) -> Any:
    """Convert a Case result field into a JSON/process transport-safe value."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, DataRef):
        return value.as_dict()
    if isinstance(value, Mapping):
        return {
            str(key): make_transport_safe(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [make_transport_safe(item, path=f"{path}[]") for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return make_transport_safe(tolist(), path=path)
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return make_transport_safe(as_dict(), path=path)
    raise TypeError(
        f"Case result field '{path}' is not transport-safe: {type(value).__name__}. "
        "Return JSON-compatible data or a DataRef."
    )


def inject_case_input_artifacts(
    case_obj: Any,
    input_artifacts: Mapping[str, DataRef],
    *,
    case_name: str,
) -> None:
    if not input_artifacts:
        return
    setter = getattr(case_obj, "set_input_artifacts", None)
    if not callable(setter):
        raise ProjectConfigurationError(
            f"Case '{case_name}' declares input_artifacts but does not implement "
            "set_input_artifacts(refs)"
        )
    setter(dict(input_artifacts))


def normalize_case_output(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    as_dict = getattr(raw, "as_dict", None)
    if callable(as_dict):
        value = as_dict()
        return dict(value) if isinstance(value, Mapping) else {"raw": value}
    return {"raw": raw}


def collect_artifact_refs(output: Mapping[str, Any]) -> dict[str, DataRef]:
    refs: dict[str, DataRef] = {}
    for field_name in ("artifact_refs", "artifacts"):
        raw = output.get(field_name)
        if not isinstance(raw, Mapping):
            continue
        for artifact_name, value in raw.items():
            ref = _coerce_data_ref(value)
            if ref is not None:
                refs[str(artifact_name)] = ref
    return refs


def case_runtime_state(case_obj: Any) -> dict[str, Any]:
    check_payload = build_case_check_payload(case_obj)
    plugins = getattr(getattr(case_obj, "plugin_manager", None), "plugins", []) or []
    providers = getattr(getattr(case_obj, "evaluation_mediator", None), "list_providers", None)
    pipeline = (
        getattr(case_obj, "representation_pipeline", None)
        or getattr(case_obj, "pipeline", None)
        or getattr(case_obj, "representation", None)
    )
    return {
        "case_class": type(case_obj).__name__,
        "problem": type(getattr(case_obj, "problem", None)).__name__,
        "pipeline": type(pipeline).__name__,
        "pipeline_variant": str(check_payload.get("pipeline_variant", "None")),
        "initializer": str(check_payload.get("initializer", "None")),
        "mutator": str(check_payload.get("mutator", "None")),
        "repair": str(check_payload.get("repair", "None")),
        "adapter": type(getattr(case_obj, "adapter", None)).__name__,
        "providers": len(tuple(providers())) if callable(providers) else 0,
        "provider_names": list(check_payload.get("providers", ()) or ()),
        "provider_details": list(check_payload.get("provider_details", ()) or ()),
        "plugins": len(tuple(plugins)),
        "plugin_names": list(check_payload.get("plugins", ()) or ()),
        "resource_context": _as_dict(getattr(case_obj, "resource_context", None)),
        "resource_binding": case_resource_binding_audit(case_obj),
        "component_overrides": _as_dict(
            getattr(case_obj, "component_override_audit", None)
        ),
    }


def _coerce_data_ref(value: Any) -> DataRef | None:
    if isinstance(value, DataRef):
        return value
    if isinstance(value, (str, Path)):
        return DataRef(uri=str(value))
    if not isinstance(value, Mapping):
        describe = getattr(value, "as_dict", None)
        if callable(describe):
            value = describe()
    if not isinstance(value, Mapping):
        return None
    payload = dict(value)
    if "uri" not in payload and payload.get("path"):
        payload["uri"] = str(payload["path"])
    if not payload.get("uri"):
        return None
    return DataRef.from_dict(payload)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return dict(value.as_dict())
    if isinstance(value, Mapping):
        return dict(value)
    return {}


__all__ = [
    "case_runtime_state",
    "collect_artifact_refs",
    "execute_case_payload",
    "inject_case_input_artifacts",
    "make_transport_safe",
    "normalize_case_output",
]
