"""Unified Project/Case/L0 runtime substrate.

This layer is intentionally semantic-neutral.  It knows about Projects, Cases,
case kinds, resource requests, leases, and ResourceContext injection.  It does
not know how an optimization solver searches or how an ML trainer fits.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from blackbase.resources import (
    InMemoryLeaseStore,
    ResourceAllocator,
    ResourceContext,
    ResourceLease,
    ResourceOffer,
    ResourcePolicy,
    ResourceRequest,
)


_CASE_LOCAL_MODULE_ROOTS = {
    "adapter",
    "adapters",
    "assembly",
    "bias",
    "capabilities",
    "case_scaffold",
    "catalog",
    "cli",
    "config",
    "evaluation",
    "inner_solver",
    "pipeline",
    "plugins",
    "problem",
    "refactor_data",
    "reporting",
    "runtime",
    "solver",
    "working_blacklist_optimizer",
    "working_integrated_optimizer",
    "working_nested_optimizer",
}
_SUPPORTED_CASE_KINDS = {"solver", "trainer"}


@dataclass(frozen=True)
class ProjectRuntimeConfig:
    """Project-level L0 configuration loaded from ``project_config.py``."""

    offer: ResourceOffer = field(default_factory=ResourceOffer)
    policy: ResourcePolicy = field(default_factory=ResourcePolicy)
    default_request: ResourceRequest = field(default_factory=ResourceRequest)
    compute_backend: str = "auto"
    execution_backend: str = "local"
    namespace: str = "project"


def load_project_runtime_config(module: Any) -> ProjectRuntimeConfig:
    """Load project-level L0 settings from a project config module."""

    l0 = dict(getattr(module, "L0", {}) or {})
    offer_payload = dict(l0.get("offer", getattr(module, "RESOURCE_OFFER", {}) or {}))
    policy_payload = dict(l0.get("policy", getattr(module, "RESOURCE_POLICY", {}) or {}))
    request_payload = dict(
        l0.get("default_request", getattr(module, "DEFAULT_RESOURCE_REQUEST", {}) or {})
    )
    return ProjectRuntimeConfig(
        offer=_coerce_offer(offer_payload),
        policy=ResourcePolicy.from_dict(policy_payload),
        default_request=ResourceRequest.from_dict(request_payload),
        compute_backend=str(l0.get("compute_backend", request_payload.get("compute_backend", "auto"))),
        execution_backend=str(
            l0.get("execution_backend", offer_payload.get("backend", offer_payload.get("resource_backend", "local")))
        ),
        namespace=str(l0.get("namespace", "project")),
    )


class ProjectL0Runtime:
    """Project-level allocator and ResourceContext injector."""

    def __init__(self, config: ProjectRuntimeConfig | None = None) -> None:
        self.config = config or ProjectRuntimeConfig()
        self.allocator = ResourceAllocator(
            offer=self.config.offer,
            policy=self.config.policy,
            lease_store=InMemoryLeaseStore(),
        )

    def acquire_case(
        self,
        case_name: str,
        *,
        request: ResourceRequest | Mapping[str, Any] | None = None,
        stage_name: str = "",
    ) -> ResourceLease:
        req = request if request is not None else self.config.default_request
        return self.allocator.acquire(
            req,
            owner_id=str(case_name),
            scope=str(stage_name or "project_case"),
        )

    def release(self, lease: ResourceLease | Mapping[str, Any] | str) -> None:
        self.allocator.release(lease)

    def resource_context(
        self,
        lease: ResourceLease,
        *,
        case_name: str,
        stage_name: str = "",
    ) -> ResourceContext:
        namespace = ".".join(
            part
            for part in (self.config.namespace, str(stage_name or ""), str(case_name))
            if part
        )
        return ResourceContext.from_mapping(
            lease.resource_context(
                compute_backend=self.config.compute_backend,
                execution_backend=self.config.execution_backend,
                namespace=namespace,
                metadata={
                    "substrate": "blackbase.Project/Case/Scaffold/L0",
                    "case_name": str(case_name),
                    "stage_name": str(stage_name or ""),
                },
            )
        )


def case_import_context(
    project_root: Path | str,
    case_name: str,
    *,
    extra_import_paths: Sequence[Path | str] = (),
) -> Iterator[Path]:
    return _case_import_context(project_root, case_name, extra_import_paths=extra_import_paths)


@contextmanager
def _case_import_context(
    project_root: Path | str,
    case_name: str,
    *,
    extra_import_paths: Sequence[Path | str] = (),
) -> Iterator[Path]:
    """Make one Project/Case scaffold importable with legacy-local isolation."""

    root = Path(project_root).resolve()
    case_root = root / "cases" / str(case_name)
    inserted = _prepend_sys_path((root, case_root, *tuple(Path(p) for p in extra_import_paths)))
    _purge_case_local_modules()
    try:
        yield case_root
    finally:
        for item in inserted:
            try:
                sys.path.remove(item)
            except ValueError:
                pass


class CaseBuilderProxy:
    """Lazy builder proxy that imports a case inside its Project context."""

    def __init__(
        self,
        project_root: Path | str,
        case_name: str,
        case_kind: str = "solver",
        *,
        extra_import_paths: Sequence[Path | str] = (),
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.case_name = str(case_name)
        self.case_kind = _normalize_case_kind(case_kind)
        self.extra_import_paths = tuple(Path(p).resolve() for p in extra_import_paths)

    def _resolve_builder(self):
        module_name, func_name = _builder_target(self.case_kind)
        import_path = f"cases.{self.case_name}.{module_name}"
        try:
            module = importlib.import_module(import_path)
        except Exception as exc:
            raise AttributeError(
                f"Case '{self.case_name}' (kind={self.case_kind}) must expose {module_name}.py:{func_name}()."
            ) from exc
        builder = getattr(module, func_name, None)
        if callable(builder):
            return builder
        raise AttributeError(
            f"Case '{self.case_name}' (kind={self.case_kind}) missing callable {module_name}.py:{func_name}()."
        )

    def _load(self):
        with _case_import_context(
            self.project_root,
            self.case_name,
            extra_import_paths=self.extra_import_paths,
        ):
            return self._resolve_builder()

    def accepts_parameter(self, name: str) -> bool:
        try:
            sig = inspect.signature(self._load())
        except (TypeError, ValueError):
            return False
        return str(name) in sig.parameters

    def __call__(self, *args: Any, **kwargs: Any):
        with _case_import_context(
            self.project_root,
            self.case_name,
            extra_import_paths=self.extra_import_paths,
        ):
            builder = self._resolve_builder()
            return builder(*args, **kwargs)


def load_case_builder(
    project_root: Path | str,
    case_name: str,
    case_kind: str = "solver",
    *,
    extra_import_paths: Sequence[Path | str] = (),
) -> CaseBuilderProxy:
    """Load case builder by semantic kind."""

    return CaseBuilderProxy(
        project_root,
        case_name,
        case_kind=case_kind,
        extra_import_paths=extra_import_paths,
    )


def load_case_kind(
    project_root: Path | str,
    case_name: str,
    *,
    stage: Mapping[str, Any] | None = None,
    default: str = "solver",
) -> str:
    """Resolve case semantic kind from stage override or ``.case`` marker."""

    stage = dict(stage or {})
    case_kinds = dict(stage.get("case_kinds", stage.get("kinds", {})) or {})
    if case_name in case_kinds:
        return _normalize_case_kind(case_kinds.get(case_name))
    case_root = Path(project_root).resolve() / "cases" / str(case_name)
    marker_kind = _read_case_marker_kind(case_root / ".case")
    if marker_kind:
        return marker_kind
    return _normalize_case_kind(default)


def load_case_resource_request(
    case_name: str,
    *,
    project_root: Path | str | None = None,
    stage: Mapping[str, Any] | None = None,
    default: ResourceRequest | Mapping[str, Any] | None = None,
    extra_import_paths: Sequence[Path | str] = (),
) -> ResourceRequest:
    """Resolve a case requirement without letting the case allocate resources."""

    stage = dict(stage or {})
    stage_requests = dict(stage.get("resource_requests", stage.get("resources", {})) or {})
    payload = stage_requests.get(case_name)
    if payload is None:
        try:
            if project_root is None:
                cfg_module = importlib.import_module(f"cases.{case_name}.config")
            else:
                with _case_import_context(
                    project_root,
                    case_name,
                    extra_import_paths=extra_import_paths,
                ):
                    cfg_module = importlib.import_module(f"cases.{case_name}.config")
            get_case_config = getattr(cfg_module, "get_case_config", None)
            get_project_config = getattr(cfg_module, "get_project_config", None)
            case_config = get_case_config() if callable(get_case_config) else None
            if case_config is None and callable(get_project_config):
                case_config = get_project_config()
            payload = getattr(case_config, "resource_request", None)
        except Exception:
            payload = None
    if payload is None:
        return default if isinstance(default, ResourceRequest) else ResourceRequest.from_dict(default)
    if isinstance(payload, ResourceRequest):
        return payload
    return ResourceRequest.from_dict(payload)


def build_case(builder: Any, *, resource_context: Mapping[str, Any] | ResourceContext | None = None):
    """Call a case builder with ResourceContext when it accepts one."""

    payload = _as_dict(resource_context) if resource_context is not None else None
    accepts = getattr(builder, "accepts_parameter", None)
    if callable(accepts) and accepts("resource_context"):
        return builder(resource_context=payload)
    try:
        sig = inspect.signature(builder)
    except (TypeError, ValueError):
        return builder()
    if "resource_context" in sig.parameters:
        return builder(resource_context=payload)
    built = builder()
    setter = getattr(built, "set_resource_context", None)
    if callable(setter) and payload is not None:
        setter(payload)
    return built


def run_case(case_obj: Any):
    """Run a solver/trainer-like case using the shared execution surface."""

    for name in ("run", "fit", "step"):
        fn = getattr(case_obj, name, None)
        if callable(fn):
            return fn()
    return case_obj


def iter_group_stages(config_module: Any, group_name: str) -> Sequence[Mapping[str, Any]]:
    groups = getattr(config_module, "GROUPS", {}) or {}
    stages = getattr(config_module, "STAGES", []) or []
    stage_by_name = {str(stage.get("name", "")): stage for stage in stages}
    selected = groups.get(group_name, {})
    selected_names = selected.get("stages", []) if isinstance(selected, Mapping) else []
    if not selected_names:
        return tuple(stages)
    return tuple(stage_by_name[name] for name in selected_names if name in stage_by_name)


def _prepend_sys_path(paths: Sequence[Path]) -> list[str]:
    inserted: list[str] = []
    for path in reversed([Path(p).resolve() for p in paths]):
        text = str(path)
        if text in sys.path:
            continue
        sys.path.insert(0, text)
        inserted.append(text)
    return inserted


def _purge_case_local_modules() -> None:
    for name in list(sys.modules):
        root = str(name).split(".", 1)[0]
        if root in _CASE_LOCAL_MODULE_ROOTS:
            sys.modules.pop(name, None)


def _normalize_case_kind(kind: str | None) -> str:
    value = str(kind or "").strip().lower()
    return value if value in _SUPPORTED_CASE_KINDS else "solver"


def _builder_target(case_kind: str) -> tuple[str, str]:
    if _normalize_case_kind(case_kind) == "trainer":
        return ("build_trainer", "build_trainer")
    return ("build_solver", "build_solver")


def _read_case_marker_kind(marker_path: Path) -> str | None:
    if not marker_path.is_file():
        return None
    try:
        text = marker_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = marker_path.read_text(encoding="utf-8-sig", errors="replace")
    kind: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip().lower() != "kind":
            continue
        kind = value.strip().strip('"').strip("'")
        break
    return _normalize_case_kind(kind) if kind else None


def _coerce_offer(payload: Mapping[str, Any] | None) -> ResourceOffer:
    data = dict(payload or {})
    known = {
        "threads": int(data.get("threads", data.get("workers", 1)) or 1),
        "gpus": int(data.get("gpus", 0) or 0),
        "backend": str(data.get("backend", data.get("resource_backend", "local"))),
        "device_tokens": tuple(data.get("device_tokens", ()) or ()),
        "metadata": dict(data.get("metadata", {}) or {}),
    }
    for key, value in data.items():
        if key not in {"threads", "workers", "gpus", "backend", "resource_backend", "device_tokens", "metadata"}:
            known["metadata"][key] = value
    return ResourceOffer(**known)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return dict(value.as_dict())
    if isinstance(value, Mapping):
        return dict(value)
    return {}
