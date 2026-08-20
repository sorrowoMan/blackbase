"""Unified Project/Case/L0 runtime substrate.

This layer is intentionally semantic-neutral.  It knows about Projects, Cases,
case kinds, resource requests, leases, and ResourceContext injection.  It does
not know how an optimization solver searches or how an ML trainer fits.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4

from blackbase.resources import (
    ArtifactAuthority,
    CancellationRef,
    CancellationToken,
    InMemoryLeaseStore,
    RedisLeaseStore,
    ResourceAllocator,
    ResourceContext,
    ResourceLease,
    ResourceOffer,
    ResourcePolicy,
    ResourceRequest,
    RedisBudgetAuthority,
    SQLiteBudgetAuthority,
    SQLiteLeaseStore,
    TerminationPolicy,
)

from .case_binding import bind_case_resource_context


_CASE_LOCAL_MODULE_ROOTS = {
    "adapter",
    "adapters",
    "assembly",
    "bias",
    "capabilities",
    "case_scaffold",
    "cases",
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
_PROJECT_IMPORT_LOCK = threading.RLock()


def text_declares_check_argument(text: str) -> bool:
    """Return whether source registers a real ``--check`` CLI argument."""

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_add_argument = (
            isinstance(func, ast.Attribute) and func.attr == "add_argument"
        ) or (
            isinstance(func, ast.Name) and func.id == "add_argument"
        )
        if not is_add_argument:
            continue
        if any(isinstance(arg, ast.Constant) and arg.value == "--check" for arg in node.args):
            return True
    return False


def path_declares_check_argument(path: Path | str) -> bool:
    """Resolve a direct or formally delegated ``--check`` CLI contract."""

    run_entry = Path(path)
    try:
        source = run_entry.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return False
    if text_declares_check_argument(source):
        return True
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    shared_cli_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module not in {"blackbase.project", "blackbase.project.case_cli"}:
            continue
        for alias in node.names:
            if alias.name == "run_standard_case_cli":
                shared_cli_names.add(str(alias.asname or alias.name))
    if shared_cli_names and any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in shared_cli_names
        for node in ast.walk(tree)
    ):
        return True

    delegated_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module != "build_solver":
            continue
        for alias in node.names:
            delegated_names.add(str(alias.asname or alias.name))
    if not delegated_names:
        return False
    calls_delegate = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in delegated_names
        for node in ast.walk(tree)
    )
    if not calls_delegate:
        return False
    build_entry = run_entry.with_name("build_solver.py")
    try:
        build_source = build_entry.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return False
    return text_declares_check_argument(build_source)


def load_resource_context_from_env(
    framework: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load the Project-granted ResourceContext passed to a CLI-mode Case."""

    source = os.environ if environ is None else environ
    keys: list[str] = []
    framework_name = str(framework or "").strip().upper()
    if framework_name:
        keys.append(f"{framework_name}_RESOURCE_CONTEXT_JSON")
    keys.append("BLACKBASE_RESOURCE_CONTEXT_JSON")
    for key in keys:
        raw = str(source.get(key, "") or "").strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid JSON in {key}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"{key} must contain a JSON object")
        return dict(payload)
    return {}
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
    lease_backend: str = "memory"
    lease_path: str = ".blackbase/l0_leases.sqlite"
    lease_redis_url: str = ""
    lease_redis_url_env: str = "BLACKBASE_REDIS_URL"
    lease_ttl_seconds: float = 30.0
    lease_heartbeat_seconds: float = 10.0
    budgets: Mapping[str, int] = field(default_factory=dict)
    budget_scope: str = ""
    control_path: str = ".blackbase/l0_controls.sqlite"
    artifact_path: str = ".blackbase/artifacts"
    artifact_allow_unsafe_serializers: bool = False
    termination: TerminationPolicy = field(default_factory=TerminationPolicy)

    def __post_init__(self) -> None:
        backend = str(self.lease_backend or "memory").strip().lower()
        redis_url = str(self.lease_redis_url or "").strip()
        redis_url_env = str(self.lease_redis_url_env or "").strip()
        ttl = float(self.lease_ttl_seconds)
        heartbeat = float(self.lease_heartbeat_seconds)
        budgets = {
            str(name).strip(): int(limit)
            for name, limit in dict(self.budgets or {}).items()
            if str(name).strip()
        }
        termination = (
            self.termination
            if isinstance(self.termination, TerminationPolicy)
            else TerminationPolicy.from_dict(self.termination)
        )
        if backend not in {"memory", "sqlite", "redis"}:
            raise ValueError("L0.lease_backend must be 'memory', 'sqlite', or 'redis'")
        if ttl <= 0:
            raise ValueError("L0.lease_ttl_seconds must be positive")
        if heartbeat <= 0 or heartbeat >= ttl:
            raise ValueError(
                "L0.lease_heartbeat_seconds must be positive and smaller than lease_ttl_seconds"
            )
        if backend == "redis" and not redis_url and not redis_url_env:
            raise ValueError(
                "Redis L0 lease authority requires L0.lease_redis_url or "
                "L0.lease_redis_url_env"
            )
        if any(limit < 0 for limit in budgets.values()):
            raise ValueError("L0.budgets limits must be non-negative")
        if budgets and backend == "memory":
            raise ValueError(
                "L0.budgets requires lease_backend='sqlite' or 'redis' for shared authority"
            )
        object.__setattr__(self, "lease_backend", backend)
        object.__setattr__(self, "lease_path", str(self.lease_path or ".blackbase/l0_leases.sqlite"))
        object.__setattr__(self, "lease_redis_url", redis_url)
        object.__setattr__(self, "lease_redis_url_env", redis_url_env)
        object.__setattr__(self, "lease_ttl_seconds", ttl)
        object.__setattr__(self, "lease_heartbeat_seconds", heartbeat)
        object.__setattr__(self, "budgets", budgets)
        object.__setattr__(self, "budget_scope", str(self.budget_scope or "").strip())
        object.__setattr__(self, "control_path", str(self.control_path or ".blackbase/l0_controls.sqlite"))
        object.__setattr__(self, "artifact_path", str(self.artifact_path or ".blackbase/artifacts"))
        object.__setattr__(
            self,
            "artifact_allow_unsafe_serializers",
            bool(self.artifact_allow_unsafe_serializers),
        )
        object.__setattr__(self, "termination", termination)


def load_project_runtime_config(module: Any) -> ProjectRuntimeConfig:
    """Load project-level L0 settings from a project config module."""

    l0 = dict(getattr(module, "L0", {}) or {})
    offer_payload = dict(l0.get("offer", getattr(module, "RESOURCE_OFFER", {}) or {}))
    policy_payload = dict(l0.get("policy", getattr(module, "RESOURCE_POLICY", {}) or {}))
    request_payload = dict(
        l0.get("default_request", getattr(module, "DEFAULT_RESOURCE_REQUEST", {}) or {})
    )
    artifact_payload = l0.get("artifacts", l0.get("artifact_store", {})) or {}
    if not isinstance(artifact_payload, Mapping):
        raise ValueError("L0.artifacts must be a mapping")
    termination_payload = l0.get("termination", {}) or {}
    if not isinstance(termination_payload, Mapping):
        raise ValueError("L0.termination must be a mapping")
    return ProjectRuntimeConfig(
        offer=_coerce_offer(offer_payload),
        policy=ResourcePolicy.from_dict(policy_payload),
        default_request=ResourceRequest.from_dict(request_payload),
        compute_backend=str(l0.get("compute_backend", request_payload.get("compute_backend", "auto"))),
        execution_backend=str(
            l0.get("execution_backend", offer_payload.get("backend", offer_payload.get("resource_backend", "local")))
        ),
        namespace=str(l0.get("namespace", "project")),
        lease_backend=str(l0.get("lease_backend", "memory")),
        lease_path=str(l0.get("lease_path", ".blackbase/l0_leases.sqlite")),
        lease_redis_url=str(l0.get("lease_redis_url", "")),
        lease_redis_url_env=str(l0.get("lease_redis_url_env", "BLACKBASE_REDIS_URL")),
        lease_ttl_seconds=float(l0.get("lease_ttl_seconds", 30.0) or 30.0),
        lease_heartbeat_seconds=float(l0.get("lease_heartbeat_seconds", 10.0) or 10.0),
        budgets=dict(l0.get("budgets", {}) or {}),
        budget_scope=str(l0.get("budget_scope", "") or ""),
        control_path=str(l0.get("control_path", ".blackbase/l0_controls.sqlite")),
        artifact_path=str(artifact_payload.get("path", ".blackbase/artifacts")),
        artifact_allow_unsafe_serializers=bool(
            artifact_payload.get("allow_unsafe_serializers", False)
        ),
        termination=TerminationPolicy.from_dict(termination_payload),
    )


class ResourceLeaseFenceError(RuntimeError):
    """Raised when a Case result no longer owns the current L0 fence."""


class ResourceLeaseGuard:
    """Renews one Project L0 lease and records loss of authority."""

    def __init__(self, runtime: "ProjectL0Runtime", lease: ResourceLease) -> None:
        self.runtime = runtime
        self.lease = lease
        self._stop = threading.Event()
        self._lost = threading.Event()
        renewed = self.runtime.allocator.renew(self.lease)
        if renewed is None:
            self._lost.set()
        self._thread = threading.Thread(
            target=self._run,
            name=f"blackbase-l0-heartbeat-{lease.lease_id}",
            daemon=True,
        )
        if not self._lost.is_set():
            self._thread.start()

    @property
    def lost(self) -> bool:
        return self._lost.is_set()

    def _run(self) -> None:
        interval = self.runtime.config.lease_heartbeat_seconds
        while not self._stop.wait(interval):
            if self.runtime.allocator.renew(self.lease) is None:
                self._lost.set()
                return

    def assert_current(self) -> None:
        if self.lost or not self.runtime.allocator.is_current(self.lease):
            raise ResourceLeaseFenceError(
                f"Project L0 lease fence is no longer current: "
                f"lease_id='{self.lease.lease_id}' token={self.lease.fencing_token}"
            )

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(
                timeout=max(0.1, self.runtime.config.lease_heartbeat_seconds * 2.0)
            )


class ProjectL0Runtime:
    """Project-level allocator and ResourceContext injector."""

    def __init__(
        self,
        config: ProjectRuntimeConfig | None = None,
        *,
        project_root: Path | str | None = None,
        durable: bool = True,
        lease_redis_client: Any = None,
    ) -> None:
        self.config = config or ProjectRuntimeConfig()
        self.project_root = (
            Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
        )
        artifact_root = Path(self.config.artifact_path)
        if not artifact_root.is_absolute():
            artifact_root = self.project_root / artifact_root
        artifact_root = artifact_root.resolve()
        try:
            artifact_root.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError(
                f"L0.artifacts.path must stay within the Project root: '{artifact_root}'"
            ) from exc
        self.artifact_authority = ArtifactAuthority(
            backend="filesystem",
            root=str(artifact_root),
            namespace=self.config.namespace,
            allow_unsafe_serializers=self.config.artifact_allow_unsafe_serializers,
        ).as_dict()
        configured_lease_backend = self.config.lease_backend
        lease_backend = configured_lease_backend if durable else "memory"
        self._control_backend = lease_backend
        budget_scope = self.config.budget_scope or f"run-{uuid4().hex}"
        self.budget_authority = None
        self._cancellation_redis_client = None
        self.budget_authority_metadata: dict[str, Any] = {}
        if lease_backend == "sqlite":
            lease_path = Path(self.config.lease_path)
            if not lease_path.is_absolute():
                lease_path = self.project_root / lease_path
            lease_store = SQLiteLeaseStore(
                lease_path,
                namespace=self.config.namespace,
            )
            self.lease_authority = {
                "backend": "sqlite",
                "path": str(lease_path.resolve()),
                "namespace": self.config.namespace,
                "ttl_seconds": self.config.lease_ttl_seconds,
                "heartbeat_seconds": self.config.lease_heartbeat_seconds,
            }
            if self.config.budgets:
                self.budget_authority = SQLiteBudgetAuthority(
                    lease_path,
                    namespace=self.config.namespace,
                    scope=budget_scope,
                )
                self.budget_authority_metadata = {
                    "backend": "sqlite",
                    "path": str(lease_path.resolve()),
                    "namespace": self.config.namespace,
                    "scope": budget_scope,
                    "budgets": dict(self.config.budgets),
                }
        elif lease_backend == "redis":
            redis_url = str(self.config.lease_redis_url or "").strip()
            if not redis_url:
                redis_url = str(
                    os.environ.get(self.config.lease_redis_url_env, "") or ""
                ).strip()
            if lease_redis_client is None and not redis_url:
                raise ValueError(
                    "Redis L0 lease authority requires L0.lease_redis_url or "
                    f"environment variable {self.config.lease_redis_url_env}"
                )
            lease_store = RedisLeaseStore(
                redis_url or "redis://localhost:6379/0",
                namespace=self.config.namespace,
                client=lease_redis_client,
            )
            self._cancellation_redis_client = lease_store.client
            self.lease_authority = {
                "backend": "redis",
                "namespace": self.config.namespace,
                "redis_url_env": self.config.lease_redis_url_env,
                "ttl_seconds": self.config.lease_ttl_seconds,
                "heartbeat_seconds": self.config.lease_heartbeat_seconds,
            }
            if self.config.budgets:
                self.budget_authority = RedisBudgetAuthority(
                    redis_url or "redis://localhost:6379/0",
                    namespace=self.config.namespace,
                    scope=budget_scope,
                    client=lease_store.client,
                )
                self.budget_authority_metadata = {
                    "backend": "redis",
                    "namespace": self.config.namespace,
                    "scope": budget_scope,
                    "redis_url_env": self.config.lease_redis_url_env,
                    "budgets": dict(self.config.budgets),
                }
        else:
            lease_store = InMemoryLeaseStore()
            self.lease_authority = {
                "backend": "memory",
                "namespace": self.config.namespace,
                "ttl_seconds": self.config.lease_ttl_seconds,
                "heartbeat_seconds": self.config.lease_heartbeat_seconds,
            }
        if not durable and configured_lease_backend == "sqlite":
            lease_path = Path(self.config.lease_path)
            if not lease_path.is_absolute():
                lease_path = self.project_root / lease_path
            self.lease_authority = {
                "backend": "sqlite",
                "path": str(lease_path.resolve()),
                "namespace": self.config.namespace,
                "ttl_seconds": self.config.lease_ttl_seconds,
                "heartbeat_seconds": self.config.lease_heartbeat_seconds,
                "check_only": True,
            }
        elif not durable and configured_lease_backend == "redis":
            self.lease_authority = {
                "backend": "redis",
                "namespace": self.config.namespace,
                "redis_url_env": self.config.lease_redis_url_env,
                "ttl_seconds": self.config.lease_ttl_seconds,
                "heartbeat_seconds": self.config.lease_heartbeat_seconds,
                "check_only": True,
            }
        if self.budget_authority is not None:
            for budget_name, limit in self.config.budgets.items():
                self.budget_authority.configure(budget_name, limit)
        self.allocator = ResourceAllocator(
            offer=self.config.offer,
            policy=self.config.policy,
            lease_store=lease_store,
            lease_ttl_seconds=self.config.lease_ttl_seconds,
        )

    def new_cancellation_ref(self, *, deadline_at: float = 0.0) -> CancellationRef:
        """Issue a transport-safe cancellation reference for one Case run."""

        if self._control_backend == "redis":
            ref = CancellationRef(
                backend="redis",
                namespace=self.config.namespace,
                redis_url_env=self.config.lease_redis_url_env,
                deadline_at=deadline_at,
            )
        else:
            control_path = Path(self.config.control_path)
            if not control_path.is_absolute():
                control_path = self.project_root / control_path
            ref = CancellationRef(
                backend="sqlite",
                namespace=self.config.namespace,
                path=str(control_path.resolve()),
                deadline_at=deadline_at,
            )
        CancellationToken(ref, redis_client=self._cancellation_redis_client)
        return ref

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

    def start_lease_guard(self, lease: ResourceLease) -> ResourceLeaseGuard:
        return ResourceLeaseGuard(self, lease)

    def assert_current(self, lease: ResourceLease | Mapping[str, Any]) -> None:
        item = lease if isinstance(lease, ResourceLease) else ResourceLease.from_dict(lease)
        if not self.allocator.is_current(item):
            raise ResourceLeaseFenceError(
                f"Project L0 lease fence is no longer current: "
                f"lease_id='{item.lease_id}' token={item.fencing_token}"
            )

    def assert_resource_context_current(self, resource_context: Mapping[str, Any]) -> None:
        lease = dict(resource_context.get("lease", {}) or {})
        if not lease:
            raise ResourceLeaseFenceError("Case result omitted its Project L0 lease")
        self.assert_current(lease)

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
                    "lease_authority": dict(self.lease_authority),
                    "artifact_authority": dict(self.artifact_authority),
                    **(
                        {"budget_authority": dict(self.budget_authority_metadata)}
                        if self.budget_authority_metadata
                        else {}
                    ),
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


def project_import_context(*import_paths: Path | str) -> Iterator[tuple[Path, ...]]:
    """Temporarily expose project-local paths without leaking short-name modules."""

    return _project_import_context(import_paths)


@contextmanager
def _project_import_context(import_paths: Sequence[Path | str]) -> Iterator[tuple[Path, ...]]:
    resolved = tuple(Path(path).resolve() for path in import_paths)
    # sys.path and sys.modules are process-global. Serializing this import scope
    # prevents two in-process external workers from importing different
    # projects through the same short ``cases.*`` module names.
    with _PROJECT_IMPORT_LOCK:
        inserted = _prepend_sys_path(resolved)
        previous_modules = _case_local_modules()
        _purge_case_local_modules()
        try:
            yield resolved
        finally:
            _purge_case_local_modules()
            sys.modules.update(previous_modules)
            for item in inserted:
                try:
                    sys.path.remove(item)
                except ValueError:
                    pass


@contextmanager
def _case_import_context(
    project_root: Path | str,
    case_name: str,
    *,
    extra_import_paths: Sequence[Path | str] = (),
) -> Iterator[Path]:
    """Make one Project/Case scaffold importable with project-local isolation."""

    root = Path(project_root).resolve()
    case_root = root / "cases" / str(case_name)
    with _project_import_context((root, case_root, *tuple(Path(p) for p in extra_import_paths))):
        yield case_root


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
    """Load the canonical Case builder; semantic kind does not change its path."""

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


def build_case(
    builder: Any,
    *,
    resource_context: Mapping[str, Any] | ResourceContext | None = None,
    component_overrides: Mapping[str, Any] | None = None,
):
    """Call a canonical Case builder under the required shared injection contract."""

    payload = _as_dict(resource_context) if resource_context is not None else None
    overrides = dict(component_overrides or {})
    accepts = getattr(builder, "accepts_parameter", None)
    if callable(accepts):
        missing = [
            name
            for name in ("resource_context", "component_overrides")
            if not accepts(name)
        ]
    else:
        try:
            parameters = inspect.signature(builder).parameters
        except (TypeError, ValueError) as exc:
            raise TypeError("canonical Case builder must expose an inspectable signature") from exc
        missing = [
            name
            for name in ("resource_context", "component_overrides")
            if name not in parameters
        ]
    if missing:
        raise TypeError(
            "canonical Case builder must accept keyword parameters "
            f"resource_context and component_overrides; missing={missing}"
        )
    case_obj = builder(
        resource_context=payload,
        component_overrides=overrides,
    )
    bind_case_resource_context(case_obj, payload)
    return case_obj


def run_case(case_obj: Any, *, case_kind: str = "solver"):
    """Run a solver/trainer-like case using the shared execution surface."""

    order = ("fit", "run", "step") if _normalize_case_kind(case_kind) == "trainer" else ("run", "fit", "step")
    for name in order:
        fn = getattr(case_obj, name, None)
        if callable(fn):
            raw_output = fn()
            exporter = getattr(case_obj, "export_case_result", None)
            if callable(exporter):
                return exporter(raw_output)
            return raw_output
    return case_obj


def close_case_after_build_check(case_obj: Any) -> dict[str, Any]:
    """Release resources acquired while validating a built Case.

    A build check does not enter the normal Solver/Trainer lifecycle, so its
    object cannot rely on ``run()``/``fit()`` to reach teardown.  The first
    inspectable zero-argument close hook is executed exactly once.
    """

    for name in ("close_after_build_check", "close", "teardown"):
        hook = getattr(case_obj, name, None)
        if not callable(hook):
            continue
        try:
            inspect.signature(hook).bind()
        except (TypeError, ValueError):
            continue
        hook()
        return {"status": "closed", "hook": name}
    return {"status": "unavailable", "hook": None}


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


def _case_local_modules() -> dict[str, Any]:
    return {
        name: module
        for name, module in sys.modules.items()
        if str(name).split(".", 1)[0] in _CASE_LOCAL_MODULE_ROOTS
    }


def _purge_case_local_modules() -> None:
    for name in list(_case_local_modules()):
        root = str(name).split(".", 1)[0]
        if root in _CASE_LOCAL_MODULE_ROOTS:
            sys.modules.pop(name, None)


def _normalize_case_kind(kind: str | None) -> str:
    value = str(kind or "").strip().lower()
    return value if value in _SUPPORTED_CASE_KINDS else "solver"


def _builder_target(case_kind: str) -> tuple[str, str]:
    _normalize_case_kind(case_kind)
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
