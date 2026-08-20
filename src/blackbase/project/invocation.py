"""Single recursive Case execution and invocation boundary."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Condition, RLock
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4

from blackbase.resources import (
    ArtifactPublisher,
    ArtifactSerializer,
    BudgetAccount,
    BudgetClaim,
    BudgetHandle,
    CancellationRef,
    CancellationRequested,
    CancellationToken,
    CaseDeadlineExceeded,
    DataRef,
    ResourceContext,
    ResourceRequest,
    build_budget_authority_from_resource_context,
)

from .case_execution import (
    case_runtime_state,
    collect_artifact_refs,
    inject_case_input_artifacts,
    make_transport_safe,
    normalize_case_output,
)
from .execution import (
    CaseFailure,
    CaseRunIdentity,
    CaseRunRequest,
    CaseRunResult,
    ChildResourceGrant,
    ExecutionControl,
    ProjectConfigurationError,
)
from .runtime import (
    build_case,
    case_import_context,
    close_case_after_build_check,
    load_case_builder,
    load_case_resource_request,
    run_case,
)


@dataclass
class CaseRuntimeContext:
    """Non-serializable runtime services injected into one built Case."""

    request: CaseRunRequest
    cancellation_tokens: tuple[CancellationToken, ...]
    invoker: "CaseInvoker"
    artifact_publisher: ArtifactPublisher | None = None
    _events: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _artifact_refs: dict[str, DataRef] = field(default_factory=dict, repr=False)

    @property
    def identity(self) -> CaseRunIdentity:
        return self.request.identity

    @property
    def control(self) -> ExecutionControl:
        return self.request.control

    @property
    def resource_context(self) -> ResourceContext:
        return ResourceContext.from_mapping(self.request.resource_context)

    def checkpoint(self) -> None:
        for token in self.cancellation_tokens:
            token.checkpoint()

    def cancel(self, reason: str = "cancelled by Case") -> bool:
        return self.cancellation_tokens[-1].cancel(reason)

    def invoke(self, request: CaseRunRequest) -> CaseRunResult:
        self.checkpoint()
        result = self.invoker.invoke(request)
        self.checkpoint()
        return result

    @property
    def artifact_refs(self) -> Mapping[str, DataRef]:
        return dict(self._artifact_refs)

    def register_artifact_serializer(
        self,
        serializer: ArtifactSerializer,
        *,
        replace: bool = False,
    ) -> None:
        if self.artifact_publisher is None:
            raise ProjectConfigurationError(
                "Case runtime has no Project artifact authority"
            )
        self.artifact_publisher.register_serializer(serializer, replace=replace)

    def publish_artifact(
        self,
        name: str,
        value: Any,
        *,
        serializer: str | ArtifactSerializer = "auto",
        kind: str = "artifact",
        media_type: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> DataRef:
        """Publish through the Project authority; never mint an unresolved ref."""

        if self.artifact_publisher is None:
            raise ProjectConfigurationError(
                "Case runtime has no Project artifact authority; run the Case through "
                "the standard Project substrate or attach a formal provider"
            )
        artifact_name = str(name or "").strip()
        if artifact_name in self._artifact_refs:
            raise ProjectConfigurationError(
                f"Case artifact '{artifact_name}' was already published"
            )
        self.checkpoint()
        ref = self.artifact_publisher.publish(
            artifact_name,
            value,
            serializer=serializer,
            kind=kind,
            media_type=media_type,
            metadata=metadata,
        )
        self.checkpoint()
        self._artifact_refs[artifact_name] = ref
        self.record_event(
            "artifact.published",
            {"name": artifact_name, "ref": ref.as_dict()},
        )
        return ref

    def record_event(self, topic: str, payload: Mapping[str, Any] | None = None) -> None:
        self._events.append(
            {
                "topic": str(topic),
                "payload": make_transport_safe(dict(payload or {}), path="runtime_event"),
                "recorded_at": time.time(),
            }
        )

    def audit(self) -> dict[str, Any]:
        return {
            "identity": self.identity.as_dict(),
            "control": self.control.as_dict(),
            "child_invocations": self.invoker.audit(),
            "artifact_refs": {
                name: ref.as_dict() for name, ref in self._artifact_refs.items()
            },
            "events": list(self._events),
        }


class _ChildGrantPool:
    """Atomically partitions one parent grant among concurrent child calls."""

    def __init__(self, parent_request: CaseRunRequest) -> None:
        self.parent_request = parent_request
        parent = ResourceContext.from_mapping(parent_request.resource_context)
        grant = dict(parent.grant or {})
        lease_resources = dict(dict(parent.lease or {}).get("resources", {}) or {})
        self._total_workers = max(
            1,
            int(grant.get("workers", lease_resources.get("workers", 1)) or 1),
        )
        self._total_threads = max(1, int(grant.get("threads", parent.threads) or parent.threads))
        self._total_gpus = max(0, int(grant.get("gpus", 0) or 0))
        self._total_memory_mb = max(
            0.0,
            float(grant.get("memory_mb", lease_resources.get("memory_mb", 0.0)) or 0.0),
        )
        self._total_gpu_memory_mb = max(
            0.0,
            float(
                grant.get(
                    "gpu_memory_mb",
                    lease_resources.get("gpu_memory_mb", 0.0),
                )
                or 0.0
            ),
        )
        self._device_tokens = tuple(str(item) for item in grant.get("device_tokens", ()) or ())
        self._parent_backend = str(
            grant.get("backend", lease_resources.get("backend", "local")) or "local"
        )
        self._parent_compute_backend = str(
            parent.compute_backend
            or grant.get("compute_backend", lease_resources.get("compute_backend", "auto"))
            or "auto"
        )
        self._parent_device = str(parent.device or grant.get("device", "cpu") or "cpu")
        self._parent_capabilities = tuple(
            str(item)
            for item in grant.get(
                "capabilities",
                lease_resources.get("capabilities", ()),
            )
            or ()
        )
        self._available_workers = self._total_workers
        self._available_threads = self._total_threads
        self._available_gpus = self._total_gpus
        self._available_memory_mb = self._total_memory_mb
        self._available_gpu_memory_mb = self._total_gpu_memory_mb
        self._available_tokens = list(self._device_tokens)
        self._condition = Condition(RLock())
        self._active: dict[str, ChildResourceGrant] = {}

    @contextmanager
    def acquire(
        self,
        request: ResourceRequest,
        *,
        identity: CaseRunIdentity,
        control: ExecutionControl,
        checkpoint: Any,
    ) -> Iterator[ChildResourceGrant]:
        workers = int(request.workers)
        threads = int(request.threads)
        requested_tokens = tuple(str(item) for item in request.device_tokens)
        if len(set(requested_tokens)) != len(requested_tokens):
            raise ProjectConfigurationError("child Case device_tokens must be unique")
        gpus = max(int(request.gpus), len(requested_tokens))
        memory_mb = max(0.0, float(request.memory_mb or 0.0))
        gpu_memory_mb = max(0.0, float(request.gpu_memory_mb or 0.0))
        backend, compute_backend, device = self._validate_qualitative_request(request)
        if workers > self._total_workers:
            raise ProjectConfigurationError(
                f"child Case requests {workers} workers but parent grant has "
                f"{self._total_workers}"
            )
        if threads > self._total_threads:
            raise ProjectConfigurationError(
                f"child Case requests {threads} threads but parent grant has {self._total_threads}"
            )
        if gpus > self._total_gpus:
            raise ProjectConfigurationError(
                f"child Case requests {gpus} GPUs but parent grant has {self._total_gpus}"
            )
        if memory_mb > self._total_memory_mb:
            raise ProjectConfigurationError(
                f"child Case requests {memory_mb:g} MB memory but parent grant has "
                f"{self._total_memory_mb:g} MB"
            )
        if gpu_memory_mb > self._total_gpu_memory_mb:
            raise ProjectConfigurationError(
                f"child Case requests {gpu_memory_mb:g} MB GPU memory but parent grant has "
                f"{self._total_gpu_memory_mb:g} MB"
            )
        if requested_tokens and not set(requested_tokens).issubset(self._device_tokens):
            raise ProjectConfigurationError(
                "child Case requests device tokens outside the parent grant"
            )
        allocated_tokens: tuple[str, ...] = ()
        with self._condition:
            while True:
                checkpoint()
                tokens_available = (
                    all(token in self._available_tokens for token in requested_tokens)
                    if requested_tokens
                    else (
                        len(self._available_tokens) >= gpus
                        if self._device_tokens
                        else True
                    )
                )
                if (
                    self._available_workers >= workers
                    and self._available_threads >= threads
                    and self._available_gpus >= gpus
                    and self._available_memory_mb >= memory_mb
                    and self._available_gpu_memory_mb >= gpu_memory_mb
                    and tokens_available
                ):
                    break
                remaining = control.deadline_at - time.time() if control.deadline_at > 0 else 0.05
                if control.deadline_at > 0 and remaining <= 0:
                    checkpoint()
                self._condition.wait(timeout=min(0.05, max(0.001, remaining)))
            self._available_workers -= workers
            self._available_threads -= threads
            self._available_gpus -= gpus
            self._available_memory_mb -= memory_mb
            self._available_gpu_memory_mb -= gpu_memory_mb
            if requested_tokens:
                allocated_tokens = requested_tokens
            elif gpus:
                allocated_tokens = tuple(self._available_tokens[:gpus])
            for token in allocated_tokens:
                self._available_tokens.remove(token)
            parent_context = ResourceContext.from_mapping(self.parent_request.resource_context)
            parent_lease = dict(parent_context.lease or {})
            namespace = ".".join(
                item
                for item in (
                    parent_context.namespace,
                    "child",
                    identity.case_run_id,
                )
                if item
            )
            resources = {
                "workers": workers,
                "threads": threads,
                "gpus": gpus,
                "memory_mb": memory_mb or None,
                "gpu_memory_mb": gpu_memory_mb or None,
                "device_tokens": list(allocated_tokens),
                "backend": backend,
                "compute_backend": compute_backend,
                "device": device,
                "capabilities": list(request.capabilities),
            }
            grant = ChildResourceGrant(
                grant_id=f"child-grant-{uuid4().hex}",
                parent_lease_id=str(parent_lease.get("lease_id", "")),
                parent_case_run_id=self.parent_request.identity.case_run_id,
                namespace=namespace,
                resources=resources,
                fencing_token=int(parent_lease.get("fencing_token", 0) or 0),
                metadata={"resource_request": request.as_dict()},
            )
            self._active[grant.grant_id] = grant
        try:
            yield grant
        finally:
            with self._condition:
                self._active.pop(grant.grant_id, None)
                self._available_workers += workers
                self._available_threads += threads
                self._available_gpus += gpus
                self._available_memory_mb += memory_mb
                self._available_gpu_memory_mb += gpu_memory_mb
                self._available_tokens.extend(allocated_tokens)
                ordered = {token: index for index, token in enumerate(self._device_tokens)}
                self._available_tokens.sort(key=lambda token: ordered.get(token, len(ordered)))
                self._condition.notify_all()

    def _validate_qualitative_request(
        self,
        request: ResourceRequest,
    ) -> tuple[str, str, str]:
        backend = self._resolve_bounded_value(
            request.backend,
            parent=self._parent_backend,
            field="backend",
        )
        compute_backend = self._resolve_bounded_value(
            request.compute_backend,
            parent=self._parent_compute_backend,
            field="compute_backend",
            parent_auto_is_open=True,
        )
        if (
            compute_backend.lower().startswith(("cuda", "gpu", "mps", "xpu", "tpu"))
            and self._total_gpus <= 0
            and not self._device_tokens
        ):
            raise ProjectConfigurationError(
                f"child Case compute_backend='{compute_backend}' requires an accelerator "
                "outside the parent grant"
            )
        requested_capabilities = set(str(item) for item in request.capabilities)
        if not requested_capabilities.issubset(self._parent_capabilities):
            missing = sorted(requested_capabilities.difference(self._parent_capabilities))
            raise ProjectConfigurationError(
                "child Case requests capabilities outside the parent grant: "
                f"{missing}"
            )

        requested_device = str(request.device or "auto").strip()
        if requested_device.lower() in {"", "auto", "any", "none"}:
            device = self._parent_device
        elif requested_device.lower() == "cpu":
            device = "cpu"
        else:
            parent_device = self._parent_device.lower()
            token_match = requested_device in self._device_tokens
            accelerator_parent = (
                self._total_gpus > 0
                or bool(self._device_tokens)
                or parent_device.startswith(("cuda", "gpu", "mps", "xpu", "tpu"))
            )
            if not accelerator_parent or (
                ":" in requested_device
                and not token_match
                and requested_device.lower() != parent_device
            ):
                raise ProjectConfigurationError(
                    f"child Case device '{requested_device}' is outside parent device grant "
                    f"'{self._parent_device}'"
                )
            device = requested_device
        return backend, compute_backend, device

    @staticmethod
    def _resolve_bounded_value(
        requested: str,
        *,
        parent: str,
        field: str,
        parent_auto_is_open: bool = False,
    ) -> str:
        child_value = str(requested or "auto").strip()
        parent_value = str(parent or "auto").strip()
        if child_value.lower() in {"", "auto", "any", "none"}:
            return parent_value
        if child_value.lower() == parent_value.lower():
            return child_value
        if parent_auto_is_open and parent_value.lower() in {"", "auto", "any", "none"}:
            return child_value
        raise ProjectConfigurationError(
            f"child Case {field}='{child_value}' is outside parent grant "
            f"'{parent_value}'"
        )

    def audit(self) -> dict[str, Any]:
        with self._condition:
            return {
                "total_workers": self._total_workers,
                "available_workers": self._available_workers,
                "total_threads": self._total_threads,
                "available_threads": self._available_threads,
                "total_gpus": self._total_gpus,
                "available_gpus": self._available_gpus,
                "total_memory_mb": self._total_memory_mb,
                "available_memory_mb": self._available_memory_mb,
                "total_gpu_memory_mb": self._total_gpu_memory_mb,
                "available_gpu_memory_mb": self._available_gpu_memory_mb,
                "parent_backend": self._parent_backend,
                "parent_compute_backend": self._parent_compute_backend,
                "parent_device": self._parent_device,
                "parent_capabilities": list(self._parent_capabilities),
                "active_grants": [item.as_dict() for item in self._active.values()],
            }


@dataclass
class _BudgetDelegation:
    handle: BudgetHandle
    parent_account: BudgetAccount
    parent_claim: BudgetClaim


class CaseInvoker:
    """Public parent-to-child Case invocation service."""

    def __init__(
        self,
        executor: "CaseExecutor",
        parent_request: CaseRunRequest,
        *,
        cancellation_tokens: Sequence[CancellationToken],
    ) -> None:
        self.executor = executor
        self.parent_request = parent_request
        self.cancellation_tokens = tuple(cancellation_tokens)
        self._grants = _ChildGrantPool(parent_request)
        self._records: list[dict[str, Any]] = []
        self._lock = RLock()

    def _checkpoint(self) -> None:
        for token in self.cancellation_tokens:
            token.checkpoint()

    def invoke(self, request: CaseRunRequest) -> CaseRunResult:
        started_at = time.time()
        # A request object may be submitted repeatedly.  Its default identity is
        # only a request-side placeholder; each actual child execution gets a
        # fresh invocation and Case-run namespace.
        fallback_identity = self.parent_request.identity.child()
        child_control = self.parent_request.control.derive_child(request.control)
        prepared = replace(request, identity=fallback_identity, control=child_control)
        try:
            if request.project_name != self.parent_request.project_name:
                raise ProjectConfigurationError(
                    "child Case project_name must match the invoking parent Project"
                )
            if request.control.ancestor_cancellations:
                raise ProjectConfigurationError(
                    "child Case cannot supply an ancestor cancellation chain; "
                    "the invoking parent owns control lineage"
                )
            if request.resource_context or request.child_grant is not None:
                raise ProjectConfigurationError(
                    "child Case cannot supply an effective resource grant; "
                    "the invoking parent owns resource authorization"
                )
            if request.budget_handles:
                raise ProjectConfigurationError(
                    "child Case cannot supply budget handles; "
                    "the invoking parent owns budget delegation"
                )
            child_identity = request.identity
            if not child_identity.parent_case_run_id:
                child_identity = fallback_identity
            elif child_identity.parent_case_run_id != self.parent_request.identity.case_run_id:
                raise ProjectConfigurationError(
                    "child Case identity does not reference the invoking parent Case"
                )
            elif (
                child_identity.project_run_id != self.parent_request.identity.project_run_id
                or child_identity.root_run_id != self.parent_request.identity.root_run_id
                or child_identity.depth != self.parent_request.identity.depth + 1
            ):
                raise ProjectConfigurationError(
                    "child Case identity is inconsistent with parent run lineage"
                )
            prepared = replace(prepared, identity=child_identity)
            self._checkpoint()
            resource_request = self._resource_request(prepared)
            child_tokens = self.executor.cancellation_tokens(child_control)
            with self._grants.acquire(
                resource_request,
                identity=child_identity,
                control=child_control,
                checkpoint=lambda: self.executor.checkpoint(child_tokens),
            ) as grant:
                delegations = self._delegate_budgets(prepared, child_identity)
                try:
                    budget_handles = {
                        **dict(prepared.budget_handles),
                        **{item.handle.budget: item.handle for item in delegations},
                    }
                    resource_context = self._child_resource_context(
                        grant,
                        identity=child_identity,
                        control=child_control,
                        budget_handles=budget_handles,
                    )
                    effective = replace(
                        prepared,
                        resource_request=resource_request.as_dict(),
                        resource_context=resource_context,
                        child_grant=grant,
                        budget_handles=budget_handles,
                    )
                    result = self.executor.execute(effective)
                    usage = self._finalize_budgets(delegations)
                    result = replace(
                        result,
                        budget_usage={**dict(result.budget_usage), **usage},
                    )
                except BaseException:
                    self._finalize_budgets(delegations)
                    raise
        except CaseDeadlineExceeded as exc:
            result = self.executor.failure_result(
                prepared,
                exc,
                phase="invoke",
                status="timed_out",
                started_at=started_at,
            )
        except CancellationRequested as exc:
            result = self.executor.failure_result(
                prepared,
                exc,
                phase="invoke",
                status="cancelled",
                started_at=started_at,
            )
        except BaseException as exc:
            result = self.executor.failure_result(
                prepared,
                exc,
                phase="invoke",
                status="failed",
                started_at=started_at,
            )
        with self._lock:
            self._records.append(result.as_dict())
        return result

    def _resource_request(self, request: CaseRunRequest) -> ResourceRequest:
        if request.resource_request:
            return ResourceRequest.from_dict(request.resource_request)
        return load_case_resource_request(
            request.case_name,
            project_root=self.executor.project_root,
            default=ResourceRequest(),
            extra_import_paths=self.executor.extra_python_paths,
        )

    def _delegate_budgets(
        self,
        request: CaseRunRequest,
        identity: CaseRunIdentity,
    ) -> tuple[_BudgetDelegation, ...]:
        if not request.budget_request:
            return ()
        parent_context = ResourceContext.from_mapping(self.parent_request.resource_context)
        metadata = dict(parent_context.metadata or {})
        parent_handles = dict(metadata.get("budget_handles", {}) or {})
        authority_descriptor = dict(metadata.get("budget_authority", {}) or {})
        declared = dict(authority_descriptor.get("budgets", {}) or {})
        lease = dict(parent_context.lease or {})
        output: list[_BudgetDelegation] = []
        pending: tuple[BudgetAccount, BudgetClaim] | None = None
        try:
            for semantic_name, requested in request.budget_request.items():
                amount = int(requested)
                if semantic_name not in parent_handles and semantic_name not in declared:
                    raise ProjectConfigurationError(
                        f"parent Case was not granted budget '{semantic_name}'"
                    )
                parent_account = BudgetAccount.from_resource_context(
                    semantic_name,
                    parent_context,
                )
                if parent_account.allowance(amount) < amount:
                    raise ProjectConfigurationError(
                        f"child Case budget request exceeds parent allowance: "
                        f"{semantic_name}={amount}"
                    )
                parent_claim = parent_account.reserve(amount)
                pending = (parent_account, parent_claim)
                parent_handle_payload = parent_handles.get(semantic_name)
                parent_handle = (
                    BudgetHandle.from_dict(parent_handle_payload)
                    if isinstance(parent_handle_payload, Mapping)
                    else None
                )
                descriptor = (
                    dict(parent_handle.authority)
                    if parent_handle is not None
                    else authority_descriptor
                )
                authority_budget = f"{semantic_name}::child::{identity.case_run_id}"
                authority = None
                if descriptor:
                    authority = build_budget_authority_from_resource_context(
                        {
                            "lease": lease,
                            "metadata": {"budget_authority": descriptor},
                        }
                    )
                    if authority is None:  # pragma: no cover - guarded by descriptor
                        raise RuntimeError("cannot reconstruct child budget authority")
                    authority.configure(authority_budget, amount)
                handle = BudgetHandle(
                    handle_id=f"budget-handle-{uuid4().hex}",
                    budget=semantic_name,
                    authority_budget=(
                        authority_budget if authority is not None else semantic_name
                    ),
                    limit=amount,
                    authority=descriptor,
                    lease_id=str(lease.get("lease_id", "")),
                    fencing_token=int(lease.get("fencing_token", 0) or 0),
                    parent_reservation_id=parent_claim.reservation_id,
                    metadata={
                        "parent_case_run_id": self.parent_request.identity.case_run_id,
                        "child_case_run_id": identity.case_run_id,
                    },
                )
                output.append(_BudgetDelegation(handle, parent_account, parent_claim))
                pending = None
        except BaseException:
            if pending is not None:
                pending[0].cancel(pending[1])
            for delegation in output:
                delegation.parent_account.cancel(delegation.parent_claim)
            raise
        return tuple(output)

    @staticmethod
    def _finalize_budgets(
        delegations: Sequence[_BudgetDelegation],
    ) -> dict[str, Any]:
        usage: dict[str, Any] = {}
        for delegation in delegations:
            handle = delegation.handle
            committed = 0
            reserved = 0
            if handle.authority:
                authority = build_budget_authority_from_resource_context(
                    {
                        "lease": {
                            "lease_id": handle.lease_id,
                            "fencing_token": handle.fencing_token,
                        },
                        "metadata": {"budget_authority": dict(handle.authority)},
                    }
                )
                if authority is not None:
                    snapshot = authority.status(handle.authority_budget)
                    committed = int(snapshot.committed)
                    reserved = int(snapshot.reserved)
            used = min(handle.limit, committed + reserved)
            claim = delegation.parent_claim
            if claim.active:
                additional = max(0, used - claim.consumed)
                if additional:
                    delegation.parent_account.consume(claim, additional)
                delegation.parent_account.complete(claim)
            usage[handle.budget] = {
                "handle_id": handle.handle_id,
                "limit": handle.limit,
                "committed": committed,
                "outstanding_reserved": reserved,
                "charged_to_parent": used,
                "returned_to_parent": max(0, handle.limit - used),
            }
        return usage

    def _child_resource_context(
        self,
        grant: ChildResourceGrant,
        *,
        identity: CaseRunIdentity,
        control: ExecutionControl,
        budget_handles: Mapping[str, BudgetHandle],
    ) -> dict[str, Any]:
        parent = ResourceContext.from_mapping(self.parent_request.resource_context)
        resources = dict(grant.resources)
        metadata = {
            **dict(parent.metadata),
            "parent_case_run_id": self.parent_request.identity.case_run_id,
            "run_identity": identity.as_dict(),
            "execution_control": control.as_dict(),
            "child_grant": grant.as_dict(),
            "budget_handles": {
                name: handle.as_dict() for name, handle in budget_handles.items()
            },
        }
        return ResourceContext(
            scope="nested_case",
            execution_backend=parent.execution_backend,
            compute_backend=str(resources.get("compute_backend", parent.compute_backend)),
            device=str(resources.get("device", parent.device)),
            threads=int(resources.get("threads", 1) or 1),
            nested=True,
            namespace=grant.namespace,
            grant=resources,
            lease=dict(parent.lease),
            metadata=metadata,
        ).as_dict()

    def audit(self) -> dict[str, Any]:
        with self._lock:
            return {
                "parent_case_run_id": self.parent_request.identity.case_run_id,
                "resource_pool": self._grants.audit(),
                "results": list(self._records),
            }


class CaseExecutor:
    """Backend-neutral executor returning one CaseRunResult envelope."""

    def __init__(
        self,
        project_root: Path | str,
        *,
        extra_python_paths: Sequence[Path | str] = (),
        redis_client: Any = None,
        supervision_enabled: bool = True,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.extra_python_paths = tuple(Path(path).resolve() for path in extra_python_paths)
        self.redis_client = redis_client
        self.supervision_enabled = bool(supervision_enabled)

    def cancellation_tokens(
        self,
        control: ExecutionControl,
    ) -> tuple[CancellationToken, ...]:
        return tuple(
            CancellationToken(ref, redis_client=self.redis_client)
            for ref in (*control.ancestor_cancellations, control.cancellation)
        )

    @staticmethod
    def checkpoint(tokens: Sequence[CancellationToken]) -> None:
        for token in tokens:
            token.checkpoint()

    def execute(self, request: CaseRunRequest) -> CaseRunResult:
        if self.supervision_enabled and request.control.termination.requires_isolation:
            from .supervision import execute_case_payload_supervised

            supervised_started_at = time.time()
            try:
                return CaseRunResult.from_dict(
                    execute_case_payload_supervised(
                        {
                            "project_root": str(self.project_root),
                            "request": request.as_dict(),
                            "extra_python_paths": [
                                str(path) for path in self.extra_python_paths
                            ],
                        },
                        redis_client=self.redis_client,
                    )
                )
            except BaseException as exc:
                return self.failure_result(
                    request,
                    exc,
                    phase="supervise",
                    status="failed",
                    started_at=supervised_started_at,
                )
        started_at = time.time()
        phase = "control"
        tokens: tuple[CancellationToken, ...] = ()
        runtime_context: CaseRuntimeContext | None = None
        try:
            tokens = self.cancellation_tokens(request.control)
            self.checkpoint(tokens)
            phase = "build"
            with case_import_context(
                self.project_root,
                request.case_name,
                extra_import_paths=self.extra_python_paths,
            ):
                builder = load_case_builder(
                    self.project_root,
                    request.case_name,
                    case_kind=request.case_kind,
                    extra_import_paths=self.extra_python_paths,
                )
                case_obj = build_case(
                    builder,
                    resource_context=request.resource_context,
                    component_overrides=request.component_overrides,
                )
            inject_case_input_artifacts(
                case_obj,
                request.input_artifacts,
                case_name=request.case_name,
            )
            self._inject_inputs(case_obj, request.inputs)
            invoker = CaseInvoker(
                self._child_executor(),
                request,
                cancellation_tokens=tokens,
            )
            artifact_publisher = ArtifactPublisher.from_resource_context(
                request.resource_context,
                project_run_id=request.identity.project_run_id,
                case_run_id=request.identity.case_run_id,
                redis_client=self.redis_client,
            )
            runtime_context = CaseRuntimeContext(
                request,
                tokens,
                invoker,
                artifact_publisher=artifact_publisher,
            )
            self._inject_runtime(case_obj, runtime_context)
            runtime_state = make_transport_safe(
                case_runtime_state(case_obj),
                path="runtime_state",
            )
            build_check_cleanup = {"status": "not_required", "hook": None}
            if bool(request.metadata.get("check_only", False)):
                phase = "cleanup"
                build_check_cleanup = close_case_after_build_check(case_obj)
                output = {}
                result_status = "built"
            else:
                phase = "run"
                runtime_context.checkpoint()
                raw_output = run_case(case_obj, case_kind=request.case_kind)
                runtime_context.checkpoint()
                phase = "serialize"
                output = normalize_case_output(raw_output)
                output = make_transport_safe(output, path="output")
                result_status = "succeeded"
            artifact_refs = dict(runtime_context.artifact_refs)
            for name, ref in collect_artifact_refs(output).items():
                prior = artifact_refs.get(name)
                if prior is not None and prior != ref:
                    raise ProjectConfigurationError(
                        f"Case output artifact '{name}' conflicts with the ref "
                        "published through case_runtime"
                    )
                artifact_refs[name] = ref
            finished_at = time.time()
            resource_usage = {
                "authorized_grant": (
                    request.child_grant.as_dict()
                    if request.child_grant is not None
                    else dict(request.resource_context.get("grant", {}) or {})
                ),
                "effective_context": dict(runtime_state.get("resource_context", {}) or {}),
                "binding": dict(runtime_state.get("resource_binding", {}) or {}),
            }
            return CaseRunResult(
                request=request,
                status=result_status,
                output=output,
                artifact_refs=artifact_refs,
                resource_usage=resource_usage,
                budget_usage=self._budget_usage(request),
                started_at=started_at,
                finished_at=finished_at,
                metadata={
                    "runtime_state": runtime_state,
                    "runtime_audit": runtime_context.audit(),
                    "build_check_cleanup": build_check_cleanup,
                },
            )
        except CaseDeadlineExceeded as exc:
            return self.failure_result(
                request,
                exc,
                phase=phase,
                status="timed_out",
                started_at=started_at,
                runtime_context=runtime_context,
            )

        except CancellationRequested as exc:
            return self.failure_result(
                request,
                exc,
                phase=phase,
                status="cancelled",
                started_at=started_at,
                runtime_context=runtime_context,
            )
        except BaseException as exc:
            return self.failure_result(
                request,
                exc,
                phase=phase,
                status="failed",
                started_at=started_at,
                runtime_context=runtime_context,
            )

    def _child_executor(self) -> "CaseExecutor":
        """Nested Cases get their own supervisor even inside an isolated parent."""

        return CaseExecutor(
            self.project_root,
            extra_python_paths=self.extra_python_paths,
            redis_client=self.redis_client,
            supervision_enabled=True,
        )

    def failure_result(
        self,
        request: CaseRunRequest,
        exc: BaseException,
        *,
        phase: str,
        status: str,
        started_at: float,
        runtime_context: CaseRuntimeContext | None = None,
    ) -> CaseRunResult:
        return CaseRunResult(
            request=request,
            status=status,
            started_at=started_at,
            finished_at=time.time(),
            exit_code=1,
            failure=CaseFailure.from_exception(exc, phase=phase),
            budget_usage=self._budget_usage(request),
            metadata=(
                {} if runtime_context is None else {"runtime_audit": runtime_context.audit()}
            ),
        )

    @staticmethod
    def _inject_runtime(case_obj: Any, runtime_context: CaseRuntimeContext) -> None:
        setter = getattr(case_obj, "set_case_runtime", None)
        if callable(setter):
            setter(runtime_context)
            return
        try:
            setattr(case_obj, "case_runtime", runtime_context)
        except (AttributeError, TypeError) as exc:
            raise ProjectConfigurationError(
                "built Case must allow case_runtime injection or implement "
                "set_case_runtime(runtime)"
            ) from exc

    @staticmethod
    def _inject_inputs(case_obj: Any, inputs: Mapping[str, Any]) -> None:
        if not inputs:
            return
        setter = getattr(case_obj, "set_case_inputs", None)
        if callable(setter):
            setter(dict(inputs))
            return
        try:
            setattr(case_obj, "case_inputs", dict(inputs))
        except (AttributeError, TypeError) as exc:
            raise ProjectConfigurationError(
                "Case declares inputs but does not support set_case_inputs(inputs)"
            ) from exc

    @staticmethod
    def _budget_usage(request: CaseRunRequest) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for semantic_name, handle in request.budget_handles.items():
            if not handle.authority:
                output[semantic_name] = {
                    "handle_id": handle.handle_id,
                    "limit": handle.limit,
                    "committed": None,
                    "remaining": None,
                }
                continue
            authority = build_budget_authority_from_resource_context(
                {
                    "lease": {
                        "lease_id": handle.lease_id,
                        "fencing_token": handle.fencing_token,
                    },
                    "metadata": {"budget_authority": dict(handle.authority)},
                }
            )
            if authority is None:  # pragma: no cover - guarded by descriptor
                continue
            snapshot = authority.status(handle.authority_budget)
            output[semantic_name] = {
                "handle_id": handle.handle_id,
                **snapshot.as_dict(),
            }
        return output


__all__ = ["CaseExecutor", "CaseInvoker", "CaseRuntimeContext"]
