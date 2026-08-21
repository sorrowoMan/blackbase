"""Single recursive Case execution and invocation boundary."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import RLock
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
    ResourceGrantPool,
    ResourceContext,
    ResourceRequest,
    ResourceSubgrantError,
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

    def invoke(
        self,
        request: CaseRunRequest,
        *,
        intermediate_cancellations: Sequence[CancellationRef] = (),
    ) -> CaseRunResult:
        self.checkpoint()
        result = self.invoker.invoke(
            request,
            intermediate_cancellations=intermediate_cancellations,
        )
        self.checkpoint()
        return result

    def cancellation_token(self, ref: CancellationRef) -> CancellationToken:
        """Create a runtime token for a parent-authorized intermediate scope."""

        if not isinstance(ref, CancellationRef):
            raise TypeError("cancellation_token requires a CancellationRef")
        return CancellationToken(ref, redis_client=self.invoker.executor.redis_client)

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


class _CaseChildGrantAdapter:
    """Project-specific projection over the single shared resource ledger."""

    def __init__(self, parent_request: CaseRunRequest) -> None:
        self.parent_request = parent_request
        self._pool = ResourceGrantPool(parent_request.resource_context)

    @contextmanager
    def acquire(
        self,
        request: ResourceRequest,
        *,
        identity: CaseRunIdentity,
        control: ExecutionControl,
        checkpoint: Any,
    ) -> Iterator[ChildResourceGrant]:
        parent = ResourceContext.from_mapping(self.parent_request.resource_context)
        parent_lease = dict(parent.lease or {})
        try:
            with self._pool.acquire(
                request,
                scope="nested_case",
                namespace_suffix=f"child.{identity.case_run_id}",
                checkpoint=checkpoint,
                deadline_at=control.deadline_at,
                metadata={
                    "parent_case_run_id": self.parent_request.identity.case_run_id,
                    "run_identity": identity.as_dict(),
                    "execution_control": control.as_dict(),
                    "resource_request": request.as_dict(),
                },
            ) as subgrant:
                yield ChildResourceGrant(
                    grant_id=subgrant.grant_id,
                    parent_lease_id=str(parent_lease.get("lease_id", "")),
                    parent_case_run_id=self.parent_request.identity.case_run_id,
                    namespace=subgrant.resource_context.namespace,
                    resources=subgrant.resources,
                    fencing_token=int(parent_lease.get("fencing_token", 0) or 0),
                    metadata={
                        "resource_request": request.as_dict(),
                        "resource_subgrant": subgrant.as_dict(),
                    },
                )
        except ResourceSubgrantError as exc:
            raise ProjectConfigurationError(str(exc)) from exc

    def audit(self) -> dict[str, Any]:
        return self._pool.audit()


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
        self._grants = _CaseChildGrantAdapter(parent_request)
        self._records: list[dict[str, Any]] = []
        self._lock = RLock()

    def _checkpoint(self) -> None:
        for token in self.cancellation_tokens:
            token.checkpoint()

    def invoke(
        self,
        request: CaseRunRequest,
        *,
        intermediate_cancellations: Sequence[CancellationRef] = (),
    ) -> CaseRunResult:
        started_at = time.time()
        # A request object may be submitted repeatedly.  Its default identity is
        # only a request-side placeholder; each actual child execution gets a
        # fresh invocation and Case-run namespace.
        fallback_identity = self.parent_request.identity.child()
        child_control = self.parent_request.control.derive_child(
            request.control,
            intermediate_cancellations=intermediate_cancellations,
        )
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
