"""Single recursive Case execution and invocation boundary."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Iterator, Mapping, Sequence
from uuid import uuid4

from blackbase.resources import (
    ArtifactPublisher,
    ArtifactPublicationReceipt,
    FilesystemArtifactPublicationLedger,
    ArtifactSerializer,
    BudgetAccount,
    BudgetClaim,
    BudgetHandle,
    BudgetSettlementRecord,
    CancellationHeartbeat,
    CancellationRef,
    CancellationRequested,
    CancellationToken,
    CaseDeadlineExceeded,
    DataRef,
    ResourceGrantPool,
    ResourceContext,
    ResourceRequest,
    PoolScheduler,
    ResourceSubgrantError,
    SQLiteBudgetSettlementJournal,
    SharedBudgetFenceError,
    build_budget_authority_from_resource_context,
    minimal_budget_authority_ref,
)

from .case_execution import (
    case_runtime_state,
    collect_artifact_refs,
    inject_case_input_artifacts,
    make_transport_safe,
    normalize_case_output,
)
from .execution import (
    attach_failure_evidence,
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
class ArtifactPublicationTransaction:
    """Stage Artifact refs until one Case-local finalization commit."""

    runtime: "CaseRuntimeContext"
    label: str
    transaction_id: str = field(default_factory=lambda: uuid4().hex)
    _refs: dict[str, DataRef] = field(default_factory=dict, repr=False)
    _reserved_names: set[str] = field(default_factory=set, repr=False)
    _receipts: dict[str, ArtifactPublicationReceipt] = field(
        default_factory=dict,
        repr=False,
    )
    _active: bool = field(default=True, repr=False)
    _prepared: bool = field(default=False, repr=False)

    @property
    def active(self) -> bool:
        return bool(self._active)

    @property
    def artifact_refs(self) -> Mapping[str, DataRef]:
        return dict(self._refs)

    @property
    def publication_receipts(self) -> Mapping[str, ArtifactPublicationReceipt]:
        return dict(self._receipts)

    def publish(
        self,
        name: str,
        value: Any,
        *,
        serializer: str | ArtifactSerializer = "auto",
        kind: str = "artifact",
        media_type: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> DataRef:
        if not self._active:
            raise RuntimeError("artifact publication transaction is closed")
        if self._prepared:
            raise RuntimeError("artifact publication transaction is already prepared")
        runtime = self.runtime
        if runtime.artifact_publisher is None:
            raise ProjectConfigurationError(
                "Case runtime has no Project artifact authority"
            )
        artifact_name = str(name or "").strip()
        if not artifact_name:
            raise ValueError("artifact name must be non-empty")
        with runtime._artifact_lock:
            if (
                artifact_name in runtime._artifact_refs
                or artifact_name in runtime._artifact_publications
                or artifact_name in self._reserved_names
            ):
                raise ProjectConfigurationError(
                    f"Case artifact '{artifact_name}' was already published or staged"
                )
            runtime.artifact_publisher.reserve_publication(
                artifact_name,
                self.transaction_id,
            )
            self._reserved_names.add(artifact_name)
        try:
            runtime.checkpoint()
            ref = runtime.artifact_publisher.publish(
                artifact_name,
                value,
                serializer=serializer,
                kind=kind,
                media_type=media_type,
                metadata={
                    **dict(metadata or {}),
                    "publication_transaction_id": self.transaction_id,
                    "publication_transaction_label": self.label,
                },
            )
            runtime.artifact_publisher.record_provisional_publication(
                artifact_name,
                self.transaction_id,
                ref,
            )
        except BaseException:
            self.abort("publish_failed")
            raise
        with runtime._artifact_lock:
            self._refs[artifact_name] = ref
        try:
            runtime.checkpoint()
        except BaseException:
            self.abort("post_publish_control_failure")
            raise
        runtime.record_event(
            "artifact.staged",
            {
                "name": artifact_name,
                "transaction_id": self.transaction_id,
                "ref": ref.as_dict(),
            },
        )
        return ref

    def commit(self) -> Mapping[str, DataRef]:
        raise RuntimeError(
            "Artifact transactions can only be sealed by CaseRuntimeContext after "
            "result serialization and cleanup; call prepare() from semantic layers"
        )

    def _seal(self) -> Mapping[str, DataRef]:
        """Authority-only durable commit used by the Case finalization coordinator."""

        if not self._active:
            raise RuntimeError("artifact publication transaction is closed")
        runtime = self.runtime
        runtime.checkpoint()
        with runtime._artifact_lock:
            conflicts = sorted(set(runtime._artifact_refs).intersection(self._refs))
            if conflicts:
                raise ProjectConfigurationError(
                    f"Case artifacts were published concurrently: {conflicts!r}"
                )
            finalized = dict(self.prepare())
            receipts = runtime.artifact_publisher.commit_publications(
                self.transaction_id,
                finalized,
                metadata={
                    "transaction_label": self.label,
                    "case_finalization_sealed": True,
                    "case_run_schema_version": 3,
                },
            )
            runtime._artifact_refs.update(finalized)
            runtime._artifact_publications.update(receipts)
            self._refs = finalized
            self._receipts = receipts
            self._active = False
            runtime._artifact_transactions.pop(self.transaction_id, None)
        runtime.record_event(
            "artifact.transaction_committed",
            {
                "transaction_id": self.transaction_id,
                "label": self.label,
                "artifact_names": sorted(finalized),
                "publication_receipts": {
                    name: receipt.as_dict() for name, receipt in receipts.items()
                },
            },
        )
        return dict(finalized)

    def prepare(self) -> Mapping[str, DataRef]:
        """Freeze final DataRefs without publishing an authority receipt.

        Case semantic layers use this phase to construct and serialize their
        result.  Only the CaseExecutor may call ``commit`` after cleanup and
        control shutdown have succeeded.
        """

        if not self._active:
            raise RuntimeError("artifact publication transaction is closed")
        with self.runtime._artifact_lock:
            self._prepared = True
            return dict(self._refs)

    def abort(self, reason: str = "aborted") -> None:
        if not self._active:
            return
        runtime = self.runtime
        with runtime._artifact_lock:
            staged_names = sorted(self._refs)
            self._active = False
            runtime._artifact_transactions.pop(self.transaction_id, None)
        if runtime.artifact_publisher is not None:
            runtime.artifact_publisher.abort_publications(
                self.transaction_id,
                self._refs,
                reason=str(reason or "aborted"),
                artifact_names=tuple(self._reserved_names),
            )
        runtime.record_event(
            "artifact.transaction_aborted",
            {
                "transaction_id": self.transaction_id,
                "label": self.label,
                "reason": str(reason or "aborted"),
                "provisional_artifact_names": staged_names,
            },
        )


@dataclass
class CaseRuntimeContext:
    """Non-serializable runtime services injected into one built Case."""

    request: CaseRunRequest
    cancellation_tokens: tuple[CancellationToken, ...]
    invoker: "CaseInvoker"
    stage_scheduler: PoolScheduler
    artifact_publisher: ArtifactPublisher | None = None
    _events: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _artifact_refs: dict[str, DataRef] = field(default_factory=dict, repr=False)
    _artifact_publications: dict[str, ArtifactPublicationReceipt] = field(
        default_factory=dict,
        repr=False,
    )
    _inherited_artifact_publications: list[ArtifactPublicationReceipt] = field(
        default_factory=list,
        repr=False,
    )
    _artifact_transactions: dict[str, ArtifactPublicationTransaction] = field(
        default_factory=dict,
        repr=False,
    )
    _finalization_transaction: ArtifactPublicationTransaction | None = field(
        default=None,
        repr=False,
    )
    _finalization_observers: list[
        tuple[
            str,
            Callable[[Mapping[str, ArtifactPublicationReceipt]], None],
        ]
    ] = field(default_factory=list, repr=False)
    _finalization_observer_failures: list[dict[str, Any]] = field(
        default_factory=list,
        repr=False,
    )
    _finalization_observers_notified: bool = field(default=False, repr=False)
    _finalization_committed_refs: dict[str, DataRef] = field(
        default_factory=dict,
        repr=False,
    )
    _artifact_lock: RLock = field(default_factory=RLock, repr=False)

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
        if result.ok and result.artifact_publications:
            with self._artifact_lock:
                known = {
                    item.receipt_digest
                    for item in self._inherited_artifact_publications
                }
                for receipt in result.artifact_publications.values():
                    if receipt.receipt_digest not in known:
                        self._inherited_artifact_publications.append(receipt)
                        known.add(receipt.receipt_digest)
        self.checkpoint()
        return result

    def cancellation_token(self, ref: CancellationRef) -> CancellationToken:
        """Create a runtime token for a parent-authorized intermediate scope."""

        if not isinstance(ref, CancellationRef):
            raise TypeError("cancellation_token requires a CancellationRef")
        return CancellationToken(ref, redis_client=self.invoker.executor.redis_client)

    def stage_worker_capacity(self, requested_workers: int) -> int:
        """Clamp Case-local fanout to the Project-authorized parent grant."""

        context = self.resource_context
        grant = dict(context.grant or {})
        workers = max(1, int(grant.get("workers", context.threads) or 1))
        threads = max(1, int(grant.get("threads", context.threads) or 1))
        return max(1, min(int(requested_workers), workers, threads))

    def stage_executor(self, max_workers: int):
        """Borrow an executor view from the Case-scoped L0 scheduler."""

        return self.stage_scheduler.as_executor(
            self.stage_worker_capacity(max_workers)
        )

    @property
    def artifact_refs(self) -> Mapping[str, DataRef]:
        return dict(self._artifact_refs)

    @property
    def artifact_publications(self) -> Mapping[str, ArtifactPublicationReceipt]:
        return dict(self._artifact_publications)

    @property
    def finalization_artifact_refs(self) -> Mapping[str, DataRef]:
        with self._artifact_lock:
            transaction = self._finalization_transaction
            if transaction is not None and transaction.active:
                return dict(transaction.artifact_refs)
            return dict(self._finalization_committed_refs)

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
        """Stage one Artifact in the Case-wide finalization transaction.

        The returned ``DataRef`` is usable inside the Case, but no authority
        receipt exists until CaseExecutor has serialized the result and closed
        runtime resources successfully.
        """

        if self.artifact_publisher is None:
            raise ProjectConfigurationError(
                "Case runtime has no Project artifact authority; run the Case through "
                "the standard Project substrate or attach a formal provider"
            )
        artifact_name = str(name or "").strip()
        transaction = self.begin_finalization_transaction()
        try:
            ref = transaction.publish(
                artifact_name,
                value,
                serializer=serializer,
                kind=kind,
                media_type=media_type,
                metadata=metadata,
            )
        except BaseException:
            transaction.abort("artifact_stage_failed")
            raise
        return ref

    def begin_artifact_transaction(
        self,
        label: str = "case_finalization",
    ) -> ArtifactPublicationTransaction:
        """Create a Case-local transaction whose refs are invisible until commit."""

        transaction = ArtifactPublicationTransaction(
            runtime=self,
            label=str(label or "case_finalization"),
        )
        with self._artifact_lock:
            self._artifact_transactions[transaction.transaction_id] = transaction
        self.record_event(
            "artifact.transaction_started",
            {
                "transaction_id": transaction.transaction_id,
                "label": transaction.label,
            },
        )
        return transaction

    def begin_finalization_transaction(
        self,
        label: str = "case_finalization",
    ) -> ArtifactPublicationTransaction:
        """Return the one shared publication transaction for final evidence.

        Semantic layers and checkpoint plugins stage into this transaction;
        the Solver lifecycle commits it once after every prepare hook passes.
        """

        with self._artifact_lock:
            current = self._finalization_transaction
            if current is not None and current.active:
                return current
            if self._finalization_committed_refs:
                raise RuntimeError("Case finalization transaction is already committed")
        transaction = self.begin_artifact_transaction(
            label=str(label or "case_finalization")
        )
        with self._artifact_lock:
            current = self._finalization_transaction
            if current is not None and current.active:
                transaction.abort("superseded_finalization_transaction")
                return current
            self._finalization_transaction = transaction
        return transaction

    def commit_finalization_transaction(self) -> Mapping[str, DataRef]:
        with self._artifact_lock:
            transaction = self._finalization_transaction
            if transaction is None:
                return dict(self._finalization_committed_refs)
            if not transaction.active:
                return dict(self._finalization_committed_refs)
        committed = dict(transaction._seal())
        with self._artifact_lock:
            self._finalization_committed_refs = dict(committed)
        return committed

    def abort_finalization_transaction(self, reason: str = "aborted") -> None:
        with self._artifact_lock:
            transaction = self._finalization_transaction
        if transaction is not None and transaction.active:
            transaction.abort(reason)

    def register_finalization_observer(
        self,
        observer: Callable[[Mapping[str, ArtifactPublicationReceipt]], None],
        *,
        name: str = "",
    ) -> None:
        """Register a non-veto observer that runs only after authority seal.

        Semantic runtimes register their ``on_*_finalized`` notification here.
        The Case executor owns invocation: cleanup, output serialization and the
        Artifact ledger seal must all have succeeded before an observer runs.
        Observer failures are diagnostic and can never revoke an issued receipt.
        """

        if not callable(observer):
            raise TypeError("finalization observer must be callable")
        label = str(name or getattr(observer, "__qualname__", "observer")).strip()
        with self._artifact_lock:
            if self._finalization_observers_notified:
                raise RuntimeError("Case finalization observers were already notified")
            self._finalization_observers.append((label or "observer", observer))

    def notify_finalization_observers(self) -> tuple[dict[str, Any], ...]:
        """Notify every post-seal observer and retain bounded failure evidence."""

        with self._artifact_lock:
            if self._finalization_observers_notified:
                return tuple(dict(item) for item in self._finalization_observer_failures)
            self._finalization_observers_notified = True
            observers = tuple(self._finalization_observers)
            publications = dict(self._artifact_publications)
        failures: list[dict[str, Any]] = []
        for name, observer in observers:
            try:
                observer(publications)
            except BaseException as exc:
                evidence = {
                    "observer": name,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                failures.append(evidence)
                self.record_event("case.finalization_observer_failed", evidence)
        with self._artifact_lock:
            self._finalization_observer_failures = list(failures)
        return tuple(dict(item) for item in failures)

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
            "artifact_publications": {
                name: receipt.as_dict()
                for name, receipt in self._artifact_publications.items()
            },
            "inherited_artifact_publications": [
                receipt.as_dict()
                for receipt in self._inherited_artifact_publications
            ],
            "active_artifact_transactions": sorted(self._artifact_transactions),
            "finalization_transaction": (
                None
                if self._finalization_transaction is None
                else {
                    "transaction_id": self._finalization_transaction.transaction_id,
                    "active": self._finalization_transaction.active,
                    "artifact_names": sorted(
                        self._finalization_transaction.artifact_refs
                    ),
                }
            ),
            "finalization_observers": {
                "registered": len(self._finalization_observers),
                "notified": self._finalization_observers_notified,
                "failures": [
                    dict(item) for item in self._finalization_observer_failures
                ],
            },
            "events": list(self._events),
            "stage_scheduler": self.stage_scheduler.report(),
        }

    def close(self, *, preserve_finalization: bool = False) -> None:
        """Join Case-local stage work before the parent resource lease may end."""

        with self._artifact_lock:
            active_transactions = tuple(self._artifact_transactions.values())
        for transaction in active_transactions:
            if preserve_finalization and transaction is self._finalization_transaction:
                continue
            transaction.abort("case_runtime_closed")
        self.stage_scheduler.shutdown(wait=True)


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
    settlement_id: str


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
        self._budget_settlement_journal = SQLiteBudgetSettlementJournal(
            self.executor.project_root / ".blackbase" / "budget_settlements.sqlite",
            namespace=self.parent_request.project_name or "project",
        )
        self._budget_settlement_reconciliation: list[dict[str, Any]] = []

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
        self._reconcile_budget_settlements()
        owned_child_token: CancellationToken | None = None
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
            owned_child_token = child_tokens[-1]
            with self._grants.acquire(
                resource_request,
                identity=child_identity,
                control=child_control,
                checkpoint=lambda: self.executor.checkpoint(child_tokens),
            ) as grant:
                delegations = self._delegate_budgets(prepared, child_identity)
                settlement_attempted = False
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
                    settlement_attempted = True
                    usage = self._finalize_budgets(delegations)
                    result = replace(
                        result,
                        budget_usage={**dict(result.budget_usage), **usage},
                    )
                except BaseException as primary_error:
                    if not settlement_attempted:
                        try:
                            self._finalize_budgets(delegations)
                        except BaseException as settlement_error:
                            attach_failure_evidence(
                                primary_error,
                                "budget_settlement",
                                {
                                    "error_type": type(settlement_error).__name__,
                                    "message": str(settlement_error),
                                },
                            )
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
        if owned_child_token is not None:
            try:
                owned_child_token.retire()
            except BaseException as exc:
                cleanup_failure = CaseFailure.from_exception(
                    exc,
                    phase="control_cleanup",
                )
                if result.failure is None:
                    result = replace(
                        result,
                        status="failed",
                        exit_code=1,
                        failure=cleanup_failure,
                        finished_at=time.time(),
                    )
                else:
                    result = replace(
                        result,
                        metadata={
                            **dict(result.metadata),
                            "control_cleanup_failure": cleanup_failure.as_dict(),
                        },
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
        pending_settlement_id: str | None = None
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
                handle_id = f"budget-handle-{uuid4().hex}"
                parent_reservation_id = f"budget-parent-{uuid4().hex}"
                handle = BudgetHandle(
                    handle_id=handle_id,
                    budget=semantic_name,
                    authority_budget=(
                        authority_budget if authority is not None else semantic_name
                    ),
                    limit=amount,
                    authority=descriptor,
                    lease_id=str(lease.get("lease_id", "")),
                    fencing_token=int(lease.get("fencing_token", 0) or 0),
                    parent_reservation_id=parent_reservation_id,
                    metadata={
                        "parent_case_run_id": self.parent_request.identity.case_run_id,
                        "child_case_run_id": identity.case_run_id,
                    },
                )
                settlement_id = f"budget-settlement:{handle.handle_id}"
                self._budget_settlement_journal.prepare(
                    BudgetSettlementRecord(
                        settlement_id=settlement_id,
                        project_run_id=identity.project_run_id,
                        parent_case_run_id=self.parent_request.identity.case_run_id,
                        child_case_run_id=identity.case_run_id,
                        budget=semantic_name,
                        parent_reservation_id=parent_reservation_id,
                        parent_authority_ref=minimal_budget_authority_ref(
                            parent_context.as_dict()
                        ),
                        child_handle=handle.as_dict(),
                        requested_amount=amount,
                    )
                )
                pending_settlement_id = settlement_id
                parent_claim = parent_account.reserve(
                    amount,
                    reservation_id=parent_reservation_id,
                )
                pending = (parent_account, parent_claim)
                self._budget_settlement_journal.mark_reserved(settlement_id)
                output.append(
                    _BudgetDelegation(
                        handle,
                        parent_account,
                        parent_claim,
                        settlement_id,
                    )
                )
                pending = None
                pending_settlement_id = None
        except BaseException as exc:
            if pending is not None:
                pending[0].cancel(pending[1])
            if pending_settlement_id is not None:
                self._budget_settlement_journal.mark_reclaimed(
                    pending_settlement_id,
                    exc,
                )
            for delegation in output:
                delegation.parent_account.cancel(delegation.parent_claim)
                self._budget_settlement_journal.mark_reclaimed(
                    delegation.settlement_id,
                    exc,
                )
            raise
        return tuple(output)

    def _finalize_budgets(
        self,
        delegations: Sequence[_BudgetDelegation],
    ) -> dict[str, Any]:
        usage: dict[str, Any] = {}
        failures: list[BaseException] = []
        for delegation in delegations:
            handle = delegation.handle
            item_usage: dict[str, Any] = {}
            try:
                item_usage = self._budget_usage_for_handle(handle)
                self._budget_settlement_journal.mark_ready(
                    delegation.settlement_id,
                    item_usage,
                )
                used = int(item_usage["charged_to_parent"])
                claim = delegation.parent_claim
                if claim.active:
                    additional = max(0, used - claim.consumed)
                    if additional:
                        delegation.parent_account.consume(claim, additional)
                    delegation.parent_account.complete(claim)
                item_usage["settlement_status"] = "settled"
                self._budget_settlement_journal.mark_settled(
                    delegation.settlement_id,
                    item_usage,
                )
            except BaseException as exc:
                item_usage = {
                    **item_usage,
                    "handle_id": handle.handle_id,
                    "limit": handle.limit,
                    "settlement_status": "retry_required",
                    "settlement_error": {
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
                self._budget_settlement_journal.mark_retry(
                    delegation.settlement_id,
                    exc,
                    usage=item_usage,
                )
                failures.append(exc)
            usage[handle.budget] = item_usage
        if failures:
            raise RuntimeError(
                "one or more child budget settlements require retry: "
                + "; ".join(
                    f"{type(exc).__name__}: {exc}" for exc in failures
                )
            ) from failures[0]
        return usage

    @staticmethod
    def _budget_usage_for_handle(handle: BudgetHandle) -> dict[str, Any]:
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
        return {
            "handle_id": handle.handle_id,
            "limit": handle.limit,
            "committed": committed,
            "outstanding_reserved": reserved,
            "charged_to_parent": used,
            "returned_to_parent": max(0, handle.limit - used),
        }

    def _reconcile_budget_settlements(self) -> None:
        # Sweep the whole Project namespace, not only the current run.  Old
        # run-scoped claims are either idempotently completed or rejected by
        # their lease fence; neither may become invisible journal debt.
        pending = self._budget_settlement_journal.pending()
        for record in pending:
            if record.status in {"reserve_intent", "prepared"}:
                try:
                    authority = build_budget_authority_from_resource_context(
                        record.parent_authority_ref
                    )
                    if authority is None:
                        raise SharedBudgetFenceError(
                            "budget settlement has no durable parent authority"
                        )
                    reservation = authority.reservation(record.parent_reservation_id)
                    if record.status == "reserve_intent":
                        if reservation is not None and reservation.status == "active":
                            authority.cancel(reservation)
                        error = SharedBudgetFenceError(
                            "write-ahead budget intent did not reach child execution"
                        )
                        archived = self._budget_settlement_journal.mark_reclaimed(
                            record.settlement_id,
                            error,
                            usage=record.usage,
                        )
                        self._budget_settlement_reconciliation.append(
                            {
                                "settlement_id": record.settlement_id,
                                "status": "reclaimed",
                                "error_type": type(error).__name__,
                                "message": str(error),
                                "attempts": archived.attempts,
                            }
                        )
                        continue
                    if reservation is not None and reservation.status == "active":
                        # Another live invocation owns this intent.  Settling it here
                        # would close the parent's reservation before the child has
                        # reached a terminal execution boundary.
                        continue
                    error = SharedBudgetFenceError(
                        "prepared budget settlement lost its active parent reservation"
                    )
                    archived = self._budget_settlement_journal.mark_reclaimed(
                        record.settlement_id,
                        error,
                        usage=record.usage,
                    )
                    self._budget_settlement_reconciliation.append(
                        {
                            "settlement_id": record.settlement_id,
                            "status": "reclaimed",
                            "error_type": type(error).__name__,
                            "message": str(error),
                            "attempts": archived.attempts,
                        }
                    )
                except SharedBudgetFenceError as exc:
                    archived = self._budget_settlement_journal.mark_reclaimed(
                        record.settlement_id,
                        exc,
                        usage=record.usage,
                    )
                    self._budget_settlement_reconciliation.append(
                        {
                            "settlement_id": record.settlement_id,
                            "status": "reclaimed",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                            "attempts": archived.attempts,
                        }
                    )
                except BaseException as exc:
                    # Preserve the write-ahead state.  A transient authority
                    # outage must not turn an unstarted child into terminal debt.
                    self._budget_settlement_reconciliation.append(
                        {
                            "settlement_id": record.settlement_id,
                            "status": "retry_required",
                            "record_status": record.status,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                continue
            if record.status not in {"settlement_ready", "retry_required"}:
                continue
            try:
                handle = BudgetHandle.from_dict(record.child_handle)
                usage = self._budget_usage_for_handle(handle)
                authority = build_budget_authority_from_resource_context(
                    record.parent_authority_ref
                )
                if authority is not None:
                    authority.complete(
                        record.parent_reservation_id,
                        completed=int(usage["charged_to_parent"]),
                    )
                usage["settlement_status"] = "settled"
                self._budget_settlement_journal.mark_settled(
                    record.settlement_id,
                    usage,
                )
                self._budget_settlement_reconciliation.append(
                    {
                        "settlement_id": record.settlement_id,
                        "status": "settled",
                        "usage": usage,
                    }
                )
            except SharedBudgetFenceError as exc:
                archived = self._budget_settlement_journal.mark_reclaimed(
                    record.settlement_id,
                    exc,
                    usage=record.usage,
                )
                self._budget_settlement_reconciliation.append(
                    {
                        "settlement_id": record.settlement_id,
                        "status": "reclaimed",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "attempts": archived.attempts,
                    }
                )
            except BaseException as exc:
                self._budget_settlement_journal.mark_retry(
                    record.settlement_id,
                    exc,
                    usage=record.usage,
                )
                self._budget_settlement_reconciliation.append(
                    {
                        "settlement_id": record.settlement_id,
                        "status": "retry_required",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )

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
                "budget_settlement_reconciliation": list(
                    self._budget_settlement_reconciliation
                ),
                "pending_budget_settlements": [
                    item.as_dict()
                    for item in self._budget_settlement_journal.pending(
                        project_run_id=self.parent_request.identity.project_run_id
                    )
                ],
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
        return (
            CancellationToken(control.cancellation, redis_client=self.redis_client),
        )

    @staticmethod
    def checkpoint(tokens: Sequence[CancellationToken]) -> None:
        for token in tokens:
            token.checkpoint()

    def execute(self, request: CaseRunRequest) -> CaseRunResult:
        if self.supervision_enabled and request.control.termination.requires_isolation:
            if request.control.cancellation.process_local:
                return self.failure_result(
                    request,
                    ProjectConfigurationError(
                        "isolated Case execution requires SQLite or Redis "
                        "cancellation authority; memory is process-local"
                    ),
                    phase="supervise",
                    status="failed",
                    started_at=time.time(),
                )
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
        stage_scheduler: PoolScheduler | None = None
        control_heartbeat: CancellationHeartbeat | None = None
        pending_result: CaseRunResult
        try:
            tokens = self.cancellation_tokens(request.control)
            control_heartbeat = CancellationHeartbeat(tokens[-1])
            self.checkpoint(tokens)
            artifact_publisher = ArtifactPublisher.from_resource_context(
                request.resource_context,
                project_run_id=request.identity.project_run_id,
                case_run_id=request.identity.case_run_id,
                redis_client=self.redis_client,
            )
            if request.input_artifact_bindings:
                if artifact_publisher is None:
                    raise ProjectConfigurationError(
                        "authoritative input_artifacts require an Artifact authority "
                        "in the effective ResourceContext"
                    )
                for input_name, binding in request.input_artifact_bindings.items():
                    receipt = binding.publication
                    ledger = FilesystemArtifactPublicationLedger(
                        artifact_publisher.authority,
                        project_run_id=receipt.project_run_id,
                        case_run_id=receipt.case_run_id,
                    )
                    if not ledger.verify(receipt):
                        raise ProjectConfigurationError(
                            f"input ArtifactBinding '{input_name}' failed authority "
                            "receipt or physical-content verification"
                        )
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
            declared_check_inputs = dict(
                request.metadata.get("declared_input_artifacts", {}) or {}
            )
            if (
                bool(request.metadata.get("check_only", False))
                and declared_check_inputs
                and not callable(getattr(case_obj, "set_input_artifacts", None))
            ):
                raise ProjectConfigurationError(
                    f"Case '{request.case_name}' declares input_artifacts but does not "
                    "implement set_input_artifacts(refs)"
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
            parent_context = ResourceContext.from_mapping(request.resource_context)
            parent_grant = dict(parent_context.grant or {})
            stage_capacity = max(
                1,
                min(
                    int(parent_grant.get("workers", parent_context.threads) or 1),
                    int(parent_grant.get("threads", parent_context.threads) or 1),
                ),
            )
            stage_scheduler = PoolScheduler(total_threads=stage_capacity)
            runtime_context = CaseRuntimeContext(
                request,
                tokens,
                invoker,
                stage_scheduler,
                artifact_publisher=artifact_publisher,
            )
            runtime_context._inherited_artifact_publications.extend(
                binding.publication
                for binding in request.input_artifact_bindings.values()
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
            artifact_publications = dict(runtime_context.artifact_publications)
            for name, ref in collect_artifact_refs(output).items():
                prior = artifact_refs.get(name)
                if prior is not None and prior != ref:
                    raise ProjectConfigurationError(
                        f"Case output artifact '{name}' conflicts with the ref "
                        "published through case_runtime"
                    )
                artifact_refs[name] = ref
                if name not in artifact_publications:
                    inherited = [
                        receipt
                        for receipt in runtime_context._inherited_artifact_publications
                        if receipt.ref == ref
                    ]
                    if len(inherited) == 1:
                        artifact_publications[name] = inherited[0]
                    elif len(inherited) > 1:
                        raise ProjectConfigurationError(
                            f"Case output artifact '{name}' matches multiple inherited "
                            "publication receipts"
                        )
            resource_usage = {
                "authorized_grant": (
                    request.child_grant.as_dict()
                    if request.child_grant is not None
                    else dict(request.resource_context.get("grant", {}) or {})
                ),
                "effective_context": dict(runtime_state.get("resource_context", {}) or {}),
                "binding": dict(runtime_state.get("resource_binding", {}) or {}),
            }
            pending_result = CaseRunResult(
                request=request,
                status=result_status,
                output=output,
                artifact_refs=artifact_refs,
                artifact_publications=artifact_publications,
                resource_usage=resource_usage,
                budget_usage=self._budget_usage(request),
                started_at=started_at,
                finished_at=time.time(),
                metadata={
                    "runtime_state": runtime_state,
                    "build_check_cleanup": build_check_cleanup,
                },
            )
        except CaseDeadlineExceeded as exc:
            pending_result = self.failure_result(
                request,
                exc,
                phase=phase,
                status="timed_out",
                started_at=started_at,
                runtime_context=runtime_context,
            )

        except CancellationRequested as exc:
            pending_result = self.failure_result(
                request,
                exc,
                phase=phase,
                status="cancelled",
                started_at=started_at,
                runtime_context=runtime_context,
            )
        except BaseException as exc:
            pending_result = self.failure_result(
                request,
                exc,
                phase=phase,
                status="failed",
                started_at=started_at,
                runtime_context=runtime_context,
            )
        final_result = self._finalize_local_result(
            pending_result,
            runtime_context=runtime_context,
            stage_scheduler=stage_scheduler,
        )
        if not final_result.ok and runtime_context is not None:
            runtime_context.abort_finalization_transaction(
                "case_finalization_failed_before_seal"
            )
        if control_heartbeat is not None:
            heartbeat_failure: BaseException | None = None
            try:
                control_heartbeat.assert_current()
            except BaseException as exc:
                heartbeat_failure = exc
            try:
                control_heartbeat.close()
            except BaseException as exc:
                if heartbeat_failure is None:
                    heartbeat_failure = exc
                else:
                    attach_failure_evidence(
                        heartbeat_failure,
                        "heartbeat_close",
                        {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
            if heartbeat_failure is not None and final_result.failure is None:
                if runtime_context is not None:
                    runtime_context.abort_finalization_transaction(
                        "control_heartbeat_failure"
                    )
                failure_result = self.failure_result(
                    request,
                    heartbeat_failure,
                    phase="control_heartbeat",
                    status="failed",
                    started_at=started_at,
                    runtime_context=runtime_context,
                )
                return replace(
                    failure_result,
                    metadata=dict(final_result.metadata),
                )
            if heartbeat_failure is not None:
                metadata = dict(final_result.metadata)
                metadata["control_heartbeat_failure"] = CaseFailure.from_exception(
                    heartbeat_failure,
                    phase="control_heartbeat",
                ).as_dict()
                final_result = replace(final_result, metadata=metadata)
        if final_result.ok and runtime_context is not None:
            try:
                committed = dict(runtime_context.commit_finalization_transaction())
            except BaseException as exc:
                try:
                    runtime_context.abort_finalization_transaction(
                        f"finalization_seal_failed:{type(exc).__name__}"
                    )
                except BaseException as abort_exc:
                    attach_failure_evidence(
                        exc,
                        "finalization_abort",
                        {
                            "error_type": type(abort_exc).__name__,
                            "message": str(abort_exc),
                        },
                    )
                failed = self.failure_result(
                    request,
                    exc,
                    phase="finalization_seal",
                    status="failed",
                    started_at=started_at,
                    runtime_context=runtime_context,
                )
                return replace(failed, metadata=dict(final_result.metadata))
            publications = dict(runtime_context.artifact_publications)
            refs = dict(committed)
            for name, ref in collect_artifact_refs(final_result.output).items():
                refs.setdefault(name, ref)
                if name not in publications:
                    inherited = [
                        receipt
                        for receipt in runtime_context._inherited_artifact_publications
                        if receipt.ref == ref
                    ]
                    if len(inherited) == 1:
                        publications[name] = inherited[0]
            final_result = replace(
                final_result,
                artifact_refs=refs,
                artifact_publications=publications,
                diagnostic_artifact_refs={
                    name: ref
                    for name, ref in final_result.diagnostic_artifact_refs.items()
                    if name not in publications
                },
                diagnostic_artifact_publications={
                    name: receipt
                    for name, receipt in final_result.diagnostic_artifact_publications.items()
                    if name not in publications
                },
            )
            observer_failures = runtime_context.notify_finalization_observers()
            metadata = dict(final_result.metadata)
            try:
                metadata["runtime_audit"] = runtime_context.audit()
            except BaseException as audit_exc:
                prior_audit = metadata.get("runtime_audit", {})
                metadata["runtime_audit"] = (
                    dict(prior_audit) if isinstance(prior_audit, Mapping) else {}
                )
                metadata["post_seal_audit_failure"] = CaseFailure.from_exception(
                    audit_exc,
                    phase="post_seal_audit",
                ).as_dict()
            metadata["finalization_observers"] = {
                "status": "degraded" if observer_failures else "succeeded",
                "failure_count": len(observer_failures),
                "failures": [dict(item) for item in observer_failures],
            }
            final_result = replace(final_result, metadata=metadata)
        return final_result

    @staticmethod
    def _finalize_local_result(
        result: CaseRunResult,
        *,
        runtime_context: CaseRuntimeContext | None,
        stage_scheduler: PoolScheduler | None,
    ) -> CaseRunResult:
        """Close local runtime services before freezing lifecycle evidence."""

        cleanup_started_at = time.time()
        cleanup_error: BaseException | None = None
        try:
            if runtime_context is not None:
                runtime_context.close(preserve_finalization=result.ok)
            elif stage_scheduler is not None:
                stage_scheduler.shutdown(wait=True)
        except BaseException as exc:
            cleanup_error = exc
        scheduler_shutdown_finished_at = time.time()

        scheduler_report: Mapping[str, Any] = {}
        audit_error: BaseException | None = None
        if runtime_context is not None:
            try:
                runtime_audit = runtime_context.audit()
                scheduler_report = dict(runtime_audit.get("stage_scheduler", {}) or {})
            except BaseException as exc:
                audit_error = exc
                runtime_audit = {
                    "identity": runtime_context.identity.as_dict(),
                    "stage_scheduler": {},
                    "audit_error": CaseFailure.from_exception(
                        exc,
                        phase="cleanup_audit",
                    ).as_dict(),
                }
        elif stage_scheduler is not None:
            try:
                scheduler_report = stage_scheduler.report()
            except BaseException as exc:
                audit_error = exc
                scheduler_report = {}
            runtime_audit = {"stage_scheduler": dict(scheduler_report)}
        else:
            runtime_audit = {}

        cleanup_finished_at = time.time()

        cleanup_failure = (
            None
            if cleanup_error is None
            else CaseFailure.from_exception(cleanup_error, phase="cleanup")
        )
        audit_failure = (
            None
            if audit_error is None
            else CaseFailure.from_exception(audit_error, phase="cleanup_audit")
        )
        cleanup_evidence = {
            "status": (
                "failed"
                if cleanup_error is not None
                else (
                    "degraded"
                    if audit_error is not None
                    else (
                        "not_required"
                        if runtime_context is None and stage_scheduler is None
                        else "succeeded"
                    )
                )
            ),
            "started_at": cleanup_started_at,
            "scheduler_shutdown_finished_at": scheduler_shutdown_finished_at,
            "finished_at": cleanup_finished_at,
            "elapsed_seconds": max(0.0, cleanup_finished_at - cleanup_started_at),
            "scheduler": dict(scheduler_report),
            "failure": None if cleanup_failure is None else cleanup_failure.as_dict(),
            "audit_failure": None if audit_failure is None else audit_failure.as_dict(),
        }
        metadata = dict(result.metadata or {})
        metadata["runtime_audit"] = runtime_audit
        metadata["cleanup"] = cleanup_evidence

        if cleanup_failure is not None and result.failure is None:
            return replace(
                result,
                status="failed",
                exit_code=1,
                failure=cleanup_failure,
                finished_at=cleanup_finished_at,
                metadata=metadata,
            )
        return replace(
            result,
            finished_at=cleanup_finished_at,
            metadata=metadata,
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
            artifact_refs=(
                {}
                if runtime_context is None
                else {
                    **dict(runtime_context.artifact_refs),
                    **dict(runtime_context.finalization_artifact_refs),
                }
            ),
            artifact_publications=(
                {}
                if runtime_context is None
                else runtime_context.artifact_publications
            ),
            started_at=started_at,
            finished_at=time.time(),
            exit_code=1,
            failure=CaseFailure.from_exception(exc, phase=phase),
            budget_usage=self._budget_usage(request),
            # Runtime audit is frozen only after scheduler cleanup by
            # _finalize_local_result().  Keeping it out of this provisional
            # envelope also prevents an audit error from masking the primary
            # Case failure.
            metadata={},
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
