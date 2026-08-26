"""Versioned execution contracts for recursive Project/Case orchestration."""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from uuid import uuid4

from blackbase.resources import (
    ArtifactBinding,
    ArtifactPublicationReceipt,
    BudgetHandle,
    CancellationRef,
    DataRef,
    TerminationPolicy,
)
from blackbase.wire import freeze_wire_mapping, thaw_wire_mapping


CASE_RUN_SCHEMA_VERSION = 3
CASE_RUN_STATUSES = frozenset(
    {
        "pending",
        "running",
        "succeeded",
        "built",
        "checked",
        "resumed",
        "failed",
        "cancelled",
        "timed_out",
        "skipped",
    }
)
CASE_RUN_SUCCESS_STATUSES = frozenset(
    {"succeeded", "built", "checked", "resumed"}
)


@dataclass(frozen=True)
class CaseRunIdentity:
    """Stable identity and lineage for one Case invocation."""

    project_run_id: str = ""
    root_run_id: str = ""
    case_run_id: str = ""
    parent_case_run_id: str = ""
    invocation_id: str = ""
    attempt: int = 0
    depth: int = 0

    def __post_init__(self) -> None:
        project_run_id = str(self.project_run_id or f"project-run-{uuid4().hex}")
        root_run_id = str(self.root_run_id or project_run_id)
        case_run_id = str(self.case_run_id or f"case-run-{uuid4().hex}")
        invocation_id = str(self.invocation_id or case_run_id)
        attempt = max(0, int(self.attempt or 0))
        depth = max(0, int(self.depth or 0))
        parent = str(self.parent_case_run_id or "")
        if depth > 0 and not parent:
            raise ValueError("nested Case identity requires parent_case_run_id")
        object.__setattr__(self, "project_run_id", project_run_id)
        object.__setattr__(self, "root_run_id", root_run_id)
        object.__setattr__(self, "case_run_id", case_run_id)
        object.__setattr__(self, "parent_case_run_id", parent)
        object.__setattr__(self, "invocation_id", invocation_id)
        object.__setattr__(self, "attempt", attempt)
        object.__setattr__(self, "depth", depth)

    def child(self, *, invocation_id: str = "", case_run_id: str = "") -> "CaseRunIdentity":
        return CaseRunIdentity(
            project_run_id=self.project_run_id,
            root_run_id=self.root_run_id,
            case_run_id=str(case_run_id or f"case-run-{uuid4().hex}"),
            parent_case_run_id=self.case_run_id,
            invocation_id=str(invocation_id or f"invoke-{uuid4().hex}"),
            attempt=0,
            depth=self.depth + 1,
        )

    def for_attempt(self, attempt: int) -> "CaseRunIdentity":
        """Derive one concrete execution identity for a logical retry."""

        attempt_number = max(0, int(attempt))
        if attempt_number == self.attempt:
            return self
        return CaseRunIdentity(
            project_run_id=self.project_run_id,
            root_run_id=self.root_run_id,
            case_run_id=f"{self.invocation_id}:attempt:{attempt_number}",
            parent_case_run_id=self.parent_case_run_id,
            invocation_id=self.invocation_id,
            attempt=attempt_number,
            depth=self.depth,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "CaseRunIdentity":
        data = dict(payload or {})
        return cls(
            project_run_id=str(data.get("project_run_id", "")),
            root_run_id=str(data.get("root_run_id", "")),
            case_run_id=str(data.get("case_run_id", "")),
            parent_case_run_id=str(data.get("parent_case_run_id", "")),
            invocation_id=str(data.get("invocation_id", "")),
            attempt=int(data.get("attempt", 0) or 0),
            depth=int(data.get("depth", 0) or 0),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_run_id": self.project_run_id,
            "root_run_id": self.root_run_id,
            "case_run_id": self.case_run_id,
            "parent_case_run_id": self.parent_case_run_id,
            "invocation_id": self.invocation_id,
            "attempt": self.attempt,
            "depth": self.depth,
        }


@dataclass(frozen=True)
class ExecutionControl:
    """Fixed-size online cancellation control with durable lineage authority."""

    cancellation: CancellationRef = field(default_factory=CancellationRef)
    ancestor_cancellations: tuple[CancellationRef, ...] = ()
    termination: TerminationPolicy = field(default_factory=TerminationPolicy)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        current = (
            self.cancellation
            if isinstance(self.cancellation, CancellationRef)
            else CancellationRef.from_dict(self.cancellation)
        )
        ancestors = tuple(
            item if isinstance(item, CancellationRef) else CancellationRef.from_dict(item)
            for item in tuple(self.ancestor_cancellations or ())
        )
        if ancestors:
            raise ValueError(
                "ExecutionControl no longer transports ancestor references; "
                "use CancellationRef.parent_control_id and the durable authority"
            )
        termination = (
            self.termination
            if isinstance(self.termination, TerminationPolicy)
            else TerminationPolicy.from_dict(self.termination)
        )
        object.__setattr__(self, "cancellation", current)
        object.__setattr__(self, "ancestor_cancellations", ancestors)
        object.__setattr__(self, "termination", termination)
        object.__setattr__(
            self,
            "metadata",
            freeze_wire_mapping(
                _transport_safe(self.metadata, path="control.metadata"),
                path="control.metadata",
            ),
        )

    @property
    def deadline_at(self) -> float:
        return self.cancellation.deadline_at

    @classmethod
    def with_timeout(
        cls,
        timeout_seconds: float | None,
        *,
        backend: str = "memory",
        namespace: str = "blackbase",
        path: str = "",
        redis_url_env: str = "BLACKBASE_REDIS_URL",
        active_ttl_seconds: float = 0.0,
        heartbeat_seconds: float = 0.0,
        retention_seconds: float = 0.0,
        termination: TerminationPolicy | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ExecutionControl":
        timeout = 0.0 if timeout_seconds is None else float(timeout_seconds)
        if timeout < 0:
            raise ValueError("timeout_seconds must be non-negative")
        deadline = 0.0 if timeout == 0 else time.time() + timeout
        return cls(
            cancellation=CancellationRef(
                backend=backend,
                namespace=namespace,
                path=path,
                redis_url_env=redis_url_env,
                deadline_at=deadline,
                active_ttl_seconds=active_ttl_seconds,
                heartbeat_seconds=heartbeat_seconds,
                retention_seconds=retention_seconds,
            ),
            termination=(
                termination
                if isinstance(termination, TerminationPolicy)
                else TerminationPolicy.from_dict(termination)
            ),
            metadata=dict(metadata or {}),
        )

    def derive_child(
        self,
        child: "ExecutionControl | CancellationRef",
        *,
        intermediate_cancellations: Sequence[CancellationRef] = (),
    ) -> "ExecutionControl":
        if isinstance(child, ExecutionControl):
            child_cancellation = child.cancellation
            termination = child.termination
            child_metadata = dict(child.metadata)
        else:
            child_cancellation = child
            termination = self.termination
            child_metadata = {}
        intermediates = tuple(
            item
            if isinstance(item, CancellationRef)
            else CancellationRef.from_dict(item)
            for item in tuple(intermediate_cancellations or ())
        )
        expected_parent = self.cancellation
        for intermediate in intermediates:
            if (
                intermediate.backend != self.cancellation.backend
                or intermediate.namespace != self.cancellation.namespace
                or intermediate.path != self.cancellation.path
            ):
                raise ValueError(
                    "intermediate cancellation must use the parent authority"
                )
            expected_digest = hashlib.sha256(
                (
                    f"{expected_parent.lineage_digest}:"
                    f"{expected_parent.control_id}:{intermediate.control_id}"
                ).encode("utf-8")
            ).hexdigest()
            if (
                intermediate.parent_control_id != expected_parent.control_id
                or intermediate.root_control_id
                != (expected_parent.root_control_id or expected_parent.control_id)
                or intermediate.lineage_depth != expected_parent.lineage_depth + 1
                or intermediate.lineage_digest != expected_digest
            ):
                raise ValueError(
                    "intermediate cancellation does not extend the parent lineage"
                )
            expected_parent = intermediate
        parent_ref = expected_parent
        if child_cancellation.control_id in {
            self.cancellation.control_id,
            self.cancellation.root_control_id,
            *(item.control_id for item in intermediates),
        }:
            raise ValueError("child cancellation control_id would create a lineage cycle")
        child_deadlines = [
            value
            for value in (parent_ref.deadline_at, child_cancellation.deadline_at)
            if value > 0
        ]
        effective_deadline = min(child_deadlines) if child_deadlines else 0.0
        lineage_depth = int(parent_ref.lineage_depth) + 1
        lineage_digest = hashlib.sha256(
            (
                f"{parent_ref.lineage_digest}:"
                f"{parent_ref.control_id}:{child_cancellation.control_id}"
            ).encode("utf-8")
        ).hexdigest()
        cancellation = CancellationRef(
            control_id=child_cancellation.control_id,
            backend=parent_ref.backend,
            namespace=parent_ref.namespace,
            path=parent_ref.path,
            redis_url_env=parent_ref.redis_url_env,
            deadline_at=effective_deadline,
            created_at=child_cancellation.created_at,
            parent_control_id=parent_ref.control_id,
            root_control_id=(
                parent_ref.root_control_id or parent_ref.control_id
            ),
            lineage_depth=lineage_depth,
            lineage_digest=lineage_digest,
            active_ttl_seconds=parent_ref.active_ttl_seconds,
            heartbeat_seconds=parent_ref.heartbeat_seconds,
            retention_seconds=parent_ref.retention_seconds,
        )
        return ExecutionControl(
            cancellation=cancellation,
            ancestor_cancellations=(),
            termination=termination,
            metadata={
                **dict(self.metadata),
                **child_metadata,
                "parent_control_id": self.cancellation.control_id,
                "lineage_root_control_id": cancellation.root_control_id,
                "lineage_depth": cancellation.lineage_depth,
                "lineage_digest": cancellation.lineage_digest,
                "intermediate_control_ids": [
                    item.control_id for item in intermediates
                ],
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ExecutionControl":
        data = dict(payload or {})
        cancellation_payload = data.get("cancellation")
        if not isinstance(cancellation_payload, Mapping):
            cancellation_payload = {}
        return cls(
            cancellation=CancellationRef.from_dict(cancellation_payload),
            ancestor_cancellations=tuple(
                CancellationRef.from_dict(item)
                for item in data.get("ancestor_cancellations", ()) or ()
            ),
            termination=TerminationPolicy.from_dict(data.get("termination")),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "cancellation": self.cancellation.as_dict(),
            "ancestor_cancellations": [item.as_dict() for item in self.ancestor_cancellations],
            "deadline_at": self.deadline_at,
            "termination": self.termination.as_dict(),
            "metadata": thaw_wire_mapping(self.metadata),
        }


@dataclass(frozen=True)
class ChildResourceGrant:
    """Bounded subgrant issued under a parent Case lease."""

    grant_id: str
    parent_lease_id: str
    parent_case_run_id: str
    namespace: str
    resources: Mapping[str, Any] = field(default_factory=dict)
    fencing_token: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        grant_id = str(self.grant_id or "").strip()
        if not grant_id:
            raise ValueError("child grant_id must be non-empty")
        resources = dict(self.resources or {})
        if int(resources.get("threads", 1) or 1) < 1:
            raise ValueError("child grant threads must be positive")
        object.__setattr__(self, "grant_id", grant_id)
        object.__setattr__(self, "parent_lease_id", str(self.parent_lease_id or ""))
        object.__setattr__(self, "parent_case_run_id", str(self.parent_case_run_id or ""))
        object.__setattr__(self, "namespace", str(self.namespace or ""))
        object.__setattr__(
            self,
            "resources",
            freeze_wire_mapping(
                _transport_safe(resources, path="child_grant.resources"),
                path="child_grant.resources",
            ),
        )
        object.__setattr__(self, "fencing_token", max(0, int(self.fencing_token or 0)))
        object.__setattr__(
            self,
            "metadata",
            freeze_wire_mapping(
                _transport_safe(self.metadata, path="child_grant.metadata"),
                path="child_grant.metadata",
            ),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChildResourceGrant":
        return cls(
            grant_id=str(payload.get("grant_id", "")),
            parent_lease_id=str(payload.get("parent_lease_id", "")),
            parent_case_run_id=str(payload.get("parent_case_run_id", "")),
            namespace=str(payload.get("namespace", "")),
            resources=dict(payload.get("resources", {}) or {}),
            fencing_token=int(payload.get("fencing_token", 0) or 0),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "grant_id": self.grant_id,
            "parent_lease_id": self.parent_lease_id,
            "parent_case_run_id": self.parent_case_run_id,
            "namespace": self.namespace,
            "resources": thaw_wire_mapping(self.resources),
            "fencing_token": self.fencing_token,
            "metadata": thaw_wire_mapping(self.metadata),
        }


@dataclass(frozen=True)
class CaseFailure:
    """Structured, transport-safe failure retained by every execution backend."""

    kind: str
    message: str
    phase: str = "run"
    retryable: bool = False
    traceback_ref: DataRef | None = None
    cause: Mapping[str, Any] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.kind or "").strip():
            raise ValueError("failure kind must be non-empty")
        ref = self.traceback_ref
        if ref is not None and not isinstance(ref, DataRef):
            ref = DataRef.from_dict(ref)
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "message", str(self.message or ""))
        object.__setattr__(self, "phase", str(self.phase or "run"))
        object.__setattr__(self, "retryable", bool(self.retryable))
        object.__setattr__(self, "traceback_ref", ref)
        object.__setattr__(
            self,
            "cause",
            freeze_wire_mapping(
                _transport_safe(self.cause, path="failure.cause"),
                path="failure.cause",
            ),
        )
        object.__setattr__(
            self,
            "details",
            freeze_wire_mapping(
                _transport_safe(self.details, path="failure.details"),
                path="failure.details",
            ),
        )

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        phase: str,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> "CaseFailure":
        failure_details = collect_failure_evidence(exc)
        failure_details.update(dict(details or {}))
        if isinstance(exc, CaseInvocationError):
            result = exc.result
            child_failure = result.failure
            nested_cause = (
                child_failure.as_dict()
                if child_failure is not None
                else {
                    "kind": "CaseRunFailure",
                    "message": result.error or f"child Case status={result.status}",
                    "phase": "run",
                    "retryable": False,
                    "traceback_ref": None,
                    "cause": {},
                    "details": {},
                }
            )
            failure_details["child_case"] = {
                "identity": result.identity.as_dict(),
                "project_name": result.request.project_name,
                "stage_name": result.request.stage_name,
                "case_name": result.request.case_name,
                "case_kind": result.request.case_kind,
                "status": result.status,
                "exit_code": result.exit_code,
            }
            return cls(
                kind="CaseInvocationError",
                message=str(exc),
                phase=phase,
                retryable=(
                    bool(child_failure.retryable)
                    if child_failure is not None
                    else bool(retryable)
                ),
                cause=nested_cause,
                details=failure_details,
            )
        return cls(
            kind=type(exc).__name__,
            message=str(exc),
            phase=phase,
            retryable=retryable,
            details=failure_details,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CaseFailure":
        raw_ref = payload.get("traceback_ref")
        return cls(
            kind=str(payload.get("kind", "Failure")),
            message=str(payload.get("message", "")),
            phase=str(payload.get("phase", "run")),
            retryable=bool(payload.get("retryable", False)),
            traceback_ref=(
                DataRef.from_dict(raw_ref) if isinstance(raw_ref, Mapping) else None
            ),
            cause=dict(payload.get("cause", {}) or {}),
            details=dict(payload.get("details", {}) or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "phase": self.phase,
            "retryable": self.retryable,
            "traceback_ref": None if self.traceback_ref is None else self.traceback_ref.as_dict(),
            "cause": thaw_wire_mapping(self.cause),
            "details": thaw_wire_mapping(self.details),
        }


_FAILURE_EVIDENCE_ATTR = "_blackbase_failure_evidence"


def attach_failure_evidence(
    exc: BaseException,
    key: str,
    evidence: Any,
) -> BaseException:
    """Attach one formally serializable evidence field to an exception.

    ``CaseFailure.from_exception`` is the transport boundary and consumes this
    public carrier.  Callers should not rely on arbitrary exception attributes
    surviving a Case/process boundary.
    """

    name = str(key or "").strip()
    if not name:
        raise ValueError("failure evidence key must be non-empty")
    current = dict(getattr(exc, _FAILURE_EVIDENCE_ATTR, {}) or {})
    current[name] = evidence
    setattr(exc, _FAILURE_EVIDENCE_ATTR, current)
    return exc


def collect_failure_evidence(exc: BaseException) -> dict[str, Any]:
    """Return detached formal evidence carried by ``exc``."""

    raw = getattr(exc, _FAILURE_EVIDENCE_ATTR, {})
    return dict(raw) if isinstance(raw, Mapping) else {}


@dataclass(frozen=True)
class CaseRunRequest:
    """Complete transport-safe request for one standard Case invocation."""

    project_name: str
    stage_name: str
    case_name: str
    case_kind: str = "solver"
    mode: str = "build"
    identity: CaseRunIdentity = field(default_factory=CaseRunIdentity)
    control: ExecutionControl = field(default_factory=ExecutionControl)
    resource_request: Mapping[str, Any] = field(default_factory=dict)
    budget_request: Mapping[str, int] = field(default_factory=dict)
    resource_context: Mapping[str, Any] = field(default_factory=dict)
    child_grant: ChildResourceGrant | None = None
    budget_handles: Mapping[str, BudgetHandle] = field(default_factory=dict)
    component_overrides: Mapping[str, Any] = field(default_factory=dict)
    input_artifacts: Mapping[str, DataRef] = field(default_factory=dict)
    input_artifact_bindings: Mapping[str, ArtifactBinding] = field(default_factory=dict)
    inputs: Mapping[str, Any] = field(default_factory=dict)
    argv: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = CASE_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.schema_version) != CASE_RUN_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported CaseRunRequest schema_version={self.schema_version}"
            )
        identity = (
            self.identity
            if isinstance(self.identity, CaseRunIdentity)
            else CaseRunIdentity.from_dict(self.identity)
        )
        control = (
            self.control
            if isinstance(self.control, ExecutionControl)
            else ExecutionControl.from_dict(self.control)
        )
        child_grant = self.child_grant
        if child_grant is not None and not isinstance(child_grant, ChildResourceGrant):
            child_grant = ChildResourceGrant.from_dict(child_grant)
        handles = {
            str(key): value if isinstance(value, BudgetHandle) else BudgetHandle.from_dict(value)
            for key, value in dict(self.budget_handles or {}).items()
        }
        refs = {
            str(key): value if isinstance(value, DataRef) else DataRef.from_dict(value)
            for key, value in dict(self.input_artifacts or {}).items()
        }
        bindings = {
            str(key): (
                value
                if isinstance(value, ArtifactBinding)
                else ArtifactBinding.from_dict(value)
            )
            for key, value in dict(self.input_artifact_bindings or {}).items()
        }
        if refs and not bindings:
            raise ValueError(
                "formal input_artifacts require authority-issued ArtifactBinding values; "
                "put ordinary DataRef values in request.inputs instead"
            )
        bound_refs = {key: binding.ref for key, binding in bindings.items()}
        if refs and refs != bound_refs:
            raise ValueError("input_artifacts do not match input_artifact_bindings")
        refs = bound_refs
        object.__setattr__(self, "project_name", str(self.project_name))
        object.__setattr__(self, "stage_name", str(self.stage_name))
        object.__setattr__(self, "case_name", str(self.case_name))
        object.__setattr__(self, "case_kind", str(self.case_kind or "solver"))
        object.__setattr__(self, "mode", str(self.mode or "build"))
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "control", control)
        object.__setattr__(
            self,
            "resource_request",
            freeze_wire_mapping(
                _transport_safe(
                    self.resource_request,
                    path="request.resource_request",
                ),
                path="request.resource_request",
            ),
        )
        budgets = {
            str(key): int(value)
            for key, value in dict(self.budget_request or {}).items()
        }
        if any(value < 0 for value in budgets.values()):
            raise ValueError("Case budget_request values must be non-negative")
        object.__setattr__(self, "budget_request", MappingProxyType(budgets))
        object.__setattr__(
            self,
            "resource_context",
            freeze_wire_mapping(
                _transport_safe(
                    self.resource_context,
                    path="request.resource_context",
                ),
                path="request.resource_context",
            ),
        )
        object.__setattr__(self, "child_grant", child_grant)
        object.__setattr__(self, "budget_handles", MappingProxyType(handles))
        object.__setattr__(
            self,
            "component_overrides",
            freeze_wire_mapping(
                _transport_safe(
                    self.component_overrides,
                    path="request.component_overrides",
                ),
                path="request.component_overrides",
            ),
        )
        object.__setattr__(self, "input_artifacts", MappingProxyType(refs))
        object.__setattr__(
            self,
            "input_artifact_bindings",
            MappingProxyType(bindings),
        )
        object.__setattr__(
            self,
            "inputs",
            freeze_wire_mapping(
                _transport_safe(self.inputs, path="request.inputs"),
                path="request.inputs",
            ),
        )
        object.__setattr__(self, "argv", tuple(str(item) for item in self.argv))
        object.__setattr__(
            self,
            "metadata",
            freeze_wire_mapping(
                _transport_safe(self.metadata, path="request.metadata"),
                path="request.metadata",
            ),
        )
        object.__setattr__(self, "schema_version", CASE_RUN_SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CaseRunRequest":
        schema_version = int(payload.get("schema_version", 0) or 0)
        if schema_version != CASE_RUN_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported CaseRunRequest schema_version={schema_version}; "
                f"expected {CASE_RUN_SCHEMA_VERSION}. Live older requests cannot be "
                "upgraded because they do not carry the current cancellation and "
                "Artifact authority bindings; upgrade every worker before dispatch"
            )
        raw_grant = payload.get("child_grant")
        return cls(
            project_name=str(payload.get("project_name", "")),
            stage_name=str(payload.get("stage_name", "")),
            case_name=str(payload.get("case_name", "")),
            case_kind=str(payload.get("case_kind", "solver")),
            mode=str(payload.get("mode", "build")),
            identity=CaseRunIdentity.from_dict(payload.get("identity")),
            control=ExecutionControl.from_dict(payload.get("control")),
            resource_request=dict(payload.get("resource_request", {}) or {}),
            budget_request={
                str(key): int(value)
                for key, value in dict(payload.get("budget_request", {}) or {}).items()
            },
            resource_context=dict(payload.get("resource_context", {}) or {}),
            child_grant=(
                ChildResourceGrant.from_dict(raw_grant)
                if isinstance(raw_grant, Mapping)
                else None
            ),
            budget_handles={
                str(key): BudgetHandle.from_dict(value)
                for key, value in dict(payload.get("budget_handles", {}) or {}).items()
            },
            component_overrides=dict(payload.get("component_overrides", {}) or {}),
            input_artifacts={
                str(key): DataRef.from_dict(value)
                for key, value in dict(payload.get("input_artifacts", {}) or {}).items()
            },
            input_artifact_bindings={
                str(key): ArtifactBinding.from_dict(value)
                for key, value in dict(
                    payload.get("input_artifact_bindings", {}) or {}
                ).items()
            },
            inputs=dict(payload.get("inputs", {}) or {}),
            argv=tuple(payload.get("argv", ()) or ()),
            metadata=dict(payload.get("metadata", {}) or {}),
            schema_version=schema_version,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CASE_RUN_SCHEMA_VERSION,
            "project_name": self.project_name,
            "stage_name": self.stage_name,
            "case_name": self.case_name,
            "case_kind": self.case_kind,
            "mode": self.mode,
            "identity": self.identity.as_dict(),
            "control": self.control.as_dict(),
            "resource_request": _transport_safe(
                self.resource_request, path="request.resource_request"
            ),
            "budget_request": dict(self.budget_request),
            "resource_context": _transport_safe(
                self.resource_context, path="request.resource_context"
            ),
            "child_grant": None if self.child_grant is None else self.child_grant.as_dict(),
            "budget_handles": {
                key: handle.as_dict() for key, handle in self.budget_handles.items()
            },
            "component_overrides": _transport_safe(
                self.component_overrides, path="request.component_overrides"
            ),
            "input_artifacts": {
                key: ref.as_dict() for key, ref in self.input_artifacts.items()
            },
            "input_artifact_bindings": {
                key: binding.as_dict()
                for key, binding in self.input_artifact_bindings.items()
            },
            "inputs": _transport_safe(self.inputs, path="request.inputs"),
            "argv": list(self.argv),
            "metadata": _transport_safe(self.metadata, path="request.metadata"),
        }


@dataclass(frozen=True)
class CaseRunResult:
    """Versioned result envelope shared by every Case execution backend."""

    request: CaseRunRequest
    status: str
    output: Mapping[str, Any] = field(default_factory=dict)
    artifact_refs: Mapping[str, DataRef] = field(default_factory=dict)
    artifact_publications: Mapping[str, ArtifactPublicationReceipt] = field(
        default_factory=dict
    )
    diagnostic_artifact_refs: Mapping[str, DataRef] = field(default_factory=dict)
    diagnostic_artifact_publications: Mapping[
        str, ArtifactPublicationReceipt
    ] = field(default_factory=dict)
    resource_usage: Mapping[str, Any] = field(default_factory=dict)
    budget_usage: Mapping[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    finished_at: float = 0.0
    elapsed_seconds: float = 0.0
    exit_code: int = 0
    failure: CaseFailure | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = CASE_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.schema_version) != CASE_RUN_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported CaseRunResult schema_version={self.schema_version}"
            )
        request = (
            self.request
            if isinstance(self.request, CaseRunRequest)
            else CaseRunRequest.from_dict(self.request)
        )
        status = str(self.status or "pending").strip().lower()
        if status not in CASE_RUN_STATUSES:
            raise ValueError(f"unsupported CaseRunResult status '{status}'")
        supplied_refs = {
            str(key): value if isinstance(value, DataRef) else DataRef.from_dict(value)
            for key, value in dict(self.artifact_refs or {}).items()
        }
        publications = {
            str(key): (
                value
                if isinstance(value, ArtifactPublicationReceipt)
                else ArtifactPublicationReceipt.from_dict(value)
            )
            for key, value in dict(self.artifact_publications or {}).items()
        }
        diagnostic_refs = {
            str(key): value if isinstance(value, DataRef) else DataRef.from_dict(value)
            for key, value in dict(self.diagnostic_artifact_refs or {}).items()
        }
        diagnostic_publications = {
            str(key): (
                value
                if isinstance(value, ArtifactPublicationReceipt)
                else ArtifactPublicationReceipt.from_dict(value)
            )
            for key, value in dict(
                self.diagnostic_artifact_publications or {}
            ).items()
        }
        failure = self.failure
        if failure is not None and not isinstance(failure, CaseFailure):
            failure = CaseFailure.from_dict(failure)
        started = max(0.0, float(self.started_at or 0.0))
        finished = max(0.0, float(self.finished_at or 0.0))
        elapsed = max(0.0, float(self.elapsed_seconds or 0.0))
        if started > 0 and finished > 0:
            elapsed = max(0.0, finished - started)
        succeeded = (
            int(self.exit_code or 0) == 0
            and status in CASE_RUN_SUCCESS_STATUSES
            and failure is None
        )
        formal_refs: dict[str, DataRef] = {}
        formal_publications: dict[str, ArtifactPublicationReceipt] = {}
        for name, receipt in publications.items():
            if receipt.project_run_id != request.identity.project_run_id:
                raise ValueError(
                    f"Artifact publication '{name}' is outside the Case run lineage"
                )
            supplied = supplied_refs.get(name)
            if supplied is not None and supplied != receipt.ref:
                raise ValueError(
                    f"Artifact publication '{name}' disagrees with its DataRef"
                )
            if succeeded:
                formal_refs[name] = receipt.ref
                formal_publications[name] = receipt
            else:
                diagnostic_refs[name] = receipt.ref
                diagnostic_publications[name] = receipt
        for name, ref in supplied_refs.items():
            if name not in formal_refs:
                diagnostic_refs.setdefault(name, ref)
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "output",
            freeze_wire_mapping(
                _transport_safe(self.output, path="result.output"),
                path="result.output",
            ),
        )
        object.__setattr__(self, "artifact_refs", MappingProxyType(formal_refs))
        object.__setattr__(
            self,
            "artifact_publications",
            MappingProxyType(formal_publications),
        )
        object.__setattr__(
            self,
            "diagnostic_artifact_refs",
            MappingProxyType(diagnostic_refs),
        )
        object.__setattr__(
            self,
            "diagnostic_artifact_publications",
            MappingProxyType(diagnostic_publications),
        )
        object.__setattr__(
            self,
            "resource_usage",
            freeze_wire_mapping(
                _transport_safe(self.resource_usage, path="result.resource_usage"),
                path="result.resource_usage",
            ),
        )
        object.__setattr__(
            self,
            "budget_usage",
            freeze_wire_mapping(
                _transport_safe(self.budget_usage, path="result.budget_usage"),
                path="result.budget_usage",
            ),
        )
        object.__setattr__(self, "started_at", started)
        object.__setattr__(self, "finished_at", finished)
        object.__setattr__(self, "elapsed_seconds", elapsed)
        object.__setattr__(self, "exit_code", int(self.exit_code or 0))
        object.__setattr__(self, "failure", failure)
        object.__setattr__(
            self,
            "metadata",
            freeze_wire_mapping(
                _transport_safe(self.metadata, path="result.metadata"),
                path="result.metadata",
            ),
        )
        object.__setattr__(self, "schema_version", CASE_RUN_SCHEMA_VERSION)

    @property
    def identity(self) -> CaseRunIdentity:
        return self.request.identity

    @property
    def control(self) -> ExecutionControl:
        return self.request.control

    @property
    def error(self) -> str:
        if self.failure is None:
            return ""
        return f"{self.failure.kind}: {self.failure.message}"

    @property
    def ok(self) -> bool:
        return (
            self.exit_code == 0
            and self.status in CASE_RUN_SUCCESS_STATUSES
            and self.failure is None
        )

    def raise_for_failure(self, message: str = "") -> "CaseRunResult":
        """Raise a structured nested invocation error when this result failed."""

        if not self.ok:
            raise CaseInvocationError(self, message=message)
        return self

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CaseRunResult":
        schema_version = int(payload.get("schema_version", 0) or 0)
        if schema_version != CASE_RUN_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported CaseRunResult schema_version={schema_version}; "
                f"expected {CASE_RUN_SCHEMA_VERSION}. Historical v1 results are "
                "migrated only through ProjectRunManifest resume; live worker "
                "results require a stack-wide worker upgrade"
            )
        raw_failure = payload.get("failure")
        return cls(
            request=CaseRunRequest.from_dict(dict(payload.get("request", {}) or {})),
            status=str(payload.get("status", "pending")),
            output=dict(payload.get("output", {}) or {}),
            artifact_refs={
                str(key): DataRef.from_dict(value)
                for key, value in dict(payload.get("artifact_refs", {}) or {}).items()
            },
            artifact_publications={
                str(key): ArtifactPublicationReceipt.from_dict(value)
                for key, value in dict(
                    payload.get("artifact_publications", {}) or {}
                ).items()
            },
            diagnostic_artifact_refs={
                str(key): DataRef.from_dict(value)
                for key, value in dict(
                    payload.get("diagnostic_artifact_refs", {}) or {}
                ).items()
            },
            diagnostic_artifact_publications={
                str(key): ArtifactPublicationReceipt.from_dict(value)
                for key, value in dict(
                    payload.get("diagnostic_artifact_publications", {}) or {}
                ).items()
            },
            resource_usage=dict(payload.get("resource_usage", {}) or {}),
            budget_usage=dict(payload.get("budget_usage", {}) or {}),
            started_at=float(payload.get("started_at", 0.0) or 0.0),
            finished_at=float(payload.get("finished_at", 0.0) or 0.0),
            elapsed_seconds=float(payload.get("elapsed_seconds", 0.0) or 0.0),
            exit_code=int(payload.get("exit_code", 0) or 0),
            failure=(
                CaseFailure.from_dict(raw_failure)
                if isinstance(raw_failure, Mapping)
                else None
            ),
            metadata=dict(payload.get("metadata", {}) or {}),
            schema_version=schema_version,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CASE_RUN_SCHEMA_VERSION,
            "request": self.request.as_dict(),
            "identity": self.identity.as_dict(),
            "status": self.status,
            "ok": self.ok,
            "output": _transport_safe(self.output, path="result.output"),
            "artifact_refs": {
                key: ref.as_dict() for key, ref in self.artifact_refs.items()
            },
            "artifact_publications": {
                key: receipt.as_dict()
                for key, receipt in self.artifact_publications.items()
            },
            "diagnostic_artifact_refs": {
                key: ref.as_dict()
                for key, ref in self.diagnostic_artifact_refs.items()
            },
            "diagnostic_artifact_publications": {
                key: receipt.as_dict()
                for key, receipt in self.diagnostic_artifact_publications.items()
            },
            "resource_usage": _transport_safe(
                self.resource_usage, path="result.resource_usage"
            ),
            "budget_usage": _transport_safe(
                self.budget_usage, path="result.budget_usage"
            ),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": self.elapsed_seconds,
            "exit_code": self.exit_code,
            "failure": None if self.failure is None else self.failure.as_dict(),
            "error": self.error,
            "metadata": _transport_safe(self.metadata, path="result.metadata"),
        }


class CaseInvocationError(RuntimeError):
    """Failure raised by a parent while retaining the complete child envelope."""

    def __init__(self, result: CaseRunResult, *, message: str = "") -> None:
        if not isinstance(result, CaseRunResult):
            raise TypeError("CaseInvocationError.result must be a CaseRunResult")
        if result.ok:
            raise ValueError("cannot raise CaseInvocationError for a successful result")
        self.result = result
        label = "/".join(
            (
                result.request.project_name,
                result.request.stage_name,
                result.request.case_name,
            )
        )
        prefix = str(message or "child Case invocation failed").strip()
        detail = result.error or f"status={result.status}, exit_code={result.exit_code}"
        super().__init__(f"{prefix}: {label}: {detail}")


@dataclass(frozen=True)
class ProjectRunResult:
    """Structured Project result retained independently from the CLI exit code."""

    project_name: str
    group: str
    case_results: Sequence[CaseRunResult] = ()
    artifact_registry: Mapping[str, DataRef] = field(default_factory=dict)
    artifact_bindings: Mapping[str, ArtifactBinding] = field(default_factory=dict)
    status: str = "ok"
    exit_code: int = 0
    run_id: str = ""
    manifest_path: str = ""
    resumed_from: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_name", str(self.project_name))
        object.__setattr__(self, "group", str(self.group))
        object.__setattr__(self, "case_results", tuple(self.case_results or ()))
        object.__setattr__(
            self,
            "artifact_registry",
            MappingProxyType(dict(self.artifact_registry or {})),
        )
        bindings = {
            str(key): (
                value
                if isinstance(value, ArtifactBinding)
                else ArtifactBinding.from_dict(value)
            )
            for key, value in dict(self.artifact_bindings or {}).items()
        }
        if dict(self.artifact_registry or {}) != {
            key: binding.ref for key, binding in bindings.items()
        }:
            raise ValueError(
                "ProjectRunResult artifact_registry must be the DataRef view of "
                "artifact_bindings"
            )
        object.__setattr__(
            self,
            "artifact_bindings",
            MappingProxyType(bindings),
        )
        object.__setattr__(self, "status", str(self.status or "unknown"))
        object.__setattr__(self, "exit_code", int(self.exit_code or 0))
        object.__setattr__(self, "run_id", str(self.run_id or ""))
        object.__setattr__(self, "manifest_path", str(self.manifest_path or ""))
        object.__setattr__(self, "resumed_from", str(self.resumed_from or ""))

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and all(item.ok for item in self.case_results)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CASE_RUN_SCHEMA_VERSION,
            "project_name": self.project_name,
            "group": self.group,
            "status": self.status,
            "exit_code": self.exit_code,
            "run_id": self.run_id,
            "manifest_path": self.manifest_path,
            "resumed_from": self.resumed_from,
            "cases": [item.as_dict() for item in self.case_results],
            "artifact_registry": {
                key: ref.as_dict() for key, ref in self.artifact_registry.items()
            },
            "artifact_bindings": {
                key: binding.as_dict()
                for key, binding in self.artifact_bindings.items()
            },
        }


class ProjectConfigurationError(ValueError):
    """Raised when a Project declares execution semantics the runtime cannot honor."""


def _transport_safe(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, DataRef):
        return value.as_dict()
    if isinstance(value, Mapping):
        return {
            str(key): _transport_safe(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_transport_safe(item, path=f"{path}[]") for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _transport_safe(tolist(), path=path)
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return _transport_safe(as_dict(), path=path)
    raise TypeError(
        f"Case protocol field '{path}' is not transport-safe: {type(value).__name__}"
    )


__all__ = [
    "CASE_RUN_SCHEMA_VERSION",
    "CASE_RUN_STATUSES",
    "CaseFailure",
    "CaseInvocationError",
    "CaseRunIdentity",
    "CaseRunRequest",
    "CaseRunResult",
    "ChildResourceGrant",
    "ExecutionControl",
    "ProjectConfigurationError",
    "ProjectRunResult",
]
