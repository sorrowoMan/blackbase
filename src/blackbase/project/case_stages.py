"""Composable child-Case stages over the canonical invocation protocol.

This module contains orchestration only.  It never constructs a Solver or a
Trainer and therefore remains independent of downstream semantic frameworks.
"""

from __future__ import annotations

import time
from concurrent.futures import CancelledError, Executor, FIRST_COMPLETED, Future, wait
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol, Sequence

from blackbase.resources import (
    ArtifactBinding,
    CancellationHeartbeat,
    CancellationRef,
    CancellationToken,
    DataRef,
)

from .execution import (
    attach_failure_evidence,
    CaseRunRequest,
    CaseRunResult,
    CaseFailure,
    ExecutionControl,
    ProjectConfigurationError,
)


class CaseInvocationRuntime(Protocol):
    """Minimum parent runtime surface required by :class:`CaseStageRunner`."""

    request: CaseRunRequest

    def checkpoint(self) -> None: ...

    def invoke(
        self,
        request: CaseRunRequest,
        *,
        intermediate_cancellations: Sequence[CancellationRef] = (),
    ) -> CaseRunResult: ...

    def cancellation_token(self, ref: CancellationRef) -> CancellationToken: ...

    def stage_worker_capacity(self, requested_workers: int) -> int: ...

    def stage_executor(self, max_workers: int) -> Executor: ...


@dataclass(frozen=True)
class ChildCaseCall:
    """One complete child Case invocation inside a parent Case stage."""

    name: str
    case_name: str
    case_kind: str = "solver"
    mode: str = "build"
    resource_request: Mapping[str, Any] = field(default_factory=dict)
    budget_request: Mapping[str, int] = field(default_factory=dict)
    component_overrides: Mapping[str, Any] = field(default_factory=dict)
    input_artifacts: Mapping[str, DataRef] = field(default_factory=dict)
    input_artifact_bindings: Mapping[str, ArtifactBinding] = field(default_factory=dict)
    artifact_bindings: Mapping[str, str] = field(default_factory=dict)
    inputs: Mapping[str, Any] = field(default_factory=dict)
    argv: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        case_name = str(self.case_name or "").strip()
        if not name:
            raise ValueError("child Case call name must be non-empty")
        if not case_name:
            raise ValueError("child Case name must be non-empty")
        timeout = self.timeout_seconds
        if timeout is not None and float(timeout) < 0:
            raise ValueError("child Case timeout_seconds must be non-negative")
        refs = {
            str(key): value if isinstance(value, DataRef) else DataRef.from_dict(value)
            for key, value in dict(self.input_artifacts or {}).items()
        }
        bound_inputs = {
            str(key): (
                value
                if isinstance(value, ArtifactBinding)
                else ArtifactBinding.from_dict(value)
            )
            for key, value in dict(self.input_artifact_bindings or {}).items()
        }
        if refs and not bound_inputs:
            raise ValueError(
                "ChildCaseCall input_artifacts require input_artifact_bindings"
            )
        if refs and refs != {key: value.ref for key, value in bound_inputs.items()}:
            raise ValueError(
                "ChildCaseCall input_artifacts do not match input_artifact_bindings"
            )
        budgets = {str(key): int(value) for key, value in dict(self.budget_request or {}).items()}
        if any(value < 0 for value in budgets.values()):
            raise ValueError("child Case budget values must be non-negative")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "case_name", case_name)
        object.__setattr__(self, "case_kind", str(self.case_kind or "solver"))
        object.__setattr__(self, "mode", str(self.mode or "build"))
        object.__setattr__(self, "resource_request", dict(self.resource_request or {}))
        object.__setattr__(self, "budget_request", budgets)
        object.__setattr__(self, "component_overrides", dict(self.component_overrides or {}))
        object.__setattr__(
            self,
            "input_artifacts",
            {key: value.ref for key, value in bound_inputs.items()},
        )
        object.__setattr__(self, "input_artifact_bindings", bound_inputs)
        object.__setattr__(
            self,
            "artifact_bindings",
            {str(key): str(value) for key, value in dict(self.artifact_bindings or {}).items()},
        )
        object.__setattr__(self, "inputs", dict(self.inputs or {}))
        object.__setattr__(self, "argv", tuple(str(item) for item in tuple(self.argv or ())))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        object.__setattr__(self, "timeout_seconds", None if timeout is None else float(timeout))

    def build_request(
        self,
        runtime: CaseInvocationRuntime,
        *,
        stage_name: str,
        artifacts: Mapping[str, ArtifactBinding],
    ) -> CaseRunRequest:
        resolved_bindings = dict(self.input_artifact_bindings)
        for input_name, artifact_name in self.artifact_bindings.items():
            try:
                resolved_bindings[input_name] = artifacts[artifact_name]
            except KeyError as exc:
                raise ProjectConfigurationError(
                    f"child Case call '{self.name}' requires missing artifact "
                    f"'{artifact_name}' for input '{input_name}'"
                ) from exc
        control = ExecutionControl.with_timeout(self.timeout_seconds)
        return CaseRunRequest(
            project_name=runtime.request.project_name,
            stage_name=str(stage_name),
            case_name=self.case_name,
            case_kind=self.case_kind,
            mode=self.mode,
            control=control,
            resource_request=self.resource_request,
            budget_request=self.budget_request,
            component_overrides=self.component_overrides,
            input_artifacts={
                name: binding.ref for name, binding in resolved_bindings.items()
            },
            input_artifact_bindings=resolved_bindings,
            inputs=self.inputs,
            argv=self.argv,
            metadata={**dict(self.metadata), "child_call_name": self.name},
        )


@dataclass(frozen=True)
class CaseStage:
    """A serial or parallel group of complete child Case calls."""

    name: str
    calls: tuple[ChildCaseCall, ...]
    policy: str = "serial"
    failure_policy: str = "fail_fast"
    max_workers: int = 0
    cancellation_grace_seconds: float = 5.0

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        calls = tuple(self.calls or ())
        policy = str(self.policy or "serial").strip().lower()
        failure = str(self.failure_policy or "fail_fast").strip().lower()
        if not name:
            raise ValueError("Case stage name must be non-empty")
        if not calls:
            raise ValueError("Case stage requires at least one child call")
        if any(not isinstance(call, ChildCaseCall) for call in calls):
            raise TypeError("Case stage calls must be ChildCaseCall values")
        names = tuple(call.name for call in calls)
        if len(names) != len(set(names)):
            raise ValueError("child Case call names must be unique within a stage")
        if policy not in {"serial", "parallel"}:
            raise ValueError("Case stage policy must be 'serial' or 'parallel'")
        if failure not in {"fail_fast", "continue"}:
            raise ValueError("Case stage failure_policy must be 'fail_fast' or 'continue'")
        grace = float(self.cancellation_grace_seconds)
        if grace < 0:
            raise ValueError("Case stage cancellation_grace_seconds must be non-negative")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "calls", calls)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "failure_policy", failure)
        object.__setattr__(self, "max_workers", max(0, int(self.max_workers or 0)))
        object.__setattr__(self, "cancellation_grace_seconds", grace)


@dataclass(frozen=True)
class CaseStageResult:
    """Structured stage result retaining every child Case envelope."""

    stage_name: str
    results: Mapping[str, CaseRunResult]
    artifact_refs: Mapping[str, DataRef]
    stopped_early: bool = False
    cancelled_calls: tuple[str, ...] = ()
    cancellation_overdue_calls: tuple[str, ...] = ()
    control_cleanup: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.results) and all(result.ok for result in self.results.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "ok": self.ok,
            "stopped_early": self.stopped_early,
            "cancelled_calls": list(self.cancelled_calls),
            "cancellation_overdue_calls": list(self.cancellation_overdue_calls),
            "control_cleanup": dict(self.control_cleanup),
            "results": {name: result.as_dict() for name, result in self.results.items()},
            "artifact_refs": {name: ref.as_dict() for name, ref in self.artifact_refs.items()},
        }


class CaseStageRunner:
    """Run complete child Cases while preserving lineage, grants and failures."""

    def __init__(
        self,
        runtime: CaseInvocationRuntime,
        stages: Sequence[CaseStage],
        *,
        artifact_refs: Mapping[str, DataRef] | None = None,
        artifact_bindings: Mapping[str, ArtifactBinding] | None = None,
    ) -> None:
        self.runtime = runtime
        self.stages = tuple(stages or ())
        if not self.stages:
            raise ValueError("CaseStageRunner requires at least one stage")
        if any(not isinstance(stage, CaseStage) for stage in self.stages):
            raise TypeError("stages must contain CaseStage values")
        names = tuple(stage.name for stage in self.stages)
        if len(names) != len(set(names)):
            raise ValueError("Case stage names must be unique")
        supplied_refs = {
            str(key): value if isinstance(value, DataRef) else DataRef.from_dict(value)
            for key, value in dict(artifact_refs or {}).items()
        }
        self._artifact_bindings = {
            str(key): (
                value
                if isinstance(value, ArtifactBinding)
                else ArtifactBinding.from_dict(value)
            )
            for key, value in dict(artifact_bindings or {}).items()
        }
        bound_refs = {
            key: binding.ref for key, binding in self._artifact_bindings.items()
        }
        if supplied_refs and supplied_refs != bound_refs:
            raise ValueError(
                "CaseStageRunner artifact_refs require matching ArtifactBinding values"
            )
        self._artifacts = bound_refs
        self._results: list[CaseStageResult] = []

    @property
    def artifact_refs(self) -> Mapping[str, DataRef]:
        return dict(self._artifacts)

    @property
    def results(self) -> tuple[CaseStageResult, ...]:
        return tuple(self._results)

    def run(self) -> tuple[CaseStageResult, ...]:
        for stage in self.stages:
            self.runtime.checkpoint()
            result = self._run_stage(stage)
            self._results.append(result)
            self.runtime.checkpoint()
            if not result.ok and stage.failure_policy == "fail_fast":
                break
        return self.results

    def _run_stage(self, stage: CaseStage) -> CaseStageResult:
        if stage.policy == "parallel":
            return self._run_parallel(stage)
        return self._run_serial(stage)

    def _run_serial(self, stage: CaseStage) -> CaseStageResult:
        results: dict[str, CaseRunResult] = {}
        stopped = False
        for call in stage.calls:
            self.runtime.checkpoint()
            result = self._invoke(stage, call, dict(self._artifact_bindings))
            results[call.name] = result
            if result.ok:
                self._merge_artifacts(call, result)
            if not result.ok and stage.failure_policy == "fail_fast":
                stopped = True
                break
        return CaseStageResult(stage.name, results, dict(self._artifacts), stopped)

    def _run_parallel(self, stage: CaseStage) -> CaseStageResult:
        snapshot = dict(self._artifact_bindings)
        requested_workers = stage.max_workers or len(stage.calls)
        workers = self.runtime.stage_worker_capacity(requested_workers)
        results: dict[str, CaseRunResult] = {}
        requests = {
            call.name: call.build_request(
                self.runtime,
                stage_name=stage.name,
                artifacts=snapshot,
            )
            for call in stage.calls
        }
        stage_ref = self.runtime.request.control.derive_child(
            CancellationRef()
        ).cancellation
        stage_token = self.runtime.cancellation_token(stage_ref)
        heartbeat: CancellationHeartbeat | None = None
        primary_error: BaseException | None = None
        stage_result: CaseStageResult | None = None
        cleanup_evidence: dict[str, Any] = {
            "schema": "blackbase.stage_control_cleanup/v1",
            "control_id": stage_ref.control_id,
            "heartbeat_closed": False,
            "retired": False,
            "issues": [],
        }
        stopped = False
        cancellation_deadline: float | None = None
        cancelled: list[str] = []
        cancellation_overdue: set[str] = set()
        try:
            heartbeat = CancellationHeartbeat(stage_token)
            with self.runtime.stage_executor(workers) as pool:
                futures: dict[Future[CaseRunResult], ChildCaseCall] = {
                    pool.submit(
                        self._invoke_request,
                        requests[call.name],
                        (stage_ref,),
                    ): call
                    for call in stage.calls
                }
                pending = set(futures)
                while pending:
                    heartbeat.assert_current()
                    done, _not_done = wait(
                        pending,
                        timeout=0.05,
                        return_when=FIRST_COMPLETED,
                    )
                    if (
                        cancellation_deadline is not None
                        and time.monotonic() >= cancellation_deadline
                    ):
                        cancellation_overdue.update(
                            futures[future].name
                            for future in pending
                            if not future.cancelled() and not future.done()
                        )
                        # A cooperative thread cannot be abandoned safely.  The
                        # grace deadline is audit evidence only; structured thread
                        # stages continue joining every started child.  Callers that
                        # need bounded termination must request isolated execution.
                        cancellation_deadline = None
                    if not done:
                        self.runtime.checkpoint()
                        continue
                    pending.difference_update(done)
                    for future in done:
                        call = futures[future]
                        result = self._future_result(call, requests[call.name], future)
                        if call.name in cancellation_overdue:
                            result = replace(
                                result,
                                metadata={
                                    **dict(result.metadata),
                                    "stage_cancellation_overdue": True,
                                },
                            )
                        results[call.name] = result
                        if (
                            not result.ok
                            and stage.failure_policy == "fail_fast"
                            and not stopped
                        ):
                            stopped = True
                            stage_token.cancel(
                                f"parallel stage '{stage.name}' failed at child '{call.name}'"
                            )
                            for sibling in pending:
                                sibling.cancel()
                            cancellation_deadline = (
                                time.monotonic() + stage.cancellation_grace_seconds
                            )
                for future, call in futures.items():
                    if future.cancelled():
                        cancelled.append(call.name)
                    if call.name not in results:
                        results[call.name] = self._future_result(
                            call,
                            requests[call.name],
                            future,
                        )
            heartbeat.assert_current()
            for call in stage.calls:
                result = results[call.name]
                if result.ok:
                    self._merge_artifacts(call, result)
            ordered = {call.name: results[call.name] for call in stage.calls}
            cancelled_set = set(cancelled)
            stage_result = CaseStageResult(
                stage.name,
                ordered,
                dict(self._artifacts),
                stopped,
                tuple(call.name for call in stage.calls if call.name in cancelled_set),
                tuple(
                    call.name for call in stage.calls
                    if call.name in cancellation_overdue
                ),
            )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            cleanup_issues: list[dict[str, str]] = []
            if heartbeat is not None:
                try:
                    heartbeat.close()
                    cleanup_evidence["heartbeat_closed"] = True
                except BaseException as exc:
                    cleanup_issues.append(
                        {"phase": "heartbeat_close", "type": type(exc).__name__, "message": str(exc)}
                    )
            try:
                stage_token.retire()
                cleanup_evidence["retired"] = True
            except BaseException as exc:
                cleanup_issues.append(
                    {"phase": "control_retire", "type": type(exc).__name__, "message": str(exc)}
                )
            cleanup_evidence["issues"] = cleanup_issues
            if cleanup_issues:
                if primary_error is not None:
                    attach_failure_evidence(
                        primary_error,
                        "stage_control_cleanup",
                        cleanup_evidence,
                    )
                    setattr(primary_error, "_blackbase_stage_control_cleanup", cleanup_evidence)
                    add_note = getattr(primary_error, "add_note", None)
                    if callable(add_note):
                        add_note(
                            "Parallel Stage cancellation-control cleanup failed: "
                            f"{cleanup_issues!r}"
                        )
                else:
                    error = RuntimeError(
                        "parallel Stage cancellation-control cleanup failed"
                    )
                    attach_failure_evidence(
                        error,
                        "stage_control_cleanup",
                        cleanup_evidence,
                    )
                    setattr(error, "_blackbase_stage_control_cleanup", cleanup_evidence)
                    raise error
        if stage_result is None:  # pragma: no cover - guarded by the try block
            raise RuntimeError("parallel Stage completed without a result")
        return replace(stage_result, control_cleanup=cleanup_evidence)

    @staticmethod
    def _cancelled_result(
        request: CaseRunRequest,
        *,
        kind: str,
        message: str,
    ) -> CaseRunResult:
        now = time.time()
        return CaseRunResult(
            request=request,
            status="cancelled",
            started_at=now,
            finished_at=now,
            exit_code=1,
            failure=CaseFailure(kind=kind, message=message),
        )

    @classmethod
    def _future_result(
        cls,
        call: ChildCaseCall,
        request: CaseRunRequest,
        future: Future[CaseRunResult],
    ) -> CaseRunResult:
        try:
            return future.result()
        except CancelledError:
            return cls._cancelled_result(
                request,
                kind="stage_cancelled_before_start",
                message="child Case was cancelled before execution",
            )
        except BaseException as exc:
            now = time.time()
            return CaseRunResult(
                request=request,
                status="failed",
                started_at=now,
                finished_at=now,
                exit_code=1,
                failure=CaseFailure(
                    kind="stage_child_exception",
                    message=f"child Case '{call.name}' raised outside its result envelope",
                    cause={"type": type(exc).__name__, "message": str(exc)},
                ),
            )

    def _invoke(
        self,
        stage: CaseStage,
        call: ChildCaseCall,
        artifacts: Mapping[str, ArtifactBinding],
    ) -> CaseRunResult:
        request = call.build_request(
            self.runtime,
            stage_name=stage.name,
            artifacts=artifacts,
        )
        return self._invoke_request(request)

    def _invoke_request(
        self,
        request: CaseRunRequest,
        intermediate_cancellations: Sequence[CancellationRef] = (),
    ) -> CaseRunResult:
        if not intermediate_cancellations:
            return self.runtime.invoke(request)
        return self.runtime.invoke(
            request,
            intermediate_cancellations=intermediate_cancellations,
        )

    def _merge_artifacts(self, call: ChildCaseCall, result: CaseRunResult) -> None:
        for name, ref in result.artifact_refs.items():
            if name not in result.artifact_publications:
                continue
            qualified = f"{call.name}.{name}"
            if qualified in self._artifacts and self._artifacts[qualified] != ref:
                raise ProjectConfigurationError(
                    f"child Case artifact '{qualified}' was published more than once"
                )
            self._artifacts[qualified] = ref
            self._artifact_bindings[qualified] = ArtifactBinding(
                ref=ref,
                publication=result.artifact_publications[name],
            )
            if name not in self._artifacts:
                self._artifacts[name] = ref
                self._artifact_bindings[name] = self._artifact_bindings[qualified]


__all__ = [
    "CaseInvocationRuntime",
    "CaseStage",
    "CaseStageResult",
    "CaseStageRunner",
    "ChildCaseCall",
]
