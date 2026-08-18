"""Composable child-Case stages over the canonical invocation protocol.

This module contains orchestration only.  It never constructs a Solver or a
Trainer and therefore remains independent of downstream semantic frameworks.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from blackbase.resources import DataRef

from .execution import (
    CaseRunRequest,
    CaseRunResult,
    ExecutionControl,
    ProjectConfigurationError,
)


class CaseInvocationRuntime(Protocol):
    """Minimum parent runtime surface required by :class:`CaseStageRunner`."""

    request: CaseRunRequest

    def checkpoint(self) -> None: ...

    def invoke(self, request: CaseRunRequest) -> CaseRunResult: ...


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
        object.__setattr__(self, "input_artifacts", refs)
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
        artifacts: Mapping[str, DataRef],
    ) -> CaseRunRequest:
        resolved = dict(self.input_artifacts)
        for input_name, artifact_name in self.artifact_bindings.items():
            try:
                resolved[input_name] = artifacts[artifact_name]
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
            input_artifacts=resolved,
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
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "calls", calls)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "failure_policy", failure)
        object.__setattr__(self, "max_workers", max(0, int(self.max_workers or 0)))


@dataclass(frozen=True)
class CaseStageResult:
    """Structured stage result retaining every child Case envelope."""

    stage_name: str
    results: Mapping[str, CaseRunResult]
    artifact_refs: Mapping[str, DataRef]
    stopped_early: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.results) and all(result.ok for result in self.results.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "ok": self.ok,
            "stopped_early": self.stopped_early,
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
        self._artifacts = {
            str(key): value if isinstance(value, DataRef) else DataRef.from_dict(value)
            for key, value in dict(artifact_refs or {}).items()
        }
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
            result = self._invoke(stage, call, dict(self._artifacts))
            results[call.name] = result
            self._merge_artifacts(call, result)
            if not result.ok and stage.failure_policy == "fail_fast":
                stopped = True
                break
        return CaseStageResult(stage.name, results, dict(self._artifacts), stopped)

    def _run_parallel(self, stage: CaseStage) -> CaseStageResult:
        snapshot = dict(self._artifacts)
        workers = stage.max_workers or len(stage.calls)
        results: dict[str, CaseRunResult] = {}
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(stage.calls)))) as pool:
            futures = {
                pool.submit(self._invoke, stage, call, snapshot): call
                for call in stage.calls
            }
            for future in as_completed(futures):
                call = futures[future]
                result = future.result()
                results[call.name] = result
        for call in stage.calls:
            self._merge_artifacts(call, results[call.name])
        ordered = {call.name: results[call.name] for call in stage.calls}
        return CaseStageResult(stage.name, ordered, dict(self._artifacts), False)

    def _invoke(
        self,
        stage: CaseStage,
        call: ChildCaseCall,
        artifacts: Mapping[str, DataRef],
    ) -> CaseRunResult:
        request = call.build_request(
            self.runtime,
            stage_name=stage.name,
            artifacts=artifacts,
        )
        return self.runtime.invoke(request)

    def _merge_artifacts(self, call: ChildCaseCall, result: CaseRunResult) -> None:
        for name, ref in result.artifact_refs.items():
            qualified = f"{call.name}.{name}"
            if qualified in self._artifacts and self._artifacts[qualified] != ref:
                raise ProjectConfigurationError(
                    f"child Case artifact '{qualified}' was published more than once"
                )
            self._artifacts[qualified] = ref
            if name not in self._artifacts:
                self._artifacts[name] = ref


__all__ = [
    "CaseInvocationRuntime",
    "CaseStage",
    "CaseStageResult",
    "CaseStageRunner",
    "ChildCaseCall",
]
