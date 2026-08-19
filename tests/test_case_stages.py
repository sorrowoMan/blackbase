from __future__ import annotations

from dataclasses import replace

from blackbase.project import (
    CaseRunRequest,
    CaseRunResult,
    CaseStage,
    CaseStageRunner,
    ChildCaseCall,
)
from blackbase.resources import DataRef


class _Runtime:
    def __init__(self) -> None:
        self.request = CaseRunRequest(
            project_name="project",
            stage_name="outer",
            case_name="parent",
            resource_context={"threads": 2, "grant": {"threads": 2, "workers": 2}},
        )
        self.requests: list[CaseRunRequest] = []
        self.checkpoints = 0

    def checkpoint(self) -> None:
        self.checkpoints += 1

    def invoke(self, request: CaseRunRequest) -> CaseRunResult:
        self.requests.append(request)
        ref = DataRef(uri=f"memory://{request.case_name}", kind="artifact", backend="memory")
        return CaseRunResult(
            request=replace(request, identity=self.request.identity.child()),
            status="succeeded",
            artifact_refs={"model": ref},
            output={"case": request.case_name},
        )


def test_case_stage_routes_artifacts_through_complete_child_requests() -> None:
    runtime = _Runtime()
    runner = CaseStageRunner(
        runtime,
        (
            CaseStage("prepare", (ChildCaseCall("train", "trainer", case_kind="trainer"),)),
            CaseStage(
                "consume",
                (
                    ChildCaseCall(
                        "solve",
                        "solver",
                        artifact_bindings={"trained_model": "train.model"},
                        timeout_seconds=3.0,
                    ),
                ),
            ),
        ),
    )

    results = runner.run()

    assert all(result.ok for result in results)
    assert runtime.requests[1].input_artifacts["trained_model"].uri == "memory://trainer"
    assert runtime.requests[1].control.deadline_at > 0
    assert runtime.requests[1].resource_context == {}
    assert runtime.checkpoints >= 4


def test_parallel_stage_retains_structured_failure_without_fabricating_output() -> None:
    runtime = _Runtime()

    def invoke(request: CaseRunRequest) -> CaseRunResult:
        runtime.requests.append(request)
        if request.case_name == "bad":
            return CaseRunResult(request=request, status="failed", exit_code=1)
        return CaseRunResult(request=request, status="succeeded", output={"ok": True})

    runtime.invoke = invoke  # type: ignore[method-assign]
    result = CaseStageRunner(
        runtime,
        (
            CaseStage(
                "parallel",
                (ChildCaseCall("good", "good"), ChildCaseCall("bad", "bad")),
                policy="parallel",
                failure_policy="continue",
            ),
        ),
    ).run()[0]

    assert not result.ok
    assert result.results["bad"].status == "failed"
    assert result.results["bad"].output == {}
