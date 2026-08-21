from __future__ import annotations

from dataclasses import replace
import threading
import time

from blackbase.project import (
    CaseRunRequest,
    CaseRunResult,
    CaseStage,
    CaseStageRunner,
    ChildCaseCall,
)
from blackbase.resources import CancellationRef, CancellationToken, DataRef


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

    def invoke(
        self,
        request: CaseRunRequest,
        *,
        intermediate_cancellations=(),
    ) -> CaseRunResult:
        del intermediate_cancellations
        self.requests.append(request)
        ref = DataRef(uri=f"memory://{request.case_name}", kind="artifact", backend="memory")
        return CaseRunResult(
            request=replace(request, identity=self.request.identity.child()),
            status="succeeded",
            artifact_refs={"model": ref},
            output={"case": request.case_name},
        )

    def cancellation_token(self, ref: CancellationRef) -> CancellationToken:
        return CancellationToken(ref)


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

    def invoke(request: CaseRunRequest, *, intermediate_cancellations=()) -> CaseRunResult:
        del intermediate_cancellations
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


def test_parallel_fail_fast_cancels_running_and_pending_siblings() -> None:
    runtime = _Runtime()
    slow_started = threading.Event()

    def invoke(request: CaseRunRequest, *, intermediate_cancellations=()) -> CaseRunResult:
        runtime.requests.append(request)
        if request.case_name == "bad":
            assert slow_started.wait(timeout=1.0)
            return CaseRunResult(request=request, status="failed", exit_code=1)
        token = runtime.cancellation_token(intermediate_cancellations[-1])
        if request.case_name == "slow":
            slow_started.set()
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if token.cancelled:
                    return CaseRunResult(request=request, status="cancelled", exit_code=1)
                time.sleep(0.005)
        if request.case_name == "queued":
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if token.cancelled:
                    return CaseRunResult(request=request, status="cancelled", exit_code=1)
                time.sleep(0.005)
        return CaseRunResult(request=request, status="succeeded")

    runtime.invoke = invoke  # type: ignore[method-assign]
    started_at = time.monotonic()
    result = CaseStageRunner(
        runtime,
        (
            CaseStage(
                "parallel",
                (
                    ChildCaseCall("bad", "bad"),
                    ChildCaseCall("slow", "slow"),
                    ChildCaseCall("queued", "queued"),
                ),
                policy="parallel",
                failure_policy="fail_fast",
                max_workers=2,
                cancellation_grace_seconds=0.5,
            ),
        ),
    ).run()[0]

    assert time.monotonic() - started_at < 0.9
    assert result.stopped_early is True
    assert result.results["bad"].status == "failed"
    assert result.results["slow"].status == "cancelled"
    assert result.results["queued"].status == "cancelled"
    assert set(result.cancelled_calls).issubset({"queued"})
    assert result.still_running_calls == ()
