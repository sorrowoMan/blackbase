from __future__ import annotations

from dataclasses import replace
import threading
import time

import pytest

from blackbase.project import (
    CaseRunRequest,
    CaseRunResult,
    CaseStage,
    CaseStageRunner,
    ChildCaseCall,
    ExecutionControl,
)
from blackbase.resources import (
    ArtifactPublicationReceipt,
    CancellationRef,
    CancellationToken,
    DataRef,
    PoolScheduler,
)


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
        self.stage_scheduler = PoolScheduler(total_threads=2)

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
        identity = self.request.identity.child()
        ref = DataRef(
            uri=f"memory://{request.case_name}",
            kind="artifact",
            backend="memory",
            checksum="sha256:" + "a" * 64,
        )
        receipt = ArtifactPublicationReceipt(
            publication_id=f"publication-{request.case_name}",
            artifact_name="model",
            ref=ref,
            project_run_id=identity.project_run_id,
            case_run_id=identity.case_run_id,
            transaction_id=f"transaction-{request.case_name}",
            authority_namespace="test",
            committed_at=time.time(),
            metadata={"case_finalization_sealed": True},
        )
        return CaseRunResult(
            request=replace(request, identity=identity),
            status="succeeded",
            artifact_refs={"model": ref},
            artifact_publications={"model": receipt},
            output={"case": request.case_name},
        )

    def cancellation_token(self, ref: CancellationRef) -> CancellationToken:
        return CancellationToken(ref)

    def stage_worker_capacity(self, requested_workers: int) -> int:
        return max(1, min(int(requested_workers), 2))

    def stage_executor(self, max_workers: int):
        return self.stage_scheduler.as_executor(
            self.stage_worker_capacity(max_workers)
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
    assert result.cancellation_overdue_calls == ()


def test_parallel_stage_caps_host_fanout_to_parent_l0_grant() -> None:
    runtime = _Runtime()
    lock = threading.Lock()
    active = 0
    peak = 0

    def invoke(request: CaseRunRequest, *, intermediate_cancellations=()) -> CaseRunResult:
        del intermediate_cancellations
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.03)
            return CaseRunResult(request=request, status="succeeded")
        finally:
            with lock:
                active -= 1

    runtime.invoke = invoke  # type: ignore[method-assign]
    calls = tuple(ChildCaseCall(f"call-{index}", f"case-{index}") for index in range(8))

    result = CaseStageRunner(
        runtime,
        (
            CaseStage(
                "parallel",
                calls,
                policy="parallel",
                max_workers=32,
            ),
        ),
    ).run()[0]

    assert result.ok
    assert peak == 2


def test_parallel_thread_stage_joins_child_that_exceeds_cancellation_grace() -> None:
    runtime = _Runtime()
    slow_started = threading.Event()
    slow_finished = threading.Event()

    def invoke(request: CaseRunRequest, *, intermediate_cancellations=()) -> CaseRunResult:
        del intermediate_cancellations
        if request.case_name == "slow":
            slow_started.set()
            time.sleep(0.12)
            slow_finished.set()
            return CaseRunResult(request=request, status="succeeded")
        assert slow_started.wait(timeout=1.0)
        return CaseRunResult(request=request, status="failed", exit_code=1)

    runtime.invoke = invoke  # type: ignore[method-assign]
    started_at = time.monotonic()
    result = CaseStageRunner(
        runtime,
        (
            CaseStage(
                "parallel",
                (
                    ChildCaseCall("slow", "slow"),
                    ChildCaseCall("bad", "bad"),
                ),
                policy="parallel",
                failure_policy="fail_fast",
                max_workers=2,
                cancellation_grace_seconds=0.01,
            ),
        ),
    ).run()[0]

    assert slow_finished.is_set()
    assert time.monotonic() - started_at >= 0.1
    assert result.cancellation_overdue_calls == ("slow",)
    assert result.results["slow"].status == "succeeded"
    assert result.results["slow"].metadata["stage_cancellation_overdue"] is True
    assert all(item.status != "running" for item in result.results.values())


def test_parallel_stage_heartbeats_and_retires_owned_control(tmp_path) -> None:
    runtime = _Runtime()
    parent_ref = CancellationRef(
        backend="sqlite",
        path=str(tmp_path / "stage-control.sqlite3"),
        active_ttl_seconds=0.12,
        heartbeat_seconds=0.03,
    )
    parent_token = CancellationToken(parent_ref)
    runtime.request = replace(
        runtime.request,
        control=ExecutionControl(cancellation=parent_ref),
    )
    captured: dict[str, CancellationToken] = {}

    def cancellation_token(ref: CancellationRef) -> CancellationToken:
        token = CancellationToken(ref)
        captured["stage"] = token
        return token

    def invoke(request: CaseRunRequest, *, intermediate_cancellations=()) -> CaseRunResult:
        assert intermediate_cancellations
        deadline = time.monotonic() + 0.24
        while time.monotonic() < deadline:
            assert captured["stage"].touch() is True
            time.sleep(0.035)
        return CaseRunResult(request=request, status="succeeded")

    runtime.cancellation_token = cancellation_token  # type: ignore[method-assign]
    runtime.invoke = invoke  # type: ignore[method-assign]
    try:
        result = CaseStageRunner(
            runtime,
            (
                CaseStage(
                    "parallel",
                    (ChildCaseCall("slow", "slow"),),
                    policy="parallel",
                ),
            ),
        ).run()[0]
    finally:
        parent_token.retire()

    assert result.ok
    assert result.control_cleanup["heartbeat_closed"] is True
    assert result.control_cleanup["retired"] is True
    assert result.control_cleanup["issues"] == []
    assert captured["stage"].touch() is False


def test_parallel_stage_exposes_control_cleanup_failure_evidence() -> None:
    runtime = _Runtime()

    class _RetireFailureToken(CancellationToken):
        def retire(self) -> None:
            raise RuntimeError("retire failed")

    runtime.cancellation_token = (  # type: ignore[method-assign]
        lambda ref: _RetireFailureToken(ref)
    )

    with pytest.raises(
        RuntimeError,
        match="cancellation-control cleanup failed",
    ) as captured:
        CaseStageRunner(
            runtime,
            (
                CaseStage(
                    "parallel",
                    (ChildCaseCall("only", "only"),),
                    policy="parallel",
                ),
            ),
        ).run()

    evidence = captured.value._blackbase_stage_control_cleanup
    assert evidence["heartbeat_closed"] is True
    assert evidence["retired"] is False
    assert evidence["issues"][0]["phase"] == "control_retire"
