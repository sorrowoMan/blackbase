from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from blackbase.project import (
    attach_failure_evidence,
    CaseExecutor,
    CaseFailure,
    CaseInvocationError,
    CaseInvoker,
    CaseRunRequest,
    CaseRunResult,
    ChildResourceGrant,
    ExecutionControl,
    execute_project,
)
from blackbase.project.scaffold import add_case, create_project
from blackbase.resources import (
    CancellationHeartbeat,
    CancellationRef,
    CancellationToken,
    BudgetHandle,
    ResourceRequest,
)
from blackbase.resources import PoolScheduler


def test_parent_case_invokes_complete_child_with_lineage_grant_budget_and_artifact(
    tmp_path,
) -> None:
    project_root = create_project(tmp_path / "nested_project", framework="blackbase")
    child_root = add_case("child", "solver", project_root=project_root)
    parent_root = add_case("parent", "solver", project_root=project_root)
    (child_root / "build_solver.py").write_text(
        """
from blackbase.resources import BudgetAccount


class Child:
    def __init__(self, resource_context=None):
        self.resource_context = dict(resource_context or {})
        self.case_inputs = {}

    def set_case_inputs(self, inputs):
        self.case_inputs = dict(inputs)

    def run(self):
        account = BudgetAccount.from_resource_context(
            "evaluations",
            self.resource_context,
        )
        claim = account.reserve(3)
        account.consume(claim, 2)
        account.complete(claim)
        answer_ref = self.case_runtime.publish_artifact(
            "answer",
            {"value": self.case_inputs["value"] * 2},
            kind="result",
        )
        return {
            "value": self.case_inputs["value"] * 2,
            "identity": self.case_runtime.identity.as_dict(),
            "control": self.case_runtime.control.as_dict(),
            "threads": self.resource_context["threads"],
            "grant": self.resource_context["metadata"]["child_grant"],
            "artifact_refs": {"answer": answer_ref},
        }


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, component_overrides
    return Child(resource_context)
""".lstrip(),
        encoding="utf-8",
    )
    (parent_root / "build_solver.py").write_text(
        """
from blackbase.project import CaseRunRequest


class Parent:
    def __init__(self, resource_context=None):
        self.resource_context = dict(resource_context or {})

    def run(self):
        child = self.case_runtime.invoke(
            CaseRunRequest(
                project_name="nested_project",
                stage_name="nested",
                case_name="child",
                resource_request={"workers": 1, "threads": 1},
                budget_request={"evaluations": 3},
                inputs={"value": 21},
            )
        )
        return {
            "child": child.as_dict(),
            "artifact_refs": {"answer": child.artifact_refs["answer"]},
        }


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, component_overrides
    return Parent(resource_context)
""".lstrip(),
        encoding="utf-8",
    )
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "nested_project"
L0 = {
    "namespace": "nested_project",
    "offer": {"threads": 2, "gpus": 0, "backend": "local"},
    "policy": {"mode": "strict", "max_workers": 1, "max_threads": 2},
    "default_request": {"workers": 1, "threads": 2, "gpus": 0},
    "lease_backend": "sqlite",
    "lease_path": ".blackbase/l0.sqlite",
    "budgets": {"evaluations": 5},
}
STAGES = [{"name": "outer", "cases": ["parent"]}]
GROUPS = {"default": {"stages": ["outer"]}}
""".lstrip(),
        encoding="utf-8",
    )

    project_result = execute_project(project_root, run_id="nested-run")

    assert project_result.ok
    parent_result = project_result.case_results[0]
    child_result = CaseRunResult.from_dict(parent_result.output["child"])
    assert child_result.ok
    assert child_result.output["value"] == 42
    assert child_result.identity.parent_case_run_id == parent_result.identity.case_run_id
    assert child_result.identity.root_run_id == parent_result.identity.root_run_id
    assert child_result.identity.depth == parent_result.identity.depth + 1
    assert child_result.control.ancestor_cancellations == ()
    assert child_result.control.cancellation.parent_control_id == (
        parent_result.request.control.cancellation.control_id
    )
    assert child_result.control.cancellation.lineage_depth == (
        parent_result.request.control.cancellation.lineage_depth + 1
    )
    assert child_result.output["threads"] == 1
    assert child_result.output["grant"]["parent_case_run_id"] == (
        parent_result.identity.case_run_id
    )
    assert child_result.budget_usage["evaluations"]["charged_to_parent"] == 2
    assert child_result.budget_usage["evaluations"]["returned_to_parent"] == 1
    assert child_result.artifact_refs["answer"].uri
    assert project_result.artifact_registry["parent.answer"] == (
        child_result.artifact_refs["answer"]
    )
    with sqlite3.connect(project_root / ".blackbase" / "l0_controls.sqlite") as connection:
        remaining_controls = connection.execute(
            "SELECT COUNT(*) FROM cancellation_controls"
        ).fetchone()[0]
    assert remaining_controls == 0


@pytest.mark.parametrize("commit", [True, False])
def test_case_artifact_transaction_exposes_all_or_no_refs(tmp_path, commit) -> None:
    project_root = create_project(
        tmp_path / f"artifact_transaction_{commit}",
        framework="blackbase",
    )
    case_root = add_case("publisher", "solver", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        f"""
class Publisher:
    def run(self):
        transaction = self.case_runtime.begin_finalization_transaction("test-finalization")
        transaction.publish("model", {{"value": 1}}, kind="model")
        transaction.publish("report", {{"ok": True}}, kind="report")
        visible_before = sorted(self.case_runtime.artifact_refs)
        if {commit!r}:
            committed = transaction.prepare()
            return {{
                "visible_before": visible_before,
                "committed": sorted(committed),
            }}
        transaction.abort("test-abort")
        return {{"visible_before": visible_before, "committed": []}}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Publisher()
""".lstrip(),
        encoding="utf-8",
    )
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "artifact_transaction"
STAGES = [{"name": "publish", "cases": ["publisher"]}]
GROUPS = {"default": {"stages": ["publish"]}}
""".lstrip(),
        encoding="utf-8",
    )

    project_result = execute_project(project_root, record=False)

    assert project_result.ok
    case_result = project_result.case_results[0]
    assert case_result.output["visible_before"] == ()
    if commit:
        assert set(case_result.artifact_refs) == {"model", "report"}
        assert set(case_result.artifact_publications) == {"model", "report"}
    else:
        assert case_result.artifact_refs == {}


def test_finalized_observer_runs_after_seal_and_cannot_veto_success(tmp_path) -> None:
    project_root = create_project(
        tmp_path / "post_seal_observer",
        framework="blackbase",
    )
    case_root = add_case("publisher", "solver", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        """
class Publisher:
    def run(self):
        transaction = self.case_runtime.begin_finalization_transaction("result")
        transaction.publish("model", {"value": 1}, kind="model")

        def observe(publications):
            receipt = publications["model"]
            assert receipt.metadata["case_finalization_sealed"] is True
            raise RuntimeError("diagnostic observer failed")

        self.case_runtime.register_finalization_observer(
            observe,
            name="test.post_seal",
        )
        return {"value": 1}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Publisher()
""".lstrip(),
        encoding="utf-8",
    )
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "post_seal_observer"
STAGES = [{"name": "publish", "cases": ["publisher"]}]
GROUPS = {"default": {"stages": ["publish"]}}
""".lstrip(),
        encoding="utf-8",
    )

    project_result = execute_project(project_root, record=False)
    result = project_result.case_results[0]

    assert project_result.ok
    assert result.ok
    assert set(result.artifact_refs) == {"model"}
    assert result.artifact_publications["model"].metadata[
        "case_finalization_sealed"
    ] is True
    observer_audit = result.metadata["finalization_observers"]
    assert observer_audit["status"] == "degraded"
    assert observer_audit["failure_count"] == 1
    assert observer_audit["failures"][0]["observer"] == "test.post_seal"
    assert observer_audit["failures"][0]["message"] == (
        "diagnostic observer failed"
    )


def test_child_resource_grants_serialize_overcommitted_parallel_invocations(tmp_path) -> None:
    project_root = create_project(tmp_path / "grant_project", framework="blackbase")
    child_root = add_case("child", "solver", project_root=project_root)
    parent_root = add_case("parent", "solver", project_root=project_root)
    (child_root / "build_solver.py").write_text(
        """
import time


class Child:
    def run(self):
        started_at = time.monotonic()
        time.sleep(0.18)
        return {"started_at": started_at, "finished_at": time.monotonic()}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Child()
""".lstrip(),
        encoding="utf-8",
    )
    (parent_root / "build_solver.py").write_text(
        """
from concurrent.futures import ThreadPoolExecutor
from blackbase.project import CaseRunRequest


class Parent:
    def _invoke(self):
        return self.case_runtime.invoke(
            CaseRunRequest(
                project_name="grant_project",
                stage_name="nested",
                case_name="child",
                resource_request={"workers": 1, "threads": 2},
            )
        )

    def run(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = [future.result() for future in (pool.submit(self._invoke), pool.submit(self._invoke))]
        return {"children": [result.as_dict() for result in results]}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Parent()
""".lstrip(),
        encoding="utf-8",
    )
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "grant_project"
L0 = {
    "namespace": "grant_project",
    "offer": {"threads": 2, "gpus": 0, "backend": "local"},
    "policy": {"mode": "strict", "max_workers": 1, "max_threads": 2},
    "default_request": {"workers": 1, "threads": 2, "gpus": 0},
}
STAGES = [{"name": "outer", "cases": ["parent"]}]
GROUPS = {"default": {"stages": ["outer"]}}
""".lstrip(),
        encoding="utf-8",
    )

    project_result = execute_project(project_root, record=False)

    assert project_result.ok
    children = [
        CaseRunResult.from_dict(payload)
        for payload in project_result.case_results[0].output["children"]
    ]
    assert all(child.ok for child in children)
    windows = sorted(
        (child.output["started_at"], child.output["finished_at"])
        for child in children
    )
    assert windows[1][0] >= windows[0][1]


def test_child_resource_grant_accounts_all_resources_and_rejects_widening(tmp_path) -> None:
    project_root = create_project(tmp_path / "bounded_grant_project", framework="blackbase")
    child_root = add_case("child", "solver", project_root=project_root)
    parent_root = add_case("parent", "solver", project_root=project_root)
    (child_root / "build_solver.py").write_text(
        """
class Child:
    def __init__(self, resource_context):
        self.resource_context = resource_context

    def run(self):
        return {"resource_context": self.resource_context}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, component_overrides
    return Child(resource_context)
""".lstrip(),
        encoding="utf-8",
    )
    (parent_root / "build_solver.py").write_text(
        """
from blackbase.project import CaseRunRequest


class Parent:
    def invoke(self, resource_request):
        return self.case_runtime.invoke(
            CaseRunRequest(
                project_name="bounded_grant_project",
                stage_name="nested",
                case_name="child",
                resource_request=resource_request,
            )
        ).as_dict()

    def run(self):
        return {
            "valid": self.invoke({
                "workers": 1,
                "threads": 1,
                "memory_mb": 256,
                "backend": "local",
                "compute_backend": "auto",
                "device": "cpu",
                "capabilities": ["base"],
            }),
            "too_many_workers": self.invoke({"workers": 2, "threads": 1}),
            "too_much_memory": self.invoke({"workers": 1, "threads": 1, "memory_mb": 2048}),
            "wrong_backend": self.invoke({"workers": 1, "threads": 1, "backend": "ray"}),
            "cuda_from_cpu": self.invoke({
                "workers": 1,
                "threads": 1,
                "compute_backend": "cuda",
                "device": "cuda:0",
            }),
            "missing_capability": self.invoke({
                "workers": 1,
                "threads": 1,
                "capabilities": ["not-granted"],
            }),
        }


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Parent()
""".lstrip(),
        encoding="utf-8",
    )
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "bounded_grant_project"
L0 = {
    "namespace": "bounded_grant_project",
    "offer": {
        "threads": 2,
        "gpus": 0,
        "backend": "local",
        "metadata": {"memory_mb": 1024},
    },
    "policy": {
        "mode": "strict",
        "max_workers": 1,
        "max_threads": 2,
        "max_memory_mb": 1024,
    },
    "default_request": {
        "workers": 1,
        "threads": 2,
        "gpus": 0,
        "memory_mb": 1024,
        "backend": "local",
        "compute_backend": "auto",
        "device": "cpu",
        "capabilities": ["base"],
    },
}
STAGES = [{"name": "outer", "cases": ["parent"]}]
GROUPS = {"default": {"stages": ["outer"]}}
""".lstrip(),
        encoding="utf-8",
    )

    project_result = execute_project(project_root, record=False)

    assert project_result.ok
    output = project_result.case_results[0].output
    valid = CaseRunResult.from_dict(output["valid"])
    assert valid.ok
    resources = valid.request.child_grant.resources
    assert resources["workers"] == 1
    assert resources["threads"] == 1
    assert resources["memory_mb"] == 256.0
    assert resources["capabilities"] == ("base",)
    for name in (
        "too_many_workers",
        "too_much_memory",
        "wrong_backend",
        "cuda_from_cpu",
        "missing_capability",
    ):
        rejected = CaseRunResult.from_dict(output[name])
        assert rejected.status == "failed"
        assert rejected.failure is not None
        assert rejected.failure.kind == "ProjectConfigurationError"


def test_case_executor_observes_durable_cancellation_and_deadline(tmp_path) -> None:
    project_root = create_project(tmp_path / "control_project", framework="blackbase")
    case_root = add_case("controlled", "solver", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        """
class Controlled:
    def run(self):
        return {"must_not_run": True}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Controlled()
""".lstrip(),
        encoding="utf-8",
    )
    executor = CaseExecutor(project_root)
    control_path = Path(project_root) / ".blackbase" / "controls.sqlite"

    cancelled_ref = CancellationRef(backend="sqlite", path=str(control_path))
    CancellationToken(cancelled_ref).cancel("test cancellation")
    cancelled = executor.execute(
        CaseRunRequest(
            project_name="control_project",
            stage_name="control",
            case_name="controlled",
            resource_request=ResourceRequest().as_dict(),
            control=ExecutionControl(cancellation=cancelled_ref),
        )
    )
    assert cancelled.status == "cancelled"
    assert cancelled.failure is not None
    assert cancelled.failure.message == "test cancellation"

    deadline_ref = CancellationRef(
        backend="sqlite",
        path=str(control_path),
        deadline_at=time.time() - 1.0,
    )
    timed_out = executor.execute(
        CaseRunRequest(
            project_name="control_project",
            stage_name="control",
            case_name="controlled",
            resource_request=ResourceRequest().as_dict(),
            control=ExecutionControl(cancellation=deadline_ref),
        )
    )
    assert timed_out.status == "timed_out"
    assert timed_out.failure is not None
    assert timed_out.failure.kind == "CaseDeadlineExceeded"


def _build_cleanup_case(tmp_path):
    project_root = create_project(tmp_path / "cleanup_project", framework="blackbase")
    case_root = add_case("clean", "solver", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        """
class Clean:
    def run(self):
        return {"value": 42}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Clean()
""".lstrip(),
        encoding="utf-8",
    )
    request = CaseRunRequest(
        project_name="cleanup_project",
        stage_name="cleanup",
        case_name="clean",
        resource_context={
            "threads": 1,
            "grant": {"workers": 1, "threads": 1},
            "metadata": {
                "artifact_authority": {
                    "backend": "filesystem",
                    "root": str(project_root / ".blackbase" / "artifacts"),
                    "namespace": "cleanup_project",
                    "schema_version": 1,
                }
            },
        },
    )
    return project_root, case_root, request


def test_case_result_freezes_runtime_audit_after_scheduler_shutdown(tmp_path) -> None:
    project_root, _case_root, request = _build_cleanup_case(tmp_path)

    result = CaseExecutor(project_root).execute(request)

    assert result.ok
    assert result.metadata["runtime_audit"]["stage_scheduler"]["shutdown"] is True
    assert result.metadata["cleanup"]["status"] == "succeeded"
    assert result.metadata["cleanup"]["scheduler"]["shutdown"] is True
    assert result.finished_at >= result.metadata["cleanup"]["finished_at"]


def test_scheduler_cleanup_failure_is_structured_and_does_not_escape(
    tmp_path,
    monkeypatch,
) -> None:
    project_root, _case_root, request = _build_cleanup_case(tmp_path)

    def fail_shutdown(self, wait=True):
        del self, wait
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(PoolScheduler, "shutdown", fail_shutdown)
    result = CaseExecutor(project_root).execute(request)

    assert result.status == "failed"
    assert result.exit_code == 1
    assert result.failure is not None
    assert result.failure.phase == "cleanup"
    assert result.failure.message == "cleanup failed"
    assert result.metadata["cleanup"]["status"] == "failed"
    assert result.metadata["cleanup"]["failure"]["phase"] == "cleanup"


def test_cleanup_failure_aborts_provisional_finalization_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    project_root, case_root, request = _build_cleanup_case(tmp_path)
    (case_root / "build_solver.py").write_text(
        """
class Publisher:
    def run(self):
        transaction = self.case_runtime.begin_finalization_transaction("cleanup")
        ref = transaction.publish("model", {"value": 42}, kind="model")
        transaction.prepare()
        return {"staged_uri": ref.uri}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Publisher()
""".lstrip(),
        encoding="utf-8",
    )

    def fail_shutdown(self, wait=True):
        del self, wait
        raise RuntimeError("cleanup failed after prepare")

    monkeypatch.setattr(PoolScheduler, "shutdown", fail_shutdown)
    result = CaseExecutor(project_root).execute(request)

    assert result.status == "failed"
    assert result.artifact_publications == {}
    assert result.artifact_refs == {}
    artifact_payloads = tuple(
        path
        for path in (project_root / ".blackbase" / "artifacts").rglob("*.json")
        if ".publication-ledger" not in path.parts
    )
    assert artifact_payloads == ()


def test_cleanup_failure_preserves_primary_case_failure(tmp_path, monkeypatch) -> None:
    project_root, case_root, request = _build_cleanup_case(tmp_path)
    (case_root / "build_solver.py").write_text(
        """
class Broken:
    def run(self):
        raise ValueError("primary failed")


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Broken()
""".lstrip(),
        encoding="utf-8",
    )

    def fail_shutdown(self, wait=True):
        del self, wait
        raise RuntimeError("cleanup also failed")

    monkeypatch.setattr(PoolScheduler, "shutdown", fail_shutdown)
    result = CaseExecutor(project_root).execute(request)

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.kind == "ValueError"
    assert result.failure.phase == "run"
    assert result.failure.message == "primary failed"
    assert result.metadata["cleanup"]["failure"]["message"] == "cleanup also failed"


def test_case_result_envelope_round_trips_strict_schema() -> None:
    request = CaseRunRequest(
        project_name="roundtrip",
        stage_name="stage",
        case_name="case",
        resource_request=ResourceRequest(threads=2).as_dict(),
        inputs={"value": [1, 2, 3]},
    )
    result = CaseRunResult(
        request=request,
        status="succeeded",
        output={"value": 6},
        started_at=1.0,
        finished_at=2.5,
    )

    restored = CaseRunResult.from_dict(result.as_dict())

    assert restored == result
    assert restored.elapsed_seconds == 1.5


def test_failed_child_result_raises_structured_invocation_error() -> None:
    request = CaseRunRequest(
        project_name="nested",
        stage_name="inner",
        case_name="child",
    )
    child_failure = CaseFailure(
        kind="EvaluationError",
        message="objective failed",
        phase="evaluate",
        retryable=True,
        details={"evaluation_id": "eval-7"},
    )
    result = CaseRunResult(
        request=request,
        status="failed",
        exit_code=1,
        failure=child_failure,
    )

    with pytest.raises(CaseInvocationError) as caught:
        result.raise_for_failure("inner optimization failed")

    assert caught.value.result is result
    parent_failure = CaseFailure.from_exception(
        caught.value,
        phase="evaluate",
        details={"candidate_id": "candidate-3"},
    )
    assert parent_failure.kind == "CaseInvocationError"
    assert parent_failure.retryable is True
    assert parent_failure.cause == child_failure.as_dict()
    assert parent_failure.details["candidate_id"] == "candidate-3"
    child_details = parent_failure.details["child_case"]
    assert child_details["identity"] == result.identity.as_dict()
    assert child_details["case_name"] == "child"
    assert child_details["status"] == "failed"


def test_formal_exception_evidence_survives_case_failure_envelope() -> None:
    error = RuntimeError("evaluation failed")
    attach_failure_evidence(
        error,
        "evaluation",
        {"event_id": "event-7", "phase": "provider"},
    )

    failure = CaseFailure.from_exception(error, phase="evaluate")

    assert failure.details["evaluation"] == {
        "event_id": "event-7",
        "phase": "provider",
    }


def test_cancellation_heartbeat_close_rejects_a_live_worker_thread() -> None:
    class _StuckThread:
        def is_alive(self):
            return True

        def join(self, timeout):
            assert timeout >= 0.1

    token = CancellationToken(CancellationRef(active_ttl_seconds=0.0))
    heartbeat = CancellationHeartbeat(token)
    heartbeat._thread = _StuckThread()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="thread did not stop"):
        heartbeat.close()


def test_successful_child_result_raise_for_failure_is_a_noop() -> None:
    result = CaseRunResult(
        request=CaseRunRequest(
            project_name="nested",
            stage_name="inner",
            case_name="child",
        ),
        status="succeeded",
        output={"value": 3},
    )

    assert result.raise_for_failure() is result


def test_reusing_child_request_mints_distinct_run_and_budget_namespaces(tmp_path) -> None:
    project_root = create_project(tmp_path / "repeated_request", framework="blackbase")
    child_root = add_case("child", "solver", project_root=project_root)
    parent_root = add_case("parent", "solver", project_root=project_root)
    (child_root / "build_solver.py").write_text(
        """
class Child:
    def run(self):
        handle = self.case_runtime.request.budget_handles["evaluations"]
        return {
            "case_run_id": self.case_runtime.identity.case_run_id,
            "invocation_id": self.case_runtime.identity.invocation_id,
            "budget_namespace": handle.authority_budget,
        }


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Child()
""".lstrip(),
        encoding="utf-8",
    )
    (parent_root / "build_solver.py").write_text(
        """
from blackbase.project import CaseRunRequest


class Parent:
    def run(self):
        request = CaseRunRequest(
            project_name="repeated_request",
            stage_name="nested",
            case_name="child",
            resource_request={"workers": 1, "threads": 1},
            budget_request={"evaluations": 1},
        )
        first = self.case_runtime.invoke(request)
        second = self.case_runtime.invoke(request)
        return {"first": first.as_dict(), "second": second.as_dict()}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Parent()
""".lstrip(),
        encoding="utf-8",
    )
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "repeated_request"
L0 = {
    "namespace": "repeated_request",
    "offer": {"threads": 1, "gpus": 0, "backend": "local"},
    "policy": {"mode": "strict", "max_workers": 1, "max_threads": 1},
    "default_request": {"workers": 1, "threads": 1},
    "lease_backend": "sqlite",
    "lease_path": ".blackbase/l0.sqlite",
    "budgets": {"evaluations": 2},
}
STAGES = [{"name": "outer", "cases": ["parent"]}]
GROUPS = {"default": {"stages": ["outer"]}}
""".lstrip(),
        encoding="utf-8",
    )

    project_result = execute_project(project_root, record=False)

    assert project_result.ok
    first = CaseRunResult.from_dict(project_result.case_results[0].output["first"])
    second = CaseRunResult.from_dict(project_result.case_results[0].output["second"])
    assert first.identity.case_run_id != second.identity.case_run_id
    assert first.identity.invocation_id != second.identity.invocation_id
    assert first.output["budget_namespace"] != second.output["budget_namespace"]


def test_child_failure_keeps_primary_cause_when_budget_settlement_also_fails(
    tmp_path,
    monkeypatch,
) -> None:
    parent = CaseRunRequest(
        project_name="project",
        stage_name="outer",
        case_name="parent",
        resource_context={"threads": 1, "grant": {"workers": 1, "threads": 1}},
    )
    executor = CaseExecutor(tmp_path)
    invoker = CaseInvoker(
        executor,
        parent,
        cancellation_tokens=(CancellationToken(parent.control.cancellation),),
    )
    grant = ChildResourceGrant(
        grant_id="grant-1",
        parent_lease_id="lease-1",
        parent_case_run_id=parent.identity.case_run_id,
        namespace="project",
        resources={"workers": 1, "threads": 1},
    )

    class _GrantPool:
        @contextmanager
        def acquire(self, *args, **kwargs):
            del args, kwargs
            yield grant

    calls = {"settlement": 0}
    invoker._grants = _GrantPool()
    delegation = SimpleNamespace(handle=BudgetHandle.local("evaluations", 1))
    monkeypatch.setattr(
        invoker,
        "_delegate_budgets",
        lambda *args: (delegation,),
    )
    monkeypatch.setattr(invoker, "_child_resource_context", lambda *args, **kwargs: {})

    def settle(_delegations):
        calls["settlement"] += 1
        raise ConnectionError("settlement authority unavailable")

    monkeypatch.setattr(invoker, "_finalize_budgets", settle)
    monkeypatch.setattr(
        executor,
        "execute",
        lambda request: (_ for _ in ()).throw(RuntimeError("child exploded")),
    )

    result = invoker.invoke(
        CaseRunRequest(
            project_name="project",
            stage_name="inner",
            case_name="child",
            resource_request={"workers": 1, "threads": 1},
            budget_request={"evaluations": 1},
        )
    )

    assert not result.ok
    assert result.failure is not None
    assert result.failure.kind == "RuntimeError"
    assert result.failure.message == "child exploded"
    assert result.failure.details["budget_settlement"] == {
        "error_type": "ConnectionError",
        "message": "settlement authority unavailable",
    }
    assert calls["settlement"] == 1
