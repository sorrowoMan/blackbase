from __future__ import annotations

import time
from pathlib import Path

from blackbase.project import (
    CaseExecutor,
    CaseRunRequest,
    CaseRunResult,
    ExecutionControl,
    execute_project,
)
from blackbase.project.scaffold import add_case, create_project
from blackbase.resources import CancellationRef, CancellationToken, ResourceRequest


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
        return {
            "value": self.case_inputs["value"] * 2,
            "identity": self.case_runtime.identity.as_dict(),
            "control": self.case_runtime.control.as_dict(),
            "threads": self.resource_context["threads"],
            "grant": self.resource_context["metadata"]["child_grant"],
            "artifact_refs": {
                "answer": {"uri": "memory://nested/answer", "kind": "result"},
            },
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
            "artifact_refs": {"answer": child.artifact_refs["answer"].as_dict()},
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
    assert child_result.control.ancestor_cancellations[-1].control_id == (
        parent_result.request.control.cancellation.control_id
    )
    assert child_result.output["threads"] == 1
    assert child_result.output["grant"]["parent_case_run_id"] == (
        parent_result.identity.case_run_id
    )
    assert child_result.budget_usage["evaluations"]["charged_to_parent"] == 2
    assert child_result.budget_usage["evaluations"]["returned_to_parent"] == 1
    assert child_result.artifact_refs["answer"].uri == "memory://nested/answer"
    assert project_result.artifact_registry["parent.answer"].uri == (
        "memory://nested/answer"
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
    assert resources["capabilities"] == ["base"]
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
