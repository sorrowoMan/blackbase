from __future__ import annotations

import time

from blackbase.project import CaseExecutor, CaseRunRequest, CaseRunResult, ExecutionControl
from blackbase.project import execute_project
from blackbase.project.scaffold import add_case, create_project
from blackbase.resources import TerminationPolicy


def _hard_policy() -> TerminationPolicy:
    return TerminationPolicy(
        mode="cooperative_then_terminate",
        grace_seconds=0.05,
        kill_grace_seconds=0.2,
        poll_interval_seconds=0.01,
    )


def test_isolated_case_is_terminated_inside_uninterruptible_call(tmp_path) -> None:
    project_root = create_project(tmp_path / "hard_stop_project", framework="blackbase")
    case_root = add_case("blocking", "solver", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        """
import time


class Blocking:
    def run(self):
        time.sleep(10.0)
        return {"unreachable": True}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Blocking()
""".lstrip(),
        encoding="utf-8",
    )
    control_path = project_root / ".blackbase" / "controls.sqlite"
    started = time.monotonic()

    result = CaseExecutor(project_root).execute(
        CaseRunRequest(
            project_name="hard_stop_project",
            stage_name="hard_stop",
            case_name="blocking",
            control=ExecutionControl.with_timeout(
                0.15,
                backend="sqlite",
                path=str(control_path),
                termination=_hard_policy(),
            ),
        )
    )

    assert time.monotonic() - started < 4.0
    assert result.status == "timed_out"
    assert result.failure is not None
    assert result.failure.kind == "CaseDeadlineExceeded"
    assert result.failure.phase == "terminate"
    assert result.failure.details["terminated"] is True


def test_parent_case_can_hard_terminate_complete_blocking_child_case(tmp_path) -> None:
    project_root = create_project(tmp_path / "nested_hard_stop", framework="blackbase")
    child_root = add_case("blocking_child", "solver", project_root=project_root)
    parent_root = add_case("parent", "solver", project_root=project_root)
    (child_root / "build_solver.py").write_text(
        """
import time


class BlockingChild:
    def run(self):
        time.sleep(10.0)
        return {"unreachable": True}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return BlockingChild()
""".lstrip(),
        encoding="utf-8",
    )
    (parent_root / "build_solver.py").write_text(
        """
from blackbase.project import CaseRunRequest, ExecutionControl
from blackbase.resources import TerminationPolicy


class Parent:
    def run(self):
        child = self.case_runtime.invoke(
            CaseRunRequest(
                project_name="nested_hard_stop",
                stage_name="nested",
                case_name="blocking_child",
                resource_request={"workers": 1, "threads": 1},
                control=ExecutionControl.with_timeout(
                    0.15,
                    termination=TerminationPolicy(
                        mode="cooperative_then_terminate",
                        grace_seconds=0.05,
                        kill_grace_seconds=0.2,
                        poll_interval_seconds=0.01,
                    ),
                ),
            )
        )
        return {"child": child.as_dict()}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Parent()
""".lstrip(),
        encoding="utf-8",
    )
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "nested_hard_stop"
L0 = {
    "offer": {"threads": 1, "gpus": 0, "backend": "local"},
    "policy": {"mode": "strict", "max_workers": 1, "max_threads": 1},
    "default_request": {"workers": 1, "threads": 1},
}
STAGES = [{"name": "outer", "cases": ["parent"]}]
GROUPS = {"default": {"stages": ["outer"]}}
""".lstrip(),
        encoding="utf-8",
    )
    started = time.monotonic()

    project_result = execute_project(project_root, record=False)

    assert time.monotonic() - started < 4.0
    assert project_result.ok
    child_result = CaseRunResult.from_dict(project_result.case_results[0].output["child"])
    assert child_result.status == "timed_out"
    assert child_result.failure is not None
    assert child_result.failure.phase == "terminate"
    assert child_result.identity.parent_case_run_id == (
        project_result.case_results[0].identity.case_run_id
    )


def test_parallel_process_backend_applies_per_case_termination_policy(tmp_path) -> None:
    project_root = create_project(tmp_path / "parallel_hard_stop", framework="blackbase")
    blocking_root = add_case("blocking", "solver", project_root=project_root)
    fast_root = add_case("fast", "solver", project_root=project_root)
    (blocking_root / "build_solver.py").write_text(
        """
import time


class Blocking:
    def run(self):
        time.sleep(10.0)
        return {"unreachable": True}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Blocking()
""".lstrip(),
        encoding="utf-8",
    )
    (fast_root / "build_solver.py").write_text(
        """
class Fast:
    def run(self):
        return {"value": 42}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Fast()
""".lstrip(),
        encoding="utf-8",
    )
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "parallel_hard_stop"
L0 = {
    "offer": {"threads": 2, "gpus": 0, "backend": "local"},
    "policy": {"mode": "strict", "max_workers": 2, "max_threads": 2},
    "default_request": {"workers": 1, "threads": 1},
}
STAGES = [{
    "name": "parallel",
    "policy": "parallel",
    "failure_policy": "run_all",
    "cases": ["blocking", "fast"],
    "case_timeout_seconds": {"blocking": 1.5},
    "case_termination": {
        "blocking": {
            "mode": "cooperative_then_terminate",
            "grace_seconds": 0.05,
            "kill_grace_seconds": 0.2,
            "poll_interval_seconds": 0.01,
        },
    },
}]
GROUPS = {"default": {"stages": ["parallel"]}}
""".lstrip(),
        encoding="utf-8",
    )
    started = time.monotonic()

    project_result = execute_project(project_root, record=False)

    assert time.monotonic() - started < 6.0
    by_name = {result.request.case_name: result for result in project_result.case_results}
    assert by_name["blocking"].status == "timed_out"
    assert by_name["blocking"].failure is not None
    assert by_name["blocking"].failure.phase == "terminate"
    assert by_name["fast"].ok
    assert by_name["fast"].output["value"] == 42
