from __future__ import annotations

from pathlib import Path

import pytest

from blackbase.project import (
    DagStagePlan,
    ProjectConfigurationError,
    add_case,
    create_project,
    execute_project,
    run_common_project_doctor,
)


def _write_config(project_root: Path, source: str) -> None:
    (project_root / "project_config.py").write_text(source.lstrip(), encoding="utf-8")


def test_dag_plan_infers_artifact_edges_and_rejects_cycles() -> None:
    plan = DagStagePlan.from_stage(
        {
            "name": "workflow",
            "policy": "dag",
            "cases": ["consumer", "gate", "producer_a", "producer_b"],
            "depends_on": {"gate": ["consumer"]},
            "input_artifacts": {
                "consumer": {
                    "a": "producer_a.result",
                    "b": "workflow.producer_b.result",
                }
            },
        }
    )

    assert plan.dependencies_for("consumer") == ("producer_a", "producer_b")
    assert plan.dependencies_for("gate") == ("consumer",)
    assert plan.topological_order == (
        "producer_a",
        "producer_b",
        "consumer",
        "gate",
    )

    with pytest.raises(ProjectConfigurationError, match="dependency cycle"):
        DagStagePlan.from_stage(
            {
                "name": "cycle",
                "policy": "dag",
                "cases": ["a", "b"],
                "depends_on": {"a": ["b"], "b": ["a"]},
            }
        )


def test_project_dag_runs_ready_cases_in_parallel_and_wakes_consumers(tmp_path) -> None:
    project_root = create_project(tmp_path / "dag_project", framework="blackbase")
    producer_source = """
import os
import time

class Case:
    def __init__(self, resource_context=None, component_overrides=None):
        self.resource_context = resource_context
        self.component_overrides = dict(component_overrides or {})
        self.case_runtime = None

    def set_case_runtime(self, runtime):
        self.case_runtime = runtime

    def run(self):
        started_at = time.time()
        time.sleep(float(self.component_overrides.get("delay", 0.2)))
        ref = self.case_runtime.publish_artifact(
            "result",
            {"producer": self.component_overrides["name"]},
            kind="dag-result",
        )
        return {
            "pid": os.getpid(),
            "started_at": started_at,
            "finished_at": time.time(),
            "artifact_refs": {"result": ref},
        }

def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config
    return Case(resource_context, component_overrides)
""".lstrip()
    for name in ("producer_a", "producer_b"):
        root = add_case(name, "solver", project_root=project_root)
        (root / "build_solver.py").write_text(producer_source, encoding="utf-8")

    consumer_root = add_case("consumer", "trainer", project_root=project_root)
    (consumer_root / "build_solver.py").write_text(
        """
import time

class Case:
    def __init__(self, resource_context=None, component_overrides=None):
        del resource_context, component_overrides
        self.input_artifacts = {}

    def set_input_artifacts(self, refs):
        self.input_artifacts = dict(refs)

    def fit(self):
        return {
            "started_at": time.time(),
            "producer_uris": sorted(ref.uri for ref in self.input_artifacts.values()),
            "finished_at": time.time(),
        }

def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config
    return Case(resource_context, component_overrides)
""".lstrip(),
        encoding="utf-8",
    )
    gate_root = add_case("gate", "solver", project_root=project_root)
    (gate_root / "build_solver.py").write_text(
        """
import time

class Case:
    def __init__(self, resource_context=None, component_overrides=None):
        del resource_context, component_overrides

    def run(self):
        now = time.time()
        return {"started_at": now, "finished_at": now}

def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config
    return Case(resource_context, component_overrides)
""".lstrip(),
        encoding="utf-8",
    )
    _write_config(
        project_root,
        """
PROJECT_NAME = "dag_project"
L0 = {
    "namespace": "dag_project",
    "offer": {"threads": 2, "gpus": 0, "backend": "local"},
    "policy": {"mode": "strict", "max_workers": 2, "max_threads": 2},
    "default_request": {"workers": 1, "threads": 1, "gpus": 0, "backend": "local"},
}
STAGES = [{
    "name": "workflow",
    "policy": "dag",
    "max_workers": 2,
    "cases": ["consumer", "gate", "producer_a", "producer_b"],
    "depends_on": {"gate": ["consumer"]},
    "component_overrides": {
        "producer_a": {"name": "a", "delay": 0.25},
        "producer_b": {"name": "b", "delay": 0.25},
    },
    "input_artifacts": {
        "consumer": {
            "a": "producer_a.result",
            "b": "producer_b.result",
        },
    },
}]
GROUPS = {"default": {"stages": ["workflow"]}}
""",
    )

    result = execute_project(project_root, record=False, run_id="dag-e2e")

    assert result.ok
    by_name = {item.request.case_name: item for item in result.case_results}
    producer_a = by_name["producer_a"].output
    producer_b = by_name["producer_b"].output
    consumer = by_name["consumer"].output
    gate = by_name["gate"].output
    assert max(producer_a["started_at"], producer_b["started_at"]) < min(
        producer_a["finished_at"], producer_b["finished_at"]
    )
    assert consumer["started_at"] >= max(
        producer_a["finished_at"], producer_b["finished_at"]
    )
    assert gate["started_at"] >= consumer["finished_at"]
    assert len(consumer["producer_uris"]) == 2
    assert by_name["consumer"].request.metadata["dag"]["dependencies"] == (
        "producer_a",
        "producer_b",
    )
    assert by_name["gate"].request.metadata["dag"]["explicit_dependencies"] == (
        "consumer",
    )


def test_project_dag_continue_policy_skips_only_blocked_descendants(tmp_path) -> None:
    project_root = create_project(tmp_path / "dag_failure", framework="blackbase")
    source = """
class Case:
    def __init__(self, resource_context=None, component_overrides=None):
        del resource_context
        self.fail = bool(dict(component_overrides or {}).get("fail", False))

    def run(self):
        if self.fail:
            raise RuntimeError("intentional DAG failure")
        return {"ok": True}

def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config
    return Case(resource_context, component_overrides)
""".lstrip()
    for name in ("failed_root", "independent", "blocked", "transitive"):
        root = add_case(name, "solver", project_root=project_root)
        (root / "build_solver.py").write_text(source, encoding="utf-8")
    _write_config(
        project_root,
        """
PROJECT_NAME = "dag_failure"
L0 = {
    "namespace": "dag_failure",
    "offer": {"threads": 2, "gpus": 0, "backend": "local"},
    "policy": {"mode": "strict", "max_workers": 2, "max_threads": 2},
    "default_request": {"workers": 1, "threads": 1, "gpus": 0, "backend": "local"},
}
STAGES = [{
    "name": "workflow",
    "policy": "dag",
    "failure_policy": "continue",
    "cases": ["failed_root", "independent", "blocked", "transitive"],
    "depends_on": {
        "blocked": ["failed_root"],
        "transitive": ["blocked"],
    },
    "component_overrides": {"failed_root": {"fail": True}},
}]
GROUPS = {"default": {"stages": ["workflow"]}}
""",
    )

    result = execute_project(project_root, record=False, run_id="dag-failure")

    assert not result.ok
    by_name = {item.request.case_name: item for item in result.case_results}
    assert by_name["failed_root"].status == "failed"
    assert by_name["independent"].status == "succeeded"
    assert by_name["blocked"].status == "skipped"
    assert by_name["blocked"].failure.kind == "DependencyFailed"
    assert by_name["transitive"].status == "skipped"
    assert by_name["transitive"].failure.kind == "DependencyFailed"


def test_project_doctor_validates_dag_before_execution(tmp_path) -> None:
    project_root = create_project(tmp_path / "dag_doctor", framework="blackbase")
    for name in ("a", "b"):
        add_case(name, "solver", project_root=project_root)
    _write_config(
        project_root,
        """
PROJECT_NAME = "dag_doctor"
STAGES = [{
    "name": "workflow",
    "policy": "dag",
    "cases": ["a", "b"],
    "depends_on": {"b": ["a"]},
}]
GROUPS = {"default": {"stages": ["workflow"]}}
""",
    )

    valid = run_common_project_doctor(project_root, strict=True)
    assert not any(item.code == "project-dag-invalid" for item in valid.diagnostics)
    assert any(item.code == "project-dag-valid" for item in valid.diagnostics)

    _write_config(
        project_root,
        """
PROJECT_NAME = "dag_doctor"
STAGES = [{
    "name": "workflow",
    "policy": "dag",
    "cases": ["a", "b"],
    "depends_on": {"a": ["b"], "b": ["a"]},
}]
GROUPS = {"default": {"stages": ["workflow"]}}
""",
    )
    invalid = run_common_project_doctor(project_root, strict=True)
    dag_error = next(
        item for item in invalid.diagnostics if item.code == "project-dag-invalid"
    )
    assert dag_error.level == "error"
    assert "dependency cycle" in dag_error.message


def test_project_dag_resume_reuses_verified_upstream_artifact(tmp_path) -> None:
    project_root = create_project(tmp_path / "dag_resume", framework="blackbase")
    producer_root = add_case("producer", "solver", project_root=project_root)
    consumer_root = add_case("consumer", "trainer", project_root=project_root)
    (producer_root / "build_solver.py").write_text(
        """
from pathlib import Path

class Case:
    def __init__(self):
        self.case_runtime = None

    def set_case_runtime(self, runtime):
        self.case_runtime = runtime

    def run(self):
        root = Path(__file__).resolve().parents[2]
        counter = root / "producer.count"
        count = int(counter.read_text()) + 1 if counter.exists() else 1
        counter.write_text(str(count))
        ref = self.case_runtime.publish_artifact("model", {"count": count})
        return {"count": count, "artifact_refs": {"model": ref}}

def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Case()
""".lstrip(),
        encoding="utf-8",
    )
    (consumer_root / "build_solver.py").write_text(
        """
from pathlib import Path

class Case:
    def __init__(self):
        self.input_artifacts = {}

    def set_input_artifacts(self, refs):
        self.input_artifacts = dict(refs)

    def fit(self):
        root = Path(__file__).resolve().parents[2]
        counter = root / "consumer.count"
        count = int(counter.read_text()) + 1 if counter.exists() else 1
        counter.write_text(str(count))
        if count == 1:
            raise RuntimeError("fail once")
        return {"count": count, "model_uri": self.input_artifacts["model"].uri}

def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Case()
""".lstrip(),
        encoding="utf-8",
    )
    _write_config(
        project_root,
        """
PROJECT_NAME = "dag_resume"
L0 = {
    "namespace": "dag_resume",
    "offer": {"threads": 1, "gpus": 0, "backend": "local"},
    "policy": {"mode": "strict", "max_workers": 1, "max_threads": 1},
    "default_request": {"workers": 1, "threads": 1, "gpus": 0, "backend": "local"},
}
STAGES = [{
    "name": "workflow",
    "policy": "dag",
    "cases": ["producer", "consumer"],
    "input_artifacts": {"consumer": {"model": "producer.model"}},
}]
GROUPS = {"default": {"stages": ["workflow"]}}
""",
    )

    first = execute_project(project_root, record=True, run_id="dag-first")
    assert not first.ok
    assert [item.status for item in first.case_results] == ["succeeded", "failed"]
    model_uri = first.artifact_registry["producer.model"].uri

    resumed = execute_project(
        project_root,
        record=True,
        run_id="dag-second",
        resume_from=first.manifest_path,
    )

    assert resumed.ok
    assert [item.status for item in resumed.case_results] == ["resumed", "succeeded"]
    assert resumed.case_results[1].output == {"count": 2, "model_uri": model_uri}
    assert (project_root / "producer.count").read_text() == "1"
    assert (project_root / "consumer.count").read_text() == "2"
