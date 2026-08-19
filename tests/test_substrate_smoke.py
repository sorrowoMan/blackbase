from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from blackbase.context import ContextStore, create_snapshot_store
from blackbase.kernel import build_pipeline_kernel
from blackbase.project.doctor import run_common_project_doctor
from blackbase.project.execution import ProjectConfigurationError
from blackbase.project.project_runner import execute_project, run_project
from blackbase.project.runtime import case_import_context, run_case
from blackbase.project.scaffold import add_case, create_project
from blackbase.resources import (
    InMemoryResourceScheduler,
    PoolScheduler,
    ResourceBudgetError,
    ResourceContext,
    ResourceAllocator,
    ResourceOffer,
    ResourcePolicy,
    ResourceRequest,
    ResourceRequirement,
    TaskEnvelope,
    WorkerDescriptor,
)
from blackbase.types import Feedback, UnknownState


def test_run_case_applies_optional_semantic_result_exporter() -> None:
    class _Case:
        def run(self):
            return {"raw_value": 3}

        def export_case_result(self, raw_output):
            return {"projected_value": int(raw_output["raw_value"]) + 1}

    assert run_case(_Case(), case_kind="solver") == {"projected_value": 4}


def test_resource_context_child_keeps_parent_lease_and_clamps_threads() -> None:
    parent = ResourceContext.from_mapping(
        {
            "scope": "optimization",
            "threads": 4,
            "namespace": "project.case",
            "grant": {"threads": 4, "workers": 4, "device_tokens": ["cuda:0"]},
            "lease": {"lease_id": "lease-parent", "owner_id": "case"},
            "metadata": {"case_name": "outer"},
        }
    )

    child = parent.derive_child(
        scope="training",
        namespace_suffix="inner",
        threads=99,
        metadata={"bridge": "nsgablack->mlblack"},
    )

    assert child.nested is True
    assert child.threads == 4
    assert child.namespace == "project.case.inner"
    assert child.lease["lease_id"] == "lease-parent"
    assert child.metadata["parent_lease_id"] == "lease-parent"
    assert child.metadata["bridge"] == "nsgablack->mlblack"


def test_project_l0_allocator_enforces_aggregate_active_lease_budget() -> None:
    allocator = ResourceAllocator(
        offer=ResourceOffer(threads=4, gpus=0, backend="local"),
        policy=ResourcePolicy(max_workers=2, max_threads=4, max_gpus=0, mode="strict"),
    )
    first = allocator.acquire(ResourceRequest(workers=1, threads=3), owner_id="case_a")

    with pytest.raises(ResourceBudgetError, match="active lease threads over budget"):
        allocator.acquire(ResourceRequest(workers=1, threads=2), owner_id="case_b")

    allocator.release(first)
    second = allocator.acquire(ResourceRequest(workers=1, threads=2), owner_id="case_b")
    assert second.threads == 2


def test_pool_scheduler_submit_with_explicit_capacity_metadata() -> None:
    pool = PoolScheduler(total_threads=2)
    try:
        assert pool.submit(lambda x: x + 1, 41).result(timeout=5) == 42

        submitted = pool.submit(
            lambda x: x * 2,
            21,
            resource_permits=2,
            task_id="task_a",
        )
        assert submitted.task_id == "task_a"
        assert submitted.result(timeout=5) == 42
        assert pool.report()["tasks_completed"] == 2
    finally:
        pool.close()


def test_l0_scheduler_grants_and_releases_worker_resources() -> None:
    scheduler = InMemoryResourceScheduler(
        workers=(
            WorkerDescriptor(
                worker_id="gpu-worker",
                executor_backend="thread",
                capabilities=("nested_eval",),
                offer={"threads": 4, "device_tokens": ("cuda:0",), "memory_mb": 2048},
                max_inflight=1,
            ),
        )
    )
    task = TaskEnvelope(
        task_id="candidate-1",
        task_type="nested_candidate_eval",
        requirement=ResourceRequirement(
            threads=2,
            gpus=1,
            memory_mb=1024,
            capabilities=("nested_eval",),
        ),
        executor_backend="thread",
        namespace="case.inner",
    )

    scheduled = scheduler.acquire(task)
    assert scheduled.lease is not None
    assert scheduled.lease.threads == 2
    assert scheduled.lease.device_tokens == ("cuda:0",)
    assert scheduled.resource_context["metadata"]["task_id"] == "candidate-1"
    with pytest.raises(ResourceBudgetError):
        scheduler.acquire(task)

    scheduler.release(scheduled)
    assert scheduler.active_leases() == ()


def test_shared_protocol_types() -> None:
    state = UnknownState(values=[1, 2], metadata={"source": "test"})
    moved = state.with_values([3, 4], stage="mutate")

    assert state.metadata == {"source": "test"}
    assert moved.metadata == {"source": "test", "stage": "mutate"}
    assert moved.as_array().tolist() == [3.0, 4.0]

    feedback = Feedback(objectives=[1.0], constraints=[0.0])
    assert feedback.ok
    assert feedback.scalar_score() == 1.0


def test_context_and_snapshot_store() -> None:
    context = ContextStore()
    context.set("generation", 3)
    assert context.snapshot()["generation"] == 3

    snapshots = create_snapshot_store(backend="memory")
    handle = snapshots.write({"population": [[1.0, 2.0]]}, key="smoke")
    record = snapshots.read(handle.key)

    assert record is not None
    assert record.data["population"] == [[1.0, 2.0]]


def test_pipeline_kernel_calls_method_style_operator() -> None:
    class Problem:
        dimension = 3

    class Initializer:
        def initialize(self, problem, context=None):
            return np.ones(problem.dimension, dtype=float)

    class Repair:
        def repair(self, value, context=None):
            return np.clip(np.asarray(value, dtype=float), 0.0, 1.0)

    kernel = build_pipeline_kernel(
        {
            "key": "smoke",
            "slots": [
                {"slot": "initializer", "operators": ["init"]},
                {"slot": "repair", "operators": ["repair"]},
            ],
        },
        operator_registry={"init": Initializer(), "repair": Repair()},
    )

    candidate = kernel.run_slot("initializer", Problem())
    repaired = kernel.run_slot("repair", [-1.0, 0.5, 2.0])

    assert candidate.tolist() == [1.0, 1.0, 1.0]
    assert repaired.tolist() == [0.0, 0.5, 1.0]


def test_project_substrate_add_case_runner_and_doctor(tmp_path) -> None:
    project_root = create_project(tmp_path / "demo", framework="blackbase")

    add_case("search_case", "solver", project_root=project_root)
    add_case("fit_case", "trainer", project_root=project_root)

    solver_case = project_root / "cases" / "search_case"
    trainer_case = project_root / "cases" / "fit_case"

    assert (solver_case / "build_solver.py").is_file()
    assert (solver_case / "run_solver.py").is_file()
    assert (trainer_case / "build_solver.py").is_file()
    assert (trainer_case / "run_solver.py").is_file()
    expected_build_alias = "from .build_solver import build_solver as build_trainer\n"
    expected_run_alias = (
        'from .run_solver import main\n\n'
        'if __name__ == "__main__":\n'
        '    raise SystemExit(main())\n'
    )
    for case_root in (solver_case, trainer_case):
        assert (case_root / "build_trainer.py").read_text(encoding="utf-8") == expected_build_alias
        assert (case_root / "run_trainer.py").read_text(encoding="utf-8") == expected_run_alias
        run_source = (case_root / "run_solver.py").read_text(encoding="utf-8")
        assert "run_standard_case_cli" in run_source
        assert "from .build_solver" not in run_source
        pipeline_main = case_root / "pipeline" / "main.py"
        assert pipeline_main.is_file()
        assert "def build_pipeline" in pipeline_main.read_text(encoding="utf-8")

    assert '"key": "solver_default"' in (solver_case / "pipeline" / "main.py").read_text(encoding="utf-8")
    assert '"key": "trainer_default"' in (trainer_case / "pipeline" / "main.py").read_text(encoding="utf-8")

    (solver_case / "build_solver.py").write_text(
        """
class Case:
    def __init__(self, resource_context=None):
        self.resource_context = dict(resource_context or {})

    def run(self):
        return {"ok": True}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, component_overrides
    return Case(resource_context)
""".lstrip(),
        encoding="utf-8",
    )
    (trainer_case / "build_solver.py").write_text(
        """
class Case:
    def __init__(self, resource_context=None):
        self.resource_context = dict(resource_context or {})

    def fit(self):
        return {"ok": True}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, component_overrides
    return Case(resource_context)
""".lstrip(),
        encoding="utf-8",
    )
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "demo"
L0 = {
    "namespace": "demo",
    "offer": {"threads": 4, "gpus": 0, "backend": "local"},
    "policy": {"mode": "strict", "max_workers": 4, "max_threads": 4},
    "default_request": {"threads": 1, "gpus": 0, "backend": "local"},
}
STAGES = [{
    "name": "main",
    "cases": ["search_case", "fit_case"],
    "case_modes": {"fit_case": "cli"},
}]
GROUPS = {"default": {"stages": ["main"]}}
""".lstrip(),
        encoding="utf-8",
    )

    assert run_project(project_root, check=True, build_check=True) == 0

    report = run_common_project_doctor(project_root, strict=True)
    errors = [item for item in report.diagnostics if item.level == "error"]
    assert not errors

    (trainer_case / "build_trainer.py").write_text(
        "def build_trainer():\n    return None\n",
        encoding="utf-8",
    )
    broken = run_common_project_doctor(project_root, strict=True)
    assert any(item.code == "case-build-trainer-not-thin-alias" for item in broken.diagnostics)


def test_case_import_context_restores_short_name_modules(tmp_path) -> None:
    project_root = tmp_path / "isolated_project"
    pipeline_root = project_root / "cases" / "demo" / "pipeline"
    pipeline_root.mkdir(parents=True)
    (pipeline_root / "__init__.py").write_text("SOURCE = 'case'\n", encoding="utf-8")

    previous = {name: module for name, module in sys.modules.items() if name == "pipeline" or name.startswith("pipeline.")}
    sentinel = ModuleType("pipeline")
    sentinel.SOURCE = "caller"
    sys.modules["pipeline"] = sentinel
    try:
        with case_import_context(project_root, "demo"):
            imported = importlib.import_module("pipeline")
            assert imported is not sentinel
            assert imported.SOURCE == "case"

        assert sys.modules.get("pipeline") is sentinel
        assert not any(name.startswith("pipeline.") for name in sys.modules)
    finally:
        for name in list(sys.modules):
            if name == "pipeline" or name.startswith("pipeline."):
                sys.modules.pop(name, None)
        sys.modules.update(previous)


def test_execute_project_returns_results_and_passes_artifact_refs_and_overrides(tmp_path) -> None:
    project_root = create_project(tmp_path / "runtime_project", framework="blackbase")
    producer_root = add_case("producer", "solver", project_root=project_root)
    consumer_root = add_case("consumer", "trainer", project_root=project_root)

    (producer_root / "build_solver.py").write_text(
        """
class Case:
    def __init__(self, resource_context=None, component_overrides=None):
        self.resource_context = dict(resource_context or {})
        self.component_overrides = dict(component_overrides or {})

    def run(self):
        return {
            "entry": "run",
            "artifact_refs": {
                "model": {"uri": "memory://trained/model", "kind": "model"},
            },
        }


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config
    return Case(resource_context, component_overrides)
""".lstrip(),
        encoding="utf-8",
    )
    (consumer_root / "build_solver.py").write_text(
        """
class Case:
    def __init__(self, resource_context=None, component_overrides=None):
        self.resource_context = dict(resource_context or {})
        self.component_overrides = dict(component_overrides or {})
        self.input_artifacts = {}

    def set_input_artifacts(self, refs):
        self.input_artifacts = dict(refs)

    def run(self):
        return {"entry": "run"}

    def fit(self):
        return {
            "entry": "fit",
            "model_uri": self.input_artifacts["model"].uri,
            "overrides": self.component_overrides,
        }


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config
    return Case(resource_context, component_overrides)
""".lstrip(),
        encoding="utf-8",
    )
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "runtime_project"
L0 = {
    "namespace": "runtime_project",
    "offer": {"threads": 2, "gpus": 0, "backend": "local"},
    "policy": {"mode": "strict", "max_workers": 2, "max_threads": 2},
    "default_request": {"threads": 1, "gpus": 0, "backend": "local"},
}
STAGES = [{
    "name": "main",
    "policy": "serial",
    "cases": ["producer", "consumer"],
    "component_overrides": {
        "consumer": {"trainer": {"max_steps": 2}},
    },
    "input_artifacts": {
        "consumer": {"model": "producer.model"},
    },
}]
GROUPS = {"default": {"stages": ["main"]}}
""".lstrip(),
        encoding="utf-8",
    )

    result = execute_project(project_root)

    assert result.ok
    assert result.exit_code == 0
    assert [item.request.case_name for item in result.case_results] == ["producer", "consumer"]
    assert result.case_results[0].output["entry"] == "run"
    assert result.case_results[1].output == {
        "entry": "fit",
        "model_uri": "memory://trained/model",
        "overrides": {"trainer": {"max_steps": 2}},
    }
    assert result.artifact_registry["main.producer.model"].kind == "model"
    assert result.artifact_registry["producer.model"].uri == "memory://trained/model"
    assert run_project(project_root) == 0


def test_execute_project_runs_parallel_cases_in_isolated_processes(tmp_path) -> None:
    project_root = create_project(tmp_path / "parallel_project", framework="blackbase")
    case_source = """
import os
import time


class Case:
    def __init__(self, resource_context=None, component_overrides=None):
        self.resource_context = dict(resource_context or {})
        self.component_overrides = dict(component_overrides or {})

    def run(self):
        started_at = time.time()
        time.sleep(0.35)
        return {
            "pid": os.getpid(),
            "started_at": started_at,
            "finished_at": time.time(),
            "lease_id": self.resource_context["lease"]["lease_id"],
            "threads": self.resource_context["threads"],
            "artifact_refs": {
                "result": {"uri": "memory://parallel/" + str(os.getpid()), "kind": "result"},
            },
        }


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config
    return Case(resource_context, component_overrides)
""".lstrip()
    for case_name in ("case_a", "case_b"):
        case_root = add_case(case_name, "solver", project_root=project_root)
        (case_root / "build_solver.py").write_text(case_source, encoding="utf-8")
    consumer_root = add_case("consumer", "trainer", project_root=project_root)
    (consumer_root / "build_solver.py").write_text(
        """
class Case:
    def __init__(self, resource_context=None, component_overrides=None):
        del resource_context, component_overrides
        self.input_artifacts = {}

    def set_input_artifacts(self, refs):
        self.input_artifacts = dict(refs)

    def fit(self):
        return {"producer_uris": sorted(ref.uri for ref in self.input_artifacts.values())}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config
    return Case(resource_context, component_overrides)
""".lstrip(),
        encoding="utf-8",
    )
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "parallel_project"
L0 = {
    "namespace": "parallel_project",
    "offer": {"threads": 2, "gpus": 0, "backend": "local"},
    "policy": {"mode": "strict", "max_workers": 2, "max_threads": 2},
    "default_request": {"workers": 1, "threads": 1, "gpus": 0, "backend": "local"},
}
STAGES = [
    {
        "name": "parallel",
        "policy": "run_all_in_parallel",
        "cases": ["case_a", "case_b"],
    },
    {
        "name": "consume",
        "policy": "serial",
        "cases": ["consumer"],
        "input_artifacts": {
            "consumer": {
                "a": "case_a.result",
                "b": "case_b.result",
            },
        },
    },
]
GROUPS = {"default": {"stages": ["parallel", "consume"]}}
""".lstrip(),
        encoding="utf-8",
    )

    result = execute_project(project_root)

    assert result.ok
    assert [item.request.case_name for item in result.case_results] == [
        "case_a",
        "case_b",
        "consumer",
    ]
    assert all(item.status == "succeeded" for item in result.case_results)
    outputs = [item.output for item in result.case_results[:2]]
    assert all(item["pid"] != os.getpid() for item in outputs)
    assert len({item["pid"] for item in outputs}) == 2
    assert len({item["lease_id"] for item in outputs}) == 2
    assert all(item["threads"] == 1 for item in outputs)
    assert max(item["started_at"] for item in outputs) < min(
        item["finished_at"] for item in outputs
    )
    assert result.case_results[2].output["producer_uris"] == sorted(
        [item["artifact_refs"]["result"]["uri"] for item in outputs]
    )
    assert result.artifact_registry["parallel.case_a.result"].kind == "result"


def test_parallel_fail_fast_records_pending_cases_as_skipped(tmp_path) -> None:
    project_root = create_project(tmp_path / "fail_fast_project", framework="blackbase")
    sources = {
        "slow": """
import time

class Case:
    def run(self):
        time.sleep(0.4)
        return {"completed": True}

def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Case()
""",
        "failing": """
class Case:
    def run(self):
        raise RuntimeError("intentional parallel failure")

def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Case()
""",
        "pending": """
class Case:
    def run(self):
        return {"must_not_start": True}

def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Case()
""",
    }
    for case_name, source in sources.items():
        case_root = add_case(case_name, "solver", project_root=project_root)
        (case_root / "build_solver.py").write_text(source.lstrip(), encoding="utf-8")
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "fail_fast_project"
L0 = {
    "namespace": "fail_fast_project",
    "offer": {"threads": 2, "gpus": 0, "backend": "local"},
    "policy": {"mode": "strict", "max_workers": 2, "max_threads": 2},
    "default_request": {"workers": 1, "threads": 1, "gpus": 0, "backend": "local"},
}
STAGES = [{
    "name": "main",
    "policy": "parallel",
    "failure_policy": "fail_fast",
    "cases": ["slow", "failing", "pending"],
}]
GROUPS = {"default": {"stages": ["main"]}}
""".lstrip(),
        encoding="utf-8",
    )

    result = execute_project(project_root)

    assert not result.ok
    assert result.exit_code == 1
    assert [item.request.case_name for item in result.case_results] == [
        "slow",
        "failing",
        "pending",
    ]
    assert result.case_results[0].status == "cancelled"
    assert result.case_results[1].status == "failed"
    assert "intentional parallel failure" in result.case_results[1].error
    assert result.case_results[2].status == "skipped"
    assert result.case_results[2].output == {}


def test_project_manifest_resumes_successful_case_and_recovers_artifacts(tmp_path) -> None:
    project_root = create_project(tmp_path / "resume_project", framework="blackbase")
    producer_root = add_case("producer", "solver", project_root=project_root)
    consumer_root = add_case("consumer", "trainer", project_root=project_root)
    (producer_root / "build_solver.py").write_text(
        """
from pathlib import Path

class Case:
    def run(self):
        root = Path(__file__).resolve().parents[2]
        counter = root / "producer.count"
        count = int(counter.read_text(encoding="utf-8")) + 1 if counter.exists() else 1
        counter.write_text(str(count), encoding="utf-8")
        return {
            "count": count,
            "artifact_refs": {
                "model": {"uri": "memory://resume/model", "kind": "model"},
            },
        }

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
        count = int(counter.read_text(encoding="utf-8")) + 1 if counter.exists() else 1
        counter.write_text(str(count), encoding="utf-8")
        if count == 1:
            raise RuntimeError("fail once after producer completed")
        return {"model_uri": self.input_artifacts["model"].uri, "count": count}

def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Case()
""".lstrip(),
        encoding="utf-8",
    )
    config_path = project_root / "project_config.py"
    config_path.write_text(
        """
PROJECT_NAME = "resume_project"
L0 = {
    "namespace": "resume_project",
    "offer": {"threads": 1, "gpus": 0, "backend": "local"},
    "policy": {"mode": "strict", "max_workers": 1, "max_threads": 1},
    "default_request": {"workers": 1, "threads": 1, "gpus": 0, "backend": "local"},
}
STAGES = [
    {"name": "produce", "cases": ["producer"]},
    {
        "name": "consume",
        "cases": ["consumer"],
        "input_artifacts": {"consumer": {"model": "producer.model"}},
    },
]
GROUPS = {"default": {"stages": ["produce", "consume"]}}
""".lstrip(),
        encoding="utf-8",
    )

    first = execute_project(project_root, run_id="attempt-one")

    assert not first.ok
    assert Path(first.manifest_path).is_file()
    first_manifest = json.loads(Path(first.manifest_path).read_text(encoding="utf-8"))
    assert first_manifest["status"] == "failed"
    assert [item["status"] for item in first_manifest["cases"]] == [
        "succeeded",
        "failed",
    ]
    assert first_manifest["artifact_registry"]["producer.model"]["uri"] == "memory://resume/model"

    resumed = execute_project(
        project_root,
        run_id="attempt-two",
        resume_from=first.manifest_path,
    )

    assert resumed.ok
    assert [item.status for item in resumed.case_results] == ["resumed", "succeeded"]
    assert resumed.case_results[1].output == {
        "model_uri": "memory://resume/model",
        "count": 2,
    }
    assert (project_root / "producer.count").read_text(encoding="utf-8") == "1"
    assert (project_root / "consumer.count").read_text(encoding="utf-8") == "2"
    assert resumed.resumed_from == first.manifest_path
    resumed_manifest = json.loads(Path(resumed.manifest_path).read_text(encoding="utf-8"))
    assert resumed_manifest["status"] == "ok"
    assert resumed_manifest["resumed_from"] == first.manifest_path

    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    with pytest.raises(ProjectConfigurationError, match="config fingerprint"):
        execute_project(project_root, resume_from=first.manifest_path)
