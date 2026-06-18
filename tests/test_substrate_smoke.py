from __future__ import annotations

import numpy as np

from blackbase.context import ContextStore, create_snapshot_store
from blackbase.kernel import build_pipeline_kernel
from blackbase.project.doctor import run_common_project_doctor
from blackbase.project.project_runner import run_project
from blackbase.project.scaffold import add_case, create_project
from blackbase.resources import PoolScheduler, PoolTaskResult
from blackbase.types import Feedback, UnknownState


def test_pool_scheduler_new_and_legacy_submit() -> None:
    pool = PoolScheduler(total_threads=2)
    try:
        assert pool.submit(lambda x: x + 1, 41).result(timeout=5) == 42

        legacy = pool.submit("task_a", 2, lambda x: x * 2, 21).result(timeout=5)
        assert isinstance(legacy, PoolTaskResult)
        assert legacy.task_id == "task_a"
        assert legacy.result == 42
        assert pool.report()["tasks_completed"] == 2
    finally:
        pool.close()


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
    assert not (solver_case / "build_trainer.py").exists()
    assert (trainer_case / "build_trainer.py").is_file()
    assert not (trainer_case / "build_solver.py").exists()

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
    (trainer_case / "build_trainer.py").write_text(
        """
class Case:
    def __init__(self, resource_context=None):
        self.resource_context = dict(resource_context or {})

    def fit(self):
        return {"ok": True}


def build_trainer(config=None, *, resource_context=None, component_overrides=None):
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
STAGES = [{"name": "main", "cases": ["search_case", "fit_case"]}]
GROUPS = {"default": {"stages": ["main"]}}
""".lstrip(),
        encoding="utf-8",
    )

    assert run_project(project_root, check=True, build_check=True) == 0

    report = run_common_project_doctor(project_root, strict=True)
    errors = [item for item in report.diagnostics if item.level == "error"]
    assert not errors

    (trainer_case / "build_solver.py").write_text("def build_solver():\n    return None\n", encoding="utf-8")
    broken = run_common_project_doctor(project_root, strict=True)
    assert any(item.code == "case-dual-build-entry" for item in broken.diagnostics)
