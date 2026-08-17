from __future__ import annotations

import json

import numpy as np
import pytest

from blackbase.resources import DataRef
from blackbase.types import (
    Feedback,
    PopulationSnapshot,
    SolveQuality,
    SolverResult,
    TrainerResult,
    UnknownState,
)


def test_shared_trainer_result_codec_round_trips_nested_protocol_types() -> None:
    state = UnknownState([1.0, 2.0], metadata={"source": "test"})
    feedback = Feedback(
        objectives=[0.25, 0.5],
        constraints=[-1.0, 0.1],
        metrics={"folds": np.asarray([1, 2])},
    )
    population = PopulationSnapshot(
        candidates=(state,),
        objectives=np.asarray([[0.25, 0.5]]),
        constraints=np.asarray([[-1.0, 0.1]]),
        generation=3,
    )
    model_ref = DataRef(
        uri="artifact://models/best",
        kind="model",
        backend="object-store",
    )
    result = TrainerResult(
        best_model=object(),
        best_model_ref=model_ref,
        best_state=state,
        best_objectives=feedback.objectives,
        best_feedback=feedback,
        history=({"score": np.float64(0.25)},),
        population=population,
        report={"best_score": np.float64(0.25)},
        artifact_refs={"best_model": model_ref},
    )

    payload = result.as_dict()
    json.dumps(payload)
    restored = TrainerResult.from_dict(payload)

    assert restored.best_model is None
    assert restored.best_model_ref == model_ref
    assert isinstance(restored.best_state, UnknownState)
    assert isinstance(restored.best_feedback, Feedback)
    assert isinstance(restored.population, PopulationSnapshot)
    assert np.allclose(restored.best_objectives, [0.25, 0.5])
    assert restored.artifact_refs["best_model"] == model_ref


def test_trainer_result_codec_rejects_opaque_model_without_artifact_ref() -> None:
    result = TrainerResult(best_model=object())

    with pytest.raises(TypeError, match="DataRef/ArtifactRef"):
        result.as_dict()


def test_shared_solver_result_codec_round_trips_pareto_and_refs() -> None:
    best = UnknownState([1.0, 2.0], metadata={"selected_by": "solver"})
    pareto = PopulationSnapshot(
        candidates=(best, UnknownState([2.0, 1.0])),
        objectives=np.asarray([[0.2, 0.8], [0.5, 0.4]]),
        constraints=np.asarray([0.0, 0.1]),
        generation=4,
    )
    history_ref = DataRef(
        uri="artifact://solver/history",
        kind="history",
        backend="object-store",
    )
    result = SolverResult(
        best_solution=best,
        best_objectives=[0.2, 0.8],
        best_constraint_violation=0.0,
        pareto_front=pareto,
        solve_status="feasible",
        termination_reason="iteration_limit",
        feasibility="feasible",
        quality=SolveQuality(
            approximate=True,
            relative_gap=0.05,
            bound=0.15,
            metrics={"hypervolume": np.float64(0.8)},
        ),
        history_ref=history_ref,
        report={"generation": np.int64(4)},
        artifact_refs={"history": history_ref},
    )

    payload = result.as_dict()
    json.dumps(payload)
    restored = SolverResult.from_dict(payload)

    assert payload["protocol_type"] == "blackbase.solver_result"
    assert isinstance(restored.best_solution, UnknownState)
    assert isinstance(restored.pareto_front, PopulationSnapshot)
    assert np.allclose(restored.best_objectives, [0.2, 0.8])
    assert restored.best_constraint_violation == 0.0
    assert restored.solve_status == "feasible"
    assert restored.termination_reason == "iteration_limit"
    assert restored.feasibility == "feasible"
    assert restored.quality.approximate
    assert restored.quality.relative_gap == 0.05
    assert restored.quality.metrics["hypervolume"] == 0.8
    assert restored.history_ref == history_ref
    assert restored.artifact_refs["history"] == history_ref


def test_solver_result_codec_rejects_opaque_solution_without_artifact_ref() -> None:
    result = SolverResult(best_solution=object())

    with pytest.raises(TypeError, match="DataRef/ArtifactRef"):
        result.as_dict()


def test_solver_result_codec_keeps_backward_compatible_terminal_defaults() -> None:
    restored = SolverResult.from_dict(
        {
            "protocol_type": "blackbase.solver_result",
            "schema_version": 1,
        }
    )

    assert restored.solve_status == "unknown"
    assert restored.termination_reason == "unknown"
    assert restored.feasibility == "unknown"
    assert restored.quality == SolveQuality()
    assert restored.quality.approximate is None


def test_solve_quality_preserves_unknown_and_explicit_approximation_states() -> None:
    assert SolveQuality.from_dict({}).approximate is None
    assert SolveQuality.from_dict({"approximate": True}).approximate is True
    assert SolveQuality.from_dict({"approximate": False}).approximate is False
    with pytest.raises(TypeError, match="bool or None"):
        SolveQuality.from_dict({"approximate": "false"})


@pytest.mark.parametrize(
    ("solve_status", "feasibility"),
    (
        ("optimal", "infeasible"),
        ("feasible", "infeasible"),
        ("infeasible", "feasible"),
        ("no_solution", "feasible"),
    ),
)
def test_solver_result_rejects_contradictory_terminal_semantics(
    solve_status: str,
    feasibility: str,
) -> None:
    with pytest.raises(ValueError, match="inconsistent SolverResult"):
        SolverResult(solve_status=solve_status, feasibility=feasibility)


def test_solver_result_rejects_nonzero_gap_for_explicit_non_approximate_result() -> None:
    with pytest.raises(ValueError, match="approximate=False"):
        SolverResult(
            solve_status="feasible",
            feasibility="feasible",
            quality=SolveQuality(approximate=False, relative_gap=0.01),
        )


def test_solver_result_rejects_nonzero_gap_for_optimal_status() -> None:
    with pytest.raises(ValueError, match="solve_status='optimal'"):
        SolverResult(
            solve_status="optimal",
            feasibility="feasible",
            quality=SolveQuality(approximate=None, absolute_gap=0.01),
        )


def test_solver_result_allows_stopped_feasible_and_unbounded_feasible() -> None:
    assert SolverResult(solve_status="stopped", feasibility="feasible")
    assert SolverResult(solve_status="unbounded", feasibility="feasible")
