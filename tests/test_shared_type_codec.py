from __future__ import annotations

import json

import numpy as np
import pytest

from blackbase.resources import DataRef
from blackbase.types import Feedback, PopulationSnapshot, TrainerResult, UnknownState


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
