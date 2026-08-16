"""
Shared protocol types used by nsgablack/mlblack bridge surfaces.

These objects stay small and numpy-oriented so they can move through Case
boundaries without importing either semantic framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .resources.model import DataRef


SHARED_TYPE_SCHEMA_VERSION = 1
_PROTOCOL_TYPE_FIELD = "protocol_type"
_SCHEMA_VERSION_FIELD = "schema_version"


def _protocol_header(type_name: str) -> dict[str, Any]:
    return {
        _PROTOCOL_TYPE_FIELD: f"blackbase.{str(type_name)}",
        _SCHEMA_VERSION_FIELD: SHARED_TYPE_SCHEMA_VERSION,
    }


def _validate_protocol_header(payload: Mapping[str, Any], type_name: str) -> None:
    protocol_type = str(payload.get(_PROTOCOL_TYPE_FIELD, "") or "")
    if protocol_type and protocol_type != f"blackbase.{type_name}":
        raise ValueError(
            f"expected blackbase.{type_name} payload, got {protocol_type}"
        )
    version = int(payload.get(_SCHEMA_VERSION_FIELD, SHARED_TYPE_SCHEMA_VERSION) or 0)
    if version != SHARED_TYPE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported {type_name} schema_version={version}; "
            f"expected {SHARED_TYPE_SCHEMA_VERSION}"
        )


def _encode_data_ref(ref: DataRef) -> dict[str, Any]:
    payload = ref.as_dict()
    payload["metadata"] = _encode_shared_value(
        payload.get("metadata", {}),
        path="data_ref.metadata",
    )
    return {
        **_protocol_header("data_ref"),
        **payload,
    }


def _encode_shared_value(value: Any, *, path: str) -> Any:
    """Encode nested shared-type fields without accepting opaque Python objects."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, DataRef):
        return _encode_data_ref(value)
    if isinstance(value, (Feedback, PopulationSnapshot, TrainerResult)):
        return value.as_dict()
    if isinstance(value, UnknownState):
        return _encode_shared_value(value.as_dict(), path=path)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _encode_shared_value(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _encode_shared_value(item, path=f"{path}[]")
            for item in value
        ]
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return _encode_shared_value(as_dict(), path=path)
    raise TypeError(
        f"shared protocol field '{path}' cannot encode {type(value).__name__}; "
        "publish the object and pass a DataRef/ArtifactRef instead"
    )


def _decode_shared_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_shared_value(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    payload = dict(value)
    protocol_type = str(payload.get(_PROTOCOL_TYPE_FIELD, "") or "")
    if protocol_type == "blackbase.data_ref":
        _validate_protocol_header(payload, "data_ref")
        return DataRef.from_dict(payload)
    if protocol_type == "blackbase.feedback":
        return Feedback.from_dict(payload)
    if protocol_type == "blackbase.population_snapshot":
        return PopulationSnapshot.from_dict(payload)
    if protocol_type == "blackbase.trainer_result":
        return TrainerResult.from_dict(payload)
    return {str(key): _decode_shared_value(item) for key, item in payload.items()}


def _coerce_data_ref(value: Any) -> DataRef | None:
    if value is None:
        return None
    if isinstance(value, DataRef):
        return value
    if isinstance(value, Mapping):
        return DataRef.from_dict(value)
    raise TypeError(f"expected DataRef-compatible value, got {type(value).__name__}")


@dataclass(frozen=True)
class Feedback:
    """Evaluation feedback for a candidate/model state."""

    objectives: np.ndarray
    constraints: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    gradients: Optional[np.ndarray] = None
    loss: Optional[float] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    residuals: Optional[np.ndarray] = None
    signals: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "objectives", np.asarray(self.objectives, dtype=float))
        object.__setattr__(self, "constraints", np.asarray(self.constraints, dtype=float))
        if self.gradients is not None:
            object.__setattr__(self, "gradients", np.asarray(self.gradients, dtype=float))
        if self.residuals is not None:
            object.__setattr__(self, "residuals", np.asarray(self.residuals, dtype=float))
        object.__setattr__(self, "metrics", dict(self.metrics or {}))
        object.__setattr__(self, "signals", dict(self.signals or {}))
        object.__setattr__(self, "info", dict(self.info or {}))

    @property
    def ok(self) -> bool:
        return self.objectives is not None and self.objectives.size > 0

    def scalar_score(self, constraint_penalty: float = 1e6) -> float:
        obj = float(np.mean(self.objectives)) if self.objectives.size > 0 else 0.0
        if self.constraints is not None and self.constraints.size > 0:
            obj += float(constraint_penalty) * float(np.sum(np.maximum(0.0, self.constraints)))
        return obj

    def as_dict(self) -> dict[str, Any]:
        return {
            **_protocol_header("feedback"),
            "objectives": self.objectives.tolist(),
            "constraints": self.constraints.tolist(),
            "gradients": None if self.gradients is None else self.gradients.tolist(),
            "loss": _encode_shared_value(self.loss, path="feedback.loss"),
            "metrics": _encode_shared_value(self.metrics, path="feedback.metrics"),
            "residuals": None if self.residuals is None else self.residuals.tolist(),
            "signals": _encode_shared_value(self.signals, path="feedback.signals"),
            "info": _encode_shared_value(self.info, path="feedback.info"),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Feedback":
        data = dict(payload or {})
        _validate_protocol_header(data, "feedback")
        return cls(
            objectives=np.asarray(data.get("objectives", ()), dtype=float),
            constraints=np.asarray(data.get("constraints", ()), dtype=float),
            gradients=(
                None
                if data.get("gradients") is None
                else np.asarray(data.get("gradients"), dtype=float)
            ),
            loss=None if data.get("loss") is None else float(data.get("loss")),
            metrics=dict(_decode_shared_value(data.get("metrics", {})) or {}),
            residuals=(
                None
                if data.get("residuals") is None
                else np.asarray(data.get("residuals"), dtype=float)
            ),
            signals=dict(_decode_shared_value(data.get("signals", {})) or {}),
            info=dict(_decode_shared_value(data.get("info", {})) or {}),
        )


@dataclass(frozen=True, init=False)
class UnknownState:
    """
    Numeric candidate state.

    Compatibility note: old mlblack code used both `meta=` and `metadata=`.
    BlackBase accepts both and exposes `.metadata` as an alias for `.meta`.
    """

    values: np.ndarray
    meta: dict[str, Any]

    def __init__(
        self,
        values: Any,
        meta: Optional[Mapping[str, Any]] = None,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        merged = dict(meta or {})
        if metadata:
            merged.update(dict(metadata))
        object.__setattr__(self, "values", np.asarray(values, dtype=float))
        object.__setattr__(self, "meta", merged)

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.meta)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.values.shape)

    @property
    def size(self) -> int:
        return int(self.values.size)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=float)

    def with_values(self, values: Any, **metadata: Any) -> "UnknownState":
        meta = dict(self.meta)
        meta.update(metadata)
        return UnknownState(values=np.asarray(values, dtype=float), meta=meta)

    def with_meta(self, **metadata: Any) -> "UnknownState":
        meta = dict(self.meta)
        meta.update(metadata)
        return UnknownState(values=self.values.copy(), meta=meta)

    def to_protocol_payload(self) -> dict[str, Any]:
        """Return the stable, JSON-safe-codec input used by shared stores."""
        return {
            "version": 1,
            "values": self.as_array(),
            "metadata": dict(self.meta),
        }

    @classmethod
    def from_protocol_payload(cls, payload: Mapping[str, Any]) -> "UnknownState":
        version = int(payload.get("version", 1) or 1)
        if version != 1:
            raise ValueError(f"unsupported UnknownState protocol version: {version}")
        return cls(
            values=np.asarray(payload.get("values", []), dtype=float),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {"values": self.as_array().tolist(), "metadata": dict(self.meta)}


@dataclass(frozen=True)
class PopulationSnapshot:
    """Snapshot of candidate states and their feedback arrays."""

    candidates: Sequence[UnknownState]
    objectives: np.ndarray
    constraints: Optional[np.ndarray] = None
    generation: int = 0
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates or ()))
        object.__setattr__(self, "objectives", np.asarray(self.objectives, dtype=float))
        if self.constraints is not None:
            object.__setattr__(self, "constraints", np.asarray(self.constraints, dtype=float))
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def as_dict(self) -> dict[str, Any]:
        return {
            **_protocol_header("population_snapshot"),
            "candidates": [
                _encode_shared_value(candidate, path="population_snapshot.candidates[]")
                for candidate in self.candidates
            ],
            "objectives": self.objectives.tolist(),
            "constraints": None if self.constraints is None else self.constraints.tolist(),
            "generation": self.generation,
            "timestamp": self.timestamp,
            "metadata": _encode_shared_value(
                self.metadata,
                path="population_snapshot.metadata",
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PopulationSnapshot":
        data = dict(payload or {})
        _validate_protocol_header(data, "population_snapshot")
        return cls(
            candidates=tuple(
                item
                if isinstance(item, UnknownState)
                else UnknownState.from_protocol_payload(item)
                for item in data.get("candidates", ()) or ()
            ),
            objectives=np.asarray(data.get("objectives", ()), dtype=float),
            constraints=(
                None
                if data.get("constraints") is None
                else np.asarray(data.get("constraints"), dtype=float)
            ),
            generation=int(data.get("generation", 0) or 0),
            timestamp=float(data.get("timestamp", 0.0) or 0.0),
            metadata=dict(_decode_shared_value(data.get("metadata", {})) or {}),
        )


@dataclass(frozen=True)
class TrainerResult:
    """Result payload for an ML-style Case run."""

    best_model: Any = None
    best_state: Any = None
    best_objectives: Optional[np.ndarray] = None
    best_feedback: Any = None
    history: tuple = ()
    population: Optional[PopulationSnapshot] = None
    report: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    best_model_ref: DataRef | None = None
    population_ref: DataRef | None = None
    history_ref: DataRef | None = None
    artifact_refs: Mapping[str, DataRef] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.best_objectives is not None:
            object.__setattr__(self, "best_objectives", np.asarray(self.best_objectives, dtype=float))
        object.__setattr__(self, "history", tuple(self.history or ()))
        object.__setattr__(self, "report", dict(self.report or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        object.__setattr__(self, "best_model_ref", _coerce_data_ref(self.best_model_ref))
        object.__setattr__(self, "population_ref", _coerce_data_ref(self.population_ref))
        object.__setattr__(self, "history_ref", _coerce_data_ref(self.history_ref))
        artifact_refs: dict[str, DataRef] = {}
        for key, value in dict(self.artifact_refs or {}).items():
            ref = _coerce_data_ref(value)
            if ref is None:
                raise TypeError(f"artifact_refs['{key}'] must be a DataRef")
            artifact_refs[str(key)] = ref
        object.__setattr__(self, "artifact_refs", artifact_refs)

    def as_dict(self) -> dict[str, Any]:
        best_model = None
        if self.best_model_ref is None:
            best_model = _encode_shared_value(
                self.best_model,
                path="trainer_result.best_model",
            )
        return {
            **_protocol_header("trainer_result"),
            "best_model": best_model,
            "best_model_ref": (
                None if self.best_model_ref is None else _encode_data_ref(self.best_model_ref)
            ),
            "best_state": _encode_shared_value(
                self.best_state,
                path="trainer_result.best_state",
            ),
            "best_objectives": (
                None if self.best_objectives is None else self.best_objectives.tolist()
            ),
            "best_feedback": _encode_shared_value(
                self.best_feedback,
                path="trainer_result.best_feedback",
            ),
            "history": (
                []
                if self.history_ref is not None
                else _encode_shared_value(self.history, path="trainer_result.history")
            ),
            "history_ref": (
                None if self.history_ref is None else _encode_data_ref(self.history_ref)
            ),
            "population": (
                None
                if self.population_ref is not None
                else _encode_shared_value(
                    self.population,
                    path="trainer_result.population",
                )
            ),
            "population_ref": (
                None if self.population_ref is None else _encode_data_ref(self.population_ref)
            ),
            "report": _encode_shared_value(self.report, path="trainer_result.report"),
            "metadata": _encode_shared_value(
                self.metadata,
                path="trainer_result.metadata",
            ),
            "artifact_refs": {
                str(key): _encode_data_ref(value)
                for key, value in self.artifact_refs.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainerResult":
        data = dict(payload or {})
        _validate_protocol_header(data, "trainer_result")
        best_state = _decode_shared_value(data.get("best_state"))
        if isinstance(best_state, Mapping) and "values" in best_state:
            best_state = UnknownState.from_protocol_payload(best_state)
        best_feedback = _decode_shared_value(data.get("best_feedback"))
        population = _decode_shared_value(data.get("population"))
        return cls(
            best_model=_decode_shared_value(data.get("best_model")),
            best_state=best_state,
            best_objectives=(
                None
                if data.get("best_objectives") is None
                else np.asarray(data.get("best_objectives"), dtype=float)
            ),
            best_feedback=best_feedback,
            history=tuple(_decode_shared_value(data.get("history", ())) or ()),
            population=population,
            report=dict(_decode_shared_value(data.get("report", {})) or {}),
            metadata=dict(_decode_shared_value(data.get("metadata", {})) or {}),
            best_model_ref=_decode_shared_value(data.get("best_model_ref")),
            population_ref=_decode_shared_value(data.get("population_ref")),
            history_ref=_decode_shared_value(data.get("history_ref")),
            artifact_refs={
                str(key): _decode_shared_value(value)
                for key, value in dict(data.get("artifact_refs", {}) or {}).items()
            },
        )


__all__ = [
    "SHARED_TYPE_SCHEMA_VERSION",
    "Feedback",
    "UnknownState",
    "PopulationSnapshot",
    "TrainerResult",
]
