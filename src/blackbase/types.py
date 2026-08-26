"""
Shared protocol types used by nsgablack/mlblack bridge surfaces.

These objects stay small and numpy-oriented so they can move through Case
boundaries without importing either semantic framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .resources.model import DataRef
from .selection import normalize_row_selector
from .state_ref import StateRef


SHARED_TYPE_SCHEMA_VERSION = 1
_PROTOCOL_TYPE_FIELD = "protocol_type"
_SCHEMA_VERSION_FIELD = "schema_version"

SOLVE_STATUSES = frozenset(
    {
        "unknown",
        "optimal",
        "feasible",
        "infeasible",
        "unbounded",
        "no_solution",
        "stopped",
        "failed",
    }
)
FEASIBILITY_STATUSES = frozenset({"unknown", "feasible", "infeasible"})

_SOLVER_STATUS_RULES: Mapping[str, Mapping[str, Any]] = {
    "optimal": {
        "forbidden_feasibility": frozenset({"infeasible"}),
        "forbids_approximate": True,
        "forbids_positive_gap": True,
        "forbids_infeasible_best": True,
    },
    "feasible": {
        "forbidden_feasibility": frozenset({"infeasible"}),
        "forbids_infeasible_best": True,
    },
    "infeasible": {
        "forbidden_feasibility": frozenset({"feasible"}),
        "forbids_feasible_best": True,
    },
    "unbounded": {
        "forbidden_feasibility": frozenset({"infeasible"}),
    },
    "no_solution": {
        "forbidden_feasibility": frozenset({"feasible"}),
        "forbids_feasible_best": True,
    },
    "stopped": {},
    "failed": {},
    "unknown": {},
}


def _protocol_header(type_name: str) -> dict[str, Any]:
    return {
        _PROTOCOL_TYPE_FIELD: f"blackbase.{str(type_name)}",
        _SCHEMA_VERSION_FIELD: SHARED_TYPE_SCHEMA_VERSION,
    }


def _readonly_array(value: Any, *, dtype: Any = float) -> np.ndarray:
    """Return an array whose public view cannot be made writeable again."""

    owner = np.array(value, dtype=dtype, copy=True)
    if owner.dtype.hasobject:
        # Object arrays cannot use an immutable byte backing safely.  Detach
        # them and expose a non-owning readonly view; wire codecs reject
        # arbitrary objects before transport.
        owner.setflags(write=False)
        view = owner.view()
        view.setflags(write=False)
        return view
    immutable = np.frombuffer(owner.tobytes(order="C"), dtype=owner.dtype)
    immutable = immutable.reshape(owner.shape)
    immutable.setflags(write=False)
    return immutable


def _freeze_shared_value(value: Any) -> Any:
    """Detach nested protocol data and expose it through immutable containers."""

    if isinstance(value, np.ndarray):
        return _readonly_array(value, dtype=value.dtype)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_shared_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_shared_value(item) for item in value)
    return value


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
    if isinstance(value, StateRef):
        return value.as_dict()
    if isinstance(
        value,
        (Feedback, PopulationSnapshot, TrainerResult, SolveQuality, SolverResult),
    ):
        return value.as_dict()
    if isinstance(value, (UnknownState, CandidateBatch)):
        return _encode_shared_value(value.as_dict(), path=path)
    if isinstance(value, np.ndarray):
        return _encode_shared_value(value.tolist(), path=path)
    if isinstance(value, np.generic):
        return _encode_shared_value(value.item(), path=path)
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
    if protocol_type == "blackbase.state_ref":
        return StateRef.from_dict(payload)
    if protocol_type == "blackbase.evaluation_provider_spec":
        from .evaluation.model import EvaluationProviderSpec

        return EvaluationProviderSpec.from_dict(payload)
    if protocol_type == "blackbase.evaluation_request":
        from .evaluation.model import EvaluationRequest

        return EvaluationRequest.from_dict(payload)
    if protocol_type == "blackbase.evaluation_binding":
        from .evaluation.model import EvaluationBinding

        return EvaluationBinding.from_dict(payload)
    if protocol_type == "blackbase.evaluation_result":
        from .evaluation.model import EvaluationResult

        return EvaluationResult.from_dict(payload)
    if protocol_type == "blackbase.state_transition_request":
        from .evaluation.state_transition import StateTransitionRequest

        return StateTransitionRequest.from_dict(payload)
    if protocol_type == "blackbase.state_transition_result":
        from .evaluation.state_transition import StateTransitionResult

        return StateTransitionResult.from_dict(payload)
    if protocol_type == "blackbase.state_materialization_request":
        from .evaluation.state_transition import StateMaterializationRequest

        return StateMaterializationRequest.from_dict(payload)
    if protocol_type == "blackbase.state_materialization_result":
        from .evaluation.state_transition import StateMaterializationResult

        return StateMaterializationResult.from_dict(payload)
    if protocol_type == "blackbase.state_release_request":
        from .evaluation.state_transition import StateReleaseRequest

        return StateReleaseRequest.from_dict(payload)
    if protocol_type == "blackbase.state_release_result":
        from .evaluation.state_transition import StateReleaseResult

        return StateReleaseResult.from_dict(payload)
    if protocol_type == "blackbase.feedback":
        return Feedback.from_dict(payload)
    if protocol_type == "blackbase.candidate_batch":
        return CandidateBatch.from_dict(payload)
    if protocol_type == "blackbase.population_snapshot":
        return PopulationSnapshot.from_dict(payload)
    if protocol_type == "blackbase.trainer_result":
        return TrainerResult.from_dict(payload)
    if protocol_type == "blackbase.solve_quality":
        return SolveQuality.from_dict(payload)
    if protocol_type == "blackbase.solver_result":
        return SolverResult.from_dict(payload)
    return {str(key): _decode_shared_value(item) for key, item in payload.items()}


def _coerce_data_ref(value: Any) -> DataRef | None:
    if value is None:
        return None
    if isinstance(value, DataRef):
        return value
    if isinstance(value, Mapping):
        return DataRef.from_dict(value)
    raise TypeError(f"expected DataRef-compatible value, got {type(value).__name__}")


def encode_shared_value(value: Any, *, path: str = "value") -> Any:
    """Encode a value using the public BlackBase wire-safe value codec."""

    return _encode_shared_value(value, path=str(path or "value"))


def decode_shared_value(value: Any) -> Any:
    """Decode a value produced by :func:`encode_shared_value`."""

    return _decode_shared_value(value)


@dataclass(frozen=True)
class Feedback:
    """Evaluation feedback for a candidate/model state."""

    objectives: np.ndarray
    constraints: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    gradients: Optional[np.ndarray] = None
    loss: Optional[float] = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    residuals: Optional[np.ndarray] = None
    signals: Mapping[str, Any] = field(default_factory=dict)
    info: Mapping[str, Any] = field(default_factory=dict)
    gradient_ref: StateRef | DataRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "objectives", _readonly_array(self.objectives))
        object.__setattr__(self, "constraints", _readonly_array(self.constraints))
        if self.gradients is not None:
            object.__setattr__(self, "gradients", _readonly_array(self.gradients))
        if self.gradient_ref is not None and not isinstance(
            self.gradient_ref,
            (StateRef, DataRef),
        ):
            raise TypeError("Feedback.gradient_ref must be a StateRef, DataRef, or None")
        if self.residuals is not None:
            object.__setattr__(self, "residuals", _readonly_array(self.residuals))
        object.__setattr__(
            self,
            "metrics",
            _freeze_shared_value(dict(self.metrics or {})),
        )
        object.__setattr__(
            self,
            "signals",
            _freeze_shared_value(dict(self.signals or {})),
        )
        object.__setattr__(
            self,
            "info",
            _freeze_shared_value(dict(self.info or {})),
        )

    @property
    def ok(self) -> bool:
        return self.objectives is not None and self.objectives.size > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            **_protocol_header("feedback"),
            "objectives": self.objectives.tolist(),
            "constraints": self.constraints.tolist(),
            "gradients": None if self.gradients is None else self.gradients.tolist(),
            "gradient_ref": _encode_shared_value(
                self.gradient_ref,
                path="feedback.gradient_ref",
            ),
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
            gradient_ref=_decode_shared_value(data.get("gradient_ref")),
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


@dataclass(frozen=True)
class UnknownState:
    """Numeric candidate state with one canonical metadata field."""

    values: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        canonical_metadata = _decode_shared_value(
            _encode_shared_value(
                dict(self.metadata or {}),
                path="unknown_state.metadata",
            )
        )
        object.__setattr__(self, "values", _readonly_array(self.values))
        object.__setattr__(
            self,
            "metadata",
            _freeze_shared_value(dict(canonical_metadata or {})),
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.values.shape)

    @property
    def size(self) -> int:
        return int(self.values.size)

    def as_array(self) -> np.ndarray:
        return np.array(self.values, dtype=float, copy=True)

    def with_values(self, values: Any, **metadata: Any) -> "UnknownState":
        current_metadata = dict(self.metadata)
        current_metadata.update(metadata)
        return UnknownState(values=np.asarray(values, dtype=float), metadata=current_metadata)

    def with_meta(self, **metadata: Any) -> "UnknownState":
        current_metadata = dict(self.metadata)
        current_metadata.update(metadata)
        return UnknownState(values=self.values.copy(), metadata=current_metadata)

    def to_protocol_payload(self) -> dict[str, Any]:
        """Return the stable, JSON-safe-codec input used by shared stores."""
        return {
            "version": 1,
            "values": self.as_array(),
            "metadata": _encode_shared_value(
                self.metadata,
                path="unknown_state.metadata",
            ),
        }

    @classmethod
    def from_protocol_payload(cls, payload: Mapping[str, Any]) -> "UnknownState":
        version = int(payload.get("version", 1) or 1)
        if version != 1:
            raise ValueError(f"unsupported UnknownState protocol version: {version}")
        return cls(
            values=np.asarray(payload.get("values", []), dtype=float),
            metadata=dict(_decode_shared_value(payload.get("metadata", {})) or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "values": self.as_array().tolist(),
            "metadata": _encode_shared_value(
                self.metadata,
                path="unknown_state.metadata",
            ),
        }


@dataclass(frozen=True)
class CandidateBatch:
    """Aligned semantic and numeric views of one candidate batch.

    Numeric algorithms consume :attr:`numeric_matrix`; representation,
    evaluation and lineage code consume :attr:`semantic_states`.  The two
    views are validated together so converting an ``UnknownState`` to a row
    never discards its decode-relevant metadata.
    """

    semantic_states: Sequence[UnknownState]
    numeric_matrix: np.ndarray
    candidate_tokens: Sequence[str | None] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        states = tuple(
            state
            if isinstance(state, UnknownState)
            else UnknownState.from_protocol_payload(state)
            if isinstance(state, Mapping)
            else UnknownState(values=state)
            for state in tuple(self.semantic_states)
        )
        matrix = np.asarray(self.numeric_matrix, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1) if matrix.size else matrix.reshape(0, 0)
        if matrix.ndim != 2:
            raise ValueError("CandidateBatch.numeric_matrix must be two-dimensional")
        if int(matrix.shape[0]) != len(states):
            raise ValueError(
                "CandidateBatch semantic/numeric row counts must match: "
                f"{len(states)} != {int(matrix.shape[0])}"
            )
        detached_states: list[UnknownState] = []
        for index, state in enumerate(states):
            values = np.asarray(state.as_array(), dtype=float).reshape(-1)
            if int(values.size) != int(matrix.shape[1]):
                raise ValueError(
                    f"CandidateBatch.semantic_states[{index}] has dimension "
                    f"{int(values.size)}; expected {int(matrix.shape[1])}"
                )
            if not np.array_equal(values, matrix[index], equal_nan=True):
                raise ValueError(
                    f"CandidateBatch.semantic_states[{index}] does not match its numeric row"
                )
            detached_states.append(
                UnknownState(
                    values=values.copy(),
                    metadata=dict(
                        _decode_shared_value(
                            _encode_shared_value(
                                dict(state.metadata),
                                path=f"candidate_batch.semantic_states[{index}].metadata",
                            )
                        )
                        or {}
                    ),
                )
            )
        tokens = tuple(self.candidate_tokens)
        if not tokens:
            tokens = (None,) * len(states)
        if len(tokens) != len(states):
            raise ValueError("CandidateBatch.candidate_tokens must align with candidate rows")
        normalized_tokens = tuple(
            None if token is None else str(token).strip() or None for token in tokens
        )
        matrix_copy = _readonly_array(matrix)
        object.__setattr__(self, "semantic_states", tuple(detached_states))
        object.__setattr__(self, "numeric_matrix", matrix_copy)
        object.__setattr__(self, "candidate_tokens", normalized_tokens)

    @classmethod
    def from_candidates(
        cls,
        candidates: Sequence[Any],
        *,
        candidate_tokens: Sequence[str | None] = (),
    ) -> "CandidateBatch":
        states = tuple(
            item
            if isinstance(item, UnknownState)
            else UnknownState(values=np.asarray(item, dtype=float).reshape(-1))
            for item in tuple(candidates)
        )
        if states:
            matrix = np.stack(
                [np.asarray(state.as_array(), dtype=float).reshape(-1) for state in states],
                axis=0,
            )
        else:
            matrix = np.empty((0, 0), dtype=float)
        return cls(
            semantic_states=states,
            numeric_matrix=matrix,
            candidate_tokens=tuple(candidate_tokens),
        )

    def numeric_rows(self) -> tuple[np.ndarray, ...]:
        """Return detached writable rows for ndarray-native algorithms."""

        return tuple(np.asarray(row, dtype=float).copy() for row in self.numeric_matrix)

    def subset(self, selector: slice | Sequence[int] | np.ndarray) -> "CandidateBatch":
        """Return an aligned semantic/numeric/token subset."""

        indices = normalize_row_selector(selector, len(self.semantic_states))
        return type(self)(
            semantic_states=tuple(
                self.semantic_states[int(index)] for index in indices
            ),
            numeric_matrix=self.numeric_matrix[indices],
            candidate_tokens=tuple(
                self.candidate_tokens[int(index)] for index in indices
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **_protocol_header("candidate_batch"),
            "semantic_states": [state.as_dict() for state in self.semantic_states],
            "numeric_matrix": self.numeric_matrix.tolist(),
            "candidate_tokens": list(self.candidate_tokens),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateBatch":
        _validate_protocol_header(payload, "candidate_batch")
        states = tuple(
            UnknownState.from_protocol_payload(dict(item))
            for item in tuple(payload.get("semantic_states", ()) or ())
        )
        return cls(
            semantic_states=states,
            numeric_matrix=np.asarray(payload.get("numeric_matrix", ()), dtype=float),
            candidate_tokens=tuple(payload.get("candidate_tokens", ()) or ()),
        )


@dataclass(frozen=True)
class PopulationSnapshot:
    """Snapshot of candidate states and their feedback arrays."""

    candidates: Sequence[UnknownState]
    objectives: np.ndarray
    constraints: Optional[np.ndarray] = None
    candidate_tokens: Sequence[str | None] = field(default_factory=tuple)
    generation: int = 0
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidates = tuple(
            candidate
            if isinstance(candidate, UnknownState)
            else UnknownState.from_protocol_payload(candidate)
            if isinstance(candidate, Mapping)
            else UnknownState(values=candidate)
            for candidate in tuple(self.candidates or ())
        )
        objectives = np.asarray(self.objectives, dtype=float)
        if objectives.ndim == 1:
            objectives = (
                objectives.reshape(1, -1)
                if len(candidates) == 1
                else objectives.reshape(-1, 1)
            )
        if objectives.ndim != 2 or objectives.shape[0] != len(candidates):
            raise ValueError(
                "PopulationSnapshot objectives must be a 2D array aligned with candidates"
            )
        tokens = tuple(self.candidate_tokens or ())
        if not tokens:
            tokens = (None,) * len(candidates)
        if len(tokens) != len(candidates):
            raise ValueError(
                "PopulationSnapshot candidate_tokens must align with candidates"
            )
        tokens = tuple(
            None if token is None else str(token).strip() or None
            for token in tokens
        )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "objectives", _readonly_array(objectives))
        if self.constraints is not None:
            constraints = np.asarray(self.constraints, dtype=float)
            if constraints.ndim == 0:
                constraints = constraints.reshape(1)
            if constraints.shape[0] != len(candidates):
                raise ValueError(
                    "PopulationSnapshot constraints must align with candidates"
                )
            object.__setattr__(self, "constraints", _readonly_array(constraints))
        object.__setattr__(self, "candidate_tokens", tokens)
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "timestamp", float(self.timestamp))
        canonical_metadata = _decode_shared_value(
            _encode_shared_value(
                dict(self.metadata or {}),
                path="population_snapshot.metadata",
            )
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_shared_value(dict(canonical_metadata or {})),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **_protocol_header("population_snapshot"),
            "candidates": [
                _encode_shared_value(candidate, path="population_snapshot.candidates[]")
                for candidate in self.candidates
            ],
            "objectives": self.objectives.tolist(),
            "constraints": None if self.constraints is None else self.constraints.tolist(),
            "candidate_tokens": list(self.candidate_tokens),
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
            candidate_tokens=tuple(data.get("candidate_tokens", ()) or ()),
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
    best_state_ref: DataRef | None = None
    population_ref: DataRef | None = None
    history_ref: DataRef | None = None
    report_ref: DataRef | None = None
    artifact_refs: Mapping[str, DataRef] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.best_objectives is not None:
            object.__setattr__(
                self,
                "best_objectives",
                _readonly_array(np.asarray(self.best_objectives, dtype=float)),
            )
        object.__setattr__(
            self,
            "history",
            _freeze_shared_value(
                _encode_shared_value(
                    tuple(self.history or ()), path="trainer_result.history"
                )
            ),
        )
        object.__setattr__(
            self,
            "report",
            _freeze_shared_value(
                _encode_shared_value(
                    dict(self.report or {}), path="trainer_result.report"
                )
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_shared_value(
                _encode_shared_value(
                    dict(self.metadata or {}), path="trainer_result.metadata"
                )
            ),
        )
        object.__setattr__(self, "best_model_ref", _coerce_data_ref(self.best_model_ref))
        object.__setattr__(self, "best_state_ref", _coerce_data_ref(self.best_state_ref))
        object.__setattr__(self, "population_ref", _coerce_data_ref(self.population_ref))
        object.__setattr__(self, "history_ref", _coerce_data_ref(self.history_ref))
        object.__setattr__(self, "report_ref", _coerce_data_ref(self.report_ref))
        artifact_refs: dict[str, DataRef] = {}
        for key, value in dict(self.artifact_refs or {}).items():
            ref = _coerce_data_ref(value)
            if ref is None:
                raise TypeError(f"artifact_refs['{key}'] must be a DataRef")
            artifact_refs[str(key)] = ref
        object.__setattr__(self, "artifact_refs", MappingProxyType(artifact_refs))

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
            "best_state": (
                None
                if self.best_state_ref is not None
                else _encode_shared_value(
                    self.best_state,
                    path="trainer_result.best_state",
                )
            ),
            "best_state_ref": (
                None
                if self.best_state_ref is None
                else _encode_data_ref(self.best_state_ref)
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
            "report": (
                {}
                if self.report_ref is not None
                else _encode_shared_value(self.report, path="trainer_result.report")
            ),
            "report_ref": (
                None if self.report_ref is None else _encode_data_ref(self.report_ref)
            ),
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
            best_state_ref=_decode_shared_value(data.get("best_state_ref")),
            population_ref=_decode_shared_value(data.get("population_ref")),
            history_ref=_decode_shared_value(data.get("history_ref")),
            report_ref=_decode_shared_value(data.get("report_ref")),
            artifact_refs={
                str(key): _decode_shared_value(value)
                for key, value in dict(data.get("artifact_refs", {}) or {}).items()
            },
        )


@dataclass(frozen=True)
class SolveQuality:
    """Optional proof/approximation quality attached to a Solver terminal state."""

    approximate: Optional[bool] = None
    absolute_gap: Optional[float] = None
    relative_gap: Optional[float] = None
    bound: Optional[float] = None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        approximate = self.approximate
        if approximate is not None and not isinstance(approximate, (bool, np.bool_)):
            raise TypeError("SolveQuality.approximate must be bool or None")
        object.__setattr__(
            self,
            "approximate",
            None if approximate is None else bool(approximate),
        )
        for name in ("absolute_gap", "relative_gap"):
            value = getattr(self, name)
            if value is None:
                continue
            normalized = float(value)
            if not np.isfinite(normalized):
                raise ValueError(f"SolveQuality.{name} must be finite")
            if normalized < 0:
                raise ValueError(f"SolveQuality.{name} must be non-negative")
            object.__setattr__(self, name, normalized)
        if self.bound is not None:
            bound = float(self.bound)
            if not np.isfinite(bound):
                raise ValueError("SolveQuality.bound must be finite")
            object.__setattr__(self, "bound", bound)
        object.__setattr__(
            self,
            "metrics",
            _freeze_shared_value(
                _encode_shared_value(
                    dict(self.metrics or {}), path="solve_quality.metrics"
                )
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **_protocol_header("solve_quality"),
            "approximate": self.approximate,
            "absolute_gap": self.absolute_gap,
            "relative_gap": self.relative_gap,
            "bound": self.bound,
            "metrics": _encode_shared_value(
                self.metrics,
                path="solve_quality.metrics",
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "SolveQuality":
        data = dict(payload or {})
        if data:
            _validate_protocol_header(data, "solve_quality")
        raw_approximate = data.get("approximate")
        return cls(
            approximate=raw_approximate,
            absolute_gap=(
                None
                if data.get("absolute_gap") is None
                else float(data.get("absolute_gap"))
            ),
            relative_gap=(
                None
                if data.get("relative_gap") is None
                else float(data.get("relative_gap"))
            ),
            bound=None if data.get("bound") is None else float(data.get("bound")),
            metrics=dict(_decode_shared_value(data.get("metrics", {})) or {}),
        )


@dataclass(frozen=True)
class SolverResult:
    """Versioned result payload for an optimization/search Case run.

    The payload describes optimization outputs without selecting a Pareto
    solution for the consumer.  Semantic layers may project ``best_solution``
    or ``pareto_front`` according to an explicit policy.
    """

    best_solution: Any = None
    best_objectives: Optional[np.ndarray] = None
    best_constraint_violation: Optional[float] = None
    best_candidate_token: str | None = None
    best_evaluation_id: str | None = None
    best_provenance: Mapping[str, Any] = field(default_factory=dict)
    pareto_front: Optional[PopulationSnapshot] = None
    solve_status: str = "unknown"
    termination_reason: str = "unknown"
    feasibility: str = "unknown"
    quality: SolveQuality = field(default_factory=SolveQuality)
    history: tuple = ()
    report: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    best_solution_ref: DataRef | None = None
    pareto_front_ref: DataRef | None = None
    history_ref: DataRef | None = None
    artifact_refs: Mapping[str, DataRef] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.best_objectives is not None:
            object.__setattr__(
                self,
                "best_objectives",
                _readonly_array(np.asarray(self.best_objectives, dtype=float).reshape(-1)),
            )
        if self.best_constraint_violation is not None:
            best_constraint_violation = float(self.best_constraint_violation)
            if not np.isfinite(best_constraint_violation):
                raise ValueError(
                    "SolverResult.best_constraint_violation must be finite"
                )
            object.__setattr__(
                self,
                "best_constraint_violation",
                best_constraint_violation,
            )
        object.__setattr__(
            self,
            "best_candidate_token",
            None
            if self.best_candidate_token is None
            else str(self.best_candidate_token).strip() or None,
        )
        object.__setattr__(
            self,
            "best_evaluation_id",
            None
            if self.best_evaluation_id is None
            else str(self.best_evaluation_id).strip() or None,
        )
        canonical_provenance = _encode_shared_value(
            dict(self.best_provenance or {}),
            path="solver_result.best_provenance",
        )
        object.__setattr__(
            self,
            "best_provenance",
            _freeze_shared_value(dict(canonical_provenance or {})),
        )
        solve_status = str(self.solve_status or "unknown").strip().lower()
        if solve_status not in SOLVE_STATUSES:
            raise ValueError(f"unsupported SolverResult solve_status '{solve_status}'")
        feasibility = str(self.feasibility or "unknown").strip().lower()
        if feasibility not in FEASIBILITY_STATUSES:
            raise ValueError(f"unsupported SolverResult feasibility '{feasibility}'")
        termination_reason = str(self.termination_reason or "unknown").strip().lower()
        quality = self.quality
        if not isinstance(quality, SolveQuality):
            if not isinstance(quality, Mapping):
                raise TypeError("SolverResult.quality must be SolveQuality-compatible")
            quality = SolveQuality.from_dict(quality)
        object.__setattr__(self, "solve_status", solve_status)
        object.__setattr__(self, "termination_reason", termination_reason)
        object.__setattr__(self, "feasibility", feasibility)
        object.__setattr__(self, "quality", quality)
        _validate_solver_result_consistency(
            solve_status=solve_status,
            feasibility=feasibility,
            quality=quality,
            best_constraint_violation=self.best_constraint_violation,
        )
        pareto_front = self.pareto_front
        if pareto_front is not None and not isinstance(pareto_front, PopulationSnapshot):
            pareto_front = PopulationSnapshot.from_dict(pareto_front)
        object.__setattr__(self, "pareto_front", pareto_front)
        object.__setattr__(
            self,
            "history",
            _freeze_shared_value(
                _encode_shared_value(
                    tuple(self.history or ()), path="solver_result.history"
                )
            ),
        )
        object.__setattr__(
            self,
            "report",
            _freeze_shared_value(
                _encode_shared_value(
                    dict(self.report or {}), path="solver_result.report"
                )
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_shared_value(
                _encode_shared_value(
                    dict(self.metadata or {}), path="solver_result.metadata"
                )
            ),
        )
        object.__setattr__(self, "best_solution_ref", _coerce_data_ref(self.best_solution_ref))
        object.__setattr__(self, "pareto_front_ref", _coerce_data_ref(self.pareto_front_ref))
        object.__setattr__(self, "history_ref", _coerce_data_ref(self.history_ref))
        artifact_refs: dict[str, DataRef] = {}
        for key, value in dict(self.artifact_refs or {}).items():
            ref = _coerce_data_ref(value)
            if ref is None:
                raise TypeError(f"artifact_refs['{key}'] must be a DataRef")
            artifact_refs[str(key)] = ref
        object.__setattr__(self, "artifact_refs", MappingProxyType(artifact_refs))

    def as_dict(self) -> dict[str, Any]:
        return {
            **_protocol_header("solver_result"),
            "best_solution": (
                None
                if self.best_solution_ref is not None
                else _encode_shared_value(
                    self.best_solution,
                    path="solver_result.best_solution",
                )
            ),
            "best_solution_ref": (
                None
                if self.best_solution_ref is None
                else _encode_data_ref(self.best_solution_ref)
            ),
            "best_objectives": (
                None if self.best_objectives is None else self.best_objectives.tolist()
            ),
            "best_constraint_violation": self.best_constraint_violation,
            "best_candidate_token": self.best_candidate_token,
            "best_evaluation_id": self.best_evaluation_id,
            "best_provenance": _encode_shared_value(
                self.best_provenance,
                path="solver_result.best_provenance",
            ),
            "solve_status": self.solve_status,
            "termination_reason": self.termination_reason,
            "feasibility": self.feasibility,
            "quality": self.quality.as_dict(),
            "pareto_front": (
                None
                if self.pareto_front_ref is not None
                else _encode_shared_value(
                    self.pareto_front,
                    path="solver_result.pareto_front",
                )
            ),
            "pareto_front_ref": (
                None
                if self.pareto_front_ref is None
                else _encode_data_ref(self.pareto_front_ref)
            ),
            "history": (
                []
                if self.history_ref is not None
                else _encode_shared_value(self.history, path="solver_result.history")
            ),
            "history_ref": (
                None if self.history_ref is None else _encode_data_ref(self.history_ref)
            ),
            "report": _encode_shared_value(self.report, path="solver_result.report"),
            "metadata": _encode_shared_value(
                self.metadata,
                path="solver_result.metadata",
            ),
            "artifact_refs": {
                str(key): _encode_data_ref(value)
                for key, value in self.artifact_refs.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SolverResult":
        data = dict(payload or {})
        _validate_protocol_header(data, "solver_result")
        best_solution = _decode_shared_value(data.get("best_solution"))
        if isinstance(best_solution, Mapping) and "values" in best_solution:
            best_solution = UnknownState.from_protocol_payload(best_solution)
        pareto_front = _decode_shared_value(data.get("pareto_front"))
        return cls(
            best_solution=best_solution,
            best_objectives=(
                None
                if data.get("best_objectives") is None
                else np.asarray(data.get("best_objectives"), dtype=float)
            ),
            best_constraint_violation=(
                None
                if data.get("best_constraint_violation") is None
                else float(data.get("best_constraint_violation"))
            ),
            best_candidate_token=data.get("best_candidate_token"),
            best_evaluation_id=data.get("best_evaluation_id"),
            best_provenance=dict(
                _decode_shared_value(data.get("best_provenance", {})) or {}
            ),
            solve_status=str(data.get("solve_status", "unknown")),
            termination_reason=str(data.get("termination_reason", "unknown")),
            feasibility=str(data.get("feasibility", "unknown")),
            quality=SolveQuality.from_dict(data.get("quality")),
            pareto_front=pareto_front,
            history=tuple(_decode_shared_value(data.get("history", ())) or ()),
            report=dict(_decode_shared_value(data.get("report", {})) or {}),
            metadata=dict(_decode_shared_value(data.get("metadata", {})) or {}),
            best_solution_ref=_decode_shared_value(data.get("best_solution_ref")),
            pareto_front_ref=_decode_shared_value(data.get("pareto_front_ref")),
            history_ref=_decode_shared_value(data.get("history_ref")),
            artifact_refs={
                str(key): _decode_shared_value(value)
                for key, value in dict(data.get("artifact_refs", {}) or {}).items()
            },
        )


def _validate_solver_result_consistency(
    *,
    solve_status: str,
    feasibility: str,
    quality: SolveQuality,
    best_constraint_violation: float | None,
) -> None:
    rules = _SOLVER_STATUS_RULES[solve_status]
    forbidden_feasibility = rules.get("forbidden_feasibility", frozenset())
    if feasibility in forbidden_feasibility:
        raise ValueError(
            "inconsistent SolverResult terminal semantics: "
            f"solve_status='{solve_status}', feasibility='{feasibility}'"
        )
    best_is_feasible = (
        None
        if best_constraint_violation is None
        else float(best_constraint_violation) <= 0.0
    )
    if feasibility == "feasible" and best_is_feasible is False:
        raise ValueError(
            "a feasible SolverResult conflicts with a positive best constraint violation"
        )
    if feasibility == "infeasible" and best_is_feasible is True:
        raise ValueError(
            "an infeasible SolverResult conflicts with a feasible declared best"
        )
    if rules.get("forbids_infeasible_best") and best_is_feasible is False:
        raise ValueError(
            f"solve_status='{solve_status}' conflicts with a positive best constraint violation"
        )
    if rules.get("forbids_feasible_best") and best_is_feasible is True:
        raise ValueError(
            f"solve_status='{solve_status}' conflicts with a feasible declared best"
        )
    positive_gap = any(
        value is not None and float(value) > 0.0
        for value in (quality.absolute_gap, quality.relative_gap)
    )
    if quality.approximate is False and positive_gap:
        raise ValueError(
            "SolveQuality.approximate=False conflicts with a positive optimality gap"
        )
    if rules.get("forbids_approximate") and quality.approximate is True:
        raise ValueError(
            "solve_status='optimal' conflicts with SolveQuality.approximate=True"
        )
    if rules.get("forbids_positive_gap") and positive_gap:
        raise ValueError("solve_status='optimal' conflicts with a positive optimality gap")


__all__ = [
    "SHARED_TYPE_SCHEMA_VERSION",
    "SOLVE_STATUSES",
    "FEASIBILITY_STATUSES",
    "encode_shared_value",
    "decode_shared_value",
    "Feedback",
    "StateRef",
    "UnknownState",
    "CandidateBatch",
    "PopulationSnapshot",
    "TrainerResult",
    "SolveQuality",
    "SolverResult",
]
