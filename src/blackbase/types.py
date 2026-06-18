"""
Shared protocol types used by nsgablack/mlblack bridge surfaces.

These objects stay small and numpy-oriented so they can move through Case
boundaries without importing either semantic framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np


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

    def __post_init__(self) -> None:
        if self.best_objectives is not None:
            object.__setattr__(self, "best_objectives", np.asarray(self.best_objectives, dtype=float))
        object.__setattr__(self, "history", tuple(self.history or ()))
        object.__setattr__(self, "report", dict(self.report or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


__all__ = [
    "Feedback",
    "UnknownState",
    "PopulationSnapshot",
    "TrainerResult",
]
