"""Adapter abstract base class.

Unified interface for algorithm adapters / optimizer adapters.

Core concept: an adapter proposes candidate solutions and consumes
evaluation feedback.  This is the "strategy layer" in the four-layer
orthogonal architecture.

All hooks are provided with sensible defaults; subclasses only need
to implement propose() and update().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..contracts import BatchDisposition


class AdapterBase(ABC):
    """Abstract base class for algorithm adapters / optimizer adapters.

    Core (must implement):
    - propose(control, context) -> Sequence[Any]
    - update(control, candidates, feedback, context) -> None

    Optional hooks (all have defaults):
    - setup / teardown — lifecycle
    - get_state / set_state — checkpoint
    - coerce_candidates — normalize propose() output
    - get_runtime_context_projection — runtime context exposure
    - set_population — population write-back
    - validate_population_snapshot — snapshot validation
    - create_local_rng — component-local RNG
    """

    name: str = ""
    priority: int = 0

    # Context contract metadata
    context_requires: tuple = ()
    context_provides: tuple = ()
    context_mutates: tuple = ()
    context_cache: tuple = ()
    context_notes: str | None = None

    # --- Core (abstract) ---

    @abstractmethod
    def propose(self, control: Any, context: Mapping[str, Any]) -> Sequence[Any]:
        """Return candidate solutions for evaluation.

        Args:
            control: The control plane (solver or trainer).
            context: Current run context.

        Returns:
            Sequence of candidate solutions.
        """
        ...

    @abstractmethod
    def update(
        self,
        control: Any,
        candidates: Sequence[Any],
        feedback: Any,
        context: Mapping[str, Any],
    ) -> None:
        """Consume evaluation feedback for candidates.

        Args:
            control: The control plane (solver or trainer).
            candidates: The candidates that were evaluated.
            feedback: Evaluation results (objectives+violations or Feedback).
            context: Current run context.
        """
        ...

    # --- Lifecycle hooks ---

    def setup(self, control: Any) -> None:
        """Called once before the run starts."""
        return None

    def teardown(self, control: Any) -> None:
        """Called once after the run ends."""
        return None

    def on_proposal_disposition(
        self,
        control: Any,
        disposition: BatchDisposition,
        context: Mapping[str, Any],
    ) -> None:
        """Reconcile proposal-owned state after partial batch admission.

        Stateful adapters that retain one pending record per proposed item
        should project that state through ``disposition.accepted_indices``.
        Stateless adapters can keep the default no-op implementation.
        """
        del control, disposition, context

    # --- State persistence ---

    def get_state(self) -> Dict[str, Any]:
        """Return serializable state for checkpointing."""
        return {}

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore state from a checkpoint."""
        return None

    # --- Candidate normalization ---

    @staticmethod
    def coerce_candidates(value: Any) -> List[Any]:
        """Normalize propose() output to a list.

        Handles None, single items, numpy arrays, and iterables.
        """
        if value is None:
            return []
        if isinstance(value, Sequence) and not hasattr(value, "__array__"):
            return list(value)
        # numpy array or similar
        if hasattr(value, "ndim"):
            if value.ndim <= 1:
                return [value]
            return [value[i] for i in range(len(value))]
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    # --- Context projection ---

    def get_runtime_context_projection(self, control: Any) -> Mapping[str, Any]:
        """Return runtime fields to expose in control plane context.
        """
        del control
        return {}

    def get_runtime_context_projection_sources(
        self,
        control: Any,
    ) -> Mapping[str, str]:
        """Return writer attribution for projected runtime fields."""
        del control
        return {}

    # --- Population write-back ---

    def set_population(
        self, population: Any, objectives: Any, violations: Any
    ) -> bool:
        """Optional population write-back contract.

        Adapters that own population/objective state should override this
        and return True when write-back succeeds. Default returns False.
        """
        return False

    @staticmethod
    def validate_population_snapshot(
        population: Any, objectives: Any, violations: Any
    ) -> tuple:
        """Normalize and validate population snapshot payload.

        Returns (population, objectives, violations) as validated arrays.
        Default implementation passes through without validation;
        frameworks can override for strict validation.
        """
        return population, objectives, violations

    # --- RNG ---

    def create_local_rng(
        self, seed: Optional[int] = None, control: Any = None
    ) -> Any:
        """Create a component-local RNG.

        Priority:
        1) explicit seed
        2) control.fork_rng() if available
        3) independent default RNG
        """
        if seed is not None:
            import random
            return random.Random(int(seed))
        if control is not None:
            fork = getattr(control, "fork_rng", None)
            if callable(fork):
                try:
                    rng = fork(self.name)
                    if rng is not None:
                        return rng
                except Exception:
                    pass
        import random
        return random.Random()

    # --- Metadata ---

    def get_context_contract(self) -> Dict[str, Any]:
        """Return context contract metadata for this adapter."""
        return {
            "requires": list(getattr(self, "context_requires", ()) or ()),
            "provides": list(getattr(self, "context_provides", ()) or ()),
            "mutates": list(getattr(self, "context_mutates", ()) or ()),
            "cache": list(getattr(self, "context_cache", ()) or ()),
            "notes": getattr(self, "context_notes", None),
        }

    def describe(self) -> Dict[str, Any]:
        """Return a human-readable description of this adapter."""
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "contract": self.get_context_contract(),
        }

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.name})>"
