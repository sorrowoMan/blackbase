"""Problem abstract base class.

Unified interface for evaluation / problem definitions.

Core concept: a problem evaluates a candidate and returns feedback.
The evaluation itself is immutable — bias and preference belong in
the Bias layer, not here.

All hooks are provided with sensible defaults; subclasses only need
to implement evaluate().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping


class ProblemBase(ABC):
    """Abstract base class for problem / evaluation definitions.

    Core (must implement):
    - evaluate(candidate, context) -> Any

    Optional hooks (all have defaults):
    - evaluate_constraints — constraint violation check
    - is_valid — bounds + constraint validity check
    """

    name: str = ""

    # Context contract metadata
    context_requires: tuple = ()
    context_provides: tuple = ()
    context_mutates: tuple = ()
    context_cache: tuple = ()
    context_notes: str | None = None

    # --- Core (abstract) ---

    @abstractmethod
    def evaluate(self, candidate: Any, context: Mapping[str, Any] | None = None) -> Any:
        """Evaluate a candidate and return feedback.

        Args:
            candidate: The candidate to evaluate (ndarray, model, state, etc.).
            context: Current run context (optional).

        Returns:
            Evaluation result. Type depends on framework:
            - nsgablack: ndarray of objective values
            - mlblack: Feedback object
        """
        ...

    # --- Optional hooks ---

    def evaluate_constraints(self, candidate: Any) -> Any:
        """Return constraint violation values.

        Convention: g(x) <= 0 means satisfied, g(x) > 0 means violated.
        Default: no constraints (returns empty).
        """
        return None

    def is_valid(self, candidate: Any) -> bool:
        """Check if a candidate satisfies bounds and constraints.

        Default: always True. Frameworks can override for strict checking.
        """
        return True

    # --- Metadata ---

    def get_context_contract(self) -> Dict[str, Any]:
        """Return context contract metadata for this problem."""
        return {
            "requires": list(getattr(self, "context_requires", ()) or ()),
            "provides": list(getattr(self, "context_provides", ()) or ()),
            "mutates": list(getattr(self, "context_mutates", ()) or ()),
            "cache": list(getattr(self, "context_cache", ()) or ()),
            "notes": getattr(self, "context_notes", None),
        }

    def describe(self) -> Dict[str, Any]:
        """Return a human-readable description of this problem."""
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "contract": self.get_context_contract(),
        }

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.name})>"
