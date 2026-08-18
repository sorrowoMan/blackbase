"""Bias abstract base class.

Unified interface for preference expression / soft adjustment.

Core concept: bias expresses preference without changing evaluation.
It is the minimal unit for context intervention — it injects preference
signals into context, and optionally adjusts feedback in multi-objective
scenarios.

Important distinction:
- Bias = preference expression (does NOT change evaluation)
- Problem = evaluation definition (including loss weighting)
- Plugin/Event = control logic (enable/disable, switch strategies)

All hooks are provided with sensible defaults; subclasses override
as needed.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Mapping, Sequence


class BiasBase(ABC):
    """Abstract base class for bias / soft-preference systems.

    Core concept: bias injects preference signals into context.
    It does NOT change the evaluation itself — that belongs to Problem.

    Optional hooks (all have defaults):
    - project_context — inject preference signals into context
    - adjust_feedback — project evaluated feedback for selection/update
    - weight/enable management
    """

    name: str = ""
    weight: float = 1.0
    enabled: bool = True

    # Context contract metadata
    context_requires: tuple = ()
    context_provides: tuple = ()
    context_mutates: tuple = ()
    context_cache: tuple = ()
    context_notes: str | None = None

    def __init__(
        self,
        name: str | None = None,
        *,
        weight: float = 1.0,
        enabled: bool = True,
    ) -> None:
        if name is not None:
            self.name = str(name)
        self.weight = float(weight)
        self.enabled = bool(enabled)

    # --- Core: preference signal injection ---

    def project_context(
        self,
        control: Any,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Inject preference signals into context.

        This is the primary mechanism for bias: express "what I prefer"
        without changing evaluation. Adapters read these signals and
        adjust their behavior accordingly.

        Default: pass through without modification.
        """
        del control
        return dict(context)

    # --- Optional: preference adjustment ---

    def adjust_feedback(
        self,
        control: Any,
        candidates: Sequence[Any],
        feedback: Sequence[Any],
        context: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        """Project evaluated feedback for selection or adapter update.

        Only meaningful in multi-objective scenarios where multiple
        equally-valid solutions exist (Pareto front). Does NOT change
        the Problem evaluation itself — only the feedback consumed by the
        selection policy.

        Default: pass through without modification.
        """
        del control, candidates, context
        return tuple(feedback)

    # --- Weight / enable management ---

    def get_weight(self) -> float:
        """Return current bias weight."""
        return self.weight

    def set_weight(self, weight: float) -> None:
        """Set bias weight (non-negative)."""
        self.weight = max(0.0, weight)

    def enable(self) -> None:
        """Enable this bias."""
        self.enabled = True

    def disable(self) -> None:
        """Disable this bias."""
        self.enabled = False

    def get_name(self) -> str:
        """Return bias name."""
        return self.name

    # --- Metadata ---

    def get_context_contract(self) -> Dict[str, Any]:
        """Return context contract metadata for this bias."""
        return {
            "requires": list(getattr(self, "context_requires", ()) or ()),
            "provides": list(getattr(self, "context_provides", ()) or ()),
            "mutates": list(getattr(self, "context_mutates", ()) or ()),
            "cache": list(getattr(self, "context_cache", ()) or ()),
            "notes": getattr(self, "context_notes", None),
        }

    def describe(self) -> Dict[str, Any]:
        """Return a human-readable description of this bias."""
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "contract": self.get_context_contract(),
        }

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.name})>"
