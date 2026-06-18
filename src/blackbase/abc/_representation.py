"""Representation abstract base class.

Unified interface for candidate solution encoding/decoding.

Core concept: a representation manages the lifecycle of candidate
solutions — creating them, decoding them into usable forms, and
optionally encoding, repairing, or mutating them.

All hooks are provided with sensible defaults; subclasses only need
to implement init() and decode().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Sequence, Tuple


class RepresentationBase(ABC):
    """Abstract base class for representation / codec systems.

    Core (must implement):
    - init(context) -> Any — create initial candidate state
    - decode(state, context) -> Any — decode candidate into usable form

    Optional hooks (all have defaults):
    - encode — reverse of decode (raises NotImplementedError by default)
    - repair — constraint repair (pass-through by default)
    - mutate — mutation operator (pass-through by default)
    - init_batch / decode_batch / repair_batch — batch variants
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
    def init(self, context: Mapping[str, Any]) -> Any:
        """Create an initial candidate state.

        Args:
            context: Current run context.

        Returns:
            An initial candidate state.
        """
        ...

    @abstractmethod
    def decode(self, state: Any, context: Mapping[str, Any]) -> Any:
        """Decode a candidate state into a usable form (model, function, etc.).

        Args:
            state: The candidate state to decode.
            context: Current run context.

        Returns:
            The decoded object (model, function, parameters, etc.).
        """
        ...

    # --- Optional hooks ---

    def encode(self, model: Any, context: Mapping[str, Any]) -> Any:
        """Encode a model back into a candidate state.

        Reverse of decode(). Not all representations support this.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.encode() is not implemented"
        )

    def repair(self, state: Any, context: Mapping[str, Any]) -> Any:
        """Repair a candidate state to satisfy constraints.

        Default: pass through without modification.
        """
        return state

    def mutate(self, state: Any, context: Mapping[str, Any]) -> Any:
        """Mutate a candidate state.

        Default: pass through without modification.
        """
        return state

    # --- Batch variants ---

    def init_batch(
        self, n: int, context: Mapping[str, Any] | None = None
    ) -> Tuple[Any, ...]:
        """Create a batch of initial candidate states."""
        ctx = dict(context or {})
        return tuple(self.init(ctx) for _ in range(int(n)))

    def decode_batch(
        self, states: Sequence[Any], context: Mapping[str, Any] | None = None
    ) -> Tuple[Any, ...]:
        """Decode a batch of candidate states."""
        ctx = dict(context or {})
        return tuple(self.decode(state, ctx) for state in tuple(states))

    def repair_batch(
        self, states: Sequence[Any], context: Mapping[str, Any] | None = None
    ) -> Tuple[Any, ...]:
        """Repair a batch of candidate states."""
        ctx = dict(context or {})
        return tuple(self.repair(state, ctx) for state in tuple(states))

    # --- Metadata ---

    def get_context_contract(self) -> Dict[str, Any]:
        """Return context contract metadata for this representation."""
        return {
            "requires": list(getattr(self, "context_requires", ()) or ()),
            "provides": list(getattr(self, "context_provides", ()) or ()),
            "mutates": list(getattr(self, "context_mutates", ()) or ()),
            "cache": list(getattr(self, "context_cache", ()) or ()),
            "notes": getattr(self, "context_notes", None),
        }

    def describe(self) -> Dict[str, Any]:
        """Return a human-readable description of this representation."""
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "contract": self.get_context_contract(),
        }

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.name})>"
