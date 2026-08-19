"""Deterministic callable binding with exactly-once execution semantics.

Framework extension points may intentionally accept more than one declared
call shape.  This module resolves a supported shape with
``inspect.signature`` before execution, so a ``TypeError`` raised by the
callable body is never mistaken for a signature mismatch and retried.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class CallCandidate:
    """One declared positional/keyword shape for a callable invocation."""

    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", dict(self.kwargs))
        object.__setattr__(self, "label", str(self.label or ""))


@dataclass(frozen=True)
class BoundCallOutcome:
    """Result plus the call candidate selected before execution."""

    value: Any
    candidate_index: int
    candidate_label: str = ""
    signature_available: bool = True


def invoke_bound_once_with_outcome(
    fn: Callable[..., Any],
    candidates: Sequence[CallCandidate],
) -> BoundCallOutcome:
    """Bind a declared candidate, then execute ``fn`` exactly once.

    For extension or builtin callables without an inspectable signature, the
    first candidate is the canonical shape and is invoked once.  Its original
    exception is allowed to propagate because retrying could duplicate side
    effects.
    """

    declared = tuple(candidates)
    if not declared:
        raise ValueError("at least one call candidate is required")
    if not all(isinstance(candidate, CallCandidate) for candidate in declared):
        raise TypeError("candidates must contain CallCandidate instances")

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        canonical = declared[0]
        value = fn(*canonical.args, **dict(canonical.kwargs))
        return BoundCallOutcome(
            value=value,
            candidate_index=0,
            candidate_label=canonical.label,
            signature_available=False,
        )

    for index, candidate in enumerate(declared):
        try:
            signature.bind(*candidate.args, **dict(candidate.kwargs))
        except TypeError:
            continue
        value = fn(*candidate.args, **dict(candidate.kwargs))
        return BoundCallOutcome(
            value=value,
            candidate_index=index,
            candidate_label=candidate.label,
            signature_available=True,
        )

    labels = ", ".join(
        candidate.label or f"candidate[{index}]"
        for index, candidate in enumerate(declared)
    )
    raise TypeError(f"{fn!r} cannot bind any declared call candidate: {labels}")


def invoke_bound_once(
    fn: Callable[..., Any],
    candidates: Sequence[CallCandidate],
) -> Any:
    """Return the value from :func:`invoke_bound_once_with_outcome`."""

    return invoke_bound_once_with_outcome(fn, candidates).value


__all__ = [
    "BoundCallOutcome",
    "CallCandidate",
    "invoke_bound_once",
    "invoke_bound_once_with_outcome",
]
