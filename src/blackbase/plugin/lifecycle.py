"""Stable lifecycle slot names shared by runtime controllers and plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


ATTEMPT_START = "attempt_start"
ATTEMPT_END = "attempt_end"
GENERATION_START = "generation_start"
GENERATION_COMMITTED = "generation_committed"
GENERATION_END = "generation_end"

RUNTIME_LIFECYCLE_SLOTS = frozenset(
    {
        ATTEMPT_START,
        ATTEMPT_END,
        GENERATION_START,
        GENERATION_COMMITTED,
        GENERATION_END,
    }
)


@dataclass(frozen=True)
class PluginLifecycleReceipt:
    """Exact Plugin participants that completed one lifecycle start hook.

    The receipt is intentionally process-local: it carries the live Plugin
    instances that must receive the matching end notification even when a
    later start participant fails or changes its enabled flag.
    """

    start_event: str
    participants: tuple[Any, ...] = ()

    @classmethod
    def capture(
        cls,
        start_event: str,
        participants: Sequence[Any],
    ) -> "PluginLifecycleReceipt":
        return cls(
            start_event=str(start_event),
            participants=tuple(participants),
        )

    @property
    def participant_names(self) -> tuple[str, ...]:
        return tuple(
            str(getattr(participant, "name", type(participant).__name__))
            for participant in self.participants
        )


class PluginLifecycleDispatchError(RuntimeError):
    """A lifecycle start hook failed after a receipt was partially built."""

    def __init__(
        self,
        *,
        event_name: str,
        plugin_name: str,
        receipt: PluginLifecycleReceipt,
        cause: BaseException,
    ) -> None:
        self.event_name = str(event_name)
        self.plugin_name = str(plugin_name)
        self.receipt = receipt
        self.cause = cause
        super().__init__(
            f"Plugin '{self.plugin_name}' failed in lifecycle start "
            f"'{self.event_name}': {cause}"
        )


class PluginLifecycleCleanupError(RuntimeError):
    """One or more matching lifecycle end hooks failed after all ran."""

    def __init__(
        self,
        *,
        event_name: str,
        errors: Sequence[tuple[str, BaseException]],
    ) -> None:
        self.event_name = str(event_name)
        self.errors = tuple(errors)
        summary = "; ".join(
            f"{name}: {type(exc).__name__}: {exc}"
            for name, exc in self.errors
        )
        super().__init__(
            f"plugin lifecycle cleanup '{self.event_name}' failed: {summary}"
        )


def normalize_lifecycle_slot(value: str) -> str:
    """Return one canonical lifecycle slot or fail closed."""

    slot = str(value or "").strip().lower()
    if slot not in RUNTIME_LIFECYCLE_SLOTS:
        raise ValueError(f"unsupported runtime lifecycle slot: {slot or '<empty>'}")
    return slot


__all__ = [
    "ATTEMPT_END",
    "ATTEMPT_START",
    "GENERATION_COMMITTED",
    "GENERATION_END",
    "GENERATION_START",
    "PluginLifecycleCleanupError",
    "PluginLifecycleDispatchError",
    "PluginLifecycleReceipt",
    "RUNTIME_LIFECYCLE_SLOTS",
    "normalize_lifecycle_slot",
]
