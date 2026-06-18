"""
Context event system for tracking context changes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class ContextEvent:
    kind: str
    key: Optional[str]
    value: Any
    timestamp: float
    source: Optional[str] = None
    generation: Optional[int] = None
    step: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "key": self.key,
            "value": self.value,
            "timestamp": float(self.timestamp),
            "source": self.source,
            "generation": self.generation,
            "step": self.step,
        }


def record_context_event(
    context: Dict[str, Any],
    *,
    kind: str,
    key: Optional[str],
    value: Any,
    source: Optional[str] = None,
    generation: Optional[int] = None,
    step: Optional[int] = None,
    events_key: str = "context_events",
) -> ContextEvent:
    event = ContextEvent(
        kind=str(kind),
        key=key,
        value=value,
        timestamp=time.time(),
        source=source,
        generation=generation,
        step=step,
    )
    context.setdefault(events_key, []).append(event.to_dict())
    return event


def apply_context_event(context: Dict[str, Any], event: Mapping[str, Any]) -> None:
    kind = str(event.get("kind", "set"))
    key = event.get("key")
    value = event.get("value")

    if kind == "set":
        if key is not None:
            context[key] = value
        return
    if kind == "update":
        if isinstance(value, Mapping):
            context.update(dict(value))
        return
    if kind == "append":
        if key is not None:
            context.setdefault(key, []).append(value)
        return
    if kind == "extend":
        if key is not None and isinstance(value, Iterable):
            context.setdefault(key, []).extend(list(value))
        return
    if kind == "delete":
        if key is not None and key in context:
            del context[key]
        return

    raise ValueError(f"Unsupported context event kind: {kind}")


def replay_context(
    base_context: Mapping[str, Any],
    events: Iterable[Mapping[str, Any]],
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    ctx = dict(base_context)
    for event in events:
        try:
            apply_context_event(ctx, event)
        except Exception:
            if strict:
                raise
            continue
    return ctx


__all__ = [
    "ContextEvent",
    "record_context_event",
    "apply_context_event",
    "replay_context",
]