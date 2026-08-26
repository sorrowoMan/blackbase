"""
Context event system for tracking context changes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .value_isolation import detach_context_value


CONTEXT_EVENT_KINDS = frozenset({"set", "update", "append", "extend", "delete"})


@dataclass(frozen=True)
class ContextEvent:
    kind: str
    key: Optional[str]
    value: Any
    timestamp: float
    source: Optional[str] = None
    generation: Optional[int] = None
    step: Optional[int] = None

    def __post_init__(self) -> None:
        kind = str(self.kind or "").strip().lower()
        if kind not in CONTEXT_EVENT_KINDS:
            raise ValueError(
                f"unsupported context event kind {kind!r}; "
                f"expected one of {sorted(CONTEXT_EVENT_KINDS)}"
            )
        key = None if self.key is None else str(self.key)
        if kind in {"set", "append", "extend", "delete"} and key is None:
            raise ValueError(f"context event kind={kind!r} requires a key")
        value = detach_context_value(
            self.value,
            path=f"context_event.{kind}.value",
        )
        if kind == "update" and not isinstance(value, Mapping):
            raise TypeError("context update event value must be a Mapping")
        if kind == "extend" and (
            not isinstance(value, Iterable)
            or isinstance(value, (str, bytes, bytearray, Mapping))
        ):
            raise TypeError("context extend event value must be a non-mapping iterable")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "value", value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "key": self.key,
            "value": detach_context_value(
                self.value,
                path=f"context_event.{self.kind}.payload",
            ),
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
    normalized = ContextEvent(
        kind=str(event.get("kind", "set")),
        key=event.get("key"),
        value=event.get("value"),
        timestamp=float(event.get("timestamp", time.time())),
        source=event.get("source"),
        generation=event.get("generation"),
        step=event.get("step"),
    )
    kind = normalized.kind
    key = normalized.key
    value = normalized.value

    if kind == "set":
        assert key is not None
        context[key] = value
        return
    if kind == "update":
        if isinstance(value, Mapping):
            context.update(dict(value))
        return
    if kind == "append":
        assert key is not None
        context.setdefault(key, []).append(value)
        return
    if kind == "extend":
        assert key is not None
        context.setdefault(key, []).extend(list(value))
        return
    if kind == "delete":
        assert key is not None
        if key in context:
            del context[key]
        return

    raise ValueError(f"Unsupported context event kind: {kind}")


def replay_context(
    base_context: Mapping[str, Any],
    events: Iterable[Mapping[str, Any]],
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    detached = detach_context_value(
        dict(base_context),
        path="context_replay.base",
    )
    if not isinstance(detached, dict):  # pragma: no cover - Mapping guarantees this
        raise TypeError("detached context replay base must be a dict")
    ctx = detached
    for event in events:
        try:
            apply_context_event(ctx, event)
        except Exception:
            if strict:
                raise
            continue
    return ctx


__all__ = [
    "CONTEXT_EVENT_KINDS",
    "ContextEvent",
    "record_context_event",
    "apply_context_event",
    "replay_context",
]
