"""Serializable deadline and cooperative-cancellation authority."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol
from uuid import uuid4


@dataclass(frozen=True)
class TerminationPolicy:
    """Serializable escalation policy for one standard Case execution.

    ``cooperative`` only propagates cancellation/deadline state.  The
    ``cooperative_then_terminate`` mode additionally requires an isolated
    execution boundary and lets its supervisor terminate that boundary after
    the configured grace period.
    """

    mode: str = "cooperative"
    grace_seconds: float = 5.0
    kill_grace_seconds: float = 1.0
    poll_interval_seconds: float = 0.05

    def __post_init__(self) -> None:
        mode = str(self.mode or "cooperative").strip().lower()
        if mode not in {"cooperative", "cooperative_then_terminate"}:
            raise ValueError(
                "termination mode must be cooperative or cooperative_then_terminate"
            )
        grace = float(self.grace_seconds)
        kill_grace = float(self.kill_grace_seconds)
        poll = float(self.poll_interval_seconds)
        if grace < 0 or kill_grace < 0:
            raise ValueError("termination grace periods must be non-negative")
        if poll <= 0:
            raise ValueError("termination poll_interval_seconds must be positive")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "grace_seconds", grace)
        object.__setattr__(self, "kill_grace_seconds", kill_grace)
        object.__setattr__(self, "poll_interval_seconds", poll)

    @property
    def requires_isolation(self) -> bool:
        return self.mode == "cooperative_then_terminate"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "TerminationPolicy":
        data = dict(payload or {})
        return cls(
            mode=str(data.get("mode", "cooperative")),
            grace_seconds=float(data.get("grace_seconds", 5.0) or 0.0),
            kill_grace_seconds=float(data.get("kill_grace_seconds", 1.0) or 0.0),
            poll_interval_seconds=float(data.get("poll_interval_seconds", 0.05) or 0.05),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "grace_seconds": self.grace_seconds,
            "kill_grace_seconds": self.kill_grace_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
        }


@dataclass(frozen=True)
class CancellationRef:
    """Transport-safe reference to one cancellation state."""

    control_id: str = field(default_factory=lambda: f"control-{uuid4().hex}")
    backend: str = "memory"
    namespace: str = "blackbase"
    path: str = ""
    redis_url_env: str = "BLACKBASE_REDIS_URL"
    deadline_at: float = 0.0
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        backend = str(self.backend or "memory").strip().lower()
        if backend not in {"memory", "sqlite", "redis"}:
            raise ValueError("cancellation backend must be memory, sqlite, or redis")
        control_id = str(self.control_id or "").strip()
        if not control_id:
            raise ValueError("control_id must be non-empty")
        if backend == "sqlite" and not str(self.path or "").strip():
            raise ValueError("sqlite cancellation requires a database path")
        object.__setattr__(self, "control_id", control_id)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "namespace", str(self.namespace or "blackbase"))
        object.__setattr__(self, "path", str(self.path or ""))
        object.__setattr__(self, "redis_url_env", str(self.redis_url_env or "BLACKBASE_REDIS_URL"))
        object.__setattr__(self, "deadline_at", max(0.0, float(self.deadline_at or 0.0)))
        object.__setattr__(self, "created_at", float(self.created_at or time.time()))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CancellationRef":
        return cls(
            control_id=str(payload.get("control_id", "")),
            backend=str(payload.get("backend", "memory")),
            namespace=str(payload.get("namespace", "blackbase")),
            path=str(payload.get("path", "")),
            redis_url_env=str(payload.get("redis_url_env", "BLACKBASE_REDIS_URL")),
            deadline_at=float(payload.get("deadline_at", 0.0) or 0.0),
            created_at=float(payload.get("created_at", 0.0) or 0.0),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "control_id": self.control_id,
            "backend": self.backend,
            "namespace": self.namespace,
            "path": self.path,
            "redis_url_env": self.redis_url_env,
            "deadline_at": self.deadline_at,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class CancellationState:
    requested: bool = False
    reason: str = ""
    requested_at: float = 0.0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "CancellationState":
        data = dict(payload or {})
        return cls(
            requested=bool(data.get("requested", False)),
            reason=str(data.get("reason", "")),
            requested_at=float(data.get("requested_at", 0.0) or 0.0),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": bool(self.requested),
            "reason": str(self.reason),
            "requested_at": float(self.requested_at),
        }


class CancellationRequested(RuntimeError):
    """Raised at a Case boundary after cooperative cancellation."""


class CaseDeadlineExceeded(TimeoutError):
    """Raised when an absolute Case deadline has passed."""


class CancellationStore(Protocol):
    def create(self, ref: CancellationRef) -> None: ...

    def read(self, ref: CancellationRef) -> CancellationState: ...

    def request(self, ref: CancellationRef, *, reason: str) -> bool: ...


class InMemoryCancellationStore:
    def __init__(self) -> None:
        self._states: dict[tuple[str, str], CancellationState] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(ref: CancellationRef) -> tuple[str, str]:
        return ref.namespace, ref.control_id

    def create(self, ref: CancellationRef) -> None:
        with self._lock:
            self._states.setdefault(self._key(ref), CancellationState())

    def read(self, ref: CancellationRef) -> CancellationState:
        with self._lock:
            return self._states.get(self._key(ref), CancellationState())

    def request(self, ref: CancellationRef, *, reason: str) -> bool:
        with self._lock:
            key = self._key(ref)
            current = self._states.get(key, CancellationState())
            if current.requested:
                return False
            self._states[key] = CancellationState(
                requested=True,
                reason=str(reason or "cancelled"),
                requested_at=time.time(),
            )
            return True


class SQLiteCancellationStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cancellation_controls (
                    namespace TEXT NOT NULL,
                    control_id TEXT NOT NULL,
                    requested INTEGER NOT NULL DEFAULT 0,
                    reason TEXT NOT NULL DEFAULT '',
                    requested_at REAL NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (namespace, control_id)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def create(self, ref: CancellationRef) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO cancellation_controls
                    (namespace, control_id, created_at)
                VALUES (?, ?, ?)
                """,
                (ref.namespace, ref.control_id, ref.created_at),
            )

    def read(self, ref: CancellationRef) -> CancellationState:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT requested, reason, requested_at
                FROM cancellation_controls
                WHERE namespace = ? AND control_id = ?
                """,
                (ref.namespace, ref.control_id),
            ).fetchone()
        if row is None:
            return CancellationState()
        return CancellationState(bool(row[0]), str(row[1]), float(row[2]))

    def request(self, ref: CancellationRef, *, reason: str) -> bool:
        self.create(ref)
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE cancellation_controls
                SET requested = 1, reason = ?, requested_at = ?
                WHERE namespace = ? AND control_id = ? AND requested = 0
                """,
                (str(reason or "cancelled"), time.time(), ref.namespace, ref.control_id),
            ).rowcount
        return updated == 1


class RedisCancellationStore:
    def __init__(
        self,
        redis_url: str,
        *,
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                import redis
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("redis cancellation backend requires the redis package") from exc
            client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.client = client

    @staticmethod
    def _key(ref: CancellationRef) -> str:
        return f"blackbase:{ref.namespace}:control:{ref.control_id}"

    def create(self, ref: CancellationRef) -> None:
        self.client.setnx(self._key(ref), json.dumps(CancellationState().as_dict()))

    def read(self, ref: CancellationRef) -> CancellationState:
        raw = self.client.get(self._key(ref))
        if raw is None:
            return CancellationState()
        return CancellationState.from_dict(json.loads(_decode_redis_value(raw)))

    def request(self, ref: CancellationRef, *, reason: str) -> bool:
        self.create(ref)
        key = self._key(ref)
        while True:
            with self.client.pipeline() as pipe:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    current = CancellationState.from_dict(
                        json.loads(_decode_redis_value(raw))
                    )
                    if current.requested:
                        pipe.unwatch()
                        return False
                    state = CancellationState(True, str(reason or "cancelled"), time.time())
                    pipe.multi()
                    pipe.set(key, json.dumps(state.as_dict()))
                    pipe.execute()
                    return True
                except Exception as exc:
                    if type(exc).__name__ != "WatchError":
                        raise


def _decode_redis_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


_MEMORY_STORES: dict[str, InMemoryCancellationStore] = {}
_MEMORY_LOCK = threading.RLock()


def build_cancellation_store(
    ref: CancellationRef,
    *,
    redis_client: Any = None,
) -> CancellationStore:
    if ref.backend == "sqlite":
        return SQLiteCancellationStore(ref.path)
    if ref.backend == "redis":
        redis_url = str(os.environ.get(ref.redis_url_env, "") or "").strip()
        if redis_client is None and not redis_url:
            raise RuntimeError(
                f"redis cancellation backend requires environment variable {ref.redis_url_env}"
            )
        return RedisCancellationStore(redis_url or "redis://localhost:6379/0", client=redis_client)
    with _MEMORY_LOCK:
        return _MEMORY_STORES.setdefault(ref.namespace, InMemoryCancellationStore())


class CancellationToken:
    """Runtime view over a serializable cancellation reference."""

    def __init__(
        self,
        ref: CancellationRef,
        *,
        store: CancellationStore | None = None,
        redis_client: Any = None,
    ) -> None:
        self.ref = ref
        self.store = store or build_cancellation_store(ref, redis_client=redis_client)
        self.store.create(ref)

    @property
    def state(self) -> CancellationState:
        return self.store.read(self.ref)

    @property
    def cancelled(self) -> bool:
        return bool(self.state.requested)

    @property
    def deadline_exceeded(self) -> bool:
        return self.ref.deadline_at > 0 and time.time() >= self.ref.deadline_at

    def remaining_seconds(self) -> float | None:
        if self.ref.deadline_at <= 0:
            return None
        return max(0.0, self.ref.deadline_at - time.time())

    def cancel(self, reason: str = "cancelled") -> bool:
        return self.store.request(self.ref, reason=str(reason or "cancelled"))

    def checkpoint(self) -> None:
        if self.deadline_exceeded:
            self.cancel("case deadline exceeded")
            raise CaseDeadlineExceeded("case deadline exceeded")
        state = self.state
        if state.requested:
            raise CancellationRequested(state.reason or "case cancellation requested")


__all__ = [
    "CancellationRef",
    "CancellationRequested",
    "CancellationState",
    "CancellationStore",
    "CancellationToken",
    "CaseDeadlineExceeded",
    "InMemoryCancellationStore",
    "RedisCancellationStore",
    "SQLiteCancellationStore",
    "TerminationPolicy",
    "build_cancellation_store",
]
