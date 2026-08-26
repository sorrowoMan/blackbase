"""Serializable deadline and cooperative-cancellation authority."""

from __future__ import annotations

import json
import hashlib
import math
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol
from uuid import uuid4


MAX_CANCELLATION_LINEAGE_DEPTH = 256


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
    """Serializable cancellation descriptor.

    ``memory`` descriptors are process-local.  SQLite/Redis descriptors are
    transport authorities and persist their immediate parent relation so one
    fixed-size child reference can observe cancellation of any ancestor.
    """

    control_id: str = field(default_factory=lambda: f"control-{uuid4().hex}")
    backend: str = "memory"
    namespace: str = "blackbase"
    path: str = ""
    redis_url_env: str = "BLACKBASE_REDIS_URL"
    deadline_at: float = 0.0
    created_at: float = field(default_factory=time.time)
    parent_control_id: str | None = None
    root_control_id: str | None = None
    lineage_depth: int = 0
    lineage_digest: str = ""
    active_ttl_seconds: float = 0.0
    heartbeat_seconds: float = 0.0
    retention_seconds: float = 0.0

    def __post_init__(self) -> None:
        backend = str(self.backend or "memory").strip().lower()
        if backend not in {"memory", "sqlite", "redis"}:
            raise ValueError("cancellation backend must be memory, sqlite, or redis")
        control_id = str(self.control_id or "").strip()
        if not control_id:
            raise ValueError("control_id must be non-empty")
        if backend == "sqlite" and not str(self.path or "").strip():
            raise ValueError("sqlite cancellation requires a database path")
        parent_control_id = (
            None
            if self.parent_control_id is None
            else str(self.parent_control_id).strip() or None
        )
        if parent_control_id == control_id:
            raise ValueError("cancellation parent_control_id must differ from control_id")
        lineage_depth = int(self.lineage_depth)
        if lineage_depth < 0 or lineage_depth > MAX_CANCELLATION_LINEAGE_DEPTH:
            raise ValueError(
                "cancellation lineage_depth must be between 0 and "
                f"{MAX_CANCELLATION_LINEAGE_DEPTH}"
            )
        if lineage_depth == 0 and parent_control_id is not None:
            raise ValueError("root cancellation cannot declare a parent_control_id")
        if lineage_depth > 0 and parent_control_id is None:
            raise ValueError("child cancellation requires parent_control_id")
        root_control_id = str(self.root_control_id or control_id).strip()
        if not root_control_id:
            raise ValueError("root_control_id must be non-empty")
        lineage_digest = str(self.lineage_digest or "").strip().lower()
        if not lineage_digest:
            lineage_digest = hashlib.sha256(
                f"root:{root_control_id}".encode("utf-8")
            ).hexdigest()
        if len(lineage_digest) != 64 or any(
            char not in "0123456789abcdef" for char in lineage_digest
        ):
            raise ValueError("lineage_digest must be a 64-character SHA-256 hex digest")
        active_ttl = float(self.active_ttl_seconds or 0.0)
        heartbeat = float(self.heartbeat_seconds or 0.0)
        retention = float(self.retention_seconds or 0.0)
        if active_ttl < 0 or heartbeat < 0 or retention < 0:
            raise ValueError("cancellation TTL, heartbeat, and retention must be non-negative")
        if active_ttl == 0 and heartbeat > 0:
            raise ValueError("cancellation heartbeat requires a positive active TTL")
        if active_ttl > 0 and heartbeat <= 0:
            raise ValueError("cancellation active TTL requires a positive heartbeat")
        if active_ttl > 0 and heartbeat * 2 > active_ttl:
            raise ValueError(
                "cancellation heartbeat must be no greater than active_ttl_seconds / 2"
            )
        object.__setattr__(self, "control_id", control_id)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "namespace", str(self.namespace or "blackbase"))
        object.__setattr__(self, "path", str(self.path or ""))
        object.__setattr__(self, "redis_url_env", str(self.redis_url_env or "BLACKBASE_REDIS_URL"))
        object.__setattr__(self, "deadline_at", max(0.0, float(self.deadline_at or 0.0)))
        object.__setattr__(self, "created_at", float(self.created_at or time.time()))
        object.__setattr__(self, "parent_control_id", parent_control_id)
        object.__setattr__(self, "root_control_id", root_control_id)
        object.__setattr__(self, "lineage_depth", lineage_depth)
        object.__setattr__(self, "lineage_digest", lineage_digest)
        object.__setattr__(self, "active_ttl_seconds", active_ttl)
        object.__setattr__(self, "heartbeat_seconds", heartbeat)
        object.__setattr__(self, "retention_seconds", retention)

    @property
    def process_local(self) -> bool:
        return self.backend == "memory"

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
            parent_control_id=payload.get("parent_control_id"),
            root_control_id=payload.get("root_control_id"),
            lineage_depth=int(payload.get("lineage_depth", 0) or 0),
            lineage_digest=str(payload.get("lineage_digest", "") or ""),
            active_ttl_seconds=float(payload.get("active_ttl_seconds", 0.0) or 0.0),
            heartbeat_seconds=float(payload.get("heartbeat_seconds", 0.0) or 0.0),
            retention_seconds=float(payload.get("retention_seconds", 0.0) or 0.0),
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
            "parent_control_id": self.parent_control_id,
            "root_control_id": self.root_control_id,
            "lineage_depth": self.lineage_depth,
            "lineage_digest": self.lineage_digest,
            "active_ttl_seconds": self.active_ttl_seconds,
            "heartbeat_seconds": self.heartbeat_seconds,
            "retention_seconds": self.retention_seconds,
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

    def touch(self, ref: CancellationRef) -> bool: ...

    def retire(self, ref: CancellationRef) -> None: ...


class InMemoryCancellationStore:
    def __init__(self) -> None:
        self._states: dict[tuple[str, str], CancellationState] = {}
        self._parents: dict[tuple[str, str], str | None] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(ref: CancellationRef) -> tuple[str, str]:
        return ref.namespace, ref.control_id

    def create(self, ref: CancellationRef) -> None:
        with self._lock:
            key = self._key(ref)
            existing_parent = self._parents.get(key, ref.parent_control_id)
            if key in self._parents and existing_parent != ref.parent_control_id:
                raise RuntimeError("cancellation control parent lineage changed")
            self._states.setdefault(key, CancellationState())
            self._parents.setdefault(key, ref.parent_control_id)

    def read(self, ref: CancellationRef) -> CancellationState:
        with self._lock:
            control_id: str | None = ref.control_id
            visited: set[str] = set()
            while control_id is not None:
                if control_id in visited:
                    raise RuntimeError("cancellation lineage contains a cycle")
                visited.add(control_id)
                if len(visited) > MAX_CANCELLATION_LINEAGE_DEPTH + 1:
                    raise RuntimeError("cancellation lineage exceeds the supported depth")
                key = (ref.namespace, control_id)
                state = self._states.get(key, CancellationState())
                if state.requested:
                    return state
                control_id = self._parents.get(key)
            return CancellationState()

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

    def touch(self, ref: CancellationRef) -> bool:
        with self._lock:
            return self._key(ref) in self._states

    def retire(self, ref: CancellationRef) -> None:
        with self._lock:
            key = self._key(ref)
            self._states.pop(key, None)
            self._parents.pop(key, None)


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
                    parent_control_id TEXT,
                    root_control_id TEXT,
                    lineage_depth INTEGER NOT NULL DEFAULT 0,
                    lineage_digest TEXT NOT NULL DEFAULT '',
                    retired_at REAL NOT NULL DEFAULT 0,
                    expires_at REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (namespace, control_id)
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(cancellation_controls)"
                ).fetchall()
            }
            migrations = {
                "parent_control_id": "ALTER TABLE cancellation_controls ADD COLUMN parent_control_id TEXT",
                "root_control_id": "ALTER TABLE cancellation_controls ADD COLUMN root_control_id TEXT",
                "lineage_depth": "ALTER TABLE cancellation_controls ADD COLUMN lineage_depth INTEGER NOT NULL DEFAULT 0",
                "lineage_digest": "ALTER TABLE cancellation_controls ADD COLUMN lineage_digest TEXT NOT NULL DEFAULT ''",
                "retired_at": "ALTER TABLE cancellation_controls ADD COLUMN retired_at REAL NOT NULL DEFAULT 0",
                "expires_at": "ALTER TABLE cancellation_controls ADD COLUMN expires_at REAL NOT NULL DEFAULT 0",
            }
            for name, statement in migrations.items():
                if name not in columns:
                    connection.execute(statement)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @staticmethod
    def _active_expires_at(ref: CancellationRef) -> float:
        ttl = float(ref.active_ttl_seconds)
        return time.time() + ttl if ttl > 0 else 0.0

    @staticmethod
    def _purge_expired(connection: sqlite3.Connection) -> None:
        connection.execute(
            "DELETE FROM cancellation_controls WHERE expires_at > 0 AND expires_at <= ?",
            (time.time(),),
        )

    def create(self, ref: CancellationRef) -> None:
        with self._connect() as connection:
            self._purge_expired(connection)
            expires_at = self._active_expires_at(ref)
            connection.execute(
                """
                INSERT OR IGNORE INTO cancellation_controls
                    (namespace, control_id, created_at, parent_control_id,
                     root_control_id, lineage_depth, lineage_digest, retired_at,
                     expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    ref.namespace,
                    ref.control_id,
                    ref.created_at,
                    ref.parent_control_id,
                    ref.root_control_id,
                    ref.lineage_depth,
                    ref.lineage_digest,
                    expires_at,
                ),
            )
            connection.execute(
                """
                UPDATE cancellation_controls
                SET parent_control_id = ?, root_control_id = ?,
                    lineage_depth = ?, lineage_digest = ?
                WHERE namespace = ? AND control_id = ?
                  AND (root_control_id IS NULL OR root_control_id = '')
                  AND (lineage_digest IS NULL OR lineage_digest = '')
                """,
                (
                    ref.parent_control_id,
                    ref.root_control_id,
                    ref.lineage_depth,
                    ref.lineage_digest,
                    ref.namespace,
                    ref.control_id,
                ),
            )
            row = connection.execute(
                """
                SELECT parent_control_id, root_control_id, lineage_depth,
                       lineage_digest, retired_at
                FROM cancellation_controls
                WHERE namespace = ? AND control_id = ?
                """,
                (ref.namespace, ref.control_id),
            ).fetchone()
            expected = (
                ref.parent_control_id,
                ref.root_control_id,
                ref.lineage_depth,
                ref.lineage_digest,
            )
            if row is not None and tuple(row[:4]) != expected:
                raise RuntimeError("cancellation control lineage changed")
            retired_at = 0.0 if row is None else float(row[4] or 0.0)
            if expires_at > 0 and retired_at <= 0:
                connection.execute(
                    """
                    UPDATE cancellation_controls SET expires_at = ?
                    WHERE namespace = ? AND control_id = ? AND retired_at <= 0
                    """,
                    (expires_at, ref.namespace, ref.control_id),
                )

    def read(self, ref: CancellationRef) -> CancellationState:
        with self._connect() as connection:
            self._purge_expired(connection)
            control_id: str | None = ref.control_id
            visited: set[str] = set()
            while control_id is not None:
                if control_id in visited:
                    raise RuntimeError("cancellation lineage contains a cycle")
                visited.add(control_id)
                if len(visited) > MAX_CANCELLATION_LINEAGE_DEPTH + 1:
                    raise RuntimeError("cancellation lineage exceeds the supported depth")
                row = connection.execute(
                    """
                    SELECT requested, reason, requested_at, parent_control_id,
                           retired_at
                    FROM cancellation_controls
                    WHERE namespace = ? AND control_id = ?
                    """,
                    (ref.namespace, control_id),
                ).fetchone()
                if row is None:
                    return CancellationState()
                state = CancellationState(bool(row[0]), str(row[1]), float(row[2]))
                if ref.active_ttl_seconds > 0 and float(row[4] or 0.0) <= 0:
                    connection.execute(
                        """
                        UPDATE cancellation_controls SET expires_at = ?
                        WHERE namespace = ? AND control_id = ?
                        """,
                        (
                            self._active_expires_at(ref),
                            ref.namespace,
                            control_id,
                        ),
                    )
                if state.requested:
                    return state
                control_id = None if row[3] is None else str(row[3])
        return CancellationState()

    def request(self, ref: CancellationRef, *, reason: str) -> bool:
        self.create(ref)
        with self._connect() as connection:
            self._purge_expired(connection)
            updated = connection.execute(
                """
                UPDATE cancellation_controls
                SET requested = 1, reason = ?, requested_at = ?
                WHERE namespace = ? AND control_id = ?
                  AND requested = 0 AND retired_at <= 0
                """,
                (str(reason or "cancelled"), time.time(), ref.namespace, ref.control_id),
            ).rowcount
        return updated == 1

    def touch(self, ref: CancellationRef) -> bool:
        with self._connect() as connection:
            self._purge_expired(connection)
            if ref.active_ttl_seconds > 0:
                updated = connection.execute(
                    """
                    UPDATE cancellation_controls SET expires_at = ?
                    WHERE namespace = ? AND control_id = ? AND retired_at <= 0
                    """,
                    (
                        self._active_expires_at(ref),
                        ref.namespace,
                        ref.control_id,
                    ),
                ).rowcount
            else:
                updated = connection.execute(
                    """
                    UPDATE cancellation_controls SET expires_at = expires_at
                    WHERE namespace = ? AND control_id = ? AND retired_at <= 0
                    """,
                    (ref.namespace, ref.control_id),
                ).rowcount
        return updated == 1

    def retire(self, ref: CancellationRef) -> None:
        with self._connect() as connection:
            retention = float(ref.retention_seconds)
            if retention > 0:
                connection.execute(
                    """
                    UPDATE cancellation_controls
                    SET retired_at = ?, expires_at = ?
                    WHERE namespace = ? AND control_id = ?
                    """,
                    (
                        time.time(),
                        time.time() + retention,
                        ref.namespace,
                        ref.control_id,
                    ),
                )
            else:
                connection.execute(
                    """
                    DELETE FROM cancellation_controls
                    WHERE namespace = ? AND control_id = ?
                    """,
                    (ref.namespace, ref.control_id),
                )


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

    @staticmethod
    def _ttl(ref: CancellationRef) -> int | None:
        ttl = float(ref.active_ttl_seconds)
        return max(1, int(math.ceil(ttl))) if ttl > 0 else None

    def create(self, ref: CancellationRef) -> None:
        payload = {
            **CancellationState().as_dict(),
            "parent_control_id": ref.parent_control_id,
            "root_control_id": ref.root_control_id,
            "lineage_depth": ref.lineage_depth,
            "lineage_digest": ref.lineage_digest,
            "retired_at": 0.0,
        }
        key = self._key(ref)
        ttl = self._ttl(ref)
        created = self.client.set(key, json.dumps(payload), nx=True, ex=ttl)
        if not created:
            raw = self.client.get(key)
            if raw is None:
                created = self.client.set(key, json.dumps(payload), nx=True, ex=ttl)
                if created:
                    return
                raw = self.client.get(key)
            if raw is None:
                raise RuntimeError("cancellation control disappeared during creation")
            existing = json.loads(_decode_redis_value(raw))
            current_lineage = (
                existing.get("parent_control_id"),
                existing.get("root_control_id"),
                int(existing.get("lineage_depth", 0) or 0),
                str(existing.get("lineage_digest", "") or ""),
            )
            expected = (
                ref.parent_control_id,
                ref.root_control_id,
                ref.lineage_depth,
                ref.lineage_digest,
            )
            if current_lineage != expected:
                raise RuntimeError("cancellation control lineage changed")
            if float(existing.get("retired_at", 0.0) or 0.0) <= 0:
                self.touch(ref)

    def read(self, ref: CancellationRef) -> CancellationState:
        control_id: str | None = ref.control_id
        visited: set[str] = set()
        while control_id is not None:
            if control_id in visited:
                raise RuntimeError("cancellation lineage contains a cycle")
            visited.add(control_id)
            if len(visited) > MAX_CANCELLATION_LINEAGE_DEPTH + 1:
                raise RuntimeError("cancellation lineage exceeds the supported depth")
            key_ref = CancellationRef(
                control_id=control_id,
                backend=ref.backend,
                namespace=ref.namespace,
                path=ref.path,
                redis_url_env=ref.redis_url_env,
            )
            raw = self.client.get(self._key(key_ref))
            if raw is None:
                return CancellationState()
            payload = json.loads(_decode_redis_value(raw))
            if float(payload.get("retired_at", 0.0) or 0.0) <= 0:
                self._touch_key(self._key(key_ref), self._ttl(ref))
            state = CancellationState.from_dict(payload)
            if state.requested:
                return state
            control_id = payload.get("parent_control_id")
        return CancellationState()

    def request(self, ref: CancellationRef, *, reason: str) -> bool:
        self.create(ref)
        key = self._key(ref)
        while True:
            with self.client.pipeline() as pipe:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    if raw is None:
                        pipe.unwatch()
                        self.create(ref)
                        continue
                    current = CancellationState.from_dict(
                        json.loads(_decode_redis_value(raw))
                    )
                    current_payload = dict(
                        json.loads(_decode_redis_value(raw)) or {}
                    )
                    if float(current_payload.get("retired_at", 0.0) or 0.0) > 0:
                        pipe.unwatch()
                        return False
                    if current.requested:
                        pipe.unwatch()
                        return False
                    state = CancellationState(True, str(reason or "cancelled"), time.time())
                    payload = {
                        **current_payload,
                        **state.as_dict(),
                    }
                    pipe.multi()
                    ttl = self._ttl(ref)
                    if ttl is None:
                        pipe.set(key, json.dumps(payload))
                    else:
                        pipe.set(key, json.dumps(payload), ex=ttl)
                    pipe.execute()
                    return True
                except Exception as exc:
                    if type(exc).__name__ != "WatchError":
                        raise

    def _touch_key(self, key: str, ttl: int | None) -> bool:
        while True:
            with self.client.pipeline() as pipe:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    if raw is None:
                        pipe.unwatch()
                        return False
                    payload = dict(json.loads(_decode_redis_value(raw)) or {})
                    if float(payload.get("retired_at", 0.0) or 0.0) > 0:
                        pipe.unwatch()
                        return False
                    if ttl is None:
                        pipe.unwatch()
                        return True
                    pipe.multi()
                    pipe.expire(key, ttl)
                    pipe.execute()
                    return True
                except Exception as exc:
                    if type(exc).__name__ != "WatchError":
                        raise

    def touch(self, ref: CancellationRef) -> bool:
        return self._touch_key(self._key(ref), self._ttl(ref))

    def retire(self, ref: CancellationRef) -> None:
        key = self._key(ref)
        retention = float(ref.retention_seconds)
        if retention > 0:
            while True:
                with self.client.pipeline() as pipe:
                    try:
                        pipe.watch(key)
                        raw = pipe.get(key)
                        if raw is None:
                            pipe.unwatch()
                            return
                        payload = dict(json.loads(_decode_redis_value(raw)) or {})
                        payload["retired_at"] = time.time()
                        pipe.multi()
                        pipe.set(
                            key,
                            json.dumps(payload),
                            ex=max(1, int(math.ceil(retention))),
                        )
                        pipe.execute()
                        return
                    except Exception as exc:
                        if type(exc).__name__ != "WatchError":
                            raise
        else:
            self.client.delete(key)


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

    def touch(self) -> bool:
        """Refresh the active control lease without changing cancellation state."""

        return bool(self.store.touch(self.ref))

    def retire(self) -> None:
        """Release this control record after its owning execution has finished."""

        self.store.retire(self.ref)

    def checkpoint(self) -> None:
        if self.deadline_exceeded:
            self.cancel("case deadline exceeded")
            raise CaseDeadlineExceeded("case deadline exceeded")
        state = self.state
        if state.requested:
            raise CancellationRequested(state.reason or "case cancellation requested")


class CancellationHeartbeat:
    """Keep one durable cancellation record alive during owned execution."""

    def __init__(self, token: CancellationToken) -> None:
        self.token = token
        self._stop = threading.Event()
        self._lost = threading.Event()
        self._failure: BaseException | None = None
        self._thread: threading.Thread | None = None
        interval = float(token.ref.heartbeat_seconds)
        if token.ref.active_ttl_seconds > 0:
            if not token.touch():
                raise RuntimeError("cancellation control lease is not current")
            self._thread = threading.Thread(
                target=self._run,
                args=(interval,),
                name=f"blackbase-control-heartbeat-{token.ref.control_id}",
                daemon=True,
            )
            self._thread.start()

    @property
    def lost(self) -> bool:
        return self._lost.is_set()

    def _run(self, interval: float) -> None:
        while not self._stop.wait(interval):
            try:
                if not self.token.touch():
                    self._lost.set()
                    return
            except BaseException as exc:
                self._failure = exc
                self._lost.set()
                return

    def assert_current(self) -> None:
        if self._failure is not None:
            raise RuntimeError("cancellation control heartbeat failed") from self._failure
        if self.lost:
            raise RuntimeError("cancellation control lease is no longer current")

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(
                timeout=max(0.1, float(self.token.ref.heartbeat_seconds) * 2.0)
            )
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("cancellation control heartbeat thread did not stop")
        self.assert_current()


__all__ = [
    "CancellationRef",
    "CancellationHeartbeat",
    "CancellationRequested",
    "CancellationState",
    "CancellationStore",
    "CancellationToken",
    "CaseDeadlineExceeded",
    "InMemoryCancellationStore",
    "MAX_CANCELLATION_LINEAGE_DEPTH",
    "RedisCancellationStore",
    "SQLiteCancellationStore",
    "TerminationPolicy",
    "build_cancellation_store",
]
