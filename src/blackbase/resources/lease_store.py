"""Durable Project L0 lease authority with monotonic fencing tokens."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from .model import ResourceLease


_SCHEMA_INITIALIZE_LOCK = threading.RLock()


class SQLiteLeaseStore:
    """SQLite lease authority with atomic aggregate-budget admission.

    A namespace uses one monotonically increasing fencing counter. Admission,
    expiry recovery, budget validation, and insertion happen in the same
    ``BEGIN IMMEDIATE`` transaction, so independent Project processes cannot
    both authorize the same last unit of capacity.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        namespace: str = "project",
        busy_timeout_seconds: float = 10.0,
    ) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = str(namespace or "project")
        self.busy_timeout_seconds = max(0.1, float(busy_timeout_seconds))
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.path),
            timeout=self.busy_timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_seconds * 1000)}")
        return connection

    def _initialize(self) -> None:
        deadline = time.monotonic() + self.busy_timeout_seconds
        with _SCHEMA_INITIALIZE_LOCK:
            while True:
                try:
                    self._initialize_once()
                    return
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.01)

    def _initialize_once(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS resource_leases (
                    namespace TEXT NOT NULL,
                    lease_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    resources_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    fencing_token INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY(namespace, lease_id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_resource_leases_active
                ON resource_leases(namespace, status, expires_at)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS resource_lease_counters (
                    namespace TEXT PRIMARY KEY,
                    next_fencing_token INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO resource_lease_counters(namespace, next_fencing_token)
                VALUES (?, 1)
                """,
                (self.namespace,),
            )

    def acquire_lease(
        self,
        lease: ResourceLease,
        *,
        validate_active: Callable[[Sequence[ResourceLease]], None],
        ttl_seconds: float = 0.0,
    ) -> ResourceLease:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._expire_locked(connection, now=now)
            active = tuple(
                _lease_from_row(row)
                for row in connection.execute(
                    """
                    SELECT * FROM resource_leases
                    WHERE namespace = ? AND status = 'active'
                    ORDER BY fencing_token
                    """,
                    (self.namespace,),
                ).fetchall()
            )
            validate_active(active)
            counter = connection.execute(
                """
                SELECT next_fencing_token FROM resource_lease_counters
                WHERE namespace = ?
                """,
                (self.namespace,),
            ).fetchone()
            if counter is None:  # pragma: no cover - initialized in schema setup
                raise RuntimeError(f"Missing lease counter for namespace='{self.namespace}'")
            fencing_token = int(counter["next_fencing_token"])
            connection.execute(
                """
                UPDATE resource_lease_counters
                SET next_fencing_token = ?
                WHERE namespace = ?
                """,
                (fencing_token + 1, self.namespace),
            )
            issued = replace(
                lease,
                status="active",
                created_at=now,
                updated_at=now,
                expires_at=(now + float(ttl_seconds) if float(ttl_seconds) > 0 else 0.0),
                fencing_token=fencing_token,
            )
            connection.execute(
                """
                INSERT INTO resource_leases(
                    namespace, lease_id, owner_id, scope, resources_json,
                    status, created_at, updated_at, expires_at, fencing_token,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _lease_row_values(self.namespace, issued),
            )
            connection.commit()
        return issued

    def create(self, lease: ResourceLease) -> None:
        self.acquire_lease(
            lease,
            validate_active=lambda active: None,
            ttl_seconds=max(0.0, lease.expires_at - time.time()),
        )

    def get(self, lease_id: str) -> ResourceLease | None:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._expire_locked(connection, now=now)
            row = connection.execute(
                """
                SELECT * FROM resource_leases
                WHERE namespace = ? AND lease_id = ?
                """,
                (self.namespace, str(lease_id)),
            ).fetchone()
            connection.commit()
        return None if row is None else _lease_from_row(row)

    def update(self, lease: ResourceLease) -> None:
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE resource_leases
                SET owner_id = ?, scope = ?, resources_json = ?, status = ?,
                    created_at = ?, updated_at = ?, expires_at = ?,
                    fencing_token = ?, metadata_json = ?
                WHERE namespace = ? AND lease_id = ?
                """,
                (
                    lease.owner_id,
                    lease.scope,
                    _dump_json(lease.resources),
                    lease.status,
                    lease.created_at,
                    lease.updated_at,
                    lease.expires_at,
                    lease.fencing_token,
                    _dump_json(lease.metadata),
                    self.namespace,
                    lease.lease_id,
                ),
            ).rowcount
        if updated != 1:
            raise KeyError(f"Unknown ResourceLease '{lease.lease_id}'")

    def delete(self, lease_id: str) -> None:
        self.release_lease(str(lease_id))

    def list(self) -> tuple[ResourceLease, ...]:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._expire_locked(connection, now=now)
            rows = connection.execute(
                """
                SELECT * FROM resource_leases
                WHERE namespace = ? AND status = 'active'
                ORDER BY fencing_token
                """,
                (self.namespace,),
            ).fetchall()
            connection.commit()
        return tuple(_lease_from_row(row) for row in rows)

    def list_all(self) -> tuple[ResourceLease, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM resource_leases
                WHERE namespace = ? ORDER BY fencing_token
                """,
                (self.namespace,),
            ).fetchall()
        return tuple(_lease_from_row(row) for row in rows)

    def renew_lease(
        self,
        lease_id: str,
        fencing_token: int,
        *,
        ttl_seconds: float,
    ) -> ResourceLease | None:
        now = time.time()
        expires_at = now + float(ttl_seconds) if float(ttl_seconds) > 0 else 0.0
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._expire_locked(connection, now=now)
            updated = connection.execute(
                """
                UPDATE resource_leases
                SET updated_at = ?, expires_at = ?
                WHERE namespace = ? AND lease_id = ? AND fencing_token = ?
                    AND status = 'active'
                """,
                (
                    now,
                    expires_at,
                    self.namespace,
                    str(lease_id),
                    int(fencing_token),
                ),
            ).rowcount
            row = None
            if updated == 1:
                row = connection.execute(
                    """
                    SELECT * FROM resource_leases
                    WHERE namespace = ? AND lease_id = ?
                    """,
                    (self.namespace, str(lease_id)),
                ).fetchone()
            connection.commit()
        return None if row is None else _lease_from_row(row)

    def release_lease(self, lease_id: str, fencing_token: int | None = None) -> bool:
        now = time.time()
        query = (
            """
            UPDATE resource_leases SET status = 'released', updated_at = ?
            WHERE namespace = ? AND lease_id = ? AND status = 'active'
            """
        )
        params: tuple[object, ...] = (now, self.namespace, str(lease_id))
        if fencing_token is not None:
            query += " AND fencing_token = ?"
            params = (*params, int(fencing_token))
        with self._connect() as connection:
            updated = connection.execute(query, params).rowcount
        return updated == 1

    def is_current(self, lease_id: str, fencing_token: int) -> bool:
        now = time.time()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM resource_leases
                WHERE namespace = ? AND lease_id = ? AND fencing_token = ?
                    AND status = 'active'
                    AND (expires_at <= 0 OR expires_at > ?)
                """,
                (
                    self.namespace,
                    str(lease_id),
                    int(fencing_token),
                    now,
                ),
            ).fetchone()
        return row is not None

    def recover_expired(self) -> int:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            recovered = self._expire_locked(connection, now=time.time())
            connection.commit()
        return recovered

    def _expire_locked(self, connection: sqlite3.Connection, *, now: float) -> int:
        return int(
            connection.execute(
                """
                UPDATE resource_leases
                SET status = 'expired', updated_at = ?
                WHERE namespace = ? AND status = 'active'
                    AND expires_at > 0 AND expires_at <= ?
                """,
                (now, self.namespace, now),
            ).rowcount
        )


class RedisLeaseStore:
    """Distributed L0 lease authority backed by a Redis lock and transaction.

    All clients sharing a namespace use the same state lock, lease-id set and
    monotonic counter. Admission holds that distributed lock while recovering
    expiry, validating the aggregate budget and committing the new fence.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        *,
        namespace: str = "project",
        client: Any = None,
        lock_timeout_seconds: float = 30.0,
        blocking_timeout_seconds: float = 10.0,
    ) -> None:
        self.redis_url = str(redis_url)
        self.namespace = str(namespace or "project")
        self.client = client if client is not None else _make_redis_client(self.redis_url)
        self.lock_timeout_seconds = max(1.0, float(lock_timeout_seconds))
        self.blocking_timeout_seconds = max(0.1, float(blocking_timeout_seconds))
        prefix = f"blackbase:l0_leases:{self.namespace}"
        self._lease_ids_key = f"{prefix}:lease_ids"
        self._counter_key = f"{prefix}:next_fencing_token"
        self._lock_key = f"{prefix}:state_lock"
        self._lease_prefix = f"{prefix}:lease:"

    def _lease_key(self, lease_id: str) -> str:
        return f"{self._lease_prefix}{str(lease_id)}"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        lock_factory = getattr(self.client, "lock", None)
        if not callable(lock_factory):
            raise RuntimeError("Redis client must provide redis-py compatible lock()")
        lock = lock_factory(
            self._lock_key,
            timeout=self.lock_timeout_seconds,
            blocking_timeout=self.blocking_timeout_seconds,
        )
        if not lock.acquire(blocking=True):
            raise RuntimeError(
                f"Timed out acquiring Redis lease authority lock '{self._lock_key}'"
            )
        try:
            yield
        finally:
            lock.release()

    def acquire_lease(
        self,
        lease: ResourceLease,
        *,
        validate_active: Callable[[Sequence[ResourceLease]], None],
        ttl_seconds: float = 0.0,
    ) -> ResourceLease:
        now = self._now()
        with self._locked():
            self._expire_locked(now=now)
            active = self._list_locked(active_only=True)
            validate_active(active)
            raw_counter = self.client.get(self._counter_key)
            fencing_token = int(_decode_redis(raw_counter)) if raw_counter is not None else 1
            issued = replace(
                lease,
                status="active",
                created_at=now,
                updated_at=now,
                expires_at=(now + float(ttl_seconds) if float(ttl_seconds) > 0 else 0.0),
                fencing_token=fencing_token,
            )
            pipe = self.client.pipeline(transaction=True)
            pipe.set(self._lease_key(issued.lease_id), _dump_json(issued.as_dict()))
            pipe.sadd(self._lease_ids_key, issued.lease_id)
            pipe.set(self._counter_key, fencing_token + 1)
            pipe.execute()
        return issued

    def create(self, lease: ResourceLease) -> None:
        self.acquire_lease(
            lease,
            validate_active=lambda active: None,
            ttl_seconds=max(0.0, lease.expires_at - self._now()),
        )

    def get(self, lease_id: str) -> ResourceLease | None:
        with self._locked():
            self._expire_locked(now=self._now())
            return self._load_locked(lease_id)

    def update(self, lease: ResourceLease) -> None:
        with self._locked():
            if self._load_locked(lease.lease_id) is None:
                raise KeyError(f"Unknown ResourceLease '{lease.lease_id}'")
            self.client.set(self._lease_key(lease.lease_id), _dump_json(lease.as_dict()))

    def delete(self, lease_id: str) -> None:
        self.release_lease(str(lease_id))

    def list(self) -> tuple[ResourceLease, ...]:
        with self._locked():
            self._expire_locked(now=self._now())
            return self._list_locked(active_only=True)

    def list_all(self) -> tuple[ResourceLease, ...]:
        with self._locked():
            return self._list_locked(active_only=False)

    def renew_lease(
        self,
        lease_id: str,
        fencing_token: int,
        *,
        ttl_seconds: float,
    ) -> ResourceLease | None:
        now = self._now()
        with self._locked():
            self._expire_locked(now=now)
            current = self._load_locked(lease_id)
            if (
                current is None
                or current.status != "active"
                or int(current.fencing_token) != int(fencing_token)
            ):
                return None
            renewed = replace(
                current,
                updated_at=now,
                expires_at=(now + float(ttl_seconds) if float(ttl_seconds) > 0 else 0.0),
            )
            self.client.set(self._lease_key(lease_id), _dump_json(renewed.as_dict()))
            return renewed

    def release_lease(self, lease_id: str, fencing_token: int | None = None) -> bool:
        now = self._now()
        with self._locked():
            self._expire_locked(now=now)
            current = self._load_locked(lease_id)
            if current is None or current.status != "active":
                return False
            if fencing_token is not None and int(current.fencing_token) != int(fencing_token):
                return False
            released = replace(current, status="released", updated_at=now)
            self.client.set(self._lease_key(lease_id), _dump_json(released.as_dict()))
            return True

    def is_current(self, lease_id: str, fencing_token: int) -> bool:
        with self._locked():
            self._expire_locked(now=self._now())
            current = self._load_locked(lease_id)
            return bool(
                current is not None
                and current.active
                and int(current.fencing_token) == int(fencing_token)
            )

    def recover_expired(self) -> int:
        with self._locked():
            return self._expire_locked(now=self._now())

    def _now(self) -> float:
        redis_time = getattr(self.client, "time", None)
        if callable(redis_time):
            raw = redis_time()
            if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                return float(raw[0]) + float(raw[1]) / 1_000_000.0
        return time.time()

    def _load_locked(self, lease_id: str) -> ResourceLease | None:
        raw = self.client.get(self._lease_key(lease_id))
        if raw is None:
            return None
        payload = json.loads(_decode_redis(raw))
        return ResourceLease.from_dict(payload)

    def _list_locked(self, *, active_only: bool) -> tuple[ResourceLease, ...]:
        leases: list[ResourceLease] = []
        for raw_id in self.client.smembers(self._lease_ids_key):
            item = self._load_locked(_decode_redis(raw_id))
            if item is None or (active_only and not item.active):
                continue
            leases.append(item)
        return tuple(sorted(leases, key=lambda item: item.fencing_token))

    def _expire_locked(self, *, now: float) -> int:
        expired: list[ResourceLease] = []
        for item in self._list_locked(active_only=False):
            if item.status == "active" and item.expires_at > 0 and item.expires_at <= now:
                expired.append(replace(item, status="expired", updated_at=now))
        if expired:
            pipe = self.client.pipeline(transaction=True)
            for item in expired:
                pipe.set(self._lease_key(item.lease_id), _dump_json(item.as_dict()))
            pipe.execute()
        return len(expired)


def _lease_row_values(namespace: str, lease: ResourceLease) -> tuple[object, ...]:
    return (
        str(namespace),
        lease.lease_id,
        lease.owner_id,
        lease.scope,
        _dump_json(lease.resources),
        lease.status,
        float(lease.created_at),
        float(lease.updated_at),
        float(lease.expires_at),
        int(lease.fencing_token),
        _dump_json(lease.metadata),
    )


def _lease_from_row(row: sqlite3.Row) -> ResourceLease:
    return ResourceLease(
        lease_id=str(row["lease_id"]),
        owner_id=str(row["owner_id"]),
        scope=str(row["scope"]),
        resources=dict(json.loads(str(row["resources_json"]))),
        status=str(row["status"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        expires_at=float(row["expires_at"]),
        fencing_token=int(row["fencing_token"]),
        metadata=dict(json.loads(str(row["metadata_json"]))),
    )


def _dump_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _decode_redis(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _make_redis_client(redis_url: str) -> Any:
    try:
        import redis  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("RedisLeaseStore requires the optional 'redis' package") from exc
    return redis.from_url(str(redis_url))


__all__ = ["RedisLeaseStore", "SQLiteLeaseStore"]
