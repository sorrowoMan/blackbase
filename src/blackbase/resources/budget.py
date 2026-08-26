"""Lease-fenced Project L0 budgets shared across Cases and processes."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping
from uuid import uuid4

from .model import ResourceLease


_SCHEMA_LOCK = threading.RLock()


class SharedBudgetError(RuntimeError):
    """Base error for the shared Project budget authority."""


class SharedBudgetExceeded(SharedBudgetError):
    """Raised when an atomic reservation would exceed the configured limit."""


class SharedBudgetFenceError(SharedBudgetError):
    """Raised when a reservation caller no longer owns its Project lease."""


class SharedBudgetConfigurationError(SharedBudgetError):
    """Raised when authorities disagree about one run-scoped budget limit."""


@dataclass(frozen=True)
class BudgetReservation:
    """One lease-fenced reservation against a run-scoped shared budget."""

    reservation_id: str
    scope: str
    budget: str
    amount: int
    lease_id: str
    fencing_token: int
    status: str = "active"
    completed: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "reservation_id": str(self.reservation_id),
            "scope": str(self.scope),
            "budget": str(self.budget),
            "amount": int(self.amount),
            "lease_id": str(self.lease_id),
            "fencing_token": int(self.fencing_token),
            "status": str(self.status),
            "completed": int(self.completed),
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BudgetReservation":
        payload = dict(value or {})
        return cls(
            reservation_id=str(payload.get("reservation_id", "")),
            scope=str(payload.get("scope", "")),
            budget=str(payload.get("budget", "")),
            amount=max(0, int(payload.get("amount", 0) or 0)),
            lease_id=str(payload.get("lease_id", "")),
            fencing_token=max(0, int(payload.get("fencing_token", 0) or 0)),
            status=str(payload.get("status", "active")),
            completed=max(0, int(payload.get("completed", 0) or 0)),
            created_at=float(payload.get("created_at", 0.0) or 0.0),
            updated_at=float(payload.get("updated_at", 0.0) or 0.0),
        )


@dataclass(frozen=True)
class BudgetSnapshot:
    """Atomic usage view for one run-scoped shared budget."""

    scope: str
    budget: str
    limit: int
    committed: int
    reserved: int
    reclaimed: int = 0

    @property
    def remaining(self) -> int:
        return max(0, int(self.limit) - int(self.committed) - int(self.reserved))

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": str(self.scope),
            "budget": str(self.budget),
            "limit": int(self.limit),
            "committed": int(self.committed),
            "reserved": int(self.reserved),
            "remaining": int(self.remaining),
            "reclaimed": int(self.reclaimed),
        }


class SQLiteBudgetAuthority:
    """Atomic shared budget authority co-located with SQLite L0 leases."""

    def __init__(
        self,
        path: Path | str,
        *,
        namespace: str,
        scope: str,
        busy_timeout_seconds: float = 10.0,
    ) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = str(namespace or "project")
        self.scope = str(scope or "run")
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
        with _SCHEMA_LOCK, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS resource_budgets (
                    namespace TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    budget TEXT NOT NULL,
                    limit_amount INTEGER NOT NULL,
                    committed_amount INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(namespace, scope, budget)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS resource_budget_reservations (
                    namespace TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    reservation_id TEXT NOT NULL,
                    budget TEXT NOT NULL,
                    amount INTEGER NOT NULL,
                    lease_id TEXT NOT NULL,
                    fencing_token INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    completed_amount INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(namespace, scope, reservation_id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_resource_budget_reservations_active
                ON resource_budget_reservations(namespace, scope, budget, status)
                """
            )

    def configure(self, budget: str, limit: int) -> None:
        name = _budget_name(budget)
        maximum = _budget_limit(limit)
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT limit_amount FROM resource_budgets
                WHERE namespace = ? AND scope = ? AND budget = ?
                """,
                (self.namespace, self.scope, name),
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO resource_budgets(
                        namespace, scope, budget, limit_amount,
                        committed_amount, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 0, ?, ?)
                    """,
                    (self.namespace, self.scope, name, maximum, now, now),
                )
            elif int(row["limit_amount"]) != maximum:
                raise SharedBudgetConfigurationError(
                    f"budget '{name}' already configured with limit={int(row['limit_amount'])}, "
                    f"requested={maximum}"
                )
            connection.commit()

    def status(self, budget: str) -> BudgetSnapshot:
        name = _budget_name(budget)
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            reclaimed = self._reclaim_stale_locked(connection, now=now)
            row = self._budget_row_locked(connection, name)
            reserved = self._reserved_locked(connection, name)
            connection.commit()
        return BudgetSnapshot(
            scope=self.scope,
            budget=name,
            limit=int(row["limit_amount"]),
            committed=int(row["committed_amount"]),
            reserved=reserved,
            reclaimed=reclaimed,
        )

    def reservation(self, reservation: BudgetReservation | str) -> BudgetReservation | None:
        """Return the durable reservation state after applying lease reclamation."""

        identifier = _reservation_id(reservation)
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._reclaim_stale_locked(connection, now=now)
            row = self._reservation_row_locked(connection, identifier)
            connection.commit()
        return None if row is None else _reservation_from_row(row)

    def reserve(
        self,
        budget: str,
        amount: int,
        *,
        lease_id: str,
        fencing_token: int,
        reservation_id: str | None = None,
    ) -> BudgetReservation:
        name = _budget_name(budget)
        requested = _reservation_amount(amount)
        now = time.time()
        identifier = str(reservation_id or f"budget-{uuid4().hex}")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_fence_locked(connection, lease_id, fencing_token, now=now)
            self._reclaim_stale_locked(connection, now=now)
            existing = self._reservation_row_locked(connection, identifier)
            if existing is not None:
                reservation = _reservation_from_row(existing)
                if (
                    reservation.budget == name
                    and reservation.amount == requested
                    and reservation.lease_id == str(lease_id)
                    and reservation.fencing_token == int(fencing_token)
                ):
                    connection.commit()
                    return reservation
                raise SharedBudgetConfigurationError(
                    f"reservation_id '{identifier}' was reused with a different request"
                )
            row = self._budget_row_locked(connection, name)
            reserved = self._reserved_locked(connection, name)
            remaining = max(
                0,
                int(row["limit_amount"]) - int(row["committed_amount"]) - reserved,
            )
            if requested > remaining:
                raise SharedBudgetExceeded(
                    f"shared budget '{name}' exceeded: requested={requested}, remaining={remaining}"
                )
            reservation = BudgetReservation(
                reservation_id=identifier,
                scope=self.scope,
                budget=name,
                amount=requested,
                lease_id=str(lease_id),
                fencing_token=int(fencing_token),
                created_at=now,
                updated_at=now,
            )
            connection.execute(
                """
                INSERT INTO resource_budget_reservations(
                    namespace, scope, reservation_id, budget, amount,
                    lease_id, fencing_token, status, completed_amount,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', 0, ?, ?)
                """,
                (
                    self.namespace,
                    self.scope,
                    identifier,
                    name,
                    requested,
                    str(lease_id),
                    int(fencing_token),
                    now,
                    now,
                ),
            )
            connection.commit()
        return reservation

    def complete(self, reservation: BudgetReservation | str, *, completed: int) -> BudgetReservation:
        identifier = _reservation_id(reservation)
        completed_count = max(0, int(completed))
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._reservation_row_locked(connection, identifier)
            if row is None:
                raise KeyError(f"Unknown budget reservation '{identifier}'")
            current = _reservation_from_row(row)
            if current.status == "completed" and current.completed == completed_count:
                connection.commit()
                return current
            if current.status != "active":
                raise SharedBudgetFenceError(
                    f"budget reservation '{identifier}' is no longer active"
                )
            if completed_count > current.amount:
                raise ValueError("completed budget units cannot exceed the reservation")
            if completed_count < current.completed:
                raise ValueError(
                    "completed budget units cannot be smaller than already consumed units"
                )
            self._assert_fence_locked(
                connection,
                current.lease_id,
                current.fencing_token,
                now=now,
            )
            delta = completed_count - current.completed
            if delta:
                connection.execute(
                    """
                    UPDATE resource_budgets
                    SET committed_amount = committed_amount + ?, updated_at = ?
                    WHERE namespace = ? AND scope = ? AND budget = ?
                    """,
                    (delta, now, self.namespace, self.scope, current.budget),
                )
            connection.execute(
                """
                UPDATE resource_budget_reservations
                SET status = 'completed', completed_amount = ?, updated_at = ?
                WHERE namespace = ? AND scope = ? AND reservation_id = ?
                """,
                (completed_count, now, self.namespace, self.scope, identifier),
            )
            connection.commit()
        return replace(current, status="completed", completed=completed_count, updated_at=now)

    def consume(
        self,
        reservation: BudgetReservation | str,
        *,
        amount: int = 1,
    ) -> BudgetReservation:
        """Permanently consume units immediately before work is dispatched."""

        identifier = _reservation_id(reservation)
        delta = _reservation_amount(amount)
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._reservation_row_locked(connection, identifier)
            if row is None:
                raise KeyError(f"Unknown budget reservation '{identifier}'")
            current = _reservation_from_row(row)
            if current.status != "active":
                raise SharedBudgetFenceError(
                    f"budget reservation '{identifier}' is no longer active"
                )
            if current.completed + delta > current.amount:
                raise ValueError("consumed budget units cannot exceed the reservation")
            self._assert_fence_locked(
                connection,
                current.lease_id,
                current.fencing_token,
                now=now,
            )
            consumed = current.completed + delta
            connection.execute(
                """
                UPDATE resource_budgets
                SET committed_amount = committed_amount + ?, updated_at = ?
                WHERE namespace = ? AND scope = ? AND budget = ?
                """,
                (delta, now, self.namespace, self.scope, current.budget),
            )
            connection.execute(
                """
                UPDATE resource_budget_reservations
                SET completed_amount = ?, updated_at = ?
                WHERE namespace = ? AND scope = ? AND reservation_id = ?
                """,
                (consumed, now, self.namespace, self.scope, identifier),
            )
            connection.commit()
        return replace(current, completed=consumed, updated_at=now)

    def cancel(self, reservation: BudgetReservation | str) -> bool:
        identifier = _reservation_id(reservation)
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._reservation_row_locked(connection, identifier)
            if row is None:
                connection.commit()
                return False
            current = _reservation_from_row(row)
            if current.status != "active" or not self._fence_current_locked(
                connection,
                current.lease_id,
                current.fencing_token,
                now=now,
            ):
                connection.commit()
                return False
            connection.execute(
                """
                UPDATE resource_budget_reservations
                SET status = 'cancelled', updated_at = ?
                WHERE namespace = ? AND scope = ? AND reservation_id = ?
                """,
                (now, self.namespace, self.scope, identifier),
            )
            connection.commit()
        return True

    def _budget_row_locked(self, connection: sqlite3.Connection, budget: str) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT * FROM resource_budgets
            WHERE namespace = ? AND scope = ? AND budget = ?
            """,
            (self.namespace, self.scope, budget),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown shared budget '{budget}' in scope '{self.scope}'")
        return row

    def _reservation_row_locked(
        self,
        connection: sqlite3.Connection,
        reservation_id: str,
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT * FROM resource_budget_reservations
            WHERE namespace = ? AND scope = ? AND reservation_id = ?
            """,
            (self.namespace, self.scope, reservation_id),
        ).fetchone()

    def _reserved_locked(self, connection: sqlite3.Connection, budget: str) -> int:
        row = connection.execute(
            """
            SELECT COALESCE(SUM(amount - completed_amount), 0) AS reserved
            FROM resource_budget_reservations
            WHERE namespace = ? AND scope = ? AND budget = ? AND status = 'active'
            """,
            (self.namespace, self.scope, budget),
        ).fetchone()
        return int(row["reserved"] if row is not None else 0)

    def _fence_current_locked(
        self,
        connection: sqlite3.Connection,
        lease_id: str,
        fencing_token: int,
        *,
        now: float,
    ) -> bool:
        try:
            row = connection.execute(
                """
                SELECT 1 FROM resource_leases
                WHERE namespace = ? AND lease_id = ? AND fencing_token = ?
                    AND status = 'active'
                    AND (expires_at <= 0 OR expires_at > ?)
                """,
                (self.namespace, str(lease_id), int(fencing_token), now),
            ).fetchone()
        except sqlite3.OperationalError as exc:
            raise SharedBudgetConfigurationError(
                "SQLite shared budgets must use the same database as the L0 lease authority"
            ) from exc
        return row is not None

    def _assert_fence_locked(
        self,
        connection: sqlite3.Connection,
        lease_id: str,
        fencing_token: int,
        *,
        now: float,
    ) -> None:
        if not self._fence_current_locked(connection, lease_id, fencing_token, now=now):
            raise SharedBudgetFenceError(
                f"Project L0 lease fence is not current: lease_id='{lease_id}' "
                f"token={int(fencing_token)}"
            )

    def _reclaim_stale_locked(self, connection: sqlite3.Connection, *, now: float) -> int:
        try:
            return int(
                connection.execute(
                    """
                    UPDATE resource_budget_reservations
                    SET status = 'reclaimed', updated_at = ?
                    WHERE namespace = ? AND scope = ? AND status = 'active'
                        AND NOT EXISTS (
                            SELECT 1 FROM resource_leases
                            WHERE resource_leases.namespace = resource_budget_reservations.namespace
                                AND resource_leases.lease_id = resource_budget_reservations.lease_id
                                AND resource_leases.fencing_token = resource_budget_reservations.fencing_token
                                AND resource_leases.status = 'active'
                                AND (resource_leases.expires_at <= 0 OR resource_leases.expires_at > ?)
                        )
                    """,
                    (now, self.namespace, self.scope, now),
                ).rowcount
            )
        except sqlite3.OperationalError as exc:
            raise SharedBudgetConfigurationError(
                "SQLite shared budgets must use the same database as the L0 lease authority"
            ) from exc


class RedisBudgetAuthority:
    """Distributed shared budget authority using the Redis L0 lease lock."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        *,
        namespace: str,
        scope: str,
        client: Any = None,
        lock_timeout_seconds: float = 30.0,
        blocking_timeout_seconds: float = 10.0,
    ) -> None:
        self.redis_url = str(redis_url)
        self.namespace = str(namespace or "project")
        self.scope = str(scope or "run")
        self.client = client if client is not None else _make_redis_client(self.redis_url)
        self.lock_timeout_seconds = max(1.0, float(lock_timeout_seconds))
        self.blocking_timeout_seconds = max(0.1, float(blocking_timeout_seconds))
        lease_prefix = f"blackbase:l0_leases:{self.namespace}"
        budget_prefix = f"blackbase:l0_budgets:{self.namespace}:{self.scope}"
        self._lock_key = f"{lease_prefix}:state_lock"
        self._lease_prefix = f"{lease_prefix}:lease:"
        self._budget_prefix = f"{budget_prefix}:budget:"
        self._reservation_ids_key = f"{budget_prefix}:reservation_ids"
        self._reservation_prefix = f"{budget_prefix}:reservation:"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        factory = getattr(self.client, "lock", None)
        if not callable(factory):
            raise RuntimeError("Redis client must provide redis-py compatible lock()")
        lock = factory(
            self._lock_key,
            timeout=self.lock_timeout_seconds,
            blocking_timeout=self.blocking_timeout_seconds,
        )
        if not lock.acquire(blocking=True):
            raise RuntimeError(f"Timed out acquiring Redis L0 state lock '{self._lock_key}'")
        try:
            yield
        finally:
            lock.release()

    def configure(self, budget: str, limit: int) -> None:
        name = _budget_name(budget)
        maximum = _budget_limit(limit)
        with self._locked():
            state = self._budget_state_locked(name)
            if state is None:
                now = self._now()
                self.client.set(
                    self._budget_key(name),
                    _dump_json({
                        "limit": maximum,
                        "committed": 0,
                        "created_at": now,
                        "updated_at": now,
                    }),
                )
            elif int(state["limit"]) != maximum:
                raise SharedBudgetConfigurationError(
                    f"budget '{name}' already configured with limit={int(state['limit'])}, "
                    f"requested={maximum}"
                )

    def status(self, budget: str) -> BudgetSnapshot:
        name = _budget_name(budget)
        with self._locked():
            reclaimed = self._reclaim_stale_locked(now=self._now())
            state = self._require_budget_locked(name)
            reserved = sum(
                item.amount - item.completed
                for item in self._reservations_locked()
                if item.budget == name and item.status == "active"
            )
        return BudgetSnapshot(
            scope=self.scope,
            budget=name,
            limit=int(state["limit"]),
            committed=int(state.get("committed", 0)),
            reserved=reserved,
            reclaimed=reclaimed,
        )

    def reservation(self, reservation: BudgetReservation | str) -> BudgetReservation | None:
        """Return the durable reservation state after applying lease reclamation."""

        identifier = _reservation_id(reservation)
        with self._locked():
            self._reclaim_stale_locked(now=self._now())
            return self._load_reservation_locked(identifier)

    def reserve(
        self,
        budget: str,
        amount: int,
        *,
        lease_id: str,
        fencing_token: int,
        reservation_id: str | None = None,
    ) -> BudgetReservation:
        name = _budget_name(budget)
        requested = _reservation_amount(amount)
        identifier = str(reservation_id or f"budget-{uuid4().hex}")
        with self._locked():
            now = self._now()
            self._assert_fence_locked(lease_id, fencing_token, now=now)
            self._reclaim_stale_locked(now=now)
            existing = self._load_reservation_locked(identifier)
            if existing is not None:
                if (
                    existing.budget == name
                    and existing.amount == requested
                    and existing.lease_id == str(lease_id)
                    and existing.fencing_token == int(fencing_token)
                ):
                    return existing
                raise SharedBudgetConfigurationError(
                    f"reservation_id '{identifier}' was reused with a different request"
                )
            state = self._require_budget_locked(name)
            reserved = sum(
                item.amount - item.completed
                for item in self._reservations_locked()
                if item.budget == name and item.status == "active"
            )
            remaining = max(0, int(state["limit"]) - int(state.get("committed", 0)) - reserved)
            if requested > remaining:
                raise SharedBudgetExceeded(
                    f"shared budget '{name}' exceeded: requested={requested}, remaining={remaining}"
                )
            item = BudgetReservation(
                reservation_id=identifier,
                scope=self.scope,
                budget=name,
                amount=requested,
                lease_id=str(lease_id),
                fencing_token=int(fencing_token),
                created_at=now,
                updated_at=now,
            )
            pipe = self.client.pipeline(transaction=True)
            pipe.set(self._reservation_key(identifier), _dump_json(item.as_dict()))
            pipe.sadd(self._reservation_ids_key, identifier)
            pipe.execute()
            return item

    def complete(self, reservation: BudgetReservation | str, *, completed: int) -> BudgetReservation:
        identifier = _reservation_id(reservation)
        completed_count = max(0, int(completed))
        with self._locked():
            now = self._now()
            current = self._load_reservation_locked(identifier)
            if current is None:
                raise KeyError(f"Unknown budget reservation '{identifier}'")
            if current.status == "completed" and current.completed == completed_count:
                return current
            if current.status != "active":
                raise SharedBudgetFenceError(
                    f"budget reservation '{identifier}' is no longer active"
                )
            if completed_count > current.amount:
                raise ValueError("completed budget units cannot exceed the reservation")
            if completed_count < current.completed:
                raise ValueError(
                    "completed budget units cannot be smaller than already consumed units"
                )
            self._assert_fence_locked(
                current.lease_id,
                current.fencing_token,
                now=now,
            )
            state = self._require_budget_locked(current.budget)
            state["committed"] = int(state.get("committed", 0)) + (
                completed_count - current.completed
            )
            state["updated_at"] = now
            completed_item = replace(
                current,
                status="completed",
                completed=completed_count,
                updated_at=now,
            )
            pipe = self.client.pipeline(transaction=True)
            pipe.set(self._budget_key(current.budget), _dump_json(state))
            pipe.set(self._reservation_key(identifier), _dump_json(completed_item.as_dict()))
            pipe.execute()
            return completed_item

    def consume(
        self,
        reservation: BudgetReservation | str,
        *,
        amount: int = 1,
    ) -> BudgetReservation:
        """Permanently consume units immediately before work is dispatched."""

        identifier = _reservation_id(reservation)
        delta = _reservation_amount(amount)
        with self._locked():
            now = self._now()
            current = self._load_reservation_locked(identifier)
            if current is None:
                raise KeyError(f"Unknown budget reservation '{identifier}'")
            if current.status != "active":
                raise SharedBudgetFenceError(
                    f"budget reservation '{identifier}' is no longer active"
                )
            if current.completed + delta > current.amount:
                raise ValueError("consumed budget units cannot exceed the reservation")
            self._assert_fence_locked(
                current.lease_id,
                current.fencing_token,
                now=now,
            )
            state = self._require_budget_locked(current.budget)
            state["committed"] = int(state.get("committed", 0)) + delta
            state["updated_at"] = now
            consumed = replace(
                current,
                completed=current.completed + delta,
                updated_at=now,
            )
            pipe = self.client.pipeline(transaction=True)
            pipe.set(self._budget_key(current.budget), _dump_json(state))
            pipe.set(self._reservation_key(identifier), _dump_json(consumed.as_dict()))
            pipe.execute()
            return consumed

    def cancel(self, reservation: BudgetReservation | str) -> bool:
        identifier = _reservation_id(reservation)
        with self._locked():
            now = self._now()
            current = self._load_reservation_locked(identifier)
            if (
                current is None
                or current.status != "active"
                or not self._fence_current_locked(
                    current.lease_id,
                    current.fencing_token,
                    now=now,
                )
            ):
                return False
            self.client.set(
                self._reservation_key(identifier),
                _dump_json(replace(current, status="cancelled", updated_at=now).as_dict()),
            )
            return True

    def _budget_key(self, budget: str) -> str:
        return f"{self._budget_prefix}{budget}"

    def _reservation_key(self, reservation_id: str) -> str:
        return f"{self._reservation_prefix}{reservation_id}"

    def _lease_key(self, lease_id: str) -> str:
        return f"{self._lease_prefix}{lease_id}"

    def _budget_state_locked(self, budget: str) -> dict[str, Any] | None:
        raw = self.client.get(self._budget_key(budget))
        return None if raw is None else dict(json.loads(_decode_redis(raw)))

    def _require_budget_locked(self, budget: str) -> dict[str, Any]:
        state = self._budget_state_locked(budget)
        if state is None:
            raise KeyError(f"Unknown shared budget '{budget}' in scope '{self.scope}'")
        return state

    def _load_reservation_locked(self, reservation_id: str) -> BudgetReservation | None:
        raw = self.client.get(self._reservation_key(reservation_id))
        if raw is None:
            return None
        return BudgetReservation.from_dict(json.loads(_decode_redis(raw)))

    def _reservations_locked(self) -> tuple[BudgetReservation, ...]:
        items = []
        for raw_id in self.client.smembers(self._reservation_ids_key):
            item = self._load_reservation_locked(_decode_redis(raw_id))
            if item is not None:
                items.append(item)
        return tuple(items)

    def _load_lease_locked(self, lease_id: str) -> ResourceLease | None:
        raw = self.client.get(self._lease_key(lease_id))
        if raw is None:
            return None
        return ResourceLease.from_dict(json.loads(_decode_redis(raw)))

    def _fence_current_locked(
        self,
        lease_id: str,
        fencing_token: int,
        *,
        now: float,
    ) -> bool:
        lease = self._load_lease_locked(str(lease_id))
        return bool(
            lease is not None
            and lease.status == "active"
            and int(lease.fencing_token) == int(fencing_token)
            and (lease.expires_at <= 0 or lease.expires_at > now)
        )

    def _assert_fence_locked(
        self,
        lease_id: str,
        fencing_token: int,
        *,
        now: float,
    ) -> None:
        if not self._fence_current_locked(lease_id, fencing_token, now=now):
            raise SharedBudgetFenceError(
                f"Project L0 lease fence is not current: lease_id='{lease_id}' "
                f"token={int(fencing_token)}"
            )

    def _reclaim_stale_locked(self, *, now: float) -> int:
        stale = [
            item
            for item in self._reservations_locked()
            if item.status == "active"
            and not self._fence_current_locked(
                item.lease_id,
                item.fencing_token,
                now=now,
            )
        ]
        if stale:
            pipe = self.client.pipeline(transaction=True)
            for item in stale:
                pipe.set(
                    self._reservation_key(item.reservation_id),
                    _dump_json(replace(item, status="reclaimed", updated_at=now).as_dict()),
                )
            pipe.execute()
        return len(stale)

    def _now(self) -> float:
        redis_time = getattr(self.client, "time", None)
        if callable(redis_time):
            raw = redis_time()
            if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                return float(raw[0]) + float(raw[1]) / 1_000_000.0
        return time.time()


def build_budget_authority_from_resource_context(
    resource_context: Mapping[str, Any] | Any,
    *,
    redis_url: str = "",
    redis_client: Any = None,
) -> SQLiteBudgetAuthority | RedisBudgetAuthority | None:
    """Reconstruct the serializable budget authority granted by Project L0."""

    if hasattr(resource_context, "as_dict"):
        payload = dict(resource_context.as_dict())
    else:
        payload = dict(resource_context or {})
    metadata = dict(payload.get("metadata", {}) or {})
    authority = dict(metadata.get("budget_authority", {}) or {})
    if not authority:
        return None
    backend = str(authority.get("backend", "")).strip().lower()
    namespace = str(authority.get("namespace", "project") or "project")
    scope = str(authority.get("scope", "") or "")
    if not scope:
        raise SharedBudgetConfigurationError("Project L0 budget authority omitted its run scope")
    if backend == "sqlite":
        path = str(authority.get("path", "") or "").strip()
        if not path:
            raise SharedBudgetConfigurationError("SQLite budget authority omitted its database path")
        return SQLiteBudgetAuthority(path, namespace=namespace, scope=scope)
    if backend == "redis":
        resolved_url = str(redis_url or "").strip()
        env_name = str(authority.get("redis_url_env", "BLACKBASE_REDIS_URL") or "")
        if not resolved_url and env_name:
            resolved_url = str(os.environ.get(env_name, "") or "").strip()
        if redis_client is None and not resolved_url:
            raise SharedBudgetConfigurationError(
                "Redis budget authority requires an injected client or "
                f"environment variable {env_name}"
            )
        return RedisBudgetAuthority(
            resolved_url or "redis://localhost:6379/0",
            namespace=namespace,
            scope=scope,
            client=redis_client,
        )
    raise SharedBudgetConfigurationError(
        f"Unsupported shared budget backend '{backend}'; expected sqlite or redis"
    )


def _budget_name(value: str) -> str:
    name = str(value or "").strip()
    if not name:
        raise ValueError("budget name must be non-empty")
    return name


def _budget_limit(value: int) -> int:
    limit = int(value)
    if limit < 0:
        raise ValueError("budget limit must be non-negative")
    return limit


def _reservation_amount(value: int) -> int:
    amount = int(value)
    if amount <= 0:
        raise ValueError("reservation amount must be positive")
    return amount


def _reservation_id(value: BudgetReservation | str) -> str:
    identifier = value.reservation_id if isinstance(value, BudgetReservation) else str(value)
    if not identifier:
        raise ValueError("reservation_id must be non-empty")
    return identifier


def _reservation_from_row(row: sqlite3.Row) -> BudgetReservation:
    return BudgetReservation(
        reservation_id=str(row["reservation_id"]),
        scope=str(row["scope"]),
        budget=str(row["budget"]),
        amount=int(row["amount"]),
        lease_id=str(row["lease_id"]),
        fencing_token=int(row["fencing_token"]),
        status=str(row["status"]),
        completed=int(row["completed_amount"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
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
        raise RuntimeError("RedisBudgetAuthority requires the optional 'redis' package") from exc
    return redis.from_url(str(redis_url))


__all__ = [
    "BudgetReservation",
    "BudgetSnapshot",
    "RedisBudgetAuthority",
    "SQLiteBudgetAuthority",
    "SharedBudgetConfigurationError",
    "SharedBudgetError",
    "SharedBudgetExceeded",
    "SharedBudgetFenceError",
    "build_budget_authority_from_resource_context",
]
