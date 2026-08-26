"""Durable journal for parent/child budget delegation settlement."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


_SETTLEMENT_SCHEMA_V1 = "blackbase.budget_settlement/v1"
_SETTLEMENT_SCHEMA = "blackbase.budget_settlement/v2"
_LOCK = threading.RLock()


@dataclass(frozen=True)
class BudgetSettlementRecord:
    settlement_id: str
    project_run_id: str
    parent_case_run_id: str
    child_case_run_id: str
    budget: str
    parent_reservation_id: str
    parent_authority_ref: Mapping[str, Any]
    child_handle: Mapping[str, Any]
    requested_amount: int = 0
    status: str = "reserve_intent"
    usage: Mapping[str, Any] = field(default_factory=dict)
    attempts: int = 0
    last_error: Mapping[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "settlement_id",
            "project_run_id",
            "parent_case_run_id",
            "child_case_run_id",
            "budget",
            "parent_reservation_id",
        ):
            value = str(getattr(self, name, "") or "").strip()
            if not value:
                raise ValueError(f"BudgetSettlementRecord.{name} must not be empty")
            object.__setattr__(self, name, value)
        status = str(self.status or "prepared").strip().lower()
        if status not in {
            "reserve_intent",
            "prepared",
            "settlement_ready",
            "retry_required",
            "settled",
            "reclaimed",
        }:
            raise ValueError(f"unsupported budget settlement status '{status}'")
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "parent_authority_ref",
            _plain_wire(dict(self.parent_authority_ref or {})),
        )
        object.__setattr__(self, "child_handle", _plain_wire(dict(self.child_handle or {})))
        object.__setattr__(self, "usage", _plain_wire(dict(self.usage or {})))
        object.__setattr__(self, "last_error", _plain_wire(dict(self.last_error or {})))
        object.__setattr__(self, "attempts", max(0, int(self.attempts or 0)))
        amount = int(self.requested_amount or 0)
        if amount < 0:
            raise ValueError("BudgetSettlementRecord.requested_amount must be non-negative")
        object.__setattr__(self, "requested_amount", amount)
        now = time.time()
        object.__setattr__(self, "created_at", float(self.created_at or now))
        object.__setattr__(self, "updated_at", float(self.updated_at or now))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": _SETTLEMENT_SCHEMA,
            "settlement_id": self.settlement_id,
            "project_run_id": self.project_run_id,
            "parent_case_run_id": self.parent_case_run_id,
            "child_case_run_id": self.child_case_run_id,
            "budget": self.budget,
            "parent_reservation_id": self.parent_reservation_id,
            "parent_authority_ref": dict(self.parent_authority_ref),
            "child_handle": dict(self.child_handle),
            "requested_amount": self.requested_amount,
            "status": self.status,
            "usage": dict(self.usage),
            "attempts": self.attempts,
            "last_error": dict(self.last_error),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BudgetSettlementRecord":
        payload = dict(value or {})
        schema = str(payload.get("schema", ""))
        if schema not in {_SETTLEMENT_SCHEMA_V1, _SETTLEMENT_SCHEMA}:
            raise ValueError("unsupported budget settlement schema")
        parent_authority_ref = dict(payload.get("parent_authority_ref", {}) or {})
        if schema == _SETTLEMENT_SCHEMA_V1:
            parent_authority_ref = minimal_budget_authority_ref(
                dict(payload.get("parent_resource_context", {}) or {})
            )
        return cls(
            settlement_id=str(payload.get("settlement_id", "")),
            project_run_id=str(payload.get("project_run_id", "")),
            parent_case_run_id=str(payload.get("parent_case_run_id", "")),
            child_case_run_id=str(payload.get("child_case_run_id", "")),
            budget=str(payload.get("budget", "")),
            parent_reservation_id=str(payload.get("parent_reservation_id", "")),
            parent_authority_ref=parent_authority_ref,
            child_handle=dict(payload.get("child_handle", {}) or {}),
            requested_amount=int(payload.get("requested_amount", 0) or 0),
            status=str(payload.get("status", "prepared")),
            usage=dict(payload.get("usage", {}) or {}),
            attempts=int(payload.get("attempts", 0) or 0),
            last_error=dict(payload.get("last_error", {}) or {}),
            created_at=float(payload.get("created_at", 0.0) or 0.0),
            updated_at=float(payload.get("updated_at", 0.0) or 0.0),
        )


class SQLiteBudgetSettlementJournal:
    """Independent local WAL recording settlement intent before child execution."""

    def __init__(self, path: Path | str, *, namespace: str) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = str(namespace or "project")
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=10.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with _LOCK, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS budget_settlement_journal (
                    namespace TEXT NOT NULL,
                    settlement_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(namespace, settlement_id)
                )
                """
            )

    def prepare(self, record: BudgetSettlementRecord) -> BudgetSettlementRecord:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload_json FROM budget_settlement_journal WHERE namespace=? AND settlement_id=?",
                (self.namespace, record.settlement_id),
            ).fetchone()
            if row is not None:
                current = BudgetSettlementRecord.from_dict(json.loads(row["payload_json"]))
                if (
                    current.parent_reservation_id != record.parent_reservation_id
                    or current.child_handle != record.child_handle
                ):
                    raise ValueError("budget settlement_id was reused with different authority")
                connection.commit()
                return current
            self._upsert_locked(connection, record)
            connection.commit()
        return record

    def mark_settled(
        self,
        settlement_id: str,
        usage: Mapping[str, Any],
    ) -> BudgetSettlementRecord:
        return self._transition(settlement_id, status="settled", usage=usage, error={})

    def mark_reserved(self, settlement_id: str) -> BudgetSettlementRecord:
        """Confirm the idempotent parent reservation after its WAL intent exists."""

        return self._transition(
            settlement_id,
            status="prepared",
            usage={},
            error={},
            increment_attempts=False,
        )

    def mark_ready(
        self,
        settlement_id: str,
        usage: Mapping[str, Any],
    ) -> BudgetSettlementRecord:
        """Declare that child execution ended and parent settlement may be retried."""

        return self._transition(
            settlement_id,
            status="settlement_ready",
            usage=usage,
            error={},
        )

    def mark_retry(
        self,
        settlement_id: str,
        error: BaseException,
        *,
        usage: Mapping[str, Any] | None = None,
    ) -> BudgetSettlementRecord:
        return self._transition(
            settlement_id,
            status="retry_required",
            usage=dict(usage or {}),
            error={"error_type": type(error).__name__, "message": str(error)},
        )

    def mark_reclaimed(
        self,
        settlement_id: str,
        error: BaseException,
        *,
        usage: Mapping[str, Any] | None = None,
    ) -> BudgetSettlementRecord:
        """Archive debt whose parent lease fence proves it was reclaimed."""

        return self._transition(
            settlement_id,
            status="reclaimed",
            usage=dict(usage or {}),
            error={"error_type": type(error).__name__, "message": str(error)},
        )

    def pending(self, *, project_run_id: str = "") -> tuple[BudgetSettlementRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM budget_settlement_journal "
                "WHERE namespace=? AND status NOT IN ('settled', 'reclaimed') "
                "ORDER BY updated_at",
                (self.namespace,),
            ).fetchall()
        records = tuple(
            BudgetSettlementRecord.from_dict(json.loads(row["payload_json"]))
            for row in rows
        )
        target = str(project_run_id or "")
        return tuple(item for item in records if not target or item.project_run_id == target)

    def retryable(self, *, project_run_id: str = "") -> tuple[BudgetSettlementRecord, ...]:
        """Return only records whose child execution reached a terminal boundary."""

        records = self.pending(project_run_id=project_run_id)
        return tuple(
            item
            for item in records
            if item.status in {"settlement_ready", "retry_required"}
        )

    def _transition(
        self,
        settlement_id: str,
        *,
        status: str,
        usage: Mapping[str, Any],
        error: Mapping[str, Any],
        increment_attempts: bool = True,
    ) -> BudgetSettlementRecord:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload_json FROM budget_settlement_journal WHERE namespace=? AND settlement_id=?",
                (self.namespace, str(settlement_id)),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown budget settlement '{settlement_id}'")
            current = BudgetSettlementRecord.from_dict(json.loads(row["payload_json"]))
            if current.status in {"settled", "reclaimed"}:
                connection.commit()
                return current
            payload = current.as_dict()
            payload.update(
                status=status,
                usage=dict(usage or current.usage),
                attempts=current.attempts + (1 if increment_attempts else 0),
                last_error=dict(error or {}),
                updated_at=time.time(),
            )
            updated = BudgetSettlementRecord.from_dict(payload)
            self._upsert_locked(connection, updated)
            connection.commit()
            return updated

    def _upsert_locked(
        self,
        connection: sqlite3.Connection,
        record: BudgetSettlementRecord,
    ) -> None:
        connection.execute(
            """
            INSERT INTO budget_settlement_journal(namespace, settlement_id, status, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(namespace, settlement_id) DO UPDATE SET
                status=excluded.status,
                payload_json=excluded.payload_json,
                updated_at=excluded.updated_at
            """,
            (
                self.namespace,
                record.settlement_id,
                record.status,
                json.dumps(record.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                record.updated_at,
            ),
        )


def _plain_wire(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_wire(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return _plain_wire(as_dict())
    raise TypeError(
        "budget settlement payload is not transport-safe: "
        f"{type(value).__name__}"
    )


def minimal_budget_authority_ref(resource_context: Mapping[str, Any]) -> dict[str, Any]:
    """Retain only fields required to reopen the parent budget authority."""

    payload = dict(resource_context or {})
    lease = dict(payload.get("lease", {}) or {})
    metadata = dict(payload.get("metadata", {}) or {})
    authority = dict(metadata.get("budget_authority", {}) or {})
    allowed = {
        key: authority[key]
        for key in (
            "backend",
            "namespace",
            "scope",
            "path",
            "redis_url_env",
        )
        if key in authority
    }
    return {
        "lease": {
            "lease_id": str(lease.get("lease_id", "")),
            "fencing_token": int(lease.get("fencing_token", 0) or 0),
        },
        "metadata": {"budget_authority": allowed},
    }


__all__ = [
    "BudgetSettlementRecord",
    "SQLiteBudgetSettlementJournal",
    "minimal_budget_authority_ref",
]
