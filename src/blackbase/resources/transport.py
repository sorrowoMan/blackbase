"""Durable task transport contracts and a SQLite reference backend."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Mapping, Protocol, Sequence
from uuid import uuid4

from .model import TaskEnvelope, TaskResult, WorkerDescriptor


FINAL_TASK_STATES = frozenset({"succeeded", "failed", "cancelled"})


class TaskTransportError(RuntimeError):
    """Base error raised by a task transport."""


class TaskLeaseError(TaskTransportError):
    """Raised when a worker attempts to mutate a task without its active lease."""


@dataclass(frozen=True)
class ClaimedTask:
    """Task plus the opaque lease token owned by one external worker."""

    task: TaskEnvelope
    worker_id: str
    lease_token: str
    attempt: int
    lease_expires_at: float


@dataclass(frozen=True)
class TaskRecord:
    """Auditable durable state for one submitted task."""

    task: TaskEnvelope
    status: str
    attempt: int = 0
    worker_id: str = ""
    lease_expires_at: float = 0.0
    result: TaskResult | None = None
    error: str = ""
    updated_at: float = 0.0

    @property
    def final(self) -> bool:
        return self.status in FINAL_TASK_STATES


class TaskTransport(Protocol):
    """Backend-neutral transport required by an external worker runtime."""

    def submit(self, task: TaskEnvelope | Mapping[str, Any]) -> TaskRecord: ...

    def claim(
        self,
        worker: WorkerDescriptor | Mapping[str, Any],
        *,
        lease_seconds: float = 30.0,
        task_types: Sequence[str] = (),
        namespaces: Sequence[str] = (),
    ) -> ClaimedTask | None: ...

    def heartbeat_task(self, claim: ClaimedTask, *, lease_seconds: float = 30.0) -> bool: ...

    def complete(self, claim: ClaimedTask, result: TaskResult | Mapping[str, Any]) -> TaskRecord: ...

    def fail(self, claim: ClaimedTask, result: TaskResult | Mapping[str, Any]) -> TaskRecord: ...

    def get(self, task_id: str) -> TaskRecord | None: ...

    def recover_expired(self) -> int: ...

    def cancel(self, task_id: str, *, reason: str = "cancelled") -> bool: ...

    def wait_result(
        self,
        task_id: str,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 0.05,
    ) -> TaskResult | None: ...

    def register_worker(
        self,
        worker: WorkerDescriptor | Mapping[str, Any],
    ) -> WorkerDescriptor: ...

    def heartbeat_worker(self, worker_id: str, *, status: str = "online") -> bool: ...

    def list_workers(self, *, max_age_seconds: float | None = None) -> tuple[WorkerDescriptor, ...]: ...

    def counts(self) -> dict[str, int]: ...


class InMemoryTaskTransport:
    """Process-local implementation of the canonical task transport.

    It follows the same claim/fencing contract as the durable transports and
    is intended for tests and single-process execution, not cross-process use.
    """

    def __init__(self) -> None:
        self._records: dict[str, TaskRecord] = {}
        self._workers: dict[str, WorkerDescriptor] = {}
        self._lease_tokens: dict[str, str] = {}
        self._lock = RLock()

    def submit(self, task: TaskEnvelope | Mapping[str, Any]) -> TaskRecord:
        envelope = task if isinstance(task, TaskEnvelope) else TaskEnvelope.from_dict(task)
        with self._lock:
            current = self._records.get(envelope.task_id)
            if current is not None:
                if current.task.as_dict() != envelope.as_dict():
                    raise TaskTransportError(
                        f"task_id='{envelope.task_id}' already exists with a different envelope"
                    )
                return current
            record = TaskRecord(task=envelope, status="queued", updated_at=time.time())
            self._records[envelope.task_id] = record
            return record

    def register_worker(
        self,
        worker: WorkerDescriptor | Mapping[str, Any],
    ) -> WorkerDescriptor:
        descriptor = worker if isinstance(worker, WorkerDescriptor) else WorkerDescriptor.from_dict(worker)
        descriptor = descriptor.heartbeat(status="online")
        with self._lock:
            self._workers[descriptor.worker_id] = descriptor
        return descriptor

    def heartbeat_worker(self, worker_id: str, *, status: str = "online") -> bool:
        with self._lock:
            current = self._workers.get(str(worker_id))
            if current is None:
                return False
            self._workers[current.worker_id] = current.heartbeat(status=status)
            return True

    def list_workers(self, *, max_age_seconds: float | None = None) -> tuple[WorkerDescriptor, ...]:
        now = time.time()
        with self._lock:
            workers = tuple(self._workers.values())
        if max_age_seconds is None:
            return workers
        return tuple(
            item
            for item in workers
            if now - float(item.last_heartbeat_at) <= float(max_age_seconds)
        )

    def claim(
        self,
        worker: WorkerDescriptor | Mapping[str, Any],
        *,
        lease_seconds: float = 30.0,
        task_types: Sequence[str] = (),
        namespaces: Sequence[str] = (),
    ) -> ClaimedTask | None:
        descriptor = self.register_worker(worker)
        accepted_types = {str(item) for item in task_types if str(item)}
        accepted_namespaces = {str(item) for item in namespaces if str(item)}
        now = time.time()
        with self._lock:
            self._recover_expired_locked(now)
            active = sum(
                1
                for item in self._records.values()
                if item.status == "leased" and item.worker_id == descriptor.worker_id
            )
            if active >= descriptor.max_inflight:
                return None
            for task_id, record in self._records.items():
                task = record.task
                if record.status != "queued":
                    continue
                if accepted_types and task.task_type not in accepted_types:
                    continue
                if accepted_namespaces and task.namespace not in accepted_namespaces:
                    continue
                if not descriptor.can_run(
                    task.requirement,
                    executor_backend=task.executor_backend,
                    active_count=active,
                ):
                    continue
                claim = ClaimedTask(
                    task=task,
                    worker_id=descriptor.worker_id,
                    lease_token=uuid4().hex,
                    attempt=int(record.attempt) + 1,
                    lease_expires_at=now + max(0.1, float(lease_seconds)),
                )
                self._records[task_id] = TaskRecord(
                    task=task,
                    status="leased",
                    attempt=claim.attempt,
                    worker_id=claim.worker_id,
                    lease_expires_at=claim.lease_expires_at,
                    updated_at=now,
                )
                self._lease_tokens[task_id] = claim.lease_token
                return claim
        return None

    def heartbeat_task(self, claim: ClaimedTask, *, lease_seconds: float = 30.0) -> bool:
        now = time.time()
        with self._lock:
            record = self._records.get(claim.task.task_id)
            if not self._claim_is_current(record, claim, now=now):
                return False
            self._records[claim.task.task_id] = TaskRecord(
                task=record.task,
                status="leased",
                attempt=record.attempt,
                worker_id=record.worker_id,
                lease_expires_at=now + max(0.1, float(lease_seconds)),
                updated_at=now,
            )
            return True

    def complete(
        self,
        claim: ClaimedTask,
        result: TaskResult | Mapping[str, Any],
    ) -> TaskRecord:
        task_result = result if isinstance(result, TaskResult) else TaskResult.from_dict(result)
        if not task_result.ok:
            raise TaskTransportError("complete() requires a successful TaskResult; use fail()")
        return self._finish(claim, task_result, status="succeeded", allow_retry=False)

    def fail(
        self,
        claim: ClaimedTask,
        result: TaskResult | Mapping[str, Any],
    ) -> TaskRecord:
        task_result = result if isinstance(result, TaskResult) else TaskResult.from_dict(result)
        if task_result.ok:
            raise TaskTransportError("fail() requires a failed TaskResult; use complete()")
        return self._finish(claim, task_result, status="failed", allow_retry=True)

    def _finish(
        self,
        claim: ClaimedTask,
        result: TaskResult,
        *,
        status: str,
        allow_retry: bool,
    ) -> TaskRecord:
        if result.task_id != claim.task.task_id:
            raise TaskLeaseError("TaskResult.task_id does not match the claimed task")
        now = time.time()
        with self._lock:
            current = self._records.get(claim.task.task_id)
            if not self._claim_is_current(current, claim, now=now):
                raise TaskLeaseError(f"task lease is not current for task_id='{claim.task.task_id}'")
            retry = bool(allow_retry and claim.attempt <= claim.task.max_retries)
            record = TaskRecord(
                task=claim.task,
                status="queued" if retry else status,
                attempt=claim.attempt,
                result=None if retry else result,
                error=str(result.error or ""),
                updated_at=now,
            )
            self._records[claim.task.task_id] = record
            self._lease_tokens.pop(claim.task.task_id, None)
            return record

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._records.get(str(task_id))

    def recover_expired(self) -> int:
        with self._lock:
            return self._recover_expired_locked(time.time())

    def _recover_expired_locked(self, now: float) -> int:
        recovered = 0
        for task_id, record in tuple(self._records.items()):
            if record.status == "leased" and record.lease_expires_at <= now:
                self._records[task_id] = TaskRecord(
                    task=record.task,
                    status="queued",
                    attempt=record.attempt,
                    error="task lease expired",
                    updated_at=now,
                )
                self._lease_tokens.pop(task_id, None)
                recovered += 1
        return recovered

    def _claim_is_current(
        self,
        record: TaskRecord | None,
        claim: ClaimedTask,
        *,
        now: float,
    ) -> bool:
        return bool(
            record is not None
            and record.status == "leased"
            and record.worker_id == claim.worker_id
            and record.attempt == claim.attempt
            and self._lease_tokens.get(claim.task.task_id) == claim.lease_token
            and record.lease_expires_at > now
        )

    def cancel(self, task_id: str, *, reason: str = "cancelled") -> bool:
        with self._lock:
            current = self._records.get(str(task_id))
            if current is None or current.final:
                return False
            self._records[str(task_id)] = TaskRecord(
                task=current.task,
                status="cancelled",
                attempt=current.attempt,
                error=str(reason),
                updated_at=time.time(),
            )
            self._lease_tokens.pop(str(task_id), None)
            return True

    def wait_result(
        self,
        task_id: str,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 0.05,
    ) -> TaskResult | None:
        deadline = None if timeout_seconds is None else time.monotonic() + max(0.0, float(timeout_seconds))
        poll = max(0.001, float(poll_interval_seconds))
        while True:
            record = self.get(task_id)
            if record is None:
                raise TaskTransportError(f"Unknown task_id='{task_id}'")
            if record.final:
                return record.result
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(poll)

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self._lock:
            for record in self._records.values():
                counts[record.status] = counts.get(record.status, 0) + 1
        return counts


class SQLiteTaskTransport:
    """SQLite-backed at-least-once task queue with atomic worker claims."""

    def __init__(self, path: Path | str, *, busy_timeout_seconds: float = 10.0) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    envelope_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT NOT NULL DEFAULT '',
                    lease_token TEXT NOT NULL DEFAULT '',
                    lease_expires_at REAL NOT NULL DEFAULT 0,
                    result_json TEXT,
                    error TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_claim
                ON tasks(status, created_at, task_id)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    descriptor_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    last_heartbeat_at REAL NOT NULL
                )
                """
            )

    def submit(self, task: TaskEnvelope | Mapping[str, Any]) -> TaskRecord:
        envelope = task if isinstance(task, TaskEnvelope) else TaskEnvelope.from_dict(task)
        envelope_json = _dump_json(envelope.as_dict(), label=f"TaskEnvelope '{envelope.task_id}'")
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT envelope_json FROM tasks WHERE task_id = ?",
                (envelope.task_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["envelope_json"]) != envelope_json:
                    connection.rollback()
                    raise TaskTransportError(
                        f"task_id='{envelope.task_id}' already exists with a different envelope"
                    )
                connection.commit()
                record = self.get(envelope.task_id)
                if record is None:  # pragma: no cover - guarded by the transaction
                    raise TaskTransportError(f"task_id='{envelope.task_id}' disappeared")
                return record
            connection.execute(
                """
                INSERT INTO tasks(
                    task_id, envelope_json, status, created_at, updated_at
                ) VALUES (?, ?, 'queued', ?, ?)
                """,
                (envelope.task_id, envelope_json, now, now),
            )
            connection.commit()
        record = self.get(envelope.task_id)
        if record is None:  # pragma: no cover - guarded by insert success
            raise TaskTransportError(f"failed to persist task_id='{envelope.task_id}'")
        return record

    def register_worker(
        self,
        worker: WorkerDescriptor | Mapping[str, Any],
    ) -> WorkerDescriptor:
        descriptor = worker if isinstance(worker, WorkerDescriptor) else WorkerDescriptor.from_dict(worker)
        descriptor = descriptor.heartbeat(status="online")
        payload = _dump_json(descriptor.as_dict(), label=f"WorkerDescriptor '{descriptor.worker_id}'")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workers(worker_id, descriptor_json, status, last_heartbeat_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    descriptor_json = excluded.descriptor_json,
                    status = excluded.status,
                    last_heartbeat_at = excluded.last_heartbeat_at
                """,
                (
                    descriptor.worker_id,
                    payload,
                    descriptor.status,
                    descriptor.last_heartbeat_at,
                ),
            )
        return descriptor

    def heartbeat_worker(self, worker_id: str, *, status: str = "online") -> bool:
        now = time.time()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT descriptor_json FROM workers WHERE worker_id = ?",
                (str(worker_id),),
            ).fetchone()
            if row is None:
                return False
            descriptor = WorkerDescriptor.from_dict(json.loads(str(row["descriptor_json"]))).heartbeat(
                status=status,
                at=now,
            )
            connection.execute(
                """
                UPDATE workers
                SET descriptor_json = ?, status = ?, last_heartbeat_at = ?
                WHERE worker_id = ?
                """,
                (
                    _dump_json(descriptor.as_dict(), label="worker heartbeat"),
                    descriptor.status,
                    now,
                    descriptor.worker_id,
                ),
            )
        return True

    def list_workers(self, *, max_age_seconds: float | None = None) -> tuple[WorkerDescriptor, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT descriptor_json, last_heartbeat_at FROM workers ORDER BY worker_id"
            ).fetchall()
        now = time.time()
        workers = []
        for row in rows:
            if max_age_seconds is not None:
                age = now - float(row["last_heartbeat_at"] or 0.0)
                if age > float(max_age_seconds):
                    continue
            workers.append(WorkerDescriptor.from_dict(json.loads(str(row["descriptor_json"]))))
        return tuple(workers)

    def claim(
        self,
        worker: WorkerDescriptor | Mapping[str, Any],
        *,
        lease_seconds: float = 30.0,
        task_types: Sequence[str] = (),
        namespaces: Sequence[str] = (),
    ) -> ClaimedTask | None:
        descriptor = self.register_worker(worker)
        lease_seconds = max(0.1, float(lease_seconds))
        accepted_types = {str(item) for item in task_types if str(item)}
        accepted_namespaces = {str(item) for item in namespaces if str(item)}
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._recover_expired_locked(connection, now=now)
            active_count = int(
                connection.execute(
                    "SELECT COUNT(*) AS count FROM tasks WHERE status = 'leased' AND worker_id = ?",
                    (descriptor.worker_id,),
                ).fetchone()["count"]
            )
            if active_count >= descriptor.max_inflight:
                connection.commit()
                return None
            rows = connection.execute(
                """
                SELECT task_id, envelope_json, attempt
                FROM tasks
                WHERE status = 'queued'
                ORDER BY created_at, task_id
                """
            ).fetchall()
            selected: tuple[sqlite3.Row, TaskEnvelope] | None = None
            for row in rows:
                envelope = TaskEnvelope.from_dict(json.loads(str(row["envelope_json"])))
                if accepted_types and envelope.task_type not in accepted_types:
                    continue
                if accepted_namespaces and envelope.namespace not in accepted_namespaces:
                    continue
                if not descriptor.can_run(
                    envelope.requirement,
                    executor_backend=envelope.executor_backend,
                    active_count=active_count,
                ):
                    continue
                selected = (row, envelope)
                break
            if selected is None:
                connection.commit()
                return None
            row, envelope = selected
            lease_token = uuid4().hex
            attempt = int(row["attempt"] or 0) + 1
            expires_at = now + lease_seconds
            updated = connection.execute(
                """
                UPDATE tasks
                SET status = 'leased', attempt = ?, worker_id = ?, lease_token = ?,
                    lease_expires_at = ?, updated_at = ?
                WHERE task_id = ? AND status = 'queued'
                """,
                (
                    attempt,
                    descriptor.worker_id,
                    lease_token,
                    expires_at,
                    now,
                    envelope.task_id,
                ),
            ).rowcount
            if updated != 1:  # pragma: no cover - BEGIN IMMEDIATE serializes claimers
                connection.rollback()
                return None
            connection.commit()
        return ClaimedTask(
            task=envelope,
            worker_id=descriptor.worker_id,
            lease_token=lease_token,
            attempt=attempt,
            lease_expires_at=expires_at,
        )

    def heartbeat_task(self, claim: ClaimedTask, *, lease_seconds: float = 30.0) -> bool:
        now = time.time()
        expires_at = now + max(0.1, float(lease_seconds))
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE tasks SET lease_expires_at = ?, updated_at = ?
                WHERE task_id = ? AND status = 'leased'
                    AND worker_id = ? AND lease_token = ? AND lease_expires_at > ?
                """,
                (
                    expires_at,
                    now,
                    claim.task.task_id,
                    claim.worker_id,
                    claim.lease_token,
                    now,
                ),
            ).rowcount
        return updated == 1

    def complete(
        self,
        claim: ClaimedTask,
        result: TaskResult | Mapping[str, Any],
    ) -> TaskRecord:
        task_result = result if isinstance(result, TaskResult) else TaskResult.from_dict(result)
        if task_result.task_id != claim.task.task_id:
            raise TaskLeaseError("TaskResult.task_id does not match the claimed task")
        if not task_result.ok:
            raise TaskTransportError("complete() requires a successful TaskResult; use fail()")
        self._finish_claim(claim, task_result, status="succeeded", allow_retry=False)
        record = self.get(claim.task.task_id)
        if record is None:  # pragma: no cover
            raise TaskTransportError("completed task disappeared")
        return record

    def fail(
        self,
        claim: ClaimedTask,
        result: TaskResult | Mapping[str, Any],
    ) -> TaskRecord:
        task_result = result if isinstance(result, TaskResult) else TaskResult.from_dict(result)
        if task_result.task_id != claim.task.task_id:
            raise TaskLeaseError("TaskResult.task_id does not match the claimed task")
        if task_result.ok:
            raise TaskTransportError("fail() requires a non-success TaskResult; use complete()")
        self._finish_claim(claim, task_result, status="failed", allow_retry=True)
        record = self.get(claim.task.task_id)
        if record is None:  # pragma: no cover
            raise TaskTransportError("failed task disappeared")
        return record

    def _finish_claim(
        self,
        claim: ClaimedTask,
        result: TaskResult,
        *,
        status: str,
        allow_retry: bool,
    ) -> None:
        now = time.time()
        retry = allow_retry and claim.attempt <= claim.task.max_retries
        next_status = "queued" if retry else status
        result_json = None if retry else _dump_json(result.as_dict(), label="TaskResult")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """
                UPDATE tasks
                SET status = ?, worker_id = '', lease_token = '', lease_expires_at = 0,
                    result_json = ?, error = ?, updated_at = ?
                WHERE task_id = ? AND status = 'leased'
                    AND worker_id = ? AND lease_token = ? AND lease_expires_at > ?
                """,
                (
                    next_status,
                    result_json,
                    result.error,
                    now,
                    claim.task.task_id,
                    claim.worker_id,
                    claim.lease_token,
                    now,
                ),
            ).rowcount
            if updated != 1:
                connection.rollback()
                raise TaskLeaseError(
                    f"Task lease is no longer active for task_id='{claim.task.task_id}'"
                )
            connection.commit()

    def recover_expired(self) -> int:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            recovered = self._recover_expired_locked(connection, now=time.time())
            connection.commit()
        return recovered

    def _recover_expired_locked(self, connection: sqlite3.Connection, *, now: float) -> int:
        rows = connection.execute(
            """
            SELECT task_id, envelope_json, attempt, worker_id
            FROM tasks
            WHERE status = 'leased' AND lease_expires_at <= ?
            """,
            (now,),
        ).fetchall()
        recovered = 0
        for row in rows:
            envelope = TaskEnvelope.from_dict(json.loads(str(row["envelope_json"])))
            attempt = int(row["attempt"] or 0)
            if attempt <= envelope.max_retries:
                next_status = "queued"
                result_json = None
            else:
                next_status = "failed"
                result_json = _dump_json(
                    TaskResult.failure(
                        task_id=envelope.task_id,
                        error=f"worker lease expired after attempt {attempt}",
                        worker_id=str(row["worker_id"] or ""),
                    ).as_dict(),
                    label="expired TaskResult",
                )
            connection.execute(
                """
                UPDATE tasks
                SET status = ?, worker_id = '', lease_token = '', lease_expires_at = 0,
                    result_json = ?, error = ?, updated_at = ?
                WHERE task_id = ? AND status = 'leased'
                """,
                (
                    next_status,
                    result_json,
                    f"worker lease expired after attempt {attempt}",
                    now,
                    envelope.task_id,
                ),
            )
            recovered += 1
        return recovered

    def cancel(self, task_id: str, *, reason: str = "cancelled") -> bool:
        result = TaskResult.failure(task_id=str(task_id), error=reason, status="cancelled")
        now = time.time()
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE tasks
                SET status = 'cancelled', result_json = ?, error = ?, updated_at = ?
                WHERE task_id = ? AND status = 'queued'
                """,
                (
                    _dump_json(result.as_dict(), label="cancelled TaskResult"),
                    reason,
                    now,
                    str(task_id),
                ),
            ).rowcount
        return updated == 1

    def get(self, task_id: str) -> TaskRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (str(task_id),),
            ).fetchone()
        return None if row is None else _record_from_row(row)

    def wait_result(
        self,
        task_id: str,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 0.05,
    ) -> TaskResult | None:
        deadline = None if timeout_seconds is None else time.monotonic() + max(0.0, timeout_seconds)
        poll = max(0.001, float(poll_interval_seconds))
        while True:
            self.recover_expired()
            record = self.get(task_id)
            if record is None:
                raise TaskTransportError(f"Unknown task_id='{task_id}'")
            if record.final:
                return record.result
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(poll)

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status"
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}


class RedisTaskTransport:
    """Redis-backed at-least-once task transport using one distributed state lock.

    Task records are stored as immutable-envelope JSON values. Queue, lease, and
    record mutations use Redis transactions while a namespace-scoped Redis lock
    serializes capability-aware selection. This keeps the contract correct across
    processes without putting framework-specific execution semantics in Redis.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        *,
        namespace: str = "blackbase:tasks",
        client: Any = None,
        lock_timeout_seconds: float = 30.0,
        blocking_timeout_seconds: float = 10.0,
    ) -> None:
        self.redis_url = str(redis_url)
        self.namespace = str(namespace or "blackbase:tasks").strip().rstrip(":")
        self.client = client if client is not None else _make_redis_client(self.redis_url)
        self.lock_timeout_seconds = max(1.0, float(lock_timeout_seconds))
        self.blocking_timeout_seconds = max(0.1, float(blocking_timeout_seconds))
        self._queue_key = f"{self.namespace}:queue"
        self._tasks_key = f"{self.namespace}:task_ids"
        self._leased_key = f"{self.namespace}:leased"
        self._workers_key = f"{self.namespace}:worker_ids"
        self._lock_key = f"{self.namespace}:state_lock"

    def _task_key(self, task_id: str) -> str:
        return f"{self.namespace}:task:{str(task_id)}"

    def _worker_key(self, worker_id: str) -> str:
        return f"{self.namespace}:worker:{str(worker_id)}"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        lock_factory = getattr(self.client, "lock", None)
        if not callable(lock_factory):
            raise TaskTransportError("Redis client must provide redis-py compatible lock()")
        lock = lock_factory(
            self._lock_key,
            timeout=self.lock_timeout_seconds,
            blocking_timeout=self.blocking_timeout_seconds,
        )
        if not lock.acquire(blocking=True):
            raise TaskTransportError(
                f"Timed out acquiring Redis task transport lock '{self._lock_key}'"
            )
        try:
            yield
        finally:
            active_error = sys.exc_info()[0] is not None
            try:
                lock.release()
            except Exception as exc:  # pragma: no cover - redis connection/ownership failure
                if not active_error:
                    raise TaskTransportError("Lost Redis task transport lock") from exc

    def submit(self, task: TaskEnvelope | Mapping[str, Any]) -> TaskRecord:
        envelope = task if isinstance(task, TaskEnvelope) else TaskEnvelope.from_dict(task)
        envelope_json = _dump_json(envelope.as_dict(), label=f"TaskEnvelope '{envelope.task_id}'")
        now = time.time()
        with self._locked():
            existing = self._load_state(envelope.task_id)
            if existing is not None:
                current_json = _dump_json(
                    TaskEnvelope.from_dict(existing["task"]).as_dict(),
                    label="existing TaskEnvelope",
                )
                if current_json != envelope_json:
                    raise TaskTransportError(
                        f"task_id='{envelope.task_id}' already exists with a different envelope"
                    )
                return _task_record_from_state(existing)
            state = _new_redis_task_state(envelope, now=now)
            pipe = self.client.pipeline(transaction=True)
            pipe.set(self._task_key(envelope.task_id), _dump_json(state, label="Redis task record"))
            pipe.sadd(self._tasks_key, envelope.task_id)
            pipe.rpush(self._queue_key, envelope.task_id)
            pipe.execute()
        record = self.get(envelope.task_id)
        if record is None:  # pragma: no cover - guarded by transaction
            raise TaskTransportError(f"failed to persist task_id='{envelope.task_id}'")
        return record

    def register_worker(
        self,
        worker: WorkerDescriptor | Mapping[str, Any],
    ) -> WorkerDescriptor:
        descriptor = worker if isinstance(worker, WorkerDescriptor) else WorkerDescriptor.from_dict(worker)
        descriptor = descriptor.heartbeat(status="online")
        payload = _dump_json(descriptor.as_dict(), label=f"WorkerDescriptor '{descriptor.worker_id}'")
        with self._locked():
            pipe = self.client.pipeline(transaction=True)
            pipe.set(self._worker_key(descriptor.worker_id), payload)
            pipe.sadd(self._workers_key, descriptor.worker_id)
            pipe.execute()
        return descriptor

    def heartbeat_worker(self, worker_id: str, *, status: str = "online") -> bool:
        with self._locked():
            raw = self.client.get(self._worker_key(worker_id))
            if raw is None:
                return False
            descriptor = WorkerDescriptor.from_dict(json.loads(_decode_redis(raw))).heartbeat(
                status=status
            )
            self.client.set(
                self._worker_key(descriptor.worker_id),
                _dump_json(descriptor.as_dict(), label="worker heartbeat"),
            )
        return True

    def list_workers(self, *, max_age_seconds: float | None = None) -> tuple[WorkerDescriptor, ...]:
        now = time.time()
        workers: list[WorkerDescriptor] = []
        for worker_id in sorted(_decode_redis(item) for item in self.client.smembers(self._workers_key)):
            raw = self.client.get(self._worker_key(worker_id))
            if raw is None:
                continue
            descriptor = WorkerDescriptor.from_dict(json.loads(_decode_redis(raw)))
            if max_age_seconds is not None:
                if now - float(descriptor.last_heartbeat_at) > float(max_age_seconds):
                    continue
            workers.append(descriptor)
        return tuple(workers)

    def claim(
        self,
        worker: WorkerDescriptor | Mapping[str, Any],
        *,
        lease_seconds: float = 30.0,
        task_types: Sequence[str] = (),
        namespaces: Sequence[str] = (),
    ) -> ClaimedTask | None:
        descriptor = self.register_worker(worker)
        lease_seconds = max(0.1, float(lease_seconds))
        accepted_types = {str(item) for item in task_types if str(item)}
        accepted_namespaces = {str(item) for item in namespaces if str(item)}
        now = time.time()
        with self._locked():
            self._recover_expired_locked(now=now)
            active_count = 0
            for task_id in self._leased_ids():
                state = self._load_state(task_id)
                if state is not None and str(state.get("worker_id", "")) == descriptor.worker_id:
                    active_count += 1
            if active_count >= descriptor.max_inflight:
                return None
            selected: tuple[str, dict[str, Any], TaskEnvelope] | None = None
            stale_queue_ids: list[str] = []
            for raw_task_id in self.client.lrange(self._queue_key, 0, -1):
                task_id = _decode_redis(raw_task_id)
                state = self._load_state(task_id)
                if state is None or str(state.get("status", "")) != "queued":
                    stale_queue_ids.append(task_id)
                    continue
                envelope = TaskEnvelope.from_dict(dict(state["task"]))
                if accepted_types and envelope.task_type not in accepted_types:
                    continue
                if accepted_namespaces and envelope.namespace not in accepted_namespaces:
                    continue
                if not descriptor.can_run(
                    envelope.requirement,
                    executor_backend=envelope.executor_backend,
                    active_count=active_count,
                ):
                    continue
                selected = (task_id, state, envelope)
                break
            for stale_id in stale_queue_ids:
                self.client.lrem(self._queue_key, 0, stale_id)
            if selected is None:
                return None
            task_id, state, envelope = selected
            attempt = int(state.get("attempt", 0) or 0) + 1
            lease_token = uuid4().hex
            expires_at = now + lease_seconds
            state.update(
                {
                    "status": "leased",
                    "attempt": attempt,
                    "worker_id": descriptor.worker_id,
                    "lease_token": lease_token,
                    "lease_expires_at": expires_at,
                    "result": None,
                    "updated_at": now,
                }
            )
            pipe = self.client.pipeline(transaction=True)
            pipe.set(self._task_key(task_id), _dump_json(state, label="Redis leased task"))
            pipe.lrem(self._queue_key, 1, task_id)
            pipe.sadd(self._leased_key, task_id)
            pipe.execute()
        return ClaimedTask(
            task=envelope,
            worker_id=descriptor.worker_id,
            lease_token=lease_token,
            attempt=attempt,
            lease_expires_at=expires_at,
        )

    def heartbeat_task(self, claim: ClaimedTask, *, lease_seconds: float = 30.0) -> bool:
        now = time.time()
        with self._locked():
            state = self._load_state(claim.task.task_id)
            if not _redis_claim_is_active(state, claim, now=now):
                return False
            state["lease_expires_at"] = now + max(0.1, float(lease_seconds))
            state["updated_at"] = now
            self.client.set(
                self._task_key(claim.task.task_id),
                _dump_json(state, label="Redis task heartbeat"),
            )
        return True

    def complete(
        self,
        claim: ClaimedTask,
        result: TaskResult | Mapping[str, Any],
    ) -> TaskRecord:
        task_result = result if isinstance(result, TaskResult) else TaskResult.from_dict(result)
        if task_result.task_id != claim.task.task_id:
            raise TaskLeaseError("TaskResult.task_id does not match the claimed task")
        if not task_result.ok:
            raise TaskTransportError("complete() requires a successful TaskResult; use fail()")
        return self._finish_claim(claim, task_result, status="succeeded", allow_retry=False)

    def fail(
        self,
        claim: ClaimedTask,
        result: TaskResult | Mapping[str, Any],
    ) -> TaskRecord:
        task_result = result if isinstance(result, TaskResult) else TaskResult.from_dict(result)
        if task_result.task_id != claim.task.task_id:
            raise TaskLeaseError("TaskResult.task_id does not match the claimed task")
        if task_result.ok:
            raise TaskTransportError("fail() requires a non-success TaskResult; use complete()")
        return self._finish_claim(claim, task_result, status="failed", allow_retry=True)

    def _finish_claim(
        self,
        claim: ClaimedTask,
        result: TaskResult,
        *,
        status: str,
        allow_retry: bool,
    ) -> TaskRecord:
        now = time.time()
        with self._locked():
            state = self._load_state(claim.task.task_id)
            if not _redis_claim_is_active(state, claim, now=now):
                self._recover_expired_locked(now=now)
                raise TaskLeaseError(
                    f"Task lease is no longer active for task_id='{claim.task.task_id}'"
                )
            retry = allow_retry and claim.attempt <= claim.task.max_retries
            state.update(
                {
                    "status": "queued" if retry else status,
                    "worker_id": "",
                    "lease_token": "",
                    "lease_expires_at": 0.0,
                    "result": None if retry else result.as_dict(),
                    "error": str(result.error or ""),
                    "updated_at": now,
                }
            )
            pipe = self.client.pipeline(transaction=True)
            pipe.set(
                self._task_key(claim.task.task_id),
                _dump_json(state, label="Redis finished task"),
            )
            pipe.srem(self._leased_key, claim.task.task_id)
            if retry:
                pipe.rpush(self._queue_key, claim.task.task_id)
            pipe.execute()
        record = self.get(claim.task.task_id)
        if record is None:  # pragma: no cover - guarded by transaction
            raise TaskTransportError("finished task disappeared")
        return record

    def recover_expired(self) -> int:
        with self._locked():
            return self._recover_expired_locked(now=time.time())

    def _recover_expired_locked(self, *, now: float) -> int:
        recovered = 0
        for task_id in self._leased_ids():
            state = self._load_state(task_id)
            if state is None:
                self.client.srem(self._leased_key, task_id)
                continue
            if str(state.get("status", "")) != "leased":
                self.client.srem(self._leased_key, task_id)
                continue
            if float(state.get("lease_expires_at", 0.0) or 0.0) > now:
                continue
            envelope = TaskEnvelope.from_dict(dict(state["task"]))
            attempt = int(state.get("attempt", 0) or 0)
            error = f"worker lease expired after attempt {attempt}"
            retry = attempt <= envelope.max_retries
            result = None
            if not retry:
                result = TaskResult.failure(
                    task_id=envelope.task_id,
                    error=error,
                    worker_id=str(state.get("worker_id", "") or ""),
                ).as_dict()
            state.update(
                {
                    "status": "queued" if retry else "failed",
                    "worker_id": "",
                    "lease_token": "",
                    "lease_expires_at": 0.0,
                    "result": result,
                    "error": error,
                    "updated_at": now,
                }
            )
            pipe = self.client.pipeline(transaction=True)
            pipe.set(self._task_key(task_id), _dump_json(state, label="Redis recovered task"))
            pipe.srem(self._leased_key, task_id)
            if retry:
                pipe.rpush(self._queue_key, task_id)
            pipe.execute()
            recovered += 1
        return recovered

    def cancel(self, task_id: str, *, reason: str = "cancelled") -> bool:
        now = time.time()
        with self._locked():
            state = self._load_state(task_id)
            if state is None or str(state.get("status", "")) != "queued":
                return False
            result = TaskResult.failure(
                task_id=str(task_id),
                error=str(reason),
                status="cancelled",
            )
            state.update(
                {
                    "status": "cancelled",
                    "result": result.as_dict(),
                    "error": str(reason),
                    "updated_at": now,
                }
            )
            pipe = self.client.pipeline(transaction=True)
            pipe.set(self._task_key(task_id), _dump_json(state, label="Redis cancelled task"))
            pipe.lrem(self._queue_key, 0, str(task_id))
            pipe.execute()
        return True

    def get(self, task_id: str) -> TaskRecord | None:
        state = self._load_state(task_id)
        return None if state is None else _task_record_from_state(state)

    def wait_result(
        self,
        task_id: str,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 0.05,
    ) -> TaskResult | None:
        deadline = None if timeout_seconds is None else time.monotonic() + max(0.0, timeout_seconds)
        poll = max(0.001, float(poll_interval_seconds))
        while True:
            self.recover_expired()
            record = self.get(task_id)
            if record is None:
                raise TaskTransportError(f"Unknown task_id='{task_id}'")
            if record.final:
                return record.result
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(poll)

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for task_id in sorted(_decode_redis(item) for item in self.client.smembers(self._tasks_key)):
            state = self._load_state(task_id)
            if state is None:
                continue
            status = str(state.get("status", "unknown"))
            counts[status] = int(counts.get(status, 0)) + 1
        return counts

    def _load_state(self, task_id: str) -> dict[str, Any] | None:
        raw = self.client.get(self._task_key(task_id))
        if raw is None:
            return None
        value = json.loads(_decode_redis(raw))
        if not isinstance(value, Mapping):
            raise TaskTransportError(f"Invalid Redis task record for task_id='{task_id}'")
        return dict(value)

    def _leased_ids(self) -> tuple[str, ...]:
        return tuple(_decode_redis(item) for item in self.client.smembers(self._leased_key))


def _record_from_row(row: sqlite3.Row) -> TaskRecord:
    result_payload = row["result_json"]
    return TaskRecord(
        task=TaskEnvelope.from_dict(json.loads(str(row["envelope_json"]))),
        status=str(row["status"]),
        attempt=int(row["attempt"] or 0),
        worker_id=str(row["worker_id"] or ""),
        lease_expires_at=float(row["lease_expires_at"] or 0.0),
        result=None if not result_payload else TaskResult.from_dict(json.loads(str(result_payload))),
        error=str(row["error"] or ""),
        updated_at=float(row["updated_at"] or 0.0),
    )


def _new_redis_task_state(envelope: TaskEnvelope, *, now: float) -> dict[str, Any]:
    return {
        "task": envelope.as_dict(),
        "status": "queued",
        "attempt": 0,
        "worker_id": "",
        "lease_token": "",
        "lease_expires_at": 0.0,
        "result": None,
        "error": "",
        "updated_at": float(now),
    }


def _task_record_from_state(state: Mapping[str, Any]) -> TaskRecord:
    result = state.get("result")
    return TaskRecord(
        task=TaskEnvelope.from_dict(dict(state.get("task", {}) or {})),
        status=str(state.get("status", "unknown")),
        attempt=int(state.get("attempt", 0) or 0),
        worker_id=str(state.get("worker_id", "") or ""),
        lease_expires_at=float(state.get("lease_expires_at", 0.0) or 0.0),
        result=(
            TaskResult.from_dict(dict(result))
            if isinstance(result, Mapping)
            else None
        ),
        error=str(state.get("error", "") or ""),
        updated_at=float(state.get("updated_at", 0.0) or 0.0),
    )


def _redis_claim_is_active(
    state: Mapping[str, Any] | None,
    claim: ClaimedTask,
    *,
    now: float,
) -> bool:
    if state is None:
        return False
    return (
        str(state.get("status", "")) == "leased"
        and str(state.get("worker_id", "")) == claim.worker_id
        and str(state.get("lease_token", "")) == claim.lease_token
        and float(state.get("lease_expires_at", 0.0) or 0.0) > float(now)
    )


def _decode_redis(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _make_redis_client(redis_url: str) -> Any:
    try:
        import redis  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("RedisTaskTransport requires the optional 'redis' package") from exc
    return redis.from_url(str(redis_url))


def _dump_json(value: Mapping[str, Any], *, label: str) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be JSON serializable: {exc}") from exc


__all__ = [
    "ClaimedTask",
    "FINAL_TASK_STATES",
    "RedisTaskTransport",
    "SQLiteTaskTransport",
    "TaskLeaseError",
    "TaskRecord",
    "TaskTransport",
    "TaskTransportError",
]
