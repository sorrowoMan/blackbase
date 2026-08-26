"""Durable state machine for Evaluation Event evidence.

The journal is intentionally small: large candidate and feedback payloads stay
in :class:`SnapshotStore`.  One journal record only points at those snapshots
and records whether the control-plane decision is still being prepared,
durably pending, being committed, or terminal.

The state machine closes the process-crash window between an Evaluation Event
and its disposition without pretending that evaluation can always be replayed.
Semantic frameworks decide whether an unresolved event is recoverable; this
module only provides atomic, idempotent evidence transitions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import hashlib
import json
import sqlite3
import threading
import time

from blackbase.evaluation.event import (
    EvaluationDispositionEnvelope,
    EvaluationDispositionVerificationReceipt,
    evaluation_disposition_digest,
)
from blackbase.wire import freeze_wire_mapping, thaw_wire_mapping


EVALUATION_EVIDENCE_RECORD_SCHEMA_V1 = "blackbase.evaluation_evidence_record/v1"
EVALUATION_EVIDENCE_RECORD_SCHEMA_V2 = "blackbase.evaluation_evidence_record/v2"
EVALUATION_EVIDENCE_ACTIVE_STATUSES = frozenset(
    {"preparing", "pending", "deciding"}
)
EVALUATION_EVIDENCE_TERMINAL_STATUSES = frozenset(
    {"committed", "rejected", "failed", "abandoned"}
)


class EvaluationEvidenceConflict(RuntimeError):
    """The requested transition contradicts already durable evidence."""


class EvaluationEvidenceRevisionConflict(EvaluationEvidenceConflict):
    """The caller tried to update a stale journal revision."""


@dataclass(frozen=True)
class EvaluationEvidenceRecord:
    """One compact, versioned journal entry for an Evaluation Event."""

    event_id: str
    run_id: str
    event_snapshot_key: str
    status: str = "preparing"
    disposition: Mapping[str, Any] | None = None
    disposition_snapshot_key: str = ""
    verification: Mapping[str, Any] | None = None
    identity: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    revision: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        event_id = str(self.event_id or "").strip()
        run_id = str(self.run_id or "").strip()
        event_snapshot_key = str(self.event_snapshot_key or "").strip()
        status = str(self.status or "").strip().lower()
        revision = int(self.revision)
        created_at = float(self.created_at)
        updated_at = float(self.updated_at)
        if not event_id or not run_id or not event_snapshot_key:
            raise ValueError(
                "evaluation evidence requires event_id, run_id, and event_snapshot_key"
            )
        if status not in (
            EVALUATION_EVIDENCE_ACTIVE_STATUSES
            | EVALUATION_EVIDENCE_TERMINAL_STATUSES
        ):
            raise ValueError(f"unsupported evaluation evidence status: {status}")
        if revision < 1:
            raise ValueError("evaluation evidence revision must be positive")
        if created_at <= 0 or updated_at <= 0 or updated_at < created_at:
            raise ValueError("evaluation evidence timestamps are invalid")

        disposition_payload: Mapping[str, Any] | None = None
        if self.disposition is not None:
            envelope = EvaluationDispositionEnvelope.from_dict(self.disposition)
            if envelope.event_id != event_id:
                raise ValueError("evaluation evidence disposition event_id mismatch")
            disposition_payload = freeze_wire_mapping(
                envelope.as_dict(),
                path="evaluation_evidence.disposition",
            )
        if status == "deciding" and disposition_payload is None:
            raise ValueError("deciding evaluation evidence requires a disposition intent")
        if status in {"committed", "rejected", "failed"}:
            if disposition_payload is None:
                raise ValueError(f"{status} evaluation evidence requires a disposition")
            envelope = EvaluationDispositionEnvelope.from_dict(disposition_payload)
            if envelope.status != status:
                raise ValueError(
                    "evaluation evidence terminal status disagrees with disposition"
                )
        if status in {"preparing", "pending"} and disposition_payload is not None:
            raise ValueError(f"{status} evaluation evidence cannot carry a disposition")

        verification_payload: Mapping[str, Any] | None = None
        if self.verification is not None:
            receipt = EvaluationDispositionVerificationReceipt.from_dict(
                self.verification
            )
            if disposition_payload is None:
                raise ValueError(
                    "evaluation evidence verification requires a disposition"
                )
            envelope = EvaluationDispositionEnvelope.from_dict(disposition_payload)
            if receipt.event_id != event_id:
                raise ValueError("evaluation evidence verification event_id mismatch")
            if receipt.event_snapshot_key != event_snapshot_key:
                raise ValueError(
                    "evaluation evidence verification Event Snapshot mismatch"
                )
            if receipt.disposition_digest != evaluation_disposition_digest(envelope):
                raise ValueError(
                    "evaluation evidence verification disposition digest mismatch"
                )
            if status not in {"committed", "rejected", "failed"}:
                raise ValueError(
                    "only a decision terminal evidence record may carry verification"
                )
            verification_payload = freeze_wire_mapping(
                receipt.as_dict(),
                path="evaluation_evidence.verification",
            )

        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "event_snapshot_key", event_snapshot_key)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "disposition", disposition_payload)
        object.__setattr__(self, "verification", verification_payload)
        object.__setattr__(
            self,
            "disposition_snapshot_key",
            str(self.disposition_snapshot_key or "").strip(),
        )
        object.__setattr__(
            self,
            "identity",
            freeze_wire_mapping(self.identity, path="evaluation_evidence.identity"),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_wire_mapping(self.metadata, path="evaluation_evidence.metadata"),
        )
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)

    @property
    def terminal(self) -> bool:
        return self.status in EVALUATION_EVIDENCE_TERMINAL_STATUSES

    @property
    def terminal_verified(self) -> bool:
        return (
            self.status in {"committed", "rejected", "failed"}
            and self.verification is not None
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": EVALUATION_EVIDENCE_RECORD_SCHEMA_V2,
            "event_id": self.event_id,
            "run_id": self.run_id,
            "event_snapshot_key": self.event_snapshot_key,
            "status": self.status,
            "disposition": (
                None
                if self.disposition is None
                else thaw_wire_mapping(self.disposition)
            ),
            "disposition_snapshot_key": self.disposition_snapshot_key,
            "verification": (
                None
                if self.verification is None
                else thaw_wire_mapping(self.verification)
            ),
            "identity": thaw_wire_mapping(self.identity),
            "metadata": thaw_wire_mapping(self.metadata),
            "revision": self.revision,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvaluationEvidenceRecord":
        data = dict(payload or {})
        schema = str(data.get("schema", "") or "")
        if schema not in {
            EVALUATION_EVIDENCE_RECORD_SCHEMA_V1,
            EVALUATION_EVIDENCE_RECORD_SCHEMA_V2,
        }:
            raise ValueError(
                f"unsupported evaluation evidence schema: {schema or '<missing>'}"
            )
        disposition = data.get("disposition")
        verification = data.get("verification")
        return cls(
            event_id=str(data.get("event_id", "")),
            run_id=str(data.get("run_id", "")),
            event_snapshot_key=str(data.get("event_snapshot_key", "")),
            status=str(data.get("status", "")),
            disposition=(dict(disposition) if isinstance(disposition, Mapping) else None),
            disposition_snapshot_key=str(
                data.get("disposition_snapshot_key", "") or ""
            ),
            verification=(
                dict(verification)
                if isinstance(verification, Mapping)
                else None
            ),
            identity=dict(data.get("identity", {}) or {}),
            metadata=dict(data.get("metadata", {}) or {}),
            revision=int(data.get("revision", 1) or 1),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            updated_at=float(data.get("updated_at", 0.0) or 0.0),
        )


RecordUpdater = Callable[
    [EvaluationEvidenceRecord | None],
    EvaluationEvidenceRecord,
]


class EvaluationEvidenceJournal(ABC):
    """Atomic and idempotent Evaluation Event evidence journal."""

    backend: str = "unknown"

    @abstractmethod
    def _atomic_update(
        self,
        event_id: str,
        updater: RecordUpdater,
    ) -> EvaluationEvidenceRecord:
        raise NotImplementedError

    @abstractmethod
    def get(self, event_id: str) -> EvaluationEvidenceRecord | None:
        raise NotImplementedError

    @abstractmethod
    def list_records(
        self,
        *,
        run_id: str | None = None,
        statuses: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> tuple[EvaluationEvidenceRecord, ...]:
        raise NotImplementedError

    @staticmethod
    def _expected(
        current: EvaluationEvidenceRecord,
        expected_revision: int | None,
    ) -> None:
        if expected_revision is not None and current.revision != int(expected_revision):
            raise EvaluationEvidenceRevisionConflict(
                "stale evaluation evidence revision: "
                f"expected={int(expected_revision)}, actual={current.revision}"
            )

    @staticmethod
    def _next(
        current: EvaluationEvidenceRecord,
        *,
        status: str,
        disposition: Mapping[str, Any] | None = None,
        disposition_snapshot_key: str | None = None,
        verification: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> EvaluationEvidenceRecord:
        merged_metadata = dict(current.metadata)
        merged_metadata.update(dict(metadata or {}))
        return EvaluationEvidenceRecord(
            event_id=current.event_id,
            run_id=current.run_id,
            event_snapshot_key=current.event_snapshot_key,
            status=status,
            disposition=(
                current.disposition if disposition is None else disposition
            ),
            disposition_snapshot_key=(
                current.disposition_snapshot_key
                if disposition_snapshot_key is None
                else disposition_snapshot_key
            ),
            verification=(
                current.verification
                if verification is None
                else verification
            ),
            identity=current.identity,
            metadata=merged_metadata,
            revision=current.revision + 1,
            created_at=current.created_at,
            updated_at=time.time(),
        )

    def reserve(
        self,
        *,
        event_id: str,
        run_id: str,
        event_snapshot_key: str,
        identity: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> EvaluationEvidenceRecord:
        """Reserve an index entry before the large Event snapshot is written."""

        requested = EvaluationEvidenceRecord(
            event_id=event_id,
            run_id=run_id,
            event_snapshot_key=event_snapshot_key,
            identity=dict(identity or {}),
            metadata=dict(metadata or {}),
        )

        def update(current: EvaluationEvidenceRecord | None) -> EvaluationEvidenceRecord:
            if current is None:
                return requested
            if (
                current.run_id == requested.run_id
                and current.event_snapshot_key == requested.event_snapshot_key
                and dict(current.identity) == dict(requested.identity)
            ):
                return current
            raise EvaluationEvidenceConflict(
                f"event_id {event_id!r} is already reserved with different identity"
            )

        return self._atomic_update(requested.event_id, update)

    def mark_event_durable(
        self,
        event_id: str,
        *,
        expected_revision: int | None = None,
    ) -> EvaluationEvidenceRecord:
        """Confirm that the referenced Evaluation Event snapshot is readable."""

        def update(current: EvaluationEvidenceRecord | None) -> EvaluationEvidenceRecord:
            if current is None:
                raise EvaluationEvidenceConflict(
                    f"evaluation evidence {event_id!r} was not reserved"
                )
            self._expected(current, expected_revision)
            if current.status == "preparing":
                return self._next(current, status="pending")
            if current.status in (
                {"pending", "deciding"} | EVALUATION_EVIDENCE_TERMINAL_STATUSES
            ):
                return current
            raise EvaluationEvidenceConflict(
                f"cannot mark event durable from status {current.status!r}"
            )

        return self._atomic_update(str(event_id), update)

    def prepare_disposition(
        self,
        envelope: EvaluationDispositionEnvelope,
        *,
        disposition_snapshot_key: str = "",
        expected_revision: int | None = None,
    ) -> EvaluationEvidenceRecord:
        """Persist the intended terminal edge before publishing its snapshot."""

        if not isinstance(envelope, EvaluationDispositionEnvelope):
            raise TypeError("envelope must be EvaluationDispositionEnvelope")
        payload = envelope.as_dict()

        def update(current: EvaluationEvidenceRecord | None) -> EvaluationEvidenceRecord:
            if current is None:
                raise EvaluationEvidenceConflict(
                    f"evaluation evidence {envelope.event_id!r} was not reserved"
                )
            self._expected(current, expected_revision)
            if envelope.event_snapshot_key != current.event_snapshot_key:
                raise EvaluationEvidenceConflict(
                    "disposition event_snapshot_key disagrees with the reserved Event"
                )
            if current.status == "pending":
                return self._next(
                    current,
                    status="deciding",
                    disposition=payload,
                    disposition_snapshot_key=disposition_snapshot_key,
                )
            if current.status == "deciding":
                if (
                    dict(current.disposition or {}) == payload
                    and current.disposition_snapshot_key
                    == str(disposition_snapshot_key or "")
                ):
                    return current
                raise EvaluationEvidenceConflict(
                    f"event {envelope.event_id!r} already has a different disposition intent"
                )
            if current.status == envelope.status:
                if dict(current.disposition or {}) == payload:
                    return current
            if current.terminal:
                raise EvaluationEvidenceConflict(
                    "cannot replace terminal evaluation evidence in status "
                    f"{current.status!r}"
                )
            raise EvaluationEvidenceConflict(
                f"cannot prepare disposition from status {current.status!r}"
            )

        return self._atomic_update(envelope.event_id, update)

    def settle(
        self,
        event_id: str,
        *,
        verification: EvaluationDispositionVerificationReceipt,
        expected_revision: int | None = None,
    ) -> EvaluationEvidenceRecord:
        """Make a previously prepared disposition terminal."""

        if not isinstance(
            verification,
            EvaluationDispositionVerificationReceipt,
        ):
            raise TypeError(
                "settle requires EvaluationDispositionVerificationReceipt"
            )

        def update(current: EvaluationEvidenceRecord | None) -> EvaluationEvidenceRecord:
            if current is None:
                raise EvaluationEvidenceConflict(
                    f"evaluation evidence {event_id!r} was not reserved"
                )
            self._expected(current, expected_revision)
            if current.disposition is None:
                raise EvaluationEvidenceConflict(
                    f"cannot settle evaluation evidence from status {current.status!r}"
                )
            envelope = EvaluationDispositionEnvelope.from_dict(current.disposition)
            expected_destination = (
                envelope.authority_snapshot_key
                if envelope.status == "committed"
                else current.disposition_snapshot_key
            )
            if verification.event_id != current.event_id:
                raise EvaluationEvidenceConflict(
                    "verification receipt belongs to another Evaluation Event"
                )
            if verification.event_snapshot_key != current.event_snapshot_key:
                raise EvaluationEvidenceConflict(
                    "verification receipt Event Snapshot disagrees with journal"
                )
            if verification.destination_snapshot_key != expected_destination:
                raise EvaluationEvidenceConflict(
                    "verification receipt destination disagrees with disposition"
                )
            if verification.disposition_digest != evaluation_disposition_digest(
                envelope
            ):
                raise EvaluationEvidenceConflict(
                    "verification receipt disposition digest mismatch"
                )
            if current.status in {"committed", "rejected", "failed"}:
                if current.status != envelope.status:
                    raise EvaluationEvidenceConflict(
                        "terminal evidence status disagrees with disposition"
                    )
                if current.verification is not None:
                    existing = EvaluationDispositionVerificationReceipt.from_dict(
                        current.verification
                    )
                    if existing.as_dict() != verification.as_dict():
                        raise EvaluationEvidenceConflict(
                            "terminal evidence already carries another verification"
                        )
                    return current
                return self._next(
                    current,
                    status=current.status,
                    verification=verification.as_dict(),
                )
            if current.status != "deciding":
                raise EvaluationEvidenceConflict(
                    f"cannot settle evaluation evidence from status {current.status!r}"
                )
            return self._next(
                current,
                status=envelope.status,
                verification=verification.as_dict(),
            )

        return self._atomic_update(str(event_id), update)

    def abandon(
        self,
        event_id: str,
        *,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
    ) -> EvaluationEvidenceRecord:
        """Archive an unresolved Event without manufacturing a decision."""

        reason_text = str(reason or "").strip()
        if not reason_text:
            raise ValueError("abandoned evaluation evidence requires a reason")

        def update(current: EvaluationEvidenceRecord | None) -> EvaluationEvidenceRecord:
            if current is None:
                raise EvaluationEvidenceConflict(
                    f"evaluation evidence {event_id!r} was not reserved"
                )
            self._expected(current, expected_revision)
            if current.status == "abandoned":
                return current
            if current.terminal:
                raise EvaluationEvidenceConflict(
                    f"cannot abandon terminal evidence in status {current.status!r}"
                )
            audit = {"abandon_reason": reason_text}
            audit.update(dict(metadata or {}))
            return self._next(current, status="abandoned", metadata=audit)

        return self._atomic_update(str(event_id), update)

    def list_unresolved(
        self,
        *,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[EvaluationEvidenceRecord, ...]:
        active = self.list_records(
            run_id=run_id,
            statuses=tuple(sorted(EVALUATION_EVIDENCE_ACTIVE_STATUSES)),
            limit=limit,
        )
        remaining = None if limit is None else max(0, int(limit) - len(active))
        if remaining == 0:
            return active
        legacy_terminal = tuple(
            record
            for record in self.list_records(
                run_id=run_id,
                statuses=("committed", "rejected", "failed"),
                limit=remaining,
            )
            if not record.terminal_verified
        )
        records = tuple(sorted((*active, *legacy_terminal), key=lambda item: item.created_at))
        return records if limit is None else records[: max(0, int(limit))]


class InMemoryEvaluationEvidenceJournal(EvaluationEvidenceJournal):
    backend = "memory"

    def __init__(self) -> None:
        self._records: dict[str, EvaluationEvidenceRecord] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _detach(record: EvaluationEvidenceRecord) -> EvaluationEvidenceRecord:
        return EvaluationEvidenceRecord.from_dict(record.as_dict())

    def _atomic_update(
        self,
        event_id: str,
        updater: RecordUpdater,
    ) -> EvaluationEvidenceRecord:
        with self._lock:
            updated = updater(self._records.get(str(event_id)))
            self._records[updated.event_id] = self._detach(updated)
            return self._detach(updated)

    def get(self, event_id: str) -> EvaluationEvidenceRecord | None:
        with self._lock:
            record = self._records.get(str(event_id))
            return None if record is None else self._detach(record)

    def list_records(
        self,
        *,
        run_id: str | None = None,
        statuses: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> tuple[EvaluationEvidenceRecord, ...]:
        wanted = None if statuses is None else {str(item).lower() for item in statuses}
        with self._lock:
            records = [
                self._detach(record)
                for record in self._records.values()
                if (run_id is None or record.run_id == str(run_id))
                and (wanted is None or record.status in wanted)
            ]
        records.sort(key=lambda item: (item.created_at, item.event_id))
        if limit is not None:
            records = records[: max(0, int(limit))]
        return tuple(records)


class SQLiteEvaluationEvidenceJournal(EvaluationEvidenceJournal):
    """Cross-process journal for filesystem-backed Snapshot stores."""

    backend = "sqlite"

    def __init__(
        self,
        path: str | Path,
        *,
        namespace: str = "default",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = str(namespace or "default")
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.path),
            timeout=self.timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_evidence_records (
                    namespace TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (namespace, event_id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_evaluation_evidence_run_status
                ON evaluation_evidence_records(namespace, run_id, status, created_at)
                """
            )

    @staticmethod
    def _decode(raw: str) -> EvaluationEvidenceRecord:
        return EvaluationEvidenceRecord.from_dict(json.loads(str(raw)))

    @staticmethod
    def _encode(record: EvaluationEvidenceRecord) -> str:
        return json.dumps(
            record.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _atomic_update(
        self,
        event_id: str,
        updater: RecordUpdater,
    ) -> EvaluationEvidenceRecord:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT payload_json FROM evaluation_evidence_records
                    WHERE namespace = ? AND event_id = ?
                    """,
                    (self.namespace, str(event_id)),
                ).fetchone()
                current = None if row is None else self._decode(row["payload_json"])
                updated = updater(current)
                connection.execute(
                    """
                    INSERT INTO evaluation_evidence_records(
                        namespace, event_id, run_id, status, revision,
                        created_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(namespace, event_id) DO UPDATE SET
                        run_id = excluded.run_id,
                        status = excluded.status,
                        revision = excluded.revision,
                        created_at = excluded.created_at,
                        payload_json = excluded.payload_json
                    """,
                    (
                        self.namespace,
                        updated.event_id,
                        updated.run_id,
                        updated.status,
                        updated.revision,
                        updated.created_at,
                        self._encode(updated),
                    ),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return EvaluationEvidenceRecord.from_dict(updated.as_dict())

    def get(self, event_id: str) -> EvaluationEvidenceRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM evaluation_evidence_records
                WHERE namespace = ? AND event_id = ?
                """,
                (self.namespace, str(event_id)),
            ).fetchone()
        return None if row is None else self._decode(row["payload_json"])

    def list_records(
        self,
        *,
        run_id: str | None = None,
        statuses: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> tuple[EvaluationEvidenceRecord, ...]:
        clauses = ["namespace = ?"]
        params: list[Any] = [self.namespace]
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(str(run_id))
        if statuses is not None and not tuple(statuses):
            return ()
        normalized_statuses = tuple(str(item).lower() for item in (statuses or ()))
        if normalized_statuses:
            clauses.append(
                "status IN (" + ",".join("?" for _ in normalized_statuses) + ")"
            )
            params.extend(normalized_statuses)
        query = (
            "SELECT payload_json FROM evaluation_evidence_records WHERE "
            + " AND ".join(clauses)
            + " ORDER BY created_at, event_id"
        )
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(self._decode(row["payload_json"]) for row in rows)


class RedisEvaluationEvidenceJournal(EvaluationEvidenceJournal):
    """Distributed journal using Redis WATCH/MULTI compare-and-set."""

    backend = "redis"

    def __init__(
        self,
        *,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "blackbase:evaluation-evidence",
        client: Any | None = None,
        max_watch_retries: int = 32,
    ) -> None:
        try:
            import redis  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "RedisEvaluationEvidenceJournal requires `redis` package."
            ) from exc
        self._redis = client if client is not None else redis.from_url(redis_url)
        self._watch_error = redis.WatchError
        self.key_prefix = str(key_prefix or "blackbase:evaluation-evidence").rstrip(":")
        self.max_watch_retries = max(1, int(max_watch_retries))

    def _key(self, event_id: str) -> str:
        digest = hashlib.sha256(str(event_id).encode("utf-8")).hexdigest()
        return f"{self.key_prefix}:record:{digest}"

    @staticmethod
    def _decode(raw: Any) -> EvaluationEvidenceRecord:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return EvaluationEvidenceRecord.from_dict(json.loads(str(raw)))

    @staticmethod
    def _encode(record: EvaluationEvidenceRecord) -> str:
        return json.dumps(
            record.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _atomic_update(
        self,
        event_id: str,
        updater: RecordUpdater,
    ) -> EvaluationEvidenceRecord:
        key = self._key(event_id)
        for _attempt in range(self.max_watch_retries):
            with self._redis.pipeline(transaction=True) as pipe:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    current = None if raw is None else self._decode(raw)
                    updated = updater(current)
                    pipe.multi()
                    pipe.set(key, self._encode(updated))
                    pipe.execute()
                    return EvaluationEvidenceRecord.from_dict(updated.as_dict())
                except self._watch_error:
                    continue
        raise EvaluationEvidenceRevisionConflict(
            f"evaluation evidence update for {event_id!r} exceeded WATCH retries"
        )

    def get(self, event_id: str) -> EvaluationEvidenceRecord | None:
        raw = self._redis.get(self._key(event_id))
        return None if raw is None else self._decode(raw)

    def list_records(
        self,
        *,
        run_id: str | None = None,
        statuses: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> tuple[EvaluationEvidenceRecord, ...]:
        wanted = None if statuses is None else {str(item).lower() for item in statuses}
        records: list[EvaluationEvidenceRecord] = []
        pattern = f"{self.key_prefix}:record:*"
        for key in self._redis.scan_iter(match=pattern, count=200):
            raw = self._redis.get(key)
            if raw is None:
                continue
            record = self._decode(raw)
            if run_id is not None and record.run_id != str(run_id):
                continue
            if wanted is not None and record.status not in wanted:
                continue
            records.append(record)
        records.sort(key=lambda item: (item.created_at, item.event_id))
        if limit is not None:
            records = records[: max(0, int(limit))]
        return tuple(records)


def create_evaluation_evidence_journal(
    *,
    backend: str = "memory",
    redis_url: str = "redis://localhost:6379/0",
    key_prefix: str = "blackbase:evaluation-evidence",
    base_dir: str | Path = "runs/snapshots",
    sqlite_path: str | Path | None = None,
) -> EvaluationEvidenceJournal:
    """Build the journal paired with a SnapshotStore backend."""

    backend_name = str(backend or "memory").strip().lower()
    if backend_name in {"memory", "inmemory", "local"}:
        return InMemoryEvaluationEvidenceJournal()
    if backend_name in {"file", "filesystem", "disk", "sqlite"}:
        path = (
            Path(sqlite_path)
            if sqlite_path is not None
            else Path(base_dir) / "evaluation_evidence.sqlite3"
        )
        return SQLiteEvaluationEvidenceJournal(
            path,
            namespace=str(key_prefix or "blackbase:evaluation-evidence"),
        )
    if backend_name == "redis":
        return RedisEvaluationEvidenceJournal(
            redis_url=redis_url,
            key_prefix=key_prefix,
        )
    raise ValueError(f"unsupported evaluation evidence journal backend: {backend}")


__all__ = [
    "EVALUATION_EVIDENCE_ACTIVE_STATUSES",
    "EVALUATION_EVIDENCE_RECORD_SCHEMA_V1",
    "EVALUATION_EVIDENCE_TERMINAL_STATUSES",
    "EvaluationEvidenceConflict",
    "EvaluationEvidenceJournal",
    "EvaluationEvidenceRecord",
    "EvaluationEvidenceRevisionConflict",
    "InMemoryEvaluationEvidenceJournal",
    "RedisEvaluationEvidenceJournal",
    "SQLiteEvaluationEvidenceJournal",
    "create_evaluation_evidence_journal",
]
