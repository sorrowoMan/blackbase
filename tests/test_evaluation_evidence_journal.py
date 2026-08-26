from __future__ import annotations

import time

import pytest

from blackbase.evaluation import (
    EvaluationDispositionEnvelope,
    EvaluationDispositionVerificationReceipt,
    EvaluationEvidenceConflict,
    EvaluationEvidenceRevisionConflict,
    InMemoryEvaluationEvidenceJournal,
    SQLiteEvaluationEvidenceJournal,
    evaluation_disposition_digest,
)


def _disposition(event_id: str, status: str = "committed") -> EvaluationDispositionEnvelope:
    return EvaluationDispositionEnvelope(
        event_id=event_id,
        status=status,
        disposition_codec="test.disposition/v1",
        disposition_payload={"accepted_indices": [0]},
        event_snapshot_key=f"events/{event_id}",
        authority_snapshot_key=(
            f"authority/{event_id}" if status == "committed" else "authority/previous"
        ),
        identity={"run_id": "run-1", "attempt": 2},
    )


def _verification(
    envelope: EvaluationDispositionEnvelope,
) -> EvaluationDispositionVerificationReceipt:
    destination = (
        envelope.authority_snapshot_key
        if envelope.status == "committed"
        else f"dispositions/{envelope.event_id}"
    )
    return EvaluationDispositionVerificationReceipt(
        event_id=envelope.event_id,
        event_snapshot_key=envelope.event_snapshot_key,
        event_snapshot_revision=1,
        event_snapshot_digest="sha256:" + "a" * 64,
        event_snapshot_schema="test.evaluation_event/v1",
        destination_snapshot_key=destination,
        destination_snapshot_revision=1,
        destination_snapshot_digest="sha256:" + "b" * 64,
        destination_snapshot_schema="test.authority/v1",
        disposition_digest=evaluation_disposition_digest(envelope),
        verifier="test.snapshot_store",
        verified_at=time.time(),
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda tmp_path: InMemoryEvaluationEvidenceJournal(),
        lambda tmp_path: SQLiteEvaluationEvidenceJournal(
            tmp_path / "evaluation-evidence.sqlite3",
            namespace="test",
        ),
    ],
)
def test_evaluation_evidence_state_machine_is_idempotent(factory, tmp_path) -> None:
    journal = factory(tmp_path)
    reserved = journal.reserve(
        event_id="event-1",
        run_id="run-1",
        event_snapshot_key="events/event-1",
        identity={"run_id": "run-1", "attempt": 2},
    )
    assert reserved.status == "preparing"
    assert journal.reserve(
        event_id="event-1",
        run_id="run-1",
        event_snapshot_key="events/event-1",
        identity={"run_id": "run-1", "attempt": 2},
    ).revision == reserved.revision

    pending = journal.mark_event_durable(
        "event-1",
        expected_revision=reserved.revision,
    )
    assert pending.status == "pending"
    disposition = _disposition("event-1")
    deciding = journal.prepare_disposition(
        disposition,
        expected_revision=pending.revision,
    )
    assert deciding.status == "deciding"
    verification = _verification(disposition)
    terminal = journal.settle(
        "event-1",
        verification=verification,
        expected_revision=deciding.revision,
    )
    assert terminal.status == "committed"
    assert terminal.terminal_verified
    assert journal.settle(
        "event-1",
        verification=verification,
    ).revision == terminal.revision
    assert journal.list_unresolved(run_id="run-1") == ()
    assert journal.list_records(run_id="run-1") == (terminal,)


def test_evaluation_evidence_cannot_settle_without_snapshot_verification() -> None:
    journal = InMemoryEvaluationEvidenceJournal()
    journal.reserve(
        event_id="event-1",
        run_id="run-1",
        event_snapshot_key="events/event-1",
    )
    journal.mark_event_durable("event-1")
    journal.prepare_disposition(_disposition("event-1"))

    with pytest.raises(TypeError, match="verification"):
        journal.settle("event-1")  # type: ignore[call-arg]


def test_evaluation_evidence_rejects_stale_or_conflicting_transitions() -> None:
    journal = InMemoryEvaluationEvidenceJournal()
    reserved = journal.reserve(
        event_id="event-1",
        run_id="run-1",
        event_snapshot_key="events/event-1",
    )
    pending = journal.mark_event_durable("event-1")
    with pytest.raises(EvaluationEvidenceRevisionConflict, match="stale"):
        journal.prepare_disposition(
            _disposition("event-1"),
            expected_revision=reserved.revision,
        )
    journal.prepare_disposition(_disposition("event-1"))
    with pytest.raises(EvaluationEvidenceConflict, match="different"):
        journal.prepare_disposition(_disposition("event-1", "rejected"))
    assert pending.revision == reserved.revision + 1


def test_evaluation_evidence_abandon_preserves_unexecuted_intent() -> None:
    journal = InMemoryEvaluationEvidenceJournal()
    journal.reserve(
        event_id="event-1",
        run_id="run-1",
        event_snapshot_key="events/event-1",
    )
    journal.mark_event_durable("event-1")
    deciding = journal.prepare_disposition(_disposition("event-1"))
    abandoned = journal.abandon(
        "event-1",
        reason="authority_snapshot_missing",
        expected_revision=deciding.revision,
    )
    assert abandoned.status == "abandoned"
    assert abandoned.disposition is not None
    assert abandoned.metadata["abandon_reason"] == "authority_snapshot_missing"
    with pytest.raises(EvaluationEvidenceConflict, match="terminal"):
        journal.prepare_disposition(_disposition("event-1"))


@pytest.mark.parametrize("status", ["committed", "rejected", "failed"])
def test_every_disposition_requires_event_snapshot_key(status: str) -> None:
    with pytest.raises(ValueError, match="event_snapshot_key"):
        EvaluationDispositionEnvelope(
            event_id="event-1",
            status=status,
            disposition_codec="test.disposition/v1",
            disposition_payload={},
            authority_snapshot_key="authority/event-1",
        )


def test_committed_disposition_requires_authority_snapshot_key() -> None:
    with pytest.raises(ValueError, match="authority_snapshot_key"):
        EvaluationDispositionEnvelope(
            event_id="event-1",
            status="committed",
            disposition_codec="test.disposition/v1",
            disposition_payload={},
            event_snapshot_key="events/event-1",
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda tmp_path: InMemoryEvaluationEvidenceJournal(),
        lambda tmp_path: SQLiteEvaluationEvidenceJournal(
            tmp_path / "empty-statuses.sqlite3",
            namespace="test",
        ),
    ],
)
def test_empty_status_filter_returns_no_records(factory, tmp_path) -> None:
    journal = factory(tmp_path)
    journal.reserve(
        event_id="event-1",
        run_id="run-1",
        event_snapshot_key="events/event-1",
    )

    assert journal.list_records(statuses=()) == ()


def test_sqlite_evaluation_evidence_survives_reopen(tmp_path) -> None:
    path = tmp_path / "journal.sqlite3"
    first = SQLiteEvaluationEvidenceJournal(path, namespace="run-space")
    first.reserve(
        event_id="event-1",
        run_id="run-1",
        event_snapshot_key="events/event-1",
    )
    first.mark_event_durable("event-1")

    second = SQLiteEvaluationEvidenceJournal(path, namespace="run-space")
    restored = second.get("event-1")
    assert restored is not None
    assert restored.status == "pending"
    assert second.list_unresolved(run_id="run-1") == (restored,)
