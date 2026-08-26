from __future__ import annotations

from blackbase.resources import (
    BudgetSettlementRecord,
    SQLiteBudgetSettlementJournal,
)


def _record() -> BudgetSettlementRecord:
    return BudgetSettlementRecord(
        settlement_id="settlement-1",
        project_run_id="run-1",
        parent_case_run_id="parent-1",
        child_case_run_id="child-1",
        budget="evaluations",
        parent_reservation_id="reservation-1",
        parent_authority_ref={
            "lease": {"lease_id": "lease-1", "fencing_token": 2},
            "metadata": {
                "budget_authority": {
                    "backend": "sqlite",
                    "namespace": "project",
                    "scope": "run-1",
                    "path": "budgets.sqlite",
                }
            },
        },
        child_handle={"handle_id": "handle-1", "metadata": {"nested": True}},
        requested_amount=3,
    )


def test_budget_settlement_retry_survives_journal_reopen(tmp_path) -> None:
    path = tmp_path / "settlements.sqlite"
    journal = SQLiteBudgetSettlementJournal(path, namespace="project")
    intent = journal.prepare(_record())
    assert intent.status == "reserve_intent"
    prepared = journal.mark_reserved(intent.settlement_id)
    assert prepared.status == "prepared"

    retry = journal.mark_retry(
        prepared.settlement_id,
        ConnectionError("authority temporarily unavailable"),
        usage={"charged_to_parent": 2},
    )
    assert retry.status == "retry_required"
    assert retry.attempts == 1

    reopened = SQLiteBudgetSettlementJournal(path, namespace="project")
    pending = reopened.pending(project_run_id="run-1")
    assert len(pending) == 1
    assert pending[0].last_error["error_type"] == "ConnectionError"

    settled = reopened.mark_settled(
        prepared.settlement_id,
        {"charged_to_parent": 2, "returned_to_parent": 1},
    )
    assert settled.status == "settled"
    assert reopened.pending(project_run_id="run-1") == ()


def test_prepared_settlement_is_not_retryable_until_child_finishes(tmp_path) -> None:
    journal = SQLiteBudgetSettlementJournal(
        tmp_path / "settlements.sqlite",
        namespace="project",
    )
    intent = journal.prepare(_record())

    assert journal.pending() == (intent,)
    assert journal.retryable() == ()

    prepared = journal.mark_reserved(intent.settlement_id)

    assert journal.pending() == (prepared,)
    assert journal.retryable() == ()

    ready = journal.mark_ready(
        prepared.settlement_id,
        {"charged_to_parent": 2, "returned_to_parent": 1},
    )

    assert ready.status == "settlement_ready"
    assert journal.retryable() == (ready,)


def test_reclaimed_settlement_is_terminal_and_not_retried(tmp_path) -> None:
    journal = SQLiteBudgetSettlementJournal(
        tmp_path / "settlements.sqlite",
        namespace="project",
    )
    intent = journal.prepare(_record())
    prepared = journal.mark_reserved(intent.settlement_id)

    reclaimed = journal.mark_reclaimed(
        prepared.settlement_id,
        RuntimeError("parent lease was fenced and reclaimed"),
    )

    assert reclaimed.status == "reclaimed"
    assert reclaimed.last_error["error_type"] == "RuntimeError"
    assert journal.pending() == ()
    assert journal.mark_settled(prepared.settlement_id, {}).status == "reclaimed"
