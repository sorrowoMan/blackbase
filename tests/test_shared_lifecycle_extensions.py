from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from blackbase import CandidateBatch, normalize_row_selector
from blackbase.contracts import BatchDisposition
from blackbase.evaluation import (
    EvaluationDispositionEnvelope,
    EvaluationEventEnvelope,
)
from blackbase.project import CaseExecutor, CaseRunRequest, ExecutionControl
from blackbase.plugin import (
    PluginBase,
    PluginLifecycleCleanupError,
    PluginLifecycleDispatchError,
)
from blackbase.resources import (
    CancellationHeartbeat,
    CancellationRef,
    CancellationToken,
    PoolScheduler,
    TerminationPolicy,
)


def test_row_selector_treats_tuple_as_rows_and_empty_as_empty() -> None:
    assert normalize_row_selector((0, 2), row_count=3).tolist() == [0, 2]
    assert normalize_row_selector((), row_count=3).tolist() == []
    batch = CandidateBatch.from_candidates([[1.0], [2.0], [3.0]])
    assert batch.subset((0, 2)).numeric_matrix[:, 0].tolist() == [1.0, 3.0]
    assert batch.subset(()).numeric_matrix.shape == (0, 1)


def test_row_selector_rejects_multidimensional_and_out_of_range() -> None:
    with pytest.raises(ValueError):
        normalize_row_selector(np.array([[0, 1]]), row_count=3)
    with pytest.raises(IndexError):
        normalize_row_selector([3], row_count=3)


def test_batch_disposition_composes_local_and_global_selection() -> None:
    local = BatchDisposition(
        proposed_count=5,
        accepted_indices=(0, 2, 4),
        reason="local",
    )
    global_result = BatchDisposition(
        proposed_count=3,
        accepted_indices=(1, 2),
        reason="global",
    )
    composed = local.compose(global_result)
    assert composed.proposed_count == 5
    assert composed.accepted_indices == (2, 4)


def test_evaluation_event_envelope_is_bounded_to_wire_values() -> None:
    envelope = EvaluationEventEnvelope(
        event_id="event-1",
        candidate_codec="candidate/v1",
        candidate_payload={"rows": [[1.0]]},
        feedback_codec="feedback/v1",
        feedback_payload={"loss": [2.0]},
        identity={"run_id": "run-1", "attempt": 2},
        evaluation_count=7,
    )
    assert EvaluationEventEnvelope.from_dict(envelope.as_dict()) == envelope
    with pytest.raises(TypeError):
        EvaluationEventEnvelope(
            event_id="event-2",
            candidate_codec="candidate/v1",
            candidate_payload={"rows": np.array([[1.0]])},
            feedback_codec="feedback/v1",
            feedback_payload={},
        )


def test_evaluation_disposition_envelope_links_event_to_authority() -> None:
    envelope = EvaluationDispositionEnvelope(
        event_id="event-1",
        status="committed",
        disposition_codec="blackbase.batch_disposition/v1",
        disposition_payload={
            "proposed_count": 2,
            "accepted_indices": [0],
            "reason": "budget",
        },
        event_snapshot_key="evaluation/event-1",
        authority_snapshot_key="population/generation-3",
        identity={"run_id": "run-1", "attempt": 4},
    )

    assert EvaluationDispositionEnvelope.from_dict(envelope.as_dict()) == envelope
    with pytest.raises(ValueError, match="committed, rejected, or failed"):
        EvaluationDispositionEnvelope(
            event_id="event-2",
            status="pending",
            disposition_codec="disposition/v1",
            disposition_payload={},
        )


def test_executor_views_reuse_the_persistent_pool_worker() -> None:
    pool = PoolScheduler(total_threads=1)
    try:
        with pool.as_executor(max_workers=1) as executor:
            first = executor.submit(
                lambda: (threading.get_ident(), threading.current_thread().name)
            ).result()
        with pool.as_executor(max_workers=1) as executor:
            second = executor.submit(
                lambda: (threading.get_ident(), threading.current_thread().name)
            ).result()
        assert first[0] == second[0]
        assert first[1].startswith("blackbase-pool")
        assert second[1].startswith("blackbase-pool")
    finally:
        pool.shutdown()


def test_executor_future_completion_publishes_audit_first() -> None:
    pool = PoolScheduler(total_threads=1)
    try:
        with pool.as_executor(max_workers=1) as executor:
            future = executor.submit(lambda: 7)
            assert future.result(timeout=1.0) == 7
            assert pool.report()["tasks_completed"] == 1
    finally:
        pool.shutdown()


def test_executor_future_failure_publishes_audit_first() -> None:
    pool = PoolScheduler(total_threads=1)
    try:
        with pool.as_executor(max_workers=1) as executor:
            future = executor.submit(
                lambda: (_ for _ in ()).throw(ValueError("boom"))
            )
            with pytest.raises(ValueError, match="boom"):
                future.result(timeout=1.0)
            assert pool.report()["tasks_failed"] == 1
    finally:
        pool.shutdown()


def test_durable_cancellation_lineage_is_constant_size_and_cascades(tmp_path) -> None:
    authority = str(tmp_path / "controls.sqlite")
    root = ExecutionControl(
        cancellation=CancellationRef(backend="sqlite", path=authority)
    )
    CancellationToken(root.cancellation)
    current = root
    encoded_sizes: list[int] = []
    for _ in range(12):
        current = current.derive_child(ExecutionControl.with_timeout(None))
        CancellationToken(current.cancellation)
        assert current.ancestor_cancellations == ()
        encoded_sizes.append(len(str(current.as_dict())))

    assert max(encoded_sizes) - min(encoded_sizes) < 256
    CancellationToken(root.cancellation).cancel("root stopped")
    assert CancellationToken(current.cancellation).cancelled is True
    assert current.cancellation.root_control_id == root.cancellation.control_id
    assert current.cancellation.lineage_depth == 12


def test_child_control_rejects_a_forged_intermediate_lineage(tmp_path) -> None:
    authority = str(tmp_path / "controls.sqlite")
    root = ExecutionControl(
        cancellation=CancellationRef(backend="sqlite", path=authority)
    )
    forged = CancellationRef(
        backend="sqlite",
        path=authority,
        parent_control_id="unrelated-parent",
        root_control_id=root.cancellation.control_id,
        lineage_depth=1,
    )

    with pytest.raises(ValueError, match="does not extend the parent lineage"):
        root.derive_child(
            ExecutionControl.with_timeout(None),
            intermediate_cancellations=(forged,),
        )


def test_durable_cancellation_control_heartbeat_keeps_active_ttl_alive(
    tmp_path,
) -> None:
    ref = CancellationRef(
        backend="sqlite",
        path=str(tmp_path / "heartbeat.sqlite"),
        active_ttl_seconds=0.2,
        heartbeat_seconds=0.05,
    )
    token = CancellationToken(ref)
    heartbeat = CancellationHeartbeat(token)
    try:
        time.sleep(0.35)
        heartbeat.assert_current()
        assert token.touch() is True
    finally:
        heartbeat.close()
        token.retire()
    assert token.touch() is False


def test_durable_cancellation_control_expires_after_owner_crash_simulation(
    tmp_path,
) -> None:
    ref = CancellationRef(
        backend="sqlite",
        path=str(tmp_path / "orphan.sqlite"),
        active_ttl_seconds=0.15,
        heartbeat_seconds=0.05,
    )
    token = CancellationToken(ref)
    assert token.touch() is True

    time.sleep(0.25)

    assert token.touch() is False


def test_retained_cancellation_control_cannot_be_reactivated_by_late_read(
    tmp_path,
) -> None:
    ref = CancellationRef(
        backend="sqlite",
        path=str(tmp_path / "retained.sqlite"),
        active_ttl_seconds=1.0,
        heartbeat_seconds=0.2,
        retention_seconds=0.3,
    )
    token = CancellationToken(ref)
    token.retire()

    assert token.touch() is False
    assert token.cancel("too late") is False
    time.sleep(0.4)
    assert token.touch() is False


def test_isolated_execution_rejects_process_local_cancellation(tmp_path) -> None:
    request = CaseRunRequest(
        project_name="project",
        stage_name="stage",
        case_name="missing",
        control=ExecutionControl(
            termination=TerminationPolicy(mode="cooperative_then_terminate")
        ),
    )
    result = CaseExecutor(tmp_path).execute(request)
    assert result.status == "failed"
    assert result.failure is not None
    assert "process-local" in result.failure.message


class _ReceiptPlugin(PluginBase):
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        fail_start: bool = False,
        fail_end: bool = False,
    ) -> None:
        super().__init__(name=name)
        self.events = events
        self.fail_start = fail_start
        self.fail_end = fail_end

    def on_generation_start(self, generation: int) -> None:
        self.events.append(f"{self.name}:start:{generation}")
        if self.fail_start:
            raise RuntimeError(f"{self.name} start failed")

    def on_generation_end(self, generation: int) -> None:
        self.events.append(f"{self.name}:end:{generation}")
        if self.fail_end:
            raise RuntimeError(f"{self.name} end failed")


class _RunReceiptPlugin(PluginBase):
    def __init__(self, name: str, events: list[str], *, fail_init: bool = False) -> None:
        super().__init__(name=name)
        self.events = events
        self.fail_init = fail_init

    def on_solver_init(self, solver) -> None:
        del solver
        self.events.append(f"{self.name}:init")
        if self.fail_init:
            raise RuntimeError(f"{self.name} init failed")

    def on_solver_finish(self, result) -> None:
        del result
        self.events.append(f"{self.name}:finish")

    def on_solver_finalization_prepare(self, result) -> None:
        del result
        self.events.append(f"{self.name}:prepare-finalization")

    def on_solver_finalized(self, result) -> None:
        del result
        self.events.append(f"{self.name}:finalized")


def test_lifecycle_receipt_closes_only_successfully_started_plugins() -> None:
    from blackbase.plugin import PluginManager

    events: list[str] = []
    manager = PluginManager(strict=True)
    manager.register(_ReceiptPlugin("first", events))
    manager.register(_ReceiptPlugin("broken", events, fail_start=True))
    manager.register(_ReceiptPlugin("never-started", events))

    with pytest.raises(PluginLifecycleDispatchError) as captured:
        manager.begin_lifecycle("on_generation_start", 4)

    manager.finish_lifecycle(
        captured.value.receipt,
        "on_generation_end",
        4,
    )
    assert events == [
        "first:start:4",
        "broken:start:4",
        "first:end:4",
    ]


def test_lifecycle_end_attempts_every_receipt_participant() -> None:
    from blackbase.plugin import PluginManager

    events: list[str] = []
    manager = PluginManager(strict=True)
    manager.register(_ReceiptPlugin("first", events, fail_end=True))
    manager.register(_ReceiptPlugin("second", events))
    receipt = manager.begin_lifecycle("on_generation_start", 2)

    with pytest.raises(PluginLifecycleCleanupError):
        manager.finish_lifecycle(receipt, "on_generation_end", 2)

    assert events[-2:] == ["first:end:2", "second:end:2"]


def test_run_init_failure_carries_only_completed_participants() -> None:
    from types import SimpleNamespace

    from blackbase.plugin import PluginManager

    events: list[str] = []
    manager = PluginManager(strict=True)
    manager.register(_RunReceiptPlugin("first", events))
    manager.register(_RunReceiptPlugin("broken", events, fail_init=True))
    manager.register(_RunReceiptPlugin("never-started", events))

    with pytest.raises(PluginLifecycleDispatchError) as captured:
        manager.on_solver_init(SimpleNamespace(plugin_strict=True))

    assert captured.value.receipt.participant_names == ("first",)
    manager.finish_lifecycle(
        captured.value.receipt,
        "on_solver_finish",
        {"status": "failed"},
    )
    assert events == [
        "first:init",
        "broken:init",
        "first:finish",
    ]


def test_finalization_prepare_and_finalized_are_distinct_receipt_phases() -> None:
    from types import SimpleNamespace

    from blackbase.plugin import PluginManager

    events: list[str] = []
    manager = PluginManager(strict=True)
    manager.register(_RunReceiptPlugin("first", events))
    receipt = manager.on_solver_init(SimpleNamespace(plugin_strict=True))

    manager.finish_lifecycle(
        receipt,
        "on_solver_finalization_prepare",
        {"ready": False},
    )
    manager.finish_lifecycle(
        receipt,
        "on_solver_finalized",
        {"ready": True},
    )

    assert events == [
        "first:init",
        "first:prepare-finalization",
        "first:finalized",
    ]
