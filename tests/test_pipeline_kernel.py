from __future__ import annotations

from concurrent.futures import CancelledError
import threading
import time

import numpy as np
import pytest

from blackbase.kernel import (
    OrchestrationPolicy,
    PipelineLateWriteRejected,
    PipelineOrchestrator,
    PipelineParallelError,
    build_pipeline_kernel,
)
from blackbase.context import ContextStore, InMemorySnapshotStore
from blackbase.resources import PoolScheduler


class _Add:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def mutate(self, x, context=None):
        del context
        return np.asarray(x, dtype=float) + self.value


class _Scale:
    def __init__(self, factor: float) -> None:
        self.factor = float(factor)

    def mutate(self, x, context=None):
        del context
        return np.asarray(x, dtype=float) * self.factor


def _kernel(merge):
    return build_pipeline_kernel(
        {
            "slots": (
                {
                    "slot": "mutate",
                    "mode": "parallel",
                    "merge": merge,
                    "operators": ("add_two", "times_two"),
                },
            )
        },
        operator_registry={"add_two": _Add(2.0), "times_two": _Scale(2.0)},
    )


def test_parallel_mean_merge_is_executed() -> None:
    out = _kernel("mean").run_slot("mutate", np.array([2.0, 4.0]), {})
    assert np.allclose(out, [4.0, 7.0])


def test_parallel_no_merge_preserves_tuple_contract() -> None:
    out = _kernel(None).run_slot("mutate", np.array([2.0, 4.0]), {})
    assert isinstance(out, tuple)
    assert len(out) == 2


class _MutateInPlace:
    def __init__(self, amount: float) -> None:
        self.amount = float(amount)

    def mutate(self, x, context=None):
        x[0] += self.amount
        context["nested"]["items"].append(self.amount)
        return x


def test_parallel_branches_receive_isolated_value_and_context() -> None:
    kernel = build_pipeline_kernel(
        {
            "slots": ({
                "slot": "mutate",
                "mode": "parallel",
                "operators": ("one", "two"),
            },)
        },
        operator_registry={"one": _MutateInPlace(1.0), "two": _MutateInPlace(2.0)},
    )
    value = np.array([10.0])
    context = {"nested": {"items": []}}

    outputs = kernel.run_slot("mutate", value, context)

    assert np.allclose(value, [10.0])
    assert context["nested"] == {"items": []}
    assert context["pipeline.parallel_report"]["completed_count"] == 2
    assert np.allclose(outputs[0], [11.0])
    assert np.allclose(outputs[1], [12.0])


class _ConcurrencyProbe:
    def __init__(self, state) -> None:
        self.state = state

    def mutate(self, x, context=None):
        del context
        with self.state["lock"]:
            self.state["active"] += 1
            self.state["peak"] = max(self.state["peak"], self.state["active"])
        time.sleep(0.02)
        with self.state["lock"]:
            self.state["active"] -= 1
        return x


def test_parallel_respects_resource_context_and_injected_pool() -> None:
    state = {"lock": threading.Lock(), "active": 0, "peak": 0}
    pool = PoolScheduler(total_threads=4)
    kernel = build_pipeline_kernel(
        {
            "slots": ({
                "slot": "mutate",
                "mode": "parallel",
                "operators": ("a", "b", "c"),
            },)
        },
        operator_registry={key: _ConcurrencyProbe(state) for key in ("a", "b", "c")},
        pool_scheduler=pool,
    )

    outputs = kernel.run_slot(
        "mutate",
        np.array([1.0]),
        {"resource_context": {"threads": 1, "namespace": "project.case"}},
    )

    assert len(outputs) == 3
    assert state["peak"] == 1
    assert pool.available() == 4


def test_shared_orchestrator_owns_switch_and_dynamic_modes() -> None:
    orchestrator = PipelineOrchestrator(strict=True)
    switch = OrchestrationPolicy(
        mode="switch",
        operators=(_Add(1.0), _Add(3.0)),
        index_key="lane",
        strict=True,
    )
    dynamic = OrchestrationPolicy(
        mode="dynamic",
        stages=((0, _Add(1.0)), (5, _Add(5.0))),
        strict=True,
    )

    switched = orchestrator.run_policy(
        switch,
        np.array([0.0]),
        {"lane": 99},
        method="mutate",
        fallback=np.array([0.0]),
    )
    early = orchestrator.run_policy(
        dynamic,
        np.array([0.0]),
        {"generation": 2},
        method="mutate",
        fallback=np.array([0.0]),
    )
    late = orchestrator.run_policy(
        dynamic,
        np.array([0.0]),
        {"generation": 7},
        method="mutate",
        fallback=np.array([0.0]),
    )

    assert np.allclose(switched, [3.0])
    assert np.allclose(early, [1.0])
    assert np.allclose(late, [5.0])


def test_dynamic_stages_are_buildable_from_the_shared_slot_spec() -> None:
    kernel = build_pipeline_kernel(
        {
            "slots": ({
                "slot": "mutate",
                "mode": "dynamic",
                "stages": {0: "early", 5: "late"},
            },)
        },
        operator_registry={"early": _Add(1.0), "late": _Add(5.0)},
    )

    assert np.allclose(
        kernel.run_slot("mutate", np.array([0.0]), {"generation": 2}),
        [1.0],
    )
    assert np.allclose(
        kernel.run_slot("mutate", np.array([0.0]), {"generation": 7}),
        [5.0],
    )


def test_non_strict_parallel_skips_only_the_failed_branch() -> None:
    class _Fail:
        def mutate(self, x, context=None):
            del x, context
            raise RuntimeError("branch failed")

    kernel = build_pipeline_kernel(
        {
            "slots": ({
                "slot": "mutate",
                "mode": "parallel",
                "operators": ("fail", "ok"),
                "strict": False,
            },)
        },
        operator_registry={"fail": _Fail(), "ok": _Add(2.0)},
        strict=True,
    )

    context = {}
    outputs = kernel.run_slot("mutate", np.array([1.0]), context)

    assert len(outputs) == 1
    assert np.allclose(outputs[0], [3.0])
    assert context["pipeline.parallel_report"]["failure_count"] == 1
    assert context["pipeline.parallel_report"]["failures"][0]["index"] == 0


def test_strict_parallel_aggregates_all_branch_failures_when_requested() -> None:
    class _FirstFail:
        def mutate(self, x, context=None):
            del x, context
            raise ValueError("first")

    class _SecondFail:
        def mutate(self, x, context=None):
            del x, context
            raise RuntimeError("second")

    kernel = build_pipeline_kernel(
        {
            "slots": ({
                "slot": "mutate",
                "mode": "parallel",
                "operators": ("first", "second"),
                "strict": True,
                "cancel_on_error": False,
            },)
        },
        operator_registry={"first": _FirstFail(), "second": _SecondFail()},
    )
    context = {}

    with pytest.raises(PipelineParallelError) as caught:
        kernel.run_slot("mutate", np.array([1.0]), context)

    failures = caught.value.failures
    assert [(item.index, item.error_type) for item in failures] == [
        (0, "ValueError"),
        (1, "RuntimeError"),
    ]
    assert context["pipeline.parallel_report"]["failure_count"] == 2


def test_parallel_timeout_returns_with_partial_results_and_audit() -> None:
    release = threading.Event()

    class _Blocking:
        def mutate(self, x, context=None):
            del context
            release.wait(timeout=1.0)
            return x

    kernel = build_pipeline_kernel(
        {
            "slots": ({
                "slot": "mutate",
                "mode": "parallel",
                "operators": ("blocking", "quick"),
                "strict": True,
                "timeout_seconds": 0.03,
            },)
        },
        operator_registry={"blocking": _Blocking(), "quick": _Add(2.0)},
    )
    context = {"resource.threads": 2}
    started = time.monotonic()

    try:
        with pytest.raises(PipelineParallelError) as caught:
            kernel.run_slot("mutate", np.array([1.0]), context)
    finally:
        release.set()

    assert time.monotonic() - started < 0.3
    assert len(caught.value.partial_results) == 1
    assert np.allclose(caught.value.partial_results[0], [3.0])
    assert any(item.timed_out for item in caught.value.failures)
    assert any(item.still_running for item in caught.value.failures)
    assert context["pipeline.parallel_report"]["timed_out"] is True
    assert context["pipeline.parallel_report"]["still_running"] is True
    assert context["pipeline.parallel_report"]["still_running_count"] == 1
    assert context["pipeline.parallel_report"]["cancellation_requested"] is True


def test_pool_scheduler_can_cancel_a_task_waiting_for_a_permit() -> None:
    pool = PoolScheduler(total_threads=1)
    entered = threading.Event()
    release = threading.Event()

    def blocking() -> str:
        entered.set()
        release.wait(timeout=1.0)
        return "done"

    first = pool.submit(blocking)
    assert entered.wait(timeout=1.0)
    second = pool.submit(lambda: "should-not-run")

    assert second.cancel() is True
    with pytest.raises(CancelledError):
        second.result(timeout=0.2)

    release.set()
    assert first.result(timeout=1.0) == "done"


def test_operator_body_type_error_is_not_retried_with_another_signature() -> None:
    calls = []

    class _SideEffectThenTypeError:
        def mutate(self, value, context=None):
            calls.append((value, context))
            raise TypeError("operator body failed")

    kernel = build_pipeline_kernel(
        {
            "slots": ({
                "slot": "mutate",
                "operators": ("side_effect",),
                "strict": True,
            },)
        },
        operator_registry={"side_effect": _SideEffectThenTypeError()},
    )

    with pytest.raises(TypeError, match="operator body failed"):
        kernel.run_slot("mutate", np.array([1.0]), {"trace": True})

    assert len(calls) == 1


def test_signature_binding_still_supports_zero_and_one_argument_operators() -> None:
    orchestrator = PipelineOrchestrator(strict=True)

    assert orchestrator.call_operator(
        lambda value: value + 1,
        2,
        {},
        "transform",
    ) == 3
    assert orchestrator.call_operator(
        lambda: 7,
        2,
        {},
        "transform",
    ) == 7


def test_parallel_timeout_signals_cooperative_cancellation_to_running_branch() -> None:
    observed = threading.Event()

    class _Cooperative:
        def mutate(self, x, context=None):
            cancellation = context["pipeline.cancel_event"]
            while not cancellation.is_set():
                cancellation.wait(timeout=0.002)
            observed.set()
            return x

    kernel = build_pipeline_kernel(
        {
            "slots": ({
                "slot": "mutate",
                "mode": "parallel",
                "operators": ("cooperative",),
                "strict": True,
                "timeout_seconds": 0.02,
            },)
        },
        operator_registry={"cooperative": _Cooperative()},
    )
    context = {"resource_context": {"threads": 1, "namespace": "project.case"}}

    with pytest.raises(PipelineParallelError):
        kernel.run_slot("mutate", np.array([1.0]), context)

    assert observed.wait(timeout=0.5)
    report = context["pipeline.parallel_report"]
    assert report["cancellation_requested"] is True
    assert report["run_token"]
    assert report["run_namespace"] == "project.case"


def test_cancelled_parallel_branch_cannot_write_context_or_snapshot_late() -> None:
    rejected_context = threading.Event()
    rejected_snapshot = threading.Event()
    rejected_owner_context = threading.Event()
    rejected_owner_snapshot = threading.Event()
    context_store = ContextStore()
    snapshot_store = InMemorySnapshotStore()
    owner = type(
        "_RuntimeOwner",
        (),
        {"context_store": context_store, "snapshot_store": snapshot_store},
    )()

    class _LateWriter:
        def mutate(self, x, context=None):
            cancellation = context["pipeline.cancel_event"]
            while not cancellation.is_set():
                cancellation.wait(timeout=0.002)
            try:
                context["context_store"].set("late", "value")
            except PipelineLateWriteRejected:
                rejected_context.set()
            try:
                context["snapshot_store"].write({"late": "value"}, key="late")
            except PipelineLateWriteRejected:
                rejected_snapshot.set()
            try:
                context["solver"].context_store.set("late_owner", "value")
            except PipelineLateWriteRejected:
                rejected_owner_context.set()
            try:
                context["solver"].snapshot_store.write(
                    {"late": "owner"}, key="late_owner"
                )
            except PipelineLateWriteRejected:
                rejected_owner_snapshot.set()
            return x

    kernel = build_pipeline_kernel(
        {
            "slots": ({
                "slot": "mutate",
                "mode": "parallel",
                "operators": ("late",),
                "strict": True,
                "timeout_seconds": 0.02,
            },)
        },
        operator_registry={"late": _LateWriter()},
    )

    with pytest.raises(PipelineParallelError):
        kernel.run_slot(
            "mutate",
            np.array([1.0]),
            {
                "context_store": context_store,
                "snapshot_store": snapshot_store,
                "solver": owner,
                "resource.threads": 1,
            },
        )

    assert rejected_context.wait(timeout=0.5)
    assert rejected_snapshot.wait(timeout=0.5)
    assert rejected_owner_context.wait(timeout=0.5)
    assert rejected_owner_snapshot.wait(timeout=0.5)
    assert context_store.get("late") is None
    assert context_store.get("late_owner") is None
    assert snapshot_store.read("late") is None
    assert snapshot_store.read("late_owner") is None


def test_parallel_runtime_handles_accept_writes_while_run_token_is_active() -> None:
    context_store = ContextStore()
    snapshot_store = InMemorySnapshotStore()

    class _Writer:
        def mutate(self, x, context=None):
            context["context_store"].set("during", context["pipeline.run_token"])
            context["snapshot_store"].write({"ok": True}, key="during")
            return x

    kernel = build_pipeline_kernel(
        {
            "slots": ({
                "slot": "mutate",
                "mode": "parallel",
                "operators": ("writer",),
                "strict": True,
            },)
        },
        operator_registry={"writer": _Writer()},
    )
    context = {
        "context_store": context_store,
        "snapshot_store": snapshot_store,
        "resource_context": {"threads": 1, "namespace": "project.case"},
    }

    outputs = kernel.run_slot("mutate", np.array([1.0]), context)

    assert len(outputs) == 1
    report = context["pipeline.parallel_report"]
    assert context_store.get("during") == report["run_token"]
    assert snapshot_store.read("during").data == {"ok": True}
    assert report["run_namespace"] == "project.case"
