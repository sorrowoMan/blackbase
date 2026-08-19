from __future__ import annotations

from blackbase.resources import (
    InMemoryTaskRuntimeBackend,
    ResourceRequirement,
    TaskEnvelope,
    TaskResult,
    WorkerDescriptor,
)


def _worker() -> WorkerDescriptor:
    return WorkerDescriptor(
        worker_id="worker-1",
        offer={"threads": 1, "gpus": 0, "backend": "local"},
        executor_backend="thread",
        capabilities=("nested_eval",),
        max_inflight=1,
    )


def test_task_runtime_preserves_claim_and_result_fencing() -> None:
    runtime = InMemoryTaskRuntimeBackend()
    task = TaskEnvelope(
        task_id="task-1",
        task_type="nested_candidate_eval",
        payload={"candidate": [1.0]},
        requirement=ResourceRequirement(threads=1, capabilities=("nested_eval",)),
        executor_backend="thread",
        namespace="run-1",
    )
    runtime.submit_many((task,))

    claim = runtime.claim_task(
        _worker(),
        run_id="run-1",
        task_types=("nested_candidate_eval",),
    )
    assert claim is not None
    runtime.complete_claim(
        claim,
        TaskResult(task_id=task.task_id, status="ok", objectives=(2.0,), violations=(0.0,)),
    )

    result = runtime.get_result("run-1", task.task_id)
    assert result is not None
    assert result.objectives == (2.0,)
    assert runtime.get_result("another-run", task.task_id) is None
