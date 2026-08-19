"""Shared task-runtime facade over the canonical :mod:`TaskTransport` contract."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .model import TaskEnvelope, TaskResult, WorkerDescriptor
from .transport import (
    ClaimedTask,
    InMemoryTaskTransport,
    RedisTaskTransport,
    SQLiteTaskTransport,
    TaskTransport,
)


@dataclass
class TaskRuntimeBackend:
    """Backend-neutral task runtime used by framework semantic workers.

    The runtime owns no Solver or Trainer behavior.  It only adapts the shared
    submit/claim/result/fencing transport to a convenient worker surface.
    """

    task_transport: TaskTransport

    def submit(self, task: TaskEnvelope | Mapping[str, Any]) -> None:
        self.task_transport.submit(task)

    def submit_many(self, tasks: Sequence[TaskEnvelope | Mapping[str, Any]]) -> None:
        for task in tuple(tasks):
            self.submit(task)

    def claim_task(
        self,
        worker: WorkerDescriptor | Mapping[str, Any],
        *,
        run_id: str | None = None,
        timeout_seconds: float = 1.0,
        lease_seconds: float = 30.0,
        task_types: Sequence[str] = (),
    ) -> ClaimedTask | None:
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        while True:
            claim = self.task_transport.claim(
                worker,
                lease_seconds=max(0.1, float(lease_seconds)),
                task_types=tuple(task_types),
                namespaces=((str(run_id),) if run_id else ()),
            )
            if claim is not None:
                return claim
            if time.monotonic() >= deadline:
                return None
            time.sleep(min(0.05, max(0.001, deadline - time.monotonic())))

    def complete_claim(self, claim: ClaimedTask, result: TaskResult | Mapping[str, Any]) -> None:
        task_result = result if isinstance(result, TaskResult) else TaskResult.from_dict(result)
        if task_result.ok:
            self.task_transport.complete(claim, task_result)
        else:
            self.task_transport.fail(claim, task_result)

    def get_result(self, run_id: str, task_id: str) -> TaskResult | None:
        record = self.task_transport.get(str(task_id))
        if record is None:
            return None
        if run_id and str(record.task.namespace) != str(run_id):
            return None
        return record.result if record.final else None

    def heartbeat(
        self,
        worker: WorkerDescriptor | str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        del payload
        if isinstance(worker, WorkerDescriptor):
            self.task_transport.register_worker(worker)
            return
        self.task_transport.heartbeat_worker(str(worker), status="online")


class InMemoryTaskRuntimeBackend(TaskRuntimeBackend):
    def __init__(self) -> None:
        super().__init__(task_transport=InMemoryTaskTransport())


class SQLiteTaskRuntimeBackend(TaskRuntimeBackend):
    def __init__(self, path: str) -> None:
        super().__init__(task_transport=SQLiteTaskTransport(path))


class RedisTaskRuntimeBackend(TaskRuntimeBackend):
    def __init__(
        self,
        *,
        redis_url: str = "redis://localhost:6379/0",
        namespace: str = "blackbase:tasks",
        client: Any = None,
    ) -> None:
        super().__init__(
            task_transport=RedisTaskTransport(
                redis_url=redis_url,
                namespace=namespace,
                client=client,
            )
        )


__all__ = [
    "InMemoryTaskRuntimeBackend",
    "RedisTaskRuntimeBackend",
    "SQLiteTaskRuntimeBackend",
    "TaskRuntimeBackend",
]
