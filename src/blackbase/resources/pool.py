"""
Shared L0 compute pool.

This module is intentionally small: it gives both nsgablack and mlblack the
same local thread-pool surface without making either framework own the pool.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any, Callable, Generic, Optional, TypeVar

from .model import ResourceOffer, ResourceRequirement, WorkerDescriptor
from .probe import build_local_worker_descriptor, detect_local_resource_offer


T = TypeVar("T")


@dataclass(frozen=True)
class PoolTaskResult(Generic[T]):
    """Legacy-compatible completed task payload."""

    task_id: str
    result: T
    worker_id: str
    ok: bool = True
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0


class PoolTask(Generic[T]):
    """Task submitted to the shared local pool."""

    def __init__(self, fn: Callable[..., T], *args, **kwargs) -> None:
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self._result: Optional[T] = None
        self._exception: Optional[BaseException] = None
        self._done = threading.Event()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def result(self, timeout: Optional[float] = None) -> T:
        finished = self._done.wait(timeout=timeout)
        if not finished:
            raise TimeoutError("PoolTask result timed out")
        if self._exception is not None:
            raise self._exception
        return self._result  # type: ignore[return-value]

    def _set_result(self, result: T) -> None:
        self._result = result
        self._done.set()

    def _set_exception(self, exc: BaseException) -> None:
        self._exception = exc
        self._done.set()


class PoolResult(Generic[T]):
    """Result handle returned by `PoolScheduler.submit`."""

    def __init__(
        self,
        task: PoolTask[T],
        worker_id: str,
        *,
        task_id: str = "",
        legacy_result: bool = False,
    ) -> None:
        self.task = task
        self.worker_id = str(worker_id)
        self.task_id = str(task_id or "")
        self.legacy_result = bool(legacy_result)
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    @property
    def done(self) -> bool:
        return self.task.done

    def result(self, timeout: Optional[float] = None) -> T | PoolTaskResult[T]:
        raw = self.task.result(timeout=timeout)
        if self.legacy_result:
            return PoolTaskResult(
                task_id=str(self.task_id or ""),
                result=raw,
                worker_id=str(self.worker_id),
                started_at=float(self.start_time or 0.0),
                finished_at=float(self.end_time or 0.0),
            )
        return raw


class PoolScheduler:
    """
    Shared local thread-pool scheduler.

    The project substrate may inject a pool into a Case. A Case may consume this
    object for local fanout, but lease ownership remains at Project/L0 level.
    """

    def __init__(self, total_threads: Optional[int] = None, *, threads: Optional[int] = None) -> None:
        requested = total_threads if total_threads is not None else threads
        if requested is None:
            requested = max(1, os.cpu_count() or 4)
        self._total_threads = max(1, int(requested))
        self._available = threading.Semaphore(self._total_threads)
        self._shutdown = False
        self._lock = threading.Lock()
        self._tasks_submitted = 0
        self._tasks_completed = 0
        self._tasks_failed = 0

    @property
    def total_threads(self) -> int:
        return int(self._total_threads)

    def available(self) -> int:
        return int(getattr(self._available, "_value", 0))

    def submit(self, *args: Any, **kwargs: Any) -> PoolResult[Any]:
        """
        Submit a task.

        Supported forms:
        - new/shared: `submit(fn, *args, **kwargs)` returns raw `fn` result
        - legacy nsgablack: `submit(task_id, workers, fn, *args, **kwargs)`
          returns a `PoolTaskResult` from `.result()`
        """
        if self._shutdown:
            raise RuntimeError("PoolScheduler has been shutdown")

        if args and callable(args[0]):
            legacy = False
            task_id = ""
            workers = 1
            fn = args[0]
            task_args = args[1:]
        elif len(args) >= 3 and callable(args[2]):
            legacy = True
            task_id = str(args[0])
            workers = max(1, int(args[1] or 1))
            fn = args[2]
            task_args = args[3:]
        else:
            raise TypeError("submit expects (fn, *args) or (task_id, workers, fn, *args)")

        permit_count = max(1, min(int(workers), self._total_threads))
        task: PoolTask[Any] = PoolTask(fn, *task_args, **kwargs)
        result = PoolResult(task, worker_id="", task_id=task_id, legacy_result=legacy)
        with self._lock:
            self._tasks_submitted += 1

        def wrapper() -> None:
            acquired = 0
            result.start_time = time.time()
            ident = threading.get_ident()
            result.worker_id = f"pool-{ident}"
            try:
                for _ in range(permit_count):
                    self._available.acquire()
                    acquired += 1
                task._set_result(fn(*task_args, **kwargs))
                with self._lock:
                    self._tasks_completed += 1
            except BaseException as exc:
                task._set_exception(exc)
                with self._lock:
                    self._tasks_failed += 1
            finally:
                result.end_time = time.time()
                for _ in range(acquired):
                    self._available.release()

        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()
        return result

    def as_executor(self, max_workers: Optional[int] = None) -> Executor:
        workers = self._total_threads if max_workers is None else max(1, int(max_workers))

        class _PoolExecutor:
            def __init__(self, pool: PoolScheduler, worker_count: int) -> None:
                self._pool = pool
                self._worker_count = max(1, min(int(worker_count), pool._total_threads))
                self._executor: Optional[ThreadPoolExecutor] = None
                self._acquired = 0

            def __enter__(self) -> ThreadPoolExecutor:
                if self._pool._shutdown:
                    raise RuntimeError("PoolScheduler has been shutdown")
                for _ in range(self._worker_count):
                    self._pool._available.acquire()
                    self._acquired += 1
                self._executor = ThreadPoolExecutor(max_workers=self._worker_count)
                return self._executor

            def __exit__(self, exc_type, exc_val, exc_tb) -> None:
                if self._executor is not None:
                    self._executor.shutdown(wait=True)
                for _ in range(self._acquired):
                    self._pool._available.release()
                self._acquired = 0

        return _PoolExecutor(self, workers)  # type: ignore[return-value]

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            self._shutdown = True

    def close(self) -> None:
        self.shutdown(wait=True)

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_threads": int(self._total_threads),
                "available_threads": int(self.available()),
                "tasks_submitted": int(self._tasks_submitted),
                "tasks_completed": int(self._tasks_completed),
                "tasks_failed": int(self._tasks_failed),
                "shutdown": bool(self._shutdown),
            }


__all__ = [
    "WorkerDescriptor",
    "ResourceOffer",
    "ResourceRequirement",
    "build_local_worker_descriptor",
    "detect_local_resource_offer",
    "PoolScheduler",
    "PoolTask",
    "PoolResult",
    "PoolTaskResult",
]
