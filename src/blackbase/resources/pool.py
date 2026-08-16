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
from concurrent.futures import (
    CancelledError,
    Executor,
    Future,
    ThreadPoolExecutor,
    wait as wait_futures,
)
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
        self._started = False
        self._lock = threading.Lock()

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

    def cancel(self) -> bool:
        """Cancel a task that has not started executing its callable."""
        with self._lock:
            if self._done.is_set() or self._started:
                return False
            self._exception = CancelledError("PoolTask was cancelled before execution")
            self._done.set()
            return True

    def _mark_started(self) -> bool:
        with self._lock:
            if self._done.is_set():
                return False
            self._started = True
            return True

    def _set_result(self, result: T) -> None:
        with self._lock:
            if self._done.is_set():
                return
            self._result = result
            self._done.set()

    def _set_exception(self, exc: BaseException) -> None:
        with self._lock:
            if self._done.is_set():
                return
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
        cancel_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self.task = task
        self.worker_id = str(worker_id)
        self.task_id = str(task_id or "")
        self.legacy_result = bool(legacy_result)
        self._cancel_callback = cancel_callback
        self._future: Optional[Future[Any]] = None
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

    def cancel(self) -> bool:
        cancelled = self.task.cancel()
        if cancelled:
            if self._future is not None:
                self._future.cancel()
            if self._cancel_callback is not None:
                self._cancel_callback()
        return cancelled


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
        self._lock = threading.Lock()
        self._capacity_changed = threading.Condition(self._lock)
        self._available_threads = self._total_threads
        self._shutdown = False
        self._executor = ThreadPoolExecutor(
            max_workers=self._total_threads,
            thread_name_prefix="blackbase-pool",
        )
        self._future_tasks: dict[Future[Any], PoolTask[Any]] = {}
        self._executor_views: set[Any] = set()
        self._task_local = threading.local()
        self._tasks_submitted = 0
        self._tasks_completed = 0
        self._tasks_failed = 0

    @property
    def total_threads(self) -> int:
        return int(self._total_threads)

    def available(self) -> int:
        with self._lock:
            return int(self._available_threads)

    def _notify_capacity_waiters(self) -> None:
        with self._capacity_changed:
            self._capacity_changed.notify_all()

    def _acquire_capacity(self, permits: int, task: PoolTask[Any]) -> bool:
        with self._capacity_changed:
            while (
                self._available_threads < permits
                and not self._shutdown
                and not task.done
            ):
                self._capacity_changed.wait()
            if self._shutdown or task.done:
                return False
            self._available_threads -= permits
            return True

    def _release_capacity(self, permits: int) -> None:
        with self._capacity_changed:
            self._available_threads = min(
                self._total_threads,
                self._available_threads + permits,
            )
            self._capacity_changed.notify_all()

    def submit(self, *args: Any, **kwargs: Any) -> PoolResult[Any]:
        """
        Submit a task.

        Supported forms:
        - new/shared: `submit(fn, *args, **kwargs)` returns raw `fn` result
        - legacy nsgablack: `submit(task_id, workers, fn, *args, **kwargs)`
          returns a `PoolTaskResult` from `.result()`
        """
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
        if int(getattr(self._task_local, "held_permits", 0)) > 0:
            raise RuntimeError(
                "submit cannot be called from a task running in the same "
                "PoolScheduler; derive a child resource context instead"
            )
        task: PoolTask[Any] = PoolTask(fn, *task_args, **kwargs)
        result = PoolResult(
            task,
            worker_id="",
            task_id=task_id,
            legacy_result=legacy,
            cancel_callback=self._notify_capacity_waiters,
        )

        def wrapper() -> None:
            acquired = False
            previous_permits = int(getattr(self._task_local, "held_permits", 0))
            try:
                acquired = self._acquire_capacity(permit_count, task)
                if not acquired:
                    task.cancel()
                    return
                if not task._mark_started():
                    return
                result.start_time = time.time()
                ident = threading.get_ident()
                result.worker_id = f"pool-{ident}"
                self._task_local.held_permits = previous_permits + permit_count
                raw_result = fn(*task_args, **kwargs)
                task._set_result(raw_result)
                with self._lock:
                    self._tasks_completed += 1
                return raw_result
            except BaseException as exc:
                task._set_exception(exc)
                with self._lock:
                    self._tasks_failed += 1
            finally:
                result.end_time = time.time()
                self._task_local.held_permits = previous_permits
                if acquired:
                    self._release_capacity(permit_count)

        with self._lock:
            if self._shutdown:
                raise RuntimeError("PoolScheduler has been shutdown")
            self._tasks_submitted += 1
            future = self._executor.submit(wrapper)
            self._future_tasks[future] = task
            result._future = future

        def discard_future(done: Future[Any]) -> None:
            with self._lock:
                self._future_tasks.pop(done, None)

        future.add_done_callback(discard_future)
        return result

    def as_executor(self, max_workers: Optional[int] = None) -> Executor:
        workers = self._total_threads if max_workers is None else max(1, int(max_workers))

        class _PoolExecutor(Executor):
            def __init__(self, pool: PoolScheduler, worker_count: int) -> None:
                self._pool = pool
                self._worker_count = max(1, min(int(worker_count), pool._total_threads))
                self._entered = False
                self._closed = False
                self._reserved = False
                self._executor: Optional[ThreadPoolExecutor] = None
                self._lease_task: PoolTask[None] = PoolTask(lambda: None)
                self._view_lock = threading.Lock()
                self._tasks: dict[Future[Any], PoolTask[Any]] = {}

            def __enter__(self) -> "_PoolExecutor":
                if int(getattr(self._pool._task_local, "held_permits", 0)) > 0:
                    raise RuntimeError(
                        "as_executor cannot be entered from a task running in the same "
                        "PoolScheduler; derive a child resource context instead"
                    )
                with self._view_lock:
                    if self._entered or self._closed:
                        raise RuntimeError("executor view cannot be entered more than once")
                with self._pool._lock:
                    if self._pool._shutdown:
                        raise RuntimeError("PoolScheduler has been shutdown")
                if not self._pool._acquire_capacity(
                    self._worker_count,
                    self._lease_task,
                ):
                    raise RuntimeError("PoolScheduler was shutdown before executor admission")
                executor: Optional[ThreadPoolExecutor] = None
                try:
                    executor = ThreadPoolExecutor(
                        max_workers=self._worker_count,
                        thread_name_prefix="blackbase-pool-view",
                    )
                    with self._view_lock:
                        self._executor = executor
                        self._reserved = True
                        self._entered = True
                    with self._pool._lock:
                        if self._pool._shutdown:
                            raise RuntimeError("PoolScheduler has been shutdown")
                        self._pool._executor_views.add(self)
                except BaseException:
                    with self._view_lock:
                        self._executor = None
                        self._reserved = False
                        self._entered = False
                        self._closed = True
                    if executor is not None:
                        executor.shutdown(wait=True, cancel_futures=True)
                    self._pool._release_capacity(self._worker_count)
                    raise
                return self

            def submit(self, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> Future[T]:
                if not callable(fn):
                    raise TypeError("executor submit expects a callable")
                if int(getattr(self._pool._task_local, "held_permits", 0)) > 0:
                    raise RuntimeError(
                        "executor submit cannot be called from a task running in the same "
                        "PoolScheduler; derive a child resource context instead"
                    )
                task: PoolTask[T] = PoolTask(fn, *args, **kwargs)

                def wrapper() -> T:
                    previous_permits = int(
                        getattr(self._pool._task_local, "held_permits", 0)
                    )
                    try:
                        if not task._mark_started():
                            raise CancelledError(
                                "Pool executor task was cancelled before execution"
                            )
                        self._pool._task_local.held_permits = previous_permits + 1
                        raw_result = fn(*args, **kwargs)
                        task._set_result(raw_result)
                        with self._pool._lock:
                            self._pool._tasks_completed += 1
                        return raw_result
                    except BaseException as exc:
                        task._set_exception(exc)
                        if not isinstance(exc, CancelledError):
                            with self._pool._lock:
                                self._pool._tasks_failed += 1
                        raise
                    finally:
                        self._pool._task_local.held_permits = previous_permits

                with self._view_lock:
                    executor = self._executor
                    if not self._entered or self._closed or executor is None:
                        raise RuntimeError("executor view is not active")
                    with self._pool._lock:
                        if self._pool._shutdown:
                            raise RuntimeError("PoolScheduler has been shutdown")
                        future: Future[T] = executor.submit(wrapper)
                        self._pool._tasks_submitted += 1
                        self._pool._future_tasks[future] = task
                    self._tasks[future] = task

                def discard_future(done: Future[Any]) -> None:
                    with self._pool._lock:
                        self._pool._future_tasks.pop(done, None)
                    with self._view_lock:
                        self._tasks.pop(done, None)
                    self._release_reservation_if_idle()

                future.add_done_callback(discard_future)
                return future

            def _release_reservation_if_idle(self) -> None:
                release = False
                with self._view_lock:
                    if self._closed and self._reserved and not self._tasks:
                        self._reserved = False
                        self._executor = None
                        release = True
                if not release:
                    return
                with self._pool._lock:
                    self._pool._executor_views.discard(self)
                self._pool._release_capacity(self._worker_count)

            def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
                with self._view_lock:
                    self._closed = True
                    pending = tuple(self._tasks.items())
                    executor = self._executor
                if cancel_futures:
                    for future, task in pending:
                        task.cancel()
                        future.cancel()
                    self._pool._notify_capacity_waiters()
                if executor is not None:
                    executor.shutdown(wait=wait, cancel_futures=cancel_futures)
                elif wait and pending:
                    wait_futures(tuple(future for future, _task in pending))
                self._release_reservation_if_idle()

            def __exit__(self, exc_type, exc_val, exc_tb) -> None:
                self.shutdown(wait=True)

        return _PoolExecutor(self, workers)  # type: ignore[return-value]

    def shutdown(self, wait: bool = True) -> None:
        with self._capacity_changed:
            if not self._shutdown:
                self._shutdown = True
            pending = tuple(self._future_tasks.items())
            views = tuple(self._executor_views)
            self._capacity_changed.notify_all()
        for future, task in pending:
            task.cancel()
            future.cancel()
        for view in views:
            view.shutdown(wait=wait, cancel_futures=True)
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def close(self) -> None:
        self.shutdown(wait=True)

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_threads": int(self._total_threads),
                "available_threads": int(self._available_threads),
                "inflight_tasks": int(len(self._future_tasks)),
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
