"""
Pipeline orchestration policy and executor.

Provides the orchestration logic for running pipeline operators
in different execution modes (serial, parallel, router).
"""

from __future__ import annotations

import concurrent.futures
import copy
import inspect
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence


# ============================================================================
# Orchestration Policy
# ============================================================================


@dataclass(frozen=True)
class ParallelBranchFailure:
    """Structured failure information for one parallel pipeline branch."""

    index: int
    operator: str
    error_type: str
    message: str
    timed_out: bool = False
    cancelled: bool = False
    still_running: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "operator": str(self.operator),
            "error_type": str(self.error_type),
            "message": str(self.message),
            "timed_out": bool(self.timed_out),
            "cancelled": bool(self.cancelled),
            "still_running": bool(self.still_running),
        }


class PipelineParallelError(RuntimeError):
    """Raised after strict parallel execution collects branch failures."""

    def __init__(
        self,
        failures: Sequence[ParallelBranchFailure],
        *,
        partial_results: Sequence[Any] = (),
    ) -> None:
        self.failures = tuple(failures)
        self.partial_results = tuple(partial_results)
        summary = "; ".join(
            f"branch[{item.index}] {item.operator}: {item.error_type}: {item.message}"
            for item in self.failures
        )
        super().__init__(f"parallel pipeline execution failed: {summary}")


class PipelineCancellationError(concurrent.futures.CancelledError):
    """Raised at an operator boundary after cooperative cancellation."""


class PipelineLateWriteRejected(RuntimeError):
    """Raised when a cancelled or completed branch writes through a runtime handle."""


class _PipelineRunControl:
    """One-way cooperative cancellation signal and late-write fence."""

    def __init__(
        self,
        *,
        run_token: str,
        namespace: str,
        parent_event: Any = None,
    ) -> None:
        self.run_token = str(run_token)
        self.namespace = str(namespace)
        self._parent_event = parent_event
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._active = True
        self._reason = ""

    @staticmethod
    def _event_is_set(event: Any) -> bool:
        checker = getattr(event, "is_set", None)
        return bool(callable(checker) and checker())

    def is_set(self) -> bool:
        return self._event.is_set() or self._event_is_set(self._parent_event)

    def set(self) -> None:
        self.cancel("operator requested cancellation")

    def wait(self, timeout: Optional[float] = None) -> bool:
        if self._parent_event is None:
            return self._event.wait(timeout=timeout)
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while not self.is_set():
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._event.wait(timeout=min(0.01, remaining))
            else:
                self._event.wait(timeout=0.01)
        return True

    @property
    def reason(self) -> str:
        if self._reason:
            return self._reason
        if self._event_is_set(self._parent_event):
            return "parent pipeline cancellation requested"
        return "pipeline cancellation requested"

    def cancel(self, reason: str) -> None:
        with self._lock:
            if not self._reason:
                self._reason = str(reason)
            self._event.set()

    def close(self) -> None:
        with self._lock:
            self._active = False

    def raise_if_cancelled(self) -> None:
        if self.is_set():
            raise PipelineCancellationError(self.reason)

    def assert_writable(self, handle_name: str) -> None:
        with self._lock:
            active = self._active
        if self.is_set() or not active:
            state = "cancelled" if self.is_set() else "completed"
            raise PipelineLateWriteRejected(
                f"pipeline {handle_name} write rejected: run_token={self.run_token!r}, "
                f"namespace={self.namespace!r}, state={state}"
            )


class _CancellationFencedHandle:
    """Read-through runtime handle whose mutating methods honor the run fence."""

    _MUTATING_METHODS = frozenset(
        {
            "set",
            "update",
            "delete",
            "clear",
            "write",
            "write_snapshot",
            "write_population_snapshot",
            "commit_population_snapshot",
            "set_context_store",
            "set_snapshot_store",
        }
    )
    _STORE_ATTRIBUTES = frozenset({"context_store", "snapshot_store"})

    def __init__(self, target: Any, control: _PipelineRunControl, name: str) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_control", control)
        object.__setattr__(self, "_name", str(name))

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._target, name)
        if name in self._STORE_ATTRIBUTES and attribute is not None:
            return _CancellationFencedHandle(
                attribute,
                self._control,
                f"{self._name}.{name}",
            )
        if name not in self._MUTATING_METHODS or not callable(attribute):
            return attribute

        def fenced_call(*args: Any, **kwargs: Any) -> Any:
            self._control.assert_writable(self._name)
            return attribute(*args, **kwargs)

        return fenced_call

    def __getitem__(self, key: Any) -> Any:
        return self._target[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._control.assert_writable(self._name)
        self._target[key] = value

    def __delitem__(self, key: Any) -> None:
        self._control.assert_writable(self._name)
        del self._target[key]

    def __contains__(self, key: Any) -> bool:
        return key in self._target

    def __iter__(self):
        return iter(self._target)

    def __len__(self) -> int:
        return len(self._target)


@dataclass(frozen=True)
class OrchestrationPolicy:
    """
    Policy for executing operators within a pipeline slot.
    
    Defines how operators should be executed, including mode selection,
    routing configuration, and merge strategy.
    """
    
    mode: str = "serial"                              # Execution mode
    operators: Sequence[Any] = ()                     # Resolved operators
    routes: Mapping[str, Any] = field(default_factory=dict)  # Route key -> operator
    selector_key: str = "strategy_id"                # Context key for selection
    index_key: str = "vns_k"                         # Context key for index
    default_operator: Any = None
    stages: Sequence[tuple[int, Any]] = ()              # generation threshold -> operator
    strict: bool = False                             # Strict operator resolution
    merge: str | Callable[..., Any] | None = None   # Merge strategy
    timeout_seconds: Optional[float] = None          # Whole-slot deadline
    cancel_on_error: bool = True                     # Cancel pending work in strict mode
    
    def select_operator(self, context: Optional[Mapping[str, Any]]) -> Any:
        """Select an operator based on context."""
        mode = str(self.mode or "serial").strip().lower()
        ctx = context or {}
        if mode == "switch":
            operators = tuple(self.operators or ())
            if not operators:
                return None
            try:
                index = int(ctx.get(self.index_key, 0))
            except (TypeError, ValueError):
                index = 0
            index = max(0, min(index, len(operators) - 1))
            return operators[index]
        if mode == "dynamic":
            try:
                generation = int(ctx.get("generation", 0))
            except (TypeError, ValueError):
                generation = 0
            selected = None
            for start, operator in sorted(
                ((int(start), operator) for start, operator in tuple(self.stages or ())),
                key=lambda item: item[0],
            ):
                if generation < start:
                    break
                selected = operator
            return selected if selected is not None else self.default_operator
        if mode == "router" and self.routes:
            selector = ctx.get(self.selector_key)
            if selector is not None:
                key = str(selector)
                if key in self.routes:
                    return self.routes[key]
            # Try index-based selection
            index = ctx.get(self.index_key)
            if index is not None:
                route_keys = list(self.routes.keys())
                try:
                    idx = int(index) % len(route_keys)
                    return self.routes[route_keys[idx]]
                except (ValueError, TypeError, IndexError):
                    pass
            # Fallback to default
            if self.default_operator is not None:
                return self.default_operator
        return None
    
    def as_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": str(self.mode),
            "operators": [op.__name__ if hasattr(op, "__name__") else str(op) for op in self.operators],
            "routes": {str(k): v.__name__ if hasattr(v, "__name__") else str(v) for k, v in self.routes.items()},
            "selector_key": str(self.selector_key),
            "index_key": str(self.index_key),
            "has_default": self.default_operator is not None,
            "stages": [int(start) for start, _ in tuple(self.stages or ())],
            "strict": bool(self.strict),
            "merge": self.merge,
            "timeout_seconds": self.timeout_seconds,
            "cancel_on_error": bool(self.cancel_on_error),
        }


# ============================================================================
# Pipeline Orchestrator
# ============================================================================

class PipelineOrchestrator:
    """
    Executes pipeline operators according to orchestration policies.
    
    Supports serial, parallel, and router execution modes.
    """
    
    def __init__(
        self,
        *,
        strict: bool = False,
        executor: Any = None,
        pool_scheduler: Any = None,
    ) -> None:
        self.strict = bool(strict)
        self.executor = executor
        self.pool_scheduler = pool_scheduler
        self.last_parallel_report: dict[str, Any] = {}

    @staticmethod
    def _runtime_handle_keys() -> frozenset[str]:
        return frozenset(
            {
                "executor",
                "parallel_executor",
                "pool",
                "pool_scheduler",
                "resource_pool",
                "solver",
                "trainer",
                "control",
                "plugin_manager",
                "context_store",
                "snapshot_store",
                "pipeline.cancel_event",
                "pipeline.run_token",
                "pipeline.run_namespace",
            }
        )

    def _copy_parallel_value(
        self,
        value: Any,
        *,
        label: str,
        strict: Optional[bool] = None,
    ) -> Any:
        try:
            return copy.deepcopy(value)
        except Exception as exc:
            copier = getattr(value, "copy", None)
            if callable(copier):
                try:
                    return copier()
                except Exception:
                    pass
            if bool(self.strict if strict is None else strict):
                raise TypeError(f"parallel {label} is not safely copyable") from exc
            return value

    def _copy_parallel_context(
        self,
        context: Optional[MutableMapping[str, Any]],
        *,
        strict: Optional[bool] = None,
        run_control: Optional[_PipelineRunControl] = None,
    ) -> Optional[MutableMapping[str, Any]]:
        if context is None:
            return None
        cloned: dict[str, Any] = {}
        handle_keys = self._runtime_handle_keys()
        for key, item in dict(context).items():
            if str(key) in handle_keys:
                if run_control is not None and str(key) in {
                    "context_store",
                    "snapshot_store",
                    "solver",
                    "trainer",
                    "control",
                }:
                    cloned[key] = _CancellationFencedHandle(
                        item,
                        run_control,
                        str(key),
                    )
                else:
                    cloned[key] = item
                continue
            cloned[key] = self._copy_parallel_value(
                item,
                label=f"context[{key!r}]",
                strict=strict,
            )
        if run_control is not None:
            cloned["pipeline.cancel_event"] = run_control
            cloned["pipeline.run_token"] = run_control.run_token
            cloned["pipeline.run_namespace"] = run_control.namespace
        return cloned

    @staticmethod
    def _context_value(context: Optional[Mapping[str, Any]], *keys: str) -> Any:
        ctx = context or {}
        for key in keys:
            if key in ctx and ctx[key] is not None:
                return ctx[key]
        return None

    def _parallel_worker_limit(
        self,
        context: Optional[Mapping[str, Any]],
        branch_count: int,
    ) -> int:
        workers = max(1, int(branch_count))
        ctx = context or {}
        raw_threads = self._context_value(ctx, "resource.threads")
        if raw_threads is None:
            resource = self._context_value(ctx, "resource_context", "resource.context", "resource")
            if resource is not None:
                try:
                    from ..resources import coerce_resource_context

                    raw_threads = coerce_resource_context(resource).threads
                except Exception:
                    raw_threads = None
        if raw_threads is not None:
            workers = min(workers, max(1, int(raw_threads)))
        return workers

    def _parallel_runtime(self, context: Optional[Mapping[str, Any]]) -> tuple[Any, Any]:
        ctx = context or {}
        executor = self._context_value(ctx, "parallel_executor", "executor") or self.executor
        pool = self._context_value(ctx, "pool_scheduler", "resource_pool", "pool") or self.pool_scheduler
        return executor, pool

    @staticmethod
    def _operator_name(operator: Any) -> str:
        return str(
            getattr(operator, "name", None)
            or getattr(operator, "__name__", None)
            or type(operator).__name__
        )

    @staticmethod
    def _remaining_timeout(deadline: Optional[float]) -> Optional[float]:
        if deadline is None:
            return None
        return max(0.0, float(deadline) - time.monotonic())

    @staticmethod
    def _cancel_requested(context: Optional[Mapping[str, Any]]) -> bool:
        event = (context or {}).get("pipeline.cancel_event")
        checker = getattr(event, "is_set", None)
        return bool(callable(checker) and checker())

    @classmethod
    def _raise_if_cancelled(cls, context: Optional[Mapping[str, Any]]) -> None:
        if cls._cancel_requested(context):
            event = (context or {}).get("pipeline.cancel_event")
            reason = str(getattr(event, "reason", "pipeline cancellation requested"))
            raise PipelineCancellationError(reason)

    def _record_parallel_report(
        self,
        context: Optional[MutableMapping[str, Any]],
        *,
        branch_count: int,
        successes: Mapping[int, Any],
        failures: Sequence[ParallelBranchFailure],
        elapsed_seconds: float,
        run_control: Optional[_PipelineRunControl] = None,
    ) -> None:
        report = {
            "branch_count": int(branch_count),
            "completed_count": int(len(successes)),
            "failure_count": int(len(failures)),
            "success_indices": sorted(int(index) for index in successes),
            "failures": [failure.as_dict() for failure in failures],
            "timed_out": any(failure.timed_out for failure in failures),
            "cancelled": any(failure.cancelled for failure in failures),
            "still_running": any(failure.still_running for failure in failures),
            "still_running_count": sum(
                1 for failure in failures if failure.still_running
            ),
            "cancellation_requested": bool(
                run_control is not None and run_control.is_set()
            ),
            "run_token": "" if run_control is None else run_control.run_token,
            "run_namespace": "" if run_control is None else run_control.namespace,
            "elapsed_seconds": max(0.0, float(elapsed_seconds)),
        }
        self.last_parallel_report = report
        if isinstance(context, MutableMapping):
            context["pipeline.parallel_report"] = dict(report)

    def _collect_future_batch(
        self,
        future_map: Mapping[Any, tuple[int, Any]],
        *,
        deadline: Optional[float],
        strict: bool,
        cancel_on_error: bool,
        run_control: _PipelineRunControl,
    ) -> tuple[dict[int, Any], list[ParallelBranchFailure], bool]:
        successes: dict[int, Any] = {}
        failures: list[ParallelBranchFailure] = []
        pending = set(future_map)
        abort = False

        def cancel_pending(*, reason: str, timed_out: bool = False) -> None:
            nonlocal abort
            abort = True
            run_control.cancel(reason)
            for future in tuple(pending):
                index, operator = future_map[future]
                cancelled = bool(future.cancel())
                still_running = not cancelled and not future.done()
                failures.append(
                    ParallelBranchFailure(
                        index=index,
                        operator=self._operator_name(operator),
                        error_type="TimeoutError" if timed_out else "CancelledError",
                        message=reason,
                        timed_out=timed_out,
                        cancelled=cancelled,
                        still_running=still_running,
                    )
                )
            pending.clear()

        while pending:
            if run_control.is_set():
                cancel_pending(reason=run_control.reason)
                break
            remaining = self._remaining_timeout(deadline)
            if remaining is not None and remaining <= 0:
                cancel_pending(reason="parallel slot deadline exceeded", timed_out=True)
                break
            done, _ = concurrent.futures.wait(
                pending,
                timeout=remaining,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                cancel_pending(reason="parallel slot deadline exceeded", timed_out=True)
                break
            batch_failed = False
            for future in done:
                pending.discard(future)
                index, operator = future_map[future]
                try:
                    successes[index] = future.result()
                except Exception as exc:
                    batch_failed = True
                    failures.append(
                        ParallelBranchFailure(
                            index=index,
                            operator=self._operator_name(operator),
                            error_type=type(exc).__name__,
                            message=str(exc),
                            cancelled=isinstance(exc, concurrent.futures.CancelledError),
                            still_running=False,
                        )
                    )
            if batch_failed and strict and cancel_on_error and pending:
                cancel_pending(reason="cancelled after strict branch failure")
        return successes, failures, abort

    def _collect_pool_batch(
        self,
        handle_map: Mapping[Any, tuple[int, Any]],
        *,
        deadline: Optional[float],
        strict: bool,
        cancel_on_error: bool,
        run_control: _PipelineRunControl,
    ) -> tuple[dict[int, Any], list[ParallelBranchFailure], bool]:
        successes: dict[int, Any] = {}
        failures: list[ParallelBranchFailure] = []
        pending = set(handle_map)
        abort = False

        def cancel_pending(*, reason: str, timed_out: bool = False) -> None:
            nonlocal abort
            abort = True
            run_control.cancel(reason)
            for handle in tuple(pending):
                index, operator = handle_map[handle]
                cancel = getattr(handle, "cancel", None)
                cancelled = bool(callable(cancel) and cancel())
                still_running = not cancelled and not bool(getattr(handle, "done", False))
                failures.append(
                    ParallelBranchFailure(
                        index=index,
                        operator=self._operator_name(operator),
                        error_type="TimeoutError" if timed_out else "CancelledError",
                        message=reason,
                        timed_out=timed_out,
                        cancelled=cancelled,
                        still_running=still_running,
                    )
                )
            pending.clear()

        while pending:
            if run_control.is_set():
                cancel_pending(reason=run_control.reason)
                break
            remaining = self._remaining_timeout(deadline)
            if remaining is not None and remaining <= 0:
                cancel_pending(reason="parallel slot deadline exceeded", timed_out=True)
                break
            done = [handle for handle in pending if bool(getattr(handle, "done", False))]
            if not done:
                time.sleep(min(0.002, remaining) if remaining is not None else 0.002)
                continue
            batch_failed = False
            for handle in done:
                pending.discard(handle)
                index, operator = handle_map[handle]
                try:
                    successes[index] = handle.result(timeout=0)
                except Exception as exc:
                    batch_failed = True
                    failures.append(
                        ParallelBranchFailure(
                            index=index,
                            operator=self._operator_name(operator),
                            error_type=type(exc).__name__,
                            message=str(exc),
                            cancelled=isinstance(exc, concurrent.futures.CancelledError),
                            still_running=False,
                        )
                    )
            if batch_failed and strict and cancel_on_error and pending:
                cancel_pending(reason="cancelled after strict branch failure")
        return successes, failures, abort
    
    def _run_serial(
        self,
        operators: Sequence[Any],
        value: Any,
        context: Optional[MutableMapping[str, Any]],
        method: str,
        *,
        strict: bool,
    ) -> Any:
        """Run operators in serial mode."""
        result = value
        for op in operators:
            if op is None:
                continue
            try:
                self._raise_if_cancelled(context)
                result = self._call_operator(op, result, context, method)
                self._raise_if_cancelled(context)
            except PipelineCancellationError:
                raise
            except Exception:
                if bool(strict):
                    raise
                return result
        return result
    
    def _run_parallel(
        self,
        operators: Sequence[Any],
        value: Any,
        context: Optional[MutableMapping[str, Any]],
        method: str,
        *,
        strict: bool,
        timeout_seconds: Optional[float] = None,
        cancel_on_error: bool = True,
    ) -> tuple[Any, ...]:
        """Run isolated branches with deadline, cancellation, and failure audit."""
        branches = tuple(operators)
        if not branches:
            self._record_parallel_report(
                context,
                branch_count=0,
                successes={},
                failures=(),
                elapsed_seconds=0.0,
            )
            return ()
        worker_limit = self._parallel_worker_limit(context, len(branches))
        context_timeout = self._context_value(context, "pipeline.parallel_timeout_seconds")
        raw_timeout = context_timeout if context_timeout is not None else timeout_seconds
        timeout = None if raw_timeout is None else max(0.0, float(raw_timeout))
        deadline = None if timeout is None else time.monotonic() + timeout
        started_at = time.monotonic()
        parent_cancel_event = (context or {}).get("pipeline.cancel_event")
        namespace = str(
            self._context_value(
                context,
                "pipeline.run_namespace",
                "resource.namespace",
            )
            or "pipeline"
        )
        resource = self._context_value(
            context,
            "resource_context",
            "resource.context",
            "resource",
        )
        if namespace == "pipeline" and resource is not None:
            namespace = str(
                getattr(resource, "namespace", None)
                or (resource.get("namespace") if isinstance(resource, Mapping) else "")
                or namespace
            )
        run_control = _PipelineRunControl(
            run_token=uuid.uuid4().hex,
            namespace=namespace,
            parent_event=parent_cancel_event,
        )

        def run_op(item: tuple[int, Any]) -> Any:
            _, op = item
            run_control.raise_if_cancelled()
            branch_value = self._copy_parallel_value(value, label="input", strict=strict)
            branch_context = self._copy_parallel_context(
                context,
                strict=strict,
                run_control=run_control,
            )
            if op is None:
                return branch_value
            run_control.raise_if_cancelled()
            result = self._call_operator(op, branch_value, branch_context, method)
            run_control.raise_if_cancelled()
            return result

        injected_executor, pool = self._parallel_runtime(context)
        indexed = tuple(enumerate(branches))
        successes: dict[int, Any] = {}
        failures: list[ParallelBranchFailure] = []
        abort = False
        owned_executor: Any = None

        try:
            if pool is not None and callable(getattr(pool, "submit", None)):
                for start in range(0, len(indexed), worker_limit):
                    batch = indexed[start : start + worker_limit]
                    handles = {pool.submit(run_op, item): item for item in batch}
                    ok, failed, abort = self._collect_pool_batch(
                        handles,
                        deadline=deadline,
                        strict=strict,
                        cancel_on_error=cancel_on_error,
                        run_control=run_control,
                    )
                    successes.update(ok)
                    failures.extend(failed)
                    if abort:
                        break
            else:
                executor = injected_executor
                if executor is None:
                    owned_executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=worker_limit
                    )
                    executor = owned_executor
                submit = getattr(executor, "submit", None)
                if not callable(submit):
                    raise TypeError("parallel executor must provide submit()")
                for start in range(0, len(indexed), worker_limit):
                    batch = indexed[start : start + worker_limit]
                    futures = {submit(run_op, item): item for item in batch}
                    ok, failed, abort = self._collect_future_batch(
                        futures,
                        deadline=deadline,
                        strict=strict,
                        cancel_on_error=cancel_on_error,
                        run_control=run_control,
                    )
                    successes.update(ok)
                    failures.extend(failed)
                    if abort:
                        break
        finally:
            try:
                if owned_executor is not None:
                    owned_executor.shutdown(wait=not abort, cancel_futures=bool(abort))
            finally:
                run_control.close()

        completed_indices = set(successes)
        failed_indices = {failure.index for failure in failures}
        if abort:
            timed_out = any(failure.timed_out for failure in failures)
            for index, operator in indexed:
                if index in completed_indices or index in failed_indices:
                    continue
                failures.append(
                    ParallelBranchFailure(
                        index=index,
                        operator=self._operator_name(operator),
                        error_type="TimeoutError" if timed_out else "CancelledError",
                        message=(
                            "parallel slot deadline exceeded before branch submission"
                            if timed_out
                            else "cancelled before branch submission"
                        ),
                        timed_out=timed_out,
                        cancelled=True,
                    )
                )

        ordered_results = tuple(successes[index] for index in sorted(successes))
        failures.sort(key=lambda item: item.index)
        self._record_parallel_report(
            context,
            branch_count=len(branches),
            successes=successes,
            failures=failures,
            elapsed_seconds=time.monotonic() - started_at,
            run_control=run_control,
        )
        if failures and strict:
            raise PipelineParallelError(failures, partial_results=ordered_results)
        return ordered_results

    def _merge_parallel_results(
        self,
        results: Sequence[Any],
        *,
        strategy: str | Callable[..., Any] | None,
        fallback: Any,
        method: str,
    ) -> Any:
        """Merge parallel branch outputs according to the declared policy."""
        outputs = tuple(results)
        if not outputs:
            return fallback
        if callable(strategy):
            return strategy(outputs, method=method, input_value=fallback)
        if strategy is None or not str(strategy).strip():
            # Preserve the historical no-merge contract.
            return outputs

        merge = str(strategy).strip().lower()
        if merge in {"last", "override"}:
            return outputs[-1]
        if merge == "first":
            return outputs[0]
        if merge in {"list", "collect"}:
            return list(outputs)

        import numpy as np

        if merge in {"sum", "mean", "avg"}:
            stacked = np.stack([np.asarray(item, dtype=float) for item in outputs], axis=0)
            if merge == "sum":
                return np.sum(stacked, axis=0)
            return np.mean(stacked, axis=0)
        if merge in {"concat", "concatenate"}:
            return np.concatenate([np.atleast_1d(np.asarray(item)) for item in outputs], axis=0)
        raise ValueError(f"unsupported parallel merge strategy: {strategy!r}")
    
    def _run_router(
        self,
        policy: OrchestrationPolicy,
        value: Any,
        context: Optional[MutableMapping[str, Any]],
        method: str,
        fallback: Any,
    ) -> Any:
        """Run operators in router mode (select one based on context)."""
        self._raise_if_cancelled(context)
        operator = policy.select_operator(context)
        if operator is None:
            if bool(policy.strict) and str(policy.mode).strip().lower() == "router":
                raise KeyError(
                    "pipeline route not found: "
                    f"selector_key={policy.selector_key!r}, "
                    f"value={(context or {}).get(policy.selector_key)!r}"
                )
            return fallback
        result = self._call_operator(operator, value, context, method)
        self._raise_if_cancelled(context)
        return result
    
    def _call_operator(
        self,
        operator: Any,
        value: Any,
        context: Optional[MutableMapping[str, Any]],
        method: str,
    ) -> Any:
        """Call an operator with the given value and context."""
        # First support component objects whose semantics live in a slot
        # method such as initialize/mutate/repair rather than __call__.
        method_fn = getattr(operator, method, None)
        if callable(method_fn):
            return self._invoke_operator_callable(
                method_fn,
                ((value, context), (value,), ()),
            )

        # Then support plain callables.
        if callable(operator):
            return self._invoke_operator_callable(
                operator,
                ((value, context), (value,), ()),
            )
        return value

    @staticmethod
    def _invoke_operator_callable(
        fn: Callable[..., Any],
        candidates: Sequence[tuple[Any, ...]],
    ) -> Any:
        """Bind once, then invoke once so body TypeError is never retried."""

        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            # Some extension/builtin callables do not expose a signature. Use
            # the canonical richest form and preserve its original exception.
            return fn(*tuple(candidates[0]))
        for args in candidates:
            try:
                signature.bind(*args)
            except TypeError:
                continue
            return fn(*args)
        raise TypeError(
            f"pipeline operator {fn!r} cannot bind any supported call form: "
            "(value, context), (value), or ()"
        )
    
    def _run_policy(
        self,
        policy: OrchestrationPolicy,
        value: Any,
        context: Optional[MutableMapping[str, Any]],
        method: str,
        fallback: Any,
    ) -> Any:
        """Run operators according to policy."""
        mode = str(policy.mode or "serial").lower()
        
        if mode == "serial":
            return self._run_serial(
                policy.operators,
                value,
                context,
                method,
                strict=bool(policy.strict),
            )
        
        if mode == "parallel":
            results = self._run_parallel(
                policy.operators,
                value,
                context,
                method,
                strict=bool(policy.strict),
                timeout_seconds=policy.timeout_seconds,
                cancel_on_error=bool(policy.cancel_on_error),
            )
            return self._merge_parallel_results(
                results,
                strategy=policy.merge,
                fallback=fallback,
                method=method,
            )
        
        if mode in {"router", "switch", "dynamic"}:
            return self._run_router(policy, value, context, method, fallback)

        if bool(policy.strict):
            raise ValueError(f"unsupported orchestration mode: {mode!r}")
        return fallback


# ============================================================================
# Pipeline Kernel Build
# ============================================================================

@dataclass
class PipelineKernelBuild:
    """
    Result of building a pipeline kernel.
    
    Contains the representation pipeline, slot runners, policies, and registry.
    """
    
    slot_runners: Mapping[str, Callable[[Any, Optional[MutableMapping[str, Any]]], Any]] = field(default_factory=dict)
    slot_policies: Mapping[str, OrchestrationPolicy] = field(default_factory=dict)
    operator_registry: Mapping[str, Any] = field(default_factory=dict)
    
    def run_slot(
        self,
        slot: str,
        value: Any,
        context: Optional[MutableMapping[str, Any]] = None,
    ) -> Any:
        """Run a specific slot with the given value."""
        key = normalize_slot_name(slot)
        runner = self.slot_runners.get(key)
        if runner is None:
            return value
        return runner(value, context)
    
    def as_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "slots": sorted(self.slot_runners.keys()),
            "slot_policies": {key: policy.as_dict() for key, policy in self.slot_policies.items()},
        }


# Import normalize_slot_name
from .spec import normalize_slot_name


def build_pipeline_kernel(
    spec: Mapping[str, Any] | None,
    *,
    operator_registry: Mapping[str, Any],
    strict: bool = True,
    executor: Any = None,
    pool_scheduler: Any = None,
) -> PipelineKernelBuild:
    """
    Build a pipeline kernel from specification.
    
    Args:
        spec: Pipeline specification
        operator_registry: Registry of available operators
        strict: Strict operator resolution mode
    
    Returns:
        PipelineKernelBuild with runners and policies
    """
    from .spec import PipelineSpec, PipelineSlotSpec, get_method_for_slot, normalize_slot_name, is_pipeline_slot
    
    parsed = PipelineSpec.from_value(spec)
    orchestrator = PipelineOrchestrator(
        strict=bool(strict),
        executor=executor,
        pool_scheduler=pool_scheduler,
    )
    slot_runners: dict[str, Callable[[Any, Optional[MutableMapping[str, Any]]], Any]] = {}
    slot_policies: dict[str, OrchestrationPolicy] = {}
    
    for slot_spec in parsed.slot_specs():
        slot_name = normalize_slot_name(slot_spec.slot)
        if not slot_name:
            continue
        
        method = str(slot_spec.method or get_method_for_slot(slot_name)).strip()
        policy = _build_policy_from_slot(slot_spec, operator_registry=operator_registry, strict=bool(strict))
        slot_policies[slot_name] = policy
        slot_runners[slot_name] = _make_slot_runner(orchestrator, policy, method=method)
    
    return PipelineKernelBuild(
        slot_runners=slot_runners,
        slot_policies=slot_policies,
        operator_registry=dict(operator_registry),
    )


def _build_policy_from_slot(
    slot: PipelineSlotSpec,
    *,
    operator_registry: Mapping[str, Any],
    strict: bool,
) -> OrchestrationPolicy:
    """Build orchestration policy from slot spec."""
    local_strict = bool(strict if slot.strict is None else slot.strict)
    
    operators = tuple(_resolve_operator(name, operator_registry, strict=local_strict) for name in slot.operators)
    routes = {
        key: _resolve_operator(name, operator_registry, strict=local_strict)
        for key, name in dict(slot.routes or {}).items()
    }
    stages = tuple(
        (
            int(start),
            _resolve_operator(name, operator_registry, strict=local_strict),
        )
        for start, name in tuple(slot.stages or ())
    )
    default_operator = _resolve_operator(slot.default_operator, operator_registry, strict=False)
    
    return OrchestrationPolicy(
        mode=str(slot.mode or "serial").strip().lower(),
        operators=operators,
        routes=routes,
        stages=stages,
        selector_key=str(slot.selector_key or "strategy_id"),
        index_key=str(slot.index_key or "vns_k"),
        default_operator=default_operator,
        strict=local_strict,
        merge=slot.merge,
        timeout_seconds=slot.timeout_seconds,
        cancel_on_error=bool(slot.cancel_on_error),
    )


def _resolve_operator(name: Optional[str], registry: Mapping[str, Any], *, strict: bool) -> Any:
    """Resolve operator from registry."""
    if name in (None, ""):
        return None
    key = str(name).strip()
    op = registry.get(key)
    if op is None and strict:
        raise KeyError(f"pipeline operator not found: {key!r}")
    return op


def _make_slot_runner(
    orchestrator: PipelineOrchestrator,
    policy: OrchestrationPolicy,
    *,
    method: str,
) -> Callable[[Any, Optional[MutableMapping[str, Any]]], Any]:
    """Create a slot runner function."""
    def _runner(value: Any, context: Optional[MutableMapping[str, Any]] = None) -> Any:
        return orchestrator._run_policy(policy, value, context, method=method, fallback=value)
    return _runner
