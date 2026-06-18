"""
NSGABlack Resources Compatibility Layer.

This module provides backward compatibility for NSGABlack's resources modules,
wrapping blackbase implementations while maintaining the original API.
"""

from __future__ import annotations

import warnings
from typing import Any

# Re-export from blackbase with deprecation warnings


def _warn_deprecated(module: str, removal_version: str = "1.0.0") -> None:
    """Issue deprecation warning for NSGABlack-specific resources modules."""
    warnings.warn(
        f"nsgablack.core.resources.{module} is deprecated. "
        f"Use blackbase.resources instead. "
        f"This will be removed in {removal_version}.",
        DeprecationWarning,
        stacklevel=3,
    )


# Import and re-export from blackbase
from blackbase.resources import (
    DataRef,
    InMemoryLeaseStore,
    ResourceAllocator,
    ResourceLease,
    ResourceOffer,
    ResourcePolicy,
    ResourceRequest,
    ResourceRequirement,
    WorkerDescriptor,
    TaskEnvelope,
    TaskResult,
    ScheduledTask,
    ResourceContext,
    ResourceEvent,
    ResourceAudit,
    coerce_resource_context,
    detect_total_memory_mb,
    detect_cuda_devices,
    detect_local_resource_offer,
    build_local_worker_descriptor,
    PoolScheduler,
    PoolTask,
    PoolResult,
    PoolTaskResult,
)


__all__ = [
    # Core model
    "DataRef",
    "InMemoryLeaseStore",
    "ResourceAllocator",
    "ResourceLease",
    "ResourceOffer",
    "ResourcePolicy",
    "ResourceRequest",
    "ResourceRequirement",
    "WorkerDescriptor",
    "TaskEnvelope",
    "TaskResult",
    "ScheduledTask",
    
    # MLBlack extensions
    "ResourceContext",
    "ResourceEvent",
    "ResourceAudit",
    "coerce_resource_context",
    
    # Probe utilities
    "detect_total_memory_mb",
    "detect_cuda_devices",
    "detect_local_resource_offer",
    "build_local_worker_descriptor",
    "PoolScheduler",
    "PoolTask",
    "PoolResult",
    "PoolTaskResult",
]


# Backward compatibility aliases for NSGABlack
# These match the original nsgablack naming conventions


def InMemoryWorkerRegistry(*args, **kwargs):
    """Deprecated: Use blackbase.resources directly."""
    _warn_deprecated("worker_registry")
    from blackbase.resources.model import WorkerDescriptor
    return _LegacyWorkerRegistry(*args, **kwargs)


def InMemoryResourceScheduler(*args, **kwargs):
    """Deprecated: Use blackbase.resources directly."""
    _warn_deprecated("scheduler")
    return _LegacyResourceScheduler(*args, **kwargs)


class _LegacyWorkerRegistry:
    """Legacy worker registry wrapper."""
    
    def __init__(self, workers=()):
        from blackbase.resources import WorkerDescriptor
        self._workers = {}
        for worker in workers:
            if isinstance(worker, WorkerDescriptor):
                self._workers[worker.worker_id] = worker
            else:
                self._workers[worker.get("worker_id", "")] = WorkerDescriptor.from_dict(worker)
    
    def register(self, worker):
        from blackbase.resources import WorkerDescriptor
        item = worker if isinstance(worker, WorkerDescriptor) else WorkerDescriptor.from_dict(worker)
        self._workers[item.worker_id] = item
        return item
    
    def unregister(self, worker_id: str) -> None:
        self._workers.pop(str(worker_id), None)
    
    def get(self, worker_id: str):
        return self._workers.get(str(worker_id))
    
    def list_workers(self, *, status=None):
        workers = list(self._workers.values())
        if status is None:
            return tuple(workers)
        return tuple(w for w in workers if str(w.status).lower() == str(status).lower())


class _LegacyResourceScheduler:
    """Small compatibility scheduler for old nsgablack imports."""

    def __init__(self, worker_registry=None):
        self._registry = worker_registry if worker_registry is not None else _LegacyWorkerRegistry()
        self._scheduled = []

    def schedule(self, task_envelope):
        workers = self._registry.list_workers(status="available")
        if not workers:
            workers = self._registry.list_workers(status="online")
        if not workers:
            return None
        worker_id = workers[0].worker_id
        self._scheduled.append((worker_id, task_envelope))
        return worker_id

    def complete(self, worker_id, task_result):
        return None

    def get_scheduled_tasks(self):
        return list(self._scheduled)
