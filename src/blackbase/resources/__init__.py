"""
Resources layer exports.
"""

from __future__ import annotations

from .model import (
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
)
from .context import (
    ResourceContext,
    ResourceEvent,
    ResourceAudit,
    coerce_resource_context,
)
from .probe import (
    detect_total_memory_mb,
    detect_cuda_devices,
    detect_local_resource_offer,
    build_local_worker_descriptor,
)
from .pool import (
    PoolScheduler,
    PoolTask,
    PoolResult,
    PoolTaskResult,
)


__all__ = [
    # model
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
    
    # context
    "ResourceContext",
    "ResourceEvent",
    "ResourceAudit",
    "coerce_resource_context",
    
    # probe
    "detect_total_memory_mb",
    "detect_cuda_devices",
    "detect_local_resource_offer",
    "build_local_worker_descriptor",

    # pool
    "PoolScheduler",
    "PoolTask",
    "PoolResult",
    "PoolTaskResult",
]
