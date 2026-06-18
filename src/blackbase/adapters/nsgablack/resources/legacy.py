"""
Legacy resources module compatibility layer.

Provides import paths for the old nsgablack.core.resources module structure,
redirecting to blackbase implementations.
"""

from __future__ import annotations

import warnings

# Issue deprecation warning
warnings.warn(
    "nsgablack.core.resources is deprecated. "
    "Use blackbase.resources instead. "
    "Migration guide: See blackbase/MIGRATION.md",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the main compatibility layer
from blackbase.adapters.nsgablack.resources import *  # noqa: F401, F403


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
    
    # Legacy wrappers
    "InMemoryWorkerRegistry",
    "InMemoryResourceScheduler",
    "PoolScheduler",
    "PoolTask",
    "PoolResult",
    "PoolTaskResult",
]
