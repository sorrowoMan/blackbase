"""
MLBlack Resources Compatibility Layer.

This module provides backward compatibility for MLBlack's resources modules,
wrapping blackbase implementations while maintaining the original API.
"""

from __future__ import annotations

import warnings
from typing import Any

# Re-export from blackbase with deprecation warnings


def _warn_deprecated(module: str, removal_version: str = "1.0.0") -> None:
    """Issue deprecation warning for MLBlack-specific resources modules."""
    warnings.warn(
        f"mlblack.core.resources.{module} is deprecated. "
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
