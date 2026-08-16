"""
Resources layer exports.
"""

from __future__ import annotations

from .model import (
    DataRef,
    InMemoryLeaseStore,
    InMemoryResourceScheduler,
    InMemoryWorkerRegistry,
    ResourceAllocator,
    ResourceBudgetError,
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
from .transport import (
    ClaimedTask,
    RedisTaskTransport,
    SQLiteTaskTransport,
    TaskLeaseError,
    TaskRecord,
    TaskTransport,
    TaskTransportError,
)
from .lease_store import RedisLeaseStore, SQLiteLeaseStore
from .budget import (
    BudgetReservation,
    BudgetSnapshot,
    RedisBudgetAuthority,
    SQLiteBudgetAuthority,
    SharedBudgetConfigurationError,
    SharedBudgetError,
    SharedBudgetExceeded,
    SharedBudgetFenceError,
    build_budget_authority_from_resource_context,
)
from .budget_account import BudgetAccount, BudgetClaim, BudgetHandle
from .artifacts import (
    ARTIFACT_AUTHORITY_SCHEMA_VERSION,
    ArtifactAuthority,
    ArtifactFenceError,
    ArtifactPublicationError,
    ArtifactPublisher,
    ArtifactSerializer,
    ArtifactSerializerRegistry,
    ArtifactStore,
    FilesystemArtifactStore,
)
from .control import (
    CancellationRef,
    CancellationRequested,
    CancellationState,
    CancellationToken,
    CaseDeadlineExceeded,
    InMemoryCancellationStore,
    RedisCancellationStore,
    SQLiteCancellationStore,
    TerminationPolicy,
    build_cancellation_store,
)


__all__ = [
    # model
    "DataRef",
    "InMemoryLeaseStore",
    "InMemoryResourceScheduler",
    "InMemoryWorkerRegistry",
    "ResourceAllocator",
    "ResourceBudgetError",
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
    # transport
    "ClaimedTask",
    "RedisTaskTransport",
    "RedisLeaseStore",
    "SQLiteTaskTransport",
    "SQLiteLeaseStore",
    "TaskLeaseError",
    "TaskRecord",
    "TaskTransport",
    "TaskTransportError",
    # shared run budgets
    "BudgetAccount",
    "BudgetClaim",
    "BudgetHandle",
    "BudgetReservation",
    "BudgetSnapshot",
    "RedisBudgetAuthority",
    "SQLiteBudgetAuthority",
    "SharedBudgetConfigurationError",
    "SharedBudgetError",
    "SharedBudgetExceeded",
    "SharedBudgetFenceError",
    "build_budget_authority_from_resource_context",
    # durable artifact publication
    "ARTIFACT_AUTHORITY_SCHEMA_VERSION",
    "ArtifactAuthority",
    "ArtifactFenceError",
    "ArtifactPublicationError",
    "ArtifactPublisher",
    "ArtifactSerializer",
    "ArtifactSerializerRegistry",
    "ArtifactStore",
    "FilesystemArtifactStore",
    # deadline and cooperative cancellation
    "CancellationRef",
    "CancellationRequested",
    "CancellationState",
    "CancellationToken",
    "CaseDeadlineExceeded",
    "InMemoryCancellationStore",
    "RedisCancellationStore",
    "SQLiteCancellationStore",
    "TerminationPolicy",
    "build_cancellation_store",
]
