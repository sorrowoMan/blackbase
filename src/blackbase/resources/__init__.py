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
)
from .grant_pool import ResourceGrantPool, ResourceSubgrant, ResourceSubgrantError
from .transport import (
    ClaimedTask,
    InMemoryTaskTransport,
    RedisTaskTransport,
    SQLiteTaskTransport,
    TaskLeaseError,
    TaskRecord,
    TaskTransport,
    TaskTransportError,
)
from .task_runtime import (
    InMemoryTaskRuntimeBackend,
    RedisTaskRuntimeBackend,
    SQLiteTaskRuntimeBackend,
    TaskRuntimeBackend,
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
from .budget_settlement import (
    BudgetSettlementRecord,
    SQLiteBudgetSettlementJournal,
    minimal_budget_authority_ref,
)
from .artifacts import (
    ARTIFACT_BINDING_SCHEMA_VERSION,
    ARTIFACT_AUTHORITY_SCHEMA_VERSION,
    ARTIFACT_PUBLICATION_RECEIPT_SCHEMA_VERSION,
    ArtifactBinding,
    ArtifactAuthority,
    ArtifactFenceError,
    ArtifactPublicationReceipt,
    ArtifactPublicationError,
    ArtifactPublisher,
    ArtifactSerializer,
    ArtifactSerializerRegistry,
    ArtifactStore,
    FilesystemArtifactPublicationLedger,
    FilesystemArtifactStore,
)
from .control import (
    CancellationHeartbeat,
    CancellationRef,
    CancellationRequested,
    CancellationState,
    CancellationToken,
    CaseDeadlineExceeded,
    InMemoryCancellationStore,
    MAX_CANCELLATION_LINEAGE_DEPTH,
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
    "ResourceGrantPool",
    "ResourceSubgrant",
    "ResourceSubgrantError",
    # transport
    "ClaimedTask",
    "InMemoryTaskTransport",
    "RedisTaskTransport",
    "RedisLeaseStore",
    "SQLiteTaskTransport",
    "SQLiteLeaseStore",
    "TaskLeaseError",
    "TaskRecord",
    "TaskTransport",
    "TaskTransportError",
    "TaskRuntimeBackend",
    "InMemoryTaskRuntimeBackend",
    "SQLiteTaskRuntimeBackend",
    "RedisTaskRuntimeBackend",
    # shared run budgets
    "BudgetAccount",
    "BudgetClaim",
    "BudgetHandle",
    "BudgetSettlementRecord",
    "SQLiteBudgetSettlementJournal",
    "minimal_budget_authority_ref",
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
    "ARTIFACT_BINDING_SCHEMA_VERSION",
    "ARTIFACT_AUTHORITY_SCHEMA_VERSION",
    "ARTIFACT_PUBLICATION_RECEIPT_SCHEMA_VERSION",
    "ArtifactBinding",
    "ArtifactAuthority",
    "ArtifactFenceError",
    "ArtifactPublicationReceipt",
    "ArtifactPublicationError",
    "ArtifactPublisher",
    "ArtifactSerializer",
    "ArtifactSerializerRegistry",
    "ArtifactStore",
    "FilesystemArtifactPublicationLedger",
    "FilesystemArtifactStore",
    # deadline and cooperative cancellation
    "CancellationHeartbeat",
    "CancellationRef",
    "CancellationRequested",
    "CancellationState",
    "CancellationToken",
    "CaseDeadlineExceeded",
    "InMemoryCancellationStore",
    "MAX_CANCELLATION_LINEAGE_DEPTH",
    "RedisCancellationStore",
    "SQLiteCancellationStore",
    "TerminationPolicy",
    "build_cancellation_store",
]
