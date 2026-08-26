"""
Context helpers (canonical keys + minimal evaluation schema + lifecycle + replay).

Recommended imports:
- `from blackbase.context import context_keys as CK`
- `from blackbase.context import build_minimal_context`
"""

from __future__ import annotations

from .context_keys import (
    CONTEXT_KEY_SET,
    METRIC_FALLBACKS,
    METRIC_KEYS,
    normalize_context_key,
    normalize_context_keys,
    register_context_keys,
    unknown_context_keys,
    validate_context_keys,
)
from .context_contracts import (
    ContextContract,
    collect_solver_contracts,
    detect_context_conflicts,
    get_component_contract,
    validate_context_contracts,
)
from .context_field_governance import (
    CONTEXT_FIELD_SCHEMA_NAME,
    CONTEXT_FIELD_SCHEMA_VERSION,
    context_field_schema_dict,
    is_canonical_context_key,
    schema_meta,
)
from .context_schema import (
    CATEGORY_CACHE,
    CATEGORY_DERIVED,
    CATEGORY_EVENT,
    CATEGORY_INPUT,
    CATEGORY_OUTPUT,
    CATEGORY_RUNTIME,
    ContextField,
    ContextSchema,
    MinimalEvaluationContext,
    RUNTIME_CONTEXT_SCHEMA,
    build_minimal_context,
    get_context_lifecycle,
    is_replayable_context,
    strip_context_for_replay,
    validate_context,
    validate_minimal_context,
)
from .context_store import (
    ContextStore,
    InMemoryContextStore,
    RedisContextStore,
    create_context_store,
)
from .context_events import (
    CONTEXT_EVENT_KINDS,
    ContextEvent,
    apply_context_event,
    record_context_event,
    replay_context,
)
from .snapshot_store import (
    GENERIC_SNAPSHOT_MARKER,
    GENERIC_SNAPSHOT_SCHEMA,
    GENERIC_SNAPSHOT_VALUE,
    SnapshotHandle,
    SnapshotRecord,
    SnapshotStore,
    FileSnapshotStore,
    InMemorySnapshotStore,
    RedisSnapshotStore,
    create_snapshot_store,
    make_snapshot_key,
    snapshot_content_digest,
    unwrap_snapshot_payload,
    wrap_snapshot_payload,
)
from .value_isolation import detach_context_value
from .redis_codec import (
    REDIS_VALUE_ENVELOPE,
    SUPPORTED_REDIS_SERIALIZERS,
    RedisValueCodec,
    RedisValueCodecError,
)
from .storage_config import StateStoreConfig
from .runtime_projection import (
    RUNTIME_PROJECTION_AUDIT_MAX_BYTES,
    RUNTIME_PROJECTION_COMPONENT_MAX_BYTES,
    RUNTIME_PROJECTION_ERROR_TYPE_MAX_BYTES,
    RUNTIME_PROJECTION_MAX_COMPONENTS,
    RUNTIME_PROJECTION_MAX_ISSUE_SAMPLES,
    RUNTIME_PROJECTION_MESSAGE_MAX_BYTES,
    RUNTIME_PROJECTION_REASON_MAX_BYTES,
    RUNTIME_PROJECTION_SCHEMA,
    RUNTIME_PROJECTION_STATUSES,
    RuntimeContextProjection,
    RuntimeProjectionAggregation,
    RuntimeProjectionComponent,
    RuntimeProjectionIssue,
    RuntimeProjectionIssueAccumulator,
    aggregate_runtime_projections,
)


__all__ = [
    # context_keys
    "context_keys",
    "CONTEXT_KEY_SET",
    "METRIC_FALLBACKS",
    "METRIC_KEYS",
    "normalize_context_key",
    "normalize_context_keys",
    "register_context_keys",
    "unknown_context_keys",
    "validate_context_keys",
    
    # context_contracts
    "ContextContract",
    "collect_solver_contracts",
    "detect_context_conflicts",
    "get_component_contract",
    "validate_context_contracts",
    
    # context_field_governance
    "CONTEXT_FIELD_SCHEMA_NAME",
    "CONTEXT_FIELD_SCHEMA_VERSION",
    "context_field_schema_dict",
    "is_canonical_context_key",
    "schema_meta",
    
    # context_schema
    "CATEGORY_CACHE",
    "CATEGORY_DERIVED",
    "CATEGORY_EVENT",
    "CATEGORY_INPUT",
    "CATEGORY_OUTPUT",
    "CATEGORY_RUNTIME",
    "ContextField",
    "ContextSchema",
    "MinimalEvaluationContext",
    "RUNTIME_CONTEXT_SCHEMA",
    "build_minimal_context",
    "get_context_lifecycle",
    "is_replayable_context",
    "strip_context_for_replay",
    "validate_context",
    "validate_minimal_context",
    
    # context_store
    "ContextStore",
    "InMemoryContextStore",
    "RedisContextStore",
    "create_context_store",
    
    # context_events
    "CONTEXT_EVENT_KINDS",
    "ContextEvent",
    "apply_context_event",
    "record_context_event",
    "replay_context",
    
    # snapshot_store
    "GENERIC_SNAPSHOT_MARKER",
    "GENERIC_SNAPSHOT_SCHEMA",
    "GENERIC_SNAPSHOT_VALUE",
    "SnapshotHandle",
    "SnapshotRecord",
    "SnapshotStore",
    "FileSnapshotStore",
    "InMemorySnapshotStore",
    "RedisSnapshotStore",
    "create_snapshot_store",
    "make_snapshot_key",
    "snapshot_content_digest",
    "unwrap_snapshot_payload",
    "wrap_snapshot_payload",
    # value isolation
    "detach_context_value",
    "REDIS_VALUE_ENVELOPE",
    "SUPPORTED_REDIS_SERIALIZERS",
    "RedisValueCodec",
    "RedisValueCodecError",
    "StateStoreConfig",
    # runtime projection
    "RUNTIME_PROJECTION_SCHEMA",
    "RUNTIME_PROJECTION_STATUSES",
    "RUNTIME_PROJECTION_MAX_COMPONENTS",
    "RUNTIME_PROJECTION_MAX_ISSUE_SAMPLES",
    "RUNTIME_PROJECTION_AUDIT_MAX_BYTES",
    "RUNTIME_PROJECTION_COMPONENT_MAX_BYTES",
    "RUNTIME_PROJECTION_REASON_MAX_BYTES",
    "RUNTIME_PROJECTION_ERROR_TYPE_MAX_BYTES",
    "RUNTIME_PROJECTION_MESSAGE_MAX_BYTES",
    "RuntimeProjectionIssue",
    "RuntimeProjectionIssueAccumulator",
    "RuntimeContextProjection",
    "RuntimeProjectionComponent",
    "RuntimeProjectionAggregation",
    "aggregate_runtime_projections",
]
