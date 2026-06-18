"""
Legacy context module compatibility layer.

Provides import paths for the old nsgablack.utils.context module structure,
redirecting to blackbase implementations.
"""

from __future__ import annotations

import warnings

# Issue deprecation warning
warnings.warn(
    "nsgablack.utils.context is deprecated. "
    "Use blackbase.context instead. "
    "Migration guide: See blackbase/MIGRATION.md",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the main compatibility layer
from blackbase.adapters.nsgablack.context import *  # noqa: F401, F403
from blackbase.context import (  # noqa: F401
    context_keys,
    ContextEvent,
    apply_context_event,
    record_context_event,
    replay_context,
    CONTEXT_FIELD_SCHEMA_NAME,
    CONTEXT_FIELD_SCHEMA_VERSION,
    context_field_schema_dict,
    is_canonical_context_key,
    schema_meta,
    CATEGORY_CACHE,
    CATEGORY_DERIVED,
    CATEGORY_EVENT,
    CATEGORY_INPUT,
    CATEGORY_OUTPUT,
    CATEGORY_RUNTIME,
)


__all__ = [
    # From context_keys
    "context_keys",
    "normalize_context_key",
    "normalize_context_keys",
    "register_context_keys",
    "unknown_context_keys",
    "validate_context_keys",
    
    # From context_contracts
    "ContextContract",
    "collect_solver_contracts",
    "detect_context_conflicts",
    "get_component_contract",
    "validate_context_contracts",
    
    # From context_store
    "ContextStore",
    "InMemoryContextStore",
    "RedisContextStore",
    "create_context_store",
    
    # From snapshot_store
    "SnapshotStore",
    "InMemorySnapshotStore",
    "RedisSnapshotStore",
    "FileSnapshotStore",
    "create_snapshot_store",
    "SnapshotHandle",
    "SnapshotRecord",
    
    # From context_schema
    "ContextField",
    "ContextSchema",
    "MinimalEvaluationContext",
    "RUNTIME_CONTEXT_SCHEMA",
    "build_minimal_context",
    "validate_context",
    "validate_minimal_context",
    "get_context_lifecycle",
    "is_replayable_context",
    "strip_context_for_replay",
    
    # Additional exports
    "ContextEvent",
    "apply_context_event",
    "record_context_event",
    "replay_context",
    "CONTEXT_FIELD_SCHEMA_NAME",
    "CONTEXT_FIELD_SCHEMA_VERSION",
    "context_field_schema_dict",
    "is_canonical_context_key",
    "schema_meta",
    "CATEGORY_CACHE",
    "CATEGORY_DERIVED",
    "CATEGORY_EVENT",
    "CATEGORY_INPUT",
    "CATEGORY_OUTPUT",
    "CATEGORY_RUNTIME",
]
