"""
NSGABlack Context Compatibility Layer.

This module provides backward compatibility for NSGABlack's context modules,
wrapping blackbase implementations while maintaining the original API.
"""

from __future__ import annotations

import warnings
from typing import Any

# Re-export from blackbase with deprecation warnings
# These will be the primary imports after migration


def _warn_deprecated(module: str, removal_version: str = "1.0.0") -> None:
    """Issue deprecation warning for NSGABlack-specific context modules."""
    warnings.warn(
        f"nsgablack.core.state.{module} is deprecated. "
        f"Use blackbase.context instead. "
        f"This will be removed in {removal_version}.",
        DeprecationWarning,
        stacklevel=3,
    )


# Import and re-export from blackbase
from blackbase.context import (
    ContextContract,
    ContextField,
    ContextSchema,
    ContextStore,
    FileSnapshotStore,
    InMemorySnapshotStore,
    MinimalEvaluationContext,
    RUNTIME_CONTEXT_SCHEMA,
    RedisContextStore,
    RedisSnapshotStore,
    SnapshotHandle,
    SnapshotRecord,
    SnapshotStore,
    build_minimal_context,
    collect_solver_contracts,
    create_context_store,
    create_snapshot_store,
    detect_context_conflicts,
    get_component_contract,
    get_context_lifecycle,
    is_replayable_context,
    normalize_context_key,
    normalize_context_keys,
    register_context_keys,
    strip_context_for_replay,
    unknown_context_keys,
    validate_context,
    validate_context_contracts,
    validate_context_keys,
    validate_minimal_context,
)


__all__ = [
    # Context keys
    "normalize_context_key",
    "normalize_context_keys",
    "register_context_keys",
    "unknown_context_keys",
    "validate_context_keys",
    
    # Context contracts
    "ContextContract",
    "collect_solver_contracts",
    "detect_context_conflicts",
    "get_component_contract",
    "validate_context_contracts",
    
    # Context store
    "ContextStore",
    "InMemoryContextStore",
    "RedisContextStore",
    "create_context_store",
    
    # Snapshot store
    "SnapshotStore",
    "InMemorySnapshotStore",
    "RedisSnapshotStore",
    "FileSnapshotStore",
    "create_snapshot_store",
    "SnapshotHandle",
    "SnapshotRecord",
    
    # Context schema
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
]


# Backward compatibility aliases for NSGABlack
InMemoryContextStore = ContextStore
