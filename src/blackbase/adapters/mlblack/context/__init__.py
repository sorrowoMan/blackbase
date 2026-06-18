"""
MLBlack Context Compatibility Layer.

This module provides backward compatibility for MLBlack's context modules,
wrapping blackbase implementations while maintaining the original API.
"""

from __future__ import annotations

import warnings
from typing import Any

# Re-export from blackbase with deprecation warnings


def _warn_deprecated(module: str, removal_version: str = "1.0.0") -> None:
    """Issue deprecation warning for MLBlack-specific context modules."""
    warnings.warn(
        f"mlblack.core.{module} is deprecated. "
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
    InMemorySnapshotStore,
    MinimalEvaluationContext,
    SnapshotHandle,
    SnapshotRecord,
    SnapshotStore,
    build_minimal_context,
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
    "detect_context_conflicts",
    "get_component_contract",
    
    # Context store
    "ContextStore",
    "InMemoryContextStore",
    "create_context_store",
    
    # Snapshot store
    "SnapshotStore",
    "InMemorySnapshotStore",
    "SnapshotHandle",
    "SnapshotRecord",
    "create_snapshot_store",
    
    # Context schema
    "ContextField",
    "ContextSchema",
    "MinimalEvaluationContext",
    "build_minimal_context",
    "validate_context",
    "validate_minimal_context",
    "get_context_lifecycle",
    "is_replayable_context",
    "strip_context_for_replay",
]


# Backward compatibility aliases for MLBlack
InMemoryContextStore = ContextStore
