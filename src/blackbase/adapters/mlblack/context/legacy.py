"""
Legacy context module compatibility layer for MLBlack.

Provides import paths for the old mlblack.core module structure,
redirecting to blackbase implementations.
"""

from __future__ import annotations

import warnings

# Issue deprecation warning
warnings.warn(
    "mlblack.core is deprecated. "
    "Use blackbase.context instead. "
    "Migration guide: See blackbase/MIGRATION.md",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the main compatibility layer
from blackbase.adapters.mlblack.context import *  # noqa: F401, F403
from blackbase.context import (  # noqa: F401
    context_keys,
    METRIC_KEYS,
    METRIC_FALLBACKS,
    CONTEXT_KEY_SET,
    CONTEXT_KEY_ALIASES,
)


__all__ = [
    # From context_keys
    "context_keys",
    "METRIC_KEYS",
    "METRIC_FALLBACKS",
    "CONTEXT_KEY_SET",
    "CONTEXT_KEY_ALIASES",
    "normalize_context_key",
    "normalize_context_keys",
    "register_context_keys",
    "unknown_context_keys",
    "validate_context_keys",
    
    # From context_contracts
    "ContextContract",
    "detect_context_conflicts",
    "get_component_contract",
    
    # From context_store
    "ContextStore",
    "InMemoryContextStore",
    "create_context_store",
    
    # From snapshot_store
    "SnapshotStore",
    "InMemorySnapshotStore",
    "SnapshotHandle",
    "SnapshotRecord",
    "create_snapshot_store",
    
    # From context_schema
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
