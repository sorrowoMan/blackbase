"""
Context field governance and validation utilities.

Provides validation for context keys and schema metadata.
"""

from __future__ import annotations

from typing import Dict

from .context_keys import CONTEXT_KEY_SET


CONTEXT_FIELD_SCHEMA_NAME = "blackbase.context_field.v1"
CONTEXT_FIELD_SCHEMA_VERSION = "1.0.0"


def context_field_schema_dict() -> Dict[str, str]:
    """Get schema metadata as a dictionary."""
    return {
        "name": CONTEXT_FIELD_SCHEMA_NAME,
        "version": CONTEXT_FIELD_SCHEMA_VERSION,
    }


def schema_meta() -> Dict[str, str]:
    """Get schema metadata."""
    return context_field_schema_dict()


def is_canonical_context_key(key: str) -> bool:
    """Check if a key is a canonical context key."""
    return str(key).strip().lower() in CONTEXT_KEY_SET