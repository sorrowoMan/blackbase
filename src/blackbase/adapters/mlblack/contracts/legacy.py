"""
Legacy contracts module compatibility layer for MLBlack.

Provides import paths for the old mlblack.core.contracts module structure,
redirecting to blackbase implementations.
"""

from __future__ import annotations

import warnings

# Issue deprecation warning
warnings.warn(
    "mlblack.core.contracts legacy path is deprecated. "
    "Use blackbase.contracts instead. "
    "Migration guide: See blackbase/MIGRATION.md",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the main compatibility layer
from blackbase.adapters.mlblack.contracts import *  # noqa: F401, F403


__all__ = [
    "ComponentContract",
    "ContractMixin",
    "combine_contracts",
]
