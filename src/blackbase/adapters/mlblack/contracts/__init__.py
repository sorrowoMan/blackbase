"""
MLBlack Contracts Compatibility Layer.

This module provides backward compatibility for MLBlack's contracts modules,
wrapping blackbase implementations while maintaining the original API.
"""

from __future__ import annotations

import warnings

# Re-export from blackbase
from blackbase.contracts import ComponentContract, ContractMixin, combine_contracts


def _warn_deprecated(module: str = "contracts", removal_version: str = "1.0.0") -> None:
    """Issue deprecation warning for MLBlack-specific contracts modules."""
    warnings.warn(
        f"mlblack.core.{module} is deprecated. "
        f"Use blackbase.contracts instead. "
        f"This will be removed in {removal_version}.",
        DeprecationWarning,
        stacklevel=3,
    )


__all__ = [
    # Re-exported from blackbase.contracts
    "ComponentContract",
    "ContractMixin",
    "combine_contracts",
]
