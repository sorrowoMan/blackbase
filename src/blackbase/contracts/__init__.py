"""
Component contract system for declaring composition contracts.

This module provides ComponentContract (serializable bridge) and ContractMixin
(component contract declaration via class attributes), migrated from mlblack
into the shared blackbase foundation.

Recommended imports:
- `from blackbase.contracts import ComponentContract, ContractMixin, combine_contracts`
"""

from __future__ import annotations

from .component_contract import (
    ComponentContract,
    ContractMixin,
    combine_contracts,
)


__all__ = [
    "ComponentContract",
    "ContractMixin",
    "combine_contracts",
]
