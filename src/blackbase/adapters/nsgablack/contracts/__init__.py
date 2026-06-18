"""NSGABlack contracts backward-compatibility adapter.

Re-exports the canonical ``blackbase.contracts`` symbols so that legacy
``nsgablack`` core contracts import paths continue to work.

.. deprecated::
    ``nsgablack`` core contracts is deprecated.  Use ``blackbase.contracts`` instead.
"""

from __future__ import annotations

import warnings

from blackbase.contracts import ComponentContract, ContractMixin, combine_contracts  # noqa: F401

warnings.warn(
    "nsgablack core contracts is deprecated. Use blackbase.contracts instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ComponentContract",
    "ContractMixin",
    "combine_contracts",
]
