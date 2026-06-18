"""Legacy path shim for ``nsgablack`` core contracts.

.. deprecated::
    This legacy path is deprecated.  Use ``blackbase.contracts`` instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "nsgablack core contracts legacy path is deprecated. Use blackbase.contracts instead.",
    DeprecationWarning,
    stacklevel=2,
)

from blackbase.adapters.nsgablack.contracts import *  # noqa: F401,F403
