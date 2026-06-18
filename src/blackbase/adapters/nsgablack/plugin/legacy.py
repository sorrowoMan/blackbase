"""Legacy path shim for ``nsgablack.plugins.base``.

.. deprecated::
    This legacy path is deprecated.  Use ``blackbase.plugin`` instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "nsgablack.plugins.base legacy path is deprecated. Use blackbase.plugin instead.",
    DeprecationWarning,
    stacklevel=2,
)

from blackbase.adapters.nsgablack.plugin import *  # noqa: F401,F403
