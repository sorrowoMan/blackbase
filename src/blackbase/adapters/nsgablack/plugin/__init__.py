"""NSGABlack plugin backward-compatibility adapter.

Re-exports the canonical ``blackbase.plugin`` symbols so that legacy
``nsgablack.plugins.base`` import paths continue to work.

.. deprecated::
    ``nsgablack.plugins.base`` is deprecated.  Use ``blackbase.plugin`` instead.
"""

from __future__ import annotations

import warnings

from blackbase.plugin import Plugin, PluginManager, report_soft_error  # noqa: F401

warnings.warn(
    "nsgablack.plugins.base is deprecated. Use blackbase.plugin instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "Plugin",
    "PluginManager",
    "report_soft_error",
]
