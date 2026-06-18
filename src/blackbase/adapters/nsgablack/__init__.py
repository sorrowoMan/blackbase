"""
NSGABlack Adapter Package.

Provides backward compatibility and migration support for NSGABlack.
"""

from __future__ import annotations

from . import context
from . import resources
from . import kernel
from . import plugin
from . import contracts


__all__ = [
    "context",
    "resources",
    "kernel",
    "plugin",
    "contracts",
]
