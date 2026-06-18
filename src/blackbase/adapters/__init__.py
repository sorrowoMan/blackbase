"""
Adapters Package.

Provides backward compatibility adapters for NSGABlack and MLBlack.
"""

from __future__ import annotations

from . import nsgablack
from . import mlblack


__all__ = [
    "nsgablack",
    "mlblack",
]
