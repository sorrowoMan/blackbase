"""
NSGABlack Adapter Package.

Provides backward compatibility and migration support for NSGABlack.
"""

from __future__ import annotations

from importlib import import_module


_MODULES = ("context", "resources", "kernel", "plugin", "contracts")


def __getattr__(name: str):
    """Load compatibility modules only when callers explicitly request them."""

    if name not in _MODULES:
        raise AttributeError(name)
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


__all__ = [
    *_MODULES,
]
