"""
BlackBase - Shared foundation for NSGABlack and MLBlack frameworks.

This package provides unified context infrastructure, resource management,
and kernel components shared between optimization and machine learning frameworks.
"""

from __future__ import annotations

from . import context
from . import resources
from . import kernel
from . import project
from . import plugin
from . import contracts
from .types import Feedback, PopulationSnapshot, TrainerResult, UnknownState


__all__ = [
    "context",
    "resources",
    "kernel",
    "project",
    "plugin",
    "contracts",
    "Feedback",
    "UnknownState",
    "PopulationSnapshot",
    "TrainerResult",
]
