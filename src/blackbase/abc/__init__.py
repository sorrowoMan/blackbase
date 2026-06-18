"""Abstract base classes for framework-customizable components.

These ABCs define the unified contract that nsgablack and mlblack share.
Each ABC includes all hooks with sensible defaults; subclasses only
implement what they need.

Design principle:
- One base class, all hooks, implement as needed.
- Same functionality has one name (no duplicate names for same concept).
- Core methods are @abstractmethod; everything else is optional.
- Framework-specific helpers stay in framework subclasses.
"""

from __future__ import annotations

from ._adapter import AdapterBase
from ._representation import RepresentationBase
from ._problem import ProblemBase
from ._bias import BiasBase

__all__ = [
    "AdapterBase",
    "RepresentationBase",
    "ProblemBase",
    "BiasBase",
]
