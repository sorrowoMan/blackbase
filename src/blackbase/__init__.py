"""
BlackBase - Shared foundation for NSGABlack and MLBlack frameworks.

This package provides unified context infrastructure, resource management,
and kernel components shared between optimization and machine learning frameworks.
"""

from __future__ import annotations

__version__ = "0.3.0"

from . import context
from . import resources
from . import kernel
from . import project
from . import plugin
from . import contracts
from .call_binding import (
    CallCandidate,
    BoundCallOutcome,
    invoke_bound_once,
    invoke_bound_once_with_outcome,
)
from .types import (
    FEASIBILITY_STATUSES,
    SHARED_TYPE_SCHEMA_VERSION,
    SOLVE_STATUSES,
    Feedback,
    PopulationSnapshot,
    SolveQuality,
    SolverResult,
    TrainerResult,
    UnknownState,
)
from .catalog import (
    Catalog,
    CatalogEntry,
    load_catalog_paths,
    load_catalog_toml,
    render_catalog_toml,
    write_catalog_shards,
)


__all__ = [
    "Catalog",
    "CatalogEntry",
    "load_catalog_paths",
    "load_catalog_toml",
    "render_catalog_toml",
    "write_catalog_shards",
    "context",
    "resources",
    "kernel",
    "project",
    "plugin",
    "contracts",
    "CallCandidate",
    "BoundCallOutcome",
    "invoke_bound_once",
    "invoke_bound_once_with_outcome",
    "SHARED_TYPE_SCHEMA_VERSION",
    "SOLVE_STATUSES",
    "FEASIBILITY_STATUSES",
    "Feedback",
    "UnknownState",
    "PopulationSnapshot",
    "TrainerResult",
    "SolveQuality",
    "SolverResult",
    "__version__",
]
