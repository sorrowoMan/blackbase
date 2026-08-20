"""
BlackBase - Shared foundation for NSGABlack and MLBlack frameworks.

This package provides unified context infrastructure, resource management,
and kernel components shared between optimization and machine learning frameworks.
"""

from __future__ import annotations

__version__ = "0.3.6"

from . import context
from . import resources
from . import kernel
from . import project
from . import plugin
from . import contracts
from . import evaluation
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
    CandidateBatch,
    Feedback,
    PopulationSnapshot,
    SolveQuality,
    SolverResult,
    TrainerResult,
    UnknownState,
    decode_shared_value,
    encode_shared_value,
)
from .state_ref import StateRef
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
    "evaluation",
    "CallCandidate",
    "BoundCallOutcome",
    "invoke_bound_once",
    "invoke_bound_once_with_outcome",
    "SHARED_TYPE_SCHEMA_VERSION",
    "SOLVE_STATUSES",
    "FEASIBILITY_STATUSES",
    "Feedback",
    "CandidateBatch",
    "StateRef",
    "encode_shared_value",
    "decode_shared_value",
    "UnknownState",
    "PopulationSnapshot",
    "TrainerResult",
    "SolveQuality",
    "SolverResult",
    "__version__",
]
