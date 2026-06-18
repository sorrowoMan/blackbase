"""
Legacy representation module compatibility layer.

Provides import paths for the old nsgablack.representation module structure,
redirecting to blackbase implementations.
"""

from __future__ import annotations

import warnings

# Issue deprecation warning
warnings.warn(
    "nsgablack.representation is deprecated. "
    "Use blackbase.kernel instead. "
    "Migration guide: See blackbase/MIGRATION.md",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the main compatibility layer
from blackbase.adapters.nsgablack.kernel import *  # noqa: F401, F403


__all__ = [
    # Spec
    "PipelineSlotSpec",
    "PipelineSpec",
    "normalize_slot_name",
    "get_method_for_slot",
    "is_pipeline_slot",
    
    # Orchestrator
    "OrchestrationPolicy",
    "PipelineOrchestrator",
    "PipelineKernelBuild",
    "build_pipeline_kernel",
    
    # Legacy wrappers
    "RepresentationPipeline",
]
