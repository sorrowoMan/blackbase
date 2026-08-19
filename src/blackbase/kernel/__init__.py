"""
Kernel layer exports.
"""

from .spec import (
    PipelineSlotSpec,
    PipelineSpec,
    normalize_slot_name,
    get_method_for_slot,
    is_pipeline_slot,
)
from .orchestrator import (
    OrchestrationPolicy,
    ParallelBranchFailure,
    PipelineCancellationError,
    PipelineLateWriteRejected,
    PipelineParallelError,
    PipelineOrchestrator,
    PipelineKernelBuild,
    build_pipeline_kernel,
)


__all__ = [
    # spec
    "PipelineSlotSpec",
    "PipelineSpec",
    "normalize_slot_name",
    "get_method_for_slot",
    "is_pipeline_slot",
    
    # orchestrator
    "OrchestrationPolicy",
    "ParallelBranchFailure",
    "PipelineCancellationError",
    "PipelineLateWriteRejected",
    "PipelineParallelError",
    "PipelineOrchestrator",
    "PipelineKernelBuild",
    "build_pipeline_kernel",
]
