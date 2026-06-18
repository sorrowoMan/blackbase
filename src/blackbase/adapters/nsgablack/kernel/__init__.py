"""
NSGABlack Kernel Compatibility Layer.

This module provides backward compatibility for NSGABlack's representation
and pipeline modules, wrapping blackbase implementations while maintaining
the original API.
"""

from __future__ import annotations

import warnings
from typing import Any

# Re-export from blackbase with deprecation warnings


def _warn_deprecated(module: str, removal_version: str = "1.0.0") -> None:
    """Issue deprecation warning for NSGABlack-specific kernel modules."""
    warnings.warn(
        f"nsgablack.representation.{module} is deprecated. "
        f"Use blackbase.kernel instead. "
        f"This will be removed in {removal_version}.",
        DeprecationWarning,
        stacklevel=3,
    )


# Import and re-export from blackbase
from blackbase.kernel import (
    PipelineSlotSpec,
    PipelineSpec,
    OrchestrationPolicy,
    PipelineOrchestrator,
    PipelineKernelBuild,
    build_pipeline_kernel,
    normalize_slot_name,
    get_method_for_slot,
    is_pipeline_slot,
)


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
]


# Backward compatibility aliases for NSGABlack representation layer


class RepresentationPipeline:
    """
    Legacy representation pipeline wrapper.
    
    Wraps the blackbase kernel implementation while maintaining
    the original RepresentationPipeline API.
    """
    
    def __init__(
        self,
        initializer=None,
        mutator=None,
        repair=None,
        encoder=None,
        decoder=None,
        **kwargs,
    ):
        from blackbase.kernel import PipelineSpec, build_pipeline_kernel
        from typing import Optional, MutableMapping
        
        self._initializer = initializer
        self._mutator = mutator
        self._repair = repair
        self._encoder = encoder
        self._decoder = decoder
        self._extra = kwargs
        
        # Build kernel
        slots = []
        if initializer is not None:
            slots.append({"slot": "initializer", "operators": ["_init"]})
        if mutator is not None:
            slots.append({"slot": "mutate", "operators": ["_mut"]})
        if repair is not None:
            slots.append({"slot": "repair", "operators": ["_rep"]})
        if encoder is not None:
            slots.append({"slot": "encode", "operators": ["_enc"]})
        if decoder is not None:
            slots.append({"slot": "decode", "operators": ["_dec"]})
        
        spec = {"key": "representation", "slots": slots} if slots else None
        registry = {
            "_init": initializer,
            "_mut": mutator,
            "_rep": repair,
            "_enc": encoder,
            "_dec": decoder,
        }
        # Remove None values
        registry = {k: v for k, v in registry.items() if v is not None}
        
        self._kernel = build_pipeline_kernel(spec, operator_registry=registry) if registry else None
    
    def initialize(self, problem, context: Optional[MutableMapping] = None):
        if self._kernel:
            return self._kernel.run_slot("initializer", problem, context)
        return problem
    
    def mutate(self, value, context: Optional[MutableMapping] = None):
        if self._kernel:
            return self._kernel.run_slot("mutate", value, context)
        return value
    
    def repair(self, value, context: Optional[MutableMapping] = None):
        if self._kernel:
            return self._kernel.run_slot("repair", value, context)
        return value
    
    def encode(self, value, context: Optional[MutableMapping] = None):
        if self._kernel:
            return self._kernel.run_slot("encode", value, context)
        return value
    
    def decode(self, value, context: Optional[MutableMapping] = None):
        if self._kernel:
            return self._kernel.run_slot("decode", value, context)
        return value
    
    def transform(self, value, context: Optional[MutableMapping] = None):
        """Generic transform through all applicable operators."""
        result = value
        if self._initializer:
            result = self.initialize(result, context)
        if self._encoder:
            result = self.encode(result, context)
        return result
