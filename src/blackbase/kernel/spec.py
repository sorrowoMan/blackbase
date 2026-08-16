"""
Pipeline slot and specification definitions.

Provides the core data structures for defining pipeline slots,
execution modes, and routing policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


# ============================================================================
# Slot Method Mapping
# ============================================================================

_SLOT_METHOD_MAP = {
    "init": "initialize",
    "initializer": "initialize",
    "mutate": "mutate",
    "repair": "repair",
    "encode": "encode",
    "decode": "decode",
    "transform": "transform",
    "codec": "encode",
    "head": "decode",
}

_PIPELINE_SLOT_NAMES = {"init", "initializer", "mutate", "repair", "encode", "decode"}


# ============================================================================
# Pipeline Slot Spec
# ============================================================================

@dataclass(frozen=True)
class PipelineSlotSpec:
    """
    Specification for a single pipeline slot.
    
    Defines how a slot should be executed, including the operators to use,
    execution mode, and routing configuration.
    """
    
    slot: str                           # Slot name: init/mutate/repair/encode/decode
    operators: Sequence[str] = ()       # Operator names to execute
    mode: str = "serial"               # Execution mode: serial/parallel/router
    method: Optional[str] = None       # Override method name
    routes: Mapping[str, str] = field(default_factory=dict)  # Route mapping for router mode
    stages: Sequence[tuple[int, str]] = ()  # Generation threshold -> operator name
    selector_key: str = "strategy_id"  # Context key for route selection
    index_key: str = "vns_k"           # Context key for index-based selection
    default_operator: Optional[str] = None
    strict: Optional[bool] = None       # Strict mode for operator resolution
    merge: Optional[str] = None        # Merge strategy for parallel results
    timeout_seconds: Optional[float] = None  # Whole-slot wall-clock deadline
    cancel_on_error: bool = True       # Stop pending branches after strict failure
    
    @classmethod
    def from_value(cls, value: Mapping[str, Any] | "PipelineSlotSpec") -> "PipelineSlotSpec":
        """Create PipelineSlotSpec from mapping or existing spec."""
        if isinstance(value, PipelineSlotSpec):
            return value
        payload = dict(value or {})
        raw_stages = payload.get("stages", ()) or ()
        stage_items = raw_stages.items() if isinstance(raw_stages, Mapping) else raw_stages
        return cls(
            slot=str(payload.get("slot", payload.get("name", ""))),
            operators=tuple(str(name) for name in payload.get("operators", ()) if str(name).strip()),
            mode=str(payload.get("mode", "serial") or "serial"),
            method=str(payload.get("method")) if payload.get("method") not in (None, "") else None,
            routes={str(k): str(v) for k, v in dict(payload.get("routes", {}) or {}).items()},
            stages=tuple((int(start), str(name)) for start, name in stage_items),
            selector_key=str(payload.get("selector_key", "strategy_id") or "strategy_id"),
            index_key=str(payload.get("index_key", "vns_k") or "vns_k"),
            default_operator=(
                str(payload.get("default_operator")).strip()
                if payload.get("default_operator", None) not in (None, "")
                else None
            ),
            strict=payload.get("strict"),
            merge=str(payload.get("merge")) if payload.get("merge") not in (None, "") else None,
            timeout_seconds=(
                float(payload.get("timeout_seconds"))
                if payload.get("timeout_seconds") not in (None, "")
                else None
            ),
            cancel_on_error=bool(payload.get("cancel_on_error", True)),
        )
    
    def as_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "slot": str(self.slot),
            "operators": list(self.operators),
            "mode": str(self.mode),
            "method": self.method,
            "routes": dict(self.routes),
            "stages": [[int(start), str(name)] for start, name in self.stages],
            "selector_key": str(self.selector_key),
            "index_key": str(self.index_key),
            "default_operator": self.default_operator,
            "strict": self.strict,
            "merge": self.merge,
            "timeout_seconds": self.timeout_seconds,
            "cancel_on_error": bool(self.cancel_on_error),
        }


# ============================================================================
# Pipeline Spec
# ============================================================================

@dataclass(frozen=True)
class PipelineSpec:
    """
    Complete pipeline specification.
    
    Defines all slots in a pipeline along with global parameters.
    """
    
    key: str = "default"
    slots: Sequence[PipelineSlotSpec | Mapping[str, Any]] = ()
    params: Mapping[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_value(cls, value: Mapping[str, Any] | "PipelineSpec" | None) -> "PipelineSpec":
        """Create PipelineSpec from mapping or existing spec."""
        if isinstance(value, PipelineSpec):
            return value
        payload = dict(value or {})
        return cls(
            key=str(payload.get("key", "default") or "default"),
            slots=tuple(PipelineSlotSpec.from_value(item) for item in payload.get("slots", ())),
            params=dict(payload.get("params", {}) or {}),
        )
    
    def slot_specs(self) -> tuple[PipelineSlotSpec, ...]:
        """Get all slot specs as a tuple."""
        return tuple(PipelineSlotSpec.from_value(item) for item in self.slots)
    
    def as_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": str(self.key),
            "slots": [spec.as_dict() if isinstance(spec, PipelineSlotSpec) else dict(spec) for spec in self.slots],
            "params": dict(self.params),
        }


# ============================================================================
# Helper Functions
# ============================================================================

def normalize_slot_name(value: str) -> str:
    """Normalize slot name to canonical form."""
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def get_method_for_slot(slot: str) -> str:
    """Get the default method name for a slot."""
    return _SLOT_METHOD_MAP.get(slot, "transform")


def is_pipeline_slot(slot: str) -> bool:
    """Check if a slot name is a known pipeline slot."""
    return normalize_slot_name(slot) in _PIPELINE_SLOT_NAMES
