"""
Pipeline orchestration policy and executor.

Provides the orchestration logic for running pipeline operators
in different execution modes (serial, parallel, router).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence


# ============================================================================
# Orchestration Policy
# ============================================================================

@dataclass(frozen=True)
class OrchestrationPolicy:
    """
    Policy for executing operators within a pipeline slot.
    
    Defines how operators should be executed, including mode selection,
    routing configuration, and merge strategy.
    """
    
    mode: str = "serial"                              # Execution mode
    operators: Sequence[Any] = ()                     # Resolved operators
    routes: Mapping[str, Any] = field(default_factory=dict)  # Route key -> operator
    selector_key: str = "strategy_id"                # Context key for selection
    index_key: str = "vns_k"                         # Context key for index
    default_operator: Any = None
    strict: bool = False                             # Strict operator resolution
    merge: Optional[str] = None                     # Merge strategy
    
    def select_operator(self, context: Optional[Mapping[str, Any]]) -> Any:
        """Select an operator based on context."""
        if self.mode == "router" and self.routes:
            ctx = context or {}
            selector = ctx.get(self.selector_key)
            if selector is not None:
                key = str(selector)
                if key in self.routes:
                    return self.routes[key]
            # Try index-based selection
            index = ctx.get(self.index_key)
            if index is not None:
                route_keys = list(self.routes.keys())
                try:
                    idx = int(index) % len(route_keys)
                    return self.routes[route_keys[idx]]
                except (ValueError, TypeError, IndexError):
                    pass
            # Fallback to default
            if self.default_operator is not None:
                return self.default_operator
        return None
    
    def as_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": str(self.mode),
            "operators": [op.__name__ if hasattr(op, "__name__") else str(op) for op in self.operators],
            "routes": {str(k): v.__name__ if hasattr(v, "__name__") else str(v) for k, v in self.routes.items()},
            "selector_key": str(self.selector_key),
            "index_key": str(self.index_key),
            "has_default": self.default_operator is not None,
            "strict": bool(self.strict),
            "merge": self.merge,
        }


# ============================================================================
# Pipeline Orchestrator
# ============================================================================

class PipelineOrchestrator:
    """
    Executes pipeline operators according to orchestration policies.
    
    Supports serial, parallel, and router execution modes.
    """
    
    def __init__(self, *, strict: bool = False) -> None:
        self.strict = bool(strict)
    
    def _run_serial(
        self,
        operators: Sequence[Any],
        value: Any,
        context: Optional[MutableMapping[str, Any]],
        method: str,
    ) -> Any:
        """Run operators in serial mode."""
        result = value
        for op in operators:
            if op is None:
                continue
            result = self._call_operator(op, result, context, method)
        return result
    
    def _run_parallel(
        self,
        operators: Sequence[Any],
        value: Any,
        context: Optional[MutableMapping[str, Any]],
        method: str,
    ) -> tuple[Any, ...]:
        """Run operators in parallel mode."""
        import concurrent.futures
        
        def run_op(op: Any) -> Any:
            if op is None:
                return value
            return self._call_operator(op, value, context, method)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = tuple(executor.map(run_op, operators))
        return results
    
    def _run_router(
        self,
        policy: OrchestrationPolicy,
        value: Any,
        context: Optional[MutableMapping[str, Any]],
        method: str,
        fallback: Any,
    ) -> Any:
        """Run operators in router mode (select one based on context)."""
        operator = policy.select_operator(context)
        if operator is None:
            return fallback
        return self._call_operator(operator, value, context, method)
    
    def _call_operator(
        self,
        operator: Any,
        value: Any,
        context: Optional[MutableMapping[str, Any]],
        method: str,
    ) -> Any:
        """Call an operator with the given value and context."""
        # First support component objects whose semantics live in a slot
        # method such as initialize/mutate/repair rather than __call__.
        method_fn = getattr(operator, method, None)
        if callable(method_fn):
            try:
                return method_fn(value, context)
            except TypeError:
                return method_fn(value)

        # Then support plain callables.
        if callable(operator):
            try:
                return operator(value, context)
            except TypeError:
                try:
                    return operator(value)
                except TypeError:
                    return operator()
        return value
    
    def _run_policy(
        self,
        policy: OrchestrationPolicy,
        value: Any,
        context: Optional[MutableMapping[str, Any]],
        method: str,
        fallback: Any,
    ) -> Any:
        """Run operators according to policy."""
        mode = str(policy.mode or "serial").lower()
        
        if mode == "serial":
            return self._run_serial(policy.operators, value, context, method)
        
        if mode == "parallel":
            return self._run_parallel(policy.operators, value, context, method)
        
        if mode == "router":
            return self._run_router(policy, value, context, method, fallback)
        
        # Default to serial
        return self._run_serial(policy.operators, value, context, method)


# ============================================================================
# Pipeline Kernel Build
# ============================================================================

@dataclass
class PipelineKernelBuild:
    """
    Result of building a pipeline kernel.
    
    Contains the representation pipeline, slot runners, policies, and registry.
    """
    
    slot_runners: Mapping[str, Callable[[Any, Optional[MutableMapping[str, Any]]], Any]] = field(default_factory=dict)
    slot_policies: Mapping[str, OrchestrationPolicy] = field(default_factory=dict)
    operator_registry: Mapping[str, Any] = field(default_factory=dict)
    
    def run_slot(
        self,
        slot: str,
        value: Any,
        context: Optional[MutableMapping[str, Any]] = None,
    ) -> Any:
        """Run a specific slot with the given value."""
        key = normalize_slot_name(slot)
        runner = self.slot_runners.get(key)
        if runner is None:
            return value
        return runner(value, context)
    
    def as_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "slots": sorted(self.slot_runners.keys()),
            "slot_policies": {key: policy.as_dict() for key, policy in self.slot_policies.items()},
        }


# Import normalize_slot_name
from .spec import normalize_slot_name


def build_pipeline_kernel(
    spec: Mapping[str, Any] | None,
    *,
    operator_registry: Mapping[str, Any],
    strict: bool = True,
) -> PipelineKernelBuild:
    """
    Build a pipeline kernel from specification.
    
    Args:
        spec: Pipeline specification
        operator_registry: Registry of available operators
        strict: Strict operator resolution mode
    
    Returns:
        PipelineKernelBuild with runners and policies
    """
    from .spec import PipelineSpec, PipelineSlotSpec, get_method_for_slot, normalize_slot_name, is_pipeline_slot
    
    parsed = PipelineSpec.from_value(spec)
    orchestrator = PipelineOrchestrator(strict=bool(strict))
    slot_runners: dict[str, Callable[[Any, Optional[MutableMapping[str, Any]]], Any]] = {}
    slot_policies: dict[str, OrchestrationPolicy] = {}
    
    for slot_spec in parsed.slot_specs():
        slot_name = normalize_slot_name(slot_spec.slot)
        if not slot_name:
            continue
        
        method = str(slot_spec.method or get_method_for_slot(slot_name)).strip()
        policy = _build_policy_from_slot(slot_spec, operator_registry=operator_registry, strict=bool(strict))
        slot_policies[slot_name] = policy
        slot_runners[slot_name] = _make_slot_runner(orchestrator, policy, method=method)
    
    return PipelineKernelBuild(
        slot_runners=slot_runners,
        slot_policies=slot_policies,
        operator_registry=dict(operator_registry),
    )


def _build_policy_from_slot(
    slot: PipelineSlotSpec,
    *,
    operator_registry: Mapping[str, Any],
    strict: bool,
) -> OrchestrationPolicy:
    """Build orchestration policy from slot spec."""
    local_strict = bool(strict if slot.strict is None else slot.strict)
    
    operators = tuple(_resolve_operator(name, operator_registry, strict=local_strict) for name in slot.operators)
    routes = {
        key: _resolve_operator(name, operator_registry, strict=local_strict)
        for key, name in dict(slot.routes or {}).items()
    }
    default_operator = _resolve_operator(slot.default_operator, operator_registry, strict=False)
    
    return OrchestrationPolicy(
        mode=str(slot.mode or "serial").strip().lower(),
        operators=operators,
        routes=routes,
        selector_key=str(slot.selector_key or "strategy_id"),
        index_key=str(slot.index_key or "vns_k"),
        default_operator=default_operator,
        strict=local_strict,
        merge=slot.merge,
    )


def _resolve_operator(name: Optional[str], registry: Mapping[str, Any], *, strict: bool) -> Any:
    """Resolve operator from registry."""
    if name in (None, ""):
        return None
    key = str(name).strip()
    op = registry.get(key)
    if op is None and strict:
        raise KeyError(f"pipeline operator not found: {key!r}")
    return op


def _make_slot_runner(
    orchestrator: PipelineOrchestrator,
    policy: OrchestrationPolicy,
    *,
    method: str,
) -> Callable[[Any, Optional[MutableMapping[str, Any]]], Any]:
    """Create a slot runner function."""
    def _runner(value: Any, context: Optional[MutableMapping[str, Any]] = None) -> Any:
        return orchestrator._run_policy(policy, value, context, method=method, fallback=value)
    return _runner
