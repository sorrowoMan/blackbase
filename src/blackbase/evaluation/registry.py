"""Provider registration and resource-safe binding for evaluation requests."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from threading import RLock
from typing import Any, Mapping, Protocol, cast, runtime_checkable

from blackbase.resources import ResourceContext, ResourceRequirement, coerce_resource_context
from blackbase.state_ref import StateRef
from blackbase.types import encode_shared_value

from .model import (
    EvaluationBinding,
    EvaluationProviderSpec,
    EvaluationRequest,
    EvaluationResult,
    StateTransitionMethodSpec,
)
from .state_transition import (
    StateMaterializationRequest,
    StateMaterializationResult,
    StateReleaseRequest,
    StateReleaseResult,
    StateTransitionRequest,
    StateTransitionResult,
)
from .conformance import (
    CopyOnWriteConformanceReport,
    verify_copy_on_write_predecessors,
)


EVALUATION_REJECTION_REASON_MAX_LENGTH = 2048


class EvaluationProviderError(RuntimeError):
    """Base error for evaluation provider registration and execution."""


class EvaluationProviderUnavailable(EvaluationProviderError):
    """No registered provider can satisfy an evaluation request and L0 grant."""

    def __init__(self, message: str, *, rejections: Mapping[str, str] | None = None) -> None:
        super().__init__(message)
        self.rejections = dict(rejections or {})


class EvaluationProviderContractError(EvaluationProviderError):
    """A provider violated the shared request/result contract."""


@runtime_checkable
class EvaluationProvider(Protocol):
    """Provider interface implemented by ML, simulation, or domain packages."""

    @property
    def spec(self) -> EvaluationProviderSpec:
        ...

    def evaluate(
        self,
        request: EvaluationRequest,
        binding: EvaluationBinding,
    ) -> EvaluationResult:
        ...


@runtime_checkable
class StateTransitionProvider(EvaluationProvider, Protocol):
    """Evaluation Provider that can execute declared state compute kernels."""

    def transition(
        self,
        request: StateTransitionRequest,
        binding: EvaluationBinding,
    ) -> StateTransitionResult:
        ...


@runtime_checkable
class StateMaterializationProvider(EvaluationProvider, Protocol):
    """Evaluation Provider that explicitly exports its live state."""

    def materialize(
        self,
        request: StateMaterializationRequest,
        binding: EvaluationBinding,
    ) -> StateMaterializationResult:
        ...


@runtime_checkable
class StateReleaseProvider(EvaluationProvider, Protocol):
    """Evaluation Provider that tears down owned state scopes."""

    def release(
        self,
        request: StateReleaseRequest,
        binding: EvaluationBinding,
    ) -> StateReleaseResult:
        ...


@dataclass(frozen=True)
class BoundEvaluationProvider:
    """One provider bound to one request under one immutable grant snapshot."""

    provider: EvaluationProvider
    binding: EvaluationBinding
    binding_digest: str = ""

    def __post_init__(self) -> None:
        digest = str(self.binding_digest or _protocol_digest(self.binding.as_dict()))
        object.__setattr__(self, "binding_digest", digest)

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        if request.request_id != self.binding.request_id:
            raise EvaluationProviderContractError(
                "bound provider cannot evaluate a different request"
            )
        if self.binding.request_digest and (
            _protocol_digest(request.as_dict()) != self.binding.request_digest
        ):
            raise EvaluationProviderContractError(
                "bound provider cannot evaluate a request whose payload changed"
            )
        if _protocol_digest(self.binding.as_dict()) != self.binding_digest:
            raise EvaluationProviderContractError(
                "evaluation binding changed after provider selection"
            )
        # Execute exactly once.  A TypeError raised inside provider code is a
        # provider failure, never a signal to guess and retry another signature.
        result = self.provider.evaluate(request, self.binding)
        if self.binding.request_digest and (
            _protocol_digest(request.as_dict()) != self.binding.request_digest
        ):
            raise EvaluationProviderContractError(
                "provider mutated the bound evaluation request"
            )
        if _protocol_digest(self.binding.as_dict()) != self.binding_digest:
            raise EvaluationProviderContractError(
                "provider mutated the authoritative evaluation binding"
            )
        if not isinstance(result, EvaluationResult):
            raise EvaluationProviderContractError(
                f"provider '{self.binding.provider_id}' returned "
                f"{type(result).__name__}; expected EvaluationResult"
            )
        if result.request_id != request.request_id:
            raise EvaluationProviderContractError(
                f"provider '{self.binding.provider_id}' returned a result for "
                "another request"
            )
        if len(result.feedback) != len(request.states):
            raise EvaluationProviderContractError(
                f"provider '{self.binding.provider_id}' returned "
                f"{len(result.feedback)} feedback items for "
                f"{len(request.states)} states"
            )
        if result.binding is None:
            return result.with_binding(self.binding)
        if result.binding != self.binding:
            raise EvaluationProviderContractError(
                f"provider '{self.binding.provider_id}' returned conflicting "
                "binding evidence"
            )
        return result


@dataclass(frozen=True)
class BoundStateTransitionProvider:
    """One owner Provider bound to one version-fenced transition request."""

    provider: StateTransitionProvider
    binding: EvaluationBinding
    method_spec: StateTransitionMethodSpec
    binding_digest: str = ""

    def __post_init__(self) -> None:
        digest = str(self.binding_digest or _protocol_digest(self.binding.as_dict()))
        object.__setattr__(self, "binding_digest", digest)

    def execute(self, request: StateTransitionRequest) -> StateTransitionResult:
        if request.request_id != self.binding.request_id:
            raise EvaluationProviderContractError(
                "bound transition provider cannot execute a different request"
            )
        if self.binding.request_digest and (
            _protocol_digest(request.as_dict()) != self.binding.request_digest
        ):
            raise EvaluationProviderContractError(
                "bound transition provider cannot execute a request whose payload changed"
            )
        if _protocol_digest(self.binding.as_dict()) != self.binding_digest:
            raise EvaluationProviderContractError(
                "state transition binding changed after provider selection"
            )
        result = self.provider.transition(request, self.binding)
        if self.binding.request_digest and (
            _protocol_digest(request.as_dict()) != self.binding.request_digest
        ):
            raise EvaluationProviderContractError(
                "provider mutated the bound state transition request"
            )
        if _protocol_digest(self.binding.as_dict()) != self.binding_digest:
            raise EvaluationProviderContractError(
                "provider mutated the authoritative state transition binding"
            )
        if not isinstance(result, StateTransitionResult):
            raise EvaluationProviderContractError(
                f"provider '{self.binding.provider_id}' returned "
                f"{type(result).__name__}; expected StateTransitionResult"
            )
        if result.request_id != request.request_id:
            raise EvaluationProviderContractError(
                f"provider '{self.binding.provider_id}' returned a transition "
                "result for another request"
            )
        if result.method_id != request.method_id:
            raise EvaluationProviderContractError(
                f"provider '{self.binding.provider_id}' changed transition method "
                f"from '{request.method_id}' to '{result.method_id}'"
            )
        _validate_transition_successor(request, result)
        result_contract_error = _transition_result_contract_mismatch(
            self.method_spec,
            result,
        )
        if result_contract_error:
            raise EvaluationProviderContractError(result_contract_error)
        missing_result_slots = sorted(
            set(self.method_spec.result_slots).difference(result.slot_refs)
        )
        if missing_result_slots:
            raise EvaluationProviderContractError(
                f"provider '{self.binding.provider_id}' omitted required result "
                f"slots: {missing_result_slots}"
            )
        if result.binding is None:
            return result.with_binding(self.binding)
        if result.binding != self.binding:
            raise EvaluationProviderContractError(
                f"provider '{self.binding.provider_id}' returned conflicting "
                "transition binding evidence"
            )
        return result


@dataclass(frozen=True)
class BoundStateMaterializationProvider:
    """One owner Provider bound to one immutable materialization request."""

    provider: StateMaterializationProvider
    binding: EvaluationBinding
    binding_digest: str = ""

    def __post_init__(self) -> None:
        digest = str(self.binding_digest or _protocol_digest(self.binding.as_dict()))
        object.__setattr__(self, "binding_digest", digest)

    def execute(
        self,
        request: StateMaterializationRequest,
    ) -> StateMaterializationResult:
        if request.request_id != self.binding.request_id:
            raise EvaluationProviderContractError(
                "bound materialization provider cannot execute a different request"
            )
        if self.binding.request_digest and (
            _protocol_digest(request.as_dict()) != self.binding.request_digest
        ):
            raise EvaluationProviderContractError(
                "bound materialization provider cannot execute a request whose payload changed"
            )
        if _protocol_digest(self.binding.as_dict()) != self.binding_digest:
            raise EvaluationProviderContractError(
                "state materialization binding changed after provider selection"
            )
        result = self.provider.materialize(request, self.binding)
        if self.binding.request_digest and (
            _protocol_digest(request.as_dict()) != self.binding.request_digest
        ):
            raise EvaluationProviderContractError(
                "provider mutated the bound state materialization request"
            )
        if _protocol_digest(self.binding.as_dict()) != self.binding_digest:
            raise EvaluationProviderContractError(
                "provider mutated the authoritative materialization binding"
            )
        if not isinstance(result, StateMaterializationResult):
            raise EvaluationProviderContractError(
                f"provider '{self.binding.provider_id}' returned "
                f"{type(result).__name__}; expected StateMaterializationResult"
            )
        if result.request_id != request.request_id:
            raise EvaluationProviderContractError(
                "provider returned a materialization result for another request"
            )
        if result.state_ref != request.state_ref:
            raise EvaluationProviderContractError(
                "state materialization changed the authoritative StateRef"
            )
        if result.target != request.target:
            raise EvaluationProviderContractError(
                "state materialization changed the requested target"
            )
        if result.binding is None:
            return result.with_binding(self.binding)
        if result.binding != self.binding:
            raise EvaluationProviderContractError(
                "provider returned conflicting materialization binding evidence"
            )
        return result


@dataclass(frozen=True)
class BoundStateReleaseProvider:
    """One owner Provider bound to one immutable release request."""

    provider: StateReleaseProvider
    binding: EvaluationBinding
    binding_digest: str = ""

    def __post_init__(self) -> None:
        digest = str(self.binding_digest or _protocol_digest(self.binding.as_dict()))
        object.__setattr__(self, "binding_digest", digest)

    def execute(self, request: StateReleaseRequest) -> StateReleaseResult:
        if request.request_id != self.binding.request_id:
            raise EvaluationProviderContractError(
                "bound release provider cannot execute a different request"
            )
        if self.binding.request_digest and (
            _protocol_digest(request.as_dict()) != self.binding.request_digest
        ):
            raise EvaluationProviderContractError(
                "bound release provider cannot execute a changed request"
            )
        result = self.provider.release(request, self.binding)
        if self.binding.request_digest and (
            _protocol_digest(request.as_dict()) != self.binding.request_digest
        ):
            raise EvaluationProviderContractError(
                "provider mutated the bound state release request"
            )
        if _protocol_digest(self.binding.as_dict()) != self.binding_digest:
            raise EvaluationProviderContractError(
                "provider mutated the authoritative release binding"
            )
        if not isinstance(result, StateReleaseResult):
            raise EvaluationProviderContractError(
                f"provider '{self.binding.provider_id}' returned "
                f"{type(result).__name__}; expected StateReleaseResult"
            )
        if result.request_id != request.request_id:
            raise EvaluationProviderContractError(
                "provider returned a release result for another request"
            )
        if result.provider_id != request.provider_id:
            raise EvaluationProviderContractError(
                "provider changed the release owner identity"
            )
        requested_ids = set(request.state_ids)
        if requested_ids:
            resolved_ids = set(result.released_state_ids).union(
                result.not_found_state_ids
            )
            if resolved_ids != requested_ids:
                missing = sorted(requested_ids.difference(resolved_ids))
                unexpected = sorted(resolved_ids.difference(requested_ids))
                raise EvaluationProviderContractError(
                    "targeted state release did not resolve every requested StateRef: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        if result.binding is None:
            return result.with_binding(self.binding)
        if result.binding != self.binding:
            raise EvaluationProviderContractError(
                "provider returned conflicting release binding evidence"
            )
        return result


class EvaluationProviderRegistry:
    """Thread-safe registry that binds providers without allocating resources."""

    def __init__(self) -> None:
        self._providers: dict[str, EvaluationProvider] = {}
        self._specs: dict[str, EvaluationProviderSpec] = {}
        self._copy_on_write_certifications: dict[
            tuple[str, str], CopyOnWriteConformanceReport
        ] = {}
        self._lock = RLock()

    def register(self, provider: EvaluationProvider, *, replace: bool = False) -> None:
        spec = EvaluationProviderSpec.from_dict(_provider_spec(provider).as_dict())
        evaluate = getattr(provider, "evaluate", None)
        if not callable(evaluate):
            raise EvaluationProviderContractError(
                f"provider '{spec.provider_id}' must implement evaluate(request, binding)"
            )
        with self._lock:
            if spec.provider_id in self._providers and not replace:
                raise EvaluationProviderContractError(
                    f"evaluation provider already registered: {spec.provider_id}"
                )
            self._providers[spec.provider_id] = provider
            self._specs[spec.provider_id] = spec
            if replace:
                stale = [
                    key
                    for key in self._copy_on_write_certifications
                    if key[0] == spec.provider_id
                ]
                for key in stale:
                    self._copy_on_write_certifications.pop(key, None)

    def unregister(self, provider_id: str) -> EvaluationProvider | None:
        with self._lock:
            key = str(provider_id)
            self._specs.pop(key, None)
            stale = [
                item
                for item in self._copy_on_write_certifications
                if item[0] == key
            ]
            for item in stale:
                self._copy_on_write_certifications.pop(item, None)
            return self._providers.pop(key, None)

    def get(self, provider_id: str) -> EvaluationProvider:
        with self._lock:
            provider = self._providers.get(str(provider_id))
        if provider is None:
            raise EvaluationProviderUnavailable(
                f"evaluation provider is not registered: {provider_id}"
            )
        return provider

    def specs(self) -> tuple[EvaluationProviderSpec, ...]:
        with self._lock:
            specs = tuple(self._specs.values())
        return tuple(sorted(specs, key=lambda item: item.provider_id))

    def copy_on_write_certifications(
        self,
    ) -> tuple[CopyOnWriteConformanceReport, ...]:
        """Return the immutable first-use certification audit."""

        with self._lock:
            reports = tuple(self._copy_on_write_certifications.values())
        return tuple(
            sorted(reports, key=lambda item: (item.provider_id, item.method_id))
        )

    def _requires_copy_on_write_certification(
        self,
        provider_id: str,
        method_id: str,
    ) -> bool:
        with self._lock:
            return (str(provider_id), str(method_id)) not in (
                self._copy_on_write_certifications
            )

    def _record_copy_on_write_certification(
        self,
        report: CopyOnWriteConformanceReport,
    ) -> None:
        with self._lock:
            self._copy_on_write_certifications.setdefault(
                (report.provider_id, report.method_id),
                report,
            )

    def bind(
        self,
        request: EvaluationRequest,
        resource_context: ResourceContext | Mapping[str, Any] | None,
    ) -> BoundEvaluationProvider:
        """Select one provider inside the supplied authoritative L0 grant."""

        if not isinstance(request, EvaluationRequest):
            raise TypeError("request must be an EvaluationRequest")
        supplied_context = coerce_resource_context(resource_context)
        context = ResourceContext.from_mapping(supplied_context.as_dict())
        with self._lock:
            providers = tuple(
                (self._providers[key], self._specs[key])
                for key in self._providers
            )
        candidates: list[
            tuple[tuple[int, int, int, int, str], EvaluationProvider, EvaluationBinding]
        ] = []
        rejections: dict[str, str] = {}
        for provider, spec in providers:
            outcome = _try_binding(spec, request, context)
            if isinstance(outcome, str):
                rejections[spec.provider_id] = _bounded_rejection_reason(outcome)
                continue
            score, binding = outcome
            candidates.append((score, provider, binding))
        if not candidates:
            bounded = dict(sorted(rejections.items())[:32])
            raise EvaluationProviderUnavailable(
                f"no evaluation provider can satisfy problem='{request.problem_id}', "
                f"mode='{request.mode}' inside the supplied L0 grant",
                rejections=bounded,
            )
        # The final provider id in the score gives deterministic ordering.  The
        # numeric parts are descending; provider id is ascending on ties.
        candidates.sort(
            key=lambda item: (
                -item[0][0],
                -item[0][1],
                -item[0][2],
                -item[0][3],
                item[0][4],
            )
        )
        _, provider, binding = candidates[0]
        return BoundEvaluationProvider(provider=provider, binding=binding)

    def bind_transition(
        self,
        request: StateTransitionRequest,
        resource_context: ResourceContext | Mapping[str, Any] | None,
    ) -> BoundStateTransitionProvider:
        """Bind a transition to its StateRef owner inside the supplied grant."""

        if not isinstance(request, StateTransitionRequest):
            raise TypeError("request must be a StateTransitionRequest")
        owner_id = request.state_ref.provider_id
        supplied_context = coerce_resource_context(resource_context)
        context = ResourceContext.from_mapping(supplied_context.as_dict())
        with self._lock:
            provider = self._providers.get(owner_id)
            spec = self._specs.get(owner_id)
        if provider is None or spec is None:
            raise EvaluationProviderUnavailable(
                f"StateRef owner provider is not registered: {owner_id}"
            )
        transition = getattr(provider, "transition", None)
        if not callable(transition):
            raise EvaluationProviderUnavailable(
                f"provider '{owner_id}' does not implement state transitions",
                rejections={owner_id: "transition(request, binding) is unavailable"},
            )
        outcome = _try_transition_binding(spec, request, context)
        if isinstance(outcome, str):
            raise EvaluationProviderUnavailable(
                f"provider '{owner_id}' cannot execute transition "
                f"'{request.method_id}' inside the supplied L0 grant",
                rejections={owner_id: _bounded_rejection_reason(outcome)},
            )
        binding, method_spec = outcome
        return BoundStateTransitionProvider(
            provider=cast(StateTransitionProvider, provider),
            binding=binding,
            method_spec=method_spec,
        )

    def bind_materialization(
        self,
        request: StateMaterializationRequest,
        resource_context: ResourceContext | Mapping[str, Any] | None,
    ) -> BoundStateMaterializationProvider:
        """Bind an explicit state export to its owner inside the L0 grant."""

        if not isinstance(request, StateMaterializationRequest):
            raise TypeError("request must be a StateMaterializationRequest")
        owner_id = request.state_ref.provider_id
        supplied_context = coerce_resource_context(resource_context)
        context = ResourceContext.from_mapping(supplied_context.as_dict())
        with self._lock:
            provider = self._providers.get(owner_id)
            spec = self._specs.get(owner_id)
        if provider is None or spec is None:
            raise EvaluationProviderUnavailable(
                f"StateRef owner provider is not registered: {owner_id}"
            )
        materialize = getattr(provider, "materialize", None)
        if not callable(materialize):
            raise EvaluationProviderUnavailable(
                f"provider '{owner_id}' does not implement state materialization",
                rejections={owner_id: "materialize(request, binding) is unavailable"},
            )
        outcome = _try_materialization_binding(spec, request, context)
        if isinstance(outcome, str):
            raise EvaluationProviderUnavailable(
                f"provider '{owner_id}' cannot materialize target "
                f"'{request.target}' inside the supplied L0 grant",
                rejections={owner_id: _bounded_rejection_reason(outcome)},
            )
        return BoundStateMaterializationProvider(
            provider=cast(StateMaterializationProvider, provider),
            binding=outcome,
        )

    def bind_release(
        self,
        request: StateReleaseRequest,
        resource_context: ResourceContext | Mapping[str, Any] | None,
    ) -> BoundStateReleaseProvider:
        """Bind scope teardown to the owning Provider inside the L0 grant."""

        if not isinstance(request, StateReleaseRequest):
            raise TypeError("request must be a StateReleaseRequest")
        supplied_context = coerce_resource_context(resource_context)
        context = ResourceContext.from_mapping(supplied_context.as_dict())
        with self._lock:
            provider = self._providers.get(request.provider_id)
            spec = self._specs.get(request.provider_id)
        if provider is None or spec is None:
            raise EvaluationProviderUnavailable(
                f"state owner provider is not registered: {request.provider_id}"
            )
        release = getattr(provider, "release", None)
        if not callable(release):
            raise EvaluationProviderUnavailable(
                f"provider '{request.provider_id}' does not implement state release",
                rejections={request.provider_id: "release(request, binding) is unavailable"},
            )
        outcome = _try_release_binding(spec, request, context)
        if isinstance(outcome, str):
            raise EvaluationProviderUnavailable(
                f"provider '{request.provider_id}' cannot release the requested scope",
                rejections={request.provider_id: _bounded_rejection_reason(outcome)},
            )
        return BoundStateReleaseProvider(
            provider=cast(StateReleaseProvider, provider),
            binding=outcome,
        )


class EvaluationGateway:
    """Small Problem-facing facade for bind-then-evaluate."""

    def __init__(self, registry: EvaluationProviderRegistry) -> None:
        if not isinstance(registry, EvaluationProviderRegistry):
            raise TypeError("registry must be EvaluationProviderRegistry")
        self.registry = registry

    def evaluate(
        self,
        request: EvaluationRequest,
        resource_context: ResourceContext | Mapping[str, Any] | None,
    ) -> EvaluationResult:
        bound = self.registry.bind(request, resource_context)
        return bound.evaluate(request)

    def transition(
        self,
        request: StateTransitionRequest,
        resource_context: ResourceContext | Mapping[str, Any] | None,
    ) -> StateTransitionResult:
        bound = self.registry.bind_transition(request, resource_context)
        result = bound.execute(request)
        provider_id = request.state_ref.provider_id
        if (
            result.status == "applied"
            and bool(request.slot_refs)
            and self.registry._requires_copy_on_write_certification(
                provider_id,
                request.method_id,
            )
        ):
            report = verify_copy_on_write_predecessors(
                self,
                request,
                result,
                bound.binding.resource_context,
            )
            self.registry._record_copy_on_write_certification(report)
        return result

    def materialize(
        self,
        request: StateMaterializationRequest,
        resource_context: ResourceContext | Mapping[str, Any] | None,
    ) -> StateMaterializationResult:
        bound = self.registry.bind_materialization(request, resource_context)
        return bound.execute(request)

    def release(
        self,
        request: StateReleaseRequest,
        resource_context: ResourceContext | Mapping[str, Any] | None,
    ) -> StateReleaseResult:
        bound = self.registry.bind_release(request, resource_context)
        return bound.execute(request)


def _provider_spec(provider: Any) -> EvaluationProviderSpec:
    spec = getattr(provider, "spec", None)
    if not isinstance(spec, EvaluationProviderSpec):
        raise EvaluationProviderContractError(
            "evaluation provider must expose an EvaluationProviderSpec as .spec"
        )
    return spec


def _try_binding(
    spec: EvaluationProviderSpec,
    request: EvaluationRequest,
    context: ResourceContext,
) -> tuple[tuple[int, int, int, int, str], EvaluationBinding] | str:
    normalized_problem_id = str(request.problem_id or "").strip().lower()
    if spec.problem_ids and normalized_problem_id not in set(spec.problem_ids):
        return (
            f"problem '{request.problem_id}' is not supported; "
            f"supported={list(spec.problem_ids)}"
        )
    if request.mode not in spec.modes:
        return f"mode '{request.mode}' is not supported"
    provided_capabilities = set(spec.capabilities)
    required = set(request.required_capabilities)
    missing_required = sorted(required.difference(provided_capabilities))
    if missing_required:
        return f"missing required capabilities: {missing_required}"
    state_error = _state_ref_mismatch(spec, request)
    if state_error:
        return state_error
    resource_error = _resource_mismatch(spec.resource_requirement, context)
    if resource_error:
        return resource_error
    requested_backend = str(context.compute_backend or "auto").strip().lower()
    provider_backend = str(spec.compute_backend or "generic").strip().lower()
    if requested_backend not in {"", "auto", "any", "generic"} and (
        provider_backend not in {"generic", requested_backend}
    ):
        return (
            f"compute backend '{provider_backend}' conflicts with L0 grant "
            f"'{requested_backend}'"
        )
    device = _select_device(spec, context)
    if device is None:
        return (
            f"supported devices {list(spec.supported_devices)} are unavailable "
            "inside the L0 grant"
        )
    preferred = set(request.preferred_capabilities)
    matched_preferred = sorted(preferred.intersection(provided_capabilities))
    missing_preferred = sorted(preferred.difference(provided_capabilities))
    first_preferred_family = (
        _device_family(spec.preferred_devices[0]) if spec.preferred_devices else ""
    )
    device_fallback = bool(
        first_preferred_family
        and _device_family(device) != first_preferred_family
    )
    degraded = bool(missing_preferred or device_fallback)
    matched = sorted(required.union(matched_preferred))
    selected_preference = _device_preference_score(spec, device)
    accelerator_selected = int(_device_family(device) in {"gpu", "mps"})
    score = (
        int(spec.priority),
        len(matched_preferred),
        selected_preference,
        accelerator_selected,
        spec.provider_id,
    )
    binding = EvaluationBinding(
        request_id=request.request_id,
        provider_id=spec.provider_id,
        resource_context=context,
        device=device,
        compute_backend=spec.compute_backend,
        request_digest=_protocol_digest(request.as_dict()),
        matched_capabilities=tuple(matched),
        missing_preferred_capabilities=tuple(missing_preferred),
        degraded=degraded,
        audit={
            "status": "degraded" if degraded else "bound",
            "allocator": "project_l0",
            "allocation_performed": False,
            "provider_priority": int(spec.priority),
            "device_fallback": device_fallback,
            "resource_requirement": spec.resource_requirement.as_dict(),
            "grant_namespace": context.namespace,
        },
    )
    return score, binding


def _try_transition_binding(
    spec: EvaluationProviderSpec,
    request: StateTransitionRequest,
    context: ResourceContext,
) -> tuple[EvaluationBinding, StateTransitionMethodSpec] | str:
    method_spec = spec.transition_method(request.method_id)
    if method_spec is None:
        return (
            f"transition method '{request.method_id}' is not declared; "
            f"available={list(spec.transition_method_ids)}"
        )
    request_error = _transition_request_contract_mismatch(method_spec, request)
    if request_error:
        return request_error
    if spec.state_kinds:
        state_kind = str(request.state_ref.state_kind or "opaque").strip().lower()
        if state_kind not in set(spec.state_kinds):
            return f"StateRef kind '{state_kind}' is not supported"
    resource_error = _resource_mismatch(spec.resource_requirement, context)
    if resource_error:
        return resource_error
    requested_backend = str(context.compute_backend or "auto").strip().lower()
    provider_backend = str(spec.compute_backend or "generic").strip().lower()
    if requested_backend not in {"", "auto", "any", "generic"} and (
        provider_backend not in {"generic", requested_backend}
    ):
        return (
            f"compute backend '{provider_backend}' conflicts with L0 grant "
            f"'{requested_backend}'"
        )
    device = _select_device(
        spec,
        context,
        required_device=request.state_ref.device,
    )
    if device is None:
        return (
            f"supported devices {list(spec.supported_devices)} are unavailable "
            "inside the L0 grant"
        )
    state_device_family = _device_family(request.state_ref.device)
    if _device_family(device) != state_device_family:
        return (
            f"StateRef device '{request.state_ref.device}' is incompatible with "
            f"bound device '{device}'"
        )
    first_preferred_family = (
        _device_family(spec.preferred_devices[0]) if spec.preferred_devices else ""
    )
    device_fallback = bool(
        first_preferred_family
        and _device_family(device) != first_preferred_family
    )
    binding = EvaluationBinding(
        request_id=request.request_id,
        provider_id=spec.provider_id,
        resource_context=context,
        device=device,
        compute_backend=spec.compute_backend,
        request_digest=_protocol_digest(request.as_dict()),
        matched_capabilities=(f"state.transition:{request.method_id}",),
        degraded=device_fallback,
        audit={
            "status": "degraded" if device_fallback else "bound",
            "allocator": "project_l0",
            "allocation_performed": False,
            "operation": "state_transition",
            "method_id": request.method_id,
            "expected_state_version": request.state_ref.version,
            "device_fallback": device_fallback,
            "resource_requirement": spec.resource_requirement.as_dict(),
            "grant_namespace": context.namespace,
        },
    )
    return binding, method_spec


def _try_materialization_binding(
    spec: EvaluationProviderSpec,
    request: StateMaterializationRequest,
    context: ResourceContext,
) -> EvaluationBinding | str:
    if request.target not in set(spec.materialization_targets):
        return (
            f"materialization target '{request.target}' is not declared; "
            f"available={list(spec.materialization_targets)}"
        )
    if spec.state_kinds:
        state_kind = str(request.state_ref.state_kind or "opaque").strip().lower()
        if state_kind not in set(spec.state_kinds):
            return f"StateRef kind '{state_kind}' is not supported"
    resource_error = _resource_mismatch(spec.resource_requirement, context)
    if resource_error:
        return resource_error
    requested_backend = str(context.compute_backend or "auto").strip().lower()
    provider_backend = str(spec.compute_backend or "generic").strip().lower()
    if requested_backend not in {"", "auto", "any", "generic"} and (
        provider_backend not in {"generic", requested_backend}
    ):
        return (
            f"compute backend '{provider_backend}' conflicts with L0 grant "
            f"'{requested_backend}'"
        )
    device = _select_device(
        spec,
        context,
        required_device=request.state_ref.device,
    )
    if device is None or _device_family(device) != _device_family(request.state_ref.device):
        return (
            f"StateRef device '{request.state_ref.device}' is unavailable "
            "inside the L0 grant"
        )
    return EvaluationBinding(
        request_id=request.request_id,
        provider_id=spec.provider_id,
        resource_context=context,
        device=device,
        compute_backend=spec.compute_backend,
        request_digest=_protocol_digest(request.as_dict()),
        matched_capabilities=(f"state.materialize:{request.target}",),
        audit={
            "status": "bound",
            "allocator": "project_l0",
            "allocation_performed": False,
            "operation": "state_materialization",
            "target": request.target,
            "expected_state_version": request.state_ref.version,
            "grant_namespace": context.namespace,
        },
    )


def _try_release_binding(
    spec: EvaluationProviderSpec,
    request: StateReleaseRequest,
    context: ResourceContext,
) -> EvaluationBinding | str:
    resource_error = _resource_mismatch(spec.resource_requirement, context)
    if resource_error:
        return resource_error
    if not context.namespace:
        return "state release requires a non-empty authoritative L0 namespace"
    if request.scope_id != context.namespace:
        return (
            f"release scope '{request.scope_id}' conflicts with L0 namespace "
            f"'{context.namespace}'"
        )
    return EvaluationBinding(
        request_id=request.request_id,
        provider_id=spec.provider_id,
        resource_context=context,
        device=str(context.device or "cpu"),
        compute_backend=spec.compute_backend,
        request_digest=_protocol_digest(request.as_dict()),
        matched_capabilities=("state.release",),
        audit={
            "status": "bound",
            "allocator": "project_l0",
            "allocation_performed": False,
            "operation": "state_release",
            "scope_id": request.scope_id,
            "trajectory_id": request.trajectory_id,
            "grant_namespace": context.namespace,
        },
    )


def _transition_request_contract_mismatch(
    method_spec: StateTransitionMethodSpec,
    request: StateTransitionRequest,
) -> str | None:
    groups = (
        (
            "operands",
            set(request.operands),
            set(method_spec.required_operands),
            set(method_spec.optional_operands),
            method_spec.allow_additional_operands,
        ),
        (
            "parameters",
            set(request.parameters),
            set(method_spec.required_parameters),
            set(method_spec.optional_parameters),
            method_spec.allow_additional_parameters,
        ),
        (
            "slots",
            set(request.slot_refs),
            set(method_spec.required_slots),
            set(method_spec.optional_slots),
            method_spec.allow_additional_slots,
        ),
    )
    for label, actual, required, optional, allow_additional in groups:
        missing = sorted(required.difference(actual))
        if missing:
            return (
                f"transition method '{method_spec.method_id}' is missing required "
                f"{label}: {missing}"
            )
        unexpected = sorted(actual.difference(required.union(optional)))
        if unexpected and not allow_additional:
            return (
                f"transition method '{method_spec.method_id}' received undeclared "
                f"{label}: {unexpected}"
            )
    inline_operands = set(method_spec.inline_operands)
    for name, allowed_kinds in method_spec.operand_state_kinds.items():
        if name not in request.operands:
            continue
        refs = tuple(_iter_nested_state_refs(request.operands[name]))
        if not refs and name not in inline_operands:
            return (
                f"transition operand '{name}' requires a StateRef with kind in "
                f"{list(allowed_kinds)}"
            )
        invalid = sorted(
            {
                str(ref.state_kind)
                for ref in refs
                if str(ref.state_kind).strip().lower() not in set(allowed_kinds)
            }
        )
        if invalid:
            return (
                f"transition operand '{name}' has unsupported StateRef kinds "
                f"{invalid}; expected={list(allowed_kinds)}"
            )
    for name, allowed_kinds in method_spec.slot_state_kinds.items():
        ref = request.slot_refs.get(name)
        if ref is None:
            continue
        if str(ref.state_kind).strip().lower() not in set(allowed_kinds):
            return (
                f"transition slot '{name}' has StateRef kind '{ref.state_kind}'; "
                f"expected={list(allowed_kinds)}"
            )
    return None


def _transition_result_contract_mismatch(
    method_spec: StateTransitionMethodSpec,
    result: StateTransitionResult,
) -> str | None:
    for name, allowed_kinds in method_spec.result_slot_state_kinds.items():
        ref = result.slot_refs.get(name)
        if ref is None:
            continue
        if str(ref.state_kind).strip().lower() not in set(allowed_kinds):
            return (
                f"transition result slot '{name}' has StateRef kind "
                f"'{ref.state_kind}'; expected={list(allowed_kinds)}"
            )
    return None


def _iter_nested_state_refs(value: Any):
    if isinstance(value, StateRef):
        yield value
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_nested_state_refs(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_nested_state_refs(item)


def _validate_transition_successor(
    request: StateTransitionRequest,
    result: StateTransitionResult,
) -> None:
    previous = request.state_ref
    successor = result.state_ref
    if successor.provider_id != previous.provider_id:
        raise EvaluationProviderContractError(
            "state transition returned a successor owned by another provider"
        )
    if successor.scope_id != previous.scope_id:
        raise EvaluationProviderContractError(
            "state transition changed StateRef.scope_id"
        )
    if successor.trajectory_id != previous.trajectory_id:
        raise EvaluationProviderContractError(
            "state transition changed StateRef.trajectory_id"
        )
    if successor.state_kind != previous.state_kind:
        raise EvaluationProviderContractError(
            "state transition changed StateRef.state_kind"
        )
    if successor.device != previous.device:
        raise EvaluationProviderContractError(
            "state transition changed StateRef.device without an explicit transfer"
        )
    if successor.transport_scope != previous.transport_scope:
        raise EvaluationProviderContractError(
            "state transition changed StateRef.transport_scope without an explicit transfer"
        )
    if result.status == "skipped":
        if successor != previous:
            raise EvaluationProviderContractError(
                "a skipped state transition must return the original StateRef"
            )
        if dict(result.slot_refs) != dict(request.slot_refs):
            raise EvaluationProviderContractError(
                "a skipped state transition must return the original optimizer slots"
            )
        return
    if successor == previous:
        raise EvaluationProviderContractError(
            "an applied state transition must return a successor StateRef"
        )
    if successor.state_id == previous.state_id and (
        successor.version != previous.version + 1
    ):
        raise EvaluationProviderContractError(
            "an in-place state transition must increment StateRef.version by one"
        )
    missing_slots = sorted(set(request.slot_refs).difference(result.slot_refs))
    if missing_slots:
        raise EvaluationProviderContractError(
            f"state transition dropped optimizer slots: {missing_slots}"
        )
    for slot_name, slot in result.slot_refs.items():
        if slot.scope_id != successor.scope_id or slot.device != successor.device:
            raise EvaluationProviderContractError(
                f"optimizer slot '{slot_name}' is outside the successor state scope/device"
            )
        if slot.transport_scope != successor.transport_scope:
            raise EvaluationProviderContractError(
                f"optimizer slot '{slot_name}' changed transport scope"
            )
        previous_slot = request.slot_refs.get(slot_name)
        if previous_slot is None:
            continue
        if slot.state_id == previous_slot.state_id:
            raise EvaluationProviderContractError(
                f"applied optimizer slot '{slot_name}' must be copy-on-write; "
                "return a successor with a new state_id so the caller can abort safely"
            )


def _state_ref_mismatch(
    spec: EvaluationProviderSpec,
    request: EvaluationRequest,
) -> str | None:
    supported_kinds = set(spec.state_kinds)
    for state in request.states:
        if not isinstance(state, StateRef):
            continue
        if state.provider_id != spec.provider_id:
            return (
                f"StateRef '{state.state_id}' is owned by provider "
                f"'{state.provider_id}'"
            )
        state_kind = str(state.state_kind or "opaque").strip().lower()
        if supported_kinds and state_kind not in supported_kinds:
            return (
                f"StateRef kind '{state_kind}' is not supported by provider "
                f"'{spec.provider_id}'"
            )
    return None


def _resource_mismatch(
    requirement: ResourceRequirement,
    context: ResourceContext,
) -> str | None:
    grant = dict(context.grant or {})
    available_threads = int(grant.get("threads", context.threads) or context.threads)
    if int(requirement.threads) > available_threads:
        return (
            f"requires {requirement.threads} threads, grant has {available_threads}"
        )
    available_tokens = tuple(
        str(value) for value in grant.get("device_tokens", ()) or ()
    )
    available_gpus = int(grant.get("gpus", 0) or 0)
    if available_gpus <= 0:
        available_gpus = sum(
            1 for token in available_tokens if _device_family(token) in {"gpu", "mps"}
        )
    if available_gpus <= 0 and _device_family(context.device) in {"gpu", "mps"}:
        available_gpus = 1
    if int(requirement.gpus) > available_gpus:
        return f"requires {requirement.gpus} GPUs, grant has {available_gpus}"
    if requirement.device_tokens:
        missing_tokens = sorted(set(requirement.device_tokens).difference(available_tokens))
        if missing_tokens:
            return f"missing required device tokens: {missing_tokens}"
    grant_backend = str(grant.get("backend", "") or "").strip().lower()
    required_backend = str(requirement.resource_backend or "").strip().lower()
    if (
        grant_backend
        and required_backend not in {"", "any", "auto"}
        and grant_backend != required_backend
    ):
        return (
            f"resource backend '{required_backend}' conflicts with grant "
            f"'{grant_backend}'"
        )
    grant_capabilities = set(str(value) for value in grant.get("capabilities", ()) or ())
    missing_resource_capabilities = sorted(
        set(requirement.capabilities).difference(grant_capabilities)
    )
    if missing_resource_capabilities:
        return (
            "L0 grant lacks provider resource capabilities: "
            f"{missing_resource_capabilities}"
        )
    memory_error = _bounded_quantity_mismatch(
        "memory_mb", requirement.memory_mb, grant.get("memory_mb")
    )
    if memory_error:
        return memory_error
    gpu_memory_error = _bounded_quantity_mismatch(
        "gpu_memory_mb", requirement.gpu_memory_mb, grant.get("gpu_memory_mb")
    )
    if gpu_memory_error:
        return gpu_memory_error
    return None


def _bounded_quantity_mismatch(
    name: str,
    required: float | None,
    available: Any,
) -> str | None:
    if required is None:
        return None
    if available is None:
        return f"requires {name}={required}, but the L0 grant does not authorize it"
    if float(required) > float(available):
        return f"requires {name}={required}, grant has {available}"
    return None


def _select_device(
    spec: EvaluationProviderSpec,
    context: ResourceContext,
    *,
    required_device: str | None = None,
) -> str | None:
    grant = dict(context.grant or {})
    available: list[str] = []
    device_tokens = tuple(
        str(value) for value in grant.get("device_tokens", ()) or ()
    )
    current = str(context.device or "cpu").strip().lower()
    current_is_unresolved_token = (
        current in {value.strip().lower() for value in device_tokens}
        and not _is_physical_device_name(current)
    )
    if current not in {"", "auto", "none"} and not current_is_unresolved_token:
        available.append(current)
    available.extend(
        _normalize_physical_device_name(value)
        for value in device_tokens
        if _is_physical_device_name(value)
    )
    resolved_devices = dict(grant.get("resolved_devices", {}) or {})
    available.extend(
        str(resolved_devices[token]).strip().lower()
        for token in device_tokens
        if token in resolved_devices and str(resolved_devices[token]).strip()
    )
    if (
        int(grant.get("gpus", 0) or 0) > 0
        and not device_tokens
        and not any(
        _device_family(value) in {"gpu", "mps"} for value in available
        )
    ):
        available.append("gpu")
    # A CPU grant remains available for orchestration even when an accelerator
    # is also granted, unless a provider explicitly excludes CPU.
    if "cpu" not in available:
        available.append("cpu")
    compatible = [
        value
        for value in available
        if any(_device_pattern_matches(pattern, value) for pattern in spec.supported_devices)
    ]
    if required_device is not None:
        return _matching_device(str(required_device), compatible)
    for preferred in spec.preferred_devices:
        selected = _matching_device(preferred, compatible)
        if selected is not None:
            return selected
    for supported in spec.supported_devices:
        selected = _matching_device(supported, compatible)
        if selected is not None:
            return selected
    return None


def _is_physical_device_name(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    return (
        normalized == "gpu"
        or normalized.startswith(("cuda:", "gpu:"))
        or normalized == "mps"
        or normalized.startswith("mps:")
    )


def _normalize_physical_device_name(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized.startswith("gpu:"):
        return "cuda:" + normalized.split(":", 1)[1]
    return normalized


def _matching_device(requested: str, available: list[str]) -> str | None:
    requested_name = str(requested).strip().lower()
    requested_family = _device_family(requested_name)
    for value in available:
        if value == requested_name:
            return value
    if ":" in requested_name:
        return None
    for value in available:
        if _device_family(value) == requested_family:
            return value
    return None


def _device_pattern_matches(pattern: str, device: str) -> bool:
    normalized_pattern = str(pattern).strip().lower()
    normalized_device = str(device).strip().lower()
    if ":" in normalized_pattern:
        return normalized_pattern == normalized_device
    return _device_family(normalized_pattern) == _device_family(normalized_device)


def _device_preference_score(spec: EvaluationProviderSpec, device: str) -> int:
    family = _device_family(device)
    for index, preferred in enumerate(spec.preferred_devices):
        if _device_family(preferred) == family:
            return len(spec.preferred_devices) - index
    return 0


def _device_family(value: str) -> str:
    normalized = str(value or "cpu").strip().lower()
    if normalized.startswith("cuda") or normalized.startswith("gpu"):
        return "gpu"
    if normalized.startswith("mps"):
        return "mps"
    return "cpu" if normalized.startswith("cpu") else normalized


def _protocol_digest(payload: Mapping[str, Any]) -> str:
    wire_payload = encode_shared_value(
        dict(payload),
        path="evaluation.protocol_digest",
    )
    encoded = json.dumps(
        wire_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _bounded_rejection_reason(value: Any) -> str:
    message = str(value)
    if len(message) <= EVALUATION_REJECTION_REASON_MAX_LENGTH:
        return message
    digest = sha256(message.encode("utf-8")).hexdigest()
    keep = EVALUATION_REJECTION_REASON_MAX_LENGTH - len(digest) - 20
    return f"{message[:max(0, keep)]}… [sha256:{digest}]"


__all__ = [
    "BoundEvaluationProvider",
    "BoundStateTransitionProvider",
    "EvaluationGateway",
    "EvaluationProvider",
    "EvaluationProviderContractError",
    "EvaluationProviderError",
    "EvaluationProviderRegistry",
    "EvaluationProviderUnavailable",
    "EVALUATION_REJECTION_REASON_MAX_LENGTH",
    "StateTransitionProvider",
]
