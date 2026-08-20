"""Shared contracts for the Problem Evaluation Layer.

These are semantic-neutral protocol objects.  They describe what an evaluation
needs and what a provider returned; they neither allocate resources nor encode
an optimization algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from uuid import uuid4

from blackbase.resources import DataRef, ResourceContext, ResourceRequirement
from blackbase.state_ref import StateRef
from blackbase.types import (
    Feedback,
    UnknownState,
    decode_shared_value,
    encode_shared_value,
)


EVALUATION_SCHEMA_VERSION = 1
EVALUATION_IDENTIFIER_MAX_LENGTH = 256
EVALUATION_DECLARATION_MAX_ITEMS = 128
EVALUATION_MATERIALIZATION_TARGETS = frozenset({"unknown_state", "data_ref"})
CAPABILITY_POLICIES = frozenset({"required", "preferred", "optional"})
EVALUATION_MODES = frozenset({"evaluate", "train", "validate", "predict", "sample"})


@dataclass(frozen=True)
class CapabilityRequirement:
    """One semantic capability requested from an evaluation provider."""

    name: str
    policy: str = "required"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name or "").strip().lower()
        policy = str(self.policy or "required").strip().lower()
        if not name:
            raise ValueError("CapabilityRequirement.name must not be empty")
        _validate_identifier_length(name, field_name="CapabilityRequirement.name")
        if policy not in CAPABILITY_POLICIES:
            raise ValueError(
                "CapabilityRequirement.policy must be one of "
                f"{sorted(CAPABILITY_POLICIES)}"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "metadata", _freeze_wire_mapping(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "policy": self.policy,
            "metadata": _thaw_wire(self.metadata),
        }

    @classmethod
    def from_value(
        cls,
        value: "CapabilityRequirement | Mapping[str, Any] | str",
    ) -> "CapabilityRequirement":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(name=value)
        if isinstance(value, Mapping):
            return cls(
                name=str(value.get("name", "")),
                policy=str(value.get("policy", "required")),
                metadata=dict(value.get("metadata", {}) or {}),
            )
        raise TypeError(
            "capability requirement must be a string, mapping, or "
            "CapabilityRequirement"
        )


@dataclass(frozen=True)
class StateTransitionMethodSpec:
    """Static operand/parameter/slot contract for one provider compute kernel."""

    method_id: str
    required_operands: tuple[str, ...] = ()
    optional_operands: tuple[str, ...] = ()
    required_parameters: tuple[str, ...] = ()
    optional_parameters: tuple[str, ...] = ()
    required_slots: tuple[str, ...] = ()
    optional_slots: tuple[str, ...] = ()
    result_slots: tuple[str, ...] = ()
    operand_state_kinds: Mapping[str, Sequence[str] | str] = field(default_factory=dict)
    inline_operands: tuple[str, ...] = ()
    slot_state_kinds: Mapping[str, Sequence[str] | str] = field(default_factory=dict)
    result_slot_state_kinds: Mapping[str, Sequence[str] | str] = field(default_factory=dict)
    allow_additional_operands: bool = False
    allow_additional_parameters: bool = False
    allow_additional_slots: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        method_id = str(self.method_id or "").strip().lower()
        if not method_id:
            raise ValueError("StateTransitionMethodSpec.method_id must not be empty")
        _validate_identifier_length(
            method_id,
            field_name="StateTransitionMethodSpec.method_id",
        )
        normalized_fields = {
            "required_operands": _normalized_names(self.required_operands),
            "optional_operands": _normalized_names(self.optional_operands),
            "required_parameters": _normalized_names(self.required_parameters),
            "optional_parameters": _normalized_names(self.optional_parameters),
            "required_slots": _normalized_names(self.required_slots),
            "optional_slots": _normalized_names(self.optional_slots),
            "result_slots": _normalized_names(self.result_slots),
        }
        for field_name, values in normalized_fields.items():
            if len(values) > EVALUATION_DECLARATION_MAX_ITEMS:
                raise ValueError(
                    f"StateTransitionMethodSpec.{field_name} exceeds "
                    f"{EVALUATION_DECLARATION_MAX_ITEMS} items"
                )
            for value in values:
                _validate_identifier_length(
                    value,
                    field_name=f"StateTransitionMethodSpec.{field_name}",
                )
        for required_name, optional_name in (
            ("required_operands", "optional_operands"),
            ("required_parameters", "optional_parameters"),
            ("required_slots", "optional_slots"),
        ):
            overlap = set(normalized_fields[required_name]).intersection(
                normalized_fields[optional_name]
            )
            if overlap:
                raise ValueError(
                    f"{required_name} and {optional_name} overlap: {sorted(overlap)}"
                )
        declared_operands = set(normalized_fields["required_operands"]).union(
            normalized_fields["optional_operands"]
        )
        declared_slots = set(normalized_fields["required_slots"]).union(
            normalized_fields["optional_slots"]
        )
        declared_result_slots = set(normalized_fields["result_slots"])
        operand_state_kinds = _normalized_state_kind_map(
            self.operand_state_kinds,
            field_name="operand_state_kinds",
        )
        slot_state_kinds = _normalized_state_kind_map(
            self.slot_state_kinds,
            field_name="slot_state_kinds",
        )
        result_slot_state_kinds = _normalized_state_kind_map(
            self.result_slot_state_kinds,
            field_name="result_slot_state_kinds",
        )
        inline_operands = _normalized_names(self.inline_operands)
        for field_name, actual, declared, allow_additional in (
            (
                "operand_state_kinds",
                set(operand_state_kinds),
                declared_operands,
                bool(self.allow_additional_operands),
            ),
            (
                "inline_operands",
                set(inline_operands),
                declared_operands,
                bool(self.allow_additional_operands),
            ),
            (
                "slot_state_kinds",
                set(slot_state_kinds),
                declared_slots,
                bool(self.allow_additional_slots),
            ),
            (
                "result_slot_state_kinds",
                set(result_slot_state_kinds),
                declared_result_slots,
                False,
            ),
        ):
            unknown = actual.difference(declared)
            if unknown and not allow_additional:
                raise ValueError(
                    f"StateTransitionMethodSpec.{field_name} names undeclared fields: "
                    f"{sorted(unknown)}"
                )
        object.__setattr__(self, "method_id", method_id)
        for field_name, value in normalized_fields.items():
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "operand_state_kinds", operand_state_kinds)
        object.__setattr__(self, "inline_operands", inline_operands)
        object.__setattr__(self, "slot_state_kinds", slot_state_kinds)
        object.__setattr__(
            self,
            "result_slot_state_kinds",
            result_slot_state_kinds,
        )
        object.__setattr__(
            self,
            "allow_additional_operands",
            bool(self.allow_additional_operands),
        )
        object.__setattr__(
            self,
            "allow_additional_parameters",
            bool(self.allow_additional_parameters),
        )
        object.__setattr__(
            self,
            "allow_additional_slots",
            bool(self.allow_additional_slots),
        )
        object.__setattr__(self, "metadata", _freeze_wire_mapping(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "required_operands": list(self.required_operands),
            "optional_operands": list(self.optional_operands),
            "required_parameters": list(self.required_parameters),
            "optional_parameters": list(self.optional_parameters),
            "required_slots": list(self.required_slots),
            "optional_slots": list(self.optional_slots),
            "result_slots": list(self.result_slots),
            "operand_state_kinds": {
                key: list(value) for key, value in self.operand_state_kinds.items()
            },
            "inline_operands": list(self.inline_operands),
            "slot_state_kinds": {
                key: list(value) for key, value in self.slot_state_kinds.items()
            },
            "result_slot_state_kinds": {
                key: list(value)
                for key, value in self.result_slot_state_kinds.items()
            },
            "allow_additional_operands": self.allow_additional_operands,
            "allow_additional_parameters": self.allow_additional_parameters,
            "allow_additional_slots": self.allow_additional_slots,
            "metadata": _thaw_wire(self.metadata),
        }

    @classmethod
    def from_value(
        cls,
        value: "StateTransitionMethodSpec | Mapping[str, Any] | str",
    ) -> "StateTransitionMethodSpec":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            # String shorthand remains useful for simple providers, but is
            # explicitly permissive because it carries no shape contract.
            return cls(
                method_id=value,
                allow_additional_operands=True,
                allow_additional_parameters=True,
                allow_additional_slots=True,
            )
        if isinstance(value, Mapping):
            return cls(
                method_id=str(value.get("method_id", value.get("name", ""))),
                required_operands=tuple(value.get("required_operands", ()) or ()),
                optional_operands=tuple(value.get("optional_operands", ()) or ()),
                required_parameters=tuple(value.get("required_parameters", ()) or ()),
                optional_parameters=tuple(value.get("optional_parameters", ()) or ()),
                required_slots=tuple(value.get("required_slots", ()) or ()),
                optional_slots=tuple(value.get("optional_slots", ()) or ()),
                result_slots=tuple(value.get("result_slots", ()) or ()),
                operand_state_kinds=dict(
                    value.get("operand_state_kinds", {}) or {}
                ),
                inline_operands=tuple(value.get("inline_operands", ()) or ()),
                slot_state_kinds=dict(value.get("slot_state_kinds", {}) or {}),
                result_slot_state_kinds=dict(
                    value.get("result_slot_state_kinds", {}) or {}
                ),
                allow_additional_operands=bool(
                    value.get("allow_additional_operands", False)
                ),
                allow_additional_parameters=bool(
                    value.get("allow_additional_parameters", False)
                ),
                allow_additional_slots=bool(
                    value.get("allow_additional_slots", False)
                ),
                metadata=dict(value.get("metadata", {}) or {}),
            )
        raise TypeError(
            "transition method must be a string, mapping, or "
            "StateTransitionMethodSpec"
        )


@dataclass(frozen=True)
class EvaluationProviderSpec:
    """Static, auditable declaration made by an evaluation provider.

    ``resource_requirement`` is a hard minimum checked against an already
    granted L0 ResourceContext.  ``preferred_devices`` is only a binding
    preference and can never mint a GPU or other resource.
    """

    provider_id: str
    problem_ids: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    resource_requirement: ResourceRequirement = field(
        default_factory=lambda: ResourceRequirement(resource_backend="any")
    )
    supported_devices: tuple[str, ...] = ("cpu",)
    preferred_devices: tuple[str, ...] = ()
    compute_backend: str = "generic"
    modes: tuple[str, ...] = ("evaluate",)
    priority: int = 0
    state_kinds: tuple[str, ...] = ()
    materialization_targets: tuple[str, ...] = ()
    transition_methods: Sequence[
        StateTransitionMethodSpec | Mapping[str, Any] | str
    ] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        provider_id = str(self.provider_id or "").strip()
        if not provider_id:
            raise ValueError("EvaluationProviderSpec.provider_id must not be empty")
        _validate_identifier_length(
            provider_id,
            field_name="EvaluationProviderSpec.provider_id",
        )
        requirement = self.resource_requirement
        if isinstance(requirement, Mapping):
            requirement = ResourceRequirement.from_dict(requirement)
        if not isinstance(requirement, ResourceRequirement):
            raise TypeError("resource_requirement must be ResourceRequirement-compatible")
        requirement = ResourceRequirement.from_dict(requirement.as_dict())
        supported = _normalized_names(self.supported_devices or ("cpu",))
        preferred = _normalized_names(self.preferred_devices or supported)
        for field_name, values in (
            ("supported_devices", supported),
            ("preferred_devices", preferred),
        ):
            if len(values) > EVALUATION_DECLARATION_MAX_ITEMS:
                raise ValueError(
                    f"EvaluationProviderSpec.{field_name} exceeds "
                    f"{EVALUATION_DECLARATION_MAX_ITEMS} items"
                )
            for value in values:
                _validate_identifier_length(
                    value,
                    field_name=f"EvaluationProviderSpec.{field_name}",
                )
        unsupported_preferences = {
            value
            for value in preferred
            if not _device_preference_supported(value, supported)
        }
        if unsupported_preferences:
            raise ValueError(
                "preferred_devices must be a subset of supported_devices: "
                f"{sorted(unsupported_preferences)}"
            )
        modes = _normalized_names(self.modes or ("evaluate",))
        invalid_modes = set(modes).difference(EVALUATION_MODES)
        if invalid_modes:
            raise ValueError(f"unsupported evaluation modes: {sorted(invalid_modes)}")
        if int(requirement.gpus) > 0 and not any(
            _device_family(value) in {"gpu", "mps"} for value in supported
        ):
            raise ValueError(
                "a provider with gpus > 0 must support a GPU/MPS device"
            )
        object.__setattr__(self, "provider_id", provider_id)
        problem_ids = _normalized_names(self.problem_ids)
        if len(problem_ids) > EVALUATION_DECLARATION_MAX_ITEMS:
            raise ValueError(
                "EvaluationProviderSpec.problem_ids exceeds "
                f"{EVALUATION_DECLARATION_MAX_ITEMS} items"
            )
        for value in problem_ids:
            _validate_identifier_length(
                value,
                field_name="EvaluationProviderSpec.problem_ids",
            )
        object.__setattr__(self, "problem_ids", problem_ids)
        capabilities = _normalized_names(self.capabilities)
        if len(capabilities) > EVALUATION_DECLARATION_MAX_ITEMS:
            raise ValueError(
                "EvaluationProviderSpec.capabilities exceeds "
                f"{EVALUATION_DECLARATION_MAX_ITEMS} items"
            )
        for value in capabilities:
            _validate_identifier_length(
                value,
                field_name="EvaluationProviderSpec.capabilities",
            )
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "resource_requirement", requirement)
        object.__setattr__(self, "supported_devices", supported)
        object.__setattr__(self, "preferred_devices", preferred)
        object.__setattr__(self, "compute_backend", str(self.compute_backend or "generic"))
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "priority", int(self.priority))
        state_kinds = _normalized_names(self.state_kinds)
        if len(state_kinds) > EVALUATION_DECLARATION_MAX_ITEMS:
            raise ValueError(
                "EvaluationProviderSpec.state_kinds exceeds "
                f"{EVALUATION_DECLARATION_MAX_ITEMS} items"
            )
        for value in state_kinds:
            _validate_identifier_length(
                value,
                field_name="EvaluationProviderSpec.state_kinds",
            )
        object.__setattr__(self, "state_kinds", state_kinds)
        materialization_targets = _normalized_names(self.materialization_targets)
        if len(materialization_targets) > EVALUATION_DECLARATION_MAX_ITEMS:
            raise ValueError(
                "EvaluationProviderSpec.materialization_targets exceeds "
                f"{EVALUATION_DECLARATION_MAX_ITEMS} items"
            )
        for value in materialization_targets:
            _validate_identifier_length(
                value,
                field_name="EvaluationProviderSpec.materialization_targets",
            )
        invalid_materialization_targets = set(materialization_targets).difference(
            EVALUATION_MATERIALIZATION_TARGETS
        )
        if invalid_materialization_targets:
            raise ValueError(
                "unsupported state materialization targets: "
                f"{sorted(invalid_materialization_targets)}"
            )
        object.__setattr__(
            self,
            "materialization_targets",
            materialization_targets,
        )
        transition_methods = tuple(
            StateTransitionMethodSpec.from_value(value)
            for value in (self.transition_methods or ())
        )
        if len(transition_methods) > EVALUATION_DECLARATION_MAX_ITEMS:
            raise ValueError(
                "EvaluationProviderSpec.transition_methods exceeds "
                f"{EVALUATION_DECLARATION_MAX_ITEMS} items"
            )
        duplicate_methods = _duplicates(
            item.method_id for item in transition_methods
        )
        if duplicate_methods:
            raise ValueError(
                "transition method IDs must be unique: "
                f"{sorted(duplicate_methods)}"
            )
        object.__setattr__(self, "transition_methods", transition_methods)
        object.__setattr__(self, "metadata", _freeze_wire_mapping(self.metadata))

    @property
    def transition_method_ids(self) -> tuple[str, ...]:
        return tuple(item.method_id for item in self.transition_methods)

    def transition_method(self, method_id: str) -> StateTransitionMethodSpec | None:
        normalized = str(method_id or "").strip().lower()
        return next(
            (
                item
                for item in self.transition_methods
                if item.method_id == normalized
            ),
            None,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_type": "blackbase.evaluation_provider_spec",
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "provider_id": self.provider_id,
            "problem_ids": list(self.problem_ids),
            "capabilities": list(self.capabilities),
            "resource_requirement": self.resource_requirement.as_dict(),
            "supported_devices": list(self.supported_devices),
            "preferred_devices": list(self.preferred_devices),
            "compute_backend": self.compute_backend,
            "modes": list(self.modes),
            "priority": self.priority,
            "state_kinds": list(self.state_kinds),
            "materialization_targets": list(self.materialization_targets),
            "transition_methods": [
                item.as_dict() for item in self.transition_methods
            ],
            "metadata": _thaw_wire(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvaluationProviderSpec":
        data = _validate_payload(payload, "evaluation_provider_spec")
        return cls(
            provider_id=str(data.get("provider_id", "")),
            problem_ids=tuple(data.get("problem_ids", ()) or ()),
            capabilities=tuple(data.get("capabilities", ()) or ()),
            resource_requirement=(
                ResourceRequirement.from_dict(data.get("resource_requirement", {}))
                if data.get("resource_requirement") is not None
                else ResourceRequirement(resource_backend="any")
            ),
            supported_devices=tuple(data.get("supported_devices", ("cpu",)) or ()),
            preferred_devices=tuple(data.get("preferred_devices", ()) or ()),
            compute_backend=str(data.get("compute_backend", "generic")),
            modes=tuple(data.get("modes", ("evaluate",)) or ()),
            priority=int(data.get("priority", 0) or 0),
            state_kinds=tuple(data.get("state_kinds", ()) or ()),
            materialization_targets=tuple(
                data.get("materialization_targets", ()) or ()
            ),
            transition_methods=tuple(data.get("transition_methods", ()) or ()),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class EvaluationRequest:
    """A Problem's provider-neutral request for one evaluation batch."""

    problem_id: str
    states: Sequence[Any]
    mode: str = "evaluate"
    capabilities: Sequence[CapabilityRequirement | Mapping[str, Any] | str] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: f"eval_{uuid4().hex}")

    def __post_init__(self) -> None:
        problem_id = str(self.problem_id or "").strip()
        request_id = str(self.request_id or "").strip()
        mode = str(self.mode or "evaluate").strip().lower()
        if not problem_id:
            raise ValueError("EvaluationRequest.problem_id must not be empty")
        if not request_id:
            raise ValueError("EvaluationRequest.request_id must not be empty")
        _validate_identifier_length(
            problem_id,
            field_name="EvaluationRequest.problem_id",
        )
        _validate_identifier_length(
            request_id,
            field_name="EvaluationRequest.request_id",
        )
        if mode not in EVALUATION_MODES:
            raise ValueError(f"unsupported evaluation mode: {mode}")
        states = tuple(self.states or ())
        if not states:
            raise ValueError("EvaluationRequest.states must not be empty")
        requirements = tuple(
            CapabilityRequirement.from_value(value) for value in (self.capabilities or ())
        )
        if len(requirements) > EVALUATION_DECLARATION_MAX_ITEMS:
            raise ValueError(
                "EvaluationRequest.capabilities exceeds "
                f"{EVALUATION_DECLARATION_MAX_ITEMS} items"
            )
        duplicate_names = _duplicates(item.name for item in requirements)
        if duplicate_names:
            raise ValueError(
                "EvaluationRequest capability names must be unique: "
                f"{sorted(duplicate_names)}"
            )
        object.__setattr__(self, "problem_id", problem_id)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "capabilities", requirements)
        object.__setattr__(self, "payload", _freeze_wire_mapping(self.payload))
        object.__setattr__(self, "metadata", _freeze_wire_mapping(self.metadata))

    @property
    def required_capabilities(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.capabilities if item.policy == "required")

    @property
    def preferred_capabilities(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.capabilities if item.policy == "preferred")

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_type": "blackbase.evaluation_request",
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "request_id": self.request_id,
            "problem_id": self.problem_id,
            "mode": self.mode,
            "states": [_encode_state(value) for value in self.states],
            "capabilities": [item.as_dict() for item in self.capabilities],
            "payload": _thaw_wire(self.payload),
            "metadata": _thaw_wire(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvaluationRequest":
        data = _validate_payload(payload, "evaluation_request")
        return cls(
            request_id=str(data.get("request_id", "")),
            problem_id=str(data.get("problem_id", "")),
            mode=str(data.get("mode", "evaluate")),
            states=tuple(_decode_state(value) for value in data.get("states", ()) or ()),
            capabilities=tuple(data.get("capabilities", ()) or ()),
            payload=dict(decode_shared_value(data.get("payload", {})) or {}),
            metadata=dict(decode_shared_value(data.get("metadata", {})) or {}),
        )


@dataclass(frozen=True)
class EvaluationBinding:
    """Audit evidence for a provider selected inside an existing L0 grant."""

    request_id: str
    provider_id: str
    resource_context: ResourceContext
    device: str
    compute_backend: str
    request_digest: str = ""
    matched_capabilities: tuple[str, ...] = ()
    missing_preferred_capabilities: tuple[str, ...] = ()
    degraded: bool = False
    audit: Mapping[str, Any] = field(default_factory=dict)
    binding_id: str = field(default_factory=lambda: f"binding_{uuid4().hex}")

    def __post_init__(self) -> None:
        resource_context = ResourceContext.from_mapping(
            self.resource_context.as_dict()
            if isinstance(self.resource_context, ResourceContext)
            else self.resource_context
        )
        if not str(self.request_id or "").strip():
            raise ValueError("EvaluationBinding.request_id must not be empty")
        if not str(self.provider_id or "").strip():
            raise ValueError("EvaluationBinding.provider_id must not be empty")
        if not str(self.binding_id or "").strip():
            raise ValueError("EvaluationBinding.binding_id must not be empty")
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "provider_id", str(self.provider_id))
        object.__setattr__(self, "binding_id", str(self.binding_id))
        object.__setattr__(self, "resource_context", resource_context)
        object.__setattr__(self, "device", str(self.device or "cpu"))
        object.__setattr__(self, "compute_backend", str(self.compute_backend or "generic"))
        request_digest = str(self.request_digest or "").strip().lower()
        if request_digest and (
            len(request_digest) != 64
            or any(value not in "0123456789abcdef" for value in request_digest)
        ):
            raise ValueError("EvaluationBinding.request_digest must be a SHA-256 hex digest")
        object.__setattr__(self, "request_digest", request_digest)
        object.__setattr__(
            self, "matched_capabilities", _normalized_names(self.matched_capabilities)
        )
        object.__setattr__(
            self,
            "missing_preferred_capabilities",
            _normalized_names(self.missing_preferred_capabilities),
        )
        object.__setattr__(self, "degraded", bool(self.degraded))
        object.__setattr__(self, "audit", _freeze_wire_mapping(self.audit))

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_type": "blackbase.evaluation_binding",
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "binding_id": self.binding_id,
            "request_id": self.request_id,
            "provider_id": self.provider_id,
            "resource_context": self.resource_context.as_dict(),
            "device": self.device,
            "compute_backend": self.compute_backend,
            "request_digest": self.request_digest,
            "matched_capabilities": list(self.matched_capabilities),
            "missing_preferred_capabilities": list(
                self.missing_preferred_capabilities
            ),
            "degraded": self.degraded,
            "audit": _thaw_wire(self.audit),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvaluationBinding":
        data = _validate_payload(payload, "evaluation_binding")
        return cls(
            binding_id=str(data.get("binding_id", "")),
            request_id=str(data.get("request_id", "")),
            provider_id=str(data.get("provider_id", "")),
            resource_context=ResourceContext.from_mapping(
                data.get("resource_context", {}) or {}
            ),
            device=str(data.get("device", "cpu")),
            compute_backend=str(data.get("compute_backend", "generic")),
            request_digest=str(data.get("request_digest", "")),
            matched_capabilities=tuple(data.get("matched_capabilities", ()) or ()),
            missing_preferred_capabilities=tuple(
                data.get("missing_preferred_capabilities", ()) or ()
            ),
            degraded=bool(data.get("degraded", False)),
            audit=dict(data.get("audit", {}) or {}),
        )


@dataclass(frozen=True)
class EvaluationResult:
    """Feedback batch plus optional provider-owned successor states."""

    request_id: str
    feedback: Sequence[Feedback]
    result_states: Sequence[Any | None] = ()
    artifacts: Mapping[str, DataRef] = field(default_factory=dict)
    binding: EvaluationBinding | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        request_id = str(self.request_id or "").strip()
        if not request_id:
            raise ValueError("EvaluationResult.request_id must not be empty")
        feedback = tuple(
            item if isinstance(item, Feedback) else Feedback.from_dict(item)
            for item in (self.feedback or ())
        )
        if not feedback:
            raise ValueError("EvaluationResult.feedback must not be empty")
        result_states = tuple(self.result_states or ())
        if result_states and len(result_states) != len(feedback):
            raise ValueError(
                "EvaluationResult.result_states must be empty or align with feedback"
            )
        artifacts: dict[str, DataRef] = {}
        for key, value in dict(self.artifacts or {}).items():
            if not isinstance(value, DataRef):
                raise TypeError(f"artifacts['{key}'] must be a DataRef")
            artifacts[str(key)] = value
        binding = self.binding
        if binding is not None and not isinstance(binding, EvaluationBinding):
            binding = EvaluationBinding.from_dict(binding)
        if binding is not None and binding.request_id != request_id:
            raise ValueError("EvaluationResult.binding belongs to another request")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "feedback", feedback)
        object.__setattr__(self, "result_states", result_states)
        object.__setattr__(self, "artifacts", MappingProxyType(artifacts))
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "metadata", _freeze_wire_mapping(self.metadata))

    def with_binding(self, binding: EvaluationBinding) -> "EvaluationResult":
        if binding.request_id != self.request_id:
            raise ValueError("cannot attach a binding from another request")
        return replace(self, binding=binding)

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_type": "blackbase.evaluation_result",
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "request_id": self.request_id,
            "feedback": [item.as_dict() for item in self.feedback],
            "result_states": [
                None if value is None else _encode_state(value)
                for value in self.result_states
            ],
            "artifacts": {
                key: encode_shared_value(value, path=f"evaluation_result.artifacts.{key}")
                for key, value in self.artifacts.items()
            },
            "binding": None if self.binding is None else self.binding.as_dict(),
            "metadata": _thaw_wire(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvaluationResult":
        data = _validate_payload(payload, "evaluation_result")
        return cls(
            request_id=str(data.get("request_id", "")),
            feedback=tuple(
                Feedback.from_dict(item) for item in data.get("feedback", ()) or ()
            ),
            result_states=tuple(
                None if item is None else _decode_state(item)
                for item in data.get("result_states", ()) or ()
            ),
            artifacts={
                str(key): decode_shared_value(value)
                for key, value in dict(data.get("artifacts", {}) or {}).items()
            },
            binding=(
                None
                if data.get("binding") is None
                else EvaluationBinding.from_dict(data.get("binding", {}))
            ),
            metadata=dict(decode_shared_value(data.get("metadata", {})) or {}),
        )


def _encode_state(value: Any) -> dict[str, Any]:
    if isinstance(value, UnknownState):
        return {
            "state_type": "unknown_state",
            "value": encode_shared_value(
                value.to_protocol_payload(), path="evaluation_state.unknown_state"
            ),
        }
    if isinstance(value, StateRef):
        return {"state_type": "state_ref", "value": value.as_dict()}
    if isinstance(value, DataRef):
        return {
            "state_type": "data_ref",
            "value": encode_shared_value(value, path="evaluation_state.data_ref"),
        }
    return {
        "state_type": "inline",
        "value": encode_shared_value(value, path="evaluation_state.inline"),
    }


def _decode_state(payload: Mapping[str, Any]) -> Any:
    data = dict(payload or {})
    state_type = str(data.get("state_type", "inline"))
    value = data.get("value")
    if state_type == "unknown_state":
        return UnknownState.from_protocol_payload(decode_shared_value(value))
    if state_type == "state_ref":
        return StateRef.from_dict(value or {})
    if state_type in {"data_ref", "inline"}:
        return decode_shared_value(value)
    raise ValueError(f"unsupported evaluation state_type: {state_type}")


def _validate_payload(payload: Mapping[str, Any], type_name: str) -> dict[str, Any]:
    data = dict(payload or {})
    protocol_type = str(data.get("protocol_type", "") or "")
    expected = f"blackbase.{type_name}"
    if protocol_type and protocol_type != expected:
        raise ValueError(f"expected {expected} payload, got {protocol_type}")
    version = int(data.get("schema_version", EVALUATION_SCHEMA_VERSION) or 0)
    if version != EVALUATION_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported {type_name} schema_version={version}; "
            f"expected {EVALUATION_SCHEMA_VERSION}"
        )
    return data


def _normalized_names(values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value).strip().lower() for value in values if str(value).strip()))


def _normalized_state_kind_map(
    values: Mapping[str, Sequence[str] | str] | None,
    *,
    field_name: str,
) -> Mapping[str, tuple[str, ...]]:
    normalized: dict[str, tuple[str, ...]] = {}
    for raw_name, raw_kinds in dict(values or {}).items():
        name = str(raw_name or "").strip().lower()
        if not name:
            raise ValueError(f"StateTransitionMethodSpec.{field_name} has an empty key")
        _validate_identifier_length(
            name,
            field_name=f"StateTransitionMethodSpec.{field_name}",
        )
        kinds = _normalized_names(
            (raw_kinds,) if isinstance(raw_kinds, str) else tuple(raw_kinds or ())
        )
        if not kinds:
            raise ValueError(
                f"StateTransitionMethodSpec.{field_name}['{name}'] must declare "
                "at least one state kind"
            )
        for kind in kinds:
            _validate_identifier_length(
                kind,
                field_name=f"StateTransitionMethodSpec.{field_name}['{name}']",
            )
        normalized[name] = kinds
    if len(normalized) > EVALUATION_DECLARATION_MAX_ITEMS:
        raise ValueError(
            f"StateTransitionMethodSpec.{field_name} exceeds "
            f"{EVALUATION_DECLARATION_MAX_ITEMS} items"
        )
    return MappingProxyType(normalized)


def _duplicates(values: Sequence[str] | Any) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _device_family(value: str) -> str:
    normalized = str(value or "cpu").strip().lower()
    if normalized.startswith("cuda") or normalized.startswith("gpu"):
        return "gpu"
    if normalized.startswith("mps"):
        return "mps"
    return "cpu" if normalized.startswith("cpu") else normalized


def _device_preference_supported(
    preferred: str,
    supported_devices: Sequence[str],
) -> bool:
    preferred_name = str(preferred).strip().lower()
    for supported in supported_devices:
        supported_name = str(supported).strip().lower()
        if preferred_name == supported_name:
            return True
        if ":" not in supported_name and (
            _device_family(preferred_name) == _device_family(supported_name)
        ):
            return True
    return False


def _validate_identifier_length(value: str, *, field_name: str) -> None:
    if len(str(value)) > EVALUATION_IDENTIFIER_MAX_LENGTH:
        raise ValueError(
            f"{field_name} exceeds {EVALUATION_IDENTIFIER_MAX_LENGTH} characters"
        )


def _freeze_wire_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    encoded = encode_shared_value(dict(value or {}), path="evaluation.metadata")
    return _freeze_wire(encoded)


def _freeze_wire(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_wire(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_wire(item) for item in value)
    return value


def _thaw_wire(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_wire(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_wire(item) for item in value]
    return value


__all__ = [
    "CAPABILITY_POLICIES",
    "EVALUATION_MODES",
    "EVALUATION_DECLARATION_MAX_ITEMS",
    "EVALUATION_IDENTIFIER_MAX_LENGTH",
    "EVALUATION_SCHEMA_VERSION",
    "CapabilityRequirement",
    "EvaluationBinding",
    "EvaluationProviderSpec",
    "EvaluationRequest",
    "EvaluationResult",
    "StateTransitionMethodSpec",
]
