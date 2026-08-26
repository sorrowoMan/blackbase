"""Provider-owned state transition protocol.

The optimization Adapter chooses the method and parameters.  The evaluation
provider executes the corresponding compute kernel against state it owns and
returns a version-fenced successor reference.  No framework exchanges live
tensor objects through this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from blackbase.resources import DataRef
from blackbase.state_ref import StateRef
from blackbase.types import decode_shared_value, encode_shared_value
from blackbase.wire import freeze_wire_mapping, thaw_wire_value

from .model import (
    EVALUATION_DECLARATION_MAX_ITEMS,
    EVALUATION_IDENTIFIER_MAX_LENGTH,
    EVALUATION_MATERIALIZATION_TARGETS,
    EVALUATION_SCHEMA_VERSION,
    EvaluationBinding,
)


STATE_TRANSITION_STATUSES = frozenset({"applied", "skipped"})
STATE_RELEASE_STATUSES = frozenset({"released", "not_found"})
STATE_MATERIALIZATION_TARGETS = EVALUATION_MATERIALIZATION_TARGETS


class StateVersionConflict(RuntimeError):
    """Provider-side compare-and-swap rejected a stale StateRef version."""

    def __init__(
        self,
        state_ref: StateRef,
        *,
        actual_version: int | None = None,
    ) -> None:
        self.state_ref = state_ref
        self.expected_version = int(state_ref.version)
        self.actual_version = (
            None if actual_version is None else int(actual_version)
        )
        actual = "unknown" if actual_version is None else str(int(actual_version))
        super().__init__(
            f"stale StateRef version for provider='{state_ref.provider_id}', "
            f"state_id='{state_ref.state_id}': expected={state_ref.version}, "
            f"actual={actual}"
        )


@dataclass(frozen=True)
class StateMaterializationRequest:
    """Explicit request to export one live Provider state.

    Materialization is intentionally separate from Context projection.  It is
    the auditable boundary where a process-local ``StateRef`` becomes either a
    portable ``UnknownState`` or a durable ``DataRef``.
    """

    state_ref: StateRef
    target: str = "unknown_state"
    release_after: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: f"materialize_{uuid4().hex}")

    def __post_init__(self) -> None:
        if not isinstance(self.state_ref, StateRef):
            raise TypeError("StateMaterializationRequest.state_ref must be a StateRef")
        target = str(self.target or "unknown_state").strip().lower()
        request_id = str(self.request_id or "").strip()
        if target not in STATE_MATERIALIZATION_TARGETS:
            raise ValueError(
                "StateMaterializationRequest.target must be one of "
                f"{sorted(STATE_MATERIALIZATION_TARGETS)}"
            )
        if not request_id:
            raise ValueError("StateMaterializationRequest.request_id must not be empty")
        if len(request_id) > EVALUATION_IDENTIFIER_MAX_LENGTH:
            raise ValueError("StateMaterializationRequest.request_id is too long")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "release_after", bool(self.release_after))
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(
            self,
            "metadata",
            _freeze_wire_mapping(self.metadata, path="state_materialization.metadata"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_type": "blackbase.state_materialization_request",
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "request_id": self.request_id,
            "state_ref": self.state_ref.as_dict(),
            "target": self.target,
            "release_after": self.release_after,
            "metadata": thaw_wire_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateMaterializationRequest":
        data = _validate_payload(payload, "state_materialization_request")
        return cls(
            request_id=str(data.get("request_id", "")),
            state_ref=StateRef.from_dict(data.get("state_ref", {}) or {}),
            target=str(data.get("target", "unknown_state")),
            release_after=bool(data.get("release_after", False)),
            metadata=dict(decode_shared_value(data.get("metadata", {})) or {}),
        )


@dataclass(frozen=True)
class StateMaterializationResult:
    """Portable value exported from one unchanged, version-fenced StateRef."""

    request_id: str
    state_ref: StateRef
    target: str
    value: Any
    binding: EvaluationBinding | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from blackbase.types import UnknownState

        request_id = str(self.request_id or "").strip()
        target = str(self.target or "").strip().lower()
        if not request_id:
            raise ValueError("StateMaterializationResult.request_id must not be empty")
        if not isinstance(self.state_ref, StateRef):
            raise TypeError("StateMaterializationResult.state_ref must be a StateRef")
        if target not in STATE_MATERIALIZATION_TARGETS:
            raise ValueError(
                "StateMaterializationResult.target must be one of "
                f"{sorted(STATE_MATERIALIZATION_TARGETS)}"
            )
        value = self.value
        if target == "unknown_state":
            if not isinstance(value, UnknownState):
                raise TypeError("unknown_state materialization must return UnknownState")
            value = UnknownState(
                values=np.asarray(value.as_array(), dtype=float).copy(),
                metadata=dict(
                    decode_shared_value(
                        encode_shared_value(
                            dict(value.metadata or {}),
                            path="state_materialization.value.metadata",
                        )
                    )
                    or {}
                ),
            )
        elif not isinstance(value, DataRef):
            raise TypeError("data_ref materialization must return DataRef")
        binding = self.binding
        if binding is not None and not isinstance(binding, EvaluationBinding):
            binding = EvaluationBinding.from_dict(binding)
        if binding is not None and binding.request_id != request_id:
            raise ValueError("StateMaterializationResult.binding belongs to another request")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "binding", binding)
        object.__setattr__(
            self,
            "metadata",
            _freeze_wire_mapping(self.metadata, path="state_materialization.metadata"),
        )

    def with_binding(self, binding: EvaluationBinding) -> "StateMaterializationResult":
        if binding.request_id != self.request_id:
            raise ValueError("cannot attach a binding from another request")
        return replace(self, binding=binding)

    def as_dict(self) -> dict[str, Any]:
        from blackbase.types import UnknownState

        if isinstance(self.value, UnknownState):
            encoded_value = {
                "kind": "unknown_state",
                "payload": encode_shared_value(
                    self.value.to_protocol_payload(),
                    path="state_materialization.value",
                ),
            }
        else:
            encoded_value = {
                "kind": "data_ref",
                "payload": encode_shared_value(
                    self.value,
                    path="state_materialization.value",
                ),
            }
        return {
            "protocol_type": "blackbase.state_materialization_result",
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "request_id": self.request_id,
            "state_ref": self.state_ref.as_dict(),
            "target": self.target,
            "value": encoded_value,
            "binding": None if self.binding is None else self.binding.as_dict(),
            "metadata": thaw_wire_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateMaterializationResult":
        from blackbase.types import UnknownState

        data = _validate_payload(payload, "state_materialization_result")
        encoded_value = dict(data.get("value", {}) or {})
        kind = str(encoded_value.get("kind", ""))
        decoded = decode_shared_value(encoded_value.get("payload"))
        if kind == "unknown_state":
            value = UnknownState.from_protocol_payload(dict(decoded or {}))
        elif kind == "data_ref":
            if not isinstance(decoded, DataRef):
                raise TypeError("materialized data_ref payload did not decode to DataRef")
            value = decoded
        else:
            raise ValueError(f"unsupported materialized value kind: {kind}")
        return cls(
            request_id=str(data.get("request_id", "")),
            state_ref=StateRef.from_dict(data.get("state_ref", {}) or {}),
            target=str(data.get("target", "")),
            value=value,
            binding=(
                None
                if data.get("binding") is None
                else EvaluationBinding.from_dict(data.get("binding", {}))
            ),
            metadata=dict(decode_shared_value(data.get("metadata", {})) or {}),
        )


@dataclass(frozen=True)
class StateReleaseRequest:
    """Release live Provider state owned by one scope/trajectory."""

    provider_id: str
    scope_id: str = ""
    trajectory_id: str = ""
    state_kinds: tuple[str, ...] = ()
    state_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: f"release_{uuid4().hex}")

    def __post_init__(self) -> None:
        provider_id = str(self.provider_id or "").strip()
        request_id = str(self.request_id or "").strip()
        scope_id = str(self.scope_id or "").strip()
        trajectory_id = str(self.trajectory_id or "").strip()
        if not provider_id:
            raise ValueError("StateReleaseRequest.provider_id must not be empty")
        if not request_id:
            raise ValueError("StateReleaseRequest.request_id must not be empty")
        if not scope_id:
            raise ValueError(
                "StateReleaseRequest.scope_id is required for L0 namespace authorization"
            )
        kinds = tuple(
            dict.fromkeys(
                str(value or "").strip().lower()
                for value in tuple(self.state_kinds or ())
                if str(value or "").strip()
            )
        )
        if len(kinds) > EVALUATION_DECLARATION_MAX_ITEMS:
            raise ValueError("StateReleaseRequest.state_kinds has too many items")
        state_ids = tuple(
            dict.fromkeys(
                str(value or "").strip()
                for value in tuple(self.state_ids or ())
                if str(value or "").strip()
            )
        )
        if len(state_ids) > EVALUATION_DECLARATION_MAX_ITEMS:
            raise ValueError("StateReleaseRequest.state_ids has too many items")
        if any(len(value) > EVALUATION_IDENTIFIER_MAX_LENGTH for value in state_ids):
            raise ValueError("StateReleaseRequest.state_ids contains an identifier that is too long")
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "scope_id", scope_id)
        object.__setattr__(self, "trajectory_id", trajectory_id)
        object.__setattr__(self, "state_kinds", kinds)
        object.__setattr__(self, "state_ids", state_ids)
        object.__setattr__(
            self,
            "metadata",
            _freeze_wire_mapping(self.metadata, path="state_release.metadata"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_type": "blackbase.state_release_request",
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "request_id": self.request_id,
            "provider_id": self.provider_id,
            "scope_id": self.scope_id,
            "trajectory_id": self.trajectory_id,
            "state_kinds": list(self.state_kinds),
            "state_ids": list(self.state_ids),
            "metadata": thaw_wire_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateReleaseRequest":
        data = _validate_payload(payload, "state_release_request")
        return cls(
            request_id=str(data.get("request_id", "")),
            provider_id=str(data.get("provider_id", "")),
            scope_id=str(data.get("scope_id", "")),
            trajectory_id=str(data.get("trajectory_id", "")),
            state_kinds=tuple(data.get("state_kinds", ()) or ()),
            state_ids=tuple(data.get("state_ids", ()) or ()),
            metadata=dict(decode_shared_value(data.get("metadata", {})) or {}),
        )


@dataclass(frozen=True)
class StateReleaseResult:
    """Bound evidence that a Provider scope/trajectory was torn down."""

    request_id: str
    provider_id: str
    status: str
    released_count: int = 0
    released_state_ids: tuple[str, ...] = ()
    not_found_state_ids: tuple[str, ...] = ()
    binding: EvaluationBinding | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        request_id = str(self.request_id or "").strip()
        provider_id = str(self.provider_id or "").strip()
        status = str(self.status or "").strip().lower()
        released_count = int(self.released_count)
        if not request_id or not provider_id:
            raise ValueError("StateReleaseResult request/provider ids must not be empty")
        if status not in STATE_RELEASE_STATUSES:
            raise ValueError(
                "StateReleaseResult.status must be one of "
                f"{sorted(STATE_RELEASE_STATUSES)}"
            )
        if released_count < 0:
            raise ValueError("StateReleaseResult.released_count must be non-negative")
        ids = tuple(str(value) for value in tuple(self.released_state_ids or ()))
        not_found_ids = tuple(
            str(value) for value in tuple(self.not_found_state_ids or ())
        )
        if len(ids) > EVALUATION_DECLARATION_MAX_ITEMS:
            raise ValueError("StateReleaseResult.released_state_ids has too many items")
        if len(not_found_ids) > EVALUATION_DECLARATION_MAX_ITEMS:
            raise ValueError("StateReleaseResult.not_found_state_ids has too many items")
        if len(set(ids)) != len(ids) or len(set(not_found_ids)) != len(not_found_ids):
            raise ValueError("StateReleaseResult state ids must be unique")
        if set(ids).intersection(not_found_ids):
            raise ValueError("released and not-found StateRef ids must be disjoint")
        if released_count != len(ids):
            raise ValueError(
                "StateReleaseResult.released_count must match released_state_ids"
            )
        if status == "not_found" and released_count != 0:
            raise ValueError("not_found release cannot report released states")
        if status == "released" and released_count <= 0:
            raise ValueError("released status requires released_count > 0")
        binding = self.binding
        if binding is not None and not isinstance(binding, EvaluationBinding):
            binding = EvaluationBinding.from_dict(binding)
        if binding is not None and binding.request_id != request_id:
            raise ValueError("StateReleaseResult.binding belongs to another request")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "released_count", released_count)
        object.__setattr__(self, "released_state_ids", ids)
        object.__setattr__(self, "not_found_state_ids", not_found_ids)
        object.__setattr__(self, "binding", binding)
        object.__setattr__(
            self,
            "metadata",
            _freeze_wire_mapping(self.metadata, path="state_release.metadata"),
        )

    def with_binding(self, binding: EvaluationBinding) -> "StateReleaseResult":
        if binding.request_id != self.request_id:
            raise ValueError("cannot attach a binding from another request")
        return replace(self, binding=binding)

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_type": "blackbase.state_release_result",
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "request_id": self.request_id,
            "provider_id": self.provider_id,
            "status": self.status,
            "released_count": self.released_count,
            "released_state_ids": list(self.released_state_ids),
            "not_found_state_ids": list(self.not_found_state_ids),
            "binding": None if self.binding is None else self.binding.as_dict(),
            "metadata": thaw_wire_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateReleaseResult":
        data = _validate_payload(payload, "state_release_result")
        return cls(
            request_id=str(data.get("request_id", "")),
            provider_id=str(data.get("provider_id", "")),
            status=str(data.get("status", "")),
            released_count=int(data.get("released_count", 0) or 0),
            released_state_ids=tuple(data.get("released_state_ids", ()) or ()),
            not_found_state_ids=tuple(
                data.get("not_found_state_ids", ()) or ()
            ),
            binding=(
                None
                if data.get("binding") is None
                else EvaluationBinding.from_dict(data.get("binding", {}))
            ),
            metadata=dict(decode_shared_value(data.get("metadata", {})) or {}),
        )


@dataclass(frozen=True)
class StateTransitionRequest:
    """Algorithm-selected state update to execute inside the owner Provider.

    Typical operands are ``gradient`` or ``direction``. Stateful compute
    kernels can receive prior optimizer slots through ``slot_refs`` and return
    their successor slots in :class:`StateTransitionResult`. Applied slot
    transitions are copy-on-write: a successor slot must have a new ``state_id``
    so the caller can commit or abort without destroying its predecessor.
    """

    state_ref: StateRef
    method_id: str
    operands: Mapping[str, Any] = field(default_factory=dict)
    slot_refs: Mapping[str, StateRef] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    step_index: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: f"transition_{uuid4().hex}")

    def __post_init__(self) -> None:
        if not isinstance(self.state_ref, StateRef):
            raise TypeError("StateTransitionRequest.state_ref must be a StateRef")
        method_id = str(self.method_id or "").strip().lower()
        request_id = str(self.request_id or "").strip()
        if not method_id:
            raise ValueError("StateTransitionRequest.method_id must not be empty")
        if not request_id:
            raise ValueError("StateTransitionRequest.request_id must not be empty")
        if len(method_id) > EVALUATION_IDENTIFIER_MAX_LENGTH:
            raise ValueError("StateTransitionRequest.method_id is too long")
        if len(request_id) > EVALUATION_IDENTIFIER_MAX_LENGTH:
            raise ValueError("StateTransitionRequest.request_id is too long")
        step_index = int(self.step_index)
        if step_index < 0:
            raise ValueError("StateTransitionRequest.step_index must be non-negative")
        operands: dict[str, Any] = {}
        for key, value in dict(self.operands or {}).items():
            name = _validated_key(key, field_name="operands")
            if name in operands:
                raise ValueError(f"duplicate normalized operand key: {name}")
            operands[name] = _freeze_operand(value)
        if len(operands) > EVALUATION_DECLARATION_MAX_ITEMS:
            raise ValueError("StateTransitionRequest.operands has too many items")
        slots: dict[str, StateRef] = {}
        for key, value in dict(self.slot_refs or {}).items():
            name = _validated_key(key, field_name="slot_refs")
            if name in slots:
                raise ValueError(f"duplicate normalized slot key: {name}")
            if not isinstance(value, StateRef):
                raise TypeError(f"slot_refs['{name}'] must be a StateRef")
            if value.provider_id != self.state_ref.provider_id:
                raise ValueError(
                    f"slot_refs['{name}'] is owned by provider "
                    f"'{value.provider_id}', expected '{self.state_ref.provider_id}'"
                )
            _validate_local_ref_compatibility(
                self.state_ref,
                value,
                label=f"slot_refs['{name}']",
            )
            slots[name] = value
        if len(slots) > EVALUATION_DECLARATION_MAX_ITEMS:
            raise ValueError("StateTransitionRequest.slot_refs has too many items")
        for operand_name, operand in operands.items():
            for ref in _iter_state_refs(operand):
                if ref.provider_id != self.state_ref.provider_id:
                    raise ValueError(
                        f"operand '{operand_name}' contains StateRef owned by "
                        f"'{ref.provider_id}', expected '{self.state_ref.provider_id}'"
                    )
                _validate_local_ref_compatibility(
                    self.state_ref,
                    ref,
                    label=f"operand '{operand_name}'",
                )
        parameters: dict[str, Any] = {}
        for key, value in dict(self.parameters or {}).items():
            name = _validated_key(key, field_name="parameters")
            if name in parameters:
                raise ValueError(f"duplicate normalized parameter key: {name}")
            parameters[name] = value
        if len(parameters) > EVALUATION_DECLARATION_MAX_ITEMS:
            raise ValueError("StateTransitionRequest.parameters has too many items")
        object.__setattr__(self, "method_id", method_id)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "step_index", step_index)
        object.__setattr__(self, "operands", MappingProxyType(operands))
        object.__setattr__(self, "slot_refs", MappingProxyType(slots))
        object.__setattr__(
            self,
            "parameters",
            _freeze_wire_mapping(parameters, path="state_transition.parameters"),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_wire_mapping(self.metadata, path="state_transition.metadata"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_type": "blackbase.state_transition_request",
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "request_id": self.request_id,
            "state_ref": self.state_ref.as_dict(),
            "method_id": self.method_id,
            "operands": {
                key: encode_shared_value(
                    value,
                    path=f"state_transition_request.operands.{key}",
                )
                for key, value in self.operands.items()
            },
            "slot_refs": {
                key: value.as_dict() for key, value in self.slot_refs.items()
            },
            "parameters": thaw_wire_value(self.parameters),
            "step_index": self.step_index,
            "metadata": thaw_wire_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateTransitionRequest":
        data = _validate_payload(payload, "state_transition_request")
        return cls(
            request_id=str(data.get("request_id", "")),
            state_ref=StateRef.from_dict(data.get("state_ref", {}) or {}),
            method_id=str(data.get("method_id", "")),
            operands={
                str(key): decode_shared_value(value)
                for key, value in dict(data.get("operands", {}) or {}).items()
            },
            slot_refs={
                str(key): StateRef.from_dict(value)
                for key, value in dict(data.get("slot_refs", {}) or {}).items()
            },
            parameters=dict(decode_shared_value(data.get("parameters", {})) or {}),
            step_index=int(data.get("step_index", 0) or 0),
            metadata=dict(decode_shared_value(data.get("metadata", {})) or {}),
        )


@dataclass(frozen=True)
class StateTransitionResult:
    """Version-fenced successor state and provider-owned optimizer slots."""

    request_id: str
    method_id: str
    status: str
    state_ref: StateRef
    slot_refs: Mapping[str, StateRef] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    binding: EvaluationBinding | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        request_id = str(self.request_id or "").strip()
        method_id = str(self.method_id or "").strip().lower()
        status = str(self.status or "").strip().lower()
        if not request_id:
            raise ValueError("StateTransitionResult.request_id must not be empty")
        if not method_id:
            raise ValueError("StateTransitionResult.method_id must not be empty")
        if len(method_id) > EVALUATION_IDENTIFIER_MAX_LENGTH:
            raise ValueError("StateTransitionResult.method_id is too long")
        if len(request_id) > EVALUATION_IDENTIFIER_MAX_LENGTH:
            raise ValueError("StateTransitionResult.request_id is too long")
        if status not in STATE_TRANSITION_STATUSES:
            raise ValueError(
                "StateTransitionResult.status must be one of "
                f"{sorted(STATE_TRANSITION_STATUSES)}"
            )
        if not isinstance(self.state_ref, StateRef):
            raise TypeError("StateTransitionResult.state_ref must be a StateRef")
        slots: dict[str, StateRef] = {}
        for key, value in dict(self.slot_refs or {}).items():
            name = _validated_key(key, field_name="slot_refs")
            if name in slots:
                raise ValueError(f"duplicate normalized slot key: {name}")
            if not isinstance(value, StateRef):
                raise TypeError(f"slot_refs['{name}'] must be a StateRef")
            if value.provider_id != self.state_ref.provider_id:
                raise ValueError(
                    f"slot_refs['{name}'] belongs to another provider"
                )
            _validate_local_ref_compatibility(
                self.state_ref,
                value,
                label=f"slot_refs['{name}']",
            )
            slots[name] = value
        if len(slots) > EVALUATION_DECLARATION_MAX_ITEMS:
            raise ValueError("StateTransitionResult.slot_refs has too many items")
        binding = self.binding
        if binding is not None and not isinstance(binding, EvaluationBinding):
            binding = EvaluationBinding.from_dict(binding)
        if binding is not None and binding.request_id != request_id:
            raise ValueError("StateTransitionResult.binding belongs to another request")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "method_id", method_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "slot_refs", MappingProxyType(slots))
        object.__setattr__(self, "binding", binding)
        object.__setattr__(
            self,
            "metrics",
            _freeze_wire_mapping(self.metrics, path="state_transition.metrics"),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_wire_mapping(self.metadata, path="state_transition.metadata"),
        )

    def with_binding(self, binding: EvaluationBinding) -> "StateTransitionResult":
        if binding.request_id != self.request_id:
            raise ValueError("cannot attach a binding from another request")
        return replace(self, binding=binding)

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_type": "blackbase.state_transition_result",
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "request_id": self.request_id,
            "method_id": self.method_id,
            "status": self.status,
            "state_ref": self.state_ref.as_dict(),
            "slot_refs": {
                key: value.as_dict() for key, value in self.slot_refs.items()
            },
            "metrics": thaw_wire_value(self.metrics),
            "binding": None if self.binding is None else self.binding.as_dict(),
            "metadata": thaw_wire_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateTransitionResult":
        data = _validate_payload(payload, "state_transition_result")
        return cls(
            request_id=str(data.get("request_id", "")),
            method_id=str(data.get("method_id", "")),
            status=str(data.get("status", "")),
            state_ref=StateRef.from_dict(data.get("state_ref", {}) or {}),
            slot_refs={
                str(key): StateRef.from_dict(value)
                for key, value in dict(data.get("slot_refs", {}) or {}).items()
            },
            metrics=dict(decode_shared_value(data.get("metrics", {})) or {}),
            binding=(
                None
                if data.get("binding") is None
                else EvaluationBinding.from_dict(data.get("binding", {}))
            ),
            metadata=dict(decode_shared_value(data.get("metadata", {})) or {}),
        )


def _freeze_operand(value: Any) -> Any:
    if isinstance(value, StateRef):
        return value
    if isinstance(value, DataRef):
        encode_shared_value(value, path="state_transition.operand")
        return DataRef.from_dict(value.as_dict())
    if isinstance(value, np.ndarray):
        detached = np.asarray(value).copy()
        encode_shared_value(detached, path="state_transition.operand")
        detached.setflags(write=False)
        return detached
    if isinstance(value, np.generic):
        scalar = value.item()
        encode_shared_value(scalar, path="state_transition.operand")
        return scalar
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_operand(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_operand(item) for item in value)
    raise TypeError(
        "StateTransitionRequest operands must be shared-protocol values; "
        f"got {type(value).__name__}"
    )


def _iter_state_refs(value: Any):
    if isinstance(value, StateRef):
        yield value
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_state_refs(item)
        return
    if isinstance(value, tuple):
        for item in value:
            yield from _iter_state_refs(item)


def _validated_key(value: Any, *, field_name: str) -> str:
    key = str(value or "").strip().lower()
    if not key:
        raise ValueError(f"StateTransitionRequest.{field_name} keys must not be empty")
    if len(key) > EVALUATION_IDENTIFIER_MAX_LENGTH:
        raise ValueError(
            f"StateTransitionRequest.{field_name} key exceeds "
            f"{EVALUATION_IDENTIFIER_MAX_LENGTH} characters"
        )
    return key


def _validate_local_ref_compatibility(
    state_ref: StateRef,
    other_ref: StateRef,
    *,
    label: str,
) -> None:
    if other_ref.scope_id != state_ref.scope_id:
        raise ValueError(f"{label} belongs to another StateRef scope")
    if other_ref.trajectory_id != state_ref.trajectory_id:
        raise ValueError(f"{label} belongs to another StateRef trajectory")
    if other_ref.device != state_ref.device:
        raise ValueError(f"{label} belongs to another device")
    if other_ref.transport_scope != state_ref.transport_scope:
        raise ValueError(f"{label} uses another transport scope")


def _freeze_wire_mapping(
    value: Mapping[str, Any] | None,
    *,
    path: str,
) -> Mapping[str, Any]:
    encoded = encode_shared_value(dict(value or {}), path=path)
    if not isinstance(encoded, Mapping):
        raise TypeError(f"encoded {path} must be a mapping")
    return freeze_wire_mapping(encoded, path=path)


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


__all__ = [
    "STATE_MATERIALIZATION_TARGETS",
    "STATE_RELEASE_STATUSES",
    "STATE_TRANSITION_STATUSES",
    "StateMaterializationRequest",
    "StateMaterializationResult",
    "StateReleaseRequest",
    "StateReleaseResult",
    "StateTransitionRequest",
    "StateTransitionResult",
    "StateVersionConflict",
]
