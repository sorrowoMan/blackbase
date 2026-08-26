"""Executable conformance checks for stateful Evaluation Providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .state_transition import (
    StateMaterializationRequest,
    StateTransitionRequest,
    StateTransitionResult,
)


class EvaluationProviderConformanceError(RuntimeError):
    """A Provider violated a declared transactional state guarantee."""


@dataclass(frozen=True)
class CopyOnWriteConformanceReport:
    provider_id: str
    method_id: str
    transition_request_id: str
    checked_slots: tuple[str, ...]
    predecessor_state_ids: tuple[str, ...]

    @property
    def conformant(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "blackbase.copy_on_write_conformance/v1",
            "conformant": True,
            "provider_id": self.provider_id,
            "method_id": self.method_id,
            "transition_request_id": self.transition_request_id,
            "checked_slots": list(self.checked_slots),
            "predecessor_state_ids": list(self.predecessor_state_ids),
        }


def verify_copy_on_write_predecessors(
    gateway: Any,
    request: StateTransitionRequest,
    result: StateTransitionResult,
    resource_context: Mapping[str, Any] | Any,
) -> CopyOnWriteConformanceReport:
    """Prove that every replaced optimizer-slot predecessor is still live.

    This is an explicit certification check rather than a hot-path assertion:
    it materializes predecessor slots with ``release_after=False`` after an
    applied transition.  Providers should run it in their conformance suite or
    registration audit before claiming transactional copy-on-write support.
    """

    if not isinstance(request, StateTransitionRequest):
        raise TypeError("request must be a StateTransitionRequest")
    if not isinstance(result, StateTransitionResult):
        raise TypeError("result must be a StateTransitionResult")
    if request.request_id != result.request_id:
        raise EvaluationProviderConformanceError(
            "transition result belongs to another request"
        )
    if result.status != "applied":
        return CopyOnWriteConformanceReport(
            provider_id=request.state_ref.provider_id,
            method_id=result.method_id,
            transition_request_id=request.request_id,
            checked_slots=(),
            predecessor_state_ids=(),
        )
    materialize = getattr(gateway, "materialize", None)
    if not callable(materialize):
        raise EvaluationProviderConformanceError(
            "copy-on-write certification requires a materialization gateway"
        )

    checked: list[str] = []
    predecessor_ids: list[str] = []
    for name, predecessor in request.slot_refs.items():
        successor = result.slot_refs.get(name)
        if successor is None or successor.state_id == predecessor.state_id:
            continue
        try:
            materialized = materialize(
                StateMaterializationRequest(
                    state_ref=predecessor,
                    release_after=False,
                    metadata={
                        "reason": "copy_on_write_conformance",
                        "slot_name": str(name),
                        "transition_request_id": request.request_id,
                    },
                ),
                resource_context,
            )
        except BaseException as exc:
            raise EvaluationProviderConformanceError(
                f"copy-on-write predecessor for slot '{name}' is not materializable"
            ) from exc
        returned_ref = getattr(materialized, "state_ref", None)
        if returned_ref != predecessor:
            raise EvaluationProviderConformanceError(
                f"copy-on-write predecessor materialization for slot '{name}' "
                "returned another StateRef"
            )
        checked.append(str(name))
        predecessor_ids.append(predecessor.state_id)

    return CopyOnWriteConformanceReport(
        provider_id=request.state_ref.provider_id,
        method_id=result.method_id,
        transition_request_id=request.request_id,
        checked_slots=tuple(checked),
        predecessor_state_ids=tuple(predecessor_ids),
    )


__all__ = [
    "CopyOnWriteConformanceReport",
    "EvaluationProviderConformanceError",
    "verify_copy_on_write_predecessors",
]
