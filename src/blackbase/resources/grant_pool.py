"""Concurrent partitioning of one authoritative L0 resource grant."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Condition, RLock
from typing import Any, Callable, Iterator, Mapping
from uuid import uuid4

from blackbase.wire import freeze_wire_mapping, thaw_wire_mapping

from .context import ResourceContext, coerce_resource_context
from .model import ResourceRequest, ResourceRequirement


class ResourceSubgrantError(RuntimeError):
    """A child request cannot be satisfied inside its parent grant."""


@dataclass(frozen=True)
class ResourceSubgrant:
    grant_id: str
    resource_context: ResourceContext
    resources: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "grant_id", str(self.grant_id))
        object.__setattr__(
            self,
            "resources",
            freeze_wire_mapping(self.resources, path="resource_subgrant.resources"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "grant_id": self.grant_id,
            "resources": thaw_wire_mapping(self.resources),
            "resource_context": self.resource_context.as_dict(),
        }


class ResourceGrantPool:
    """Atomically slice quantitative resources from one existing parent lease.

    This pool does not mint a lease.  Every child retains the parent's lease and
    fencing token while receiving a bounded, concurrently accounted grant view.
    """

    def __init__(self, parent: ResourceContext | Mapping[str, Any]) -> None:
        self.parent = coerce_resource_context(parent)
        grant = thaw_wire_mapping(self.parent.grant)
        lease_resources = dict(
            thaw_wire_mapping(self.parent.lease).get("resources", {}) or {}
        )
        self._totals = {
            "workers": max(
                1,
                int(
                    grant.get(
                        "workers",
                        lease_resources.get(
                            "workers",
                            grant.get("threads", self.parent.threads),
                        ),
                    )
                    or 1
                ),
            ),
            "threads": max(1, int(grant.get("threads", self.parent.threads) or 1)),
            "gpus": max(0, int(grant.get("gpus", lease_resources.get("gpus", 0)) or 0)),
            "memory_mb": max(
                0.0,
                float(grant.get("memory_mb", lease_resources.get("memory_mb", 0.0)) or 0.0),
            ),
            "gpu_memory_mb": max(
                0.0,
                float(
                    grant.get(
                        "gpu_memory_mb",
                        lease_resources.get("gpu_memory_mb", 0.0),
                    )
                    or 0.0
                ),
            ),
        }
        self._available = dict(self._totals)
        self._tokens = tuple(
            str(item)
            for item in grant.get(
                "device_tokens",
                lease_resources.get("device_tokens", ()),
            )
            or ()
        )
        self._available_tokens = list(self._tokens)
        self._capabilities = frozenset(
            str(item)
            for item in grant.get(
                "capabilities",
                lease_resources.get("capabilities", ()),
            )
            or ()
        )
        # ``backend`` is the resource-provider backend (local/redis/...), not
        # ResourceContext.execution_backend (thread/process/outer).  Legacy
        # grants that omit it are local grants.
        self._backend = str(
            grant.get("backend", lease_resources.get("backend", "local"))
            or "local"
        )
        self._compute_backend = str(
            grant.get(
                "compute_backend",
                self.parent.compute_backend
                or lease_resources.get("compute_backend", "auto"),
            )
            or self.parent.compute_backend
        )
        self._device = str(
            grant.get(
                "device",
                self.parent.device or lease_resources.get("device", "cpu"),
            )
            or self.parent.device
        )
        self._resolved_devices = dict(
            grant.get(
                "resolved_devices",
                lease_resources.get("resolved_devices", {}),
            )
            or {}
        )
        self._condition = Condition(RLock())
        self._active: dict[str, ResourceSubgrant] = {}

    @contextmanager
    def acquire(
        self,
        request: ResourceRequest | ResourceRequirement | Mapping[str, Any],
        *,
        scope: str,
        namespace_suffix: str,
        checkpoint: Callable[[], None] | None = None,
        deadline_at: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[ResourceSubgrant]:
        item = self._coerce_request(request)
        resources = self._requested_resources(item)
        requested_tokens = tuple(resources["device_tokens"])
        if len(set(requested_tokens)) != len(requested_tokens):
            raise ResourceSubgrantError("child device_tokens must be unique")
        metadata_payload = thaw_wire_mapping(
            freeze_wire_mapping(
                metadata,
                path="resource_subgrant.metadata",
            )
        )
        self._validate_static(resources)
        allocated_tokens: tuple[str, ...] = ()
        grant_id = f"subgrant-{uuid4().hex}"
        with self._condition:
            while True:
                if checkpoint is not None:
                    checkpoint()
                if deadline_at > 0 and time.time() >= float(deadline_at):
                    raise TimeoutError("resource subgrant deadline expired")
                if self._can_allocate(resources, requested_tokens):
                    break
                remaining = (
                    max(0.001, float(deadline_at) - time.time())
                    if deadline_at > 0
                    else 0.05
                )
                self._condition.wait(timeout=min(0.05, remaining))
            for name in self._available:
                self._available[name] -= resources[name]
            if requested_tokens:
                allocated_tokens = requested_tokens
            elif resources["gpus"]:
                allocated_tokens = tuple(
                    self._available_tokens[: int(resources["gpus"])]
                )
            for token in allocated_tokens:
                self._available_tokens.remove(token)
            resources["device_tokens"] = list(allocated_tokens)
            resources["resolved_devices"] = {
                token: self._resolved_devices[token]
                for token in allocated_tokens
                if token in self._resolved_devices
            }
            resolved = tuple(resources["resolved_devices"].values())
            requested_compute_backend = str(
                resources["compute_backend"] or "auto"
            )
            compute_backend = (
                self._compute_backend
                if _is_generic_resource_value(requested_compute_backend)
                else requested_compute_backend
            )
            requested_device = str(resources["device"] or "auto")
            if resolved:
                device = str(resolved[0])
            elif int(resources["gpus"]) > 0:
                device = self._device
            elif not _is_generic_resource_value(requested_device):
                device = requested_device
            else:
                device = self._device
            resources["backend"] = self._backend
            resources["compute_backend"] = compute_backend
            resources["device"] = device
            namespace = ".".join(
                value
                for value in (
                    str(self.parent.namespace or "").strip("."),
                    str(namespace_suffix or "").strip("."),
                )
                if value
            )
            child = ResourceContext(
                scope=str(scope or self.parent.scope),
                execution_backend=self.parent.execution_backend,
                compute_backend=compute_backend,
                device=device,
                threads=int(resources["threads"]),
                nested=True,
                namespace=namespace,
                grant=resources,
                lease=thaw_wire_mapping(self.parent.lease),
                metadata={
                    **thaw_wire_mapping(self.parent.metadata),
                    "parent_namespace": self.parent.namespace,
                    "parent_lease_id": str(
                        thaw_wire_mapping(self.parent.lease).get("lease_id", "")
                    ),
                    "resource_subgrant_id": grant_id,
                    **metadata_payload,
                },
            )
            subgrant = ResourceSubgrant(
                grant_id=grant_id,
                resource_context=child,
                resources=resources,
            )
            self._active[grant_id] = subgrant
        try:
            yield subgrant
        finally:
            with self._condition:
                self._active.pop(grant_id, None)
                for name in self._available:
                    self._available[name] += resources[name]
                self._available_tokens.extend(allocated_tokens)
                order = {token: index for index, token in enumerate(self._tokens)}
                self._available_tokens.sort(
                    key=lambda token: order.get(token, len(order))
                )
                self._condition.notify_all()

    def audit(self) -> dict[str, Any]:
        with self._condition:
            return {
                "parent_namespace": self.parent.namespace,
                "totals": dict(self._totals),
                "available": dict(self._available),
                "active": {
                    key: value.as_dict() for key, value in self._active.items()
                },
            }

    @staticmethod
    def _coerce_request(
        request: ResourceRequest | ResourceRequirement | Mapping[str, Any],
    ) -> ResourceRequest:
        if isinstance(request, ResourceRequest):
            return request
        if isinstance(request, ResourceRequirement):
            return request.to_resource_request()
        return ResourceRequest.from_dict(request)

    @staticmethod
    def _requested_resources(request: ResourceRequest) -> dict[str, Any]:
        return {
            "workers": int(request.workers),
            "threads": int(request.threads),
            "gpus": max(int(request.gpus), len(tuple(request.device_tokens))),
            "memory_mb": max(0.0, float(request.memory_mb or 0.0)),
            "gpu_memory_mb": max(0.0, float(request.gpu_memory_mb or 0.0)),
            "device_tokens": list(request.device_tokens),
            "backend": str(request.backend),
            "compute_backend": str(request.compute_backend),
            "device": str(request.device),
            "capabilities": list(request.capabilities),
        }

    def _validate_static(self, resources: Mapping[str, Any]) -> None:
        for name in ("workers", "threads", "gpus", "memory_mb", "gpu_memory_mb"):
            if float(resources[name]) > float(self._totals[name]):
                raise ResourceSubgrantError(
                    f"child requests {name}={resources[name]}, "
                    f"parent grant has {self._totals[name]}"
                )
        tokens = set(str(item) for item in resources["device_tokens"])
        if not tokens.issubset(self._tokens):
            raise ResourceSubgrantError(
                "child requests device tokens outside the parent grant"
            )
        capabilities = set(str(item) for item in resources["capabilities"])
        if not capabilities.issubset(self._capabilities):
            raise ResourceSubgrantError(
                "child requests capabilities outside the parent grant: "
                f"{sorted(capabilities.difference(self._capabilities))}"
            )
        backend = str(resources["backend"] or "auto").lower()
        if backend not in {"", "auto", "any", self._backend.lower()}:
            raise ResourceSubgrantError(
                f"child backend '{backend}' conflicts with parent '{self._backend}'"
            )
        compute_backend = str(resources["compute_backend"] or "auto").lower()
        parent_compute_backend = self._compute_backend.lower()
        if (
            not _is_generic_resource_value(parent_compute_backend)
            and not _is_generic_resource_value(compute_backend)
            and compute_backend != parent_compute_backend
        ):
            raise ResourceSubgrantError(
                "child compute backend "
                f"'{compute_backend}' conflicts with parent "
                f"'{self._compute_backend}'"
            )
        requested_device = str(resources["device"] or "auto")
        if _is_accelerator_device(requested_device):
            parent_has_accelerator = (
                int(self._totals["gpus"]) > 0
                or bool(self._tokens)
                or _is_accelerator_device(self._device)
            )
            if not parent_has_accelerator:
                raise ResourceSubgrantError(
                    f"child device '{requested_device}' is outside the CPU-only "
                    "parent grant"
                )
            if _is_specific_device(requested_device):
                allowed_devices = {
                    self._device.lower(),
                    *(str(token).lower() for token in self._tokens),
                    *(
                        str(device).lower()
                        for device in self._resolved_devices.values()
                    ),
                }
                if requested_device.lower() not in allowed_devices:
                    raise ResourceSubgrantError(
                        f"child device '{requested_device}' is outside the parent "
                        "device grant"
                    )

    def _can_allocate(
        self,
        resources: Mapping[str, Any],
        requested_tokens: tuple[str, ...],
    ) -> bool:
        quantities_available = all(
            float(self._available[name]) >= float(resources[name])
            for name in self._available
        )
        if not quantities_available:
            return False
        if requested_tokens:
            return all(token in self._available_tokens for token in requested_tokens)
        if self._tokens and int(resources["gpus"]) > 0:
            return len(self._available_tokens) >= int(resources["gpus"])
        # Some authoritative grants account GPU capacity quantitatively without
        # naming physical/logical tokens.  The numeric GPU ledger remains the
        # authority in that mode; tokens are only an additional exclusive
        # ledger when the parent actually supplied them.
        return True


def _is_generic_resource_value(value: str) -> bool:
    return str(value or "").strip().lower() in {"", "auto", "any", "none"}


def _is_accelerator_device(value: str) -> bool:
    device = str(value or "").strip().lower()
    return device.startswith(
        ("accelerator", "cuda", "gpu", "mps", "rocm", "xpu")
    )


def _is_specific_device(value: str) -> bool:
    device = str(value or "").strip().lower()
    return _is_accelerator_device(device) and ":" in device


__all__ = [
    "ResourceGrantPool",
    "ResourceSubgrant",
    "ResourceSubgrantError",
]
