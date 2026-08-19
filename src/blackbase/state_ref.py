"""Lightweight references to provider-owned runtime state.

``StateRef`` is deliberately different from :class:`blackbase.resources.DataRef`:
DataRef identifies a published, durable artifact, while StateRef identifies live
state owned by an evaluation provider.  A StateRef never pretends that an
in-process tensor or model is portable across a Case boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


STATE_REF_SCHEMA_VERSION = 2
STATE_REF_TRANSPORT_SCOPES = frozenset({"process", "host", "cluster"})


@dataclass(frozen=True)
class StateRef:
    """Opaque reference to live state managed by one provider.

    The reference contains no Python object and no device buffer.  The named
    provider is responsible for resolving ``state_id`` inside ``scope_id`` and
    for enforcing the optimistic ``version`` fence.
    """

    provider_id: str
    state_id: str
    state_kind: str = "opaque"
    scope_id: str = ""
    trajectory_id: str = ""
    device: str = "cpu"
    version: int = 0
    transport_scope: str = "process"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        provider_id = str(self.provider_id or "").strip()
        state_id = str(self.state_id or "").strip()
        transport_scope = str(self.transport_scope or "process").strip().lower()
        if not provider_id:
            raise ValueError("StateRef.provider_id must not be empty")
        if not state_id:
            raise ValueError("StateRef.state_id must not be empty")
        if transport_scope not in STATE_REF_TRANSPORT_SCOPES:
            raise ValueError(
                "StateRef.transport_scope must be one of "
                f"{sorted(STATE_REF_TRANSPORT_SCOPES)}"
            )
        version = int(self.version)
        if version < 0:
            raise ValueError("StateRef.version must be non-negative")
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "state_id", state_id)
        object.__setattr__(self, "state_kind", str(self.state_kind or "opaque"))
        object.__setattr__(self, "scope_id", str(self.scope_id or ""))
        object.__setattr__(self, "trajectory_id", str(self.trajectory_id or ""))
        object.__setattr__(self, "device", str(self.device or "cpu"))
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "transport_scope", transport_scope)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_freeze_json_mapping(self.metadata)),
        )

    @property
    def process_local(self) -> bool:
        return self.transport_scope == "process"

    def next_version(self, *, metadata: Mapping[str, Any] | None = None) -> "StateRef":
        """Return a fenced reference for the provider's next committed state."""

        return StateRef(
            provider_id=self.provider_id,
            state_id=self.state_id,
            state_kind=self.state_kind,
            scope_id=self.scope_id,
            trajectory_id=self.trajectory_id,
            device=self.device,
            version=self.version + 1,
            transport_scope=self.transport_scope,
            metadata=self.metadata if metadata is None else metadata,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_type": "blackbase.state_ref",
            "schema_version": STATE_REF_SCHEMA_VERSION,
            "provider_id": self.provider_id,
            "state_id": self.state_id,
            "state_kind": self.state_kind,
            "scope_id": self.scope_id,
            "trajectory_id": self.trajectory_id,
            "device": self.device,
            "version": self.version,
            "transport_scope": self.transport_scope,
            "metadata": _thaw_json_mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateRef":
        data = dict(payload or {})
        protocol_type = str(data.get("protocol_type", "") or "")
        if protocol_type and protocol_type != "blackbase.state_ref":
            raise ValueError(
                "expected blackbase.state_ref payload, "
                f"got {protocol_type}"
            )
        version = int(data.get("schema_version", STATE_REF_SCHEMA_VERSION) or 0)
        if version not in {1, STATE_REF_SCHEMA_VERSION}:
            raise ValueError(
                f"unsupported StateRef schema_version={version}; "
                f"expected {STATE_REF_SCHEMA_VERSION}"
            )
        return cls(
            provider_id=str(data.get("provider_id", "")),
            state_id=str(data.get("state_id", "")),
            state_kind=str(data.get("state_kind", "opaque")),
            scope_id=str(data.get("scope_id", "")),
            trajectory_id=str(data.get("trajectory_id", "")),
            device=str(data.get("device", "cpu")),
            version=int(data.get("version", 0) or 0),
            transport_scope=str(data.get("transport_scope", "process")),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def _freeze_json_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        str(key): _freeze_json_value(item)
        for key, item in dict(value or {}).items()
    }


def _freeze_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(_freeze_json_mapping(value))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    raise TypeError(
        "StateRef.metadata must be JSON-compatible; "
        f"got {type(value).__name__}"
    )


def _thaw_json_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        str(key): _thaw_json_value(item)
        for key, item in dict(value or {}).items()
    }


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _thaw_json_mapping(value)
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


__all__ = [
    "STATE_REF_SCHEMA_VERSION",
    "STATE_REF_TRANSPORT_SCOPES",
    "StateRef",
]
