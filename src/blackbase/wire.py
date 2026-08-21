"""Immutable helpers for transport-safe shared protocol values."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping


def freeze_wire_value(value: Any, *, path: str = "value") -> Any:
    """Detach one JSON-compatible value into recursively immutable containers."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): freeze_wire_value(item, path=f"{path}.{key}")
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_wire_value(item, path=f"{path}[]") for item in value
        )
    raise TypeError(
        f"shared protocol field '{path}' is not wire-safe: "
        f"{type(value).__name__}"
    )


def freeze_wire_mapping(
    value: Mapping[str, Any] | None,
    *,
    path: str = "value",
) -> Mapping[str, Any]:
    frozen = freeze_wire_value(dict(value or {}), path=path)
    assert isinstance(frozen, Mapping)
    return frozen


def thaw_wire_value(value: Any) -> Any:
    """Return a detached mutable JSON-compatible representation."""

    if isinstance(value, Mapping):
        return {str(key): thaw_wire_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_wire_value(item) for item in value]
    return value


def thaw_wire_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    thawed = thaw_wire_value(value or {})
    assert isinstance(thawed, dict)
    return thawed


__all__ = [
    "freeze_wire_mapping",
    "freeze_wire_value",
    "thaw_wire_mapping",
    "thaw_wire_value",
]
