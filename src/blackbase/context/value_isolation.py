"""Fail-closed recursive isolation for values crossing Context boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from blackbase.resources import DataRef
from blackbase.state_ref import StateRef


_IMMUTABLE_SCALARS = (str, bytes, int, float, bool, complex, type(None))


def detach_context_value(value: Any, *, path: str = "context") -> Any:
    """Return a recursively detached Context-safe value.

    Mutable containers and arrays are copied.  Formal references are rebuilt so
    mutable metadata cannot leak back to their publisher.  Unknown objects and
    cyclic structures are rejected rather than returned by reference.
    """

    return _detach_context_value(value, path=str(path or "context"), active=set())


def _detach_context_value(value: Any, *, path: str, active: set[int]) -> Any:
    if isinstance(value, _IMMUTABLE_SCALARS):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, DataRef):
        metadata = _detach_container(
            value.metadata,
            path=f"{path}.metadata",
            active=active,
        )
        return DataRef(
            uri=value.uri,
            kind=value.kind,
            backend=value.backend,
            media_type=value.media_type,
            checksum=value.checksum,
            size_bytes=value.size_bytes,
            metadata=metadata,
        )
    if isinstance(value, StateRef):
        return StateRef.from_dict(value.as_dict())
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(f"{path} contains an object-dtype ndarray")
        return np.array(value, copy=True, subok=False)
    if isinstance(value, Mapping):
        return _detach_container(value, path=path, active=active)
    if isinstance(value, list):
        return _detach_container(value, path=path, active=active)
    if isinstance(value, tuple):
        return _detach_container(value, path=path, active=active)
    if isinstance(value, set):
        return _detach_container(value, path=path, active=active)
    if isinstance(value, frozenset):
        return _detach_container(value, path=path, active=active)
    raise TypeError(
        f"{path} contains unsupported context value type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _detach_container(value: Any, *, path: str, active: set[int]) -> Any:
    object_id = id(value)
    if object_id in active:
        raise TypeError(f"{path} contains a cyclic context value")
    active.add(object_id)
    try:
        if isinstance(value, Mapping):
            out: dict[Any, Any] = {}
            for key, item in value.items():
                detached_key = _detach_context_value(
                    key,
                    path=f"{path}.<key>",
                    active=active,
                )
                try:
                    hash(detached_key)
                except TypeError as exc:
                    raise TypeError(f"{path} contains an unhashable mapping key") from exc
                out[detached_key] = _detach_context_value(
                    item,
                    path=f"{path}[{key!r}]",
                    active=active,
                )
            return out
        if isinstance(value, list):
            return [
                _detach_context_value(item, path=f"{path}[{index}]", active=active)
                for index, item in enumerate(value)
            ]
        if isinstance(value, tuple):
            return tuple(
                _detach_context_value(item, path=f"{path}[{index}]", active=active)
                for index, item in enumerate(value)
            )
        if isinstance(value, set):
            return {
                _detach_context_value(item, path=f"{path}.<set>", active=active)
                for item in value
            }
        if isinstance(value, frozenset):
            return frozenset(
                _detach_context_value(item, path=f"{path}.<set>", active=active)
                for item in value
            )
    finally:
        active.remove(object_id)
    raise AssertionError("unreachable context container type")


__all__ = ["detach_context_value"]
