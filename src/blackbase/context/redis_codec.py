"""Versioned, fail-closed value envelopes for Redis-backed state stores.

The safe codec never imports or reconstructs arbitrary Python classes.  The
signed-pickle codec keeps the pickle bytes inside a JSON envelope so the HMAC
is verified *before* ``pickle.loads`` is called.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import pickle
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..resources.model import DataRef
from ..state_ref import StateRef
from ..types import UnknownState


SUPPORTED_REDIS_SERIALIZERS = frozenset({"safe", "pickle_signed", "pickle_unsafe"})
REDIS_VALUE_ENVELOPE = "blackbase.redis.value/v1"
_VALUE_TAG = "__blackbase_value__"


class RedisValueCodecError(ValueError):
    """A Redis value failed envelope, integrity, or type validation."""


@dataclass(frozen=True)
class RedisValueCodec:
    """Encode one value into a versioned Redis transport envelope."""

    serializer: str = "safe"
    hmac_env_var: str = "BLACKBASE_REDIS_VALUE_HMAC_KEY"
    unsafe_allow_legacy_pickle: bool = False
    max_payload_bytes: int = 8_388_608
    envelope_scope: str = "generic"
    max_depth: int = 128
    max_nodes: int = 1_000_000

    def __post_init__(self) -> None:
        serializer = str(self.serializer or "safe").strip().lower()
        if serializer not in SUPPORTED_REDIS_SERIALIZERS:
            raise ValueError(f"unsupported Redis value serializer: {self.serializer!r}")
        max_payload_bytes = int(self.max_payload_bytes)
        if max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive")
        max_depth = int(self.max_depth)
        max_nodes = int(self.max_nodes)
        if max_depth <= 0 or max_nodes <= 0:
            raise ValueError("max_depth and max_nodes must be positive")
        object.__setattr__(self, "serializer", serializer)
        object.__setattr__(self, "hmac_env_var", str(self.hmac_env_var or "").strip())
        object.__setattr__(self, "max_payload_bytes", max_payload_bytes)
        object.__setattr__(self, "max_depth", max_depth)
        object.__setattr__(self, "max_nodes", max_nodes)
        object.__setattr__(self, "envelope_scope", str(self.envelope_scope or "generic"))

    def dumps(self, value: Any) -> bytes:
        """Serialize ``value`` without silently degrading unsupported types."""

        if self.serializer == "safe":
            payload = _to_safe_wire(
                value,
                path="value",
                depth=0,
                max_depth=self.max_depth,
                active=set(),
            )
            envelope = self._base_envelope()
            envelope["payload"] = payload
        else:
            if self.serializer == "pickle_signed" and self._hmac_key() is None:
                raise RedisValueCodecError(
                    "serializer=pickle_signed requires a non-empty HMAC key in "
                    f"environment variable {self.hmac_env_var!r}"
                )
            payload_bytes = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            envelope = self._base_envelope()
            envelope["payload_b64"] = base64.b64encode(payload_bytes).decode("ascii")
            if self.serializer == "pickle_signed":
                envelope["hmac_sha256"] = hmac.new(
                    self._require_hmac_key(),
                    payload_bytes,
                    hashlib.sha256,
                ).hexdigest()

        try:
            raw = json.dumps(
                envelope,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise RedisValueCodecError("failed to encode Redis value envelope") from exc
        self._check_size(raw)
        return raw

    def loads(self, raw: bytes | bytearray | memoryview) -> Any:
        """Validate an envelope and restore its value.

        Legacy pickle is never attempted by the default safe codec.  It is
        available only behind an explicitly unsafe migration flag.
        """

        raw_bytes = bytes(raw)
        self._check_size(raw_bytes)
        try:
            envelope = json.loads(raw_bytes.decode("utf-8"))
        except Exception as exc:
            if self.serializer == "pickle_unsafe" or self.unsafe_allow_legacy_pickle:
                return self._load_legacy_pickle(raw_bytes)
            raise RedisValueCodecError("Redis value is not a valid JSON envelope") from exc
        if not isinstance(envelope, dict):
            raise RedisValueCodecError("Redis value envelope must be a mapping")
        if envelope.get("_blackbase_envelope") != REDIS_VALUE_ENVELOPE:
            raise RedisValueCodecError("unsupported Redis value envelope")
        if str(envelope.get("scope", "")) != self.envelope_scope:
            raise RedisValueCodecError("Redis value envelope scope mismatch")
        stored_serializer = str(envelope.get("serializer", "")).strip().lower()
        if stored_serializer != self.serializer:
            raise RedisValueCodecError(
                "Redis value serializer mismatch: "
                f"stored={stored_serializer!r}, configured={self.serializer!r}"
            )

        if self.serializer == "safe":
            if "payload" not in envelope:
                raise RedisValueCodecError("safe Redis value envelope has no payload")
            nodes = [0]
            return _from_safe_wire(
                envelope["payload"],
                path="value",
                depth=0,
                max_depth=self.max_depth,
                max_nodes=self.max_nodes,
                nodes=nodes,
            )

        encoded = envelope.get("payload_b64")
        if not isinstance(encoded, str):
            raise RedisValueCodecError("pickle Redis value envelope has no payload_b64")
        try:
            payload_bytes = base64.b64decode(encoded.encode("ascii"), validate=True)
        except Exception as exc:
            raise RedisValueCodecError("invalid base64 pickle payload") from exc
        if len(payload_bytes) > self.max_payload_bytes:
            raise RedisValueCodecError("decoded pickle payload exceeds max_payload_bytes")
        if self.serializer == "pickle_signed":
            supplied = envelope.get("hmac_sha256")
            if not isinstance(supplied, str):
                raise RedisValueCodecError("signed pickle envelope has no HMAC")
            expected = hmac.new(
                self._require_hmac_key(),
                payload_bytes,
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(supplied, expected):
                raise RedisValueCodecError("Redis value HMAC verification failed")
        try:
            return pickle.loads(payload_bytes)
        except Exception as exc:
            raise RedisValueCodecError("failed to decode verified pickle payload") from exc

    def _base_envelope(self) -> dict[str, Any]:
        return {
            "_blackbase_envelope": REDIS_VALUE_ENVELOPE,
            "scope": self.envelope_scope,
            "serializer": self.serializer,
        }

    def _hmac_key(self) -> bytes | None:
        if not self.hmac_env_var:
            return None
        raw = os.environ.get(self.hmac_env_var)
        if raw is None:
            return None
        key = str(raw).encode("utf-8")
        return key or None

    def _require_hmac_key(self) -> bytes:
        key = self._hmac_key()
        if key is None:
            raise RedisValueCodecError(
                "missing Redis value HMAC key in environment variable "
                f"{self.hmac_env_var!r}"
            )
        return key

    def _check_size(self, raw: bytes) -> None:
        if len(raw) > self.max_payload_bytes:
            raise RedisValueCodecError(
                f"{self.envelope_scope} payload too large: {len(raw)} bytes > "
                f"{self.max_payload_bytes} bytes"
            )

    def _load_legacy_pickle(self, raw: bytes) -> Any:
        try:
            decoded = pickle.loads(raw)
        except Exception as exc:
            raise RedisValueCodecError("failed to decode explicitly allowed legacy pickle") from exc
        if (
            isinstance(decoded, dict)
            and "_snapshot_envelope" in decoded
            and "payload" in decoded
        ):
            return decoded.get("payload")
        return decoded


def _to_safe_wire(
    value: Any,
    *,
    path: str,
    depth: int,
    max_depth: int,
    active: set[int],
) -> Any:
    if depth > max_depth:
        raise RedisValueCodecError(f"{path} exceeds safe codec depth limit")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {_VALUE_TAG: "float", "value": _nonfinite_float_name(value)}
    if isinstance(value, complex):
        return {
            _VALUE_TAG: "complex",
            "real": _to_safe_wire(
                float(value.real), path=f"{path}.real", depth=depth + 1,
                max_depth=max_depth, active=active,
            ),
            "imag": _to_safe_wire(
                float(value.imag), path=f"{path}.imag", depth=depth + 1,
                max_depth=max_depth, active=active,
            ),
        }
    if isinstance(value, np.generic):
        return _to_safe_wire(
            value.item(), path=path, depth=depth + 1, max_depth=max_depth, active=active
        )
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {
            _VALUE_TAG: "bytes",
            "data": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if isinstance(value, UnknownState):
        return {
            _VALUE_TAG: "unknown_state",
            "payload": _to_safe_wire(
                value.to_protocol_payload(),
                path=f"{path}.payload",
                depth=depth + 1,
                max_depth=max_depth,
                active=active,
            ),
        }
    if isinstance(value, DataRef):
        return {
            _VALUE_TAG: "data_ref",
            "payload": _to_safe_wire(
                value.as_dict(), path=f"{path}.payload", depth=depth + 1,
                max_depth=max_depth, active=active,
            ),
        }
    if isinstance(value, StateRef):
        return {
            _VALUE_TAG: "state_ref",
            "payload": _to_safe_wire(
                value.as_dict(), path=f"{path}.payload", depth=depth + 1,
                max_depth=max_depth, active=active,
            ),
        }
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise RedisValueCodecError(f"{path} contains an object-dtype ndarray")
        array = np.ascontiguousarray(value)
        return {
            _VALUE_TAG: "ndarray",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        }
    if isinstance(value, Mapping):
        return _encode_container(
            value,
            tag="mapping",
            path=path,
            depth=depth,
            max_depth=max_depth,
            active=active,
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return _encode_container(
            value,
            tag=type(value).__name__,
            path=path,
            depth=depth,
            max_depth=max_depth,
            active=active,
        )
    raise RedisValueCodecError(
        f"{path} contains unsupported safe value type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _encode_container(
    value: Any,
    *,
    tag: str,
    path: str,
    depth: int,
    max_depth: int,
    active: set[int],
) -> dict[str, Any]:
    object_id = id(value)
    if object_id in active:
        raise RedisValueCodecError(f"{path} contains a cyclic value")
    active.add(object_id)
    try:
        if isinstance(value, Mapping):
            items = [
                [
                    _to_safe_wire(
                        key, path=f"{path}.<key>", depth=depth + 1,
                        max_depth=max_depth, active=active,
                    ),
                    _to_safe_wire(
                        item, path=f"{path}[{key!r}]", depth=depth + 1,
                        max_depth=max_depth, active=active,
                    ),
                ]
                for key, item in value.items()
            ]
        else:
            items = [
                _to_safe_wire(
                    item, path=f"{path}[{index}]", depth=depth + 1,
                    max_depth=max_depth, active=active,
                )
                for index, item in enumerate(value)
            ]
        return {_VALUE_TAG: tag, "items": items}
    finally:
        active.remove(object_id)


def _from_safe_wire(
    value: Any,
    *,
    path: str,
    depth: int,
    max_depth: int,
    max_nodes: int,
    nodes: list[int],
) -> Any:
    nodes[0] += 1
    if nodes[0] > max_nodes:
        raise RedisValueCodecError("safe Redis value exceeds node limit")
    if depth > max_depth:
        raise RedisValueCodecError(f"{path} exceeds safe codec depth limit")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if not isinstance(value, dict):
        raise RedisValueCodecError(f"{path} contains invalid untagged JSON value")
    tag = value.get(_VALUE_TAG)
    if not isinstance(tag, str):
        raise RedisValueCodecError(f"{path} contains an untagged mapping")
    if tag == "float":
        return _restore_nonfinite_float(value.get("value"))
    if tag == "complex":
        real = _from_safe_wire(
            value.get("real"), path=f"{path}.real", depth=depth + 1,
            max_depth=max_depth, max_nodes=max_nodes, nodes=nodes,
        )
        imag = _from_safe_wire(
            value.get("imag"), path=f"{path}.imag", depth=depth + 1,
            max_depth=max_depth, max_nodes=max_nodes, nodes=nodes,
        )
        if not isinstance(real, (int, float)) or not isinstance(imag, (int, float)):
            raise RedisValueCodecError(f"{path} contains invalid complex components")
        return complex(real, imag)
    if tag == "bytes":
        encoded = value.get("data")
        if not isinstance(encoded, str):
            raise RedisValueCodecError(f"{path} contains invalid bytes payload")
        try:
            return base64.b64decode(encoded.encode("ascii"), validate=True)
        except Exception as exc:
            raise RedisValueCodecError(f"{path} contains invalid base64 bytes") from exc
    if tag == "ndarray":
        return _restore_array(value, path=path)
    if tag in {"unknown_state", "data_ref", "state_ref"}:
        payload = _from_safe_wire(
            value.get("payload"), path=f"{path}.payload", depth=depth + 1,
            max_depth=max_depth, max_nodes=max_nodes, nodes=nodes,
        )
        if not isinstance(payload, Mapping):
            raise RedisValueCodecError(f"{path} protocol payload must be a mapping")
        if tag == "unknown_state":
            return UnknownState.from_protocol_payload(payload)
        if tag == "data_ref":
            return DataRef.from_dict(payload)
        return StateRef.from_dict(payload)
    if tag == "mapping":
        raw_items = value.get("items")
        if not isinstance(raw_items, list):
            raise RedisValueCodecError(f"{path} mapping items must be a list")
        restored: dict[Any, Any] = {}
        for index, pair in enumerate(raw_items):
            if not isinstance(pair, list) or len(pair) != 2:
                raise RedisValueCodecError(f"{path} contains an invalid mapping pair")
            key = _from_safe_wire(
                pair[0], path=f"{path}.<key:{index}>", depth=depth + 1,
                max_depth=max_depth, max_nodes=max_nodes, nodes=nodes,
            )
            try:
                hash(key)
            except TypeError as exc:
                raise RedisValueCodecError(f"{path} contains an unhashable key") from exc
            restored[key] = _from_safe_wire(
                pair[1], path=f"{path}[{key!r}]", depth=depth + 1,
                max_depth=max_depth, max_nodes=max_nodes, nodes=nodes,
            )
        return restored
    if tag in {"list", "tuple", "set", "frozenset"}:
        raw_items = value.get("items")
        if not isinstance(raw_items, list):
            raise RedisValueCodecError(f"{path} collection items must be a list")
        items = [
            _from_safe_wire(
                item, path=f"{path}[{index}]", depth=depth + 1,
                max_depth=max_depth, max_nodes=max_nodes, nodes=nodes,
            )
            for index, item in enumerate(raw_items)
        ]
        if tag == "list":
            return items
        if tag == "tuple":
            return tuple(items)
        try:
            return set(items) if tag == "set" else frozenset(items)
        except TypeError as exc:
            raise RedisValueCodecError(f"{path} contains unhashable set items") from exc
    raise RedisValueCodecError(f"{path} contains unsupported safe value tag {tag!r}")


def _restore_array(value: Mapping[str, Any], *, path: str) -> np.ndarray:
    dtype_name = value.get("dtype")
    shape = value.get("shape")
    encoded = value.get("data")
    if not isinstance(dtype_name, str) or not isinstance(shape, list) or not isinstance(encoded, str):
        raise RedisValueCodecError(f"{path} contains an invalid ndarray envelope")
    try:
        dtype = np.dtype(dtype_name)
    except Exception as exc:
        raise RedisValueCodecError(f"{path} contains an invalid ndarray dtype") from exc
    if dtype.hasobject:
        raise RedisValueCodecError(f"{path} contains an object-dtype ndarray")
    try:
        dimensions = tuple(int(item) for item in shape)
    except Exception as exc:
        raise RedisValueCodecError(f"{path} contains an invalid ndarray shape") from exc
    if any(item < 0 for item in dimensions):
        raise RedisValueCodecError(f"{path} contains a negative ndarray dimension")
    try:
        payload = base64.b64decode(encoded.encode("ascii"), validate=True)
    except Exception as exc:
        raise RedisValueCodecError(f"{path} contains invalid ndarray base64") from exc
    expected_items = math.prod(dimensions) if dimensions else 1
    expected_bytes = expected_items * int(dtype.itemsize)
    if len(payload) != expected_bytes:
        raise RedisValueCodecError(
            f"{path} ndarray byte length mismatch: {len(payload)} != {expected_bytes}"
        )
    return np.frombuffer(payload, dtype=dtype).copy().reshape(dimensions)


def _nonfinite_float_name(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


def _restore_nonfinite_float(value: Any) -> float:
    if value == "nan":
        return float("nan")
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    raise RedisValueCodecError(f"invalid non-finite float marker {value!r}")


__all__ = [
    "REDIS_VALUE_ENVELOPE",
    "SUPPORTED_REDIS_SERIALIZERS",
    "RedisValueCodec",
    "RedisValueCodecError",
]
