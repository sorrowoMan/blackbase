"""
Snapshot store backends for large runtime artifacts (population/objectives/etc).

Snapshot stores keep large objects out of context while still providing
stable references for replay, inspection, and bias usage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Mapping, Optional
import hashlib
import json
import logging
import os
import pickle
import copy
import time
import uuid

import numpy as np

from ..types import UnknownState
from .redis_codec import RedisValueCodec, RedisValueCodecError

logger = logging.getLogger(__name__)

GENERIC_SNAPSHOT_SCHEMA = "generic_payload_v1"
GENERIC_SNAPSHOT_MARKER = "__blackbase_generic_payload__"
GENERIC_SNAPSHOT_VALUE = "value"


def wrap_snapshot_payload(payload: Any) -> Dict[str, Any]:
    """Wrap an arbitrary payload for stores whose write contract is mapping-only."""
    return {
        GENERIC_SNAPSHOT_MARKER: 1,
        GENERIC_SNAPSHOT_VALUE: payload,
    }


def unwrap_snapshot_payload(value: Any) -> Any:
    """Return a generic payload from a record, while preserving normal snapshots."""
    data = value.data if hasattr(value, "data") else value
    if isinstance(data, Mapping) and data.get(GENERIC_SNAPSHOT_MARKER) == 1:
        return data.get(GENERIC_SNAPSHOT_VALUE)
    return data


def _report_soft_error(**kwargs: Any) -> None:
    """Lazy import to avoid circular imports."""
    exc = kwargs.get("exc")
    if kwargs.get("strict") and isinstance(exc, Exception):
        raise exc
    component = str(kwargs.get("component", "snapshot_store"))
    event = str(kwargs.get("event", "unknown"))
    if isinstance(exc, Exception):
        logger.warning(
            "[soft-error:fallback] %s.%s: %s: %s",
            component,
            event,
            exc.__class__.__name__,
            str(exc),
        )
    else:
        logger.warning("[soft-error:fallback] %s.%s", component, event)


@dataclass(frozen=True)
class SnapshotHandle:
    """Reference to a snapshot in the store."""
    
    key: str
    backend: str
    schema: str
    meta: Dict[str, Any]
    created_at: float
    revision: int = 1
    content_digest: str = ""
    expires_at: Optional[float] = None
    pinned: bool = False


@dataclass(frozen=True)
class SnapshotRecord(SnapshotHandle):
    """Complete snapshot record with data."""
    
    data: Dict[str, Any] = field(default_factory=dict)


def make_snapshot_key(
    *,
    prefix: Optional[str] = None,
    generation: Optional[int] = None,
    step: Optional[int] = None,
    suffix: Optional[str] = None,
) -> str:
    """Generate a unique snapshot key."""
    parts = []
    if prefix:
        parts.append(str(prefix).strip())
    if generation is not None:
        parts.append(f"gen-{int(generation)}")
    if step is not None:
        parts.append(f"step-{int(step)}")
    if suffix:
        parts.append(str(suffix).strip())
    parts.append(uuid.uuid4().hex[:8])
    return "/".join([p for p in parts if p])


class SnapshotStore(ABC):
    """Abstract snapshot store interface."""
    
    backend: str = "unknown"
    
    @abstractmethod
    def write(
        self,
        data: Mapping[str, Any],
        *,
        key: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        schema: str = "population_snapshot_v1",
        ttl_seconds: Optional[float] = None,
        write_once: bool = False,
        expected_revision: Optional[int] = None,
    ) -> SnapshotHandle:
        """Write a snapshot to the store."""
        raise NotImplementedError
    
    @abstractmethod
    def read(self, key: str) -> Optional[SnapshotRecord]:
        """Read a snapshot from the store."""
        raise NotImplementedError
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a snapshot from the store."""
        raise NotImplementedError

    @abstractmethod
    def pin(self, key: str, *, owner: str) -> SnapshotHandle:
        """Prevent authority evidence from expiring or being deleted."""
        raise NotImplementedError

    @abstractmethod
    def unpin(self, key: str, *, owner: str) -> SnapshotHandle:
        """Release one named retention owner."""
        raise NotImplementedError


def snapshot_content_digest(schema: str, data: Mapping[str, Any]) -> str:
    """Return a stable typed digest for an immutable Snapshot payload."""

    codec = RedisValueCodec(
        serializer="safe",
        max_payload_bytes=1_073_741_824,
        envelope_scope="snapshot-content-digest",
    )
    raw = codec.dumps({"schema": str(schema), "data": dict(data)})
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _process_is_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _detached_snapshot_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    codec = RedisValueCodec(
        serializer="safe",
        max_payload_bytes=1_073_741_824,
        envelope_scope="snapshot-detach",
    )
    restored = codec.loads(codec.dumps(dict(data)))
    if not isinstance(restored, dict):  # pragma: no cover - mapping input contract
        raise TypeError("Snapshot payload did not round-trip as a mapping")
    return restored


class InMemorySnapshotStore(SnapshotStore):
    """In-memory snapshot store implementation."""
    
    backend = "memory"
    
    def __init__(self, *, default_ttl_seconds: Optional[float] = None) -> None:
        self.default_ttl_seconds = default_ttl_seconds
        self._data: Dict[str, SnapshotRecord] = {}
        self._expires_at: Dict[str, float] = {}
        self._pin_restore_expires_at: Dict[str, float | None] = {}
        self._pins: Dict[str, set[str]] = {}
        self._lock = RLock()
    
    def _effective_ttl(self, ttl_seconds: Optional[float]) -> Optional[float]:
        """Get effective TTL considering default."""
        if ttl_seconds is not None:
            return float(ttl_seconds)
        if self.default_ttl_seconds is not None:
            return float(self.default_ttl_seconds)
        return None
    
    def _sweep_expired(self) -> None:
        """Remove expired snapshots."""
        if not self._expires_at:
            return
        now = time.time()
        expired = [key for key, t in self._expires_at.items() if t <= now]
        for key in expired:
            if self._pins.get(key):
                self._expires_at.pop(key, None)
                continue
            self._expires_at.pop(key, None)
            self._data.pop(key, None)
    
    def write(
        self,
        data: Mapping[str, Any],
        *,
        key: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        schema: str = "population_snapshot_v1",
        ttl_seconds: Optional[float] = None,
        write_once: bool = False,
        expected_revision: Optional[int] = None,
    ) -> SnapshotHandle:
        with self._lock:
            self._sweep_expired()
            key_text = str(key or make_snapshot_key())
            current = self._data.get(key_text)
            if write_once and current is not None:
                raise FileExistsError(f"Snapshot key is write-once: {key_text}")
            if expected_revision is not None:
                actual = 0 if current is None else int(current.revision)
                if actual != int(expected_revision):
                    raise RuntimeError(
                        f"Snapshot revision conflict for '{key_text}': "
                        f"expected={int(expected_revision)}, actual={actual}"
                    )
            revision = 1 if current is None else int(current.revision) + 1
            created_at = time.time()
            meta_payload = dict(meta or {})
            detached = _detached_snapshot_mapping(data)
            digest = snapshot_content_digest(str(schema), detached)
            ttl = self._effective_ttl(ttl_seconds)
            expires_at = (
                created_at + ttl
                if ttl is not None and ttl > 0 and not self._pins.get(key_text)
                else None
            )
            record = SnapshotRecord(
                key=key_text,
                backend=self.backend,
                schema=str(schema),
                meta=meta_payload,
                created_at=created_at,
                revision=revision,
                content_digest=digest,
                expires_at=expires_at,
                pinned=bool(self._pins.get(key_text)),
                data=detached,
            )
            self._data[key_text] = record
            if expires_at is not None:
                self._expires_at[key_text] = expires_at
            else:
                self._expires_at.pop(key_text, None)
            return SnapshotHandle(
                key=record.key,
                backend=record.backend,
                schema=record.schema,
                meta=dict(record.meta),
                created_at=record.created_at,
                revision=record.revision,
                content_digest=record.content_digest,
                expires_at=record.expires_at,
                pinned=record.pinned,
            )
    
    def read(self, key: str) -> Optional[SnapshotRecord]:
        with self._lock:
            self._sweep_expired()
            record = self._data.get(str(key))
            if record is None:
                return None
            # ``write`` validates and detaches the payload before it enters the
            # private map, and reads never expose that stored object.  Re-encoding
            # the complete payload here would make repeated reads of cumulative
            # snapshots (for example histories) quadratic in run length.  Durable
            # backends still recompute their digest on every read because their
            # bytes can be modified outside this process.
            return SnapshotRecord(
                key=record.key,
                backend=record.backend,
                schema=record.schema,
                meta=dict(record.meta),
                created_at=record.created_at,
                revision=record.revision,
                content_digest=record.content_digest,
                expires_at=record.expires_at,
                pinned=record.pinned,
                data=copy.deepcopy(record.data),
            )
    
    def delete(self, key: str) -> None:
        with self._lock:
            key_text = str(key)
            if self._pins.get(key_text):
                raise RuntimeError(f"Snapshot '{key_text}' is pinned")
            self._data.pop(key_text, None)
            self._expires_at.pop(key_text, None)
            self._pin_restore_expires_at.pop(key_text, None)

    def pin(self, key: str, *, owner: str) -> SnapshotHandle:
        with self._lock:
            key_text = str(key)
            owner_text = str(owner or "").strip()
            if not owner_text:
                raise ValueError("Snapshot pin owner must not be empty")
            record = self._data.get(key_text)
            if record is None:
                raise KeyError(f"Unknown Snapshot '{key_text}'")
            if not self._pins.get(key_text):
                self._pin_restore_expires_at[key_text] = record.expires_at
            self._pins.setdefault(key_text, set()).add(owner_text)
            self._expires_at.pop(key_text, None)
            updated = SnapshotRecord(
                **{
                    **record.__dict__,
                    "meta": dict(record.meta),
                    "data": copy.deepcopy(record.data),
                    "expires_at": None,
                    "pinned": True,
                }
            )
            self._data[key_text] = updated
            return SnapshotHandle(
                key=updated.key,
                backend=updated.backend,
                schema=updated.schema,
                meta=dict(updated.meta),
                created_at=updated.created_at,
                revision=updated.revision,
                content_digest=updated.content_digest,
                expires_at=None,
                pinned=True,
            )

    def unpin(self, key: str, *, owner: str) -> SnapshotHandle:
        with self._lock:
            key_text = str(key)
            record = self._data.get(key_text)
            if record is None:
                raise KeyError(f"Unknown Snapshot '{key_text}'")
            owners = self._pins.setdefault(key_text, set())
            owners.discard(str(owner or "").strip())
            if not owners:
                self._pins.pop(key_text, None)
            pinned = bool(owners)
            restored_expiry = (
                None
                if pinned
                else self._pin_restore_expires_at.pop(key_text, None)
            )
            updated = SnapshotRecord(
                **{
                    **record.__dict__,
                    "meta": dict(record.meta),
                    "data": copy.deepcopy(record.data),
                    "pinned": pinned,
                    "expires_at": restored_expiry if not pinned else None,
                }
            )
            self._data[key_text] = updated
            if restored_expiry is not None and not pinned:
                self._expires_at[key_text] = restored_expiry
            return SnapshotHandle(
                key=updated.key,
                backend=updated.backend,
                schema=updated.schema,
                meta=dict(updated.meta),
                created_at=updated.created_at,
                revision=updated.revision,
                content_digest=updated.content_digest,
                expires_at=updated.expires_at,
                pinned=pinned,
            )


class RedisSnapshotStore(SnapshotStore):
    """Redis-backed snapshot store."""
    
    backend = "redis"
    _ENVELOPE = "blackbase.snapshot.envelope.v1"
    
    def __init__(
        self,
        *,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "blackbase:snapshot",
        default_ttl_seconds: Optional[float] = None,
        serializer: str = "safe",
        hmac_env_var: str = "BLACKBASE_SNAPSHOT_HMAC_KEY",
        unsafe_allow_unsigned: bool = False,
        max_payload_bytes: int = 8_388_608,
    ) -> None:
        try:
            import redis  # type: ignore
        except Exception as exc:
            raise RuntimeError("RedisSnapshotStore requires `redis` package.") from exc
        self._redis = redis.from_url(redis_url)
        self._key_prefix = str(key_prefix or "blackbase:snapshot").rstrip(":")
        self.default_ttl_seconds = default_ttl_seconds
        self.serializer = str(serializer or "safe").strip().lower()
        if self.serializer not in {"safe", "pickle_signed", "pickle_unsafe"}:
            raise ValueError(f"Unsupported redis snapshot serializer: {serializer}")
        self.hmac_env_var = str(hmac_env_var or "BLACKBASE_SNAPSHOT_HMAC_KEY").strip()
        self.unsafe_allow_unsigned = bool(unsafe_allow_unsigned)
        self.max_payload_bytes = int(max_payload_bytes)
    
    def _normalize_key(self, key: str) -> str:
        """Normalize snapshot key."""
        text = str(key).strip()
        if not text:
            return ""
        prefix = f"{self._key_prefix}:"
        if text.startswith(prefix):
            return text[len(prefix):]
        prefix_slash = f"{self._key_prefix}/"
        if text.startswith(prefix_slash):
            return text[len(prefix_slash):]
        return text
    
    def _k(self, key: str) -> str:
        """Get prefixed key."""
        norm = self._normalize_key(key)
        if norm:
            return f"{self._key_prefix}:{norm}"
        return f"{self._key_prefix}:"

    def _pin_k(self, key: str) -> str:
        return f"{self._k(key)}:pins"
    
    def _effective_ttl(self, ttl_seconds: Optional[float]) -> Optional[int]:
        """Get effective TTL as integer seconds."""
        if ttl_seconds is not None:
            ttl = float(ttl_seconds)
        elif self.default_ttl_seconds is not None:
            ttl = float(self.default_ttl_seconds)
        else:
            return None
        return int(ttl) if ttl > 0 else None
    
    def _value_codec(self) -> RedisValueCodec:
        return RedisValueCodec(
            serializer=getattr(self, "serializer", "safe"),
            hmac_env_var=getattr(
                self,
                "hmac_env_var",
                "BLACKBASE_SNAPSHOT_HMAC_KEY",
            ),
            unsafe_allow_legacy_pickle=(
                bool(getattr(self, "unsafe_allow_unsigned", False))
                or getattr(self, "serializer", "safe") == "pickle_unsafe"
            ),
            max_payload_bytes=getattr(self, "max_payload_bytes", 8_388_608),
            envelope_scope="snapshot",
        )
    
    def _to_safe_obj(self, value: Any) -> Any:
        """Convert value to JSON-safe representation."""
        if isinstance(value, UnknownState):
            return {
                "__blackbase_protocol__": "UnknownState",
                "payload": self._to_safe_obj(value.to_protocol_payload()),
            }
        if isinstance(value, np.ndarray):
            return {
                "__ndarray__": value.tolist(),
                "__dtype__": str(value.dtype),
                "__shape__": list(value.shape),
            }
        if isinstance(value, np.generic):
            return value.item()
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): self._to_safe_obj(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_safe_obj(v) for v in value]
        return {"__repr__": repr(value), "__type__": value.__class__.__name__}
    
    def _from_safe_obj(self, value: Any) -> Any:
        """Restore value from JSON-safe representation."""
        if isinstance(value, dict):
            if value.get("__blackbase_protocol__") == "UnknownState":
                payload = self._from_safe_obj(value.get("payload", {}))
                if not isinstance(payload, Mapping):
                    raise ValueError("UnknownState protocol payload must be a mapping")
                return UnknownState.from_protocol_payload(payload)
            if "__ndarray__" in value:
                arr = np.asarray(value.get("__ndarray__"))
                dtype = value.get("__dtype__")
                if isinstance(dtype, str) and dtype:
                    try:
                        arr = arr.astype(dtype)
                    except Exception as exc:
                        _report_soft_error(
                            component="SnapshotStore",
                            event="redis_safe_ndarray_cast",
                            exc=exc,
                            logger=logger,
                            strict=False,
                            level="debug",
                        )
                return arr
            if "__repr__" in value and "__type__" in value:
                return str(value.get("__repr__", ""))
            return {str(k): self._from_safe_obj(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._from_safe_obj(v) for v in value]
        return value
    
    def _serialize_payload(self, payload: Dict[str, Any]) -> bytes:
        """Serialize through the shared versioned Redis value codec."""
        return self._value_codec().dumps(payload)
    
    def _deserialize_payload(self, raw: bytes) -> Optional[Dict[str, Any]]:
        """Deserialize without executing unsigned pickle data."""
        try:
            payload = self._value_codec().loads(raw)
        except RedisValueCodecError as exc:
            if self.serializer == "safe":
                legacy = self._deserialize_legacy_safe_payload(raw)
                if legacy is not None:
                    return legacy
            _report_soft_error(
                component="SnapshotStore",
                event="redis_value_decode",
                exc=exc,
                logger=logger,
                strict=False,
            )
            return None
        return payload if isinstance(payload, dict) else None

    def _deserialize_legacy_safe_payload(self, raw: bytes) -> Optional[Dict[str, Any]]:
        """Read the old JSON-safe snapshot envelope without a pickle fallback."""
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        if not isinstance(decoded, dict) or decoded.get("_snapshot_envelope") != self._ENVELOPE:
            return None
        if str(decoded.get("serializer", "")).strip().lower() != "safe":
            return None
        payload = decoded.get("payload")
        if not isinstance(payload, dict):
            return None
        restored = self._from_safe_obj(payload)
        return restored if isinstance(restored, dict) else None
    
    def write(
        self,
        data: Mapping[str, Any],
        *,
        key: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        schema: str = "population_snapshot_v1",
        ttl_seconds: Optional[float] = None,
        write_once: bool = False,
        expected_revision: Optional[int] = None,
    ) -> SnapshotHandle:
        key_text = str(key or make_snapshot_key(prefix=self._key_prefix))
        created_at = time.time()
        detached = _detached_snapshot_mapping(data)
        digest = snapshot_content_digest(str(schema), detached)
        # Reject an impossible payload before touching Redis.  Besides keeping
        # failure side-effect free, this preserves the serializer contract for
        # callers that validate payloads with an unbound store instance.
        provisional = {
            "key": key_text,
            "backend": self.backend,
            "schema": str(schema),
            "meta": dict(meta or {}),
            "created_at": created_at,
            "revision": 1,
            "content_digest": digest,
            "expires_at": None,
            "data": detached,
        }
        provisional_raw = self._serialize_payload(provisional)
        if self.max_payload_bytes > 0 and len(provisional_raw) > int(
            self.max_payload_bytes
        ):
            raise ValueError(
                "snapshot payload too large: "
                f"{len(provisional_raw)} bytes > {int(self.max_payload_bytes)} bytes"
            )
        redis_key = self._k(key_text)
        current_raw = self._redis.get(redis_key)
        current_payload = (
            self._deserialize_payload(current_raw) if current_raw is not None else None
        )
        actual_revision = (
            int(current_payload.get("revision", 1))
            if isinstance(current_payload, Mapping)
            else 0
        )
        if write_once and current_raw is not None:
            raise FileExistsError(f"Snapshot key is write-once: {key_text}")
        if expected_revision is not None and actual_revision != int(expected_revision):
            raise RuntimeError(
                f"Snapshot revision conflict for '{key_text}': "
                f"expected={int(expected_revision)}, actual={actual_revision}"
            )
        revision = actual_revision + 1
        ttl = self._effective_ttl(ttl_seconds)
        pinned = bool(self._redis.scard(self._pin_k(key_text)))
        expires_at = (
            created_at + ttl if ttl is not None and not pinned else None
        )
        payload = {
            "key": key_text,
            "backend": self.backend,
            "schema": str(schema),
            "meta": dict(meta or {}),
            "created_at": created_at,
            "revision": revision,
            "content_digest": digest,
            "expires_at": expires_at,
            "data": detached,
        }
        try:
            raw = self._serialize_payload(payload)
            if self.max_payload_bytes > 0 and len(raw) > int(self.max_payload_bytes):
                raise ValueError(
                    f"snapshot payload too large: {len(raw)} bytes > {int(self.max_payload_bytes)} bytes"
                )
        except Exception as exc:
            _report_soft_error(
                component="SnapshotStore",
                event="redis_write_serialize",
                exc=exc,
                logger=logger,
                strict=False,
            )
            raise
        if write_once:
            written = self._redis.set(
                redis_key,
                raw,
                nx=True,
                ex=(None if pinned else ttl),
            )
            if not written:
                raise FileExistsError(f"Snapshot key is write-once: {key_text}")
        elif expected_revision is not None:
            pipeline = self._redis.pipeline()
            try:
                pipeline.watch(redis_key)
                watched_raw = pipeline.get(redis_key)
                watched_payload = (
                    self._deserialize_payload(watched_raw)
                    if watched_raw is not None
                    else None
                )
                watched_revision = (
                    int(watched_payload.get("revision", 1))
                    if isinstance(watched_payload, Mapping)
                    else 0
                )
                if watched_revision != int(expected_revision):
                    raise RuntimeError(
                        f"Snapshot revision conflict for '{key_text}': "
                        f"expected={int(expected_revision)}, actual={watched_revision}"
                    )
                pipeline.multi()
                pipeline.set(redis_key, raw, ex=(None if pinned else ttl))
                pipeline.execute()
            finally:
                reset = getattr(pipeline, "reset", None)
                if callable(reset):
                    reset()
        elif ttl is None or pinned:
            self._redis.set(redis_key, raw)
        else:
            self._redis.setex(redis_key, ttl, raw)
        return SnapshotHandle(
            key=key_text,
            backend=self.backend,
            schema=str(schema),
            meta=dict(meta or {}),
            created_at=created_at,
            revision=revision,
            content_digest=digest,
            expires_at=expires_at,
            pinned=pinned,
        )
    
    def read(self, key: str) -> Optional[SnapshotRecord]:
        raw = self._redis.get(self._k(key))
        if raw is None:
            return None
        payload = self._deserialize_payload(raw)
        if not isinstance(payload, dict):
            return None
        data = dict(payload.get("data", {}) or {})
        schema = str(payload.get("schema", "population_snapshot_v1"))
        stored_digest = str(payload.get("content_digest", "") or "")
        if stored_digest and snapshot_content_digest(schema, data) != stored_digest:
            raise ValueError(f"Snapshot content digest mismatch for '{key}'")
        return SnapshotRecord(
            key=str(payload.get("key", key)),
            backend=str(payload.get("backend", self.backend)),
            schema=schema,
            meta=dict(payload.get("meta", {}) or {}),
            created_at=float(payload.get("created_at", time.time())),
            revision=int(payload.get("revision", 1) or 1),
            content_digest=stored_digest,
            expires_at=(
                None
                if payload.get("expires_at") is None
                else float(payload.get("expires_at"))
            ),
            pinned=bool(self._redis.scard(self._pin_k(key))),
            data=data,
        )
    
    def delete(self, key: str) -> None:
        if bool(self._redis.scard(self._pin_k(key))):
            raise RuntimeError(f"Snapshot '{key}' is pinned")
        self._redis.delete(self._k(key))

    def pin(self, key: str, *, owner: str) -> SnapshotHandle:
        owner_text = str(owner or "").strip()
        if not owner_text:
            raise ValueError("Snapshot pin owner must not be empty")
        if self._redis.get(self._k(key)) is None:
            raise KeyError(f"Unknown Snapshot '{key}'")
        self._redis.sadd(self._pin_k(key), owner_text)
        self._redis.persist(self._k(key))
        record = self.read(key)
        if record is None:  # pragma: no cover - guarded above
            raise KeyError(f"Unknown Snapshot '{key}'")
        return SnapshotHandle(
            key=record.key,
            backend=record.backend,
            schema=record.schema,
            meta=dict(record.meta),
            created_at=record.created_at,
            revision=record.revision,
            content_digest=record.content_digest,
            expires_at=None,
            pinned=True,
        )

    def unpin(self, key: str, *, owner: str) -> SnapshotHandle:
        self._redis.srem(self._pin_k(key), str(owner or "").strip())
        record = self.read(key)
        if record is None:
            raise KeyError(f"Unknown Snapshot '{key}'")
        still_pinned = bool(self._redis.scard(self._pin_k(key)))
        if not still_pinned and record.expires_at is not None:
            remaining = int(record.expires_at - time.time())
            if remaining <= 0:
                self._redis.delete(self._k(key))
                raise KeyError(f"Snapshot '{key}' expired after its final pin was released")
            self._redis.expire(self._k(key), remaining)
        return SnapshotHandle(
            key=record.key,
            backend=record.backend,
            schema=record.schema,
            meta=dict(record.meta),
            created_at=record.created_at,
            revision=record.revision,
            content_digest=record.content_digest,
            expires_at=record.expires_at,
            pinned=still_pinned,
        )


class FileSnapshotStore(SnapshotStore):
    """Filesystem-backed snapshot store."""
    
    backend = "file"
    
    def __init__(
        self,
        *,
        base_dir: str | os.PathLike[str] = "runs/snapshots",
        default_ttl_seconds: Optional[float] = None,
        key_prefix: str = "snapshot",
        serializer: str = "safe",
        hmac_env_var: str = "BLACKBASE_SNAPSHOT_HMAC_KEY",
        unsafe_allow_unsigned: bool = False,
        max_payload_bytes: int = 8_388_608,
    ) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl_seconds = default_ttl_seconds
        self.key_prefix = str(key_prefix or "snapshot")
        self.serializer = str(serializer or "safe").strip().lower()
        self.hmac_env_var = str(hmac_env_var or "BLACKBASE_SNAPSHOT_HMAC_KEY").strip()
        self.unsafe_allow_unsigned = bool(unsafe_allow_unsigned)
        self.max_payload_bytes = int(max_payload_bytes)
        self._value_codec()

    def _value_codec(self) -> RedisValueCodec:
        return RedisValueCodec(
            serializer=self.serializer,
            hmac_env_var=self.hmac_env_var,
            unsafe_allow_legacy_pickle=(
                bool(self.unsafe_allow_unsigned) or self.serializer == "pickle_unsafe"
            ),
            max_payload_bytes=self.max_payload_bytes,
            envelope_scope="snapshot",
        )
    
    def _effective_ttl(self, ttl_seconds: Optional[float]) -> Optional[float]:
        """Get effective TTL considering default."""
        if ttl_seconds is not None:
            return float(ttl_seconds)
        if self.default_ttl_seconds is not None:
            return float(self.default_ttl_seconds)
        return None
    
    def _normalize_key(self, key: Optional[str]) -> str:
        """Normalize snapshot key for filesystem."""
        if key:
            text = str(key).strip()
        else:
            text = make_snapshot_key(prefix=self.key_prefix)
        text = text.replace("\\", "/").lstrip("/")
        return text or make_snapshot_key(prefix=self.key_prefix)
    
    def _stem_path(self, key: Optional[str]) -> Path:
        """Get stem path for snapshot files."""
        norm = self._normalize_key(key)
        path = Path(norm)
        if path.suffix:
            path = path.with_suffix("")
        resolved = (self.base_dir / path).resolve()
        try:
            resolved.relative_to(self.base_dir)
        except ValueError as exc:
            raise ValueError(
                f"snapshot key escapes configured base_dir: {norm!r}"
            ) from exc
        return resolved
    
    def _paths(self, key: Optional[str]) -> Dict[str, Path]:
        """Get file paths for a snapshot."""
        stem = self._stem_path(key)
        return {
            "npz": stem.with_suffix(".npz"),
            "meta": stem.with_suffix(".meta.json"),
            "extras": stem.with_suffix(".extras.value"),
            "legacy_extras": stem.with_suffix(".extras.pkl"),
            "pins": stem.with_suffix(".pins.json"),
            "write_once": stem.with_suffix(".write-once"),
        }
    
    @staticmethod
    def _coerce_array(value: Any) -> Optional[np.ndarray]:
        """Coerce value to numpy array if possible."""
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (list, tuple)):
            try:
                arr = np.asarray(value)
            except Exception as exc:
                _report_soft_error(
                    component="SnapshotStore",
                    event="file_coerce_array",
                    exc=exc,
                    logger=logger,
                    strict=False,
                    level="debug",
                )
                return None
            if arr.dtype == object:
                return None
            return arr
        return None
    
    def write(
        self,
        data: Mapping[str, Any],
        *,
        key: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        schema: str = "population_snapshot_v1",
        ttl_seconds: Optional[float] = None,
        write_once: bool = False,
        expected_revision: Optional[int] = None,
    ) -> SnapshotHandle:
        created_at = time.time()
        normalized_key = self._normalize_key(key)
        paths = self._paths(normalized_key)
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)

        arrays: Dict[str, np.ndarray] = {}
        extras: Dict[str, Any] = {}
        for name, value in dict(data).items():
            if value is None:
                continue
            array = self._coerce_array(value)
            if array is None:
                extras[str(name)] = value
            else:
                arrays[str(name)] = np.asarray(array)
        raw_extras = self._value_codec().dumps(extras) if extras else None
        persisted_data = {
            **{str(name): np.asarray(value) for name, value in arrays.items()},
            **_detached_snapshot_mapping(extras),
        }
        digest = snapshot_content_digest(str(schema), persisted_data)

        transaction_id = uuid.uuid4().hex
        claim_path = paths["meta"].with_suffix(paths["meta"].suffix + ".claim")
        self._claim_file_write(claim_path, transaction_id)
        new_payloads: list[Path] = []
        committed = False
        try:
            current_meta = self._read_meta(normalized_key)
            actual_revision = (
                int(current_meta.get("revision", 1))
                if isinstance(current_meta, Mapping)
                else 0
            )
            if write_once and current_meta is not None:
                raise FileExistsError(f"Snapshot key is write-once: {normalized_key}")
            if expected_revision is not None and actual_revision != int(expected_revision):
                raise RuntimeError(
                    f"Snapshot revision conflict for '{normalized_key}': "
                    f"expected={int(expected_revision)}, actual={actual_revision}"
                )

            revision = actual_revision + 1
            stem = paths["meta"].with_suffix("")
            npz_path = stem.with_name(f"{stem.name}.r{revision}.{transaction_id}.npz")
            extras_path = stem.with_name(
                f"{stem.name}.r{revision}.{transaction_id}.extras.value"
            )
            if arrays:
                with npz_path.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
                new_payloads.append(npz_path)
            if extras:
                with extras_path.open("xb") as stream:
                    assert raw_extras is not None
                    stream.write(raw_extras)
                    stream.flush()
                    os.fsync(stream.fileno())
                new_payloads.append(extras_path)

            ttl = self._effective_ttl(ttl_seconds)
            pins = self._read_pin_owners(paths["pins"])
            expires_at = (
                created_at + ttl if ttl is not None and ttl > 0 and not pins else None
            )
            meta_payload = {
                "key": normalized_key,
                "backend": self.backend,
                "schema": str(schema),
                "meta": dict(meta or {}),
                "created_at": created_at,
                "expires_at": expires_at,
                "revision": revision,
                "content_digest": digest,
                "array_keys": sorted(arrays),
                "extra_keys": sorted(extras),
                "npz_file": npz_path.name if arrays else None,
                "extras_file": extras_path.name if extras else None,
                "extras_serializer": self.serializer,
                "transaction_id": transaction_id,
            }
            tmp_meta = paths["meta"].with_name(
                f".{paths['meta'].name}.{transaction_id}.tmp"
            )
            tmp_meta.write_text(
                json.dumps(meta_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with tmp_meta.open("r+b") as stream:
                os.fsync(stream.fileno())
            tmp_meta.replace(paths["meta"])
            committed = True
            self._cleanup_replaced_payloads(paths, current_meta, meta_payload)
            if write_once:
                paths["write_once"].write_text(
                    "blackbase.snapshot.write-once/v2",
                    encoding="utf-8",
                )
            return SnapshotHandle(
                key=normalized_key,
                backend=self.backend,
                schema=str(schema),
                meta=dict(meta or {}),
                created_at=created_at,
                revision=revision,
                content_digest=digest,
                expires_at=expires_at,
                pinned=bool(pins),
            )
        finally:
            if not committed:
                for payload_path in new_payloads:
                    try:
                        payload_path.unlink()
                    except FileNotFoundError:
                        pass
            try:
                claim_path.unlink()
            except FileNotFoundError:
                pass
    
    def _read_meta(self, key: str) -> Optional[Dict[str, Any]]:
        """Read metadata file for a snapshot."""
        paths = self._paths(key)
        if not paths["meta"].exists():
            return None
        try:
            return json.loads(paths["meta"].read_text(encoding="utf-8"))
        except Exception as exc:
            _report_soft_error(
                component="SnapshotStore",
                event="file_read_meta",
                exc=exc,
                logger=logger,
                strict=False,
            )
            return None

    @staticmethod
    def _claim_file_write(path: Path, transaction_id: str) -> None:
        payload = {
            "schema": "blackbase.snapshot.file_claim/v1",
            "transaction_id": str(transaction_id),
            "pid": os.getpid(),
            "created_at": time.time(),
        }
        for _ in range(2):
            try:
                with path.open("x", encoding="utf-8") as stream:
                    json.dump(payload, stream, sort_keys=True)
                    stream.flush()
                    os.fsync(stream.fileno())
                return
            except FileExistsError as exc:
                try:
                    current = json.loads(path.read_text(encoding="utf-8"))
                    owner_pid = int(current.get("pid", 0) or 0)
                except Exception:
                    owner_pid = 0
                if owner_pid > 0 and _process_is_alive(owner_pid):
                    raise RuntimeError(
                        f"Snapshot file write is already active: {path}"
                    ) from exc
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        raise RuntimeError(f"Could not acquire Snapshot file claim: {path}")

    @staticmethod
    def _payload_path(paths: Mapping[str, Path], value: Any, fallback: Path) -> Path:
        name = str(value or "").strip()
        if not name:
            return fallback
        if Path(name).name != name:
            raise ValueError("Snapshot payload filename must be one safe path segment")
        resolved = (paths["meta"].parent / name).resolve()
        if resolved.parent != paths["meta"].parent.resolve():
            raise ValueError("Snapshot payload filename escapes its Snapshot directory")
        return resolved

    def _cleanup_replaced_payloads(
        self,
        paths: Mapping[str, Path],
        previous: Mapping[str, Any] | None,
        current: Mapping[str, Any],
    ) -> None:
        if not previous:
            return
        for field, fallback in (
            ("npz_file", paths["npz"]),
            ("extras_file", paths["extras"]),
        ):
            old_path = self._payload_path(paths, previous.get(field), fallback)
            new_path = self._payload_path(paths, current.get(field), fallback)
            if old_path == new_path:
                continue
            try:
                old_path.unlink()
            except FileNotFoundError:
                pass
    
    def _is_expired(self, meta: Dict[str, Any]) -> bool:
        """Check if snapshot is expired."""
        expires_at = meta.get("expires_at")
        if expires_at is None:
            return False
        try:
            return float(expires_at) <= time.time()
        except Exception as exc:
            _report_soft_error(
                component="SnapshotStore",
                event="file_expiry_parse",
                exc=exc,
                logger=logger,
                strict=False,
                level="debug",
            )
            return False
    
    def read(self, key: str) -> Optional[SnapshotRecord]:
        meta = self._read_meta(key)
        if meta is None:
            return None
        if self._is_expired(meta):
            self.delete(key)
            return None
        paths = self._paths(key)
        npz_path = self._payload_path(paths, meta.get("npz_file"), paths["npz"])
        extras_path = self._payload_path(
            paths, meta.get("extras_file"), paths["extras"]
        )
        data: Dict[str, Any] = {}
        if npz_path.exists():
            try:
                with np.load(str(npz_path)) as payload:
                    for k in payload.files:
                        data[str(k)] = np.asarray(payload[k])
            except Exception as exc:
                _report_soft_error(
                    component="SnapshotStore",
                    event="file_read_npz",
                    exc=exc,
                    logger=logger,
                    strict=False,
                )
        if extras_path.exists():
            try:
                with extras_path.open("rb") as f:
                    extras = self._value_codec().loads(f.read())
                if isinstance(extras, dict):
                    data.update(extras)
            except Exception as exc:
                _report_soft_error(
                    component="SnapshotStore",
                    event="file_read_extras_value",
                    exc=exc,
                    logger=logger,
                    strict=False,
                )
        elif paths["legacy_extras"].exists():
            if not (self.serializer == "pickle_unsafe" or self.unsafe_allow_unsigned):
                _report_soft_error(
                    component="SnapshotStore",
                    event="file_legacy_pickle_blocked",
                    exc=ValueError(
                        "legacy .extras.pkl blocked; use an isolated explicit migration mode"
                    ),
                    logger=logger,
                    strict=False,
                )
            else:
                try:
                    with paths["legacy_extras"].open("rb") as f:
                        extras = pickle.load(f)
                    if isinstance(extras, dict):
                        data.update(extras)
                except Exception as exc:
                    _report_soft_error(
                        component="SnapshotStore",
                        event="file_read_legacy_extras_pickle",
                        exc=exc,
                        logger=logger,
                        strict=False,
                    )
        
        schema = str(meta.get("schema", "population_snapshot_v1"))
        stored_digest = str(meta.get("content_digest", "") or "")
        if stored_digest and snapshot_content_digest(schema, data) != stored_digest:
            raise ValueError(f"Snapshot content digest mismatch for '{key}'")
        return SnapshotRecord(
            key=str(meta.get("key", key)),
            backend=str(meta.get("backend", self.backend)),
            schema=schema,
            meta=dict(meta.get("meta", {}) or {}),
            created_at=float(meta.get("created_at", time.time())),
            revision=int(meta.get("revision", 1) or 1),
            content_digest=stored_digest,
            expires_at=(
                None
                if meta.get("expires_at") is None
                else float(meta.get("expires_at"))
            ),
            pinned=bool(self._read_pin_owners(paths["pins"])),
            data=data,
        )
    
    def delete(self, key: str) -> None:
        paths = self._paths(key)
        if self._read_pin_owners(paths["pins"]):
            raise RuntimeError(f"Snapshot '{key}' is pinned")
        meta = self._read_meta(key)
        payload_paths = set(paths.values())
        if meta is not None:
            payload_paths.add(
                self._payload_path(paths, meta.get("npz_file"), paths["npz"])
            )
            payload_paths.add(
                self._payload_path(
                    paths, meta.get("extras_file"), paths["extras"]
                )
            )
        for p in payload_paths:
            if p.exists():
                try:
                    p.unlink()
                except Exception as exc:
                    _report_soft_error(
                        component="SnapshotStore",
                        event="file_delete",
                        exc=exc,
                        logger=logger,
                        strict=False,
                        level="debug",
                    )

    @staticmethod
    def _read_pin_owners(path: Path) -> set[str]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return set()
        except Exception as exc:
            raise ValueError(f"invalid Snapshot pin record: {path}") from exc
        return {str(item) for item in list(raw or ()) if str(item)}

    def pin(self, key: str, *, owner: str) -> SnapshotHandle:
        owner_text = str(owner or "").strip()
        if not owner_text:
            raise ValueError("Snapshot pin owner must not be empty")
        record = self.read(key)
        if record is None:
            raise KeyError(f"Unknown Snapshot '{key}'")
        paths = self._paths(key)
        owners = self._read_pin_owners(paths["pins"])
        owners.add(owner_text)
        self._write_json_file(paths["pins"], sorted(owners))
        meta = self._read_meta(key)
        if meta is not None and meta.get("expires_at") is not None:
            meta["pin_restore_expires_at"] = meta.get("expires_at")
            meta["expires_at"] = None
            self._write_json_file(paths["meta"], meta)
        return SnapshotHandle(
            key=record.key,
            backend=record.backend,
            schema=record.schema,
            meta=dict(record.meta),
            created_at=record.created_at,
            revision=record.revision,
            content_digest=record.content_digest,
            expires_at=None,
            pinned=True,
        )

    def unpin(self, key: str, *, owner: str) -> SnapshotHandle:
        record = self.read(key)
        if record is None:
            raise KeyError(f"Unknown Snapshot '{key}'")
        paths = self._paths(key)
        owners = self._read_pin_owners(paths["pins"])
        owners.discard(str(owner or "").strip())
        if owners:
            self._write_json_file(paths["pins"], sorted(owners))
        else:
            try:
                paths["pins"].unlink()
            except FileNotFoundError:
                pass
            meta = self._read_meta(key)
            if meta is not None and "pin_restore_expires_at" in meta:
                meta["expires_at"] = meta.pop("pin_restore_expires_at")
                self._write_json_file(paths["meta"], meta)
                record = self.read(key)
                if record is None:
                    raise KeyError(
                        f"Snapshot '{key}' expired after its final pin was released"
                    )
        return SnapshotHandle(
            key=record.key,
            backend=record.backend,
            schema=record.schema,
            meta=dict(record.meta),
            created_at=record.created_at,
            revision=record.revision,
            content_digest=record.content_digest,
            expires_at=record.expires_at,
            pinned=bool(owners),
        )

    @staticmethod
    def _write_json_file(path: Path, payload: Any) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def create_snapshot_store(
    *,
    backend: str = "memory",
    ttl_seconds: Optional[float] = None,
    redis_url: str = "redis://localhost:6379/0",
    key_prefix: str = "blackbase:snapshot",
    base_dir: str | os.PathLike[str] = "runs/snapshots",
    serializer: str = "safe",
    hmac_env_var: str = "BLACKBASE_SNAPSHOT_HMAC_KEY",
    unsafe_allow_unsigned: bool = False,
    max_payload_bytes: int = 8_388_608,
) -> SnapshotStore:
    """Create a snapshot store instance based on backend type."""
    backend_name = str(backend or "memory").strip().lower()
    if backend_name in {"memory", "inmemory", "local"}:
        return InMemorySnapshotStore(default_ttl_seconds=ttl_seconds)
    if backend_name in {"redis"}:
        return RedisSnapshotStore(
            redis_url=redis_url,
            key_prefix=key_prefix,
            default_ttl_seconds=ttl_seconds,
            serializer=serializer,
            hmac_env_var=hmac_env_var,
            unsafe_allow_unsigned=unsafe_allow_unsigned,
            max_payload_bytes=max_payload_bytes,
        )
    if backend_name in {"file", "filesystem", "disk"}:
        return FileSnapshotStore(
            base_dir=base_dir,
            default_ttl_seconds=ttl_seconds,
            key_prefix=key_prefix,
            serializer=serializer,
            hmac_env_var=hmac_env_var,
            unsafe_allow_unsigned=unsafe_allow_unsigned,
            max_payload_bytes=max_payload_bytes,
        )
    raise ValueError(f"Unsupported snapshot store backend: {backend}")
