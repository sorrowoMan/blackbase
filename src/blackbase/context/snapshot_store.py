"""
Snapshot store backends for large runtime artifacts (population/objectives/etc).

Snapshot stores keep large objects out of context while still providing
stable references for replay, inspection, and bias usage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import json
import logging
import os
import pickle
import time
import uuid
import hashlib
import hmac

import numpy as np

from ..types import UnknownState

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


@dataclass(frozen=True)
class SnapshotRecord(SnapshotHandle):
    """Complete snapshot record with data."""
    
    data: Dict[str, Any]


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


class InMemorySnapshotStore(SnapshotStore):
    """In-memory snapshot store implementation."""
    
    backend = "memory"
    
    def __init__(self, *, default_ttl_seconds: Optional[float] = None) -> None:
        self.default_ttl_seconds = default_ttl_seconds
        self._data: Dict[str, SnapshotRecord] = {}
        self._expires_at: Dict[str, float] = {}
    
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
    ) -> SnapshotHandle:
        self._sweep_expired()
        key_text = str(key or make_snapshot_key())
        created_at = time.time()
        meta_payload = dict(meta or {})
        record = SnapshotRecord(
            key=key_text,
            backend=self.backend,
            schema=str(schema),
            meta=meta_payload,
            created_at=created_at,
            data=dict(data),
        )
        self._data[key_text] = record
        ttl = self._effective_ttl(ttl_seconds)
        if ttl is not None and ttl > 0:
            self._expires_at[key_text] = created_at + ttl
        else:
            self._expires_at.pop(key_text, None)
        return SnapshotHandle(
            key=record.key,
            backend=record.backend,
            schema=record.schema,
            meta=record.meta,
            created_at=record.created_at,
        )
    
    def read(self, key: str) -> Optional[SnapshotRecord]:
        self._sweep_expired()
        return self._data.get(str(key))
    
    def delete(self, key: str) -> None:
        key_text = str(key)
        self._data.pop(key_text, None)
        self._expires_at.pop(key_text, None)


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
    
    def _effective_ttl(self, ttl_seconds: Optional[float]) -> Optional[int]:
        """Get effective TTL as integer seconds."""
        if ttl_seconds is not None:
            ttl = float(ttl_seconds)
        elif self.default_ttl_seconds is not None:
            ttl = float(self.default_ttl_seconds)
        else:
            return None
        return int(ttl) if ttl > 0 else None
    
    def _get_hmac_key(self) -> Optional[bytes]:
        """Get HMAC key from environment variable."""
        raw = os.environ.get(self.hmac_env_var)
        if raw is None:
            return None
        key = str(raw).encode("utf-8")
        return key if key else None
    
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
        """Serialize payload based on configured serializer."""
        if self.serializer == "safe":
            envelope = {
                "_snapshot_envelope": self._ENVELOPE,
                "serializer": "safe",
                "payload": self._to_safe_obj(payload),
            }
            return json.dumps(envelope, ensure_ascii=False).encode("utf-8")
        
        if self.serializer == "pickle_signed":
            key = self._get_hmac_key()
            if key is None:
                raise ValueError(
                    f"snapshot serializer=pickle_signed requires HMAC key in env var: {self.hmac_env_var}"
                )
            payload_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            mac = hmac.new(key, payload_bytes, hashlib.sha256).hexdigest()
            envelope = {
                "_snapshot_envelope": self._ENVELOPE,
                "serializer": "pickle_signed",
                "payload": payload,
                "hmac_sha256": mac,
            }
            return pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
        
        envelope = {
            "_snapshot_envelope": self._ENVELOPE,
            "serializer": "pickle_unsafe",
            "payload": payload,
            "hmac_sha256": None,
        }
        return pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize_payload(self, raw: bytes) -> Optional[Dict[str, Any]]:
        """Deserialize payload from bytes."""
        if self.serializer == "safe":
            try:
                decoded = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                _report_soft_error(
                    component="SnapshotStore",
                    event="redis_safe_json_decode",
                    exc=exc,
                    logger=logger,
                    strict=False,
                )
                return None
            if not isinstance(decoded, dict):
                return None
            payload = decoded.get("payload")
            if not isinstance(payload, dict):
                return None
            restored = self._from_safe_obj(payload)
            return restored if isinstance(restored, dict) else None
        
        try:
            decoded = pickle.loads(raw)
        except Exception as exc:
            _report_soft_error(
                component="SnapshotStore",
                event="redis_pickle_decode",
                exc=exc,
                logger=logger,
                strict=False,
            )
            return None
        
        if not isinstance(decoded, dict):
            return None
        serializer = str(decoded.get("serializer", "")).strip().lower()
        payload = decoded.get("payload")
        mac = decoded.get("hmac_sha256")
        
        if "_snapshot_envelope" not in decoded:
            if self.serializer == "pickle_signed" and not self.unsafe_allow_unsigned:
                _report_soft_error(
                    component="SnapshotStore",
                    event="redis_pickle_unsigned_blocked",
                    exc=ValueError("unsigned legacy pickle snapshot blocked"),
                    logger=logger,
                    strict=False,
                )
                return None
            return decoded
        
        if not isinstance(payload, dict):
            return None
        if self.serializer == "pickle_signed":
            key = self._get_hmac_key()
            if key is None:
                _report_soft_error(
                    component="SnapshotStore",
                    event="redis_pickle_signed_missing_key",
                    exc=ValueError("missing snapshot HMAC key"),
                    logger=logger,
                    strict=False,
                )
                return None
            if serializer != "pickle_signed":
                if not self.unsafe_allow_unsigned:
                    _report_soft_error(
                        component="SnapshotStore",
                        event="redis_pickle_signed_unsigned_payload",
                        exc=ValueError("unsigned snapshot blocked"),
                        logger=logger,
                        strict=False,
                    )
                    return None
            else:
                expected = hmac.new(
                    key,
                    pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL),
                    hashlib.sha256,
                ).hexdigest()
                if not hmac.compare_digest(str(mac or ""), expected):
                    _report_soft_error(
                        component="SnapshotStore",
                        event="redis_pickle_signed_hmac_mismatch",
                        exc=ValueError("snapshot HMAC verification failed"),
                        logger=logger,
                        strict=False,
                    )
                    return None
        return payload
    
    def write(
        self,
        data: Mapping[str, Any],
        *,
        key: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        schema: str = "population_snapshot_v1",
        ttl_seconds: Optional[float] = None,
    ) -> SnapshotHandle:
        key_text = str(key or make_snapshot_key(prefix=self._key_prefix))
        created_at = time.time()
        payload = {
            "key": key_text,
            "backend": self.backend,
            "schema": str(schema),
            "meta": dict(meta or {}),
            "created_at": created_at,
            "data": dict(data),
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
            return SnapshotHandle(
                key=key_text,
                backend=self.backend,
                schema=str(schema),
                meta=dict(meta or {}),
                created_at=created_at,
            )
        ttl = self._effective_ttl(ttl_seconds)
        redis_key = self._k(key_text)
        if ttl is None:
            self._redis.set(redis_key, raw)
        else:
            self._redis.setex(redis_key, ttl, raw)
        return SnapshotHandle(
            key=key_text,
            backend=self.backend,
            schema=str(schema),
            meta=dict(meta or {}),
            created_at=created_at,
        )
    
    def read(self, key: str) -> Optional[SnapshotRecord]:
        raw = self._redis.get(self._k(key))
        if raw is None:
            return None
        payload = self._deserialize_payload(raw)
        if not isinstance(payload, dict):
            return None
        return SnapshotRecord(
            key=str(payload.get("key", key)),
            backend=str(payload.get("backend", self.backend)),
            schema=str(payload.get("schema", "population_snapshot_v1")),
            meta=dict(payload.get("meta", {}) or {}),
            created_at=float(payload.get("created_at", time.time())),
            data=dict(payload.get("data", {}) or {}),
        )
    
    def delete(self, key: str) -> None:
        self._redis.delete(self._k(key))


class FileSnapshotStore(SnapshotStore):
    """Filesystem-backed snapshot store."""
    
    backend = "file"
    
    def __init__(
        self,
        *,
        base_dir: str | os.PathLike[str] = "runs/snapshots",
        default_ttl_seconds: Optional[float] = None,
        key_prefix: str = "snapshot",
    ) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl_seconds = default_ttl_seconds
        self.key_prefix = str(key_prefix or "snapshot")
    
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
        return (self.base_dir / path).resolve()
    
    def _paths(self, key: Optional[str]) -> Dict[str, Path]:
        """Get file paths for a snapshot."""
        stem = self._stem_path(key)
        return {
            "npz": stem.with_suffix(".npz"),
            "meta": stem.with_suffix(".meta.json"),
            "extras": stem.with_suffix(".extras.pkl"),
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
    ) -> SnapshotHandle:
        created_at = time.time()
        paths = self._paths(key)
        for p in paths.values():
            p.parent.mkdir(parents=True, exist_ok=True)
        
        arrays: Dict[str, np.ndarray] = {}
        extras: Dict[str, Any] = {}
        for k, v in dict(data).items():
            if v is None:
                continue
            arr = self._coerce_array(v)
            if arr is not None:
                arrays[str(k)] = np.asarray(arr)
            else:
                extras[str(k)] = v
        
        if arrays:
            tmp_npz = paths["npz"].with_suffix(paths["npz"].suffix + ".tmp")
            with tmp_npz.open("wb") as f:
                np.savez_compressed(f, **arrays)
            tmp_npz.replace(paths["npz"])
        
        if extras:
            tmp_extra = paths["extras"].with_suffix(paths["extras"].suffix + ".tmp")
            with tmp_extra.open("wb") as f:
                pickle.dump(extras, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_extra.replace(paths["extras"])
        else:
            if paths["extras"].exists():
                try:
                    paths["extras"].unlink()
                except Exception as exc:
                    _report_soft_error(
                        component="SnapshotStore",
                        event="file_write_cleanup_extras",
                        exc=exc,
                        logger=logger,
                        strict=False,
                        level="debug",
                    )
        
        ttl = self._effective_ttl(ttl_seconds)
        meta_payload = {
            "key": self._normalize_key(key),
            "backend": self.backend,
            "schema": str(schema),
            "meta": dict(meta or {}),
            "created_at": created_at,
            "expires_at": (created_at + ttl) if ttl is not None and ttl > 0 else None,
            "array_keys": sorted(arrays.keys()),
            "extra_keys": sorted(extras.keys()),
        }
        tmp_meta = paths["meta"].with_suffix(paths["meta"].suffix + ".tmp")
        tmp_meta.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_meta.replace(paths["meta"])
        
        key_text = meta_payload["key"]
        return SnapshotHandle(
            key=key_text,
            backend=self.backend,
            schema=str(schema),
            meta=dict(meta or {}),
            created_at=created_at,
        )
    
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
        data: Dict[str, Any] = {}
        if paths["npz"].exists():
            try:
                with np.load(str(paths["npz"])) as payload:
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
        if paths["extras"].exists():
            try:
                with paths["extras"].open("rb") as f:
                    extras = pickle.load(f)
                if isinstance(extras, dict):
                    data.update(extras)
            except Exception as exc:
                _report_soft_error(
                    component="SnapshotStore",
                    event="file_read_extras_pickle",
                    exc=exc,
                    logger=logger,
                    strict=False,
                )
        
        return SnapshotRecord(
            key=str(meta.get("key", key)),
            backend=str(meta.get("backend", self.backend)),
            schema=str(meta.get("schema", "population_snapshot_v1")),
            meta=dict(meta.get("meta", {}) or {}),
            created_at=float(meta.get("created_at", time.time())),
            data=data,
        )
    
    def delete(self, key: str) -> None:
        paths = self._paths(key)
        for p in paths.values():
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
        )
    raise ValueError(f"Unsupported snapshot store backend: {backend}")
