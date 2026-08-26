"""
Context store backends for runtime context synchronization.

This module keeps context semantics unchanged while allowing the storage
backend to be switched (in-memory by default, Redis optionally).
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Mapping, Optional

from .redis_codec import RedisValueCodec, RedisValueCodecError
from .value_isolation import detach_context_value


class _ContextStoreABC(ABC):
    """Abstract context key-value store."""

    supports_atomic_patch: bool = False
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the store."""
        raise NotImplementedError
    
    @abstractmethod
    def set(self, key: str, value: Any, *, ttl_seconds: Optional[float] = None) -> None:
        """Set a value in the store with optional TTL."""
        raise NotImplementedError
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a key from the store."""
        raise NotImplementedError
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all keys from the store."""
        raise NotImplementedError
    
    @abstractmethod
    def snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of all key-value pairs."""
        raise NotImplementedError
    
    def update(self, values: Dict[str, Any], *, ttl_seconds: Optional[float] = None) -> None:
        """Update multiple values at once."""
        if self.supports_atomic_patch:
            self.apply_patch(values, ttl_seconds=ttl_seconds)
            return
        for key, value in values.items():
            self.set(str(key), value, ttl_seconds=ttl_seconds)

    def apply_patch(
        self,
        values: Mapping[str, Any],
        *,
        delete_keys: Iterable[str] = (),
        ttl_seconds: Optional[float] = None,
    ) -> None:
        """Atomically apply a set/delete patch when the backend supports it.

        Custom stores must override this method and set
        ``supports_atomic_patch = True`` before callers may rely on atomic
        visibility.  The built-in memory and Redis backends provide that
        guarantee.
        """

        raise NotImplementedError(
            f"{type(self).__name__} does not support atomic context patches"
        )

    def __getitem__(self, key: str) -> Any:
        missing = object()
        value = self.get(str(key), missing)
        if value is missing:
            raise KeyError(str(key))
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(str(key), value)

    def __delitem__(self, key: str) -> None:
        if str(key) not in self:
            raise KeyError(str(key))
        self.delete(str(key))

    def __iter__(self):
        return iter(self.snapshot())

    def __len__(self) -> int:
        return len(self.snapshot())

    def __contains__(self, key: object) -> bool:
        missing = object()
        return self.get(str(key), missing) is not missing

    def items(self):
        return self.snapshot().items()

    def keys(self):
        return self.snapshot().keys()

    def values(self):
        return self.snapshot().values()


class InMemoryContextStore(_ContextStoreABC):
    """Default context store backend - in-memory implementation."""

    supports_atomic_patch = True
    
    def __init__(self, *, default_ttl_seconds: Optional[float] = None) -> None:
        self.default_ttl_seconds = default_ttl_seconds
        self._data: Dict[str, Any] = {}
        self._expires_at: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def _effective_ttl(self, ttl_seconds: Optional[float]) -> Optional[float]:
        """Get effective TTL considering default."""
        if ttl_seconds is not None:
            return float(ttl_seconds)
        if self.default_ttl_seconds is not None:
            return float(self.default_ttl_seconds)
        return None
    
    def _sweep_expired(self) -> None:
        """Remove expired entries."""
        if not self._expires_at:
            return
        now = time.time()
        expired = [key for key, t in self._expires_at.items() if t <= now]
        for key in expired:
            self._expires_at.pop(key, None)
            self._data.pop(key, None)
    
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            self._sweep_expired()
            return self._data.get(str(key), default)
    
    def set(self, key: str, value: Any, *, ttl_seconds: Optional[float] = None) -> None:
        with self._lock:
            self._sweep_expired()
            k = str(key)
            self._data[k] = value
            ttl = self._effective_ttl(ttl_seconds)
            if ttl is not None and ttl > 0:
                self._expires_at[k] = time.time() + ttl
            else:
                self._expires_at.pop(k, None)
    
    def delete(self, key: str) -> None:
        with self._lock:
            k = str(key)
            self._data.pop(k, None)
            self._expires_at.pop(k, None)
    
    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._expires_at.clear()
    
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            self._sweep_expired()
            return dict(self._data)

    def apply_patch(
        self,
        values: Mapping[str, Any],
        *,
        delete_keys: Iterable[str] = (),
        ttl_seconds: Optional[float] = None,
    ) -> None:
        normalized_values = {str(key): value for key, value in values.items()}
        normalized_deletes = tuple(str(key) for key in delete_keys)
        ttl = self._effective_ttl(ttl_seconds)
        expires_at = time.time() + ttl if ttl is not None and ttl > 0 else None
        with self._lock:
            self._sweep_expired()
            next_data = dict(self._data)
            next_expiry = dict(self._expires_at)
            for key in normalized_deletes:
                next_data.pop(key, None)
                next_expiry.pop(key, None)
            for key, value in normalized_values.items():
                next_data[key] = value
                if expires_at is None:
                    next_expiry.pop(key, None)
                else:
                    next_expiry[key] = expires_at
            self._data = next_data
            self._expires_at = next_expiry
    
    def __iter__(self):
        return iter(self.snapshot())
    
    def items(self):
        return self.snapshot().items()
    
    def keys(self):
        return self.snapshot().keys()
    
    def values(self):
        return self.snapshot().values()

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            self._sweep_expired()
            return self._data[str(key)]

    def __contains__(self, key: str) -> bool:
        with self._lock:
            self._sweep_expired()
            return str(key) in self._data

    def __len__(self) -> int:
        with self._lock:
            self._sweep_expired()
            return len(self._data)


class ContextStore(InMemoryContextStore):
    """
    Default context store - in-memory implementation.
    
    This is the primary class that users should use for context storage.
    For other backends (like Redis), use create_context_store().
    """
    pass


class RedisContextStore(_ContextStoreABC):
    """Redis-backed context store with a versioned, fail-closed value codec."""

    supports_atomic_patch = True
    
    def __init__(
        self,
        *,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "blackbase:context",
        default_ttl_seconds: Optional[float] = None,
        serializer: str = "safe",
        hmac_env_var: str = "BLACKBASE_CONTEXT_HMAC_KEY",
        unsafe_allow_legacy_pickle: bool = False,
        max_payload_bytes: int = 262_144,
    ) -> None:
        try:
            import redis  # type: ignore
        except Exception as exc:
            raise RuntimeError("RedisContextStore requires `redis` package.") from exc
        self._redis = redis.from_url(redis_url)
        self._key_prefix = str(key_prefix).rstrip(":")
        self.default_ttl_seconds = default_ttl_seconds
        self.serializer = str(serializer or "safe").strip().lower()
        self.hmac_env_var = str(hmac_env_var or "BLACKBASE_CONTEXT_HMAC_KEY").strip()
        self.unsafe_allow_legacy_pickle = bool(unsafe_allow_legacy_pickle)
        self.max_payload_bytes = int(max_payload_bytes)
        if self.serializer == "safe" and self.unsafe_allow_legacy_pickle:
            raise ValueError(
                "serializer='safe' cannot enable unsafe_allow_legacy_pickle; "
                "use an explicit pickle_unsafe migration store"
            )
        self._codec = RedisValueCodec(
            serializer=self.serializer,
            hmac_env_var=self.hmac_env_var,
            unsafe_allow_legacy_pickle=self.unsafe_allow_legacy_pickle,
            max_payload_bytes=self.max_payload_bytes,
            envelope_scope="context",
        )
    
    def _k(self, key: str) -> str:
        """Get prefixed key."""
        return f"{self._key_prefix}:{str(key)}"
    
    def _effective_ttl(self, ttl_seconds: Optional[float]) -> Optional[int]:
        """Get effective TTL as integer seconds."""
        if ttl_seconds is not None:
            ttl = float(ttl_seconds)
        elif self.default_ttl_seconds is not None:
            ttl = float(self.default_ttl_seconds)
        else:
            return None
        return int(ttl) if ttl > 0 else None
    
    def get(self, key: str, default: Any = None) -> Any:
        raw = self._redis.get(self._k(key))
        if raw is None:
            return default
        try:
            return self._codec.loads(raw)
        except RedisValueCodecError as exc:
            raise RedisValueCodecError(
                f"failed to decode Redis context key {str(key)!r}"
            ) from exc
    
    def set(self, key: str, value: Any, *, ttl_seconds: Optional[float] = None) -> None:
        try:
            payload = self._serialize_context_value(value, key=str(key))
        except Exception as exc:
            raise ValueError(
                f"RedisContextStore failed to serialize value for key {str(key)!r}"
            ) from exc
        ttl = self._effective_ttl(ttl_seconds)
        redis_key = self._k(key)
        if ttl is None:
            self._redis.set(redis_key, payload)
        else:
            self._redis.setex(redis_key, ttl, payload)
    
    def delete(self, key: str) -> None:
        self._redis.delete(self._k(key))

    def apply_patch(
        self,
        values: Mapping[str, Any],
        *,
        delete_keys: Iterable[str] = (),
        ttl_seconds: Optional[float] = None,
    ) -> None:
        normalized_values = {str(key): value for key, value in values.items()}
        normalized_deletes = tuple(str(key) for key in delete_keys)
        serialized: Dict[str, bytes] = {}
        for key, value in normalized_values.items():
            try:
                serialized[key] = self._serialize_context_value(value, key=key)
            except Exception as exc:
                raise ValueError(
                    f"RedisContextStore failed to serialize value for key {key!r}"
                ) from exc

        ttl = self._effective_ttl(ttl_seconds)
        pipeline = self._redis.pipeline(transaction=True)
        for key in normalized_deletes:
            pipeline.delete(self._k(key))
        for key, payload in serialized.items():
            redis_key = self._k(key)
            if ttl is None:
                pipeline.set(redis_key, payload)
            else:
                pipeline.setex(redis_key, ttl, payload)
        pipeline.execute()

    def _serialize_context_value(self, value: Any, *, key: str) -> bytes:
        detached = detach_context_value(
            value,
            path=f"redis_context[{key!r}]",
        )
        return self._codec.dumps(detached)
    
    def clear(self) -> None:
        pattern = f"{self._key_prefix}:*"
        keys: Iterable[Any] = self._redis.scan_iter(match=pattern, count=200)
        pipeline = self._redis.pipeline(transaction=False)
        has_data = False
        for k in keys:
            pipeline.delete(k)
            has_data = True
        if has_data:
            pipeline.execute()
    
    def snapshot(self) -> Dict[str, Any]:
        pattern = f"{self._key_prefix}:*"
        prefix = f"{self._key_prefix}:"
        out: Dict[str, Any] = {}
        for raw_key in self._redis.scan_iter(match=pattern, count=200):
            try:
                key_text = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else str(raw_key)
                if not key_text.startswith(prefix):
                    continue
                short_key = key_text[len(prefix):]
                raw = self._redis.get(raw_key)
                if raw is None:
                    continue
                out[short_key] = self._codec.loads(raw)
            except RedisValueCodecError as exc:
                raise RedisValueCodecError(
                    f"failed to decode Redis context key {short_key!r}"
                ) from exc
        return out


def create_context_store(
    *,
    backend: str = "memory",
    ttl_seconds: Optional[float] = None,
    redis_url: str = "redis://localhost:6379/0",
    key_prefix: str = "blackbase:context",
    serializer: str = "safe",
    hmac_env_var: str = "BLACKBASE_CONTEXT_HMAC_KEY",
    unsafe_allow_legacy_pickle: bool = False,
    max_payload_bytes: int = 262_144,
) -> _ContextStoreABC:
    """Create a context store instance based on backend type."""
    backend_name = str(backend or "memory").strip().lower()
    if backend_name in {"memory", "inmemory", "local"}:
        return InMemoryContextStore(default_ttl_seconds=ttl_seconds)
    if backend_name in {"redis"}:
        return RedisContextStore(
            redis_url=redis_url,
            key_prefix=key_prefix,
            default_ttl_seconds=ttl_seconds,
            serializer=serializer,
            hmac_env_var=hmac_env_var,
            unsafe_allow_legacy_pickle=unsafe_allow_legacy_pickle,
            max_payload_bytes=max_payload_bytes,
        )
    raise ValueError(f"Unsupported context store backend: {backend}")
