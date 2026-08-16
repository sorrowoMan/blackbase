"""
Context store backends for runtime context synchronization.

This module keeps context semantics unchanged while allowing the storage
backend to be switched (in-memory by default, Redis optionally).
"""

from __future__ import annotations

import pickle
import warnings
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional


class _ContextStoreABC(ABC):
    """Abstract context key-value store."""
    
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
        for key, value in values.items():
            self.set(str(key), value, ttl_seconds=ttl_seconds)

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
    
    def __init__(self, *, default_ttl_seconds: Optional[float] = None) -> None:
        self.default_ttl_seconds = default_ttl_seconds
        self._data: Dict[str, Any] = {}
        self._expires_at: Dict[str, float] = {}
    
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
        self._sweep_expired()
        return self._data.get(str(key), default)
    
    def set(self, key: str, value: Any, *, ttl_seconds: Optional[float] = None) -> None:
        self._sweep_expired()
        k = str(key)
        self._data[k] = value
        ttl = self._effective_ttl(ttl_seconds)
        if ttl is not None and ttl > 0:
            self._expires_at[k] = time.time() + ttl
        else:
            self._expires_at.pop(k, None)
    
    def delete(self, key: str) -> None:
        k = str(key)
        self._data.pop(k, None)
        self._expires_at.pop(k, None)
    
    def clear(self) -> None:
        self._data.clear()
        self._expires_at.clear()
    
    def snapshot(self) -> Dict[str, Any]:
        self._sweep_expired()
        return dict(self._data)
    
    def __iter__(self):
        self._sweep_expired()
        return iter(self._data.keys())
    
    def items(self):
        self._sweep_expired()
        return self._data.items()
    
    def keys(self):
        self._sweep_expired()
        return self._data.keys()
    
    def values(self):
        self._sweep_expired()
        return self._data.values()

    def __getitem__(self, key: str) -> Any:
        self._sweep_expired()
        return self._data[str(key)]

    def __contains__(self, key: str) -> bool:
        self._sweep_expired()
        return str(key) in self._data

    def __len__(self) -> int:
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
    """Redis-backed context store (optional dependency)."""
    
    def __init__(
        self,
        *,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "blackbase:context",
        default_ttl_seconds: Optional[float] = None,
    ) -> None:
        try:
            import redis  # type: ignore
        except Exception as exc:
            raise RuntimeError("RedisContextStore requires `redis` package.") from exc
        self._redis = redis.from_url(redis_url)
        self._key_prefix = str(key_prefix).rstrip(":")
        self.default_ttl_seconds = default_ttl_seconds
    
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
            return pickle.loads(raw)
        except Exception:
            return default
    
    def set(self, key: str, value: Any, *, ttl_seconds: Optional[float] = None) -> None:
        try:
            payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            warnings.warn(
                f"RedisContextStore failed to pickle value for key '{key}': {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        ttl = self._effective_ttl(ttl_seconds)
        redis_key = self._k(key)
        if ttl is None:
            self._redis.set(redis_key, payload)
        else:
            self._redis.setex(redis_key, ttl, payload)
    
    def delete(self, key: str) -> None:
        self._redis.delete(self._k(key))
    
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
        out: Dict[str, Any] = {}
        for raw_key in self._redis.scan_iter(match=pattern, count=200):
            try:
                key_text = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else str(raw_key)
                short_key = key_text.split(":", 1)[1] if ":" in key_text else key_text
                raw = self._redis.get(raw_key)
                if raw is None:
                    continue
                out[short_key] = pickle.loads(raw)
            except Exception:
                continue
        return out


def create_context_store(
    *,
    backend: str = "memory",
    ttl_seconds: Optional[float] = None,
    redis_url: str = "redis://localhost:6379/0",
    key_prefix: str = "blackbase:context",
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
        )
    raise ValueError(f"Unsupported context store backend: {backend}")
