"""Configuration for shared ContextStore and SnapshotStore infrastructure."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StateStoreConfig:
    """One explicit configuration for Context, Snapshot and projection limits.

    Semantic layers may choose their own namespace and snapshot schema, but the
    storage mechanisms and lightweight-context limits belong to the shared
    substrate.
    """

    context_store_backend: str = "memory"
    context_store_ttl_seconds: float | None = None
    context_store_redis_url: str = "redis://localhost:6379/0"
    context_store_key_prefix: str = "blackbase:context"

    snapshot_store_backend: str = "memory"
    snapshot_store_ttl_seconds: float | None = None
    snapshot_store_redis_url: str = "redis://localhost:6379/0"
    snapshot_store_key_prefix: str = "blackbase:snapshot"
    snapshot_store_dir: str | None = None
    snapshot_store_serializer: str = "safe"
    snapshot_store_hmac_env_var: str = "BLACKBASE_SNAPSHOT_HMAC_KEY"
    snapshot_store_unsafe_allow_unsigned: bool = False
    snapshot_store_max_payload_bytes: int = 8_388_608

    context_inline_candidate_max_bytes: int = 4_096
    runtime_context_projection_field_max_bytes: int = 4_096
    runtime_context_projection_total_max_bytes: int = 32_768
    snapshot_schema: str = "generic_snapshot_v1"

    def __post_init__(self) -> None:
        for field_name in (
            "snapshot_store_max_payload_bytes",
            "context_inline_candidate_max_bytes",
            "runtime_context_projection_field_max_bytes",
            "runtime_context_projection_total_max_bytes",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")

    def as_dict(self) -> dict[str, Any]:
        """Return a detached mapping suitable for constructor projection."""

        return asdict(self)


__all__ = ["StateStoreConfig"]
