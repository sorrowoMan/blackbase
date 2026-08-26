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
    context_store_serializer: str = "safe"
    context_store_hmac_env_var: str = "BLACKBASE_CONTEXT_HMAC_KEY"
    context_store_unsafe_allow_legacy_pickle: bool = False
    context_store_max_payload_bytes: int = 262_144

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
            "context_store_max_payload_bytes",
            "snapshot_store_max_payload_bytes",
            "context_inline_candidate_max_bytes",
            "runtime_context_projection_field_max_bytes",
            "runtime_context_projection_total_max_bytes",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")
        allowed_serializers = {"safe", "pickle_signed", "pickle_unsafe"}
        for field_name in (
            "context_store_serializer",
            "snapshot_store_serializer",
        ):
            serializer = str(getattr(self, field_name) or "").strip().lower()
            if serializer not in allowed_serializers:
                raise ValueError(
                    f"{field_name} must be one of {sorted(allowed_serializers)}"
                )
            object.__setattr__(self, field_name, serializer)
        if (
            self.context_store_serializer == "safe"
            and bool(self.context_store_unsafe_allow_legacy_pickle)
        ):
            raise ValueError(
                "safe Context serializer cannot enable legacy pickle migration"
            )
        if (
            self.context_store_serializer == "pickle_signed"
            and not str(self.context_store_hmac_env_var or "").strip()
        ):
            raise ValueError(
                "context_store_hmac_env_var is required for pickle_signed"
            )
        if (
            self.snapshot_store_serializer == "pickle_signed"
            and not str(self.snapshot_store_hmac_env_var or "").strip()
        ):
            raise ValueError(
                "snapshot_store_hmac_env_var is required for pickle_signed"
            )

    def as_dict(self) -> dict[str, Any]:
        """Return a detached mapping suitable for constructor projection."""

        return asdict(self)


__all__ = ["StateStoreConfig"]
