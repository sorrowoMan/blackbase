from __future__ import annotations

from blackbase.context import (
    GENERIC_SNAPSHOT_SCHEMA,
    InMemorySnapshotStore,
    unwrap_snapshot_payload,
    wrap_snapshot_payload,
)
from blackbase.context.snapshot_store import RedisSnapshotStore
from blackbase.types import UnknownState
import numpy as np


def test_generic_snapshot_payload_roundtrip_has_one_logical_layer() -> None:
    store = InMemorySnapshotStore()
    payload = {"weights": [1, 2, 3]}
    handle = store.write(
        wrap_snapshot_payload(payload),
        key="model",
        schema=GENERIC_SNAPSHOT_SCHEMA,
    )

    assert unwrap_snapshot_payload(store.read(handle.key)) == payload


def test_redis_safe_codec_roundtrips_unknown_state_structure() -> None:
    store = object.__new__(RedisSnapshotStore)
    store.serializer = "safe"
    state = UnknownState(
        values=np.asarray([1.5, 2.5], dtype=np.float32),
        metadata={"source": "unit", "mask": np.asarray([1, 0], dtype=np.int8)},
    )

    raw = store._serialize_payload({"data": {"candidates": (state,)}})
    restored = store._deserialize_payload(raw)

    candidate = restored["data"]["candidates"][0]
    assert isinstance(candidate, UnknownState)
    assert np.allclose(candidate.as_array(), [1.5, 2.5])
    assert candidate.metadata["source"] == "unit"
    assert np.array_equal(candidate.metadata["mask"], [1, 0])
