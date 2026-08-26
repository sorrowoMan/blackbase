from __future__ import annotations

import base64
import json
import pickle

import numpy as np
import pytest

from blackbase.context import (
    REDIS_VALUE_ENVELOPE,
    RedisSnapshotStore,
    RedisValueCodec,
    RedisValueCodecError,
)
from blackbase.types import UnknownState


_PICKLE_EXECUTED = False


def _mark_pickle_executed():
    global _PICKLE_EXECUTED
    _PICKLE_EXECUTED = True
    return {"executed": True}


class _MaliciousPickle:
    def __reduce__(self):
        return (_mark_pickle_executed, ())


def test_safe_codec_roundtrips_unknown_state_without_type_degradation() -> None:
    codec = RedisValueCodec(serializer="safe", envelope_scope="snapshot")
    state = UnknownState(
        values=np.asarray([1.25, 2.5], dtype=np.float32),
        metadata={"mask": np.asarray([1, 0], dtype=np.int8), "roles": ("a", "b")},
    )

    restored = codec.loads(codec.dumps({"states": (state,)}))

    assert isinstance(restored["states"], tuple)
    restored_state = restored["states"][0]
    assert isinstance(restored_state, UnknownState)
    assert np.array_equal(restored_state.as_array(), state.as_array())
    assert np.array_equal(restored_state.metadata["mask"], [1, 0])
    assert restored_state.metadata["roles"] == ("a", "b")


def test_signed_pickle_verifies_hmac_before_deserialization(monkeypatch) -> None:
    global _PICKLE_EXECUTED
    _PICKLE_EXECUTED = False
    monkeypatch.setenv("BLACKBASE_TEST_CODEC_HMAC", "test-secret")
    codec = RedisValueCodec(
        serializer="pickle_signed",
        hmac_env_var="BLACKBASE_TEST_CODEC_HMAC",
        envelope_scope="snapshot",
    )
    malicious = pickle.dumps(_MaliciousPickle(), protocol=pickle.HIGHEST_PROTOCOL)
    raw = json.dumps(
        {
            "_blackbase_envelope": REDIS_VALUE_ENVELOPE,
            "scope": "snapshot",
            "serializer": "pickle_signed",
            "payload_b64": base64.b64encode(malicious).decode("ascii"),
            "hmac_sha256": "0" * 64,
        }
    ).encode("utf-8")

    with pytest.raises(RedisValueCodecError, match="HMAC verification failed"):
        codec.loads(raw)

    assert _PICKLE_EXECUTED is False


def test_signed_pickle_uses_json_outer_envelope_and_roundtrips(monkeypatch) -> None:
    monkeypatch.setenv("BLACKBASE_TEST_CODEC_HMAC", "test-secret")
    codec = RedisValueCodec(
        serializer="pickle_signed",
        hmac_env_var="BLACKBASE_TEST_CODEC_HMAC",
        envelope_scope="context",
    )

    raw = codec.dumps({"value": (1, 2, 3)})
    outer = json.loads(raw.decode("utf-8"))

    assert outer["_blackbase_envelope"] == REDIS_VALUE_ENVELOPE
    assert outer["serializer"] == "pickle_signed"
    assert "payload_b64" in outer and "hmac_sha256" in outer
    assert codec.loads(raw) == {"value": (1, 2, 3)}


def test_safe_codec_rejects_unsigned_legacy_pickle_without_execution() -> None:
    global _PICKLE_EXECUTED
    _PICKLE_EXECUTED = False
    codec = RedisValueCodec(serializer="safe", envelope_scope="context")

    with pytest.raises(RedisValueCodecError, match="valid JSON envelope"):
        codec.loads(pickle.dumps(_MaliciousPickle()))

    assert _PICKLE_EXECUTED is False


def test_snapshot_store_reads_legacy_json_safe_envelope() -> None:
    store = object.__new__(RedisSnapshotStore)
    store.serializer = "safe"
    store.hmac_env_var = "BLACKBASE_TEST_UNUSED_HMAC"
    store.unsafe_allow_unsigned = False
    store.max_payload_bytes = 8_388_608
    legacy = json.dumps(
        {
            "_snapshot_envelope": store._ENVELOPE,
            "serializer": "safe",
            "payload": {
                "key": "legacy",
                "backend": "redis",
                "schema": "legacy/v1",
                "meta": {},
                "created_at": 1.0,
                "data": {"array": {"__ndarray__": [1, 2], "__dtype__": "int64"}},
            },
        }
    ).encode("utf-8")

    restored = store._deserialize_payload(legacy)

    assert restored is not None
    assert np.array_equal(restored["data"]["array"], [1, 2])


def test_snapshot_signed_pickle_integration_uses_shared_codec(monkeypatch) -> None:
    monkeypatch.setenv("BLACKBASE_TEST_SNAPSHOT_HMAC", "snapshot-secret")
    store = object.__new__(RedisSnapshotStore)
    store.serializer = "pickle_signed"
    store.hmac_env_var = "BLACKBASE_TEST_SNAPSHOT_HMAC"
    store.unsafe_allow_unsigned = False
    store.max_payload_bytes = 8_388_608
    payload = {"key": "signed", "data": {"values": (1, 2)}}

    raw = store._serialize_payload(payload)

    assert raw.startswith(b"{")
    assert store._deserialize_payload(raw) == payload
