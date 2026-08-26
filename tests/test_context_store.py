from __future__ import annotations

import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from blackbase.context import (
    ContextStore,
    RedisContextStore,
    RedisValueCodecError,
    StateStoreConfig,
)
from blackbase.resources import DataRef
from blackbase.state_ref import StateRef


_PICKLE_EXECUTED = False


def _mark_pickle_executed():
    global _PICKLE_EXECUTED
    _PICKLE_EXECUTED = True
    return "executed"


class _MaliciousPickle:
    def __reduce__(self):
        return (_mark_pickle_executed, ())


class _FakeRedisPipeline:
    def __init__(self, owner: "_FakeRedis", *, transaction: bool) -> None:
        self.owner = owner
        self.transaction = bool(transaction)
        self.commands = []

    def delete(self, key):
        self.commands.append(("delete", str(key), None))
        return self

    def set(self, key, value):
        self.commands.append(("set", str(key), value))
        return self

    def setex(self, key, ttl, value):
        self.commands.append(("setex", str(key), (int(ttl), value)))
        return self

    def execute(self):
        for operation, key, payload in self.commands:
            if operation == "delete":
                self.owner.values.pop(key, None)
            elif operation == "set":
                self.owner.values[key] = payload
            else:
                self.owner.values[key] = payload[1]
        self.owner.executed_pipelines.append(self)
        return [True] * len(self.commands)


class _FakeRedis:
    def __init__(self) -> None:
        self.values = {}
        self.pipeline_requests = []
        self.executed_pipelines = []

    def pipeline(self, transaction=True):
        pipeline = _FakeRedisPipeline(self, transaction=transaction)
        self.pipeline_requests.append(pipeline)
        return pipeline

    def scan_iter(self, *, match, count):
        del count
        prefix = str(match).removesuffix("*")
        return iter(
            key.encode("utf-8")
            for key in sorted(self.values)
            if key.startswith(prefix)
        )

    def get(self, key):
        normalized = key.decode("utf-8") if isinstance(key, bytes) else str(key)
        return self.values.get(normalized)


def test_state_store_config_validates_limits_and_returns_detached_payload() -> None:
    config = StateStoreConfig(
        context_store_backend="redis",
        context_inline_candidate_max_bytes=512,
        runtime_context_projection_total_max_bytes=4096,
    )

    payload = config.as_dict()
    payload["context_store_backend"] = "memory"

    assert config.context_store_backend == "redis"
    assert config.context_inline_candidate_max_bytes == 512
    with pytest.raises(ValueError, match="must be positive"):
        StateStoreConfig(snapshot_store_max_payload_bytes=0)
    with pytest.raises(ValueError, match="must be one of"):
        StateStoreConfig(context_store_serializer="unknown")
    with pytest.raises(ValueError, match="cannot enable legacy pickle"):
        StateStoreConfig(
            context_store_serializer="safe",
            context_store_unsafe_allow_legacy_pickle=True,
        )


def test_in_memory_context_patch_sets_and_deletes_as_one_backend_operation() -> None:
    store = ContextStore()
    store.update({"best_x": [1.0], "best_objective": 10.0, "untouched": True})

    store.apply_patch(
        {"best_candidate_ref": "snapshot://new", "best_objective": 2.0},
        delete_keys=("best_x",),
    )

    assert store.supports_atomic_patch is True
    assert store.snapshot() == {
        "best_candidate_ref": "snapshot://new",
        "best_objective": 2.0,
        "untouched": True,
    }


def test_redis_context_patch_pre_serializes_before_transaction(monkeypatch) -> None:
    fake = _FakeRedis()
    monkeypatch.setitem(
        sys.modules,
        "redis",
        SimpleNamespace(from_url=lambda _url: fake),
    )
    store = RedisContextStore(redis_url="redis://test")
    fake.values[store._k("old")] = store._codec.dumps("old")

    store.apply_patch(
        {"new": {"value": 3}},
        delete_keys=("old",),
    )

    assert len(fake.pipeline_requests) == 1
    assert fake.pipeline_requests[0].transaction is True
    assert store._k("old") not in fake.values
    assert store._codec.loads(fake.values[store._k("new")]) == {"value": 3}


def test_redis_context_patch_rejects_unserializable_values_before_mutation(
    monkeypatch,
) -> None:
    fake = _FakeRedis()
    monkeypatch.setitem(
        sys.modules,
        "redis",
        SimpleNamespace(from_url=lambda _url: fake),
    )
    store = RedisContextStore(redis_url="redis://test")
    old_key = store._k("old")
    fake.values[old_key] = store._codec.dumps("old")

    with pytest.raises(ValueError, match="failed to serialize"):
        store.apply_patch(
            {"bad": lambda value: value},
            delete_keys=("old",),
        )

    assert fake.pipeline_requests == []
    assert store._codec.loads(fake.values[old_key]) == "old"


def test_redis_context_snapshot_removes_the_complete_colon_delimited_prefix(
    monkeypatch,
) -> None:
    fake = _FakeRedis()
    monkeypatch.setitem(
        sys.modules,
        "redis",
        SimpleNamespace(from_url=lambda _url: fake),
    )
    store = RedisContextStore(
        redis_url="redis://test",
        key_prefix="blackbase:context:run-1",
    )
    fake.values[store._k("project.signal")] = store._codec.dumps({"ready": True})

    assert store.snapshot() == {"project.signal": {"ready": True}}


def test_redis_context_safe_codec_preserves_formal_lightweight_types(monkeypatch) -> None:
    fake = _FakeRedis()
    monkeypatch.setitem(
        sys.modules,
        "redis",
        SimpleNamespace(from_url=lambda _url: fake),
    )
    store = RedisContextStore(redis_url="redis://test")
    value = {
        ("stage", 1): (
            b"payload",
            {"ready", "committed"},
            np.asarray([1.5, 2.5], dtype=np.float32),
            DataRef(uri="artifact://model", metadata={"fold": 2}),
            StateRef(provider_id="provider", state_id="state-1"),
        )
    }

    store.apply_patch({"runtime": value})
    restored = store.snapshot()["runtime"]

    assert tuple(restored) == (("stage", 1),)
    payload = restored[("stage", 1)]
    assert isinstance(payload, tuple)
    assert payload[0] == b"payload"
    assert payload[1] == {"ready", "committed"}
    assert np.array_equal(payload[2], np.asarray([1.5, 2.5], dtype=np.float32))
    assert isinstance(payload[3], DataRef) and payload[3].metadata["fold"] == 2
    assert isinstance(payload[4], StateRef) and payload[4].state_id == "state-1"


def test_redis_context_safe_codec_never_falls_back_to_legacy_pickle(monkeypatch) -> None:
    global _PICKLE_EXECUTED
    _PICKLE_EXECUTED = False
    fake = _FakeRedis()
    monkeypatch.setitem(
        sys.modules,
        "redis",
        SimpleNamespace(from_url=lambda _url: fake),
    )
    store = RedisContextStore(redis_url="redis://test")
    fake.values[store._k("attacker")] = pickle.dumps(_MaliciousPickle())

    with pytest.raises(RedisValueCodecError, match="failed to decode"):
        store.get("attacker")

    assert _PICKLE_EXECUTED is False


def test_redis_context_payload_limit_is_enforced_before_mutation(monkeypatch) -> None:
    fake = _FakeRedis()
    monkeypatch.setitem(
        sys.modules,
        "redis",
        SimpleNamespace(from_url=lambda _url: fake),
    )
    store = RedisContextStore(redis_url="redis://test", max_payload_bytes=128)

    with pytest.raises(ValueError, match="failed to serialize"):
        store.apply_patch({"oversized": "x" * 512})

    assert fake.pipeline_requests == []
