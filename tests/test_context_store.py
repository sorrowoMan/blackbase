from __future__ import annotations

import pickle
import sys
from types import SimpleNamespace

import pytest

from blackbase.context import ContextStore, RedisContextStore, StateStoreConfig


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
    fake.values[store._k("old")] = pickle.dumps("old")

    store.apply_patch(
        {"new": {"value": 3}},
        delete_keys=("old",),
    )

    assert len(fake.pipeline_requests) == 1
    assert fake.pipeline_requests[0].transaction is True
    assert store._k("old") not in fake.values
    assert pickle.loads(fake.values[store._k("new")]) == {"value": 3}


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
    fake.values[old_key] = pickle.dumps("old")

    with pytest.raises(ValueError, match="failed to serialize"):
        store.apply_patch(
            {"bad": lambda value: value},
            delete_keys=("old",),
        )

    assert fake.pipeline_requests == []
    assert pickle.loads(fake.values[old_key]) == "old"
