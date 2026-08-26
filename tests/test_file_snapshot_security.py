from __future__ import annotations

import pickle

import numpy as np
import pytest

from blackbase.context import FileSnapshotStore, RedisValueCodecError
from blackbase.types import UnknownState


_PICKLE_EXECUTED = False


def _mark_pickle_executed():
    global _PICKLE_EXECUTED
    _PICKLE_EXECUTED = True
    return {"executed": True}


class _MaliciousPickle:
    def __reduce__(self):
        return (_mark_pickle_executed, ())


def test_file_snapshot_safe_extras_roundtrip_protocol_values(tmp_path) -> None:
    store = FileSnapshotStore(base_dir=tmp_path, serializer="safe")
    state = UnknownState(
        values=np.asarray([1.0, 2.0], dtype=np.float32),
        metadata={"source": "file-safe"},
    )

    handle = store.write({"states": (state,)}, key="run/state")
    record = store.read(handle.key)

    assert record is not None
    assert isinstance(record.data["states"], tuple)
    assert isinstance(record.data["states"][0], UnknownState)
    assert record.data["states"][0].metadata["source"] == "file-safe"
    assert store._paths(handle.key)["extras"].suffix == ".value"


def test_file_snapshot_key_cannot_escape_base_dir(tmp_path) -> None:
    store = FileSnapshotStore(base_dir=tmp_path / "snapshots")
    outside = tmp_path / "outside.meta.json"

    with pytest.raises(ValueError, match="escapes configured base_dir"):
        store.write({"value": 1}, key="../outside")

    assert not outside.exists()


def test_file_snapshot_safe_mode_blocks_legacy_pickle_without_execution(tmp_path) -> None:
    global _PICKLE_EXECUTED
    _PICKLE_EXECUTED = False
    store = FileSnapshotStore(base_dir=tmp_path, serializer="safe")
    handle = store.write({"array": np.asarray([1])}, key="legacy")
    paths = store._paths(handle.key)
    paths["legacy_extras"].write_bytes(pickle.dumps({"bad": _MaliciousPickle()}))

    record = store.read(handle.key)

    assert record is not None
    assert "bad" not in record.data
    assert _PICKLE_EXECUTED is False


def test_file_snapshot_safe_mode_rejects_unsupported_extras(tmp_path) -> None:
    store = FileSnapshotStore(base_dir=tmp_path, serializer="safe")

    with pytest.raises(RedisValueCodecError, match="unsupported safe value type"):
        store.write({"callable": lambda: None}, key="unsupported")
