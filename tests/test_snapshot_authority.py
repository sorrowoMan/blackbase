from __future__ import annotations

from pathlib import Path

import pytest

from blackbase.context import FileSnapshotStore, InMemorySnapshotStore


@pytest.mark.parametrize(
    "factory",
    [
        lambda tmp_path: InMemorySnapshotStore(),
        lambda tmp_path: FileSnapshotStore(base_dir=tmp_path / "snapshots"),
    ],
)
def test_authoritative_snapshot_is_write_once_digest_bound_and_pinned(
    factory,
    tmp_path,
) -> None:
    store = factory(tmp_path)
    handle = store.write(
        {"population": [[1.0, 2.0]]},
        key="authority/run/step-1",
        schema="test.authority/v1",
        write_once=True,
    )

    assert handle.revision == 1
    assert handle.content_digest.startswith("sha256:")
    with pytest.raises(FileExistsError):
        store.write(
            {"population": [[9.0, 9.0]]},
            key=handle.key,
            schema="test.authority/v1",
            write_once=True,
        )

    record = store.read(handle.key)
    assert record is not None
    record.data["population"][0][0] = -100.0
    reread = store.read(handle.key)
    assert reread is not None
    assert reread.data["population"][0][0] == 1.0

    pinned = store.pin(handle.key, owner="evidence:event-1")
    assert pinned.pinned is True
    with pytest.raises(RuntimeError, match="pinned"):
        store.delete(handle.key)
    store.unpin(handle.key, owner="evidence:event-1")
    store.delete(handle.key)
    assert store.read(handle.key) is None


def test_snapshot_compare_and_swap_advances_revision() -> None:
    store = InMemorySnapshotStore()
    first = store.write({"value": 1}, key="mutable", schema="test/v1")
    second = store.write(
        {"value": 2},
        key="mutable",
        schema="test/v1",
        expected_revision=first.revision,
    )
    assert second.revision == 2
    with pytest.raises(RuntimeError, match="revision conflict"):
        store.write(
            {"value": 3},
            key="mutable",
            schema="test/v1",
            expected_revision=first.revision,
        )


def test_file_snapshot_failed_meta_commit_preserves_old_revision_and_can_retry(
    tmp_path,
    monkeypatch,
) -> None:
    store = FileSnapshotStore(base_dir=tmp_path / "snapshots")
    first = store.write({"value": [1]}, key="cas", schema="test/v1")
    original_replace = Path.replace
    failed = False

    def fail_one_meta_commit(path: Path, target):
        nonlocal failed
        if not failed and str(target).endswith(".meta.json"):
            failed = True
            raise OSError("injected meta commit failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_one_meta_commit)
    with pytest.raises(OSError, match="injected"):
        store.write(
            {"value": [2]},
            key="cas",
            schema="test/v1",
            expected_revision=first.revision,
        )

    retained = store.read("cas")
    assert retained is not None
    assert retained.revision == 1
    assert retained.data["value"].tolist() == [1]
    retried = store.write(
        {"value": [2]},
        key="cas",
        schema="test/v1",
        expected_revision=first.revision,
    )
    assert retried.revision == 2


@pytest.mark.parametrize(
    "factory",
    [
        lambda tmp_path: InMemorySnapshotStore(),
        lambda tmp_path: FileSnapshotStore(base_dir=tmp_path / "snapshots"),
    ],
)
def test_final_unpin_restores_original_expiry(factory, tmp_path) -> None:
    store = factory(tmp_path)
    written = store.write({"value": 1}, key="ttl", ttl_seconds=60)
    store.pin("ttl", owner="archive")
    released = store.unpin("ttl", owner="archive")
    assert released.pinned is False
    assert released.expires_at == pytest.approx(written.expires_at)
