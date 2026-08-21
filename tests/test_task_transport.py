from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from blackbase.resources import (
    ClaimedTask,
    RedisLeaseStore,
    RedisBudgetAuthority,
    ResourceAllocator,
    ResourceBudgetError,
    ResourceOffer,
    ResourcePolicy,
    ResourceRequest,
    ResourceRequirement,
    RedisTaskTransport,
    SQLiteTaskTransport,
    TaskEnvelope,
    TaskLeaseError,
    TaskResult,
    TaskTransportError,
    SharedBudgetExceeded,
    WorkerDescriptor,
)
from blackbase.project import CaseRunRequest
from blackbase.project.external_worker import ExternalCaseWorker
from blackbase.project.runtime import ProjectL0Runtime, ProjectRuntimeConfig
from blackbase.project.scaffold import add_case, create_project


class _FakeRedisLock:
    def __init__(self, lock: threading.RLock) -> None:
        self._lock = lock

    def acquire(self, blocking: bool = True) -> bool:
        return bool(self._lock.acquire(blocking=blocking))

    def release(self) -> None:
        self._lock.release()


class _FakeRedisPipeline:
    def __init__(self, client: "_FakeRedis") -> None:
        self.client = client
        self.ops: list[tuple[str, tuple]] = []

    def __getattr__(self, name: str):
        def enqueue(*args):
            self.ops.append((name, args))
            return self

        return enqueue

    def execute(self):
        with self.client._data_lock:
            return [getattr(self.client, name)(*args) for name, args in self.ops]


class _FakeRedis:
    def __init__(self) -> None:
        self._values: dict[str, object] = {}
        self._lists: dict[str, list[object]] = {}
        self._sets: dict[str, set[object]] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._data_lock = threading.RLock()

    def lock(self, name, timeout=None, blocking_timeout=None):
        del timeout, blocking_timeout
        with self._data_lock:
            lock = self._locks.setdefault(str(name), threading.RLock())
        return _FakeRedisLock(lock)

    def pipeline(self, transaction=True):
        del transaction
        return _FakeRedisPipeline(self)

    def set(self, key, value):
        with self._data_lock:
            self._values[str(key)] = value
        return True

    def get(self, key):
        with self._data_lock:
            return self._values.get(str(key))

    def time(self):
        now = time.time()
        seconds = int(now)
        return seconds, int((now - seconds) * 1_000_000)

    def sadd(self, key, *values):
        with self._data_lock:
            target = self._sets.setdefault(str(key), set())
            before = len(target)
            target.update(values)
            return len(target) - before

    def srem(self, key, *values):
        with self._data_lock:
            target = self._sets.setdefault(str(key), set())
            before = len(target)
            target.difference_update(values)
            return before - len(target)

    def smembers(self, key):
        with self._data_lock:
            return set(self._sets.get(str(key), set()))

    def rpush(self, key, *values):
        with self._data_lock:
            target = self._lists.setdefault(str(key), [])
            target.extend(values)
            return len(target)

    def lrange(self, key, start, end):
        with self._data_lock:
            values = list(self._lists.get(str(key), []))
        stop = len(values) if int(end) == -1 else int(end) + 1
        return values[int(start):stop]

    def lrem(self, key, count, value):
        with self._data_lock:
            values = self._lists.setdefault(str(key), [])
            limit = abs(int(count))
            removed = 0
            output = []
            for item in values:
                if item == value and (limit == 0 or removed < limit):
                    removed += 1
                else:
                    output.append(item)
            self._lists[str(key)] = output
            return removed


def _worker(worker_id: str, *, capabilities=("project_case",)) -> WorkerDescriptor:
    return WorkerDescriptor(
        worker_id=worker_id,
        executor_backend="external",
        resource_backend="local",
        capabilities=capabilities,
        offer=ResourceOffer(threads=2, gpus=0, backend="local"),
        max_inflight=1,
    )


def _task(task_id: str, *, max_retries: int = 0) -> TaskEnvelope:
    return TaskEnvelope(
        task_id=task_id,
        task_type="project_case",
        payload={"case_name": "demo"},
        requirement=ResourceRequirement(
            threads=1,
            resource_backend="local",
            capabilities=("project_case",),
        ),
        executor_backend="external",
        namespace="project.stage.case",
        max_retries=max_retries,
    )


def test_sqlite_transport_submit_is_idempotent_and_json_strict(tmp_path) -> None:
    transport = SQLiteTaskTransport(tmp_path / "transport.sqlite")
    task = _task("task-1")

    first = transport.submit(task)
    second = transport.submit(task)

    assert first.status == "queued"
    assert second.task.as_dict() == first.task.as_dict()
    assert transport.counts() == {"queued": 1}

    changed = TaskEnvelope.from_dict({**task.as_dict(), "payload": {"case_name": "changed"}})
    with pytest.raises(TaskTransportError, match="different envelope"):
        transport.submit(changed)

    with pytest.raises(TypeError, match="not wire-safe"):
        TaskEnvelope(
            task_id="unsafe",
            task_type="project_case",
            payload={"value": object()},
        )


def test_sqlite_transport_claim_is_atomic_and_checks_worker_contract(tmp_path) -> None:
    path = tmp_path / "transport.sqlite"
    transport = SQLiteTaskTransport(path)
    transport.submit(_task("atomic-task"))
    workers = (_worker("worker-a"), _worker("worker-b"))

    def claim(worker: WorkerDescriptor):
        return SQLiteTaskTransport(path).claim(worker, lease_seconds=2.0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(claim, workers))

    owned = [item for item in claims if item is not None]
    assert len(owned) == 1
    assert owned[0].task.task_id == "atomic-task"
    assert transport.get("atomic-task").status == "leased"

    transport.submit(_task("capability-task"))
    incompatible = _worker("plain-worker", capabilities=())
    assert transport.claim(incompatible) is None


def test_sqlite_transport_complete_roundtrip_and_rejects_stale_lease(tmp_path) -> None:
    transport = SQLiteTaskTransport(tmp_path / "transport.sqlite")
    transport.submit(_task("complete-task"))
    claim = transport.claim(_worker("worker"), lease_seconds=2.0)
    assert isinstance(claim, ClaimedTask)

    result = TaskResult.success(
        task_id="complete-task",
        worker_id="worker",
        output={"value": 42, "artifact_refs": {"model": {"uri": "memory://model"}}},
    )
    completed = transport.complete(claim, result)

    assert completed.status == "succeeded"
    assert completed.result is not None
    assert completed.result.output["value"] == 42
    assert transport.wait_result("complete-task", timeout_seconds=0.1).output["value"] == 42

    with pytest.raises(TaskLeaseError, match="no longer active"):
        transport.complete(claim, result)


def test_sqlite_transport_rejects_result_state_mismatch(tmp_path) -> None:
    transport = SQLiteTaskTransport(tmp_path / "transport.sqlite")
    transport.submit(_task("state-task"))
    claim = transport.claim(_worker("worker"), lease_seconds=2.0)
    assert claim is not None

    with pytest.raises(TaskTransportError, match="use fail"):
        transport.complete(
            claim,
            TaskResult.failure(task_id="state-task", error="wrong completion path"),
        )
    with pytest.raises(TaskTransportError, match="use complete"):
        transport.fail(
            claim,
            TaskResult.success(task_id="state-task"),
        )

    transport.fail(
        claim,
        TaskResult.failure(task_id="state-task", error="expected"),
    )


def test_sqlite_transport_failure_retry_and_expired_lease_recovery(tmp_path) -> None:
    transport = SQLiteTaskTransport(tmp_path / "transport.sqlite")
    worker = _worker("worker")
    transport.submit(_task("retry-task", max_retries=1))
    first_claim = transport.claim(worker, lease_seconds=2.0)
    assert first_claim is not None and first_claim.attempt == 1

    retry_record = transport.fail(
        first_claim,
        TaskResult.failure(task_id="retry-task", error="transient", worker_id="worker"),
    )
    assert retry_record.status == "queued"
    assert retry_record.result is None

    second_claim = transport.claim(worker, lease_seconds=2.0)
    assert second_claim is not None and second_claim.attempt == 2
    terminal = transport.fail(
        second_claim,
        TaskResult.failure(task_id="retry-task", error="permanent", worker_id="worker"),
    )
    assert terminal.status == "failed"
    assert terminal.result is not None
    assert terminal.result.error == "permanent"

    transport.submit(_task("expired-task", max_retries=0))
    expired_claim = transport.claim(worker, lease_seconds=0.1)
    assert expired_claim is not None
    time.sleep(0.12)
    assert transport.recover_expired() == 1
    expired = transport.get("expired-task")
    assert expired is not None and expired.status == "failed"
    assert expired.result is not None
    assert "lease expired" in expired.result.error
    assert transport.heartbeat_task(expired_claim) is False


def test_sqlite_transport_worker_registry_heartbeat(tmp_path) -> None:
    transport = SQLiteTaskTransport(tmp_path / "transport.sqlite")
    worker = transport.register_worker(_worker("worker"))

    assert [item.worker_id for item in transport.list_workers()] == ["worker"]
    assert transport.heartbeat_worker(worker.worker_id, status="idle") is True
    assert transport.list_workers()[0].status == "idle"
    assert transport.heartbeat_worker("missing") is False


def test_redis_transport_matches_atomic_lease_retry_and_worker_contract() -> None:
    client = _FakeRedis()
    transport = RedisTaskTransport(client=client, namespace="test:transport")
    task = _task("redis-atomic", max_retries=1)

    first = transport.submit(task)
    second = transport.submit(task)
    assert first.status == "queued"
    assert second.task.as_dict() == first.task.as_dict()

    changed = TaskEnvelope.from_dict({**task.as_dict(), "payload": {"case_name": "changed"}})
    with pytest.raises(TaskTransportError, match="different envelope"):
        transport.submit(changed)

    workers = (_worker("redis-a"), _worker("redis-b"))

    def claim(worker: WorkerDescriptor):
        return RedisTaskTransport(client=client, namespace="test:transport").claim(
            worker,
            lease_seconds=2.0,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(claim, workers))
    owned = [item for item in claims if item is not None]
    assert len(owned) == 1
    active = owned[0]
    assert active.task.task_id == task.task_id

    retry = transport.fail(
        active,
        TaskResult.failure(task_id=task.task_id, error="transient", worker_id=active.worker_id),
    )
    assert retry.status == "queued"
    second_claim = transport.claim(_worker("redis-a"), lease_seconds=2.0)
    assert second_claim is not None and second_claim.attempt == 2
    completed = transport.complete(
        second_claim,
        TaskResult.success(
            task_id=task.task_id,
            worker_id=second_claim.worker_id,
            output={"value": 42},
        ),
    )
    assert completed.status == "succeeded"
    assert transport.wait_result(task.task_id, timeout_seconds=0.1).output == {"value": 42}
    assert transport.counts() == {"succeeded": 1}
    assert {item.worker_id for item in transport.list_workers()} == {"redis-a", "redis-b"}


def test_redis_transport_recovers_expired_lease_and_rejects_stale_owner() -> None:
    transport = RedisTaskTransport(client=_FakeRedis(), namespace="test:expiry")
    transport.submit(_task("redis-expired"))
    claim = transport.claim(_worker("redis-worker"), lease_seconds=0.1)
    assert claim is not None

    time.sleep(0.12)
    assert transport.heartbeat_task(claim) is False
    with pytest.raises(TaskLeaseError, match="no longer active"):
        transport.complete(
            claim,
            TaskResult.success(task_id=claim.task.task_id, worker_id=claim.worker_id),
        )
    record = transport.get(claim.task.task_id)
    assert record is not None and record.status == "failed"
    assert record.result is not None and "lease expired" in record.result.error


def _redis_allocator(client: _FakeRedis, *, ttl_seconds: float = 30.0) -> ResourceAllocator:
    return ResourceAllocator(
        offer=ResourceOffer(threads=1, gpus=0, backend="local"),
        policy=ResourcePolicy(max_workers=1, max_threads=1, max_gpus=0),
        lease_store=RedisLeaseStore(client=client, namespace="lease-test"),
        lease_ttl_seconds=ttl_seconds,
    )


def test_redis_lease_authority_admission_is_atomic_and_fenced() -> None:
    client = _FakeRedis()
    barrier = threading.Barrier(2)

    def acquire(owner_id: str):
        allocator = _redis_allocator(client)
        barrier.wait(timeout=2.0)
        try:
            return allocator.acquire(
                ResourceRequest(workers=1, threads=1),
                owner_id=owner_id,
            )
        except ResourceBudgetError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        leases = list(pool.map(acquire, ("redis-worker-a", "redis-worker-b")))

    admitted = [lease for lease in leases if lease is not None]
    assert len(admitted) == 1
    assert admitted[0].fencing_token == 1
    store = RedisLeaseStore(client=client, namespace="lease-test")
    assert store.list() == tuple(admitted)

    assert store.release_lease(admitted[0].lease_id, admitted[0].fencing_token)
    replacement = _redis_allocator(client).acquire(
        ResourceRequest(workers=1, threads=1),
        owner_id="replacement",
    )
    assert replacement.fencing_token == 2
    assert not store.is_current(admitted[0].lease_id, admitted[0].fencing_token)


def test_redis_lease_authority_expires_and_rejects_stale_renewal() -> None:
    client = _FakeRedis()
    allocator = _redis_allocator(client, ttl_seconds=0.1)
    stale = allocator.acquire(
        ResourceRequest(workers=1, threads=1),
        owner_id="stale",
    )

    time.sleep(0.12)
    assert allocator.renew(stale) is None
    assert allocator.lease_store.recover_expired() == 0
    replacement = allocator.acquire(
        ResourceRequest(workers=1, threads=1),
        owner_id="replacement",
    )
    assert replacement.fencing_token == stale.fencing_token + 1
    history = allocator.lease_store.list_all()
    assert [item.status for item in history] == ["expired", "active"]


def test_redis_shared_budget_is_atomic_and_reclaims_stale_reservations() -> None:
    client = _FakeRedis()
    allocator = _redis_allocator(client, ttl_seconds=1.0)
    lease = allocator.acquire(
        ResourceRequest(workers=1, threads=1),
        owner_id="budget-owner",
    )
    first = RedisBudgetAuthority(
        client=client,
        namespace="lease-test",
        scope="budget-run",
    )
    second = RedisBudgetAuthority(
        client=client,
        namespace="lease-test",
        scope="budget-run",
    )
    first.configure("evaluations", 5)
    barrier = threading.Barrier(2)

    def reserve(authority):
        barrier.wait(timeout=2.0)
        try:
            return authority.reserve(
                "evaluations",
                3,
                lease_id=lease.lease_id,
                fencing_token=lease.fencing_token,
            )
        except SharedBudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        reservations = list(pool.map(reserve, (first, second)))

    admitted = [item for item in reservations if item is not None]
    assert len(admitted) == 1
    first.complete(admitted[0], completed=2)
    pending = second.reserve(
        "evaluations",
        3,
        lease_id=lease.lease_id,
        fencing_token=lease.fencing_token,
    )
    assert pending.amount == 3

    allocator.release(lease)
    recovered = first.status("evaluations")
    assert recovered.reclaimed == 1
    assert recovered.committed == 2
    assert recovered.reserved == 0
    assert recovered.remaining == 3


def test_project_l0_runtime_uses_redis_authority_without_exposing_url(tmp_path) -> None:
    client = _FakeRedis()
    runtime = ProjectL0Runtime(
        ProjectRuntimeConfig(
            offer=ResourceOffer(threads=1, gpus=0, backend="local"),
            policy=ResourcePolicy(max_workers=1, max_threads=1, max_gpus=0),
            default_request=ResourceRequest(workers=1, threads=1),
            namespace="redis-project",
            lease_backend="redis",
            lease_redis_url="redis://user:secret@example.invalid/0",
            lease_redis_url_env="TEST_BLACKBASE_REDIS_URL",
            lease_ttl_seconds=1.0,
            lease_heartbeat_seconds=0.1,
        ),
        project_root=tmp_path,
        lease_redis_client=client,
    )
    lease = runtime.acquire_case("case-a", stage_name="stage")
    context = runtime.resource_context(lease, case_name="case-a", stage_name="stage")
    authority = context.metadata["lease_authority"]

    assert authority == {
        "backend": "redis",
        "namespace": "redis-project",
        "redis_url_env": "TEST_BLACKBASE_REDIS_URL",
        "ttl_seconds": 1.0,
        "heartbeat_seconds": 0.1,
    }
    assert "secret" not in str(context.as_dict())
    assert runtime.allocator.is_current(lease)
    runtime.release(lease)
    assert not runtime.allocator.is_current(lease)


def test_external_worker_validates_and_renews_redis_project_fence(tmp_path) -> None:
    project_root = create_project(tmp_path / "redis_fence_worker", framework="blackbase")
    case_root = add_case("remote_case", "solver", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        """
class Case:
    def __init__(self, resource_context=None):
        self.resource_context = dict(resource_context or {})

    def run(self):
        return {"fencing_token": self.resource_context["lease"]["fencing_token"]}

def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, component_overrides
    return Case(resource_context)
""".lstrip(),
        encoding="utf-8",
    )
    client = _FakeRedis()
    runtime = ProjectL0Runtime(
        ProjectRuntimeConfig(
            offer=ResourceOffer(threads=1, gpus=0, backend="local"),
            policy=ResourcePolicy(max_workers=1, max_threads=1, max_gpus=0),
            default_request=ResourceRequest(workers=1, threads=1),
            namespace="redis-worker-project",
            lease_backend="redis",
            lease_redis_url_env="TEST_BLACKBASE_REDIS_URL",
            lease_ttl_seconds=1.0,
            lease_heartbeat_seconds=0.1,
        ),
        project_root=project_root,
        lease_redis_client=client,
    )
    lease = runtime.acquire_case("remote_case", stage_name="external")
    resource_context = runtime.resource_context(
        lease,
        case_name="remote_case",
        stage_name="external",
    ).as_dict()
    transport = RedisTaskTransport(client=client, namespace="redis-worker-tasks")
    task = TaskEnvelope(
        task_id="redis-fenced-case",
        task_type="project_case",
        payload={
            "project_root": str(project_root),
            "request": CaseRunRequest(
                project_name="redis-worker-project",
                stage_name="external",
                case_name="remote_case",
                resource_request=ResourceRequest(workers=1, threads=1).as_dict(),
                resource_context=resource_context,
            ).as_dict(),
            "extra_python_paths": [],
        },
        requirement=ResourceRequirement(
            threads=1,
            resource_backend="local",
            capabilities=("project_case",),
        ),
        executor_backend="external",
        namespace="redis-worker-project.external.remote_case",
    )
    transport.submit(task)
    worker = ExternalCaseWorker(
        transport,
        _worker("redis-fence-worker"),
        allowed_project_root=project_root,
        lease_seconds=1.0,
        heartbeat_interval_seconds=0.1,
        lease_redis_client=client,
    )

    assert worker.run_once()
    record = transport.get(task.task_id)
    assert record is not None and record.status == "succeeded"
    assert record.result is not None
    assert record.result.metadata["lease_fence_validated"] is True
    assert record.result.metadata["fencing_token"] == lease.fencing_token
    assert record.result.output["output"]["fencing_token"] == lease.fencing_token
    runtime.release(lease)

    revoked = runtime.acquire_case("remote_case", stage_name="external")
    revoked_context = runtime.resource_context(
        revoked,
        case_name="remote_case",
        stage_name="external",
    ).as_dict()
    revoked_task = TaskEnvelope.from_dict(
        {
            **task.as_dict(),
            "task_id": "redis-revoked-case",
            "payload": {
                **dict(task.payload),
                "request": CaseRunRequest(
                    project_name="redis-worker-project",
                    stage_name="external",
                    case_name="remote_case",
                    resource_request=ResourceRequest(workers=1, threads=1).as_dict(),
                    resource_context=revoked_context,
                ).as_dict(),
            },
        }
    )
    transport.submit(revoked_task)
    runtime.release(revoked)

    assert worker.run_once()
    rejected = transport.get(revoked_task.task_id)
    assert rejected is not None and rejected.status == "failed"
    assert rejected.result is not None
    assert "Project L0 fence is not current" in rejected.result.error


def test_external_case_worker_retries_case_and_returns_named_artifacts(tmp_path) -> None:
    project_root = create_project(tmp_path / "worker_project", framework="blackbase")
    case_root = add_case("remote_case", "solver", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        """
from pathlib import Path

class Case:
    def __init__(self, resource_context=None):
        self.resource_context = dict(resource_context or {})

    def run(self):
        root = Path(__file__).resolve().parents[2]
        counter = root / "remote_case.count"
        count = int(counter.read_text(encoding="utf-8")) + 1 if counter.exists() else 1
        counter.write_text(str(count), encoding="utf-8")
        if count == 1:
            raise RuntimeError("retry me")
        model_ref = self.case_runtime.publish_artifact(
            "model",
            {"count": count},
            kind="model",
        )
        return {
            "count": count,
            "lease_id": self.resource_context["lease"]["lease_id"],
            "artifact_refs": {"model": model_ref},
        }

def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, component_overrides
    return Case(resource_context)
""".lstrip(),
        encoding="utf-8",
    )
    transport = SQLiteTaskTransport(project_root / ".blackbase" / "tasks.sqlite")
    task = _task("external-case", max_retries=1)
    payload = {
        "project_root": str(project_root),
        "request": CaseRunRequest(
            project_name="worker-project",
            stage_name="external",
            case_name="remote_case",
            resource_request=ResourceRequest(workers=1, threads=1).as_dict(),
            resource_context={
                "scope": "remote",
                "threads": 1,
                "lease": {"lease_id": "project-l0-lease"},
                "metadata": {
                    "artifact_authority": {
                        "backend": "filesystem",
                        "root": str(project_root / ".blackbase" / "artifacts"),
                        "namespace": "worker-project",
                        "schema_version": 1,
                    },
                },
            },
        ).as_dict(),
        "extra_python_paths": [],
    }
    task = TaskEnvelope.from_dict({**task.as_dict(), "payload": payload})
    transport.submit(task)
    worker = ExternalCaseWorker(
        transport,
        _worker("external-worker"),
        allowed_project_root=project_root,
        lease_seconds=1.0,
        heartbeat_interval_seconds=0.1,
    )

    assert worker.run_once() is True
    retry = transport.get(task.task_id)
    assert retry is not None and retry.status == "queued" and retry.attempt == 1

    assert worker.run_once() is True
    completed = transport.get(task.task_id)
    assert completed is not None and completed.status == "succeeded"
    assert completed.attempt == 2
    assert completed.result is not None
    worker_payload = completed.result.output
    assert worker_payload["output"]["count"] == 2
    assert worker_payload["output"]["lease_id"] == "project-l0-lease"
    assert Path(worker_payload["artifact_refs"]["model"]["uri"]).is_file()
    assert completed.result.artifact_refs[0].kind == "model"
    assert (project_root / "remote_case.count").read_text(encoding="utf-8") == "2"
    assert transport.list_workers()[0].status == "idle"
