from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from blackbase.context import RedisContextStore, RedisSnapshotStore
from blackbase.project.external_worker import ExternalCaseWorker
from blackbase.evaluation import (
    EvaluationDispositionEnvelope,
    EvaluationDispositionVerificationReceipt,
    RedisEvaluationEvidenceJournal,
    evaluation_disposition_digest,
)
from blackbase.project.project_runner import execute_project
from blackbase.project.scaffold import add_case, create_project
from blackbase.resources import (
    CancellationHeartbeat,
    CancellationRef,
    CancellationToken,
    RedisBudgetAuthority,
    RedisLeaseStore,
    RedisTaskTransport,
    ResourceAllocator,
    ResourceBudgetError,
    ResourceOffer,
    ResourcePolicy,
    ResourceRequest,
    ResourceRequirement,
    SharedBudgetExceeded,
    TaskEnvelope,
    TaskLeaseError,
    TaskResult,
    WorkerDescriptor,
)
from blackbase.types import UnknownState


REDIS_URL = str(os.environ.get("BLACKBASE_TEST_REDIS_URL", "") or "").strip()
pytestmark = pytest.mark.skipif(
    not REDIS_URL,
    reason="set BLACKBASE_TEST_REDIS_URL to run live Redis integration tests",
)


def _redis_client():
    redis = pytest.importorskip("redis")
    return redis.from_url(REDIS_URL, socket_timeout=2.0)


def _cleanup_keys(client, *patterns: str) -> None:
    keys: list[bytes] = []
    for pattern in patterns:
        keys.extend(client.scan_iter(match=pattern, count=100))
    if keys:
        client.delete(*tuple(dict.fromkeys(keys)))


def _allocator(namespace: str, *, ttl_seconds: float) -> ResourceAllocator:
    return ResourceAllocator(
        offer=ResourceOffer(threads=1, gpus=0, backend="local"),
        policy=ResourcePolicy(max_workers=1, max_threads=1, max_gpus=0),
        lease_store=RedisLeaseStore(REDIS_URL, namespace=namespace),
        lease_ttl_seconds=ttl_seconds,
    )


def test_live_redis_evaluation_evidence_journal_is_durable_and_atomic() -> None:
    namespace = f"live-evaluation-evidence-{uuid4().hex}"
    prefix = f"blackbase:test:{namespace}"
    client = _redis_client()
    journal = RedisEvaluationEvidenceJournal(
        redis_url=REDIS_URL,
        key_prefix=prefix,
    )
    try:
        assert client.ping()
        reserved = journal.reserve(
            event_id="event-1",
            run_id="run-1",
            event_snapshot_key="events/event-1",
        )
        pending = journal.mark_event_durable(
            "event-1",
            expected_revision=reserved.revision,
        )
        disposition = EvaluationDispositionEnvelope(
                event_id="event-1",
                status="committed",
                disposition_codec="test/v1",
                disposition_payload={"accepted_indices": [0]},
                event_snapshot_key="events/event-1",
                authority_snapshot_key="authority/event-1",
            )
        deciding = journal.prepare_disposition(
            disposition,
            expected_revision=pending.revision,
        )
        assert journal.settle(
            "event-1",
            verification=EvaluationDispositionVerificationReceipt(
                event_id="event-1",
                event_snapshot_key="events/event-1",
                destination_snapshot_key="authority/event-1",
                disposition_digest=evaluation_disposition_digest(disposition),
                verifier="test.redis_snapshot_store",
                verified_at=time.time(),
            ),
            expected_revision=deciding.revision,
        ).status == "committed"
        reopened = RedisEvaluationEvidenceJournal(
            redis_url=REDIS_URL,
            key_prefix=prefix,
        )
        assert reopened.get("event-1").status == "committed"
    finally:
        _cleanup_keys(client, f"{prefix}:*")


def test_live_redis_cancellation_control_is_heartbeated_and_retired() -> None:
    suffix = uuid4().hex
    namespace = f"live-control-{suffix}"
    client = _redis_client()
    ref = CancellationRef(
        backend="redis",
        namespace=namespace,
        redis_url_env="BLACKBASE_TEST_REDIS_URL",
        active_ttl_seconds=1.0,
        heartbeat_seconds=0.2,
        retention_seconds=2.0,
    )
    token = CancellationToken(ref)
    heartbeat = CancellationHeartbeat(token)
    key = f"blackbase:{namespace}:control:{ref.control_id}"
    try:
        assert client.ttl(key) > 0
        time.sleep(1.2)
        heartbeat.assert_current()
        assert client.exists(key) == 1
        assert client.ttl(key) > 0
        assert token.cancel("live cancellation") is True
        assert token.cancelled is True
        assert client.ttl(key) > 0
    finally:
        heartbeat.close()
        token.retire()
        assert client.exists(key) == 1
        assert token.touch() is False
        assert token.cancel("late cancellation") is False
        _cleanup_keys(client, f"blackbase:{namespace}:control:*")
    assert client.exists(key) == 0


def test_live_redis_context_and_safe_snapshot_preserve_protocol_values() -> None:
    suffix = uuid4().hex
    context_prefix = f"blackbase:live-context:{suffix}"
    snapshot_prefix = f"blackbase:live-snapshot:{suffix}"
    client = _redis_client()
    context = RedisContextStore(
        redis_url=REDIS_URL,
        key_prefix=context_prefix,
    )
    snapshots = RedisSnapshotStore(
        redis_url=REDIS_URL,
        key_prefix=snapshot_prefix,
        serializer="safe",
    )
    try:
        context.set("project.signal", {"ready": True})
        context.apply_patch({"generation": 4, "phase": "committed"})
        raw_context = client.get(f"{context_prefix}:project.signal")
        assert raw_context is not None
        context_envelope = json.loads(raw_context.decode("utf-8"))
        assert context_envelope["serializer"] == "safe"
        assert context_envelope["scope"] == "context"
        assert context.snapshot() == {
            "project.signal": {"ready": True},
            "generation": 4,
            "phase": "committed",
        }
        context._redis.connection_pool.disconnect()
        assert context.get("generation") == 4

        state = UnknownState(
            values=np.asarray([1.5, 2.5], dtype=np.float32),
            metadata={
                "source": "live-redis",
                "mask": np.asarray([1, 0], dtype=np.int8),
            },
        )
        handle = snapshots.write(
            {"candidates": (state,)},
            key="population",
            schema="blackbase.live_snapshot/v1",
        )
        restored = snapshots.read(handle.key).data["candidates"][0]
        assert isinstance(restored, UnknownState)
        assert np.allclose(restored.as_array(), [1.5, 2.5])
        assert restored.metadata["source"] == "live-redis"
        assert np.array_equal(restored.metadata["mask"], [1, 0])
    finally:
        _cleanup_keys(
            client,
            f"{context_prefix}:*",
            f"{snapshot_prefix}:*",
        )


def test_live_redis_lease_admission_ttl_and_fencing() -> None:
    namespace = f"live-lease-{uuid4().hex}"
    pattern = f"blackbase:l0_leases:{namespace}:*"
    client = _redis_client()
    barrier = threading.Barrier(2)

    def acquire(owner_id: str):
        allocator = _allocator(namespace, ttl_seconds=5.0)
        barrier.wait(timeout=2.0)
        try:
            return allocator.acquire(
                ResourceRequest(workers=1, threads=1),
                owner_id=owner_id,
            )
        except ResourceBudgetError:
            return None

    try:
        assert client.ping()
        with ThreadPoolExecutor(max_workers=2) as pool:
            leases = list(pool.map(acquire, ("project-a", "project-b")))
        admitted = [lease for lease in leases if lease is not None]
        assert len(admitted) == 1
        assert admitted[0].fencing_token == 1

        store = RedisLeaseStore(REDIS_URL, namespace=namespace)
        assert store.release_lease(admitted[0].lease_id, admitted[0].fencing_token)
        expiring_allocator = _allocator(namespace, ttl_seconds=0.15)
        expiring = expiring_allocator.acquire(
            ResourceRequest(workers=1, threads=1),
            owner_id="expiring",
        )
        assert expiring.fencing_token == 2

        time.sleep(0.2)
        assert expiring_allocator.renew(expiring) is None
        replacement = expiring_allocator.acquire(
            ResourceRequest(workers=1, threads=1),
            owner_id="replacement",
        )
        assert replacement.fencing_token == 3
        assert [item.status for item in store.list_all()] == [
            "released",
            "expired",
            "active",
        ]
    finally:
        _cleanup_keys(client, pattern)


def test_live_redis_shared_budget_is_atomic_and_lease_fenced() -> None:
    namespace = f"live-budget-{uuid4().hex}"
    scope = f"run-{uuid4().hex}"
    client = _redis_client()
    allocator = _allocator(namespace, ttl_seconds=5.0)
    lease = allocator.acquire(ResourceRequest(threads=1), owner_id="budget-owner")
    first = RedisBudgetAuthority(
        REDIS_URL,
        namespace=namespace,
        scope=scope,
    )
    second = RedisBudgetAuthority(
        REDIS_URL,
        namespace=namespace,
        scope=scope,
    )
    first.configure("evaluations", 3)
    barrier = threading.Barrier(2)

    def reserve(authority):
        barrier.wait(timeout=2.0)
        try:
            return authority.reserve(
                "evaluations",
                2,
                lease_id=lease.lease_id,
                fencing_token=lease.fencing_token,
            )
        except SharedBudgetExceeded:
            return None

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            reservations = list(pool.map(reserve, (first, second)))
        admitted = [item for item in reservations if item is not None]
        assert len(admitted) == 1
        first.complete(admitted[0], completed=2)
        pending = second.reserve(
            "evaluations",
            1,
            lease_id=lease.lease_id,
            fencing_token=lease.fencing_token,
        )
        assert pending.amount == 1

        allocator.release(lease)
        status = first.status("evaluations")
        assert status.committed == 2
        assert status.reserved == 0
        assert status.remaining == 1
        assert status.reclaimed == 1
    finally:
        _cleanup_keys(
            client,
            f"blackbase:l0_leases:{namespace}:*",
            f"blackbase:l0_budgets:{namespace}:*",
        )


def test_live_redis_project_external_worker_roundtrip(tmp_path) -> None:
    suffix = uuid4().hex
    lease_namespace = f"live-project-{suffix}"
    task_namespace = f"blackbase:live-tasks:{suffix}"
    client = _redis_client()
    project_root = create_project(tmp_path / "live_redis_project", framework="blackbase")
    case_root = add_case("remote_case", "solver", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        """
class Case:
    def __init__(self, resource_context=None):
        self.resource_context = dict(resource_context or {})

    def run(self):
        lease = dict(self.resource_context["lease"])
        return {
            "lease_id": lease["lease_id"],
            "fencing_token": lease["fencing_token"],
        }

def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, component_overrides
    return Case(resource_context)
""".lstrip(),
        encoding="utf-8",
    )
    (project_root / "project_config.py").write_text(
        f"""
PROJECT_NAME = "live_redis_project"
L0 = {{
    "namespace": {lease_namespace!r},
    "offer": {{"threads": 1, "gpus": 0, "backend": "local"}},
    "policy": {{"max_workers": 1, "max_threads": 1, "max_gpus": 0}},
    "default_request": {{"workers": 1, "threads": 1}},
    "lease_backend": "redis",
    "lease_redis_url_env": "BLACKBASE_TEST_REDIS_URL",
    "lease_ttl_seconds": 1.0,
    "lease_heartbeat_seconds": 0.1,
}}
STAGES = [{{
    "name": "external",
    "policy": "external",
    "cases": ["remote_case"],
    "external": {{
        "backend": "redis",
        "redis_url": {REDIS_URL!r},
        "namespace": {task_namespace!r},
        "queue_timeout_seconds": 3.0,
        "poll_interval_seconds": 0.01,
    }},
}}]
GROUPS = {{"default": {{"stages": ["external"]}}}}
""".lstrip(),
        encoding="utf-8",
    )
    transport = RedisTaskTransport(REDIS_URL, namespace=task_namespace)
    worker = ExternalCaseWorker(
        transport,
        WorkerDescriptor(
            worker_id=f"live-worker-{suffix}",
            executor_backend="external",
            resource_backend="local",
            capabilities=("project_case",),
            offer=ResourceOffer(threads=1, gpus=0, backend="local"),
        ),
        allowed_project_root=project_root,
        lease_seconds=1.0,
        heartbeat_interval_seconds=0.1,
        lease_redis_url=REDIS_URL,
    )
    stop_worker = threading.Event()
    worker_thread = threading.Thread(
        target=worker.run_forever,
        kwargs={
            "poll_interval_seconds": 0.01,
            "stop_event": stop_worker,
            "max_tasks": 1,
        },
        daemon=True,
    )
    worker_thread.start()

    try:
        result = execute_project(project_root, run_id=f"live-{suffix}")
        worker_thread.join(timeout=5.0)

        assert not worker_thread.is_alive()
        assert result.ok
        assert result.case_results[0].status == "succeeded"
        output = result.case_results[0].output
        assert output["fencing_token"] > 0
        record = transport.get(f"project:live-{suffix}:external:remote_case")
        assert record is not None and record.status == "succeeded"
        assert record.result is not None
        assert record.result.metadata["lease_fence_validated"] is True
        assert record.result.metadata["fencing_token"] == output["fencing_token"]
        manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
        assert REDIS_URL not in json.dumps(manifest, ensure_ascii=False)
        assert not list(
            client.scan_iter(
                match=f"blackbase:{lease_namespace}:control:*",
                count=100,
            )
        )
    finally:
        stop_worker.set()
        worker_thread.join(timeout=2.0)
        _cleanup_keys(
            client,
            f"blackbase:l0_leases:{lease_namespace}:*",
            f"blackbase:{lease_namespace}:control:*",
            f"{task_namespace}:*",
        )


def test_live_redis_task_lease_crash_recovery_rejects_stale_owner() -> None:
    suffix = uuid4().hex
    namespace = f"blackbase:live-recovery:{suffix}"
    client = _redis_client()
    transport = RedisTaskTransport(REDIS_URL, namespace=namespace)
    task = TaskEnvelope(
        task_id=f"live-recovery-{suffix}",
        task_type="project_case",
        payload={"case_name": "recovery"},
        requirement=ResourceRequirement(
            threads=1,
            resource_backend="local",
            capabilities=("project_case",),
        ),
        executor_backend="external",
        namespace="live.recovery",
        max_retries=1,
    )
    first_worker = WorkerDescriptor(
        worker_id=f"crashed-worker-{suffix}",
        executor_backend="external",
        resource_backend="local",
        capabilities=("project_case",),
        offer=ResourceOffer(threads=1, gpus=0, backend="local"),
    )
    second_worker = WorkerDescriptor(
        worker_id=f"replacement-worker-{suffix}",
        executor_backend="external",
        resource_backend="local",
        capabilities=("project_case",),
        offer=ResourceOffer(threads=1, gpus=0, backend="local"),
    )

    try:
        transport.submit(task)
        stale_claim = transport.claim(first_worker, lease_seconds=0.15)
        assert stale_claim is not None and stale_claim.attempt == 1
        time.sleep(0.2)

        assert transport.recover_expired() == 1
        assert transport.get(task.task_id).status == "queued"
        replacement_claim = transport.claim(second_worker, lease_seconds=1.0)
        assert replacement_claim is not None and replacement_claim.attempt == 2
        completed = transport.complete(
            replacement_claim,
            TaskResult.success(
                task_id=task.task_id,
                worker_id=second_worker.worker_id,
                output={"recovered": True},
            ),
        )
        assert completed.status == "succeeded"
        with pytest.raises(TaskLeaseError, match="no longer active"):
            transport.complete(
                stale_claim,
                TaskResult.success(
                    task_id=task.task_id,
                    worker_id=first_worker.worker_id,
                ),
            )
    finally:
        _cleanup_keys(client, f"{namespace}:*")
