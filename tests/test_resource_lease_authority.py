from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

import pytest

from blackbase.project.doctor import run_common_project_doctor
from blackbase.project.runtime import (
    ProjectL0Runtime,
    ProjectRuntimeConfig,
    ResourceLeaseFenceError,
)
from blackbase.project.project_runner import execute_project
from blackbase.project.scaffold import add_case, create_project
from blackbase.resources import (
    ResourceAllocator,
    ResourceBudgetError,
    InMemoryLeaseStore,
    ResourceLease,
    ResourceOffer,
    ResourcePolicy,
    ResourceRequest,
    SQLiteLeaseStore,
)


def test_resource_lease_is_recursively_immutable_and_store_safe() -> None:
    lease = ResourceLease(
        lease_id="lease-immutable",
        resources={"threads": 2, "resolved_devices": {"gpu-a": "cuda:0"}},
        metadata={"audit": {"source": "test"}},
        fencing_token=3,
    )
    store = InMemoryLeaseStore()
    store.create(lease)

    observed = store.get(lease.lease_id)
    assert observed is lease
    with pytest.raises(FrozenInstanceError):
        observed.fencing_token = 99
    with pytest.raises(TypeError):
        observed.resources["threads"] = 99
    with pytest.raises(TypeError):
        observed.resources["resolved_devices"]["gpu-a"] = "cuda:9"
    with pytest.raises(TypeError):
        observed.metadata["audit"]["source"] = "rewritten"
    assert store.get(lease.lease_id).as_dict() == lease.as_dict()


def _allocator(path, *, ttl_seconds: float = 30.0) -> ResourceAllocator:
    return ResourceAllocator(
        offer=ResourceOffer(threads=1, gpus=0, backend="local"),
        policy=ResourcePolicy(max_workers=1, max_threads=1, max_gpus=0),
        lease_store=SQLiteLeaseStore(path, namespace="test-project"),
        lease_ttl_seconds=ttl_seconds,
    )


def test_sqlite_lease_authority_persists_budget_and_monotonic_fence(tmp_path) -> None:
    path = tmp_path / "leases.sqlite"
    first_allocator = _allocator(path)
    first = first_allocator.acquire(
        ResourceRequest(workers=1, threads=1, memory_mb=1),
        owner_id="case-a",
        scope="stage",
    )

    assert first.fencing_token == 1
    assert first.expires_at > first.created_at
    assert first.resource_context()["fencing_token"] == 1
    assert first.resource_context()["lease"]["fencing_token"] == 1

    second_allocator = _allocator(path)
    with pytest.raises(ResourceBudgetError, match="active lease"):
        second_allocator.acquire(
            ResourceRequest(workers=1, threads=1, memory_mb=1),
            owner_id="case-b",
            scope="stage",
        )

    first_allocator.release(first)
    second = second_allocator.acquire(
        ResourceRequest(workers=1, threads=1, memory_mb=1),
        owner_id="case-b",
        scope="stage",
    )
    assert second.fencing_token == 2
    assert second_allocator.is_current(second)
    assert not second_allocator.is_current(first)


def test_sqlite_lease_expiry_rejects_stale_renewal_and_reopens_budget(tmp_path) -> None:
    path = tmp_path / "expiry.sqlite"
    first_allocator = _allocator(path, ttl_seconds=0.1)
    first = first_allocator.acquire(
        ResourceRequest(workers=1, threads=1, memory_mb=1),
        owner_id="crashed-case",
    )

    time.sleep(0.12)
    store = SQLiteLeaseStore(path, namespace="test-project")
    assert store.recover_expired() == 1
    assert first_allocator.renew(first) is None
    assert not first_allocator.is_current(first)

    second = _allocator(path, ttl_seconds=1.0).acquire(
        ResourceRequest(workers=1, threads=1, memory_mb=1),
        owner_id="recovered-case",
    )
    assert second.fencing_token > first.fencing_token


def test_sqlite_lease_admission_is_atomic_across_allocators(tmp_path) -> None:
    path = tmp_path / "atomic.sqlite"
    barrier = threading.Barrier(2)

    def acquire(owner_id: str):
        allocator = _allocator(path)
        barrier.wait(timeout=2.0)
        try:
            return allocator.acquire(
                ResourceRequest(workers=1, threads=1, memory_mb=1),
                owner_id=owner_id,
            )
        except ResourceBudgetError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        leases = list(pool.map(acquire, ("worker-a", "worker-b")))

    admitted = [lease for lease in leases if lease is not None]
    assert len(admitted) == 1
    assert admitted[0].fencing_token == 1
    assert len(SQLiteLeaseStore(path, namespace="test-project").list()) == 1


def test_project_l0_guard_renews_and_detects_revoked_fence(tmp_path) -> None:
    runtime = ProjectL0Runtime(
        ProjectRuntimeConfig(
            offer=ResourceOffer(threads=1, gpus=0, backend="local"),
            policy=ResourcePolicy(max_workers=1, max_threads=1, max_gpus=0),
            default_request=ResourceRequest(threads=1, memory_mb=1),
            namespace="guard-test",
            lease_backend="sqlite",
            lease_path="leases.sqlite",
            lease_ttl_seconds=0.15,
            lease_heartbeat_seconds=0.03,
        ),
        project_root=tmp_path,
    )
    lease = runtime.acquire_case("case-a", stage_name="stage")
    guard = runtime.start_lease_guard(lease)
    try:
        time.sleep(0.22)
        guard.assert_current()
        assert runtime.allocator.is_current(lease)

        assert runtime.allocator.lease_store.release_lease(
            lease.lease_id,
            lease.fencing_token,
        )
        with pytest.raises(ResourceLeaseFenceError, match="no longer current"):
            guard.assert_current()
    finally:
        guard.close()


def test_project_runner_keeps_short_ttl_lease_alive_until_case_finishes(tmp_path) -> None:
    project_root = create_project(tmp_path / "lease_guard_project", framework="blackbase")
    case_root = add_case("slow_case", "solver", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        """
import time

class Case:
    def __init__(self, resource_context):
        self.resource_context = dict(resource_context)

    def run(self):
        time.sleep(0.22)
        return {
            "fencing_token": self.resource_context["lease"]["fencing_token"],
            "expires_at": self.resource_context["lease"]["expires_at"],
        }

def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, component_overrides
    return Case(resource_context)
""".lstrip(),
        encoding="utf-8",
    )
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "lease_guard_project"
L0 = {
    "namespace": "lease_guard_project",
    "offer": {"threads": 1, "gpus": 0, "backend": "local"},
    "policy": {"max_workers": 1, "max_threads": 1, "max_gpus": 0},
    "default_request": {"workers": 1, "threads": 1, "memory_mb": 1},
    "lease_backend": "sqlite",
    "lease_path": ".blackbase/l0_leases.sqlite",
    "lease_ttl_seconds": 0.12,
    "lease_heartbeat_seconds": 0.03,
}
STAGES = [{"name": "slow", "cases": ["slow_case"]}]
GROUPS = {"default": {"stages": ["slow"]}}
""".lstrip(),
        encoding="utf-8",
    )

    result = execute_project(project_root, run_id="short-ttl")

    assert result.ok
    assert result.case_results[0].output["fencing_token"] > 0
    store = SQLiteLeaseStore(
        project_root / ".blackbase" / "l0_leases.sqlite",
        namespace="lease_guard_project",
    )
    assert store.list() == ()
    assert store.list_all()[0].status == "released"


def test_project_runner_rejects_output_after_lease_fence_is_revoked(tmp_path) -> None:
    project_root = create_project(tmp_path / "revoked_fence_project", framework="blackbase")
    case_root = add_case("revoker", "solver", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        """
from blackbase.resources import SQLiteLeaseStore

class Case:
    def __init__(self, resource_context):
        self.resource_context = dict(resource_context)

    def run(self):
        lease = dict(self.resource_context["lease"])
        authority = dict(self.resource_context["metadata"]["lease_authority"])
        store = SQLiteLeaseStore(authority["path"], namespace=authority["namespace"])
        store.release_lease(lease["lease_id"], lease["fencing_token"])
        return {"must_not_be_accepted": True}

def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, component_overrides
    return Case(resource_context)
""".lstrip(),
        encoding="utf-8",
    )
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "revoked_fence_project"
L0 = {
    "namespace": "revoked_fence_project",
    "offer": {"threads": 1, "gpus": 0, "backend": "local"},
    "policy": {"max_workers": 1, "max_threads": 1, "max_gpus": 0},
    "default_request": {"workers": 1, "threads": 1, "memory_mb": 1},
    "lease_backend": "sqlite",
    "lease_path": ".blackbase/l0_leases.sqlite",
    "lease_ttl_seconds": 1.0,
    "lease_heartbeat_seconds": 0.1,
}
STAGES = [{"name": "revoke", "cases": ["revoker"]}]
GROUPS = {"default": {"stages": ["revoke"]}}
""".lstrip(),
        encoding="utf-8",
    )

    result = execute_project(project_root, run_id="revoked")

    assert not result.ok
    assert result.case_results[0].status == "failed"
    assert "ResourceLeaseFenceError" in result.case_results[0].error
    assert result.case_results[0].output == {}


def test_project_doctor_validates_durable_l0_lease_configuration(tmp_path) -> None:
    project_root = create_project(tmp_path / "invalid_l0_project", framework="blackbase")
    generated = (project_root / "project_config.py").read_text(encoding="utf-8")
    assert '"lease_backend": "sqlite"' in generated
    assert '"lease_path": ".blackbase/l0_leases.sqlite"' in generated

    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "invalid_l0_project"
L0 = {
    "lease_backend": "http",
    "lease_ttl_seconds": 0,
    "lease_heartbeat_seconds": 10,
}
STAGES = []
GROUPS = {"default": {"stages": []}}
""".lstrip(),
        encoding="utf-8",
    )

    report = run_common_project_doctor(project_root, strict=True)
    codes = {item.code for item in report.diagnostics if item.level == "error"}
    assert "project-l0-lease-backend-invalid" in codes
    assert "project-l0-lease-ttl-invalid" in codes
    assert "project-l0-lease-heartbeat-invalid" in codes


def test_project_doctor_requires_redis_lease_connection_source(tmp_path) -> None:
    project_root = create_project(tmp_path / "redis_l0_project", framework="blackbase")
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "redis_l0_project"
L0 = {
    "lease_backend": "redis",
    "lease_redis_url": "",
    "lease_redis_url_env": "",
    "lease_ttl_seconds": 30,
    "lease_heartbeat_seconds": 10,
}
STAGES = []
GROUPS = {"default": {"stages": []}}
""".lstrip(),
        encoding="utf-8",
    )

    report = run_common_project_doctor(project_root, strict=True)
    codes = {item.code for item in report.diagnostics if item.level == "error"}
    assert "project-l0-lease-redis-connection-missing" in codes


def test_project_doctor_rejects_shared_budgets_without_durable_authority(tmp_path) -> None:
    project_root = create_project(tmp_path / "memory_budget_project", framework="blackbase")
    config_path = project_root / "project_config.py"
    config_path.write_text(
        '''
PROJECT_NAME = "memory_budget_project"
L0 = {
    "lease_backend": "memory",
    "lease_ttl_seconds": 30,
    "lease_heartbeat_seconds": 10,
    "budgets": {"evaluations": 100},
}
STAGES = []
GROUPS = {}
''',
        encoding="utf-8",
    )

    report = run_common_project_doctor(project_root)
    codes = {item.code for item in report.diagnostics}

    assert "project-l0-budget-authority-not-durable" in codes
