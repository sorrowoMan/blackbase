from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from blackbase.project.runtime import ProjectL0Runtime, ProjectRuntimeConfig
from blackbase.resources import (
    BudgetAccount,
    ResourceAllocator,
    ResourceOffer,
    ResourcePolicy,
    ResourceRequest,
    SQLiteBudgetAuthority,
    SQLiteLeaseStore,
    SharedBudgetExceeded,
    build_budget_authority_from_resource_context,
)


def _allocator(path, *, ttl_seconds: float = 30.0) -> ResourceAllocator:
    return ResourceAllocator(
        offer=ResourceOffer(threads=2, gpus=0, backend="local"),
        policy=ResourcePolicy(max_workers=2, max_threads=2, max_gpus=0),
        lease_store=SQLiteLeaseStore(path, namespace="budget-project"),
        lease_ttl_seconds=ttl_seconds,
    )


def test_sqlite_shared_budget_reservation_is_atomic_across_process_clients(tmp_path) -> None:
    path = tmp_path / "l0.sqlite"
    allocator = _allocator(path)
    leases = (
        allocator.acquire(ResourceRequest(threads=1), owner_id="a"),
        allocator.acquire(ResourceRequest(threads=1), owner_id="b"),
    )
    first = SQLiteBudgetAuthority(path, namespace="budget-project", scope="run-a")
    second = SQLiteBudgetAuthority(path, namespace="budget-project", scope="run-a")
    first.configure("evaluations", 3)
    barrier = threading.Barrier(2)

    def reserve(index: int):
        barrier.wait(timeout=2.0)
        authority = first if index == 0 else second
        lease = leases[index]
        try:
            return authority.reserve(
                "evaluations",
                2,
                lease_id=lease.lease_id,
                fencing_token=lease.fencing_token,
            )
        except SharedBudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        reservations = list(pool.map(reserve, (0, 1)))

    admitted = [item for item in reservations if item is not None]
    assert len(admitted) == 1
    status = first.status("evaluations")
    assert status.reserved == 2
    assert status.remaining == 1

    first.complete(admitted[0], completed=2)
    status = second.status("evaluations")
    assert status.committed == 2
    assert status.reserved == 0
    assert status.remaining == 1


def test_sqlite_shared_budget_reclaims_only_uncommitted_work_after_lease_loss(tmp_path) -> None:
    path = tmp_path / "expiry.sqlite"
    allocator = _allocator(path)
    stale_lease = allocator.acquire(ResourceRequest(threads=1), owner_id="crashed")
    authority = SQLiteBudgetAuthority(path, namespace="budget-project", scope="run-b")
    authority.configure("evaluations", 2)
    authority.reserve(
        "evaluations",
        2,
        lease_id=stale_lease.lease_id,
        fencing_token=stale_lease.fencing_token,
    )

    allocator.release(stale_lease)
    recovered = authority.status("evaluations")

    assert recovered.reclaimed == 1
    assert recovered.committed == 0
    assert recovered.reserved == 0
    assert recovered.remaining == 2


def test_project_runtime_grants_a_serializable_run_scoped_budget_authority(tmp_path) -> None:
    runtime = ProjectL0Runtime(
        ProjectRuntimeConfig(
            offer=ResourceOffer(threads=1, gpus=0, backend="local"),
            policy=ResourcePolicy(max_workers=1, max_threads=1, max_gpus=0),
            default_request=ResourceRequest(threads=1),
            namespace="runtime-budget",
            lease_backend="sqlite",
            lease_path="l0.sqlite",
            lease_ttl_seconds=1.0,
            lease_heartbeat_seconds=0.2,
            budgets={"evaluations": 5},
        ),
        project_root=tmp_path,
    )
    lease = runtime.acquire_case("solver")
    context = runtime.resource_context(lease, case_name="solver").as_dict()

    metadata = context["metadata"]["budget_authority"]
    assert metadata["backend"] == "sqlite"
    assert metadata["budgets"] == {"evaluations": 5}
    assert metadata["scope"].startswith("run-")

    reconstructed = build_budget_authority_from_resource_context(context)
    assert reconstructed is not None
    reservation = reconstructed.reserve(
        "evaluations",
        3,
        lease_id=lease.lease_id,
        fencing_token=lease.fencing_token,
    )
    reconstructed.complete(reservation, completed=2)
    assert runtime.budget_authority.status("evaluations").as_dict() == {
        "scope": metadata["scope"],
        "budget": "evaluations",
        "limit": 5,
        "committed": 2,
        "reserved": 0,
        "remaining": 3,
        "reclaimed": 0,
    }


def test_budget_account_owns_case_side_claim_lifecycle(tmp_path) -> None:
    runtime = ProjectL0Runtime(
        ProjectRuntimeConfig(
            offer=ResourceOffer(threads=1, gpus=0, backend="local"),
            policy=ResourcePolicy(max_workers=1, max_threads=1, max_gpus=0),
            default_request=ResourceRequest(threads=1),
            namespace="runtime-account",
            lease_backend="sqlite",
            lease_path="account.sqlite",
            budgets={"evaluations": 4},
        ),
        project_root=tmp_path,
    )
    lease = runtime.acquire_case("solver")
    context = runtime.resource_context(lease, case_name="solver")
    account = BudgetAccount.from_resource_context("evaluations", context)

    claim = account.reserve(3)
    assert account.active_reserved == 3
    account.consume(claim, 2)
    assert claim.consumed == 2
    assert account.active_reserved == 1

    account.complete(claim)
    assert claim.status == "completed"
    assert account.active_reserved == 0
    assert runtime.budget_authority.status("evaluations").as_dict() == {
        "scope": runtime.budget_authority.scope,
        "budget": "evaluations",
        "limit": 4,
        "committed": 2,
        "reserved": 0,
        "remaining": 2,
        "reclaimed": 0,
    }


def test_consumed_units_survive_cancellation_while_unused_units_are_returned(tmp_path) -> None:
    path = tmp_path / "partial.sqlite"
    allocator = _allocator(path)
    lease = allocator.acquire(ResourceRequest(threads=1), owner_id="partial")
    authority = SQLiteBudgetAuthority(path, namespace="budget-project", scope="run-partial")
    authority.configure("evaluations", 3)
    reservation = authority.reserve(
        "evaluations",
        3,
        lease_id=lease.lease_id,
        fencing_token=lease.fencing_token,
    )

    reservation = authority.consume(reservation, amount=2)
    in_flight = authority.status("evaluations")
    assert in_flight.committed == 2
    assert in_flight.reserved == 1
    assert in_flight.remaining == 0

    assert authority.cancel(reservation)
    cancelled = authority.status("evaluations")
    assert cancelled.committed == 2
    assert cancelled.reserved == 0
    assert cancelled.remaining == 1


def test_shared_budgets_reject_memory_only_l0_configuration() -> None:
    with pytest.raises(ValueError, match="requires lease_backend"):
        ProjectRuntimeConfig(lease_backend="memory", budgets={"evaluations": 1})
