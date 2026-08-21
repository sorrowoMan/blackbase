from __future__ import annotations

import threading
import time

import pytest

from blackbase.evaluation import (
    EvaluationGateway,
    EvaluationProviderContractError,
    EvaluationProviderRegistry,
    EvaluationProviderSpec,
    EvaluationProviderUnavailable,
    StateMaterializationRequest,
    StateMaterializationResult,
    StateReleaseRequest,
)
from blackbase.project import CaseRunRequest, CaseRunResult
from blackbase.project.case_execution import (
    collect_artifact_refs,
    make_transport_safe,
)
from blackbase.resources import (
    DataRef,
    ResourceAllocator,
    ResourceContext,
    ResourceGrantPool,
    ResourceOffer,
    ResourcePolicy,
    ResourceRequest,
    ResourceSubgrantError,
)
from blackbase.state_ref import StateRef
from blackbase.types import UnknownState


def test_shared_wire_contracts_are_recursively_immutable_and_detached() -> None:
    source = {"nested": {"values": [1, 2]}}
    context = ResourceContext(grant=source, metadata=source)
    request = CaseRunRequest(
        project_name="project",
        stage_name="stage",
        case_name="case",
        inputs=source,
    )
    result = CaseRunResult(request=request, status="succeeded", output=source)

    source["nested"]["values"][0] = 99
    assert context.grant["nested"]["values"] == (1, 2)
    assert request.inputs["nested"]["values"] == (1, 2)
    assert result.output["nested"]["values"] == (1, 2)

    with pytest.raises(TypeError):
        context.grant["nested"]["values"] = (9,)
    with pytest.raises(TypeError):
        request.inputs["nested"]["rewritten"] = True
    with pytest.raises(TypeError):
        result.output["nested"]["rewritten"] = True

    detached = context.as_dict()
    detached["grant"]["nested"]["values"][0] = 7
    assert context.grant["nested"]["values"] == (1, 2)


def test_logical_gpu_token_is_resolved_only_by_l0_authority() -> None:
    token = "logical-gpu:training"
    allocator = ResourceAllocator(
        offer=ResourceOffer(
            threads=2,
            gpus=0,
            backend="local",
            device_tokens=(token,),
            metadata={"device_bindings": {token: "cuda:1"}},
        ),
        policy=ResourcePolicy(max_workers=1, max_threads=2, max_gpus=1),
    )
    request = ResourceRequest(
        workers=1,
        threads=1,
        device_tokens=(token,),
        compute_backend="torch",
    )
    lease = allocator.acquire(request, owner_id="trainer", scope="fit")
    context = ResourceContext.from_mapping(
        lease.resource_context(compute_backend="torch", namespace="case.fit")
    )

    assert request.gpus == 1
    assert context.grant["resolved_devices"][token] == "cuda:1"
    assert context.device == "cuda:1"

    class Provider:
        spec = EvaluationProviderSpec(
            provider_id="torch.materialize/v1",
            compute_backend="torch",
            supported_devices=("gpu",),
            preferred_devices=("gpu",),
            state_kinds=("model_parameters",),
            materialization_targets=("unknown_state",),
        )

        def evaluate(self, request, binding):  # pragma: no cover
            raise AssertionError("materialization-only test provider")

        def materialize(self, materialize_request, binding):
            return StateMaterializationResult(
                request_id=materialize_request.request_id,
                state_ref=materialize_request.state_ref,
                target=materialize_request.target,
                value=UnknownState([1.0]),
            )

    provider = Provider()
    registry = EvaluationProviderRegistry()
    registry.register(provider)
    state_ref = StateRef(
        provider_id=provider.spec.provider_id,
        state_id="parameters",
        state_kind="model_parameters",
        scope_id="case.fit",
        trajectory_id="trajectory",
        device="cuda:1",
    )
    materialized = EvaluationGateway(registry).materialize(
        StateMaterializationRequest(state_ref=state_ref),
        context,
    )

    assert materialized.binding is not None
    assert materialized.binding.device == "cuda:1"


def test_unresolved_logical_gpu_token_cannot_be_used_as_a_physical_device() -> None:
    class Provider:
        spec = EvaluationProviderSpec(
            provider_id="gpu.only/v1",
            compute_backend="torch",
            supported_devices=("gpu",),
            state_kinds=("model_parameters",),
            materialization_targets=("unknown_state",),
        )

        def evaluate(self, request, binding):  # pragma: no cover
            raise AssertionError("materialization-only test provider")

        def materialize(self, request, binding):  # pragma: no cover
            raise AssertionError("unresolved token must not bind")

    registry = EvaluationProviderRegistry()
    registry.register(Provider())
    token = "logical-gpu:unresolved"
    request = StateMaterializationRequest(
        state_ref=StateRef(
            provider_id="gpu.only/v1",
            state_id="parameters",
            state_kind="model_parameters",
            scope_id="case.fit",
            trajectory_id="trajectory",
            device=token,
        )
    )

    with pytest.raises(EvaluationProviderUnavailable):
        EvaluationGateway(registry).materialize(
            request,
            ResourceContext(
                compute_backend="torch",
                device=token,
                namespace="case.fit",
                grant={
                    "threads": 1,
                    "gpus": 1,
                    "device_tokens": [token],
                },
            ),
        )


def test_state_release_requires_the_exact_authoritative_scope() -> None:
    with pytest.raises(ValueError, match="scope_id"):
        StateReleaseRequest(provider_id="provider/v1", trajectory_id="trajectory")


def test_provider_cannot_mutate_materialization_request_after_binding() -> None:
    class MutatingProvider:
        spec = EvaluationProviderSpec(
            provider_id="mutating/v1",
            state_kinds=("model_parameters",),
            materialization_targets=("unknown_state",),
        )

        def evaluate(self, request, binding):  # pragma: no cover
            raise AssertionError("materialization-only test provider")

        def materialize(self, request, binding):
            object.__setattr__(request, "target", "data_ref")
            return StateMaterializationResult(
                request_id=request.request_id,
                state_ref=request.state_ref,
                target=request.target,
                value=DataRef(uri="memory://mutated"),
            )

    registry = EvaluationProviderRegistry()
    registry.register(MutatingProvider())
    request = StateMaterializationRequest(
        state_ref=StateRef(
            provider_id="mutating/v1",
            state_id="parameters",
            state_kind="model_parameters",
        )
    )

    with pytest.raises(
        EvaluationProviderContractError,
        match="mutated the bound state materialization request",
    ):
        EvaluationGateway(registry).materialize(
            request,
            ResourceContext(grant={"threads": 1, "gpus": 0}),
        )


def test_resource_grant_pool_partitions_and_rejects_parent_widening() -> None:
    parent = ResourceContext(
        namespace="project.case",
        threads=2,
        grant={
            "workers": 1,
            "threads": 2,
            "gpus": 0,
            "memory_mb": 256,
            "backend": "local",
            "capabilities": ["nested_eval"],
        },
        lease={"lease_id": "lease-parent", "fencing_token": 7},
    )
    pool = ResourceGrantPool(parent)

    with pytest.raises(ResourceSubgrantError, match="workers"):
        with pool.acquire(
            ResourceRequest(workers=2, threads=1, memory_mb=0),
            scope="child",
            namespace_suffix="too-wide",
        ):
            pass

    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first() -> None:
        with pool.acquire(
            ResourceRequest(
                workers=1,
                threads=2,
                memory_mb=256,
                capabilities=("nested_eval",),
            ),
            scope="child",
            namespace_suffix="first",
        ) as grant:
            assert grant.resource_context.lease["lease_id"] == "lease-parent"
            assert grant.resource_context.metadata["parent_lease_id"] == "lease-parent"
            first_entered.set()
            release_first.wait(timeout=2.0)

    def second() -> None:
        first_entered.wait(timeout=2.0)
        with pool.acquire(
            ResourceRequest(workers=1, threads=1, memory_mb=0),
            scope="child",
            namespace_suffix="second",
        ):
            second_entered.set()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    second_thread.start()
    assert first_entered.wait(timeout=2.0)
    time.sleep(0.05)
    assert not second_entered.is_set()
    release_first.set()
    first_thread.join(timeout=2.0)
    second_thread.join(timeout=2.0)
    assert second_entered.is_set()


def test_resource_grant_pool_enforces_compute_backend_and_device_subset() -> None:
    cpu_pool = ResourceGrantPool(
        ResourceContext(
            # Execution strategy and resource-provider backend are orthogonal.
            # A thread worker with no explicit provider backend is still local.
            execution_backend="thread",
            compute_backend="numpy",
            device="cpu",
            threads=2,
            grant={
                "workers": 1,
                "threads": 2,
                "gpus": 0,
                "memory_mb": 64,
                "compute_backend": "numpy",
                "device": "cpu",
            },
        )
    )

    with pytest.raises(ResourceSubgrantError, match="compute backend"):
        with cpu_pool.acquire(
            ResourceRequest(
                workers=1,
                threads=1,
                memory_mb=0,
                compute_backend="torch",
            ),
            scope="child",
            namespace_suffix="wrong-compute",
        ):
            pass

    with pytest.raises(ResourceSubgrantError, match="CPU-only"):
        with cpu_pool.acquire(
            ResourceRequest(
                workers=1,
                threads=1,
                memory_mb=0,
                device="cuda:0",
            ),
            scope="child",
            namespace_suffix="wrong-device",
        ):
            pass

    with pytest.raises(TypeError, match="resource_subgrant.metadata"):
        with cpu_pool.acquire(
            ResourceRequest(workers=1, threads=2, memory_mb=64),
            scope="child",
            namespace_suffix="bad-metadata",
            metadata={"unsafe": object()},
        ):
            pass
    with cpu_pool.acquire(
        ResourceRequest(workers=1, threads=2, memory_mb=64),
        scope="child",
        namespace_suffix="after-bad-metadata",
    ) as recovered:
        assert recovered.resources["threads"] == 2

    gpu_pool = ResourceGrantPool(
        ResourceContext(
            execution_backend="local",
            compute_backend="auto",
            device="cuda:0",
            threads=2,
            grant={
                "workers": 1,
                "threads": 2,
                "gpus": 1,
                "memory_mb": 64,
                "gpu_memory_mb": 128,
                "backend": "local",
                "compute_backend": "auto",
                "device": "cuda:0",
                "device_tokens": ["accelerator:gpu-0"],
                "resolved_devices": {"accelerator:gpu-0": "cuda:0"},
            },
        )
    )
    with gpu_pool.acquire(
        ResourceRequest(
            workers=1,
            threads=1,
            gpus=1,
            memory_mb=0,
            gpu_memory_mb=128,
            compute_backend="torch",
            device="cuda:0",
            device_tokens=("accelerator:gpu-0",),
        ),
        scope="child",
        namespace_suffix="gpu",
    ) as grant:
        assert grant.resource_context.compute_backend == "torch"
        assert grant.resource_context.device == "cuda:0"
        assert grant.resources["compute_backend"] == "torch"
        assert grant.resources["device"] == "cuda:0"

    with pytest.raises(ResourceSubgrantError, match="must be unique"):
        with gpu_pool.acquire(
            ResourceRequest(
                workers=1,
                threads=1,
                memory_mb=0,
                device_tokens=(
                    "accelerator:gpu-0",
                    "accelerator:gpu-0",
                ),
            ),
            scope="child",
            namespace_suffix="duplicate-token",
        ):
            pass

    with pytest.raises(ResourceSubgrantError, match="outside the parent device"):
        with gpu_pool.acquire(
            ResourceRequest(
                workers=1,
                threads=1,
                memory_mb=0,
                device="cuda:1",
            ),
            scope="child",
            namespace_suffix="wrong-physical-device",
        ):
            pass


def test_artifact_collection_requires_a_formal_data_ref_marker() -> None:
    output = {
        "artifact_refs": {
            "path": "C:/tmp/model.bin",
            "plain_mapping": {"uri": "memory://unpublished", "kind": "model"},
        }
    }
    assert collect_artifact_refs(output) == {}

    safe = make_transport_safe(
        {"artifact_refs": {"model": DataRef(uri="memory://published")}},
        path="output",
    )
    assert collect_artifact_refs(safe)["model"].uri == "memory://published"
