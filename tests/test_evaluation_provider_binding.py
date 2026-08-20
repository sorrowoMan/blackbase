from __future__ import annotations

import json

import numpy as np
import pytest

from blackbase.evaluation import (
    CapabilityRequirement,
    EvaluationBinding,
    EvaluationGateway,
    EvaluationProviderContractError,
    EvaluationProviderRegistry,
    EvaluationProviderSpec,
    EvaluationProviderUnavailable,
    EvaluationRequest,
    EvaluationResult,
    StateMaterializationRequest,
    StateMaterializationResult,
    StateReleaseRequest,
    StateReleaseResult,
    StateTransitionRequest,
    StateTransitionResult,
    StateTransitionMethodSpec,
    StateVersionConflict,
)
from blackbase.resources import ResourceContext, ResourceRequirement
from blackbase.state_ref import StateRef
from blackbase.types import (
    CandidateBatch,
    Feedback,
    UnknownState,
    decode_shared_value,
    encode_shared_value,
)


class RecordingProvider:
    def __init__(self, spec: EvaluationProviderSpec, *, fail: bool = False) -> None:
        self.spec = spec
        self.calls = 0
        self.fail = fail

    def evaluate(self, request, binding):
        self.calls += 1
        if self.fail:
            raise TypeError("provider body failed")
        return EvaluationResult(
            request_id=request.request_id,
            feedback=tuple(
                Feedback(objectives=np.asarray([float(index)]))
                for index, _ in enumerate(request.states)
            ),
            result_states=tuple(request.states),
        )


class TransitionRecordingProvider(RecordingProvider):
    def __init__(
        self,
        spec: EvaluationProviderSpec,
        *,
        invalid_version: bool = False,
        fail: bool = False,
    ) -> None:
        super().__init__(spec)
        self.transition_calls = 0
        self.invalid_version = invalid_version
        self.transition_fail = fail

    def transition(self, request, binding):
        self.transition_calls += 1
        if self.transition_fail:
            raise TypeError("transition body failed")
        next_version = (
            request.state_ref.version + 2
            if self.invalid_version
            else request.state_ref.version + 1
        )
        successor = StateRef(
            provider_id=request.state_ref.provider_id,
            state_id=request.state_ref.state_id,
            state_kind=request.state_ref.state_kind,
            scope_id=request.state_ref.scope_id,
            trajectory_id=request.state_ref.trajectory_id,
            device=request.state_ref.device,
            version=next_version,
            transport_scope=request.state_ref.transport_scope,
        )
        previous_m = request.slot_refs.get("m")
        next_m = (
            StateRef(
                provider_id=request.state_ref.provider_id,
                state_id="adam-m",
                state_kind="optimizer_slot",
                scope_id=request.state_ref.scope_id,
                trajectory_id=request.state_ref.trajectory_id,
                device=request.state_ref.device,
                version=0,
            )
            if previous_m is None
            else previous_m.next_version()
        )
        return StateTransitionResult(
            request_id=request.request_id,
            method_id=request.method_id,
            status="applied",
            state_ref=successor,
            slot_refs={"m": next_m},
            metrics={"update_norm": 0.25},
        )


class MaterializationRecordingProvider(RecordingProvider):
    def __init__(self, spec: EvaluationProviderSpec) -> None:
        super().__init__(spec)
        self.materialization_calls = 0

    def materialize(self, request, binding):
        self.materialization_calls += 1
        return StateMaterializationResult(
            request_id=request.request_id,
            state_ref=request.state_ref,
            target=request.target,
            value=UnknownState(
                [1.0, 2.0],
                metadata={"source": "provider", "version": request.state_ref.version},
            ),
        )


class ReleaseRecordingProvider(RecordingProvider):
    def release(self, request, binding):
        return StateReleaseResult(
            request_id=request.request_id,
            provider_id=request.provider_id,
            status="released",
            released_count=2,
            released_state_ids=("m", "v"),
        )


def _request(*capabilities) -> EvaluationRequest:
    return EvaluationRequest(
        problem_id="neural.cross_entropy/v1",
        states=(UnknownState([1.0, 2.0]),),
        mode="train",
        capabilities=capabilities,
    )


def test_state_ref_is_wire_safe_but_not_presented_as_an_artifact() -> None:
    ref = StateRef(
        provider_id="torch.autograd/v1",
        state_id="parameter-set-7",
        state_kind="model_parameters",
        scope_id="case-1.run-2",
        trajectory_id="trajectory-9",
        device="cuda:0",
        version=3,
        transport_scope="process",
    )

    payload = encode_shared_value({"state": ref})
    json.dumps(payload)
    restored = decode_shared_value(payload)["state"]

    assert restored == ref
    assert restored.process_local is True
    assert restored.as_dict()["protocol_type"] == "blackbase.state_ref"
    with pytest.raises(TypeError):
        restored.metadata["rewritten"] = True


def test_candidate_batch_keeps_semantic_and_numeric_views_aligned() -> None:
    batch = CandidateBatch.from_candidates(
        (
            UnknownState([1.0, 2.0], metadata={"architecture": "a"}),
            UnknownState([1.0, 2.0], metadata={"architecture": "b"}),
        ),
        candidate_tokens=("candidate-a", "candidate-b"),
    )

    assert np.array_equal(batch.numeric_matrix[0], batch.numeric_matrix[1])
    assert batch.semantic_states[0].metadata != batch.semantic_states[1].metadata
    assert batch.candidate_tokens == ("candidate-a", "candidate-b")
    restored = decode_shared_value(encode_shared_value(batch))
    assert isinstance(restored, CandidateBatch)
    assert restored.semantic_states[1].metadata["architecture"] == "b"


def test_candidate_batch_views_remain_immutable_after_validation() -> None:
    source_values = np.array([1.0, 2.0])
    source_metadata = {"architecture": {"layers": [2, 4]}}
    batch = CandidateBatch.from_candidates(
        (UnknownState(source_values, metadata=source_metadata),),
        candidate_tokens=("candidate:1",),
    )

    source_values[0] = 99.0
    source_metadata["architecture"]["layers"][0] = 99
    assert batch.semantic_states[0].as_array().tolist() == [1.0, 2.0]
    assert batch.numeric_matrix.tolist() == [[1.0, 2.0]]
    assert tuple(batch.semantic_states[0].metadata["architecture"]["layers"]) == (2, 4)

    with pytest.raises(ValueError):
        batch.semantic_states[0].values[0] = 9.0
    with pytest.raises(ValueError):
        batch.numeric_matrix[0, 0] = 9.0
    with pytest.raises(TypeError):
        batch.semantic_states[0].metadata["architecture"] = {"layers": [9]}
    with pytest.raises(TypeError):
        batch.semantic_states[0].metadata["architecture"]["layers"] = (9,)

    with pytest.raises(ValueError):
        batch.semantic_states[0].values.setflags(write=True)
    with pytest.raises(ValueError):
        batch.numeric_matrix.setflags(write=True)


def test_state_release_is_bound_to_provider_scope_and_trajectory() -> None:
    provider = ReleaseRecordingProvider(
        EvaluationProviderSpec(provider_id="torch.release/v1", compute_backend="torch")
    )
    registry = EvaluationProviderRegistry()
    registry.register(provider)
    request = StateReleaseRequest(
        provider_id=provider.spec.provider_id,
        scope_id="case.run",
        trajectory_id="trajectory-9",
    )
    result = EvaluationGateway(registry).release(
        request,
        ResourceContext(
            compute_backend="torch",
            namespace="case.run",
            grant={"threads": 1, "gpus": 0},
        ),
    )

    assert result.status == "released"
    assert result.released_count == 2
    assert result.binding is not None
    assert result.binding.audit["operation"] == "state_release"
    assert decode_shared_value(encode_shared_value(request)) == request


def test_feedback_can_carry_a_provider_owned_gradient_ref() -> None:
    gradient_ref = StateRef(
        provider_id="torch.autograd/v1",
        state_id="gradient-7",
        state_kind="gradient",
        device="cuda:0",
    )
    feedback = Feedback(
        objectives=[0.5],
        loss=0.5,
        gradient_ref=gradient_ref,
    )

    restored = Feedback.from_dict(feedback.as_dict())

    assert restored.gradients is None
    assert restored.gradient_ref == gradient_ref


def test_state_materialization_is_versioned_bound_and_wire_safe() -> None:
    provider = MaterializationRecordingProvider(
        EvaluationProviderSpec(
            provider_id="torch.materialize/v1",
            supported_devices=("cpu",),
            compute_backend="torch",
            state_kinds=("model_parameters",),
            materialization_targets=("unknown_state",),
        )
    )
    registry = EvaluationProviderRegistry()
    registry.register(provider)
    state_ref = StateRef(
        provider_id=provider.spec.provider_id,
        state_id="parameters-1",
        state_kind="model_parameters",
        scope_id="case.run",
        trajectory_id="fit-7",
        device="cpu",
        version=3,
    )
    request = StateMaterializationRequest(
        state_ref=state_ref,
        release_after=True,
    )

    result = EvaluationGateway(registry).materialize(
        request,
        ResourceContext(
            compute_backend="torch",
            device="cpu",
            grant={"threads": 1, "gpus": 0},
        ),
    )

    assert isinstance(result.value, UnknownState)
    assert np.allclose(result.value.as_array(), [1.0, 2.0])
    assert result.binding is not None
    assert result.binding.audit["operation"] == "state_materialization"
    assert result.binding.audit["allocation_performed"] is False
    assert provider.materialization_calls == 1

    restored_request = decode_shared_value(encode_shared_value(request))
    restored_result = decode_shared_value(encode_shared_value(result))
    assert restored_request == request
    assert restored_request.release_after is True
    assert isinstance(restored_result, StateMaterializationResult)
    assert np.allclose(restored_result.value.as_array(), result.value.as_array())


def test_provider_spec_rejects_unrequestable_materialization_target() -> None:
    with pytest.raises(ValueError, match="unsupported state materialization targets"):
        EvaluationProviderSpec(
            provider_id="unsafe.materializer/v1",
            materialization_targets=("python_pickle",),
        )


def test_registry_prefers_gpu_inside_an_existing_gpu_grant() -> None:
    provider = RecordingProvider(
        EvaluationProviderSpec(
            provider_id="torch.autograd/v1",
            capabilities=("autograd.backward",),
            supported_devices=("cpu", "gpu"),
            preferred_devices=("gpu", "cpu"),
            compute_backend="torch",
            modes=("train", "evaluate"),
        )
    )
    registry = EvaluationProviderRegistry()
    registry.register(provider)
    request = _request("autograd.backward")
    grant = ResourceContext.from_mapping(
        {
            "compute_backend": "torch",
            "device": "cuda:0",
            "threads": 4,
            "grant": {
                "backend": "local",
                "threads": 4,
                "gpus": 1,
                "device_tokens": ["cuda:0"],
            },
        }
    )

    result = EvaluationGateway(registry).evaluate(request, grant)

    assert result.binding is not None
    assert result.binding.device == "cuda:0"
    assert result.binding.audit["allocation_performed"] is False
    assert result.binding.resource_context.as_dict() == grant.as_dict()
    assert provider.calls == 1


def test_one_provider_can_fall_back_to_cpu_without_adapter_changes() -> None:
    provider = RecordingProvider(
        EvaluationProviderSpec(
            provider_id="torch.autograd/v1",
            capabilities=("autograd.backward",),
            supported_devices=("cpu", "gpu"),
            preferred_devices=("gpu", "cpu"),
            compute_backend="torch",
            modes=("train",),
        )
    )
    registry = EvaluationProviderRegistry()
    registry.register(provider)

    bound = registry.bind(
        _request("autograd.backward"),
        ResourceContext(
            compute_backend="torch",
            device="cpu",
            threads=2,
            grant={"backend": "local", "threads": 2, "gpus": 0},
        ),
    )

    assert bound.binding.device == "cpu"
    assert bound.binding.degraded is True
    assert bound.binding.audit["device_fallback"] is True


def test_required_semantic_capability_is_matched_against_provider_not_l0() -> None:
    registry = EvaluationProviderRegistry()
    registry.register(
        RecordingProvider(
            EvaluationProviderSpec(
                provider_id="numpy.loss/v1",
                capabilities=("loss.forward",),
                supported_devices=("cpu",),
                modes=("train",),
            )
        )
    )

    with pytest.raises(EvaluationProviderUnavailable) as error:
        registry.bind(
            _request("autograd.backward"),
            ResourceContext(grant={"threads": 1, "gpus": 0}),
        )

    assert "missing required capabilities" in error.value.rejections["numpy.loss/v1"]


def test_registry_does_not_bind_a_provider_configured_for_another_problem() -> None:
    wrong = RecordingProvider(
        EvaluationProviderSpec(
            provider_id="torch.problem-a/v1",
            problem_ids=("problem.a/v1",),
            capabilities=("autograd.backward",),
            modes=("train",),
            priority=100,
        )
    )
    correct = RecordingProvider(
        EvaluationProviderSpec(
            provider_id="torch.problem-b/v1",
            problem_ids=("neural.cross_entropy/v1",),
            capabilities=("autograd.backward",),
            modes=("train",),
        )
    )
    registry = EvaluationProviderRegistry()
    registry.register(wrong)
    registry.register(correct)

    bound = registry.bind(
        _request("autograd.backward"),
        ResourceContext(grant={"threads": 1, "gpus": 0}),
    )

    assert bound.binding.provider_id == "torch.problem-b/v1"


def test_problem_identity_binding_is_case_normalized() -> None:
    provider = RecordingProvider(
        EvaluationProviderSpec(
            provider_id="torch.case-normalized/v1",
            problem_ids=("Neural.Cross_Entropy/V1",),
            capabilities=("autograd.backward",),
            modes=("train",),
        )
    )
    registry = EvaluationProviderRegistry()
    registry.register(provider)
    request = EvaluationRequest(
        problem_id="NEURAL.CROSS_ENTROPY/V1",
        states=(UnknownState([1.0]),),
        mode="train",
        capabilities=("autograd.backward",),
    )

    bound = registry.bind(
        request,
        ResourceContext(grant={"threads": 1, "gpus": 0}),
    )

    assert bound.binding.provider_id == provider.spec.provider_id


def test_gpu_only_provider_cannot_mint_gpu_outside_l0_grant() -> None:
    registry = EvaluationProviderRegistry()
    registry.register(
        RecordingProvider(
            EvaluationProviderSpec(
                provider_id="cuda.kernel/v1",
                capabilities=("loss.forward",),
                resource_requirement=ResourceRequirement(gpus=1),
                supported_devices=("gpu",),
                modes=("train",),
            )
        )
    )

    with pytest.raises(EvaluationProviderUnavailable) as error:
        registry.bind(
            _request("loss.forward"),
            ResourceContext(device="cpu", grant={"threads": 1, "gpus": 0}),
        )

    assert "requires 1 GPUs" in error.value.rejections["cuda.kernel/v1"]


def test_state_ref_can_only_bind_to_its_owner_provider() -> None:
    owner = RecordingProvider(
        EvaluationProviderSpec(
            provider_id="torch.autograd/v1",
            capabilities=("loss.forward",),
            state_kinds=("model_parameters",),
            priority=0,
        )
    )
    unrelated = RecordingProvider(
        EvaluationProviderSpec(
            provider_id="other/v1",
            capabilities=("loss.forward",),
            priority=100,
        )
    )
    registry = EvaluationProviderRegistry()
    registry.register(owner)
    registry.register(unrelated)
    request = EvaluationRequest(
        problem_id="neural.cross_entropy/v1",
        states=(
            StateRef(
                provider_id="torch.autograd/v1",
                state_id="parameters-1",
                state_kind="model_parameters",
            ),
        ),
        capabilities=("loss.forward",),
    )

    bound = registry.bind(request, ResourceContext(grant={"threads": 1, "gpus": 0}))

    assert bound.binding.provider_id == "torch.autograd/v1"


def test_preferred_capability_is_audited_without_becoming_a_hard_requirement() -> None:
    provider = RecordingProvider(
        EvaluationProviderSpec(
            provider_id="cpu.loss/v1",
            capabilities=("loss.forward",),
            supported_devices=("cpu",),
            modes=("train",),
        )
    )
    registry = EvaluationProviderRegistry()
    registry.register(provider)
    request = _request(
        "loss.forward",
        CapabilityRequirement("autograd.backward", policy="preferred"),
    )

    bound = registry.bind(request, ResourceContext(grant={"threads": 1, "gpus": 0}))

    assert bound.binding.degraded is True
    assert bound.binding.missing_preferred_capabilities == ("autograd.backward",)


def test_request_and_result_round_trip_state_and_binding_envelopes() -> None:
    request = EvaluationRequest(
        problem_id="linear.least_squares/v1",
        states=(UnknownState([1.0]), {"symbol": "beta"}),
        capabilities=("loss.forward",),
        payload={"batch": [1, 2, 3]},
        request_id="eval-fixed",
    )
    restored_request = decode_shared_value(encode_shared_value(request))

    assert isinstance(restored_request, EvaluationRequest)
    assert isinstance(restored_request.states[0], UnknownState)
    assert restored_request.states[1] == {"symbol": "beta"}
    assert restored_request.required_capabilities == ("loss.forward",)

    binding = EvaluationBinding(
        request_id=request.request_id,
        provider_id="numpy.loss/v1",
        resource_context=ResourceContext(grant={"threads": 1, "gpus": 0}),
        device="cpu",
        compute_backend="numpy",
    )
    result = EvaluationResult(
        request_id=request.request_id,
        feedback=(Feedback(objectives=[0.5], loss=0.5), Feedback(objectives=[0.4])),
        result_states=(
            StateRef(provider_id="numpy.loss/v1", state_id="state-1"),
            None,
        ),
        binding=binding,
    )
    restored_result = decode_shared_value(encode_shared_value(result))

    assert isinstance(restored_result, EvaluationResult)
    assert restored_result.binding == binding
    assert isinstance(restored_result.result_states[0], StateRef)


def test_provider_internal_type_error_is_not_retried() -> None:
    provider = RecordingProvider(
        EvaluationProviderSpec(
            provider_id="broken/v1",
            capabilities=("loss.forward",),
            modes=("train",),
        ),
        fail=True,
    )
    registry = EvaluationProviderRegistry()
    registry.register(provider)
    request = _request("loss.forward")
    bound = registry.bind(
        request,
        ResourceContext(grant={"threads": 1, "gpus": 0}),
    )

    with pytest.raises(TypeError, match="provider body failed"):
        bound.evaluate(request)

    assert provider.calls == 1


def test_bound_evaluation_state_cannot_be_mutated_after_binding() -> None:
    provider = RecordingProvider(
        EvaluationProviderSpec(
            provider_id="numpy.loss/v1",
            capabilities=("loss.forward",),
            modes=("evaluate",),
        )
    )
    registry = EvaluationProviderRegistry()
    registry.register(provider)
    state = UnknownState([1.0])
    request = EvaluationRequest(
        problem_id="mutable/v1",
        states=(state,),
        capabilities=("loss.forward",),
    )
    bound = registry.bind(
        request,
        ResourceContext(grant={"threads": 1, "gpus": 0}),
    )
    with pytest.raises(ValueError):
        state.values[0] = 2.0

    result = bound.evaluate(request)

    assert len(result.feedback) == 1
    assert provider.calls == 1


def test_bound_provider_rejects_wrong_feedback_cardinality() -> None:
    class BadCardinalityProvider(RecordingProvider):
        def evaluate(self, request, binding):
            self.calls += 1
            return EvaluationResult(
                request_id=request.request_id,
                feedback=(Feedback(objectives=[1.0]),),
            )

    provider = BadCardinalityProvider(
        EvaluationProviderSpec(
            provider_id="bad-cardinality/v1",
            capabilities=("loss.forward",),
            modes=("evaluate",),
        )
    )
    registry = EvaluationProviderRegistry()
    registry.register(provider)
    request = EvaluationRequest(
        problem_id="batch/v1",
        states=(UnknownState([1.0]), UnknownState([2.0])),
        capabilities=("loss.forward",),
    )
    bound = registry.bind(request, ResourceContext(grant={"threads": 1, "gpus": 0}))

    with pytest.raises(EvaluationProviderContractError, match="2 states"):
        bound.evaluate(request)


def test_state_transition_executes_owner_kernel_and_versions_state_and_slots() -> None:
    provider = TransitionRecordingProvider(
        EvaluationProviderSpec(
            provider_id="torch.autograd/v1",
            capabilities=("autograd.backward",),
            supported_devices=("cpu", "gpu"),
            preferred_devices=("gpu", "cpu"),
            compute_backend="torch",
            state_kinds=("model_parameters",),
            transition_methods=(
                StateTransitionMethodSpec(
                    method_id="gradient.adam",
                    required_operands=("gradient",),
                    operand_state_kinds={"gradient": ("gradient",)},
                    required_parameters=("learning_rate",),
                    optional_parameters=("beta1", "beta2"),
                    optional_slots=("m", "v"),
                    result_slots=("m",),
                    slot_state_kinds={"m": ("optimizer_slot",)},
                    result_slot_state_kinds={"m": ("optimizer_slot",)},
                ),
                "gradient.sgd",
            ),
        )
    )
    registry = EvaluationProviderRegistry()
    registry.register(provider)
    restored_spec = decode_shared_value(encode_shared_value(provider.spec))
    assert isinstance(restored_spec, EvaluationProviderSpec)
    assert restored_spec.transition_method_ids == (
        "gradient.adam",
        "gradient.sgd",
    )
    restored_adam = restored_spec.transition_method("gradient.adam")
    assert restored_adam is not None
    assert restored_adam.operand_state_kinds["gradient"] == ("gradient",)
    assert restored_adam.result_slot_state_kinds["m"] == ("optimizer_slot",)
    state_ref = StateRef(
        provider_id="torch.autograd/v1",
        state_id="parameters-1",
        state_kind="model_parameters",
        scope_id="case.run",
        trajectory_id="fit-7",
        device="cuda:0",
        version=4,
    )
    gradient_ref = StateRef(
        provider_id="torch.autograd/v1",
        state_id="gradient-1",
        state_kind="gradient",
        scope_id="case.run",
        trajectory_id="fit-7",
        device="cuda:0",
    )
    request = StateTransitionRequest(
        state_ref=state_ref,
        method_id="gradient.adam",
        operands={"gradient": gradient_ref},
        parameters={"learning_rate": 1e-3, "beta1": 0.9, "beta2": 0.999},
        step_index=5,
    )
    grant = ResourceContext(
        compute_backend="torch",
        device="cuda:0",
        threads=2,
        grant={
            "backend": "local",
            "threads": 2,
            "gpus": 1,
            "device_tokens": ["cuda:0"],
        },
    )

    result = EvaluationGateway(registry).transition(request, grant)

    assert result.state_ref.version == 5
    assert result.slot_refs["m"].provider_id == "torch.autograd/v1"
    assert result.slot_refs["m"].trajectory_id == "fit-7"
    assert result.binding is not None
    assert result.binding.audit["method_id"] == "gradient.adam"
    assert result.binding.audit["allocation_performed"] is False
    assert provider.transition_calls == 1

    restored = decode_shared_value(encode_shared_value(result))
    assert isinstance(restored, StateTransitionResult)
    assert restored.state_ref == result.state_ref


def test_transition_rejects_slots_from_another_trajectory() -> None:
    state_ref = StateRef(
        provider_id="torch.autograd/v1",
        state_id="parameters-1",
        state_kind="model_parameters",
        scope_id="case.run",
        trajectory_id="fit-a",
    )
    foreign_slot = StateRef(
        provider_id=state_ref.provider_id,
        state_id="adam-m",
        state_kind="optimizer_slot.m",
        scope_id=state_ref.scope_id,
        trajectory_id="fit-b",
    )

    with pytest.raises(ValueError, match="another StateRef trajectory"):
        StateTransitionRequest(
            state_ref=state_ref,
            method_id="gradient.adam",
            operands={"gradient": [1.0]},
            slot_refs={"m": foreign_slot},
        )


def test_state_transition_request_detaches_inline_gradient_and_round_trips() -> None:
    gradient = np.asarray([1.0, 2.0])
    request = StateTransitionRequest(
        state_ref=StateRef(
            provider_id="numpy.gradient/v1",
            state_id="parameters-1",
            state_kind="model_parameters",
        ),
        method_id="gradient.sgd",
        operands={"gradient": gradient},
        parameters={"learning_rate": 0.1},
    )
    gradient[0] = 999.0

    detached = request.operands["gradient"]
    assert np.allclose(detached, [1.0, 2.0])
    assert detached.flags.writeable is False

    restored = decode_shared_value(encode_shared_value(request))
    assert isinstance(restored, StateTransitionRequest)
    assert restored.method_id == "gradient.sgd"
    assert restored.operands["gradient"] == (1.0, 2.0)


def test_transition_method_must_be_declared_by_owner_provider() -> None:
    provider = TransitionRecordingProvider(
        EvaluationProviderSpec(
            provider_id="numpy.gradient/v1",
            transition_methods=("gradient.sgd",),
            state_kinds=("model_parameters",),
        )
    )
    registry = EvaluationProviderRegistry()
    registry.register(provider)
    request = StateTransitionRequest(
        state_ref=StateRef(
            provider_id="numpy.gradient/v1",
            state_id="parameters-1",
            state_kind="model_parameters",
        ),
        method_id="gradient.adam",
        operands={"gradient": [1.0]},
    )

    with pytest.raises(EvaluationProviderUnavailable) as error:
        registry.bind_transition(
            request,
            ResourceContext(grant={"threads": 1, "gpus": 0}),
        )

    assert "is not declared" in error.value.rejections["numpy.gradient/v1"]


def test_structured_transition_method_rejects_missing_operand_before_execution() -> None:
    provider = TransitionRecordingProvider(
        EvaluationProviderSpec(
            provider_id="numpy.structured/v1",
            state_kinds=("model_parameters",),
            transition_methods=(
                StateTransitionMethodSpec(
                    method_id="gradient.sgd",
                    required_operands=("gradient",),
                    required_parameters=("learning_rate",),
                ),
            ),
        )
    )
    registry = EvaluationProviderRegistry()
    registry.register(provider)
    request = StateTransitionRequest(
        state_ref=StateRef(
            provider_id="numpy.structured/v1",
            state_id="parameters-1",
            state_kind="model_parameters",
        ),
        method_id="gradient.sgd",
        parameters={"learning_rate": 0.1},
    )

    with pytest.raises(EvaluationProviderUnavailable) as error:
        registry.bind_transition(
            request,
            ResourceContext(grant={"threads": 1, "gpus": 0}),
        )

    assert "missing required operands" in error.value.rejections[
        "numpy.structured/v1"
    ]
    assert provider.transition_calls == 0


def test_structured_transition_method_rejects_wrong_operand_state_kind() -> None:
    provider = TransitionRecordingProvider(
        EvaluationProviderSpec(
            provider_id="numpy.typed/v1",
            state_kinds=("model_parameters",),
            transition_methods=(
                StateTransitionMethodSpec(
                    method_id="gradient.sgd",
                    required_operands=("gradient",),
                    operand_state_kinds={"gradient": ("gradient",)},
                    required_parameters=("learning_rate",),
                ),
            ),
        )
    )
    registry = EvaluationProviderRegistry()
    registry.register(provider)
    state_ref = StateRef(
        provider_id=provider.spec.provider_id,
        state_id="parameters-1",
        state_kind="model_parameters",
        scope_id="run.scope",
        trajectory_id="trajectory-1",
    )
    request = StateTransitionRequest(
        state_ref=state_ref,
        method_id="gradient.sgd",
        operands={
            "gradient": StateRef(
                provider_id=provider.spec.provider_id,
                state_id="direction-1",
                state_kind="search_direction",
                scope_id=state_ref.scope_id,
                trajectory_id=state_ref.trajectory_id,
            )
        },
        parameters={"learning_rate": 0.1},
    )

    with pytest.raises(EvaluationProviderUnavailable) as error:
        registry.bind_transition(
            request,
            ResourceContext(grant={"threads": 1, "gpus": 0}),
        )

    assert "unsupported StateRef kinds" in error.value.rejections[provider.spec.provider_id]
    assert provider.transition_calls == 0


def test_transition_provider_cannot_skip_version_fence_or_retry_type_error() -> None:
    invalid = TransitionRecordingProvider(
        EvaluationProviderSpec(
            provider_id="numpy.invalid/v1",
            transition_methods=("gradient.sgd",),
            state_kinds=("model_parameters",),
        ),
        invalid_version=True,
    )
    registry = EvaluationProviderRegistry()
    registry.register(invalid)
    request = StateTransitionRequest(
        state_ref=StateRef(
            provider_id="numpy.invalid/v1",
            state_id="parameters-1",
            state_kind="model_parameters",
        ),
        method_id="gradient.sgd",
        operands={"gradient": [1.0]},
    )
    bound = registry.bind_transition(
        request,
        ResourceContext(grant={"threads": 1, "gpus": 0}),
    )

    with pytest.raises(EvaluationProviderContractError, match="increment"):
        bound.execute(request)
    assert invalid.transition_calls == 1

    failing = TransitionRecordingProvider(
        EvaluationProviderSpec(
            provider_id="numpy.failing/v1",
            transition_methods=("gradient.sgd",),
            state_kinds=("model_parameters",),
        ),
        fail=True,
    )
    failing_registry = EvaluationProviderRegistry()
    failing_registry.register(failing)
    failing_request = StateTransitionRequest(
        state_ref=StateRef(
            provider_id="numpy.failing/v1",
            state_id="parameters-1",
            state_kind="model_parameters",
        ),
        method_id="gradient.sgd",
        operands={"gradient": [1.0]},
    )
    failing_bound = failing_registry.bind_transition(
        failing_request,
        ResourceContext(grant={"threads": 1, "gpus": 0}),
    )

    with pytest.raises(TypeError, match="transition body failed"):
        failing_bound.execute(failing_request)
    assert failing.transition_calls == 1


def test_state_version_conflict_preserves_expected_and_actual_versions() -> None:
    state_ref = StateRef(
        provider_id="torch.autograd/v1",
        state_id="parameters-1",
        version=4,
    )

    conflict = StateVersionConflict(state_ref, actual_version=5)

    assert conflict.state_ref == state_ref
    assert conflict.expected_version == 4
    assert conflict.actual_version == 5
    assert "expected=4" in str(conflict)
