from __future__ import annotations

import json

import pytest

from blackbase.abc import AdapterBase
from blackbase.context import (
    RUNTIME_PROJECTION_AUDIT_MAX_BYTES,
    RUNTIME_PROJECTION_COMPONENT_MAX_BYTES,
    RUNTIME_PROJECTION_MAX_ISSUE_SAMPLES,
    RUNTIME_PROJECTION_MESSAGE_MAX_BYTES,
    RuntimeContextProjection,
    RuntimeProjectionComponent,
    RuntimeProjectionIssue,
    RuntimeProjectionIssueAccumulator,
    aggregate_runtime_projections,
)


def test_runtime_context_projection_preserves_mapping_compatibility_and_audit() -> None:
    issue = RuntimeProjectionIssue(component="child-2", reason="error")
    projection = RuntimeContextProjection(
        fields={"metric": 3},
        status="degraded",
        component_count=2,
        successful_component_count=1,
        failed_component_count=1,
        issue_samples=(issue,),
    )

    audit = projection.as_audit()
    assert dict(projection) == {"metric": 3}
    assert projection["metric"] == 3
    assert audit["status"] == "degraded"
    assert audit["failed_component_count"] == 1
    assert audit["issue_count"] == 1
    assert len(audit["audit_digest"]) == 64


def test_runtime_context_projection_rejects_unknown_or_inconsistent_status() -> None:
    with pytest.raises(ValueError, match="runtime projection status"):
        RuntimeContextProjection(status="healthy")

    with pytest.raises(ValueError, match="inconsistent with component counts"):
        RuntimeContextProjection(
            status="ok",
            component_count=1,
            failed_component_count=1,
            issue_samples=(RuntimeProjectionIssue("child", "error"),),
        )


def test_runtime_context_projection_rejects_incomplete_classification() -> None:
    with pytest.raises(ValueError, match="classifications must sum"):
        RuntimeContextProjection(
            status="ok",
            component_count=2,
            successful_component_count=1,
        )


def test_runtime_context_projection_requires_typed_issues_and_exact_issue_count() -> None:
    with pytest.raises(TypeError, match="RuntimeProjectionIssue"):
        RuntimeContextProjection(
            status="error",
            component_count=1,
            failed_component_count=1,
            issue_samples=({"component": "child", "reason": "error"},),
        )

    with pytest.raises(ValueError, match="issue_count must equal"):
        RuntimeContextProjection(
            status="error",
            component_count=1,
            failed_component_count=1,
        )


def test_runtime_projection_issue_fields_and_total_audit_are_bounded() -> None:
    accumulator = RuntimeProjectionIssueAccumulator()
    for index in range(32):
        accumulator.add(
            RuntimeProjectionIssue(
                component=f"child-{index}-" + ("组件" * 1_000),
                reason="internal_error_" + ("r" * 1_000),
                error_type="VeryLongError" + ("E" * 1_000),
                message="错误" * 10_000,
            )
        )

    projection = RuntimeContextProjection(
        status="error",
        component_count=32,
        failed_component_count=32,
        issue_samples=accumulator.samples,
        issue_sample_limit=accumulator.sample_limit,
        issue_count=accumulator.count,
        issue_digest=accumulator.digest,
        audit_truncated=accumulator.truncated,
    )
    audit = projection.as_audit()
    encoded = json.dumps(audit, ensure_ascii=False).encode("utf-8")

    assert accumulator.count == 32
    assert len(accumulator.samples) == RUNTIME_PROJECTION_MAX_ISSUE_SAMPLES
    assert audit["issue_count"] == 32
    assert audit["issue_sample_count"] < audit["issue_count"]
    assert audit["audit_truncated"] is True
    assert len(encoded) <= RUNTIME_PROJECTION_AUDIT_MAX_BYTES
    assert all(
        len(item["component"].encode("utf-8"))
        <= RUNTIME_PROJECTION_COMPONENT_MAX_BYTES
        for item in audit["issue_samples"]
    )
    assert all(
        len(item.get("message", "").encode("utf-8"))
        <= RUNTIME_PROJECTION_MESSAGE_MAX_BYTES
        for item in audit["issue_samples"]
    )


def test_runtime_projection_digest_is_stable_for_same_evidence() -> None:
    def build() -> RuntimeContextProjection:
        issue = RuntimeProjectionIssue(
            component="child",
            reason="error",
            error_type="TypeError",
            message="failed",
        )
        return RuntimeContextProjection(
            status="error",
            component_count=1,
            failed_component_count=1,
            issue_samples=(issue,),
        )

    first = build()
    second = build()

    assert first.issue_digest == second.issue_digest
    assert first.audit_digest == second.audit_digest
    assert first.as_audit() == second.as_audit()


def test_nested_cause_digest_propagates_into_parent_evidence() -> None:
    first_cause = "a" * 64
    second_cause = "b" * 64

    first_issue = RuntimeProjectionIssue(
        component="child",
        reason="nested_error",
        cause_digest=first_cause,
    )
    second_issue = RuntimeProjectionIssue(
        component="child",
        reason="nested_error",
        cause_digest=second_cause.upper(),
    )
    first = RuntimeContextProjection(
        status="error",
        component_count=1,
        failed_component_count=1,
        issue_samples=(first_issue,),
    )
    second = RuntimeContextProjection(
        status="error",
        component_count=1,
        failed_component_count=1,
        issue_samples=(second_issue,),
    )

    assert first_issue.as_dict()["cause_digest"] == first_cause
    assert second_issue.as_dict()["cause_digest"] == second_cause
    assert first_issue.digest != second_issue.digest
    assert first.issue_digest != second.issue_digest
    assert first.audit_digest != second.audit_digest


def test_runtime_projection_issue_rejects_invalid_cause_digest() -> None:
    with pytest.raises(ValueError, match="cause_digest"):
        RuntimeProjectionIssue(
            component="child",
            reason="nested_error",
            cause_digest="not-a-sha256-digest",
        )

    with pytest.raises(TypeError, match="cause_digest"):
        RuntimeProjectionIssue(
            component="child",
            reason="nested_error",
            cause_digest=None,  # type: ignore[arg-type]
        )


def test_runtime_projection_aggregation_invokes_children_once_and_propagates_health() -> None:
    calls = {"good": 0, "nested": 0}

    def good(_control):
        calls["good"] += 1
        return {"shared": "child", "good_metric": 3}

    nested_projection = RuntimeContextProjection(
        status="error",
        component_count=1,
        failed_component_count=1,
        issue_samples=(RuntimeProjectionIssue("grandchild", "error"),),
    )

    def nested(_control):
        calls["nested"] += 1
        return nested_projection

    aggregation = aggregate_runtime_projections(
        object(),
        (
            RuntimeProjectionComponent("child.good", good),
            RuntimeProjectionComponent("child.nested", nested),
            RuntimeProjectionComponent("child.unavailable"),
        ),
        fields={"shared": "owner"},
        field_sources={"shared": "owner"},
    )
    projection = aggregation.projection
    audit = projection.as_audit()

    assert calls == {"good": 1, "nested": 1}
    assert projection.status == "degraded"
    assert projection.component_count == 3
    assert projection.successful_component_count == 1
    assert projection.failed_component_count == 1
    assert projection.unavailable_component_count == 1
    assert projection["shared"] == "owner"
    assert projection["good_metric"] == 3
    assert dict(aggregation.field_sources) == {
        "shared": "owner",
        "good_metric": "child.good",
    }
    assert dict(projection.field_sources) == dict(aggregation.field_sources)
    assert audit["issue_samples"][0]["reason"] == "nested_error"
    assert audit["issue_samples"][0]["cause_digest"] == (
        nested_projection.audit_digest
    )


def test_runtime_projection_aggregation_rejects_invalid_child_result() -> None:
    aggregation = aggregate_runtime_projections(
        None,
        (RuntimeProjectionComponent("child.invalid", lambda _control: []),),
    )

    projection = aggregation.projection
    assert projection.status == "error"
    assert projection.invalid_component_count == 1
    assert projection.as_audit()["issue_samples"][0]["reason"] == "invalid_result"


def test_nested_runtime_projection_preserves_leaf_field_writer() -> None:
    leaf = RuntimeContextProjection(
        fields={"leaf_metric": 7},
        field_sources={"leaf_metric": "adapter.leaf"},
    )
    middle = aggregate_runtime_projections(
        None,
        (RuntimeProjectionComponent("adapter.middle", lambda _control: leaf),),
    ).projection
    root = aggregate_runtime_projections(
        None,
        (RuntimeProjectionComponent("adapter.root", lambda _control: middle),),
    ).projection

    assert dict(root.field_sources) == {"leaf_metric": "adapter.leaf"}
    assert root.field_source_digest == middle.field_source_digest
    assert root.as_audit()["field_source_count"] == 1
    assert root.as_audit()["field_source_digest"] == root.field_source_digest


def test_field_writer_digest_is_separate_from_health_digest() -> None:
    first = RuntimeContextProjection(
        fields={"metric": 1},
        field_sources={"metric": "adapter.first"},
    )
    second = RuntimeContextProjection(
        fields={"metric": 1},
        field_sources={"metric": "adapter.second"},
    )

    assert first.audit_digest == second.audit_digest
    assert first.field_source_digest != second.field_source_digest


def test_runtime_projection_rejects_source_for_absent_field_and_freezes_maps() -> None:
    with pytest.raises(ValueError, match="absent field"):
        RuntimeContextProjection(field_sources={"missing": "adapter.child"})

    projection = RuntimeContextProjection(
        fields={"metric": 1},
        field_sources={"metric": "adapter.child"},
    )
    with pytest.raises(TypeError):
        projection.fields["metric"] = 2  # type: ignore[index]
    with pytest.raises(TypeError):
        projection.field_sources["metric"] = "rewritten"  # type: ignore[index]


def test_adapter_base_runtime_projection_defaults_to_empty_mapping() -> None:
    class MinimalAdapter(AdapterBase):
        def propose(self, control, context):
            del control, context
            return []

        def update(self, control, candidates, feedback, context):
            del control, candidates, feedback, context

    assert MinimalAdapter().get_runtime_context_projection(None) == {}
