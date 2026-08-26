from __future__ import annotations

from blackbase.project import CaseRunRequest, CaseRunResult
from blackbase.project.run_manifest import ProjectRunManifest
from blackbase.resources import CancellationRef


def _legacy_ref_payload(ref: CancellationRef) -> dict[str, object]:
    payload = ref.as_dict()
    for key in (
        "parent_control_id",
        "root_control_id",
        "lineage_depth",
        "lineage_digest",
        "active_ttl_seconds",
        "heartbeat_seconds",
        "retention_seconds",
    ):
        payload.pop(key, None)
    return payload


def test_manifest_resume_compacts_completed_case_run_v1_lineage() -> None:
    request = CaseRunRequest(
        project_name="project",
        stage_name="stage",
        case_name="case",
    )
    result = CaseRunResult(request=request, status="succeeded", output={"value": 3})
    payload = result.as_dict()
    payload["schema_version"] = 1
    payload["request"]["schema_version"] = 1
    current = CancellationRef(deadline_at=20.0)
    ancestor = CancellationRef(deadline_at=10.0)
    payload["request"]["control"]["cancellation"] = _legacy_ref_payload(current)
    payload["request"]["control"]["ancestor_cancellations"] = [
        _legacy_ref_payload(ancestor)
    ]
    manifest = ProjectRunManifest(
        run_id="run",
        project_name="project",
        group="default",
        framework="blackbase",
        config_fingerprint="fingerprint",
        cases=({"result": payload},),
    )

    restored = manifest.successful_cases()[("stage", "case")]

    assert restored.status == "resumed"
    assert restored.schema_version == 3
    assert restored.request.schema_version == 3
    assert restored.control.ancestor_cancellations == ()
    assert restored.control.deadline_at == 10.0
    evidence = restored.control.metadata["historical_v1_lineage"]
    assert evidence["ancestor_count"] == 1
    assert len(evidence["digest"]) == 64
    assert restored.metadata["case_run_schema_migration"]["from"] == 1
    assert restored.metadata["case_run_schema_migration"]["to"] == 3
