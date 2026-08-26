from __future__ import annotations

from pathlib import Path

import pytest

from blackbase.project import CaseRunRequest, add_case, create_project, execute_project
from blackbase.resources import (
    ArtifactBinding,
    ArtifactAuthority,
    ArtifactPublicationError,
    ArtifactPublisher,
)


def test_artifact_name_reservation_and_commit_are_durable_and_idempotent(
    tmp_path,
) -> None:
    authority = ArtifactAuthority(root=str(tmp_path / "artifacts"), namespace="test")
    first = ArtifactPublisher(
        authority,
        project_run_id="run-1",
        case_run_id="case-1",
    )
    competing = ArtifactPublisher(
        authority,
        project_run_id="run-1",
        case_run_id="case-1",
    )
    first.reserve_publication("model", "transaction-1")
    with pytest.raises(ArtifactPublicationError, match="reserved"):
        competing.reserve_publication("model", "transaction-2")

    ref = first.publish("model", {"weights": [1, 2, 3]}, serializer="json")
    receipts = first.commit_publications("transaction-1", {"model": ref})
    receipt = receipts["model"]
    assert first.verify_publication(receipt) is True
    with pytest.raises(ValueError, match="Case-finalization-sealed"):
        ArtifactBinding(ref=ref, publication=receipt)
    assert first.commit_publications("transaction-1", {"model": ref}) == receipts

    Path(ref.uri).write_text("tampered", encoding="utf-8")
    assert first.verify_publication(receipt) is False


def test_artifact_commit_rejects_content_tampered_before_receipt(tmp_path) -> None:
    authority = ArtifactAuthority(root=str(tmp_path / "artifacts"), namespace="test")
    publisher = ArtifactPublisher(
        authority,
        project_run_id="run-1",
        case_run_id="case-1",
    )
    publisher.reserve_publication("model", "transaction-1")
    ref = publisher.publish("model", {"weights": [1, 2, 3]}, serializer="json")
    publisher.record_provisional_publication("model", "transaction-1", ref)
    Path(ref.uri).write_text("tampered-before-seal", encoding="utf-8")

    with pytest.raises(ArtifactPublicationError, match="before authority seal"):
        publisher.commit_publications("transaction-1", {"model": ref})


def test_artifact_scavenger_tombstones_stale_provisional_transaction(tmp_path) -> None:
    authority = ArtifactAuthority(root=str(tmp_path / "artifacts"), namespace="test")
    publisher = ArtifactPublisher(
        authority,
        project_run_id="run-1",
        case_run_id="case-1",
    )
    publisher.reserve_publication("model", "transaction-1")
    ref = publisher.publish("model", {"weights": [1, 2, 3]}, serializer="json")
    publisher.record_provisional_publication("model", "transaction-1", ref)

    recovered = publisher.publication_ledger.scavenge_stale_reservations(
        stale_after_seconds=1,
        now=10**12,
    )

    assert recovered == (
        {
            "transaction_id": "transaction-1",
            "status": "scavenged",
            "artifact_names": ["model"],
        },
    )
    assert not Path(ref.uri).exists()


def test_durable_commit_is_not_revoked_by_name_index_compaction_failure(
    tmp_path,
    monkeypatch,
) -> None:
    authority = ArtifactAuthority(root=str(tmp_path / "artifacts"), namespace="test")
    publisher = ArtifactPublisher(
        authority,
        project_run_id="run-1",
        case_run_id="case-1",
    )
    publisher.reserve_publication("model", "transaction-1")
    ref = publisher.publish("model", {"weights": [1, 2, 3]}, serializer="json")
    publisher.record_provisional_publication("model", "transaction-1", ref)
    original_replace = publisher.publication_ledger._write_json_replace

    def fail_committed_index(path, payload):
        if payload.get("status") == "committed":
            raise OSError("name index unavailable")
        return original_replace(path, payload)

    monkeypatch.setattr(
        publisher.publication_ledger,
        "_write_json_replace",
        fail_committed_index,
    )

    receipts = publisher.commit_publications(
        "transaction-1",
        {"model": ref},
    )

    assert publisher.verify_publication(receipts["model"])
    assert publisher.commit_publications(
        "transaction-1",
        {"model": ref},
    ) == receipts


def test_case_request_rejects_bare_data_ref_as_authoritative_input() -> None:
    from blackbase.resources import DataRef

    with pytest.raises(ValueError, match="ArtifactBinding"):
        CaseRunRequest(
            project_name="project",
            stage_name="consume",
            case_name="consumer",
            input_artifacts={"model": DataRef(uri="memory://untrusted")},
        )


def test_failed_case_artifact_is_diagnostic_not_registry_input(tmp_path) -> None:
    project_root = create_project(tmp_path / "failed_artifact", framework="blackbase")
    case_root = add_case("publisher", "solver", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        """
class Publisher:
    def run(self):
        self.case_runtime.publish_artifact(
            "model",
            {"weights": [1, 2, 3]},
            serializer="json",
            kind="model",
        )
        raise RuntimeError("final result publication failed")


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Publisher()
""".lstrip(),
        encoding="utf-8",
    )
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "failed_artifact"
STAGES = [{"name": "publish", "cases": ["publisher"]}]
GROUPS = {"default": {"stages": ["publish"]}}
""".lstrip(),
        encoding="utf-8",
    )

    result = execute_project(project_root, record=False)

    assert result.ok is False
    assert result.artifact_registry == {}
    case_result = result.case_results[0]
    assert case_result.artifact_refs == {}
    assert set(case_result.diagnostic_artifact_refs) == {"model"}
    assert case_result.diagnostic_artifact_publications == {}
    assert not Path(case_result.diagnostic_artifact_refs["model"].uri).exists()
