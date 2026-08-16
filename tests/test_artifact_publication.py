from __future__ import annotations

from pathlib import Path

import pytest

from blackbase.project import execute_project
from blackbase.project.runtime import ProjectL0Runtime, ProjectRuntimeConfig
from blackbase.project.scaffold import add_case, create_project
from blackbase.resources import (
    ArtifactFenceError,
    ArtifactPublisher,
    ResourceOffer,
    ResourcePolicy,
    ResourceRequest,
)


def test_case_runtime_publishes_real_artifact_before_returning_data_ref(tmp_path) -> None:
    project_root = create_project(tmp_path / "artifact_project", framework="blackbase")
    case_root = add_case("producer", "solver", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        """
class Producer:
    def run(self):
        ref = self.case_runtime.publish_artifact(
            "model",
            {"weights": [1.0, 2.0], "bias": 0.5},
            kind="model",
            metadata={"provider": "test"},
        )
        return {"published_ref": ref.as_dict()}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Producer()
""".lstrip(),
        encoding="utf-8",
    )
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "artifact_project"
L0 = {
    "namespace": "artifact_project",
    "offer": {"threads": 1, "gpus": 0, "backend": "local"},
    "policy": {"mode": "strict", "max_workers": 1, "max_threads": 1},
    "default_request": {"workers": 1, "threads": 1},
    "lease_backend": "sqlite",
    "lease_path": ".blackbase/l0.sqlite",
    "artifacts": {"path": ".blackbase/artifacts"},
}
STAGES = [{"name": "publish", "cases": ["producer"]}]
GROUPS = {"default": {"stages": ["publish"]}}
""".lstrip(),
        encoding="utf-8",
    )

    project_result = execute_project(project_root, record=False)

    assert project_result.ok
    result = project_result.case_results[0]
    ref = result.artifact_refs["model"]
    path = Path(ref.uri)
    assert path.is_file()
    assert path.is_relative_to(project_root / ".blackbase" / "artifacts")
    assert ref.backend == "filesystem"
    assert ref.kind == "model"
    assert ref.checksum.startswith("sha256:")
    assert ref.size_bytes == path.stat().st_size
    assert ref.metadata["provider"] == "test"
    assert result.output["published_ref"] == ref.as_dict()
    assert project_result.artifact_registry["producer.model"] == ref


def test_artifact_publisher_rejects_stale_project_lease_fence(tmp_path) -> None:
    runtime = ProjectL0Runtime(
        ProjectRuntimeConfig(
            offer=ResourceOffer(threads=1),
            policy=ResourcePolicy(max_workers=1, max_threads=1),
            default_request=ResourceRequest(workers=1, threads=1),
            lease_backend="sqlite",
            lease_path=".blackbase/l0.sqlite",
        ),
        project_root=tmp_path,
    )
    lease = runtime.acquire_case("producer", request=ResourceRequest())
    context = runtime.resource_context(lease, case_name="producer").as_dict()
    publisher = ArtifactPublisher.from_resource_context(
        context,
        project_run_id="project-run",
        case_run_id="case-run",
    )
    assert publisher is not None
    runtime.release(lease)

    with pytest.raises(ArtifactFenceError, match="stale Project lease fence"):
        publisher.publish("model", {"value": 1}, kind="model")


def test_opaque_artifact_is_not_given_a_fake_ref_without_unsafe_authority(tmp_path) -> None:
    project_root = create_project(tmp_path / "safe_artifact_project", framework="blackbase")
    case_root = add_case("producer", "solver", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        """
class Producer:
    def run(self):
        self.case_runtime.publish_artifact("model", object(), kind="model")
        return {"unreachable": True}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Producer()
""".lstrip(),
        encoding="utf-8",
    )
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "safe_artifact_project"
L0 = {
    "offer": {"threads": 1, "gpus": 0, "backend": "local"},
    "policy": {"mode": "strict", "max_workers": 1, "max_threads": 1},
    "default_request": {"workers": 1, "threads": 1},
}
STAGES = [{"name": "publish", "cases": ["producer"]}]
GROUPS = {"default": {"stages": ["publish"]}}
""".lstrip(),
        encoding="utf-8",
    )

    project_result = execute_project(project_root, record=False)

    assert not project_result.ok
    result = project_result.case_results[0]
    assert result.failure is not None
    assert result.failure.kind == "ArtifactPublicationError"
    assert not result.artifact_refs
    assert not project_result.artifact_registry
