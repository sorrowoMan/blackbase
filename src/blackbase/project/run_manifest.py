"""Durable Project run records and resume state."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from blackbase.resources import CancellationRef, DataRef

from .execution import CaseRunRequest, CaseRunResult, ProjectConfigurationError


MANIFEST_SCHEMA_VERSION = 2


def _migrate_historical_case_result_v1_to_v2(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Compact one completed v1 result for manifest-only resume.

    A live v1 request cannot be upgraded safely because its transported
    ancestor token list has no registered v2 parent authority. Completed
    manifest results no longer execute, so their lineage can be reduced to
    bounded evidence while preserving the effective historical deadline.
    """

    migrated = dict(payload or {})
    if int(migrated.get("schema_version", 0) or 0) != 1:
        return migrated
    request = dict(migrated.get("request", {}) or {})
    if int(request.get("schema_version", 0) or 0) != 1:
        raise ValueError("CaseRunResult v1 contains a non-v1 request envelope")
    control = dict(request.get("control", {}) or {})
    raw_current = dict(control.get("cancellation", {}) or {})
    raw_ancestors = [
        dict(item or {})
        for item in tuple(control.get("ancestor_cancellations", ()) or ())
        if isinstance(item, Mapping)
    ]
    lineage_payload = [*raw_ancestors, raw_current]
    lineage_digest = hashlib.sha256(
        json.dumps(
            lineage_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    deadlines = [
        float(item.get("deadline_at", 0.0) or 0.0)
        for item in lineage_payload
        if float(item.get("deadline_at", 0.0) or 0.0) > 0.0
    ]
    current = CancellationRef.from_dict(raw_current or {}).as_dict()
    current["deadline_at"] = min(deadlines) if deadlines else 0.0
    current["parent_control_id"] = None
    current["root_control_id"] = current["control_id"]
    current["lineage_depth"] = 0
    current["lineage_digest"] = CancellationRef.from_dict(current).lineage_digest
    control_metadata = dict(control.get("metadata", {}) or {})
    control_metadata["historical_v1_lineage"] = {
        "ancestor_count": len(raw_ancestors),
        "digest": lineage_digest,
        "migration": "manifest_resume_only",
    }
    control["cancellation"] = current
    control["ancestor_cancellations"] = []
    control["metadata"] = control_metadata
    request_metadata = dict(request.get("metadata", {}) or {})
    request_metadata["case_run_schema_migration"] = {
        "from": 1,
        "to": 2,
        "mode": "historical_manifest_result",
    }
    request["control"] = control
    request["metadata"] = request_metadata
    request["schema_version"] = 2
    result_metadata = dict(migrated.get("metadata", {}) or {})
    result_metadata["case_run_schema_migration"] = {
        "from": 1,
        "to": 2,
        "mode": "historical_manifest_result",
    }
    migrated["request"] = request
    migrated["metadata"] = result_metadata
    migrated["schema_version"] = 2
    return migrated


def _migrate_historical_case_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade completed manifest evidence without trusting old input refs.

    Live pre-v3 requests remain rejected. A completed result is safe to retain as
    history, but its old bare ``input_artifacts`` cannot be promoted back into an
    authority capability, so those refs move into explicit audit metadata.
    """

    migrated = dict(payload or {})
    original_version = int(migrated.get("schema_version", 0) or 0)
    if original_version == 1:
        migrated = _migrate_historical_case_result_v1_to_v2(migrated)
    if int(migrated.get("schema_version", 0) or 0) != 2:
        return migrated
    request = dict(migrated.get("request", {}) or {})
    if int(request.get("schema_version", 0) or 0) != 2:
        raise ValueError("CaseRunResult v2 contains a non-v2 request envelope")
    request_metadata = dict(request.get("metadata", {}) or {})
    historical_inputs = dict(request.get("input_artifacts", {}) or {})
    if historical_inputs:
        request_metadata["historical_unbound_input_artifacts"] = historical_inputs
    request_metadata["case_run_schema_migration"] = {
        "from": original_version,
        "to": 3,
        "mode": "historical_manifest_result",
    }
    request["input_artifacts"] = {}
    request["input_artifact_bindings"] = {}
    request["metadata"] = request_metadata
    request["schema_version"] = 3
    result_metadata = dict(migrated.get("metadata", {}) or {})
    result_metadata["case_run_schema_migration"] = {
        "from": original_version,
        "to": 3,
        "mode": "historical_manifest_result",
    }
    migrated["request"] = request
    migrated["metadata"] = result_metadata
    migrated["schema_version"] = 3
    return migrated


@dataclass(frozen=True)
class ProjectRunManifest:
    """Serializable recovery record for one Project execution attempt."""

    run_id: str
    project_name: str
    group: str
    framework: str
    config_fingerprint: str
    status: str = "running"
    exit_code: int = 0
    started_at: float = 0.0
    updated_at: float = 0.0
    resumed_from: str = ""
    cases: Sequence[Mapping[str, Any]] = ()
    artifact_registry: Mapping[str, DataRef] = field(default_factory=dict)
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "run_id": str(self.run_id),
            "project_name": str(self.project_name),
            "group": str(self.group),
            "framework": str(self.framework),
            "config_fingerprint": str(self.config_fingerprint),
            "status": str(self.status),
            "exit_code": int(self.exit_code),
            "started_at": float(self.started_at),
            "updated_at": float(self.updated_at),
            "resumed_from": str(self.resumed_from),
            "cases": [dict(item) for item in self.cases],
            "artifact_registry": {
                key: ref.as_dict() for key, ref in self.artifact_registry.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProjectRunManifest":
        schema_version = int(payload.get("schema_version", 0) or 0)
        if schema_version != MANIFEST_SCHEMA_VERSION:
            raise ProjectConfigurationError(
                f"Unsupported Project run manifest schema_version={schema_version}; "
                f"expected {MANIFEST_SCHEMA_VERSION}"
            )
        return cls(
            run_id=str(payload.get("run_id", "")),
            project_name=str(payload.get("project_name", "")),
            group=str(payload.get("group", "default")),
            framework=str(payload.get("framework", "blackbase")),
            config_fingerprint=str(payload.get("config_fingerprint", "")),
            status=str(payload.get("status", "unknown")),
            exit_code=int(payload.get("exit_code", 0) or 0),
            started_at=float(payload.get("started_at", 0.0) or 0.0),
            updated_at=float(payload.get("updated_at", 0.0) or 0.0),
            resumed_from=str(payload.get("resumed_from", "")),
            cases=tuple(dict(item) for item in payload.get("cases", ()) or ()),
            artifact_registry={
                str(key): DataRef.from_dict(value)
                for key, value in dict(payload.get("artifact_registry", {}) or {}).items()
            },
            schema_version=schema_version,
        )

    def successful_cases(self) -> dict[tuple[str, str], CaseRunResult]:
        successful: dict[tuple[str, str], CaseRunResult] = {}
        for record in self.cases:
            payload = record.get("result")
            if not isinstance(payload, Mapping):
                continue
            result = CaseRunResult.from_dict(
                _migrate_historical_case_result(payload)
            )
            if not result.ok:
                continue
            request = result.request
            successful[(request.stage_name, request.case_name)] = replace(
                result,
                status="resumed",
            )
        return successful

    def external_tasks(self) -> dict[tuple[str, str], dict[str, Any]]:
        """Return durable external task links keyed by Stage/Case."""

        tasks: dict[tuple[str, str], dict[str, Any]] = {}
        for record in self.cases:
            external_task = record.get("external_task")
            if not isinstance(external_task, Mapping):
                continue
            task_id = str(external_task.get("task_id", "") or "")
            if not task_id:
                continue
            key = (
                str(record.get("stage_name", "stage")),
                str(record.get("case_name", "case")),
            )
            tasks[key] = dict(external_task)
        return tasks


class ProjectRunRecorder:
    """Atomically persists Case completion and artifact recovery state."""

    def __init__(
        self,
        *,
        project_root: Path,
        project_name: str,
        group: str,
        framework: str,
        config_fingerprint: str,
        case_order: Sequence[tuple[str, str]],
        run_id: str | None = None,
        resumed_from: str = "",
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.run_id = str(run_id or new_run_id())
        if (
            not self.run_id
            or self.run_id in {".", ".."}
            or Path(self.run_id).name != self.run_id
            or "/" in self.run_id
            or "\\" in self.run_id
        ):
            raise ProjectConfigurationError(
                "run_id must be one non-empty filesystem-safe path segment"
            )
        self.path = self.project_root / ".blackbase" / "runs" / self.run_id / "manifest.json"
        if self.path.exists():
            raise ProjectConfigurationError(
                f"Project run manifest already exists for run_id='{self.run_id}'"
            )
        self.project_name = str(project_name)
        self.group = str(group)
        self.framework = str(framework)
        self.config_fingerprint = str(config_fingerprint)
        self.resumed_from = str(resumed_from)
        self.started_at = time.time()
        self._order = {key: index for index, key in enumerate(case_order)}
        self._records: dict[tuple[str, str], dict[str, Any]] = {}
        self._artifacts: dict[str, DataRef] = {}
        self.write(status="running", exit_code=0)

    def record_case(self, result: CaseRunResult) -> None:
        request = result.request
        key = (request.stage_name, request.case_name)
        prior = self._records.get(key, {})
        record = {
            "stage_name": request.stage_name,
            "case_name": request.case_name,
            "case_kind": request.case_kind,
            "mode": request.mode,
            "status": result.status,
            "exit_code": result.exit_code,
            "elapsed_seconds": result.elapsed_seconds,
            "error": result.error,
            "artifact_refs": {
                name: ref.as_dict() for name, ref in result.artifact_refs.items()
            },
            "artifact_publications": {
                name: receipt.as_dict()
                for name, receipt in result.artifact_publications.items()
            },
            "diagnostic_artifact_refs": {
                name: ref.as_dict()
                for name, ref in result.diagnostic_artifact_refs.items()
            },
            "result": result.as_dict(),
        }
        if isinstance(prior.get("external_task"), Mapping):
            record["external_task"] = dict(prior["external_task"])
        self._records[key] = record
        for name, ref in (result.artifact_refs.items() if result.ok else ()):
            if name not in result.artifact_publications:
                continue
            self._artifacts[f"{request.stage_name}.{request.case_name}.{name}"] = ref
            self._artifacts[f"{request.case_name}.{name}"] = ref
            self._artifacts.setdefault(str(name), ref)
        self.write(status="running", exit_code=0)

    def record_external_task(
        self,
        request: CaseRunRequest,
        external_task: Mapping[str, Any],
    ) -> None:
        """Persist the broker task identity before waiting for a worker result."""

        key = (request.stage_name, request.case_name)
        prior = self._records.get(key, {})
        self._records[key] = {
            "stage_name": request.stage_name,
            "case_name": request.case_name,
            "case_kind": request.case_kind,
            "mode": request.mode,
            "status": "submitted",
            "exit_code": 0,
            "elapsed_seconds": float(prior.get("elapsed_seconds", 0.0) or 0.0),
            "error": "",
            "artifact_refs": dict(prior.get("artifact_refs", {}) or {}),
            "external_task": {
                **dict(external_task),
                "task_id": str(external_task.get("task_id", "")),
                "recorded_at": time.time(),
            },
        }
        self.write(status="running", exit_code=0)

    def seed_artifacts(self, refs: Mapping[str, DataRef]) -> None:
        self._artifacts.update(dict(refs))
        self.write(status="running", exit_code=0)

    def finish(self, *, status: str, exit_code: int) -> None:
        self.write(status=status, exit_code=exit_code)

    def write(self, *, status: str, exit_code: int) -> None:
        records = sorted(
            self._records.items(),
            key=lambda item: self._order.get(item[0], len(self._order)),
        )
        manifest = ProjectRunManifest(
            run_id=self.run_id,
            project_name=self.project_name,
            group=self.group,
            framework=self.framework,
            config_fingerprint=self.config_fingerprint,
            status=status,
            exit_code=exit_code,
            started_at=self.started_at,
            updated_at=time.time(),
            resumed_from=self.resumed_from,
            cases=tuple(record for _, record in records),
            artifact_registry=dict(self._artifacts),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


def project_config_fingerprint(
    project_root: Path | str,
    *,
    group: str,
    framework: str,
) -> str:
    root = Path(project_root).resolve()
    digest = hashlib.sha256()
    tracked_paths = [root / "project_config.py"]
    cases_root = root / "cases"
    if cases_root.is_dir():
        tracked_paths.extend(
            path
            for path in cases_root.rglob("*")
            if path.is_file() and (path.suffix == ".py" or path.name == ".case")
        )
    for path in sorted(tracked_paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    digest.update(str(group).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(framework).encode("utf-8"))
    return digest.hexdigest()


def load_resume_manifest(project_root: Path | str, resume_from: Path | str) -> tuple[Path, ProjectRunManifest]:
    root = Path(project_root).resolve()
    supplied = Path(resume_from)
    candidates = [supplied]
    if not supplied.is_absolute():
        candidates.extend((root / supplied, root / ".blackbase" / "runs" / supplied))
    manifest_path: Path | None = None
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            resolved = resolved / "manifest.json"
        if resolved.is_file():
            manifest_path = resolved
            break
    if manifest_path is None:
        raise ProjectConfigurationError(f"Project run manifest not found: {resume_from}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectConfigurationError(f"Cannot read Project run manifest '{manifest_path}': {exc}") from exc
    return manifest_path, ProjectRunManifest.from_dict(payload)


def validate_resume_manifest(
    manifest: ProjectRunManifest,
    *,
    project_name: str,
    group: str,
    framework: str,
    config_fingerprint: str,
) -> None:
    expected = (str(project_name), str(group), str(framework), str(config_fingerprint))
    actual = (
        manifest.project_name,
        manifest.group,
        manifest.framework,
        manifest.config_fingerprint,
    )
    if actual != expected:
        raise ProjectConfigurationError(
            "Resume manifest does not match the current Project/group/framework/config fingerprint"
        )


def new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:10]}"


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "ProjectRunManifest",
    "ProjectRunRecorder",
    "load_resume_manifest",
    "new_run_id",
    "project_config_fingerprint",
    "validate_resume_manifest",
]
