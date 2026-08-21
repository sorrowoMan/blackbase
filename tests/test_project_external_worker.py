from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import threading
import time

from blackbase.project.external_worker import ExternalCaseWorker
from blackbase.project import CaseRunRequest, CaseRunResult
from blackbase.project.doctor import run_common_project_doctor
from blackbase.project.project_runner import execute_project
from blackbase.project.run_manifest import ProjectRunRecorder, project_config_fingerprint
from blackbase.project.scaffold import add_case, create_project
from blackbase.resources import (
    ResourceOffer,
    ResourceAllocator,
    ResourcePolicy,
    ResourceRequest,
    ResourceRequirement,
    SQLiteLeaseStore,
    SQLiteTaskTransport,
    TaskEnvelope,
    TaskResult,
    WorkerDescriptor,
)


def _run_one_external_worker(project_root: str, transport_path: str) -> None:
    worker = WorkerDescriptor(
        worker_id="integration-worker",
        executor_backend="external",
        resource_backend="local",
        capabilities=("project_case",),
        offer=ResourceOffer(threads=1, gpus=0, backend="local"),
        max_inflight=1,
    )
    runtime = ExternalCaseWorker(
        SQLiteTaskTransport(transport_path),
        worker,
        allowed_project_root=project_root,
        lease_seconds=1.0,
        heartbeat_interval_seconds=0.1,
    )
    runtime.run_forever(poll_interval_seconds=0.02, max_tasks=1)


def _build_external_project(project_root: Path, *, queue_timeout_seconds: float) -> Path:
    case_root = add_case("external_case", "solver", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        """
import os

class Case:
    def __init__(self, resource_context=None):
        self.resource_context = dict(resource_context or {})

    def run(self):
        report_ref = self.case_runtime.publish_artifact(
            "report",
            {"pid": os.getpid()},
            kind="report",
        )
        return {
            "pid": os.getpid(),
            "lease_id": self.resource_context["lease"]["lease_id"],
            "artifact_refs": {"report": report_ref},
        }

def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, component_overrides
    return Case(resource_context)
""".lstrip(),
        encoding="utf-8",
    )
    transport_path = project_root / ".blackbase" / "external_tasks.sqlite"
    (project_root / "project_config.py").write_text(
        f"""
PROJECT_NAME = "external_project"
L0 = {{
    "namespace": "external_project",
    "offer": {{"threads": 1, "gpus": 0, "backend": "local"}},
    "policy": {{"mode": "strict", "max_workers": 1, "max_threads": 1}},
    "default_request": {{"workers": 1, "threads": 1, "gpus": 0, "backend": "local"}},
    "lease_backend": "sqlite",
    "lease_path": ".blackbase/l0_leases.sqlite",
    "lease_ttl_seconds": 1.0,
    "lease_heartbeat_seconds": 0.1,
}}
STAGES = [{{
    "name": "external",
    "policy": "external",
    "cases": ["external_case"],
    "external": {{
        "backend": "sqlite",
        "transport_path": {str(transport_path)!r},
        "queue_timeout_seconds": {float(queue_timeout_seconds)!r},
        "poll_interval_seconds": 0.01,
    }},
}}]
GROUPS = {{"default": {{"stages": ["external"]}}}}
""".lstrip(),
        encoding="utf-8",
    )
    return transport_path


def test_project_external_stage_runs_case_in_worker_process(tmp_path) -> None:
    project_root = create_project(tmp_path / "external_project", framework="blackbase")
    transport_path = _build_external_project(project_root, queue_timeout_seconds=5.0)
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_run_one_external_worker,
        args=(str(project_root), str(transport_path)),
    )
    process.start()
    try:
        result = execute_project(project_root, run_id="external-success")
        process.join(timeout=10.0)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)

    assert process.exitcode == 0
    assert result.ok
    assert result.case_results[0].status == "succeeded"
    assert result.case_results[0].output["pid"] != os.getpid()
    assert result.case_results[0].output["lease_id"].startswith("lease-external_case-external-")
    assert result.artifact_registry["external.external_case.report"].kind == "report"
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["cases"][0]["external_task"]["task_id"] == (
        "project:external-success:external:external_case"
    )
    assert manifest["cases"][0]["external_task"]["broker_status"] == "succeeded"
    transport = SQLiteTaskTransport(transport_path)
    assert transport.counts() == {"succeeded": 1}
    task_record = transport.get("project:external-success:external:external_case")
    assert task_record is not None and task_record.result is not None
    assert task_record.result.metadata["lease_fence_validated"] is True
    assert task_record.result.metadata["fencing_token"] > 0
    assert transport.list_workers()[0].worker_id == "integration-worker"


def test_external_worker_hard_terminates_blocking_case_after_deadline(tmp_path) -> None:
    project_root = create_project(tmp_path / "external_hard_stop", framework="blackbase")
    case_root = add_case("external_case", "solver", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        """
import time


class Case:
    def run(self):
        time.sleep(10.0)
        return {"unreachable": True}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Case()
""".lstrip(),
        encoding="utf-8",
    )
    transport_path = project_root / ".blackbase" / "external_tasks.sqlite"
    (project_root / "project_config.py").write_text(
        f"""
PROJECT_NAME = "external_hard_stop"
L0 = {{
    "namespace": "external_hard_stop",
    "offer": {{"threads": 1, "gpus": 0, "backend": "local"}},
    "policy": {{"mode": "strict", "max_workers": 1, "max_threads": 1}},
    "default_request": {{"workers": 1, "threads": 1}},
    "lease_backend": "sqlite",
    "lease_path": ".blackbase/l0_leases.sqlite",
    "lease_ttl_seconds": 2.0,
    "lease_heartbeat_seconds": 0.2,
}}
STAGES = [{{
    "name": "external",
    "policy": "external",
    "cases": ["external_case"],
    "timeout_seconds": 1.5,
    "termination": {{
        "mode": "cooperative_then_terminate",
        "grace_seconds": 0.05,
        "kill_grace_seconds": 0.2,
        "poll_interval_seconds": 0.01,
    }},
    "external": {{
        "backend": "sqlite",
        "transport_path": {str(transport_path)!r},
        "queue_timeout_seconds": 5.0,
        "poll_interval_seconds": 0.01,
    }},
}}]
GROUPS = {{"default": {{"stages": ["external"]}}}}
""".lstrip(),
        encoding="utf-8",
    )
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_run_one_external_worker,
        args=(str(project_root), str(transport_path)),
    )
    process.start()
    started = time.monotonic()
    try:
        result = execute_project(project_root, run_id="external-hard-stop", record=False)
        process.join(timeout=10.0)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)

    assert time.monotonic() - started < 7.0
    assert process.exitcode == 0
    assert not result.ok
    case_result = result.case_results[0]
    assert case_result.status == "timed_out"
    assert case_result.failure is not None
    assert case_result.failure.phase == "terminate"
    assert case_result.failure.details["terminated"] is True


def test_external_worker_retry_uses_attempt_specific_case_identity(tmp_path) -> None:
    project_root = create_project(tmp_path / "retry_identity", framework="blackbase")
    case_root = add_case("external_case", "solver", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        """
class Case:
    def run(self):
        identity = self.case_runtime.identity
        if identity.attempt == 1:
            raise RuntimeError("fail first worker attempt")
        return {"identity": identity.as_dict()}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Case()
""".lstrip(),
        encoding="utf-8",
    )
    transport_path = project_root / ".blackbase" / "retry_tasks.sqlite"
    transport = SQLiteTaskTransport(transport_path)
    base_request = CaseRunRequest(
        project_name="retry_identity",
        stage_name="external",
        case_name="external_case",
        resource_request=ResourceRequest(workers=1, threads=1).as_dict(),
        resource_context={
            "threads": 1,
            "grant": {
                "workers": 1,
                "threads": 1,
                "gpus": 0,
                "memory_mb": 512,
                "backend": "local",
                "compute_backend": "auto",
                "device": "cpu",
                "capabilities": [],
            },
        },
    )
    transport.submit(
        TaskEnvelope(
            task_id="retry-case",
            task_type="project_case",
            payload={
                "project_root": str(project_root),
                "request": base_request.as_dict(),
                "extra_python_paths": [],
            },
            requirement=ResourceRequirement(
                threads=1,
                resource_backend="local",
                capabilities=("project_case",),
            ),
            executor_backend="external",
            namespace="retry_identity.external.external_case",
            max_retries=1,
        )
    )
    worker = ExternalCaseWorker(
        transport,
        WorkerDescriptor(
            worker_id="retry-worker",
            executor_backend="external",
            resource_backend="local",
            capabilities=("project_case",),
            offer=ResourceOffer(threads=1, gpus=0, backend="local"),
        ),
        allowed_project_root=project_root,
        lease_seconds=1.0,
        heartbeat_interval_seconds=0.1,
    )

    assert worker.run_once()
    after_first = transport.get("retry-case")
    assert after_first is not None
    assert after_first.status == "queued"
    assert after_first.attempt == 1
    assert worker.run_once()

    completed = transport.get("retry-case")
    assert completed is not None and completed.result is not None
    assert completed.status == "succeeded"
    assert completed.attempt == 2
    result = CaseRunResult.from_dict(completed.result.output)
    assert result.identity.attempt == 2
    assert result.identity.invocation_id == base_request.identity.invocation_id
    assert result.identity.case_run_id.endswith(":attempt:2")
    assert result.identity.case_run_id != base_request.identity.case_run_id
    assert result.request.resource_context["metadata"]["run_identity"]["attempt"] == 2
    assert completed.result.resource_context["metadata"]["run_identity"]["attempt"] == 2


def test_project_external_stage_fails_when_no_worker_claims_task(tmp_path) -> None:
    project_root = create_project(tmp_path / "no_worker_project", framework="blackbase")
    transport_path = _build_external_project(project_root, queue_timeout_seconds=0.15)

    result = execute_project(project_root, run_id="external-no-worker")

    assert not result.ok
    assert result.exit_code == 1
    assert result.case_results[0].status == "failed"
    assert "No compatible external worker" in result.case_results[0].error
    assert SQLiteTaskTransport(transport_path).counts() == {"cancelled": 1}


def test_project_resume_reconciles_submit_before_manifest_crash_window(tmp_path) -> None:
    project_root = create_project(tmp_path / "recover_external", framework="blackbase")
    transport_path = _build_external_project(project_root, queue_timeout_seconds=0.15)
    fingerprint = project_config_fingerprint(
        project_root,
        group="default",
        framework="blackbase",
    )
    recorder = ProjectRunRecorder(
        project_root=project_root,
        project_name="external_project",
        group="default",
        framework="blackbase",
        config_fingerprint=fingerprint,
        case_order=(("external", "external_case"),),
        run_id="crashed-attempt",
    )
    task_id = "project:crashed-attempt:external:external_case"

    transport = SQLiteTaskTransport(transport_path)
    task = TaskEnvelope(
        task_id=task_id,
        task_type="project_case",
        payload={
            "project_root": str(project_root),
            "request": CaseRunRequest(
                project_name="external_project",
                stage_name="external",
                case_name="external_case",
                resource_request=ResourceRequest(workers=1, threads=1).as_dict(),
            ).as_dict(),
            "extra_python_paths": [],
        },
        requirement=ResourceRequirement(
            threads=1,
            resource_backend="local",
            capabilities=("project_case",),
        ),
        executor_backend="external",
        namespace="external_project.external.external_case",
    )
    transport.submit(task)
    worker = WorkerDescriptor(
        worker_id="finished-before-crash",
        executor_backend="external",
        resource_backend="local",
        capabilities=("project_case",),
        offer=ResourceOffer(threads=1, gpus=0, backend="local"),
    )
    claim = transport.claim(worker, lease_seconds=1.0)
    assert claim is not None
    lease_allocator = ResourceAllocator(
        offer=ResourceOffer(threads=1, gpus=0, backend="local"),
        policy=ResourcePolicy(max_workers=1, max_threads=1, max_gpus=0),
        lease_store=SQLiteLeaseStore(
            project_root / ".blackbase" / "l0_leases.sqlite",
            namespace="external_project",
        ),
        lease_ttl_seconds=10.0,
    )
    old_lease = lease_allocator.acquire(
        ResourceRequest(workers=1, threads=1, memory_mb=1),
        owner_id="external_case",
        scope="external",
    )
    old_resource_context = old_lease.resource_context(
        namespace="external_project.external.external_case",
        metadata={
            "lease_authority": {
                "backend": "sqlite",
                "path": str(project_root / ".blackbase" / "l0_leases.sqlite"),
                "namespace": "external_project",
                "ttl_seconds": 10.0,
                "heartbeat_seconds": 1.0,
            }
        },
    )
    transport.complete(
        claim,
        TaskResult(
            task_id=task_id,
            status="ok",
            worker_id=worker.worker_id,
            output=CaseRunResult(
                request=CaseRunRequest(
                    project_name="external_project",
                    stage_name="external",
                    case_name="external_case",
                    resource_request=ResourceRequest(workers=1, threads=1).as_dict(),
                    resource_context=old_resource_context,
                ),
                status="succeeded",
                output={
                    "recovered": True,
                    "artifact_refs": {
                        "report": {
                            "uri": "memory://recovered/report",
                            "kind": "report",
                        },
                    },
                },
                artifact_refs={
                    "report": {
                        "uri": "memory://recovered/report",
                        "kind": "report",
                    }
                },
                metadata={"runtime_state": {"source": "broker-reconciliation"}},
            ).as_dict(),
            resource_context=old_resource_context,
            metadata={
                "lease_fence_validated": True,
                "fencing_token": old_lease.fencing_token,
            },
        ),
    )
    lease_allocator.release(old_lease)

    result = execute_project(
        project_root,
        run_id="recovered-attempt",
        resume_from=recorder.path,
    )

    assert result.ok
    assert result.case_results[0].status == "resumed"
    assert result.case_results[0].output["recovered"] is True
    assert (
        result.case_results[0].request.as_dict()["resource_context"]
        == old_resource_context
    )
    assert result.artifact_registry["external.external_case.report"].uri == "memory://recovered/report"
    assert transport.counts() == {"succeeded": 1}
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["cases"][0]["external_task"]["task_id"] == task_id
    assert manifest["cases"][0]["external_task"]["reconciled"] is True


def test_project_resume_adopts_live_external_task_without_double_allocation(tmp_path) -> None:
    project_root = create_project(tmp_path / "adopt_live_external", framework="blackbase")
    transport_path = _build_external_project(project_root, queue_timeout_seconds=3.0)
    case_root = project_root / "cases" / "external_case"
    (case_root / "build_solver.py").write_text(
        """
import time

class Case:
    def __init__(self, resource_context=None):
        self.resource_context = dict(resource_context or {})

    def run(self):
        time.sleep(0.3)
        return {"lease_id": self.resource_context["lease"]["lease_id"]}

def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, component_overrides
    return Case(resource_context)
""".lstrip(),
        encoding="utf-8",
    )
    fingerprint = project_config_fingerprint(
        project_root,
        group="default",
        framework="blackbase",
    )
    recorder = ProjectRunRecorder(
        project_root=project_root,
        project_name="external_project",
        group="default",
        framework="blackbase",
        config_fingerprint=fingerprint,
        case_order=(("external", "external_case"),),
        run_id="live-crashed-attempt",
    )
    lease_store = SQLiteLeaseStore(
        project_root / ".blackbase" / "l0_leases.sqlite",
        namespace="external_project",
    )
    allocator = ResourceAllocator(
        offer=ResourceOffer(threads=1, gpus=0, backend="local"),
        policy=ResourcePolicy(max_workers=1, max_threads=1, max_gpus=0),
        lease_store=lease_store,
        lease_ttl_seconds=1.0,
    )
    old_lease = allocator.acquire(
        ResourceRequest(workers=1, threads=1),
        owner_id="external_case",
        scope="external",
    )
    resource_context = old_lease.resource_context(
        namespace="external_project.external.external_case",
        metadata={
            "lease_authority": {
                "backend": "sqlite",
                "path": str(project_root / ".blackbase" / "l0_leases.sqlite"),
                "namespace": "external_project",
                "ttl_seconds": 1.0,
                "heartbeat_seconds": 0.05,
            }
        },
    )
    task_id = "project:live-crashed-attempt:external:external_case"
    transport = SQLiteTaskTransport(transport_path)
    transport.submit(
        TaskEnvelope(
            task_id=task_id,
            task_type="project_case",
            payload={
                "project_root": str(project_root),
                "request": CaseRunRequest(
                    project_name="external_project",
                    stage_name="external",
                    case_name="external_case",
                    resource_request=ResourceRequest(workers=1, threads=1).as_dict(),
                    resource_context=resource_context,
                ).as_dict(),
                "extra_python_paths": [],
            },
            requirement=ResourceRequirement(
                threads=1,
                resource_backend="local",
                capabilities=("project_case",),
            ),
            executor_backend="external",
            namespace="external_project.external.external_case",
        )
    )
    worker = ExternalCaseWorker(
        transport,
        WorkerDescriptor(
            worker_id="live-recovery-worker",
            executor_backend="external",
            resource_backend="local",
            capabilities=("project_case",),
            offer=ResourceOffer(threads=1, gpus=0, backend="local"),
        ),
        allowed_project_root=project_root,
        lease_seconds=1.0,
        heartbeat_interval_seconds=0.05,
    )
    worker_thread = threading.Thread(target=worker.run_once, daemon=True)
    worker_thread.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        record = transport.get(task_id)
        if record is not None and record.status == "leased":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("external worker did not lease the recovery task")

    result = execute_project(
        project_root,
        run_id="live-recovered-attempt",
        resume_from=recorder.path,
    )
    worker_thread.join(timeout=3.0)

    assert not worker_thread.is_alive()
    assert result.ok
    assert result.case_results[0].status == "resumed"
    assert result.case_results[0].output["lease_id"] == old_lease.lease_id
    assert transport.counts() == {"succeeded": 1}
    assert lease_store.list() == ()


def test_project_resume_replaces_task_whose_project_fence_is_stale(tmp_path) -> None:
    project_root = create_project(tmp_path / "replace_stale_external", framework="blackbase")
    transport_path = _build_external_project(project_root, queue_timeout_seconds=3.0)
    fingerprint = project_config_fingerprint(
        project_root,
        group="default",
        framework="blackbase",
    )
    recorder = ProjectRunRecorder(
        project_root=project_root,
        project_name="external_project",
        group="default",
        framework="blackbase",
        config_fingerprint=fingerprint,
        case_order=(("external", "external_case"),),
        run_id="stale-crashed-attempt",
    )
    lease_store = SQLiteLeaseStore(
        project_root / ".blackbase" / "l0_leases.sqlite",
        namespace="external_project",
    )
    allocator = ResourceAllocator(
        offer=ResourceOffer(threads=1, gpus=0, backend="local"),
        policy=ResourcePolicy(max_workers=1, max_threads=1, max_gpus=0),
        lease_store=lease_store,
        lease_ttl_seconds=10.0,
    )
    stale_lease = allocator.acquire(
        ResourceRequest(workers=1, threads=1),
        owner_id="external_case",
        scope="external",
    )
    stale_context = stale_lease.resource_context(
        namespace="external_project.external.external_case",
        metadata={
            "lease_authority": {
                "backend": "sqlite",
                "path": str(project_root / ".blackbase" / "l0_leases.sqlite"),
                "namespace": "external_project",
                "ttl_seconds": 10.0,
                "heartbeat_seconds": 1.0,
            }
        },
    )
    allocator.release(stale_lease)

    stale_task_id = "project:stale-crashed-attempt:external:external_case"
    transport = SQLiteTaskTransport(transport_path)
    transport.submit(
        TaskEnvelope(
            task_id=stale_task_id,
            task_type="project_case",
            payload={
                "project_root": str(project_root),
                "request": CaseRunRequest(
                    project_name="external_project",
                    stage_name="external",
                    case_name="external_case",
                    resource_request=ResourceRequest(workers=1, threads=1).as_dict(),
                    resource_context=stale_context,
                ).as_dict(),
                "extra_python_paths": [],
            },
            requirement=ResourceRequirement(
                threads=1,
                resource_backend="local",
                capabilities=("project_case",),
            ),
            executor_backend="external",
            namespace="external_project.external.external_case",
            max_retries=0,
        )
    )
    stale_claim = transport.claim(
        WorkerDescriptor(
            worker_id="dead-worker",
            executor_backend="external",
            resource_backend="local",
            capabilities=("project_case",),
            offer=ResourceOffer(threads=1, gpus=0, backend="local"),
        ),
        lease_seconds=0.15,
    )
    assert stale_claim is not None

    replacement_worker = ExternalCaseWorker(
        transport,
        WorkerDescriptor(
            worker_id="replacement-worker",
            executor_backend="external",
            resource_backend="local",
            capabilities=("project_case",),
            offer=ResourceOffer(threads=1, gpus=0, backend="local"),
        ),
        allowed_project_root=project_root,
        lease_seconds=1.0,
        heartbeat_interval_seconds=0.05,
    )

    def run_replacement_worker() -> None:
        time.sleep(0.3)
        replacement_worker.run_once()
        replacement_worker.run_once()

    worker_thread = threading.Thread(target=run_replacement_worker, daemon=True)
    worker_thread.start()
    result = execute_project(
        project_root,
        run_id="stale-recovered-attempt",
        resume_from=recorder.path,
    )
    worker_thread.join(timeout=3.0)

    assert not worker_thread.is_alive()
    assert result.ok
    assert result.case_results[0].status == "succeeded"
    assert result.case_results[0].output["lease_id"] != stale_lease.lease_id
    assert transport.get(stale_task_id).status == "failed"
    assert transport.get(
        "project:stale-recovered-attempt:external:external_case"
    ).status == "succeeded"


def test_project_doctor_rejects_invalid_external_stage_config(tmp_path) -> None:
    project_root = create_project(tmp_path / "invalid_external", framework="blackbase")
    add_case("external_case", "solver", project_root=project_root)
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "invalid_external"
STAGES = [{
    "name": "external",
    "policy": "external",
    "mode": "cli",
    "cases": ["external_case"],
    "external": {"backend": "http"},
}]
GROUPS = {"default": {"stages": ["external"]}}
""".lstrip(),
        encoding="utf-8",
    )

    report = run_common_project_doctor(project_root, strict=True)
    codes = {item.code for item in report.diagnostics if item.level == "error"}

    assert "project-external-backend-unsupported" in codes
    assert "project-external-transport-path-missing" in codes
    assert "project-external-cli-mode" in codes


def test_project_external_check_reports_transport_without_submitting(tmp_path, capsys) -> None:
    project_root = create_project(tmp_path / "external_check", framework="blackbase")
    transport_path = _build_external_project(project_root, queue_timeout_seconds=1.0)

    result = execute_project(project_root, check=True, build_check=True, record=False)
    output = capsys.readouterr().out

    assert result.ok
    assert result.status == "checked"
    assert '"stage_execution_backend": "external"' in output
    assert '"external_transport_backend": "sqlite"' in output
    assert str(transport_path).replace("\\", "\\\\") in output
    assert not transport_path.exists()


def test_project_external_check_accepts_redis_transport_contract(tmp_path, capsys) -> None:
    project_root = create_project(tmp_path / "redis_external_check", framework="blackbase")
    case_root = add_case("external_case", "solver", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        """
class Case:
    def runtime_report(self):
        return {"adapter": "none", "providers": [], "plugins": []}

def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return Case()
""".lstrip(),
        encoding="utf-8",
    )
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "redis_external"
L0 = {
    "namespace": "redis_external",
    "lease_backend": "redis",
    "lease_redis_url_env": "BLACKBASE_TEST_REDIS_URL",
    "lease_ttl_seconds": 30,
    "lease_heartbeat_seconds": 10,
}
STAGES = [{
    "name": "external",
    "policy": "external",
    "cases": ["external_case"],
    "external": {
        "backend": "redis",
        "redis_url": "redis://127.0.0.1:6379/0",
        "namespace": "redis_external:tasks",
    },
}]
GROUPS = {"default": {"stages": ["external"]}}
""".lstrip(),
        encoding="utf-8",
    )

    report = run_common_project_doctor(project_root, strict=True)
    assert not [item for item in report.diagnostics if item.level == "error"]

    result = execute_project(project_root, check=True, build_check=True, record=False)
    output = capsys.readouterr().out
    assert result.ok
    assert '"external_transport_backend": "redis"' in output
    assert '"external_transport_namespace": "redis_external:tasks"' in output
    assert '"redis_url_env": "BLACKBASE_TEST_REDIS_URL"' in output
    assert '"check_only": true' in output
    assert "redis://127.0.0.1" not in output
