"""Shared Project/Case runner for standard scaffold projects."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from blackbase.resources import (
    CancellationToken,
    DataRef,
    RedisTaskTransport,
    ResourceLease,
    ResourceRequirement,
    SQLiteTaskTransport,
    TaskEnvelope,
    TaskTransport,
    TerminationPolicy,
)

from .case_execution import execute_case_payload
from .execution import (
    CaseFailure,
    CaseRunIdentity,
    CaseRunRequest,
    CaseRunResult,
    ExecutionControl,
    ProjectConfigurationError,
    ProjectRunResult,
)
from .invocation import CaseExecutor
from .run_manifest import (
    ProjectRunRecorder,
    load_resume_manifest,
    project_config_fingerprint,
    validate_resume_manifest,
)
from .runtime import (
    ProjectL0Runtime,
    ResourceLeaseFenceError,
    iter_group_stages,
    load_case_kind,
    load_case_resource_request,
    load_project_runtime_config,
    path_declares_check_argument,
)


def run_project(
    project_root: Path | str,
    *,
    group: str = "default",
    check: bool = False,
    build_check: bool = False,
    case_args: Sequence[str] | None = None,
    framework: str = "blackbase",
    resource_env_var: str | None = None,
    extra_python_paths: Sequence[Path | str] = (),
    record: bool = True,
    run_id: str | None = None,
    resume_from: Path | str | None = None,
) -> int:
    """Compatibility CLI surface returning only the Project exit code."""

    return execute_project(
        project_root,
        group=group,
        check=check,
        build_check=build_check,
        case_args=case_args,
        framework=framework,
        resource_env_var=resource_env_var,
        extra_python_paths=extra_python_paths,
        record=record,
        run_id=run_id,
        resume_from=resume_from,
    ).exit_code


def execute_project(
    project_root: Path | str,
    *,
    group: str = "default",
    check: bool = False,
    build_check: bool = False,
    case_args: Sequence[str] | None = None,
    framework: str = "blackbase",
    resource_env_var: str | None = None,
    extra_python_paths: Sequence[Path | str] = (),
    record: bool = True,
    run_id: str | None = None,
    resume_from: Path | str | None = None,
) -> ProjectRunResult:
    """Execute a Project and retain structured Case results and artifact refs."""

    root = Path(project_root).resolve()
    config_module = _load_project_config(root, framework=framework)
    runtime = ProjectL0Runtime(
        load_project_runtime_config(config_module),
        project_root=root,
        durable=not check,
    )
    project_name = str(getattr(config_module, "PROJECT_NAME", root.name))
    case_args = tuple(case_args or ())
    resource_env_var = resource_env_var or f"{str(framework).upper()}_RESOURCE_CONTEXT_JSON"
    stages = tuple(iter_group_stages(config_module, str(group)))
    stage_names = tuple(str(stage.get("name", "stage")) for stage in stages)
    if len(stage_names) != len(set(stage_names)):
        raise ProjectConfigurationError("Project group stage names must be unique for recovery")
    for stage in stages:
        stage_name = str(stage.get("name", "stage"))
        _require_supported_stage_policy(stage, stage_name=stage_name)
        declared_cases = tuple(str(name) for name in (stage.get("cases", ()) or ()))
        if len(declared_cases) != len(set(declared_cases)):
            raise ProjectConfigurationError(
                f"Stage '{stage_name}' contains duplicate Case names; recovery keys must be unique"
            )
        if _stage_is_external(stage):
            _external_stage_config(stage, project_root=root)
            stage_mode = str(stage.get("mode", "build") or "build")
            case_modes = stage.get("case_modes", {})
            if not isinstance(case_modes, Mapping):
                raise ProjectConfigurationError(
                    f"Stage '{stage_name}' field 'case_modes' must be a mapping"
                )
            invalid_modes = {
                case_name: str(case_modes.get(case_name, stage_mode) or "build")
                for case_name in declared_cases
                if str(case_modes.get(case_name, stage_mode) or "build") != "build"
            }
            if invalid_modes:
                raise ProjectConfigurationError(
                    f"External Stage '{stage_name}' requires mode='build': {invalid_modes}"
                )
    if resume_from is not None and check:
        raise ProjectConfigurationError("--resume-from cannot be combined with --check")
    fingerprint = project_config_fingerprint(root, group=str(group), framework=str(framework))
    resume_successes: dict[tuple[str, str], CaseRunResult] = {}
    resume_external_tasks: dict[tuple[str, str], dict[str, Any]] = {}
    resume_run_id = ""
    resumed_manifest_path = ""
    resumed_artifacts: dict[str, DataRef] = {}
    if resume_from is not None:
        resume_path, resume_manifest = load_resume_manifest(root, resume_from)
        validate_resume_manifest(
            resume_manifest,
            project_name=project_name,
            group=str(group),
            framework=str(framework),
            config_fingerprint=fingerprint,
        )
        resume_successes = resume_manifest.successful_cases()
        resume_external_tasks = resume_manifest.external_tasks()
        resume_run_id = resume_manifest.run_id
        resumed_manifest_path = str(resume_path)
        resumed_artifacts = dict(resume_manifest.artifact_registry)
    exit_code = 0
    case_results: list[CaseRunResult] = []
    artifact_registry: dict[str, DataRef] = dict(resumed_artifacts)
    case_order = tuple(
        (str(stage.get("name", "stage")), str(case_name))
        for stage in stages
        for case_name in tuple(stage.get("cases", ()) or ())
    )
    recorder = None
    if record and not check:
        recorder = ProjectRunRecorder(
            project_root=root,
            project_name=project_name,
            group=str(group),
            framework=str(framework),
            config_fingerprint=fingerprint,
            case_order=case_order,
            run_id=run_id,
            resumed_from=resumed_manifest_path,
        )
        if artifact_registry:
            recorder.seed_artifacts(artifact_registry)
    execution_run_id = (
        recorder.run_id
        if recorder is not None
        else str(run_id or f"ephemeral-{uuid4().hex}")
    )

    def retain(result: CaseRunResult) -> None:
        case_results.append(result)
        if recorder is not None:
            recorder.record_case(result)

    def finish(current_exit_code: int) -> ProjectRunResult:
        result = _project_result(
            project_name,
            group,
            case_results,
            artifact_registry,
            current_exit_code,
            check=check,
            run_id="" if recorder is None else recorder.run_id,
            manifest_path="" if recorder is None else str(recorder.path),
            resumed_from=resumed_manifest_path,
        )
        if recorder is not None:
            recorder.finish(status=result.status, exit_code=result.exit_code)
        return result

    for stage in stages:
        stage_name = str(stage.get("name", "stage"))
        case_names = tuple(str(name) for name in (stage.get("cases", ()) or ()))
        stage_cli_args = dict(stage.get("case_args", {}) or {})
        stage_modes = dict(stage.get("case_modes", {}) or {})
        external_audit: dict[str, Any] = {}
        if _stage_is_external(stage):
            config = _external_stage_config(stage, project_root=root)
            external_audit = _external_transport_audit(config)
        if (_stage_is_parallel(stage) or _stage_is_external(stage)) and not check:
            resumed_by_name = {
                case_name: resume_successes[(stage_name, case_name)]
                for case_name in case_names
                if (stage_name, case_name) in resume_successes
            }
            pending_names = tuple(name for name in case_names if name not in resumed_by_name)
            parallel_results, parallel_artifacts, parallel_exit_code = _execute_parallel_stage(
                project_root=root,
                project_name=project_name,
                stage=stage,
                case_names=pending_names,
                runtime=runtime,
                artifact_registry=artifact_registry,
                case_args=case_args,
                framework=framework,
                extra_python_paths=extra_python_paths,
                on_case_result=None if recorder is None else recorder.record_case,
                on_external_task=(
                    None if recorder is None else recorder.record_external_task
                ),
                execution_backend="external" if _stage_is_external(stage) else "process",
                execution_run_id=execution_run_id,
                resume_run_id=resume_run_id,
                resume_external_tasks=resume_external_tasks,
            )
            pending_by_name = {item.request.case_name: item for item in parallel_results}
            for case_name in case_names:
                if case_name in resumed_by_name:
                    retain(resumed_by_name[case_name])
                else:
                    retain(pending_by_name[case_name])
            for case_name, refs in parallel_artifacts:
                _register_case_artifacts(
                    artifact_registry,
                    stage_name=stage_name,
                    case_name=case_name,
                    artifact_refs=refs,
                )
            exit_code = exit_code or parallel_exit_code
            if parallel_exit_code and _stage_fail_fast(stage):
                return finish(exit_code)
            continue
        for case_name in case_names:
            resumed = resume_successes.get((stage_name, case_name))
            if resumed is not None:
                retain(resumed)
                continue
            started_at = time.time()
            lease = None
            lease_guard = None
            case_kind = load_case_kind(root, case_name, stage=stage, default="solver")
            mode = str(stage_modes.get(case_name, stage.get("mode", "build")) or "build")
            argv = tuple(stage_cli_args.get(case_name, ())) + case_args
            component_overrides = _case_mapping(stage, "component_overrides", case_name)
            input_artifacts: dict[str, DataRef] = {}
            effective_request: CaseRunRequest | None = None
            identity = _case_identity(execution_run_id, stage_name, case_name)
            control = _case_control(runtime, stage, case_name)
            try:
                input_artifacts = _resolve_case_input_artifacts(
                    stage,
                    case_name=case_name,
                    artifact_registry=artifact_registry,
                )
                if mode == "cli" and (component_overrides or input_artifacts):
                    raise ProjectConfigurationError(
                        f"CLI-mode Case '{case_name}' cannot receive in-process component_overrides "
                        "or input_artifacts; use mode='build' or a persisted manifest protocol."
                    )
                request = load_case_resource_request(
                    case_name,
                    project_root=root,
                    stage=stage,
                    default=runtime.config.default_request,
                    extra_import_paths=extra_python_paths,
                )
                lease = runtime.acquire_case(case_name, request=request, stage_name=stage_name)
                lease_guard = runtime.start_lease_guard(lease)
                resource_context = runtime.resource_context(
                    lease,
                    case_name=case_name,
                    stage_name=stage_name,
                )
                resource_context = _resource_context_with_run_contract(
                    resource_context,
                    identity=identity,
                    control=control,
                )
                effective_request = CaseRunRequest(
                    project_name=project_name,
                    stage_name=stage_name,
                    case_name=case_name,
                    case_kind=case_kind,
                    mode=mode,
                    identity=identity,
                    control=control,
                    resource_request=request.as_dict(),
                    resource_context=_as_dict(resource_context),
                    component_overrides=component_overrides,
                    input_artifacts=input_artifacts,
                    argv=argv,
                    metadata={"check_only": bool(check)},
                )
                if mode == "cli":
                    if check and "--check" not in argv:
                        argv = ("--check", *argv)
                    if check and build_check:
                        # A build check validates the canonical builder contract.  Do
                        # that directly instead of cold-starting one Python process per
                        # CLI Case; the CLI entry remains the authority for real runs.
                        build_request = replace(
                            effective_request,
                            mode="build",
                            argv=argv,
                            metadata={
                                **dict(effective_request.metadata),
                                "check_only": True,
                                "configured_mode": "cli",
                                "execution_mode": "build_check",
                            },
                        )
                        result = CaseExecutor(
                            root,
                            extra_python_paths=extra_python_paths,
                        ).execute(build_request)
                        lease_guard.assert_current()
                        runtime_state = dict(result.metadata.get("runtime_state", {}) or {})
                        _print_project_check(
                            project_name=project_name,
                            stage_name=stage_name,
                            case_name=case_name,
                            mode=case_kind,
                            request=request,
                            resource_context=resource_context,
                            state={**runtime_state, **external_audit},
                            label="project-check",
                        )
                        retain(result)
                        if not result.ok:
                            exit_code = exit_code or result.exit_code or 1
                            _print_project_message(
                                {
                                    "project": project_name,
                                    "stage": stage_name,
                                    "case": case_name,
                                    "mode": "build_check_error",
                                    "error": result.error,
                                },
                                label="project-error",
                            )
                            if _stage_fail_fast(stage):
                                return finish(exit_code)
                        continue
                    _print_project_check(
                        project_name=project_name,
                        stage_name=stage_name,
                        case_name=case_name,
                        mode="cli",
                        request=request,
                        resource_context=resource_context,
                        label="project-check" if check else "project-runtime",
                    )
                    rc = 0
                    if not check:
                        rc = _run_case_cli(
                            project_root=root,
                            case_name=case_name,
                            case_kind=case_kind,
                            argv=argv,
                            resource_context=resource_context,
                            resource_env_var=resource_env_var,
                            extra_python_paths=extra_python_paths,
                        )
                        exit_code = exit_code or rc
                    lease_guard.assert_current()
                    finished_at = time.time()
                    result = CaseRunResult(
                        request=replace(effective_request, argv=argv),
                        status="checked" if check else ("succeeded" if rc == 0 else "failed"),
                        output={"returncode": int(rc)},
                        started_at=started_at,
                        finished_at=finished_at,
                        exit_code=rc,
                        failure=(
                            None
                            if rc == 0
                            else CaseFailure(
                                kind="CLIExitError",
                                message=f"CLI Case exited with code {rc}",
                                phase="cli",
                            )
                        ),
                    )
                    retain(result)
                    if rc != 0 and _stage_fail_fast(stage):
                        return finish(exit_code)
                    continue

                if check and not build_check:
                    _print_project_check(
                        project_name=project_name,
                        stage_name=stage_name,
                        case_name=case_name,
                        mode=case_kind,
                        request=request,
                        resource_context=resource_context,
                        state=external_audit,
                        label="project-check",
                    )
                    lease_guard.assert_current()
                    retain(
                        CaseRunResult(
                            request=effective_request,
                            status="checked",
                            started_at=started_at,
                            finished_at=time.time(),
                        )
                    )
                    continue

                result = CaseExecutor(
                    root,
                    extra_python_paths=extra_python_paths,
                ).execute(effective_request)
                lease_guard.assert_current()
                runtime_state = dict(result.metadata.get("runtime_state", {}) or {})
                _print_project_check(
                    project_name=project_name,
                    stage_name=stage_name,
                    case_name=case_name,
                    mode=case_kind,
                    request=request,
                    resource_context=resource_context,
                    state={**runtime_state, **external_audit},
                    label="project-check" if check else "project-runtime",
                )
                _register_case_artifacts(
                    artifact_registry,
                    stage_name=stage_name,
                    case_name=case_name,
                    artifact_refs=result.artifact_refs,
                )
                retain(result)
                if not result.ok:
                    exit_code = exit_code or result.exit_code or 1
                    _print_project_message(
                        {
                            "project": project_name,
                            "stage": stage_name,
                            "case": case_name,
                            "mode": "execution_error",
                            "error": result.error,
                        },
                        label="project-error",
                    )
                    if _stage_fail_fast(stage):
                        return finish(exit_code)
            except Exception as exc:
                exit_code = exit_code or 1
                if effective_request is None:
                    effective_request = CaseRunRequest(
                        project_name=project_name,
                        stage_name=stage_name,
                        case_name=case_name,
                        case_kind=case_kind,
                        mode=mode,
                        identity=identity,
                        control=control,
                        component_overrides=component_overrides,
                        input_artifacts=input_artifacts,
                        argv=argv,
                    )
                error = f"{type(exc).__name__}: {exc}"
                retain(
                    CaseRunResult(
                        request=effective_request,
                        status="failed",
                        started_at=started_at,
                        finished_at=time.time(),
                        exit_code=1,
                        failure=CaseFailure.from_exception(exc, phase="project"),
                    )
                )
                _print_project_message(
                    {
                        "project": project_name,
                        "stage": stage_name,
                        "case": case_name,
                        "mode": "execution_error",
                        "error": error,
                    },
                    label="project-error",
                )
                if _stage_fail_fast(stage):
                    return finish(exit_code)
            finally:
                if lease_guard is not None:
                    lease_guard.close()
                if lease is not None:
                    runtime.release(lease)
    return finish(exit_code)


def _project_result(
    project_name: str,
    group: str,
    case_results: Sequence[CaseRunResult],
    artifact_registry: Mapping[str, DataRef],
    exit_code: int,
    *,
    check: bool,
    run_id: str = "",
    manifest_path: str = "",
    resumed_from: str = "",
) -> ProjectRunResult:
    status = "failed" if int(exit_code) else ("checked" if check else "ok")
    return ProjectRunResult(
        project_name=project_name,
        group=group,
        case_results=tuple(case_results),
        artifact_registry=dict(artifact_registry),
        status=status,
        exit_code=int(exit_code),
        run_id=run_id,
        manifest_path=manifest_path,
        resumed_from=resumed_from,
    )


def _require_supported_stage_policy(stage: Mapping[str, Any], *, stage_name: str) -> None:
    policy = str(stage.get("policy", "serial") or "serial").strip().lower()
    if policy in {
        "serial",
        "sequential",
        "run_all_serial",
        "parallel",
        "run_all_in_parallel",
        "external",
        "external_workers",
    }:
        return
    raise ProjectConfigurationError(
        f"Stage '{stage_name}' declares unsupported execution policy '{policy}'. "
        "Supported policies are serial, parallel, and external."
    )


def _stage_is_parallel(stage: Mapping[str, Any]) -> bool:
    policy = str(stage.get("policy", "serial") or "serial").strip().lower()
    return policy in {"parallel", "run_all_in_parallel"}


def _stage_is_external(stage: Mapping[str, Any]) -> bool:
    policy = str(stage.get("policy", "serial") or "serial").strip().lower()
    return policy in {"external", "external_workers"}


def _execute_parallel_stage(
    *,
    project_root: Path,
    project_name: str,
    stage: Mapping[str, Any],
    case_names: Sequence[str],
    runtime: ProjectL0Runtime,
    artifact_registry: Mapping[str, DataRef],
    case_args: Sequence[str],
    framework: str,
    extra_python_paths: Sequence[Path | str],
    on_case_result: Callable[[CaseRunResult], None] | None = None,
    on_external_task: Callable[[CaseRunRequest, Mapping[str, Any]], None] | None = None,
    execution_backend: str = "process",
    execution_run_id: str = "",
    resume_run_id: str = "",
    resume_external_tasks: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> tuple[list[CaseRunResult], list[tuple[str, dict[str, DataRef]]], int]:
    """Execute independent Cases in isolated worker processes under Project L0."""

    stage_name = str(stage.get("name", "stage"))
    stage_modes = dict(stage.get("case_modes", {}) or {})
    stage_cli_args = dict(stage.get("case_args", {}) or {})
    max_workers = max(
        1,
        min(
            int(stage.get("max_workers", runtime.config.policy.max_workers) or 1),
            int(runtime.config.policy.max_workers),
            len(case_names) or 1,
        ),
    )
    prepared: deque[dict[str, Any]] = deque()
    results_by_index: dict[int, CaseRunResult] = {}
    artifacts_by_index: dict[int, tuple[str, dict[str, DataRef]]] = {}
    exit_code = 0
    external_transport: TaskTransport | None = None
    external_config: dict[str, Any] = {}
    resumable_external = dict(resume_external_tasks or {})
    if execution_backend == "external":
        external_config = _external_stage_config(stage, project_root=project_root)
        external_transport = _build_external_transport(external_config)

    def retain_parallel_result(index: int, result: CaseRunResult) -> None:
        results_by_index[int(index)] = result
        if on_case_result is not None:
            on_case_result(result)

    preparation_stopped = False
    for index, case_name in enumerate(case_names):
        case_kind = "solver"
        mode = str(stage_modes.get(case_name, stage.get("mode", "build")) or "build")
        argv = tuple(stage_cli_args.get(case_name, ())) + tuple(case_args)
        component_overrides = _case_mapping(stage, "component_overrides", case_name)
        input_artifacts: dict[str, DataRef] = {}
        identity = _case_identity(execution_run_id, stage_name, case_name)
        control = _case_control(runtime, stage, case_name)
        if preparation_stopped:
            retain_parallel_result(index, CaseRunResult(
                request=CaseRunRequest(
                    project_name=project_name,
                    stage_name=stage_name,
                    case_name=case_name,
                    case_kind=case_kind,
                    mode=mode,
                    identity=identity,
                    control=control,
                    component_overrides=component_overrides,
                    argv=argv,
                ),
                status="skipped",
                exit_code=1,
                failure=CaseFailure(
                    kind="FailFastSkip",
                    message="Not prepared because parallel stage failure_policy=fail_fast",
                    phase="prepare",
                ),
            ))
            continue
        try:
            case_kind = load_case_kind(project_root, case_name, stage=stage, default="solver")
            if mode != "build":
                raise ProjectConfigurationError(
                    f"Parallel stage '{stage_name}' requires mode='build' for Case '{case_name}'; "
                    "CLI fanout needs an explicit external worker backend."
                )
            input_artifacts = _resolve_case_input_artifacts(
                stage,
                case_name=case_name,
                artifact_registry=artifact_registry,
            )
            request = load_case_resource_request(
                case_name,
                project_root=project_root,
                stage=stage,
                default=runtime.config.default_request,
                extra_import_paths=extra_python_paths,
            )
            prepared.append(
                {
                    "index": index,
                    "case_name": case_name,
                    "case_kind": case_kind,
                    "mode": mode,
                    "argv": argv,
                    "component_overrides": component_overrides,
                    "input_artifacts": input_artifacts,
                    "resource_request": request,
                    "identity": identity,
                    "control": control,
                }
            )
        except Exception as exc:
            exit_code = 1
            request_contract = CaseRunRequest(
                project_name=project_name,
                stage_name=stage_name,
                case_name=case_name,
                case_kind=case_kind,
                mode=mode,
                identity=identity,
                control=control,
                component_overrides=component_overrides,
                input_artifacts=input_artifacts,
                argv=argv,
            )
            retain_parallel_result(index, _failed_case_result(request_contract, exc))
            _print_parallel_error(project_name, stage_name, case_name, exc)
            if _stage_fail_fast(stage):
                preparation_stopped = True

    running: dict[Any, dict[str, Any]] = {}
    stop_launching = bool(exit_code and _stage_fail_fast(stage))
    cancellation_signalled = False
    executor_type = ThreadPoolExecutor if execution_backend == "external" else ProcessPoolExecutor
    with executor_type(max_workers=max_workers) as pool:
        while (prepared and not stop_launching) or running:
            while prepared and not stop_launching and len(running) < max_workers:
                item = prepared[0]
                lease = None
                lease_guard = None
                reconciled_resource_context: dict[str, Any] = {}
                if execution_backend == "external":
                    if external_transport is None:  # pragma: no cover - initialized above
                        raise RuntimeError("external transport was not initialized")
                    prior_info = resumable_external.get(
                        (stage_name, str(item["case_name"])),
                        {},
                    )
                    prior_task_id = str(prior_info.get("task_id", "") or "")
                    if not prior_task_id and resume_run_id:
                        prior_task_id = _external_task_id(
                            resume_run_id,
                            stage_name,
                            str(item["case_name"]),
                        )
                    item["external_prior_task_id"] = prior_task_id
                    if prior_task_id:
                        prior_record = _reconcile_prior_external_task(
                            external_transport,
                            prior_task_id,
                            runtime=runtime,
                            timeout_seconds=float(
                                external_config["queue_timeout_seconds"]
                            ),
                            poll_interval_seconds=float(
                                external_config["poll_interval_seconds"]
                            ),
                        )
                        if prior_record is not None:
                            item["external_task_id"] = prior_task_id
                            item["external_broker_record"] = prior_record
                            item["external_reconciled"] = True
                            prior_request_payload = prior_record.task.payload.get("request")
                            if isinstance(prior_request_payload, Mapping):
                                reconciled_resource_context = dict(
                                    CaseRunRequest.from_dict(
                                        prior_request_payload
                                    ).resource_context
                                )
                            if not reconciled_resource_context and prior_record.result is not None:
                                reconciled_resource_context = dict(
                                    prior_record.result.resource_context or {}
                                )

                if bool(item.get("external_reconciled")) and not reconciled_resource_context:
                    prepared.popleft()
                    exit_code = 1
                    contract = _parallel_case_request(
                        item,
                        project_name=project_name,
                        stage_name=stage_name,
                    )
                    exc = ResourceLeaseFenceError(
                        "Reconciled external task omitted its Project ResourceContext"
                    )
                    retain_parallel_result(
                        int(item["index"]),
                        _failed_case_result(contract, exc),
                    )
                    _print_parallel_error(
                        project_name,
                        stage_name,
                        item["case_name"],
                        exc,
                    )
                    if _stage_fail_fast(stage):
                        stop_launching = True
                    continue

                if reconciled_resource_context:
                    lease_payload = dict(
                        reconciled_resource_context.get("lease", {}) or {}
                    )
                    if lease_payload:
                        candidate_lease = ResourceLease.from_dict(lease_payload)
                        if runtime.allocator.is_current(candidate_lease):
                            lease = candidate_lease
                            lease_guard = runtime.start_lease_guard(lease)
                elif not bool(item.get("external_reconciled")):
                    try:
                        lease = runtime.acquire_case(
                            item["case_name"],
                            request=item["resource_request"],
                            stage_name=stage_name,
                        )
                    except Exception as exc:
                        if running:
                            break
                        prepared.popleft()
                        exit_code = 1
                        contract = _parallel_case_request(
                            item,
                            project_name=project_name,
                            stage_name=stage_name,
                        )
                        retain_parallel_result(
                            int(item["index"]),
                            _failed_case_result(contract, exc),
                        )
                        _print_parallel_error(
                            project_name,
                            stage_name,
                            item["case_name"],
                            exc,
                        )
                        if _stage_fail_fast(stage):
                            stop_launching = True
                        continue
                    lease_guard = runtime.start_lease_guard(lease)

                prepared.popleft()
                raw_resource_context = (
                    reconciled_resource_context
                    if reconciled_resource_context
                    else runtime.resource_context(
                        lease,
                        case_name=item["case_name"],
                        stage_name=stage_name,
                    )
                )
                resource_context = _resource_context_with_run_contract(
                    raw_resource_context,
                    identity=item["identity"],
                    control=item["control"],
                )
                contract = _parallel_case_request(
                    item,
                    project_name=project_name,
                    stage_name=stage_name,
                    resource_context=resource_context,
                )
                item["lease"] = lease
                item["lease_guard"] = lease_guard
                item["request_contract"] = contract
                item["started_at"] = time.monotonic()
                payload = {
                    "project_root": str(project_root),
                    "request": contract.as_dict(),
                    "extra_python_paths": [str(Path(path).resolve()) for path in extra_python_paths],
                    "framework": str(framework),
                }
                try:
                    if execution_backend == "external":
                        if external_transport is None:  # pragma: no cover - initialized above
                            raise RuntimeError("external transport was not initialized")
                        requirement_payload = item["resource_request"].requirement().as_dict()
                        requirement_payload["capabilities"] = list(
                            dict.fromkeys(
                                (
                                    "project_case",
                                    *tuple(requirement_payload.get("capabilities", ()) or ()),
                                )
                            )
                        )
                        prior_task_id = str(item.get("external_prior_task_id", ""))
                        task_id = str(item.get("external_task_id", ""))
                        reconciled = bool(item.get("external_reconciled"))
                        broker_record = item.get("external_broker_record")
                        if not task_id:
                            task_id = _external_task_id(
                                execution_run_id,
                                stage_name,
                                str(item["case_name"]),
                            )
                            envelope = TaskEnvelope(
                                task_id=task_id,
                                task_type="project_case",
                                payload=payload,
                                requirement=ResourceRequirement.from_dict(requirement_payload),
                                executor_backend="external",
                                input_refs=tuple(contract.input_artifacts.values()),
                                parent_task_id=(
                                    contract.identity.parent_case_run_id or None
                                ),
                                trace_id=contract.identity.root_run_id,
                                namespace=str(contract.resource_context.get("namespace", "default")),
                                max_retries=int(external_config["max_retries"]),
                                metadata={
                                    "project_name": project_name,
                                    "stage_name": stage_name,
                                    "case_name": item["case_name"],
                                    "project_run_id": execution_run_id,
                                    "run_identity": contract.identity.as_dict(),
                                    "execution_control": contract.control.as_dict(),
                                },
                            )
                            broker_record = external_transport.submit(envelope)
                        item["external_task_id"] = task_id
                        item["external_reconciled"] = reconciled
                        if on_external_task is not None:
                            on_external_task(
                                contract,
                                _external_task_manifest_payload(
                                    external_config,
                                    task_id=task_id,
                                    broker_status=(
                                        "unknown" if broker_record is None else broker_record.status
                                    ),
                                    reconciled=reconciled,
                                    resumed_from_task_id=(prior_task_id if reconciled else ""),
                                ),
                            )
                        future = pool.submit(
                            _await_external_case_task,
                            external_transport,
                            task_id,
                            queue_timeout_seconds=float(external_config["queue_timeout_seconds"]),
                            poll_interval_seconds=float(external_config["poll_interval_seconds"]),
                        )
                    else:
                        future = pool.submit(execute_case_payload, payload)
                except Exception as exc:
                    if lease_guard is not None:
                        lease_guard.close()
                    if lease is not None:
                        runtime.release(lease)
                    exit_code = 1
                    retain_parallel_result(
                        int(item["index"]),
                        _failed_case_result(contract, exc),
                    )
                    _print_parallel_error(project_name, stage_name, item["case_name"], exc)
                    if _stage_fail_fast(stage):
                        stop_launching = True
                    continue
                running[future] = item

            if not running:
                continue
            if stop_launching and not cancellation_signalled:
                for active in running.values():
                    active_contract = active.get("request_contract")
                    if isinstance(active_contract, CaseRunRequest):
                        CancellationToken(
                            active_contract.control.cancellation
                        ).cancel("parallel sibling failed under fail_fast policy")
                    if execution_backend == "external" and external_transport is not None:
                        task_id = str(active.get("external_task_id", "") or "")
                        if task_id:
                            external_transport.cancel(
                                task_id,
                                reason="parallel sibling failed under fail_fast policy",
                            )
                cancellation_signalled = True
            done, _ = wait(tuple(running), timeout=0.05, return_when=FIRST_COMPLETED)
            if not done:
                for active in running.values():
                    active_contract = active.get("request_contract")
                    if not isinstance(active_contract, CaseRunRequest):
                        continue
                    deadline = active_contract.control.deadline_at
                    if deadline > 0 and time.time() >= deadline:
                        CancellationToken(
                            active_contract.control.cancellation
                        ).cancel("case deadline exceeded")
                continue
            for future in done:
                item = running.pop(future)
                contract = item["request_contract"]
                elapsed = time.monotonic() - float(item["started_at"])
                try:
                    future_value = future.result()
                    if item["lease_guard"] is not None:
                        item["lease_guard"].assert_current()
                    if execution_backend == "external":
                        worker_result, task_result_payload = future_value
                        worker_result = dict(worker_result)
                        task_result_payload = dict(task_result_payload)
                        _assert_external_result_fence(runtime, task_result_payload)
                    else:
                        worker_result = dict(future_value)
                    case_result = CaseRunResult.from_dict(worker_result)
                    if bool(item.get("external_reconciled")) and case_result.ok:
                        case_result = replace(case_result, status="resumed")
                    refs = dict(case_result.artifact_refs)
                    if execution_backend == "external" and on_external_task is not None:
                        completed_task_id = str(item.get("external_task_id", ""))
                        completed_record = (
                            None
                            if external_transport is None or not completed_task_id
                            else external_transport.get(completed_task_id)
                        )
                        on_external_task(
                            contract,
                            _external_task_manifest_payload(
                                external_config,
                                task_id=completed_task_id,
                                broker_status=(
                                    "unknown"
                                    if completed_record is None
                                    else completed_record.status
                                ),
                                reconciled=bool(item.get("external_reconciled")),
                                resumed_from_task_id=(
                                    str(item.get("external_task_id", ""))
                                    if bool(item.get("external_reconciled"))
                                    else ""
                                ),
                            ),
                        )
                    retain_parallel_result(int(item["index"]), case_result)
                    artifacts_by_index[int(item["index"])] = (str(item["case_name"]), refs)
                    if not case_result.ok:
                        exit_code = 1
                        if _stage_fail_fast(stage):
                            stop_launching = True
                    _print_project_check(
                        project_name=project_name,
                        stage_name=stage_name,
                        case_name=item["case_name"],
                        mode=(
                            "external-worker"
                            if execution_backend == "external"
                            else "parallel-process"
                        ),
                        request=item["resource_request"],
                        # External retries return an attempt-specific request;
                        # audit the effective context actually used by the worker.
                        resource_context=case_result.request.resource_context,
                        state=dict(case_result.metadata.get("runtime_state", {}) or {}),
                        label="project-runtime",
                    )
                except Exception as exc:
                    exit_code = 1
                    if execution_backend == "external" and on_external_task is not None:
                        failed_task_id = str(item.get("external_task_id", ""))
                        failed_record = (
                            None
                            if external_transport is None or not failed_task_id
                            else external_transport.get(failed_task_id)
                        )
                        on_external_task(
                            contract,
                            _external_task_manifest_payload(
                                external_config,
                                task_id=failed_task_id,
                                broker_status=(
                                    "unknown" if failed_record is None else failed_record.status
                                ),
                                reconciled=bool(item.get("external_reconciled")),
                                resumed_from_task_id=(
                                    failed_task_id
                                    if bool(item.get("external_reconciled"))
                                    else ""
                                ),
                            ),
                        )
                    retain_parallel_result(
                        int(item["index"]),
                        _failed_case_result(
                            contract,
                            exc,
                            elapsed_seconds=elapsed,
                        ),
                    )
                    _print_parallel_error(project_name, stage_name, item["case_name"], exc)
                    if _stage_fail_fast(stage):
                        stop_launching = True
                finally:
                    if item["lease_guard"] is not None:
                        item["lease_guard"].close()
                    if item["lease"] is not None:
                        runtime.release(item["lease"])

    if prepared:
        for item in prepared:
            contract = _parallel_case_request(
                item,
                project_name=project_name,
                stage_name=stage_name,
            )
            retain_parallel_result(int(item["index"]), CaseRunResult(
                request=contract,
                status="skipped",
                exit_code=1,
                failure=CaseFailure(
                    kind="FailFastSkip",
                    message="Not started because parallel stage failure_policy=fail_fast",
                    phase="schedule",
                ),
            ))
        exit_code = 1

    ordered_results = [results_by_index[index] for index in sorted(results_by_index)]
    ordered_artifacts = [artifacts_by_index[index] for index in sorted(artifacts_by_index)]
    return ordered_results, ordered_artifacts, exit_code


def _external_stage_config(
    stage: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    raw = stage.get("external", {})
    if not isinstance(raw, Mapping):
        raise ProjectConfigurationError("Stage field 'external' must be a mapping")
    config = dict(raw)
    backend = str(config.get("backend", "sqlite") or "sqlite").strip().lower()
    if backend not in {"sqlite", "redis"}:
        raise ProjectConfigurationError(
            f"Unsupported external transport backend '{backend}'; expected sqlite or redis"
        )
    transport_path = None
    redis_url = ""
    namespace = ""
    if backend == "sqlite":
        if "transport_path" not in config:
            raise ProjectConfigurationError(
                "SQLite external Stage must declare external.transport_path explicitly"
            )
        transport_path = Path(str(config["transport_path"]))
        if not transport_path.is_absolute():
            transport_path = Path(project_root).resolve() / transport_path
        transport_path = transport_path.resolve()
    else:
        redis_url = str(config.get("redis_url", "") or "").strip()
        namespace = str(config.get("namespace", "") or "").strip().rstrip(":")
        if not redis_url:
            raise ProjectConfigurationError(
                "Redis external Stage must declare external.redis_url explicitly"
            )
        if not namespace:
            raise ProjectConfigurationError(
                "Redis external Stage must declare external.namespace explicitly"
            )
    queue_timeout = float(config.get("queue_timeout_seconds", 30.0) or 30.0)
    poll_interval = float(config.get("poll_interval_seconds", 0.05) or 0.05)
    if queue_timeout <= 0:
        raise ProjectConfigurationError("external.queue_timeout_seconds must be positive")
    if poll_interval <= 0:
        raise ProjectConfigurationError("external.poll_interval_seconds must be positive")
    return {
        "backend": backend,
        "transport_path": transport_path,
        "redis_url": redis_url,
        "namespace": namespace,
        "queue_timeout_seconds": queue_timeout,
        "poll_interval_seconds": poll_interval,
        "max_retries": max(0, int(config.get("max_retries", 0) or 0)),
    }


def _build_external_transport(config: Mapping[str, Any]) -> TaskTransport:
    backend = str(config.get("backend", "sqlite"))
    if backend == "redis":
        return RedisTaskTransport(
            str(config["redis_url"]),
            namespace=str(config["namespace"]),
        )
    return SQLiteTaskTransport(Path(config["transport_path"]))


def _external_transport_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    audit = {
        "stage_execution_backend": "external",
        "external_transport_backend": str(config["backend"]),
        "external_max_retries": int(config["max_retries"]),
    }
    if config["backend"] == "sqlite":
        audit["external_transport_path"] = str(config["transport_path"])
    else:
        audit["external_transport_namespace"] = str(config["namespace"])
    return audit


def _external_task_manifest_payload(
    config: Mapping[str, Any],
    *,
    task_id: str,
    broker_status: str,
    reconciled: bool,
    resumed_from_task_id: str = "",
) -> dict[str, Any]:
    payload = {
        "task_id": str(task_id),
        "backend": str(config["backend"]),
        "broker_status": str(broker_status),
        "reconciled": bool(reconciled),
        "resumed_from_task_id": str(resumed_from_task_id),
    }
    if config["backend"] == "sqlite":
        payload["transport_path"] = str(config["transport_path"])
    else:
        payload["namespace"] = str(config["namespace"])
    return payload


def _reconcile_prior_external_task(
    transport: TaskTransport,
    task_id: str,
    *,
    runtime: ProjectL0Runtime,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> Any | None:
    """Return an adoptable old task or retire it before a fresh grant is issued."""

    deadline = time.monotonic() + max(0.1, float(timeout_seconds))
    poll = max(0.001, float(poll_interval_seconds))
    while True:
        transport.recover_expired()
        record = transport.get(task_id)
        if record is None or record.status in {"failed", "cancelled"}:
            return None
        if record.status == "succeeded":
            return record
        if record.status == "queued":
            if transport.cancel(
                task_id,
                reason="superseded by resumed Project attempt",
            ):
                return None
            continue
        if record.status == "leased":
            request_payload = record.task.payload.get("request")
            resource_context = (
                dict(CaseRunRequest.from_dict(request_payload).resource_context)
                if isinstance(request_payload, Mapping)
                else {}
            )
            lease_payload = dict(resource_context.get("lease", {}) or {})
            if lease_payload:
                lease = ResourceLease.from_dict(lease_payload)
                if runtime.allocator.is_current(lease):
                    return record
            if time.monotonic() >= deadline:
                raise ResourceLeaseFenceError(
                    f"Prior external task '{task_id}' still holds a broker lease "
                    "after its Project L0 fence expired"
                )
            time.sleep(poll)
            continue
        return None


def _await_external_case_task(
    transport: TaskTransport,
    task_id: str,
    *,
    queue_timeout_seconds: float,
    poll_interval_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Wait for a durable Case task while preserving queued-vs-leased semantics."""

    queue_deadline = time.monotonic() + float(queue_timeout_seconds)
    poll = max(0.001, float(poll_interval_seconds))
    previous_status = "queued"
    while True:
        transport.recover_expired()
        record = transport.get(task_id)
        if record is None:
            raise RuntimeError(f"External task disappeared: {task_id}")
        if record.status == "succeeded":
            if record.result is None:
                raise RuntimeError(f"External task has no TaskResult: {task_id}")
            if not record.result.ok:
                raise RuntimeError(record.result.error or "external worker returned a failed result")
            return dict(record.result.output), record.result.as_dict()
        if record.status in {"failed", "cancelled"}:
            if record.result is not None and record.result.output:
                candidate = dict(record.result.output)
                if "schema_version" in candidate and "request" in candidate:
                    CaseRunResult.from_dict(candidate)
                    return candidate, record.result.as_dict()
            error = record.error
            if record.result is not None and record.result.error:
                error = record.result.error
            raise RuntimeError(
                f"External Case task '{task_id}' ended as {record.status}: {error}"
            )
        if record.status == "queued" and previous_status != "queued":
            queue_deadline = time.monotonic() + float(queue_timeout_seconds)
        if record.status == "queued" and time.monotonic() >= queue_deadline:
            if transport.cancel(task_id, reason="external worker queue timeout"):
                raise TimeoutError(
                    f"No compatible external worker claimed task '{task_id}' "
                    f"within {queue_timeout_seconds:.3f}s"
                )
        previous_status = record.status
        time.sleep(poll)


def _external_task_id(run_id: str, stage_name: str, case_name: str) -> str:
    """Deterministic broker identity closes submit-before-manifest crash windows."""

    return f"project:{str(run_id)}:{str(stage_name)}:{str(case_name)}"


def _assert_external_result_fence(
    runtime: ProjectL0Runtime,
    task_result: Mapping[str, Any],
) -> None:
    """Require a worker-side fence check when durable L0 authority is enabled."""

    if runtime.config.lease_backend not in {"sqlite", "redis"}:
        return
    metadata = dict(task_result.get("metadata", {}) or {})
    resource_context = dict(task_result.get("resource_context", {}) or {})
    lease = dict(resource_context.get("lease", {}) or {})
    result_token = int(metadata.get("fencing_token", 0) or 0)
    lease_token = int(lease.get("fencing_token", 0) or 0)
    if not bool(metadata.get("lease_fence_validated")):
        raise ResourceLeaseFenceError(
            "External worker result was not committed under a validated Project L0 fence"
        )
    if result_token <= 0 or result_token != lease_token:
        raise ResourceLeaseFenceError(
            "External worker result fencing token does not match its ResourceContext"
        )


def _case_identity(
    execution_run_id: str,
    stage_name: str,
    case_name: str,
) -> CaseRunIdentity:
    case_run_id = f"case:{execution_run_id}:{stage_name}:{case_name}"
    return CaseRunIdentity(
        project_run_id=execution_run_id,
        root_run_id=execution_run_id,
        case_run_id=case_run_id,
        invocation_id=case_run_id,
    )


def _case_control(
    runtime: ProjectL0Runtime,
    stage: Mapping[str, Any],
    case_name: str,
) -> ExecutionControl:
    raw_timeouts = stage.get("case_timeout_seconds", {})
    if raw_timeouts is None:
        raw_timeouts = {}
    if not isinstance(raw_timeouts, Mapping):
        raise ProjectConfigurationError("Stage field 'case_timeout_seconds' must be a mapping")
    raw_timeout = raw_timeouts.get(case_name, stage.get("timeout_seconds", 0.0))
    timeout = float(raw_timeout or 0.0)
    if timeout < 0:
        raise ProjectConfigurationError("Case timeout_seconds must be non-negative")
    deadline = 0.0 if timeout == 0 else time.time() + timeout
    termination_payload = runtime.config.termination.as_dict()
    stage_termination = stage.get("termination", {}) or {}
    if not isinstance(stage_termination, Mapping):
        raise ProjectConfigurationError("Stage field 'termination' must be a mapping")
    termination_payload.update(dict(stage_termination))
    case_terminations = stage.get("case_termination", {}) or {}
    if not isinstance(case_terminations, Mapping):
        raise ProjectConfigurationError("Stage field 'case_termination' must be a mapping")
    case_termination = case_terminations.get(case_name, {}) or {}
    if not isinstance(case_termination, Mapping):
        raise ProjectConfigurationError(
            f"Stage case_termination['{case_name}'] must be a mapping"
        )
    termination_payload.update(dict(case_termination))
    return ExecutionControl(
        cancellation=runtime.new_cancellation_ref(deadline_at=deadline),
        termination=TerminationPolicy.from_dict(termination_payload),
        metadata={"timeout_seconds": timeout},
    )


def _resource_context_with_run_contract(
    resource_context: Any,
    *,
    identity: CaseRunIdentity,
    control: ExecutionControl,
) -> dict[str, Any]:
    payload = _as_dict(resource_context)
    metadata = dict(payload.get("metadata", {}) or {})
    metadata.update(
        {
            "run_identity": identity.as_dict(),
            "execution_control": control.as_dict(),
        }
    )
    payload["metadata"] = metadata
    return payload


def _parallel_case_request(
    item: Mapping[str, Any],
    *,
    project_name: str,
    stage_name: str,
    resource_context: Any = None,
) -> CaseRunRequest:
    return CaseRunRequest(
        project_name=project_name,
        stage_name=stage_name,
        case_name=str(item["case_name"]),
        case_kind=str(item["case_kind"]),
        mode=str(item["mode"]),
        identity=item["identity"],
        control=item["control"],
        resource_request=item["resource_request"].as_dict(),
        resource_context=_as_dict(resource_context),
        component_overrides=dict(item["component_overrides"]),
        input_artifacts=dict(item["input_artifacts"]),
        argv=tuple(item["argv"]),
    )


def _failed_case_result(
    request: CaseRunRequest,
    exc: BaseException,
    *,
    elapsed_seconds: float = 0.0,
) -> CaseRunResult:
    return CaseRunResult(
        request=request,
        status="failed",
        elapsed_seconds=elapsed_seconds,
        exit_code=1,
        failure=CaseFailure.from_exception(exc, phase="project"),
    )


def _print_parallel_error(project_name: str, stage_name: str, case_name: str, exc: BaseException) -> None:
    _print_project_message(
        {
            "project": project_name,
            "stage": stage_name,
            "case": case_name,
            "mode": "parallel_execution_error",
            "error": f"{type(exc).__name__}: {exc}",
        },
        label="project-error",
    )


def _stage_fail_fast(stage: Mapping[str, Any]) -> bool:
    policy = str(stage.get("failure_policy", "fail_fast") or "fail_fast").strip().lower()
    return policy not in {"continue", "continue_on_error"}


def _case_mapping(stage: Mapping[str, Any], field_name: str, case_name: str) -> dict[str, Any]:
    values = stage.get(field_name, {})
    if not isinstance(values, Mapping):
        raise ProjectConfigurationError(f"Stage field '{field_name}' must be a mapping")
    selected = values.get(case_name, {})
    if selected is None:
        return {}
    if not isinstance(selected, Mapping):
        raise ProjectConfigurationError(
            f"Stage field '{field_name}.{case_name}' must be a mapping"
        )
    return dict(selected)


def _resolve_case_input_artifacts(
    stage: Mapping[str, Any],
    *,
    case_name: str,
    artifact_registry: Mapping[str, DataRef],
) -> dict[str, DataRef]:
    declared = _case_mapping(stage, "input_artifacts", case_name)
    resolved: dict[str, DataRef] = {}
    for input_name, registry_key in declared.items():
        key = str(registry_key)
        if key not in artifact_registry:
            raise ProjectConfigurationError(
                f"Case '{case_name}' requires missing artifact ref '{key}' for input '{input_name}'"
            )
        resolved[str(input_name)] = artifact_registry[key]
    return resolved


def _inject_case_input_artifacts(
    case_obj: Any,
    input_artifacts: Mapping[str, DataRef],
    *,
    case_name: str,
) -> None:
    if not input_artifacts:
        return
    setter = getattr(case_obj, "set_input_artifacts", None)
    if not callable(setter):
        raise ProjectConfigurationError(
            f"Case '{case_name}' declares input_artifacts but does not implement "
            "set_input_artifacts(refs)"
        )
    setter(dict(input_artifacts))


def _normalize_case_output(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    as_dict = getattr(raw, "as_dict", None)
    if callable(as_dict):
        value = as_dict()
        return dict(value) if isinstance(value, Mapping) else {"raw": value}
    return {"raw": raw}


def _collect_artifact_refs(output: Mapping[str, Any]) -> dict[str, DataRef]:
    refs: dict[str, DataRef] = {}
    for field_name in ("artifact_refs", "artifacts"):
        raw = output.get(field_name)
        if not isinstance(raw, Mapping):
            continue
        for artifact_name, value in raw.items():
            ref = _coerce_data_ref(value)
            if ref is not None:
                refs[str(artifact_name)] = ref
    return refs


def _coerce_data_ref(value: Any) -> DataRef | None:
    if isinstance(value, DataRef):
        return value
    if isinstance(value, (str, Path)):
        return DataRef(uri=str(value))
    if not isinstance(value, Mapping):
        describe = getattr(value, "as_dict", None)
        if callable(describe):
            value = describe()
    if not isinstance(value, Mapping):
        return None
    payload = dict(value)
    if "uri" not in payload and payload.get("path"):
        payload["uri"] = str(payload["path"])
    if not payload.get("uri"):
        return None
    return DataRef.from_dict(payload)


def _register_case_artifacts(
    registry: dict[str, DataRef],
    *,
    stage_name: str,
    case_name: str,
    artifact_refs: Mapping[str, DataRef],
) -> None:
    for artifact_name, ref in artifact_refs.items():
        qualified = f"{stage_name}.{case_name}.{artifact_name}"
        case_qualified = f"{case_name}.{artifact_name}"
        registry[qualified] = ref
        registry[case_qualified] = ref
        registry.setdefault(str(artifact_name), ref)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a standard Project/Case/Scaffold project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--group", default="default", help="Execution group from project_config.py.")
    parser.add_argument("--check", action="store_true", help="Print assembly/resource audit without running cases.")
    parser.add_argument(
        "--build-check",
        action="store_true",
        help="In --check mode, instantiate each case builder or CLI check surface.",
    )
    parser.add_argument("--run-id", default=None, help="Optional durable run manifest identifier.")
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Prior run id, run directory, or manifest.json to resume from.",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable .blackbase/runs manifest persistence for this run.",
    )
    parser.add_argument(
        "case_args",
        nargs=argparse.REMAINDER,
        help="Optional arguments forwarded to CLI-mode cases after '--'.",
    )
    return parser


def main(
    project_root: Path | str | None = None,
    argv: Sequence[str] | None = None,
    *,
    framework: str = "blackbase",
    resource_env_var: str | None = None,
    extra_python_paths: Sequence[Path | str] = (),
) -> int:
    args = build_parser().parse_args(argv)
    case_args = tuple(args.case_args or ())
    if case_args and case_args[0] == "--":
        case_args = case_args[1:]
    root = Path(project_root).resolve() if project_root is not None else Path.cwd()
    return run_project(
        root,
        group=str(args.group),
        check=bool(args.check),
        build_check=bool(args.build_check),
        case_args=case_args,
        framework=framework,
        resource_env_var=resource_env_var,
        extra_python_paths=extra_python_paths,
        record=not bool(args.no_record),
        run_id=args.run_id,
        resume_from=args.resume_from,
    )


def _load_project_config(project_root: Path, *, framework: str):
    path = Path(project_root).resolve() / "project_config.py"
    spec = importlib.util.spec_from_file_location(f"{framework}_project_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import project_config.py from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[call-arg]
    return module


def _run_case_cli(
    *,
    project_root: Path,
    case_name: str,
    case_kind: str,
    argv: Sequence[str],
    resource_context: Any,
    resource_env_var: str,
    extra_python_paths: Sequence[Path | str],
) -> int:
    case_root = Path(project_root).resolve() / "cases" / str(case_name)
    argv = [str(item) for item in argv]
    entry_module = "run_solver"
    entry_name = f"{entry_module}.py"
    run_entry = case_root / entry_name
    if not run_entry.is_file():
        _print_project_message(
            {
                "case": str(case_name),
                "mode": "cli_entry_missing",
                "reason": f"case kind={case_kind} requires {entry_name}",
            }
        )
        return 2
    if "--check" in argv and not path_declares_check_argument(run_entry):
        _print_project_message(
            {
                "case": str(case_name),
                "mode": "cli_check_unavailable",
                "reason": f"case {entry_name} does not declare --check",
            }
        )
        return 0
    env = os.environ.copy()
    payload = json.dumps(_as_dict(resource_context), ensure_ascii=False)
    env["BLACKBASE_RESOURCE_CONTEXT_JSON"] = payload
    env[str(resource_env_var)] = payload
    python_path_parts = [
        str(case_root),
        str(Path(project_root).resolve()),
        *(str(Path(p).resolve()) for p in extra_python_paths),
    ]
    current_python_path = env.get("PYTHONPATH", "")
    if current_python_path:
        python_path_parts.append(current_python_path)
    env["PYTHONPATH"] = os.pathsep.join(python_path_parts)
    cmd = [sys.executable, "-m", f"cases.{case_name}.{entry_module}", *argv]
    proc = subprocess.run(cmd, cwd=str(Path(project_root).resolve()), env=env, check=False)
    return int(proc.returncode)


def _case_runtime_state(case_obj: Any) -> dict[str, Any]:
    plugins = getattr(getattr(case_obj, "plugin_manager", None), "plugins", []) or []
    providers = getattr(getattr(case_obj, "evaluation_mediator", None), "list_providers", None)
    pipeline = (
        getattr(case_obj, "representation_pipeline", None)
        or getattr(case_obj, "pipeline", None)
        or getattr(case_obj, "representation", None)
    )
    return {
        "case_class": type(case_obj).__name__,
        "problem": type(getattr(case_obj, "problem", None)).__name__,
        "pipeline": type(pipeline).__name__,
        "adapter": type(getattr(case_obj, "adapter", None)).__name__,
        "providers": len(tuple(providers())) if callable(providers) else 0,
        "plugins": len(tuple(plugins)),
        "resource_context": _as_dict(getattr(case_obj, "resource_context", None)),
    }


def _print_project_check(
    *,
    project_name: str,
    stage_name: str,
    case_name: str,
    mode: str,
    request: Any,
    resource_context: Any,
    state: Mapping[str, Any] | None = None,
    label: str = "project-check",
) -> None:
    _print_project_message(
        {
            "project": str(project_name),
            "stage": str(stage_name),
            "case": str(case_name),
            "mode": str(mode),
            "resource_request": _as_dict(request),
            "resource_context": _as_dict(resource_context),
            "runtime_state": dict(state or {}),
        },
        label=label,
    )


def _print_project_message(payload: Mapping[str, Any], *, label: str = "project-runtime") -> None:
    print(f"[{label}] " + json.dumps(dict(payload), ensure_ascii=False, sort_keys=True), flush=True)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return dict(value.as_dict())
    if isinstance(value, Mapping):
        return dict(value)
    return {}
