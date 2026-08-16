"""Shared Project / Case / Scaffold / L0 substrate."""

from __future__ import annotations

from .cli import add_project_subcommands, handle_project_command
from .check_output import (
    build_case_check_payload,
    format_case_check,
    format_resource_context_summary,
    print_case_check,
    print_resource_context_summary,
)
from .doctor import DoctorDiagnostic, DoctorReport, format_doctor_report, run_common_project_doctor
from .execution import (
    CASE_RUN_SCHEMA_VERSION,
    CaseFailure,
    CaseRunIdentity,
    CaseRunRequest,
    CaseRunResult,
    ChildResourceGrant,
    ExecutionControl,
    ProjectConfigurationError,
    ProjectRunResult,
)
from .invocation import CaseExecutor, CaseInvoker, CaseRuntimeContext
from .project_runner import build_parser, execute_project, main, run_project
from .run_manifest import ProjectRunManifest, ProjectRunRecorder, load_resume_manifest
from .runtime import (
    CaseBuilderProxy,
    ProjectL0Runtime,
    ProjectRuntimeConfig,
    ResourceLeaseFenceError,
    ResourceLeaseGuard,
    build_case,
    case_import_context,
    iter_group_stages,
    load_case_builder,
    load_case_kind,
    load_case_resource_request,
    load_project_runtime_config,
    load_resource_context_from_env,
    project_import_context,
    run_case,
)
from .scaffold import add_case, add_component, create_project, init_project

__all__ = [
    "CaseBuilderProxy",
    "CASE_RUN_SCHEMA_VERSION",
    "CaseExecutor",
    "CaseFailure",
    "CaseInvoker",
    "CaseRunIdentity",
    "CaseRunRequest",
    "CaseRunResult",
    "CaseRuntimeContext",
    "ChildResourceGrant",
    "DoctorDiagnostic",
    "DoctorReport",
    "ProjectL0Runtime",
    "ProjectConfigurationError",
    "ProjectRunResult",
    "ProjectRunManifest",
    "ProjectRunRecorder",
    "ProjectRuntimeConfig",
    "ExecutionControl",
    "ResourceLeaseFenceError",
    "ResourceLeaseGuard",
    "add_case",
    "add_component",
    "add_project_subcommands",
    "build_case",
    "build_case_check_payload",
    "build_parser",
    "case_import_context",
    "create_project",
    "format_doctor_report",
    "format_case_check",
    "format_resource_context_summary",
    "execute_project",
    "init_project",
    "iter_group_stages",
    "load_case_builder",
    "load_case_kind",
    "load_case_resource_request",
    "load_project_runtime_config",
    "load_resume_manifest",
    "load_resource_context_from_env",
    "main",
    "project_import_context",
    "run_case",
    "run_common_project_doctor",
    "run_project",
    "print_case_check",
    "print_resource_context_summary",
    "handle_project_command",
]
