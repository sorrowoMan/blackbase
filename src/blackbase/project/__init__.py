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
    CaseInvocationError,
    CaseRunIdentity,
    CaseRunRequest,
    CaseRunResult,
    ChildResourceGrant,
    ExecutionControl,
    ProjectConfigurationError,
    ProjectRunResult,
)
from .invocation import CaseExecutor, CaseInvoker, CaseRuntimeContext
from .case_stages import CaseStage, CaseStageResult, CaseStageRunner, ChildCaseCall
from .case_binding import (
    CASE_RESOURCE_BINDING_SCHEMA_VERSION,
    CaseResourceBindingAudit,
    bind_case_resource_context,
    case_resource_binding_audit,
)
from .case_cli import run_standard_case_cli
from .catalog import CatalogScope, find_catalog_scope, iter_catalog_scopes, load_scaffold_catalog_entries
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
from .scaffold import add_case, add_component, create_project

__all__ = [
    "CaseBuilderProxy",
    "CASE_RUN_SCHEMA_VERSION",
    "CASE_RESOURCE_BINDING_SCHEMA_VERSION",
    "CaseExecutor",
    "CaseFailure",
    "CaseInvocationError",
    "CaseInvoker",
    "CaseRunIdentity",
    "CaseRunRequest",
    "CaseRunResult",
    "CaseRuntimeContext",
    "CaseResourceBindingAudit",
    "CaseStage",
    "CaseStageResult",
    "CaseStageRunner",
    "CatalogScope",
    "ChildResourceGrant",
    "ChildCaseCall",
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
    "bind_case_resource_context",
    "build_case_check_payload",
    "build_parser",
    "case_import_context",
    "case_resource_binding_audit",
    "create_project",
    "format_doctor_report",
    "format_case_check",
    "format_resource_context_summary",
    "find_catalog_scope",
    "execute_project",
    "iter_group_stages",
    "iter_catalog_scopes",
    "load_case_builder",
    "load_case_kind",
    "load_case_resource_request",
    "load_project_runtime_config",
    "load_resume_manifest",
    "load_resource_context_from_env",
    "load_scaffold_catalog_entries",
    "main",
    "project_import_context",
    "run_case",
    "run_standard_case_cli",
    "run_common_project_doctor",
    "run_project",
    "print_case_check",
    "print_resource_context_summary",
    "handle_project_command",
]
