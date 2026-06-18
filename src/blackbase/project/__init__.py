"""Shared Project / Case / Scaffold / L0 substrate."""

from __future__ import annotations

from .cli import add_project_subcommands, handle_project_command
from .doctor import DoctorDiagnostic, DoctorReport, format_doctor_report, run_common_project_doctor
from .project_runner import build_parser, main, run_project
from .runtime import (
    CaseBuilderProxy,
    ProjectL0Runtime,
    ProjectRuntimeConfig,
    build_case,
    case_import_context,
    iter_group_stages,
    load_case_builder,
    load_case_kind,
    load_case_resource_request,
    load_project_runtime_config,
    run_case,
)
from .scaffold import add_case, add_component, create_project, init_project

__all__ = [
    "CaseBuilderProxy",
    "DoctorDiagnostic",
    "DoctorReport",
    "ProjectL0Runtime",
    "ProjectRuntimeConfig",
    "add_case",
    "add_component",
    "add_project_subcommands",
    "build_case",
    "build_parser",
    "case_import_context",
    "create_project",
    "format_doctor_report",
    "init_project",
    "iter_group_stages",
    "load_case_builder",
    "load_case_kind",
    "load_case_resource_request",
    "load_project_runtime_config",
    "main",
    "run_case",
    "run_common_project_doctor",
    "run_project",
    "handle_project_command",
]
