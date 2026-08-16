"""Shared CLI helpers for Project/Case/Scaffold commands."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable


def add_project_subcommands(
    parser: argparse.ArgumentParser,
    *,
    default_case_type: str = "solver",
    include_framework_option: bool = False,
) -> None:
    """Attach common project subcommands to an argparse parser."""

    sub = parser.add_subparsers(dest="project_command", required=True)

    p_init = sub.add_parser("init", help="Create a local project scaffold")
    p_init.add_argument("path", help="Target directory for the project")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing directory")
    p_init.set_defaults(project_action="init")

    p_new = sub.add_parser("new", help="Create a new project scaffold")
    p_new.add_argument("project_name", help="Name/path of the new project directory")
    p_new.add_argument("--force", action="store_true", help="Overwrite existing directory")
    p_new.set_defaults(project_action="new")

    p_add = sub.add_parser("add-case", help="Add a solver/trainer case to the current project")
    p_add.add_argument("case_name", help="Name of the new case directory")
    p_add.add_argument(
        "--type",
        choices=("solver", "trainer"),
        default=default_case_type,
        help=f"Case type (default: {default_case_type})",
    )
    if include_framework_option:
        p_add.add_argument(
            "--framework",
            choices=("nsgablack", "mlblack"),
            default=None,
            help="Template family for case-level semantics.",
        )
    p_add.set_defaults(project_action="add-case")

    p_component = sub.add_parser(
        "add-component",
        help="Add one component file into the matching case component folder",
    )
    p_component.add_argument("--case", dest="case_name", default=None, help="Target case name")
    p_component.add_argument(
        "--kind",
        choices=("problem", "pipeline", "adapter", "bias", "plugin"),
        required=True,
        help="Component kind mapped to problem/pipeline/adapter/bias/plugins",
    )
    p_component.add_argument("--name", required=True, help="Component file name without .py")
    p_component.add_argument(
        "--slot",
        default=None,
        help="Optional pipeline slot, e.g. main/init/mutate/repair/codec/head/transform",
    )
    p_component.set_defaults(project_action="add-component")

    p_doctor = sub.add_parser("doctor", help="Check project structure and contracts")
    p_doctor.add_argument("--path", default=".", help="Project/package root to inspect")
    p_doctor.add_argument("--strict", action="store_true", help="Escalate shared warnings where applicable")
    p_doctor.set_defaults(project_action="doctor")

    p_run = sub.add_parser("run", help="Run a standard Project/Case scaffold")
    p_run.add_argument("--path", default=".", help="Project root to run")
    p_run.add_argument("--group", default="default", help="Execution group from project_config.py")
    p_run.add_argument("--check", action="store_true", help="Print assembly/resource audit without running cases")
    p_run.add_argument("--build-check", action="store_true", help="Instantiate builders in check mode")
    p_run.add_argument("--run-id", default=None, help="Optional durable run manifest identifier")
    p_run.add_argument(
        "--resume-from",
        default=None,
        help="Prior run id, run directory, or manifest.json to resume from",
    )
    p_run.add_argument("--no-record", action="store_true", help="Disable run manifest persistence")
    p_run.add_argument("case_args", nargs=argparse.REMAINDER, help="Arguments forwarded after '--'")
    p_run.set_defaults(project_action="run")


def handle_project_command(
    args: argparse.Namespace,
    *,
    framework: str,
    create_project: Callable[..., Any],
    add_case: Callable[..., Any],
    add_component: Callable[..., Any],
    run_project_doctor: Callable[..., Any],
    format_doctor_report: Callable[..., str],
    run_project: Callable[..., int],
) -> int:
    """Dispatch one common project subcommand."""

    action = str(getattr(args, "project_action", getattr(args, "project_command", "")))
    if action == "init":
        root = create_project(Path(args.path), force=bool(getattr(args, "force", False)))
        print(f"Project created at: {root}")
        return 0
    if action == "new":
        root = create_project(Path(args.project_name), force=bool(getattr(args, "force", False)))
        print(f"Project created at: {root}")
        return 0
    if action == "add-case":
        target_framework = getattr(args, "framework", None) or framework
        case_root = add_case(args.case_name, args.type, framework=target_framework)
        print(f"Case created at: {case_root}")
        return 0
    if action == "add-component":
        out = add_component(
            args.name,
            args.kind,
            case_name=getattr(args, "case_name", None),
            slot=getattr(args, "slot", None),
        )
        if out is None:
            return 1
        print(f"Created: {out}")
        return 0
    if action == "doctor":
        report = run_project_doctor(args.path, strict=bool(getattr(args, "strict", False)))
        print(format_doctor_report(report))
        return 0 if getattr(report, "ok", False) else 1
    if action == "run":
        case_args = tuple(getattr(args, "case_args", ()) or ())
        if case_args and case_args[0] == "--":
            case_args = case_args[1:]
        return int(
            run_project(
                Path(args.path),
                group=str(getattr(args, "group", "default")),
                check=bool(getattr(args, "check", False)),
                build_check=bool(getattr(args, "build_check", False)),
                case_args=case_args,
                record=not bool(getattr(args, "no_record", False)),
                run_id=getattr(args, "run_id", None),
                resume_from=getattr(args, "resume_from", None),
            )
        )
    return 2
