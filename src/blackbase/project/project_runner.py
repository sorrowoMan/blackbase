"""Shared Project/Case runner for standard scaffold projects."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .runtime import (
    ProjectL0Runtime,
    build_case,
    case_import_context,
    iter_group_stages,
    load_case_builder,
    load_case_kind,
    load_case_resource_request,
    load_project_runtime_config,
    run_case,
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
) -> int:
    root = Path(project_root).resolve()
    config_module = _load_project_config(root, framework=framework)
    runtime = ProjectL0Runtime(load_project_runtime_config(config_module))
    project_name = str(getattr(config_module, "PROJECT_NAME", root.name))
    case_args = tuple(case_args or ())
    resource_env_var = resource_env_var or f"{str(framework).upper()}_RESOURCE_CONTEXT_JSON"
    exit_code = 0

    for stage in iter_group_stages(config_module, str(group)):
        stage_name = str(stage.get("name", "stage"))
        case_names = tuple(str(name) for name in (stage.get("cases", ()) or ()))
        stage_cli_args = dict(stage.get("case_args", {}) or {})
        stage_modes = dict(stage.get("case_modes", {}) or {})
        for case_name in case_names:
            case_kind = load_case_kind(root, case_name, stage=stage, default="solver")
            request = load_case_resource_request(
                case_name,
                project_root=root,
                stage=stage,
                default=runtime.config.default_request,
                extra_import_paths=extra_python_paths,
            )
            lease = runtime.acquire_case(case_name, request=request, stage_name=stage_name)
            try:
                resource_context = runtime.resource_context(
                    lease,
                    case_name=case_name,
                    stage_name=stage_name,
                )
                mode = str(stage_modes.get(case_name, stage.get("mode", "build")) or "build")
                if mode == "cli":
                    argv = tuple(stage_cli_args.get(case_name, ())) + case_args
                    if check and "--check" not in argv:
                        argv = ("--check", *argv)
                    _print_project_check(
                        project_name=project_name,
                        stage_name=stage_name,
                        case_name=case_name,
                        mode="cli",
                        request=request,
                        resource_context=resource_context,
                    )
                    if build_check or not check:
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
                    continue

                if check and not build_check:
                    _print_project_check(
                        project_name=project_name,
                        stage_name=stage_name,
                        case_name=case_name,
                        mode=case_kind,
                        request=request,
                        resource_context=resource_context,
                    )
                    continue

                with case_import_context(root, case_name, extra_import_paths=extra_python_paths):
                    builder = load_case_builder(
                        root,
                        case_name,
                        case_kind=case_kind,
                        extra_import_paths=extra_python_paths,
                    )
                    case_obj = build_case(builder, resource_context=resource_context)
                    _print_project_check(
                        project_name=project_name,
                        stage_name=stage_name,
                        case_name=case_name,
                        mode=case_kind,
                        request=request,
                        resource_context=resource_context,
                        state=_case_runtime_state(case_obj),
                    )
                    if not check:
                        run_case(case_obj)
            finally:
                runtime.release(lease)
    return int(exit_code)


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
    entry_module = "run_trainer" if case_kind == "trainer" else "run_solver"
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
    if "--check" in argv and not _text_declares_check_argument(_read_text(run_entry)):
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
        }
    )


def _print_project_message(payload: Mapping[str, Any]) -> None:
    print("[project-check] " + json.dumps(dict(payload), ensure_ascii=False, sort_keys=True), flush=True)


def _text_declares_check_argument(text: str) -> bool:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_add_argument = (
            isinstance(func, ast.Attribute) and func.attr == "add_argument"
        ) or (
            isinstance(func, ast.Name) and func.id == "add_argument"
        )
        if not is_add_argument:
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and arg.value == "--check":
                return True
    return False


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig", errors="replace")
    except Exception:
        return ""


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return dict(value.as_dict())
    if isinstance(value, Mapping):
        return dict(value)
    return {}
