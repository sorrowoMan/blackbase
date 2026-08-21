"""Shared scaffold filesystem operations for Project/Case substrate."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Callable, Mapping


ComponentTemplateProvider = Callable[[str, str, str | None], str]


_COMPONENT_KIND_TO_DIR = {
    "problem": "problem",
    "pipeline": "pipeline",
    "adapter": "adapter",
    "bias": "bias",
    "plugin": "plugins",
}
_CASE_DIRS = ("problem", "pipeline", "adapter", "bias", "plugins", "evaluation", "runtime", "solver")
_PIPELINE_DEFAULT_SLOT = "custom"
_PIPELINE_SLOT_TO_SUBDIR = {
    "entry": ".",
    "main": ".",
    "representation": "representation",
    "init": "operators/init",
    "mutate": "operators/mutate",
    "repair": "operators/repair",
    "encode": "operators/encode",
    "decode": "operators/decode",
    "transform": "operators/transform",
    "codec": "operators/codec",
    "head": "operators/head",
    "custom": "operators/custom",
}


def create_project(
    project_name: str | Path,
    *,
    force: bool = False,
    framework: str = "blackbase",
    project_template: Path | str | None = None,
) -> Path:
    project_path = Path(project_name).resolve()
    if project_path.exists():
        if force:
            shutil.rmtree(project_path)
        else:
            raise FileExistsError(f"Directory already exists: {project_path}")

    template_path = Path(project_template).resolve() if project_template is not None else None
    if template_path is not None and template_path.is_dir():
        shutil.copytree(template_path, project_path, ignore=_copy_ignore)
        _rename_template_files(project_path)
        _remove_placeholder_cases(project_path)
    else:
        _write_default_project(project_path, framework=framework)
    print(f"Successfully created project '{project_path}'")
    return project_path


def add_case(
    case_name: str,
    case_type: str,
    *,
    framework: str = "blackbase",
    project_root: Path | str | None = None,
    template_by_kind: Mapping[str, Path | str] | None = None,
) -> Path:
    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    if not (root / "project_config.py").exists():
        raise FileNotFoundError("Not inside a project root. Please cd into a project directory.")
    case_kind = _normalize_case_kind(case_type)
    case_path = root / "cases" / str(case_name)
    if case_path.exists():
        raise FileExistsError(f"Case already exists: {case_path}")

    templates = dict(template_by_kind or {})
    template_path = Path(templates.get(case_kind, "")).resolve() if templates.get(case_kind) else None
    if template_path is not None and template_path.is_dir():
        shutil.copytree(template_path, case_path, ignore=_copy_ignore)
        _rename_template_files(case_path)
    else:
        case_path.mkdir(parents=True)

    _normalize_case_scaffold(case_path, case_name=case_name, case_kind=case_kind, framework=framework)
    _try_update_project_config(root, case_name=case_name)
    print(
        f"Successfully added {case_kind} case '{case_name}' "
        f"(kind={case_kind}, framework={framework})"
    )
    return case_path


def add_component(
    component_name: str,
    component_kind: str,
    *,
    case_name: str | None = None,
    slot: str | None = None,
    project_root: Path | str | None = None,
    framework: str = "blackbase",
    template_providers: Mapping[str, ComponentTemplateProvider] | None = None,
) -> Path | None:
    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    case_root = _resolve_case_root(root, case_name=case_name)
    if case_root is None:
        print("Error: Could not resolve target case. Use --case from project root, or run inside a case directory.")
        return None

    kind = str(component_kind or "").strip().lower()
    target_dir_name = _COMPONENT_KIND_TO_DIR.get(kind)
    if target_dir_name is None:
        valid = ", ".join(sorted(_COMPONENT_KIND_TO_DIR))
        print(f"Error: Unknown component kind '{component_kind}'. Use one of: {valid}.")
        return None

    pipeline_slot = _normalize_pipeline_slot(slot) if kind == "pipeline" else None
    safe_name = _normalize_component_name(component_name)
    if not safe_name:
        print("Error: component name is empty after normalization.")
        return None

    # Detect framework from .case marker if not specified
    if framework == "blackbase":
        framework = _detect_framework(case_root)

    target_dir = _resolve_component_dir(case_root, kind=kind, target_dir_name=target_dir_name, slot=pipeline_slot)
    target_dir.mkdir(parents=True, exist_ok=True)
    _ensure_package_tree(case_root, target_dir)
    target_file = target_dir / f"{safe_name}.py"
    if target_file.exists():
        print(f"Error: Component file already exists: {target_file}")
        return None

    target_file.write_text(
        _component_template(
            safe_name,
            kind,
            slot=pipeline_slot,
            framework=framework,
            template_providers=template_providers,
        ),
        encoding="utf-8",
    )
    slot_suffix = f" (slot={pipeline_slot})" if pipeline_slot is not None else ""
    print(
        f"Successfully added {kind} component '{safe_name}' "
        f"to case '{case_root.name}'{slot_suffix} at {target_file}"
    )
    return target_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage standard Project/Case scaffolds.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    p_new = subparsers.add_parser("new", help="Create a new project.")
    p_new.add_argument("project_name")
    p_new.add_argument("--force", action="store_true")
    p_add = subparsers.add_parser("add-case", help="Add a case to the current project.")
    p_add.add_argument("case_name")
    p_add.add_argument("--type", choices=("solver", "trainer"), required=True)
    p_component = subparsers.add_parser("add-component", help="Add a component file to a case.")
    p_component.add_argument("--case", dest="case_name", default=None)
    p_component.add_argument("--kind", choices=tuple(sorted(_COMPONENT_KIND_TO_DIR)), required=True)
    p_component.add_argument("--name", required=True)
    p_component.add_argument("--slot", default=None)
    p_component.add_argument("--framework", default="blackbase")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "new":
        create_project(args.project_name, force=bool(args.force))
        return 0
    if args.command == "add-case":
        add_case(args.case_name, args.type)
        return 0
    if args.command == "add-component":
        return 0 if add_component(args.name, args.kind, case_name=args.case_name, slot=args.slot, framework=args.framework) else 1
    return 2


def _write_default_project(project_path: Path, *, framework: str) -> None:
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / "cases").mkdir()
    (project_path / "cases" / "__init__.py").write_text("", encoding="utf-8")
    (project_path / "README.md").write_text(
        f"# {project_path.name}\n\n"
        "这是使用 blackbase 共享 substrate 的标准 Project / Case / Scaffold 项目。\n\n"
        "- `python run_project.py --check --build-check`：只检查装配与资源授权。\n"
        "- `python run_project.py`：执行并写入 `.blackbase/runs/<run-id>/manifest.json`。\n"
        "- `python run_project.py --resume-from <run-id>`：从兼容的运行记录恢复。\n"
        "- Stage 可声明 `policy=serial|parallel|external`；并行/外部 Case 必须使用 `mode=build`。\n"
        "- 外部 worker：`python -m blackbase.project.external_worker --project-root . "
        "--transport .blackbase/external_tasks.sqlite`。\n",
        encoding="utf-8",
    )
    (project_path / "project_config.py").write_text(_default_project_config(project_path.name), encoding="utf-8")
    (project_path / "run_project.py").write_text(_default_run_project(framework), encoding="utf-8")


def _normalize_case_scaffold(case_path: Path, *, case_name: str, case_kind: str, framework: str) -> None:
    case_path.mkdir(parents=True, exist_ok=True)
    (case_path / "__init__.py").touch()
    for dirname in _CASE_DIRS:
        directory = case_path / dirname
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").touch()
    _write_case_marker(case_path, case_name=case_name, kind=case_kind, framework=framework)
    _ensure_case_readme(
        case_path,
        case_name=case_name,
        case_kind=case_kind,
        framework=framework,
    )
    _ensure_pipeline_entry(case_path, case_kind=case_kind)
    _ensure_primary_entries(case_path, case_kind=case_kind, framework=framework)
    if not (case_path / "config.py").is_file():
        (case_path / "config.py").write_text(_case_config_template(), encoding="utf-8")


def _ensure_case_readme(
    case_path: Path,
    *,
    case_name: str,
    case_kind: str,
    framework: str,
) -> None:
    """Create the one canonical documentation entry for a Case."""

    readme = case_path / "README.md"
    if readme.is_file():
        return
    readme.write_text(
        f"# {case_name}\n\n"
        f"- 语义类型：`{case_kind}`\n"
        f"- 语义框架：`{framework}`\n"
        "- 规范装配入口：`build_solver.py`\n"
        "- 独立运行入口：`run_solver.py`\n\n"
        "组件边界、资源请求、输入输出与运行方式都应维护在本文件中；"
        "不要再创建 START_HERE、注册指南或复制的契约模板。\n",
        encoding="utf-8",
    )


def _ensure_pipeline_entry(case_path: Path, *, case_kind: str) -> None:
    """Ensure every generated Case has one executable pipeline entry."""

    pipeline_dir = case_path / "pipeline"
    pipeline_module = case_path / "pipeline.py"
    pipeline_main = pipeline_dir / "main.py"
    if not pipeline_main.is_file() and not pipeline_module.is_file():
        pipeline_main.write_text(_pipeline_entry_template(case_kind), encoding="utf-8")

    pipeline_init = pipeline_dir / "__init__.py"
    if not pipeline_init.read_text(encoding="utf-8-sig", errors="replace").strip():
        pipeline_init.write_text(
            '"""Case-level pipeline public surface."""\n\n'
            "from .main import build_pipeline, run_pipeline_slot\n\n"
            '__all__ = ["build_pipeline", "run_pipeline_slot"]\n',
            encoding="utf-8",
        )


def _ensure_primary_entries(case_path: Path, *, case_kind: str, framework: str = "blackbase") -> None:
    """Ensure the one shared Case entry shape for every semantic kind."""

    build_solver = case_path / "build_solver.py"
    if not build_solver.is_file():
        build_solver.write_text(
            _build_entry_template(case_kind, framework=framework),
            encoding="utf-8",
        )
    run_solver = case_path / "run_solver.py"
    if not run_solver.is_file():
        run_solver.write_text(_run_entry_template(case_kind, framework=framework), encoding="utf-8")
    _ensure_alias_entries(case_path)


def _ensure_alias_entries(case_path: Path) -> None:
    """Create the required trainer-named thin aliases without a second implementation."""

    alias_build = case_path / "build_trainer.py"
    if not alias_build.is_file():
        alias_build.write_text(
            "from .build_solver import build_solver as build_trainer\n",
            encoding="utf-8",
        )
    alias_run = case_path / "run_trainer.py"
    if not alias_run.is_file():
        alias_run.write_text(
            'from .run_solver import main\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n',
            encoding="utf-8",
        )


def _rename_template_files(root: Path) -> None:
    for item in root.glob("**/*.template"):
        item.rename(item.with_suffix(""))


def _copy_ignore(directory: str, names: list[str]) -> set[str]:
    del directory
    return {
        name
        for name in names
        if name == "__pycache__" or name.endswith((".pyc", ".pyo"))
    }


def _remove_placeholder_cases(project_path: Path) -> None:
    cases_dir = project_path / "cases"
    if not cases_dir.is_dir():
        return
    for case_dir in sorted(item for item in cases_dir.iterdir() if item.is_dir()):
        if _looks_like_real_case(case_dir):
            continue
        shutil.rmtree(case_dir)


def _looks_like_real_case(case_dir: Path) -> bool:
    markers = {".case", "build_solver.py", "build_trainer.py", "run_solver.py", "run_trainer.py"}
    if any((case_dir / name).is_file() for name in markers):
        return True
    for item in case_dir.rglob("*"):
        if item.is_dir():
            continue
        if item.name == "__init__.py" or item.suffix in {".pyc", ".pyo"}:
            continue
        return True
    return False


def _write_case_marker(case_path: Path, *, case_name: str, kind: str, framework: str) -> None:
    (case_path / ".case").write_text(
        f"name = {case_name}\nkind = {kind}\nframework = {framework}\n",
        encoding="utf-8",
    )


def _resolve_case_root(root: Path, *, case_name: str | None) -> Path | None:
    if case_name:
        candidate = root / "cases" / str(case_name)
        if candidate.is_dir():
            return candidate
        print(f"Error: Case '{case_name}' not found under {root / 'cases'}")
        return None
    if (root / ".case").is_file() and (root / "__init__.py").is_file():
        return root
    if (root / "project_config.py").is_file() and (root / "cases").is_dir():
        case_dirs = [
            item
            for item in sorted((root / "cases").iterdir())
            if item.is_dir() and item.name != "__pycache__"
        ]
        if len(case_dirs) == 1:
            return case_dirs[0]
    return None


def _detect_framework(case_root: Path) -> str:
    """Detect framework from .case marker file."""
    marker = case_root / ".case"
    if marker.is_file():
        for line in marker.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            if line.strip().startswith("framework"):
                _, _, value = line.partition("=")
                fw = value.strip()
                if fw:
                    return fw
    return "blackbase"


def _normalize_case_kind(kind: str | None) -> str:
    value = str(kind or "").strip().lower()
    if value not in {"solver", "trainer"}:
        raise ValueError("case type must be 'solver' or 'trainer'")
    return value


def _normalize_component_name(name: str) -> str:
    text = str(name or "").strip().replace("-", "_").replace(" ", "_")
    return "".join(ch for ch in text if ch.isalnum() or ch == "_").lower()


def _normalize_pipeline_slot(slot: str | None) -> str:
    text = str(slot or "").strip().replace("-", "_").replace(" ", "_").lower()
    cleaned = "".join(ch for ch in text if ch.isalnum() or ch == "_")
    return cleaned or _PIPELINE_DEFAULT_SLOT


def _resolve_component_dir(case_root: Path, *, kind: str, target_dir_name: str, slot: str | None) -> Path:
    base = case_root / target_dir_name
    if kind != "pipeline":
        return base
    slot_key = _normalize_pipeline_slot(slot)
    rel = _PIPELINE_SLOT_TO_SUBDIR.get(slot_key)
    if rel is None:
        rel = f"operators/{slot_key}"
    return base / rel


def _ensure_package_tree(case_root: Path, target_dir: Path) -> None:
    try:
        rel = target_dir.resolve().relative_to(case_root.resolve())
    except ValueError:
        return
    current = case_root
    for part in rel.parts:
        current = current / part
        current.mkdir(parents=True, exist_ok=True)
        init_file = current / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")


def _component_template(
    component_name: str,
    component_kind: str,
    *,
    slot: str | None = None,
    framework: str = "blackbase",
    template_providers: Mapping[str, ComponentTemplateProvider] | None = None,
) -> str:
    if component_kind == "pipeline":
        return _pipeline_component_template(component_name, slot=slot)
    provider = dict(template_providers or {}).get(str(framework or "blackbase"))
    if provider is not None:
        return provider(component_name, component_kind, slot)
    if str(framework or "blackbase") != "blackbase":
        raise ValueError(
            f"framework {framework!r} must provide its semantic component template provider"
        )
    class_name = "".join(part.capitalize() for part in component_name.split("_") if part) or "Component"
    providers = {
        "problem": _blackbase_problem_template,
        "adapter": _blackbase_adapter_template,
        "bias": _blackbase_bias_template,
        "plugin": _blackbase_plugin_template,
    }
    return providers[component_kind](class_name, component_name, component_kind)


def _blackbase_problem_template(class_name: str, component_name: str, kind: str) -> str:
    return (
        f'"""Auto-generated {kind} component: {component_name}."""\n\n'
        f"from blackbase.abc import ProblemBase\n\n\n"
        f"class {class_name}(ProblemBase):\n"
        f'    """TODO: implement evaluation logic."""\n\n'
        f"    def evaluate(self, candidate, context=None):\n"
        f'        raise NotImplementedError("TODO: implement evaluate")\n'
    )


def _blackbase_adapter_template(class_name: str, component_name: str, kind: str) -> str:
    return (
        f'"""Auto-generated {kind} component: {component_name}."""\n\n'
        f"from blackbase.abc import AdapterBase\n\n\n"
        f"class {class_name}(AdapterBase):\n"
        f'    """TODO: implement propose/update logic."""\n\n'
        f"    def propose(self, control, context):\n"
        f'        raise NotImplementedError("TODO: implement propose")\n\n'
        f"    def update(self, control, candidates, feedback, context):\n"
        f'        raise NotImplementedError("TODO: implement update")\n'
    )


def _blackbase_bias_template(class_name: str, component_name: str, kind: str) -> str:
    return (
        f'"""Auto-generated {kind} component: {component_name}."""\n\n'
        f"from blackbase.abc import BiasBase\n\n\n"
        f"class {class_name}(BiasBase):\n"
        f'    """TODO: implement preference logic."""\n\n'
        f"    def project_context(self, context):\n"
        f"        return dict(context)\n"
    )


def _blackbase_plugin_template(class_name: str, component_name: str, kind: str) -> str:
    return (
        f'"""Auto-generated {kind} component: {component_name}."""\n\n'
        f"from blackbase.plugin import PluginBase\n\n\n"
        f"class {class_name}(PluginBase):\n"
        f'    """TODO: implement plugin lifecycle hooks."""\n\n'
        f"    pass\n"
    )


def _pipeline_component_template(component_name: str, *, slot: str | None) -> str:
    slot_name = _normalize_pipeline_slot(slot)
    if slot_name in {"entry", "main"}:
        return (
            '"""Single pipeline entry composed from slot operators."""\n\n'
            "from typing import Any, Mapping\n\n"
            "from blackbase.kernel import PipelineSpec, build_pipeline_kernel\n\n"
            "def build_pipeline(*, resource_context: Mapping[str, Any] | None = None, component_overrides: Mapping[str, Any] | None = None):\n"
            "    del resource_context\n"
            "    overrides = dict(component_overrides or {})\n"
            "    registry = dict(overrides.get(\"pipeline_operators\", {}) or {})\n"
            "    spec = PipelineSpec.from_value(overrides.get(\"pipeline_spec\", {\"key\": \"default\", \"slots\": ()}))\n"
            "    kernel = build_pipeline_kernel(spec, operator_registry=registry)\n"
            "    return kernel.representation_pipeline\n"
        )

    method_by_slot = {
        "init": "initialize",
        "mutate": "mutate",
        "repair": "repair",
        "encode": "encode",
        "decode": "decode",
        "transform": "transform",
        "codec": "encode",
        "head": "decode",
        "representation": "transform",
        "custom": "transform",
    }
    method_name = method_by_slot.get(slot_name, "transform")
    class_name = "".join(part.capitalize() for part in component_name.split("_") if part) or "PipelineOperator"
    return (
        f'"""Pipeline operator: slot={slot_name}, name={component_name}."""\n\n'
        f"class {class_name}:\n"
        "    \"\"\"TODO: implement pipeline operator logic.\"\"\"\n\n"
        f"    def {method_name}(self, value, context=None):\n"
        "        del context\n"
        "        return value\n"
    )


def _pipeline_entry_template(case_kind: str) -> str:
    pipeline_key = "trainer_default" if str(case_kind) == "trainer" else "solver_default"
    return f'''"""Canonical Case-level pipeline entry.

Compose fine-grained operators here and keep ``build_solver.py`` focused on
assembling the Case lifecycle.
"""

from typing import Any, Mapping

from blackbase.kernel import PipelineSpec, build_pipeline_kernel


def _build_kernel(*, component_overrides: Mapping[str, Any] | None = None):
    overrides = dict(component_overrides or {{}})
    registry = dict(overrides.get("pipeline_operators", {{}}) or {{}})
    spec = PipelineSpec.from_value(
        overrides.get("pipeline_spec", {{"key": "{pipeline_key}", "slots": ()}})
    )
    return build_pipeline_kernel(spec, operator_registry=registry)


def build_pipeline(
    *,
    resource_context: Mapping[str, Any] | None = None,
    component_overrides: Mapping[str, Any] | None = None,
):
    """Build the Case representation pipeline from its declared slots."""

    del resource_context
    return _build_kernel(component_overrides=component_overrides).representation_pipeline


def run_pipeline_slot(
    slot: str,
    value,
    *,
    component_overrides: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
):
    """Run one declared slot through the same canonical kernel."""

    kernel = _build_kernel(component_overrides=component_overrides)
    return kernel.run_slot(slot, value, dict(context or {{}}))
'''


def _default_project_config(project_name: str) -> str:
    return f'''"""Project-level orchestration configuration."""

PROJECT_NAME = "{project_name}"

L0 = {{
    "namespace": "{project_name}",
    "offer": {{"threads": 4, "gpus": 0, "backend": "local", "device_tokens": []}},
    "policy": {{"mode": "strict", "gpu_sharing": "exclusive", "cpu_oversubscribe": False}},
    "default_request": {{"threads": 1, "gpus": 0, "backend": "local", "device": "cpu"}},
    "compute_backend": "auto",
    "execution_backend": "local",
    "lease_backend": "sqlite",
    "lease_path": ".blackbase/l0_leases.sqlite",
    # Multi-host: use lease_backend="redis" and set lease_redis_url_env.
    # Redis credentials are resolved from the worker environment, not ResourceContext.
    "lease_ttl_seconds": 30,
    "lease_heartbeat_seconds": 10,
    # Optional run-scoped hard limits shared by every Case/process.
    # Example: "budgets": {{"evaluations": 10000}},
    "budgets": {{}},
    # Artifact refs are minted only after an atomic write under this Project root.
    "artifacts": {{"path": ".blackbase/artifacts", "allow_unsafe_serializers": False}},
    # Change mode to cooperative_then_terminate for isolated hard-SLA Cases.
    "termination": {{
        "mode": "cooperative",
        "grace_seconds": 5.0,
        "kill_grace_seconds": 1.0,
        "poll_interval_seconds": 0.05,
    }},
}}

STAGES = [
    {{
        "name": "stage_1",
        "policy": "serial",
        "failure_policy": "fail_fast",
        "cases": [],
        "resource_requests": {{}},
    }},
]

GROUPS = {{"default": {{"stages": ["stage_1"]}}}}
'''


def _try_update_project_config(project_root: Path, *, case_name: str) -> None:
    config_path = project_root / "project_config.py"
    if not config_path.is_file():
        return
    text = config_path.read_text(encoding="utf-8-sig", errors="replace")
    if f'"{case_name}"' in text or f"'{case_name}'" in text:
        return
    updated = text
    for token in ('"cases": [],', '"cases": [],', "'cases': [],", "'cases': [],"):
        if token in updated:
            quote = "'" if token.startswith("'") else '"'
            updated = updated.replace(token, f"{quote}cases{quote}: [{quote}{case_name}{quote}],", 1)
            break
    for token in ('"resource_requests": {},', "'resource_requests': {},"):
        if token in updated:
            quote = "'" if token.startswith("'") else '"'
            value = (
                f"{quote}resource_requests{quote}: "
                f"{{{quote}{case_name}{quote}: "
                f"{{{quote}threads{quote}: 1, {quote}gpus{quote}: 0, {quote}backend{quote}: {quote}local{quote}}}}},"
            )
            updated = updated.replace(token, value, 1)
            break
    if updated != text:
        config_path.write_text(updated, encoding="utf-8")


def _default_run_project(framework: str) -> str:
    return f'''"""Run this project through the shared blackbase Project substrate."""

from pathlib import Path

from blackbase.project.project_runner import main


if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).resolve().parent, framework="{framework}"))
'''


def _build_entry_template(kind: str, framework: str = "blackbase") -> str:
    func = "build_solver"
    imports = (
        "from blackbase.abc import AdapterBase, RepresentationBase, ProblemBase, BiasBase\n"
        "from blackbase.plugin import PluginBase\n"
    )
    return f'''"""Canonical {kind} case assembly entry ({framework})."""

{imports}


def {func}(*, resource_context=None, component_overrides=None):
    """Assemble and return one {kind} case.

    Components inherit from blackbase ABC base classes.
    Override propose/update (adapter), init/decode (representation),
    evaluate (problem), project_context (bias) as needed.
    """

    del resource_context, component_overrides
    raise NotImplementedError("TODO: implement {func} for this case")
'''


def _run_entry_template(kind: str, *, framework: str = "blackbase") -> str:
    return f'''"""CLI entry point for this {kind} case."""

from blackbase.project import run_standard_case_cli


def main(argv=None):
    return run_standard_case_cli(__file__, framework="{framework}", argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _case_config_template() -> str:
    return '''"""Case-level component registry aggregation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CaseConfig:
    problem_key: str = "default"
    pipeline_key: str = "default"
    adapter_key: str = "default"
    resource_request: dict | None = None


def get_case_config() -> CaseConfig:
    return CaseConfig()
'''


if __name__ == "__main__":
    raise SystemExit(main())
