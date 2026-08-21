"""Common Project/Case/Scaffold doctor rules shared by frameworks."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .runtime import path_declares_check_argument


@dataclass(frozen=True)
class DoctorDiagnostic:
    level: str
    code: str
    message: str
    path: str | None = None

    def as_dict(self) -> dict[str, str]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "path": self.path or "",
        }


@dataclass(frozen=True)
class DoctorReport:
    project_root: Path
    diagnostics: Sequence[DoctorDiagnostic]

    @property
    def error_count(self) -> int:
        return sum(1 for item in self.diagnostics if item.level == "error")

    @property
    def warn_count(self) -> int:
        return sum(1 for item in self.diagnostics if item.level == "warn")

    @property
    def info_count(self) -> int:
        return sum(1 for item in self.diagnostics if item.level == "info")

    @property
    def ok(self) -> bool:
        return self.error_count == 0


def run_common_project_doctor(path: str | Path | None = None, *, strict: bool = False) -> DoctorReport:
    root = Path(path or Path.cwd()).resolve()
    diags: list[DoctorDiagnostic] = []
    projects = list(_iter_project_roots(root))
    if _is_project_root(root) and root not in projects:
        projects.insert(0, root)

    for project_root in projects:
        _check_project_root(project_root, diags, strict=strict)

    if projects:
        diags.append(
            DoctorDiagnostic(
                "info",
                "blackbase-project-scope",
                f"Validated {len(projects)} Project wrapper(s) with shared substrate rules.",
                str(root),
            )
        )
    else:
        diags.append(
            DoctorDiagnostic(
                "info",
                "blackbase-project-scope",
                "No Project wrapper found; common substrate checks only apply to Project roots/examples.",
                str(root),
            )
        )
    return DoctorReport(project_root=root, diagnostics=tuple(diags))


def format_doctor_report(report: DoctorReport) -> str:
    lines = [
        f"blackbase project doctor: {'ok' if report.ok else 'issues'}",
        f"root: {report.project_root}",
        f"summary: errors={report.error_count} warnings={report.warn_count} infos={report.info_count}",
    ]
    for item in report.diagnostics:
        suffix = "" if not item.path else f" ({item.path})"
        lines.append(f"[{item.level}] {item.code}: {item.message}{suffix}")
    return "\n".join(lines)


def iter_diagnostics_by_level(diagnostics: Iterable[DoctorDiagnostic], level: str) -> list[DoctorDiagnostic]:
    target = str(level)
    return [item for item in diagnostics if item.level == target]


def _iter_project_roots(root: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    candidates: list[Path] = []
    if _is_project_root(root):
        candidates.append(root)
    examples_cases = root / "examples" / "cases"
    if examples_cases.is_dir():
        candidates.extend(path for path in sorted(examples_cases.iterdir()) if _is_project_root(path))
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            yield resolved


def _is_project_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "project_config.py").is_file()
        and (path / "run_project.py").is_file()
        and (path / "cases").is_dir()
    )


def _check_project_root(project_root: Path, diags: list[DoctorDiagnostic], *, strict: bool) -> None:
    for filename in ("README.md", "project_config.py", "run_project.py"):
        _require_file(project_root / filename, diags)
    cases_dir = project_root / "cases"
    _require_dir(cases_dir, diags)
    _require_file(cases_dir / "__init__.py", diags)
    _check_project_config(project_root / "project_config.py", diags)

    case_count = 0
    if cases_dir.is_dir():
        for case_root in sorted(
            item
            for item in cases_dir.iterdir()
            if item.is_dir()
            and item.name != "__pycache__"
            and (
                (item / ".case").is_file()
                or (item / "build_solver.py").is_file()
                or (item / "run_solver.py").is_file()
            )
        ):
            case_count += 1
            _check_case_root(case_root, diags, strict=strict)
    diags.append(
        DoctorDiagnostic(
            "info",
            "blackbase-project-root",
            f"Checked Project root with {case_count} case(s).",
            str(project_root),
        )
    )


def _check_project_config(path: Path, diags: list[DoctorDiagnostic]) -> None:
    if not path.is_file():
        return
    source = path.read_text(encoding="utf-8-sig", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        diags.append(
            DoctorDiagnostic(
                "error",
                "project-config-syntax",
                f"project_config.py has syntax error: {exc}",
                str(path),
            )
        )
        return
    stages_value: Any = None
    l0_value: Any = None
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        if any(isinstance(target, ast.Name) and target.id == "L0" for target in targets):
            try:
                l0_value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                l0_value = None
        if any(isinstance(target, ast.Name) and target.id == "STAGES" for target in targets):
            try:
                stages_value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                return
    if isinstance(l0_value, Mapping):
        _check_l0_config(dict(l0_value), path=path, diags=diags)
    if stages_value is None:
        return
    if not isinstance(stages_value, (list, tuple)):
        diags.append(
            DoctorDiagnostic(
                "error",
                "project-stages-invalid",
                "STAGES must be a list or tuple of mappings.",
                str(path),
            )
        )
        return
    stage_names: list[str] = []
    for index, raw_stage in enumerate(stages_value):
        if not isinstance(raw_stage, Mapping):
            diags.append(
                DoctorDiagnostic(
                    "error",
                    "project-stage-invalid",
                    f"STAGES[{index}] must be a mapping.",
                    str(path),
                )
            )
            continue
        stage = dict(raw_stage)
        stage_name = str(stage.get("name", f"stage_{index}"))
        stage_names.append(stage_name)
        cases = stage.get("cases", ()) or ()
        if isinstance(cases, str) or not isinstance(cases, (list, tuple)):
            diags.append(
                DoctorDiagnostic(
                    "error",
                    "project-stage-cases-invalid",
                    f"Stage '{stage_name}' cases must be a list or tuple.",
                    str(path),
                )
            )
            continue
        case_names = tuple(str(item) for item in cases)
        for case_name in case_names:
            case_root = path.parent / "cases" / case_name
            if not case_root.is_dir():
                diags.append(
                    DoctorDiagnostic(
                        "error",
                        "project-stage-case-missing",
                        f"Stage '{stage_name}' references missing Case '{case_name}'.",
                        str(case_root),
                    )
                )
            elif not (case_root / ".case").is_file():
                diags.append(
                    DoctorDiagnostic(
                        "error",
                        "project-stage-case-marker-missing",
                        f"Stage '{stage_name}' Case '{case_name}' is missing .case.",
                        str(case_root / ".case"),
                    )
                )
        if len(case_names) != len(set(case_names)):
            diags.append(
                DoctorDiagnostic(
                    "error",
                    "project-stage-duplicate-case",
                    f"Stage '{stage_name}' contains duplicate Case names.",
                    str(path),
                )
            )
        policy = str(stage.get("policy", "serial") or "serial").lower()
        if policy not in {
            "serial",
            "sequential",
            "run_all_serial",
            "parallel",
            "run_all_in_parallel",
            "external",
            "external_workers",
        }:
            diags.append(
                DoctorDiagnostic(
                    "error",
                    "project-stage-policy-invalid",
                    f"Stage '{stage_name}' has unsupported policy '{policy}'.",
                    str(path),
                )
            )
        if policy in {"external", "external_workers"}:
            _check_external_stage(stage, stage_name=stage_name, path=path, diags=diags)
        _check_stage_timeouts(stage, stage_name=stage_name, path=path, diags=diags)
        _check_termination_config(
            stage.get("termination", {}),
            label=f"Stage '{stage_name}' termination",
            path=path,
            diags=diags,
        )
        case_terminations = stage.get("case_termination", {}) or {}
        if not isinstance(case_terminations, Mapping):
            diags.append(
                DoctorDiagnostic(
                    "error",
                    "project-stage-case-termination-invalid",
                    f"Stage '{stage_name}' case_termination must be a mapping.",
                    str(path),
                )
            )
        else:
            for case_name, raw_policy in case_terminations.items():
                _check_termination_config(
                    raw_policy,
                    label=f"Case '{case_name}' termination",
                    path=path,
                    diags=diags,
                )
    if len(stage_names) != len(set(stage_names)):
        diags.append(
            DoctorDiagnostic(
                "error",
                "project-stage-name-duplicate",
                "Project Stage names must be unique for manifest recovery.",
                str(path),
            )
        )


def _check_stage_timeouts(
    stage: Mapping[str, Any],
    *,
    stage_name: str,
    path: Path,
    diags: list[DoctorDiagnostic],
) -> None:
    try:
        timeout = float(stage.get("timeout_seconds", 0.0) or 0.0)
    except (TypeError, ValueError):
        timeout = -1.0
    if timeout < 0:
        diags.append(
            DoctorDiagnostic(
                "error",
                "project-stage-timeout-invalid",
                f"Stage '{stage_name}' timeout_seconds must be non-negative.",
                str(path),
            )
        )
    case_timeouts = stage.get("case_timeout_seconds", {}) or {}
    if not isinstance(case_timeouts, Mapping):
        diags.append(
            DoctorDiagnostic(
                "error",
                "project-stage-case-timeouts-invalid",
                f"Stage '{stage_name}' case_timeout_seconds must be a mapping.",
                str(path),
            )
        )
        return
    for case_name, raw_timeout in case_timeouts.items():
        try:
            value = float(raw_timeout)
        except (TypeError, ValueError):
            value = -1.0
        if value < 0:
            diags.append(
                DoctorDiagnostic(
                    "error",
                    "project-case-timeout-invalid",
                    f"Case '{case_name}' in Stage '{stage_name}' requires a "
                    "non-negative timeout.",
                    str(path),
                )
            )


def _check_l0_config(
    config: Mapping[str, Any],
    *,
    path: Path,
    diags: list[DoctorDiagnostic],
) -> None:
    _check_termination_config(
        config.get("termination", {}),
        label="L0.termination",
        path=path,
        diags=diags,
    )
    artifacts = config.get("artifacts", config.get("artifact_store", {})) or {}
    if not isinstance(artifacts, Mapping):
        diags.append(
            DoctorDiagnostic(
                "error",
                "project-l0-artifacts-invalid",
                "L0.artifacts must be a mapping.",
                str(path),
            )
        )
    else:
        artifact_path = str(artifacts.get("path", ".blackbase/artifacts") or "").strip()
        if not artifact_path:
            diags.append(
                DoctorDiagnostic(
                    "error",
                    "project-l0-artifact-path-missing",
                    "L0.artifacts.path must be non-empty.",
                    str(path),
                )
            )
        else:
            root = path.parent.resolve()
            candidate = Path(artifact_path)
            if not candidate.is_absolute():
                candidate = root / candidate
            try:
                candidate.resolve().relative_to(root)
            except ValueError:
                diags.append(
                    DoctorDiagnostic(
                        "error",
                        "project-l0-artifact-path-outside-root",
                        "L0.artifacts.path must stay within the Project root.",
                        str(path),
                    )
                )
    backend = str(config.get("lease_backend", "memory") or "memory").lower()
    if backend not in {"memory", "sqlite", "redis"}:
        diags.append(
            DoctorDiagnostic(
                "error",
                "project-l0-lease-backend-invalid",
                "L0.lease_backend must be 'memory', 'sqlite', or 'redis'.",
                str(path),
            )
        )
    if backend == "sqlite" and not str(config.get("lease_path", "")).strip():
        diags.append(
            DoctorDiagnostic(
                "error",
                "project-l0-lease-path-missing",
                "SQLite L0 lease authority requires L0.lease_path.",
                str(path),
            )
        )
    if backend == "redis":
        redis_url = str(config.get("lease_redis_url", "") or "").strip()
        redis_url_env = str(
            config.get("lease_redis_url_env", "BLACKBASE_REDIS_URL") or ""
        ).strip()
        if not redis_url and not redis_url_env:
            diags.append(
                DoctorDiagnostic(
                    "error",
                    "project-l0-lease-redis-connection-missing",
                    "Redis L0 lease authority requires L0.lease_redis_url or "
                    "L0.lease_redis_url_env.",
                    str(path),
                )
            )
    try:
        ttl = float(config.get("lease_ttl_seconds", 30.0))
        heartbeat = float(config.get("lease_heartbeat_seconds", 10.0))
    except (TypeError, ValueError):
        ttl = heartbeat = -1.0
    if ttl <= 0:
        diags.append(
            DoctorDiagnostic(
                "error",
                "project-l0-lease-ttl-invalid",
                "L0.lease_ttl_seconds must be positive.",
                str(path),
            )
        )
    if heartbeat <= 0 or heartbeat >= ttl:
        diags.append(
            DoctorDiagnostic(
                "error",
                "project-l0-lease-heartbeat-invalid",
                "L0.lease_heartbeat_seconds must be positive and smaller than lease_ttl_seconds.",
                str(path),
            )
        )
    raw_budgets = config.get("budgets", {}) or {}
    if not isinstance(raw_budgets, Mapping):
        diags.append(
            DoctorDiagnostic(
                "error",
                "project-l0-budgets-invalid",
                "L0.budgets must be a mapping of budget name to non-negative integer limit.",
                str(path),
            )
        )
    else:
        valid_budgets = {}
        for name, raw_limit in raw_budgets.items():
            budget_name = str(name).strip()
            try:
                budget_limit = int(raw_limit)
            except (TypeError, ValueError):
                budget_limit = -1
            if not budget_name or budget_limit < 0:
                diags.append(
                    DoctorDiagnostic(
                        "error",
                        "project-l0-budget-limit-invalid",
                        "Every L0.budgets entry requires a non-empty name and "
                        "a non-negative integer limit.",
                        str(path),
                    )
                )
                continue
            valid_budgets[budget_name] = budget_limit
        if valid_budgets and backend == "memory":
            diags.append(
                DoctorDiagnostic(
                    "error",
                    "project-l0-budget-authority-not-durable",
                    "L0.budgets requires lease_backend='sqlite' or 'redis' so "
                    "all Cases share one fenced authority.",
                    str(path),
                )
            )


def _check_termination_config(
    raw: Any,
    *,
    label: str,
    path: Path,
    diags: list[DoctorDiagnostic],
) -> None:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        diags.append(
            DoctorDiagnostic(
                "error",
                "project-termination-config-invalid",
                f"{label} must be a mapping.",
                str(path),
            )
        )
        return
    config = dict(raw)
    mode = str(config.get("mode", "cooperative") or "cooperative").strip().lower()
    if mode not in {"cooperative", "cooperative_then_terminate"}:
        diags.append(
            DoctorDiagnostic(
                "error",
                "project-termination-mode-invalid",
                f"{label}.mode must be cooperative or cooperative_then_terminate.",
                str(path),
            )
        )
    for key, default, positive in (
        ("grace_seconds", 5.0, False),
        ("kill_grace_seconds", 1.0, False),
        ("poll_interval_seconds", 0.05, True),
    ):
        try:
            value = float(config.get(key, default))
        except (TypeError, ValueError):
            value = -1.0
        invalid = value <= 0 if positive else value < 0
        if invalid:
            diags.append(
                DoctorDiagnostic(
                    "error",
                    "project-termination-timing-invalid",
                    f"{label}.{key} has an invalid value.",
                    str(path),
                )
            )


def _check_external_stage(
    stage: Mapping[str, Any],
    *,
    stage_name: str,
    path: Path,
    diags: list[DoctorDiagnostic],
) -> None:
    external = stage.get("external")
    if not isinstance(external, Mapping):
        diags.append(
            DoctorDiagnostic(
                "error",
                "project-external-config-missing",
                f"External Stage '{stage_name}' must declare an external mapping.",
                str(path),
            )
        )
        return
    config = dict(external)
    backend = str(config.get("backend", "sqlite") or "sqlite").lower()
    if backend not in {"sqlite", "redis"}:
        diags.append(
            DoctorDiagnostic(
                "error",
                "project-external-backend-unsupported",
                f"External Stage '{stage_name}' supports backend='sqlite' or 'redis'.",
                str(path),
            )
        )
    if backend != "redis" and not str(config.get("transport_path", "")).strip():
        diags.append(
            DoctorDiagnostic(
                "error",
                "project-external-transport-path-missing",
                f"External Stage '{stage_name}' must declare external.transport_path.",
                str(path),
            )
        )
    if backend == "redis" and not str(config.get("redis_url", "")).strip():
        diags.append(
            DoctorDiagnostic(
                "error",
                "project-external-redis-url-missing",
                f"External Stage '{stage_name}' must declare external.redis_url.",
                str(path),
            )
        )
    if backend == "redis" and not str(config.get("namespace", "")).strip():
        diags.append(
            DoctorDiagnostic(
                "error",
                "project-external-redis-namespace-missing",
                f"External Stage '{stage_name}' must declare external.namespace.",
                str(path),
            )
        )
    stage_mode = str(stage.get("mode", "build") or "build")
    case_modes = stage.get("case_modes", {}) or {}
    if not isinstance(case_modes, Mapping):
        diags.append(
            DoctorDiagnostic(
                "error",
                "project-external-case-modes-invalid",
                f"External Stage '{stage_name}' case_modes must be a mapping.",
                str(path),
            )
        )
        return
    invalid = {
        str(case_name): str(case_modes.get(case_name, stage_mode) or "build")
        for case_name in tuple(stage.get("cases", ()) or ())
        if str(case_modes.get(case_name, stage_mode) or "build") != "build"
    }
    if invalid:
        diags.append(
            DoctorDiagnostic(
                "error",
                "project-external-cli-mode",
                f"External Stage '{stage_name}' requires mode='build': {invalid}.",
                str(path),
            )
        )


def _check_case_root(case_root: Path, diags: list[DoctorDiagnostic], *, strict: bool) -> None:
    _require_file(case_root / "__init__.py", diags)
    marker = case_root / ".case"
    _require_file(marker, diags)
    kind = _read_case_kind(marker)
    if _case_marker_declares_key(marker, "compatibility"):
        diags.append(
            DoctorDiagnostic(
                "error",
                "case-compatibility-marker-forbidden",
                "Case markers may not opt out of the canonical build contract; "
                "remove compatibility and migrate the Case.",
                str(marker),
            )
        )
    if kind not in {"solver", "trainer"}:
        diags.append(
            DoctorDiagnostic(
                "error",
                "case-kind-invalid",
                ".case must declare kind = solver or kind = trainer.",
                str(marker),
            )
        )
        kind = "solver"

    build_solver = case_root / "build_solver.py"
    run_solver = case_root / "run_solver.py"
    build_trainer = case_root / "build_trainer.py"
    run_trainer = case_root / "run_trainer.py"
    for entry in (build_solver, run_solver, build_trainer, run_trainer):
        _require_file(entry, diags)

    for dirname in ("problem", "pipeline", "adapter", "bias", "plugins", "evaluation", "runtime"):
        _require_dir(case_root / dirname, diags)
    if (case_root / "capabilities").is_dir():
        diags.append(DoctorDiagnostic("error", "case-capabilities-dir", "Use plugins/ instead of case-level capabilities/.", str(case_root / "capabilities")))
    if (case_root / "representation").is_dir():
        diags.append(DoctorDiagnostic("error", "case-representation-dir", "Representation/codec operators belong under pipeline/.", str(case_root / "representation")))
    if (case_root / "assembly" / "scaffold.json").is_file():
        diags.append(DoctorDiagnostic("error", "case-assembly-scaffold-json", "assembly/scaffold.json is not a formal build entry.", str(case_root / "assembly" / "scaffold.json")))

    _check_build_signature(build_solver, diags=diags, strict=strict)
    _check_run_check_contract(run_solver, diags=diags, strict=strict)
    _check_thin_alias(
        build_trainer,
        source_module="build_solver",
        source_name="build_solver",
        alias_name="build_trainer",
        code="case-build-trainer-not-thin-alias",
        diags=diags,
    )
    _check_thin_alias(
        run_trainer,
        source_module="run_solver",
        source_name="main",
        alias_name="main",
        code="case-run-trainer-not-thin-alias",
        allow_main_guard=True,
        diags=diags,
    )


def _check_build_signature(
    path: Path,
    *,
    diags: list[DoctorDiagnostic],
    strict: bool,
) -> None:
    if not path.is_file():
        return
    func_name = "build_solver"
    source = path.read_text(encoding="utf-8-sig", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        diags.append(DoctorDiagnostic("error", "case-build-syntax", f"{path.name} has syntax error: {exc}", str(path)))
        return
    if any(_mutates_sys_path(node) for node in ast.walk(tree)):
        diags.append(
            DoctorDiagnostic(
                "error" if strict else "warn",
                "case-build-mutates-sys-path",
                "Canonical build_solver.py must use the shared Case import context and "
                "package-relative imports; it may not modify sys.path.",
                str(path),
            )
        )
    if any(_is_main_guard(node) for node in tree.body):
        diags.append(
            DoctorDiagnostic(
                "error" if strict else "warn",
                "case-build-executable-entry-forbidden",
                "Canonical build_solver.py is assembly-only and may not expose an "
                "executable __main__ entry; use run_solver.py for the Case CLI.",
                str(path),
            )
        )
    cli_functions = sorted(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"main", "_build_parser", "build_parser"}
    )
    imports_argparse = any(
        (
            isinstance(node, ast.Import)
            and any(alias.name == "argparse" for alias in node.names)
        )
        or (
            isinstance(node, ast.ImportFrom)
            and node.module == "argparse"
        )
        for node in tree.body
    )
    if cli_functions or imports_argparse:
        details = []
        if cli_functions:
            details.append(f"functions={cli_functions}")
        if imports_argparse:
            details.append("imports=argparse")
        diags.append(
            DoctorDiagnostic(
                "error" if strict else "warn",
                "case-build-cli-surface-forbidden",
                "Canonical build_solver.py is assembly-only; parser and CLI surfaces "
                f"belong in run_solver.py ({', '.join(details)}).",
                str(path),
            )
        )
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            target = node
            break
    if target is None:
        diags.append(DoctorDiagnostic("error", "case-build-missing-function", f"{path.name} must define {func_name}().", str(path)))
        return
    migrated_wrapper = any(
        (
            isinstance(node, ast.Name)
            and node.id in {"MigratedExampleRunner", "runpy"}
        )
        or (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "runpy"
        )
        for node in ast.walk(tree)
    )
    if migrated_wrapper:
        diags.append(
            DoctorDiagnostic(
                "error" if strict else "warn",
                "case-migrated-wrapper-forbidden",
                "A canonical Case may not delegate execution through runpy or a "
                "MigratedExampleRunner; build a real Solver/Trainer instead.",
                str(path),
            )
        )
    source_lower = source.lower()
    if "backward compatible" in source_lower or "compatibility assembly" in source_lower:
        diags.append(
            DoctorDiagnostic(
                "error" if strict else "warn",
                "case-compatibility-source-forbidden",
                "Canonical build_solver.py may not preserve a compatibility assembly; "
                "migrate or remove the old Case.",
                str(path),
            )
        )
    wrapper_classes = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        method_names = {
            item.name for item in node.body if isinstance(item, ast.FunctionDef)
        }
        if not method_names.intersection({"run", "fit"}):
            continue
        base_names = {_ast_qualified_name(base).split(".")[-1] for base in node.bases}
        if any(name.endswith(("Solver", "Trainer")) for name in base_names):
            continue
        if node.name.endswith("Runner") or node.name.endswith("OuterCase"):
            wrapper_classes.append(node.name)
    if wrapper_classes:
        diags.append(
            DoctorDiagnostic(
                "error" if strict else "warn",
                "case-private-control-wrapper-forbidden",
                "Canonical build_solver.py defines a private run/fit wrapper instead of "
                f"a Solver/Trainer control plane: {sorted(wrapper_classes)}.",
                str(path),
            )
        )
    arg_names = [arg.arg for arg in (*target.args.args, *target.args.kwonlyargs)]
    for name in ("resource_context", "component_overrides"):
        if name not in arg_names:
            level = "error" if strict else "warn"
            diags.append(
                DoctorDiagnostic(
                    level,
                    f"case-build-missing-{name.replace('_', '-')}",
                    f"{func_name}() should accept {name} for shared Project/L0 and nested overrides.",
                    str(path),
                )
            )
    for returned in _iter_function_returns(target):
        value = returned.value
        if isinstance(value, ast.Lambda) or (
            isinstance(value, ast.Name)
            and value.id in {"main", "run", "run_case", "fit"}
        ):
            diags.append(
                DoctorDiagnostic(
                    "error",
                    "case-build-returns-entry-callable",
                    "build_solver() must return a built Case/Trainer/Solver, not a CLI or runner function.",
                    str(path),
                )
            )
            break
        if isinstance(value, (ast.Dict, ast.List, ast.Tuple, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            diags.append(
                DoctorDiagnostic(
                    "error",
                    "case-build-returns-collection",
                    "build_solver() must return one built Case/Trainer/Solver; multi-Case orchestration belongs to Project.",
                    str(path),
                )
            )
            break


def _mutates_sys_path(node: ast.AST) -> bool:
    if isinstance(node, ast.Call):
        func = node.func
        return (
            isinstance(func, ast.Attribute)
            and func.attr in {"append", "clear", "extend", "insert", "pop", "remove", "reverse", "sort"}
            and _is_sys_path_expression(func.value)
        )
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.Delete):
            targets = node.targets
        else:
            targets = (node.target,)
        return any(_is_sys_path_target(target) for target in targets)
    return False


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _is_sys_path_expression(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "path"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def _is_sys_path_target(node: ast.AST) -> bool:
    if _is_sys_path_expression(node):
        return True
    return isinstance(node, ast.Subscript) and _is_sys_path_expression(node.value)


def _case_marker_declares_key(path: Path, key: str) -> bool:
    if not path.is_file():
        return False
    expected = str(key or "").strip().lower()
    for raw_line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip().lower() != expected:
            continue
        del value
        return True
    return False


def _ast_qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _ast_qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _iter_function_returns(function: ast.FunctionDef) -> Iterable[ast.Return]:
    pending: list[ast.AST] = list(function.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Return):
            yield node
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        pending.extend(ast.iter_child_nodes(node))


def _check_run_check_contract(path: Path, *, diags: list[DoctorDiagnostic], strict: bool) -> None:
    """Require a discoverable, side-effect-bounded assembly-check CLI contract."""

    if not path.is_file():
        return
    if path_declares_check_argument(path):
        return
    level = "error" if strict else "warn"
    diags.append(
        DoctorDiagnostic(
            level,
            "case-run-missing-check-contract",
            "run_solver.py should expose --check so assembly can be audited without running the Case.",
            str(path),
        )
    )


def _check_thin_alias(
    path: Path,
    *,
    source_module: str,
    source_name: str,
    alias_name: str,
    code: str,
    allow_main_guard: bool = False,
    diags: list[DoctorDiagnostic],
) -> None:
    if not path.is_file():
        return
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
    except SyntaxError as exc:
        diags.append(
            DoctorDiagnostic(
                "error",
                code,
                f"{path.name} must be a thin alias but has syntax error: {exc}",
                str(path),
            )
        )
        return

    body = list(tree.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    valid_import = False
    if body and isinstance(body[0], ast.ImportFrom):
        entry = body[0]
        valid_import = (
            entry.level == 1
            and entry.module == source_module
            and len(entry.names) == 1
            and entry.names[0].name == source_name
            and (entry.names[0].asname or entry.names[0].name) == alias_name
        )
    valid = valid_import and (
        len(body) == 1
        or (
            allow_main_guard
            and len(body) == 2
            and _is_standard_main_guard(body[1], callable_name=alias_name)
        )
    )
    if not valid:
        diags.append(
            DoctorDiagnostic(
                "error",
                code,
                (
                    f"{path.name} must only re-export "
                    f".{source_module}.{source_name} as {alias_name}."
                ),
                str(path),
            )
        )


def _is_standard_main_guard(node: ast.stmt, *, callable_name: str) -> bool:
    if not isinstance(node, ast.If) or node.orelse or len(node.body) != 1:
        return False
    test = node.test
    if not (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    ):
        return False
    statement = node.body[0]
    if not isinstance(statement, ast.Raise) or not isinstance(statement.exc, ast.Call):
        return False
    exit_call = statement.exc
    if not isinstance(exit_call.func, ast.Name) or exit_call.func.id != "SystemExit":
        return False
    if len(exit_call.args) != 1 or not isinstance(exit_call.args[0], ast.Call):
        return False
    target_call = exit_call.args[0]
    return (
        isinstance(target_call.func, ast.Name)
        and target_call.func.id == callable_name
        and not target_call.args
        and not target_call.keywords
    )


def _require_file(path: Path, diags: list[DoctorDiagnostic]) -> None:
    if not path.is_file():
        diags.append(DoctorDiagnostic("error", "missing-file", f"Required file is missing: {path.name}", str(path)))


def _require_dir(path: Path, diags: list[DoctorDiagnostic]) -> None:
    if not path.is_dir():
        diags.append(DoctorDiagnostic("error", "missing-dir", f"Required directory is missing: {path.name}", str(path)))


def _read_case_kind(marker: Path) -> str:
    if not marker.is_file():
        return ""
    text = marker.read_text(encoding="utf-8-sig", errors="replace")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip().lower() == "kind":
            return value.strip().strip('"').strip("'").lower()
    return ""
