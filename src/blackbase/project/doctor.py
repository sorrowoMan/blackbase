"""Common Project/Case/Scaffold doctor rules shared by frameworks."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


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

    case_count = 0
    if cases_dir.is_dir():
        for case_root in sorted(item for item in cases_dir.iterdir() if item.is_dir() and item.name != "__pycache__"):
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


def _check_case_root(case_root: Path, diags: list[DoctorDiagnostic], *, strict: bool) -> None:
    _require_file(case_root / "__init__.py", diags)
    marker = case_root / ".case"
    _require_file(marker, diags)
    kind = _read_case_kind(marker)
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

    primary_build = "build_trainer.py" if kind == "trainer" else "build_solver.py"
    primary_run = "run_trainer.py" if kind == "trainer" else "run_solver.py"
    other_build = "build_solver.py" if kind == "trainer" else "build_trainer.py"
    other_run = "run_solver.py" if kind == "trainer" else "run_trainer.py"
    _require_file(case_root / primary_build, diags)
    _require_file(case_root / primary_run, diags)
    if (case_root / other_build).exists():
        diags.append(
            DoctorDiagnostic(
                "error",
                "case-dual-build-entry",
                f"Case kind={kind} must not also define {other_build}.",
                str(case_root / other_build),
            )
        )
    if (case_root / other_run).exists():
        diags.append(
            DoctorDiagnostic(
                "error",
                "case-dual-run-entry",
                f"Case kind={kind} must not also define {other_run}.",
                str(case_root / other_run),
            )
        )

    for dirname in ("problem", "pipeline", "adapter", "bias", "plugins", "evaluation", "runtime"):
        _require_dir(case_root / dirname, diags)
    if (case_root / "capabilities").is_dir():
        diags.append(DoctorDiagnostic("error", "case-capabilities-dir", "Use plugins/ instead of case-level capabilities/.", str(case_root / "capabilities")))
    if (case_root / "representation").is_dir():
        diags.append(DoctorDiagnostic("error", "case-representation-dir", "Representation/codec operators belong under pipeline/.", str(case_root / "representation")))
    if (case_root / "assembly" / "scaffold.json").is_file():
        diags.append(DoctorDiagnostic("error", "case-assembly-scaffold-json", "assembly/scaffold.json is not a formal build entry.", str(case_root / "assembly" / "scaffold.json")))

    _check_build_signature(case_root / primary_build, kind=kind, diags=diags, strict=strict)


def _check_build_signature(path: Path, *, kind: str, diags: list[DoctorDiagnostic], strict: bool) -> None:
    if not path.is_file():
        return
    func_name = "build_trainer" if kind == "trainer" else "build_solver"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
    except SyntaxError as exc:
        diags.append(DoctorDiagnostic("error", "case-build-syntax", f"{path.name} has syntax error: {exc}", str(path)))
        return
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            target = node
            break
    if target is None:
        diags.append(DoctorDiagnostic("error", "case-build-missing-function", f"{path.name} must define {func_name}().", str(path)))
        return
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
