from __future__ import annotations

import json

from blackbase.evaluation import EvaluationProviderSpec

from blackbase.project import (
    build_case_check_payload,
    format_case_check,
    format_resource_context_summary,
    load_resource_context_from_env,
    run_standard_case_cli,
)
from blackbase.project.doctor import run_common_project_doctor
from blackbase.project.scaffold import add_case, create_project


class _Named:
    def __init__(self, name: str):
        self.name = name


def test_case_check_reports_attached_components_without_secrets() -> None:
    pipeline = type(
        "Pipeline",
        (),
        {"describe": lambda self: {"class": "Pipeline", "initializer": "Init", "route": "linear"}},
    )()
    case = type(
        "Case",
        (),
        {
            "problem": _Named("problem"),
            "adapter": _Named("adapter"),
            "representation_pipeline": pipeline,
            "resource_context": {"threads": 2, "api_key": "secret"},
        },
    )()

    payload = build_case_check_payload(case)

    assert payload["assembly"] == "Case"
    assert payload["problem"] == "problem"
    assert payload["pipeline"] == "Pipeline"
    assert payload["initializer"] == "Init"
    assert payload["pipeline_variant"] == "linear"
    assert payload["resource_context"] == {"threads": 2}
    assert json.loads(format_case_check(case).removeprefix("[check] ")) == payload
    resource_summary = json.loads(
        format_resource_context_summary(case.resource_context).removeprefix("[resource-context] ")
    )
    assert resource_summary == {"threads": 2}


def test_case_check_discovers_formal_problem_provider_and_state_protocol() -> None:
    provider = type(
        "Provider",
        (),
        {
            "spec": EvaluationProviderSpec(
                provider_id="ml.torch/v1",
                problem_ids=("ml.problem/v1",),
                compute_backend="torch",
                state_kinds=("model_parameters",),
                materialization_targets=("unknown_state",),
                transition_methods=("gradient.adam",),
            )
        },
    )()
    problem = type("Problem", (), {"name": "problem", "provider": provider})()
    case = type(
        "Case",
        (),
        {
            "problem": problem,
            "evaluation_provider": provider,
        },
    )()

    payload = build_case_check_payload(case)

    assert payload["providers"] == ["ml.torch/v1"]
    assert payload["provider_details"] == [
        {
            "name": "Provider",
            "provider_id": "ml.torch/v1",
            "compute_backend": "torch",
            "problem_ids": ["ml.problem/v1"],
            "transition_methods": ["gradient.adam"],
            "materialization_targets": ["unknown_state"],
        }
    ]


def test_strict_doctor_requires_case_build_check_contract(tmp_path) -> None:
    project_root = create_project(tmp_path / "project")
    case_root = add_case("search", "solver", project_root=project_root)
    run_entry = case_root / "run_solver.py"
    run_entry.write_text(
        "def main(argv=None):\n"
        "    del argv\n"
        "    return 0\n"
        "\n# Mentioning --check in a comment is not a CLI contract.\n",
        encoding="utf-8",
    )

    report = run_common_project_doctor(project_root, strict=True)

    assert any(item.code == "case-run-missing-check-contract" for item in report.diagnostics)


def test_strict_doctor_accepts_called_build_entry_check_delegate(tmp_path) -> None:
    project_root = create_project(tmp_path / "project")
    case_root = add_case("search", "solver", project_root=project_root)
    build_entry = case_root / "build_solver.py"
    build_entry.write_text(
        build_entry.read_text(encoding="utf-8")
        + "\n\ndef main(argv=None):\n"
        + "    import argparse\n"
        + "    parser = argparse.ArgumentParser()\n"
        + "    parser.add_argument('--check', action='store_true')\n"
        + "    parser.parse_args(argv)\n",
        encoding="utf-8",
    )
    (case_root / "run_solver.py").write_text(
        "from .build_solver import main as _main\n\n"
        "def main(argv=None):\n"
        "    return _main(argv)\n",
        encoding="utf-8",
    )

    report = run_common_project_doctor(project_root, strict=True)

    assert not any(item.code == "case-run-missing-check-contract" for item in report.diagnostics)


def test_doctor_rejects_build_entry_returning_cli_callable(tmp_path) -> None:
    project_root = create_project(tmp_path / "project")
    case_root = add_case("search", "solver", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        """
def main():
    return None


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return main
""".lstrip(),
        encoding="utf-8",
    )

    report = run_common_project_doctor(project_root, strict=True)

    assert any(item.code == "case-build-returns-entry-callable" for item in report.diagnostics)


def test_cli_case_loads_framework_resource_context_from_environment() -> None:
    payload = load_resource_context_from_env(
        "mlblack",
        environ={
            "MLBLACK_RESOURCE_CONTEXT_JSON": '{"threads": 2, "namespace": "project.case"}',
            "BLACKBASE_RESOURCE_CONTEXT_JSON": '{"threads": 1}',
        },
    )

    assert payload == {"threads": 2, "namespace": "project.case"}


def test_cli_resource_context_distinguishes_absent_and_explicit_empty_grant() -> None:
    assert load_resource_context_from_env("blackbase", environ={}) is None
    assert load_resource_context_from_env(
        "blackbase",
        environ={"BLACKBASE_RESOURCE_CONTEXT_JSON": "{}"},
    ) == {}


def test_standard_case_cli_treats_missing_environment_grant_as_standalone(
    tmp_path,
    monkeypatch,
) -> None:
    project_root = create_project(tmp_path / "project")
    case_root = add_case("search", "solver", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        "class Case:\n"
        "    def __init__(self):\n"
        "        self.resource_context = {'namespace': 'standalone.local'}\n\n"
        "def build_solver(config=None, *, resource_context=None, component_overrides=None):\n"
        "    assert resource_context is None\n"
        "    return Case()\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("BLACKBASE_RESOURCE_CONTEXT_JSON", raising=False)

    assert run_standard_case_cli(
        case_root / "run_solver.py",
        framework="blackbase",
        argv=("--check",),
    ) == 0


def test_doctor_rejects_build_entry_returning_multiple_cases(tmp_path) -> None:
    project_root = create_project(tmp_path / "project")
    case_root = add_case("comparison", "trainer", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        """
def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return {name: object() for name in ("a", "b")}
""".lstrip(),
        encoding="utf-8",
    )

    report = run_common_project_doctor(project_root, strict=True)

    assert any(item.code == "case-build-returns-collection" for item in report.diagnostics)


def test_doctor_rejects_migrated_runpy_wrapper(tmp_path) -> None:
    project_root = create_project(tmp_path / "project")
    case_root = add_case("legacy", "trainer", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        "import runpy\n\n"
        "class MigratedExampleRunner:\n"
        "    pass\n\n"
        "def build_solver(config=None, *, resource_context=None, component_overrides=None):\n"
        "    return MigratedExampleRunner()\n",
        encoding="utf-8",
    )

    report = run_common_project_doctor(project_root, strict=True)

    assert any(
        item.code == "case-migrated-wrapper-forbidden"
        for item in report.diagnostics
    )


def test_doctor_rejects_compatibility_marker_and_does_not_exempt_wrapper(tmp_path) -> None:
    project_root = create_project(tmp_path / "project")
    case_root = add_case("legacy", "trainer", project_root=project_root)
    marker = case_root / ".case"
    marker.write_text(
        marker.read_text(encoding="utf-8") + "compatibility = true\n",
        encoding="utf-8",
    )
    (case_root / "build_solver.py").write_text(
        "import runpy\n\n"
        "class MigratedExampleRunner:\n"
        "    pass\n\n"
        "def build_solver(config=None, *, resource_context=None, component_overrides=None):\n"
        "    return MigratedExampleRunner()\n",
        encoding="utf-8",
    )

    report = run_common_project_doctor(project_root, strict=True)

    assert any(
        item.code == "case-compatibility-marker-forbidden"
        for item in report.diagnostics
    )
    assert any(
        item.code == "case-migrated-wrapper-forbidden"
        for item in report.diagnostics
    )


def test_doctor_rejects_private_case_control_wrapper(tmp_path) -> None:
    project_root = create_project(tmp_path / "project")
    case_root = add_case("hidden", "solver", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class HiddenOuterCase:\n"
        "    def run(self):\n"
        "        return {}\n\n"
        "def build_solver(config=None, *, resource_context=None, component_overrides=None):\n"
        "    return HiddenOuterCase()\n",
        encoding="utf-8",
    )

    report = run_common_project_doctor(project_root, strict=True)

    assert any(
        item.code == "case-private-control-wrapper-forbidden"
        for item in report.diagnostics
    )


def test_doctor_rejects_compatibility_assembly_source(tmp_path) -> None:
    project_root = create_project(tmp_path / "project")
    case_root = add_case("legacy", "trainer", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        '"""Compatibility assembly for an old runner."""\n\n'
        "def build_solver(config=None, *, resource_context=None, component_overrides=None):\n"
        "    return object()\n",
        encoding="utf-8",
    )

    report = run_common_project_doctor(project_root, strict=True)

    assert any(
        item.code == "case-compatibility-source-forbidden"
        for item in report.diagnostics
    )


def test_doctor_rejects_canonical_builder_sys_path_mutation(tmp_path) -> None:
    project_root = create_project(tmp_path / "project")
    case_root = add_case("legacy_imports", "solver", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        "import sys\n\n"
        "sys.path.insert(0, '/tmp/private-case-path')\n\n"
        "def build_solver(config=None, *, resource_context=None, component_overrides=None):\n"
        "    return object()\n",
        encoding="utf-8",
    )

    report = run_common_project_doctor(project_root, strict=True)

    assert any(
        item.code == "case-build-mutates-sys-path"
        and item.level == "error"
        for item in report.diagnostics
    )


def test_doctor_rejects_executable_main_guard_in_canonical_builder(tmp_path) -> None:
    project_root = create_project(tmp_path / "project")
    case_root = add_case("double_entry", "solver", project_root=project_root)
    build_entry = case_root / "build_solver.py"
    build_entry.write_text(
        build_entry.read_text(encoding="utf-8")
        + "\n\ndef main():\n"
        + "    return 0\n\n"
        + "if __name__ == '__main__':\n"
        + "    raise SystemExit(main())\n",
        encoding="utf-8",
    )

    report = run_common_project_doctor(project_root, strict=True)

    assert any(
        item.code == "case-build-executable-entry-forbidden"
        and item.level == "error"
        for item in report.diagnostics
    )


def test_doctor_rejects_cli_surface_without_main_guard_in_canonical_builder(tmp_path) -> None:
    project_root = create_project(tmp_path / "project")
    case_root = add_case("hidden_double_entry", "solver", project_root=project_root)
    (case_root / "build_solver.py").write_text(
        "import argparse\n\n"
        "def build_solver(config=None, *, resource_context=None, component_overrides=None):\n"
        "    return object()\n\n"
        "def _build_parser():\n"
        "    return argparse.ArgumentParser()\n\n"
        "def main(argv=None):\n"
        "    return _build_parser().parse_args(argv)\n",
        encoding="utf-8",
    )

    report = run_common_project_doctor(project_root, strict=True)

    assert any(
        item.code == "case-build-cli-surface-forbidden"
        and item.level == "error"
        for item in report.diagnostics
    )


def test_doctor_ignores_cache_only_case_residue_but_rejects_missing_declared_case(tmp_path) -> None:
    project_root = create_project(tmp_path / "project")
    cache_only = project_root / "cases" / "removed_case" / "__pycache__"
    cache_only.mkdir(parents=True)
    (cache_only / "build_solver.pyc").write_bytes(b"cache")
    (project_root / "project_config.py").write_text(
        "STAGES = [{'name': 'main', 'cases': ['missing_case']}]\n"
        "GROUPS = {'default': {'stages': ['main']}}\n",
        encoding="utf-8",
    )

    report = run_common_project_doctor(project_root, strict=True)

    assert any(item.code == "project-stage-case-missing" for item in report.diagnostics)
    assert not any(
        item.path and "removed_case" in item.path and item.level == "error"
        for item in report.diagnostics
    )
