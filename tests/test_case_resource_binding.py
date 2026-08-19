from __future__ import annotations

import subprocess
import sys

import pytest

from blackbase.project import ProjectConfigurationError, build_case
from blackbase.project.scaffold import add_case, create_project


def test_build_case_binds_grant_even_when_builder_only_declares_parameter() -> None:
    class Case:
        pass

    def builder(*, resource_context=None, component_overrides=None):
        del resource_context, component_overrides
        return Case()

    case = build_case(
        builder,
        resource_context={"scope": "optimization", "threads": 2, "namespace": "p.c"},
    )

    assert case.resource_context["threads"] == 2
    assert case.resource_binding_audit["current"] is True
    assert case.resource_binding_audit["method"] == "resource_context_attribute"


def test_build_case_uses_case_resource_setter() -> None:
    class Case:
        def __init__(self) -> None:
            self.resource_context = None
            self.setter_calls = 0

        def set_resource_context(self, value) -> None:
            self.setter_calls += 1
            self.resource_context = dict(value)

    def builder(*, resource_context=None, component_overrides=None):
        del resource_context, component_overrides
        return Case()

    case = build_case(builder, resource_context={"threads": 3, "namespace": "p.c"})

    assert case.setter_calls == 1
    assert case.resource_binding_audit["method"] == "set_resource_context"
    assert case.resource_binding_audit["effective"]["threads"] == 3


def test_build_case_rejects_case_that_changes_authoritative_grant() -> None:
    class Case:
        def set_resource_context(self, value) -> None:
            self.resource_context = {**dict(value), "threads": 99}

    def builder(*, resource_context=None, component_overrides=None):
        del resource_context, component_overrides
        return Case()

    with pytest.raises(ProjectConfigurationError, match="changed the Project-authorized"):
        build_case(builder, resource_context={"threads": 1, "namespace": "p.c"})


def test_generated_case_cli_runs_directly_with_package_relative_builder_imports(tmp_path) -> None:
    project_root = create_project(tmp_path / "direct_cli", framework="blackbase")
    case_root = add_case("demo", "solver", project_root=project_root)
    (case_root / "helper.py").write_text(
        "class Case:\n"
        "    def run(self):\n"
        "        return {'ok': True}\n",
        encoding="utf-8",
    )
    (case_root / "build_solver.py").write_text(
        "from .helper import Case\n\n"
        "def build_solver(*, resource_context=None, component_overrides=None):\n"
        "    del resource_context, component_overrides\n"
        "    return Case()\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, "run_solver.py", "--check"],
        cwd=case_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"current": true' in completed.stdout
    assert '"status": "assembly ok"' in completed.stdout
