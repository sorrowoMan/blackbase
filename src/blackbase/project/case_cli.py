"""Canonical Case-local CLI that preserves the Project import contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .check_output import print_case_check, print_resource_context_summary
from .runtime import (
    build_case,
    load_case_builder,
    load_case_kind,
    load_resource_context_from_env,
    run_case,
)


def run_standard_case_cli(
    entry_file: str | Path,
    *,
    framework: str = "blackbase",
    argv: Sequence[str] | None = None,
) -> int:
    """Build/check/run one standard Case without relying on relative imports."""

    parser = argparse.ArgumentParser(description="Build or run one standard Case.")
    parser.add_argument("--check", action="store_true", help="Build only; do not run.")
    args = parser.parse_args(argv)

    case_root = Path(entry_file).resolve().parent
    if case_root.parent.name != "cases":
        raise ValueError(
            "standard Case CLI must live at <project>/cases/<case>/run_solver.py"
        )
    project_root = case_root.parent.parent
    case_name = case_root.name
    case_kind = load_case_kind(project_root, case_name)
    # ``None`` means standalone.  An explicit ``{}`` remains an authoritative
    # injected payload and must still pass post-build binding validation.
    resource_context = load_resource_context_from_env(framework)
    builder = load_case_builder(project_root, case_name, case_kind=case_kind)
    case_obj = build_case(builder, resource_context=resource_context, component_overrides={})

    if args.check:
        print_case_check(case_obj)
        return 0

    effective = getattr(case_obj, "resource_context", resource_context)
    print_resource_context_summary(effective)
    print(run_case(case_obj, case_kind=case_kind))
    return 0


__all__ = ["run_standard_case_cli"]
