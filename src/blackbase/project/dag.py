"""Validated dependency graph contract for one Project DAG Stage."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .execution import ProjectConfigurationError


DAG_STAGE_SCHEMA_VERSION = "blackbase.project_dag_stage/v1"


@dataclass(frozen=True)
class DagStagePlan:
    """Immutable, validated execution plan for a dependency-driven Stage.

    Dependencies may be declared explicitly with ``depends_on`` or inferred
    from authoritative Artifact inputs such as ``producer.model``.  The plan
    never guesses dependencies from unqualified Artifact names.
    """

    stage_name: str
    cases: tuple[str, ...]
    dependencies: Mapping[str, tuple[str, ...]]
    explicit_dependencies: Mapping[str, tuple[str, ...]]
    inferred_dependencies: Mapping[str, tuple[str, ...]]
    topological_order: tuple[str, ...]

    @classmethod
    def from_stage(cls, stage: Mapping[str, Any]) -> "DagStagePlan":
        stage_name = str(stage.get("name", "stage") or "stage")
        raw_cases = stage.get("cases", ()) or ()
        if isinstance(raw_cases, str) or not isinstance(raw_cases, Sequence):
            raise ProjectConfigurationError(
                f"DAG Stage '{stage_name}' cases must be a sequence"
            )
        cases = tuple(str(item) for item in raw_cases)
        if not cases:
            raise ProjectConfigurationError(
                f"DAG Stage '{stage_name}' requires at least one Case"
            )
        if len(cases) != len(set(cases)):
            raise ProjectConfigurationError(
                f"DAG Stage '{stage_name}' contains duplicate Case names"
            )
        case_set = frozenset(cases)

        raw_explicit = stage.get("depends_on", {}) or {}
        if not isinstance(raw_explicit, Mapping):
            raise ProjectConfigurationError(
                f"DAG Stage '{stage_name}' depends_on must be a mapping"
            )
        explicit_by_name = _normalize_named_mapping(
            raw_explicit,
            stage_name=stage_name,
            field_name="depends_on",
        )
        unknown_targets = sorted(
            key for key in explicit_by_name if key not in case_set
        )
        if unknown_targets:
            raise ProjectConfigurationError(
                f"DAG Stage '{stage_name}' declares dependencies for unknown Cases: "
                f"{unknown_targets}"
            )

        explicit: dict[str, tuple[str, ...]] = {}
        for case_name in cases:
            raw_dependencies = explicit_by_name.get(case_name, ()) or ()
            if isinstance(raw_dependencies, str) or not isinstance(
                raw_dependencies, Sequence
            ):
                raise ProjectConfigurationError(
                    f"DAG Stage '{stage_name}' depends_on['{case_name}'] must be a sequence"
                )
            normalized = tuple(dict.fromkeys(str(item) for item in raw_dependencies))
            missing = sorted(item for item in normalized if item not in case_set)
            if missing:
                raise ProjectConfigurationError(
                    f"DAG Stage '{stage_name}' Case '{case_name}' depends on missing Cases: "
                    f"{missing}"
                )
            if case_name in normalized:
                raise ProjectConfigurationError(
                    f"DAG Stage '{stage_name}' Case '{case_name}' cannot depend on itself"
                )
            explicit[case_name] = normalized

        raw_inputs = stage.get("input_artifacts", {}) or {}
        if not isinstance(raw_inputs, Mapping):
            raise ProjectConfigurationError(
                f"DAG Stage '{stage_name}' input_artifacts must be a mapping"
            )
        inputs_by_name = _normalize_named_mapping(
            raw_inputs,
            stage_name=stage_name,
            field_name="input_artifacts",
        )
        unknown_input_targets = sorted(
            key for key in inputs_by_name if key not in case_set
        )
        if unknown_input_targets:
            raise ProjectConfigurationError(
                f"DAG Stage '{stage_name}' declares Artifact inputs for unknown Cases: "
                f"{unknown_input_targets}"
            )

        inferred: dict[str, tuple[str, ...]] = {}
        for case_name in cases:
            declared_inputs = inputs_by_name.get(case_name, {}) or {}
            if not isinstance(declared_inputs, Mapping):
                raise ProjectConfigurationError(
                    f"DAG Stage '{stage_name}' input_artifacts['{case_name}'] must be a mapping"
                )
            producers: list[str] = []
            for registry_key in declared_inputs.values():
                producer = _producer_case_for_registry_key(
                    str(registry_key),
                    stage_name=stage_name,
                    case_set=case_set,
                )
                if producer is None:
                    continue
                if producer == case_name:
                    raise ProjectConfigurationError(
                        f"DAG Stage '{stage_name}' Case '{case_name}' consumes its own "
                        f"Artifact '{registry_key}'"
                    )
                if producer not in producers:
                    producers.append(producer)
            inferred[case_name] = tuple(producers)

        dependencies = {
            case_name: tuple(
                dict.fromkeys((*explicit[case_name], *inferred[case_name]))
            )
            for case_name in cases
        }
        order = _stable_topological_order(
            stage_name=stage_name,
            cases=cases,
            dependencies=dependencies,
        )
        return cls(
            stage_name=stage_name,
            cases=cases,
            dependencies=MappingProxyType(dependencies),
            explicit_dependencies=MappingProxyType(explicit),
            inferred_dependencies=MappingProxyType(inferred),
            topological_order=order,
        )

    def dependencies_for(self, case_name: str) -> tuple[str, ...]:
        name = str(case_name)
        if name not in self.dependencies:
            raise KeyError(name)
        return self.dependencies[name]

    def request_metadata(self, case_name: str) -> dict[str, Any]:
        name = str(case_name)
        return {
            "schema": DAG_STAGE_SCHEMA_VERSION,
            "stage_name": self.stage_name,
            "case_name": name,
            "dependencies": list(self.dependencies_for(name)),
            "explicit_dependencies": list(self.explicit_dependencies[name]),
            "inferred_artifact_dependencies": list(self.inferred_dependencies[name]),
            "topological_index": self.topological_order.index(name),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": DAG_STAGE_SCHEMA_VERSION,
            "stage_name": self.stage_name,
            "cases": list(self.cases),
            "dependencies": {
                name: list(self.dependencies[name]) for name in self.cases
            },
            "explicit_dependencies": {
                name: list(self.explicit_dependencies[name]) for name in self.cases
            },
            "inferred_artifact_dependencies": {
                name: list(self.inferred_dependencies[name]) for name in self.cases
            },
            "topological_order": list(self.topological_order),
        }


def _producer_case_for_registry_key(
    registry_key: str,
    *,
    stage_name: str,
    case_set: frozenset[str],
) -> str | None:
    parts = tuple(part for part in str(registry_key).split(".") if part)
    if len(parts) >= 3 and parts[0] == stage_name and parts[1] in case_set:
        return parts[1]
    if len(parts) >= 2 and parts[0] in case_set:
        return parts[0]
    return None


def _normalize_named_mapping(
    values: Mapping[Any, Any],
    *,
    stage_name: str,
    field_name: str,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_name, value in values.items():
        name = str(raw_name)
        if name in normalized:
            raise ProjectConfigurationError(
                f"DAG Stage '{stage_name}' field '{field_name}' contains duplicate "
                f"normalized key '{name}'"
            )
        normalized[name] = value
    return normalized


def _stable_topological_order(
    *,
    stage_name: str,
    cases: tuple[str, ...],
    dependencies: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    remaining = {name: set(dependencies[name]) for name in cases}
    order: list[str] = []
    while remaining:
        ready = [name for name in cases if name in remaining and not remaining[name]]
        if not ready:
            cycle_members = [name for name in cases if name in remaining]
            raise ProjectConfigurationError(
                f"DAG Stage '{stage_name}' contains a dependency cycle involving: "
                f"{cycle_members}"
            )
        for name in ready:
            order.append(name)
            remaining.pop(name)
        for pending in remaining.values():
            pending.difference_update(ready)
    return tuple(order)


__all__ = ["DAG_STAGE_SCHEMA_VERSION", "DagStagePlan"]
