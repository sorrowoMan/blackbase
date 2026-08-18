"""Formal Project/Case Catalog scope discovery and TOML aggregation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from ..catalog import CatalogEntry, load_catalog_paths


@dataclass(frozen=True)
class CatalogScope:
    root: Path
    kind: str
    project_root: Path | None = None
    case_name: str | None = None


def find_catalog_scope(start: Path | str) -> CatalogScope | None:
    """Find the nearest formal Case or Project containing ``start``."""

    path = Path(start).resolve()
    if path.is_file():
        path = path.parent
    for current in (path, *path.parents):
        if _is_case_root(current):
            project_root = current.parent.parent if current.parent.name == "cases" else None
            return CatalogScope(
                root=current,
                kind="case",
                project_root=project_root if project_root and _is_project_root(project_root) else None,
                case_name=current.name,
            )
        if _is_project_root(current):
            return CatalogScope(root=current, kind="project", project_root=current)
    return None


def iter_catalog_scopes(scope: CatalogScope | Path | str) -> tuple[CatalogScope, ...]:
    """Return deterministic Catalog sources for one formal scope."""

    resolved = scope if isinstance(scope, CatalogScope) else find_catalog_scope(scope)
    if resolved is None:
        return ()
    if resolved.kind == "case":
        return (resolved,)

    values: list[CatalogScope] = [resolved]
    cases_dir = resolved.root / "cases"
    for case_root in sorted(item for item in cases_dir.iterdir() if item.is_dir()):
        if not _is_case_root(case_root):
            continue
        values.append(
            CatalogScope(
                root=case_root,
                kind="case",
                project_root=resolved.root,
                case_name=case_root.name,
            )
        )
    return tuple(values)


def load_scaffold_catalog_entries(scope: CatalogScope | Path | str) -> tuple[CatalogEntry, ...]:
    """Load the split TOML catalogs for a Case or all Cases in a Project."""

    resolved = scope if isinstance(scope, CatalogScope) else find_catalog_scope(scope)
    if resolved is None:
        raise FileNotFoundError(f"no formal Project/Case scope found for: {scope}")

    entries: list[CatalogEntry] = []
    for source in iter_catalog_scopes(resolved):
        loaded = load_catalog_paths((source.root / "catalog" / "entries",))
        for entry in loaded:
            if source.kind == "case":
                entry = _case_scoped_entry(
                    entry,
                    source,
                    namespace_key=resolved.kind == "project",
                )
            entries.append(entry)

    keys = tuple(entry.key for entry in entries)
    if len(keys) != len(set(keys)):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError("duplicate Catalog keys across scope: " + ", ".join(duplicates[:16]))
    return tuple(entries)


def _case_scoped_entry(
    entry: CatalogEntry,
    scope: CatalogScope,
    *,
    namespace_key: bool,
) -> CatalogEntry:
    case_name = str(scope.case_name or scope.root.name)
    key = _project_case_key(entry.key, case_name=case_name) if namespace_key else entry.key
    companions = tuple(
        _project_case_key(value, case_name=case_name)
        if namespace_key and str(value).startswith("project.")
        else str(value)
        for value in entry.companions
    )
    import_path = (
        _case_import_path(entry.import_path, case_name=case_name)
        if scope.project_root is not None
        else entry.import_path
    )
    metadata = {
        **dict(entry.metadata),
        "catalog_scope_kind": "case",
        "case_root": str(scope.root),
        "case_name": case_name,
        "local_catalog_key": entry.key,
    }
    if scope.project_root is not None:
        metadata["project_root"] = str(scope.project_root)
    return replace(
        entry,
        key=key,
        import_path=import_path,
        companions=companions,
        metadata=metadata,
    )


def _project_case_key(value: str, *, case_name: str) -> str:
    suffix = str(value).strip().removeprefix("project.")
    if not suffix.startswith(f"{case_name}."):
        suffix = f"{case_name}.{suffix}"
    return f"project.{suffix}"


def _case_import_path(value: str, *, case_name: str) -> str:
    """Normalize a Case-local import to the portable Project namespace."""

    module_name, separator, symbol_name = str(value).partition(":")
    if not separator:
        return str(value)
    case_prefix = f"cases.{case_name}"
    marker = f".{case_prefix}"
    if module_name == case_prefix or module_name.startswith(f"{case_prefix}."):
        scoped_module = module_name
    elif marker in module_name:
        # Generated catalogs once recorded repository-qualified example paths.
        # Keep only the self-contained Project/Case suffix when such a Project is
        # copied elsewhere.
        scoped_module = case_prefix + module_name.rsplit(marker, 1)[1]
    else:
        scoped_module = f"{case_prefix}.{module_name}"
    return f"{scoped_module}:{symbol_name}"


def _is_project_root(path: Path) -> bool:
    return (path / "project_config.py").is_file() and (path / "cases").is_dir()


def _is_case_root(path: Path) -> bool:
    if not (path / "build_solver.py").is_file():
        return False
    return (path / ".case").is_file() or path.parent.name == "cases"


__all__ = [
    "CatalogScope",
    "find_catalog_scope",
    "iter_catalog_scopes",
    "load_scaffold_catalog_entries",
]
