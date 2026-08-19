"""Semantic-neutral catalog records and query engine.

Frameworks own discovery and contract enrichment.  blackbase owns the stable
record shape, import boundary and deterministic filtering/search behavior.
"""

from __future__ import annotations

import importlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 or explicit fallback audit
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]


@dataclass(frozen=True)
class CatalogEntry:
    key: str
    title: str
    kind: str
    import_path: str
    tags: tuple[str, ...] = ()
    summary: str = ""
    companions: tuple[str, ...] = ()
    context_requires: tuple[str, ...] = ()
    context_provides: tuple[str, ...] = ()
    context_mutates: tuple[str, ...] = ()
    context_cache: tuple[str, ...] = ()
    context_notes: tuple[str, ...] = ()
    artifact_requires: tuple[str, ...] = ()
    artifact_provides: tuple[str, ...] = ()
    phase_in: tuple[str, ...] = ()
    phase_out: tuple[str, ...] = ()
    use_when: tuple[str, ...] = ()
    minimal_wiring: tuple[str, ...] = ()
    required_companions: tuple[str, ...] = ()
    config_keys: tuple[str, ...] = ()
    example_entry: str = ""
    detail_ref: str = ""
    contract: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        key = str(self.key or "").strip()
        import_path = str(self.import_path or "").strip()
        if not key:
            raise ValueError("catalog entry key must be non-empty")
        if not import_path:
            raise ValueError("catalog entry import_path must be non-empty")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "title", str(self.title or key))
        object.__setattr__(self, "kind", str(self.kind or "component").strip().lower())
        object.__setattr__(self, "import_path", import_path)
        for name in (
            "tags",
            "companions",
            "context_requires",
            "context_provides",
            "context_mutates",
            "context_cache",
            "context_notes",
            "artifact_requires",
            "artifact_provides",
            "phase_in",
            "phase_out",
            "use_when",
            "minimal_wiring",
            "required_companions",
            "config_keys",
        ):
            object.__setattr__(self, name, tuple(str(item) for item in getattr(self, name) or ()))
        object.__setattr__(self, "summary", str(self.summary or ""))
        object.__setattr__(self, "example_entry", str(self.example_entry or ""))
        object.__setattr__(self, "detail_ref", str(self.detail_ref or ""))
        object.__setattr__(self, "contract", dict(self.contract or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def load(self) -> Any:
        module_name, separator, attribute = self.import_path.partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError(f"invalid catalog import_path: {self.import_path!r}")
        project_root = str(self.metadata.get("project_root", "") or "").strip()
        case_root = str(self.metadata.get("case_root", "") or "").strip()
        case_name = str(self.metadata.get("case_name", "") or "").strip()
        if project_root and case_name:
            from .project.runtime import case_import_context

            with case_import_context(Path(project_root), case_name):
                value: Any = importlib.import_module(module_name)
        elif case_root:
            from .project.runtime import project_import_context

            with project_import_context(Path(case_root)):
                value = importlib.import_module(module_name)
        else:
            value = importlib.import_module(module_name)
        for part in attribute.split("."):
            value = getattr(value, part)
        return value

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "kind": self.kind,
            "import_path": self.import_path,
            "tags": list(self.tags),
            "summary": self.summary,
            "companions": list(self.companions),
            "context_requires": list(self.context_requires),
            "context_provides": list(self.context_provides),
            "context_mutates": list(self.context_mutates),
            "context_cache": list(self.context_cache),
            "context_notes": list(self.context_notes),
            "artifact_requires": list(self.artifact_requires),
            "artifact_provides": list(self.artifact_provides),
            "phase_in": list(self.phase_in),
            "phase_out": list(self.phase_out),
            "use_when": list(self.use_when),
            "minimal_wiring": list(self.minimal_wiring),
            "required_companions": list(self.required_companions),
            "config_keys": list(self.config_keys),
            "example_entry": self.example_entry,
            "detail_ref": self.detail_ref,
            "contract": dict(self.contract),
            "metadata": dict(self.metadata),
        }


class Catalog:
    """Deterministic catalog query engine shared by all semantic layers."""

    def __init__(self, entries: Iterable[CatalogEntry]) -> None:
        values = tuple(entries)
        if any(not isinstance(entry, CatalogEntry) for entry in values):
            raise TypeError("catalog entries must be CatalogEntry values")
        keys = tuple(entry.key for entry in values)
        if len(keys) != len(set(keys)):
            raise ValueError("catalog entry keys must be unique")
        self._entries = values
        self._by_key = {entry.key: entry for entry in values}

    def get(self, key: str) -> CatalogEntry | None:
        return self._by_key.get(str(key))

    def show(self, key: str) -> CatalogEntry:
        entry = self.get(key)
        if entry is None:
            raise KeyError(f"catalog entry not found: {key}")
        return entry

    def list(self, *, kind: str | None = None, tag: str | None = None) -> tuple[CatalogEntry, ...]:
        values = self._entries
        if kind is not None:
            normalized = str(kind).strip().lower()
            values = tuple(entry for entry in values if entry.kind == normalized)
        if tag is not None:
            normalized = str(tag).strip().lower()
            values = tuple(
                entry
                for entry in values
                if normalized in {item.lower() for item in entry.tags}
            )
        return values

    def search(
        self,
        query: str,
        *,
        kind: str | None = None,
        kinds: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        fields: str = "all",
        limit: int = 20,
    ) -> tuple[CatalogEntry, ...]:
        tokens = tuple(token for token in re.split(r"\s+", str(query).strip().lower()) if token)
        kind_set = {str(item).strip().lower() for item in tuple(kinds or ()) if str(item).strip()}
        if kind is not None:
            kind_set.add(str(kind).strip().lower())
        tag_set = {str(item).strip().lower() for item in tuple(tags or ()) if str(item).strip()}
        field = str(fields or "all").strip().lower()
        matched: list[CatalogEntry] = []
        for entry in self._entries:
            if kind_set and entry.kind not in kind_set:
                continue
            entry_tags = {item.lower() for item in entry.tags}
            if tag_set and not tag_set.issubset(entry_tags):
                continue
            haystack = _entry_search_text(entry, field=field)
            if tokens and not all(token in haystack for token in tokens):
                continue
            matched.append(entry)
        matched.sort(key=lambda entry: (entry.kind, entry.key))
        return tuple(matched[: max(0, int(limit))])


def load_catalog_toml(path: Path | str) -> tuple[CatalogEntry, ...]:
    """Load semantic-neutral catalog entries from one TOML document."""

    source = Path(path).resolve()
    if not source.is_file():
        return ()
    with source.open("rb") as stream:
        payload = tomllib.load(stream)
    rows = payload.get("entry", ())
    if not isinstance(rows, list):
        raise ValueError(f"catalog file {source} must contain [[entry]] records")
    entries: list[CatalogEntry] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise TypeError(f"catalog entry {index} in {source} must be a mapping")
        item = dict(raw)
        key = str(item.get("key", "")).strip()
        kind = str(item.get("kind", "")).strip().lower()
        import_path = str(item.get("import_path", "")).strip()
        if not key or not kind or not import_path:
            raise ValueError(
                f"catalog entry {index} in {source} requires key, kind and import_path"
            )
        detail_ref = str(item.get("detail_ref", item.get("details_file", "")) or "").strip()
        if detail_ref:
            detail_ref = str((source.parent / detail_ref).resolve())
        entries.append(
            CatalogEntry(
                key=key,
                title=str(item.get("title", key)).strip(),
                kind=kind,
                import_path=import_path,
                tags=_string_tuple(item.get("tags")),
                summary=str(item.get("summary", "")).strip(),
                companions=_string_tuple(item.get("companions")),
                context_requires=_string_tuple(item.get("context_requires")),
                context_provides=_string_tuple(item.get("context_provides")),
                context_mutates=_string_tuple(item.get("context_mutates")),
                context_cache=_string_tuple(item.get("context_cache")),
                context_notes=_string_tuple(item.get("context_notes")),
                artifact_requires=_string_tuple(item.get("artifact_requires")),
                artifact_provides=_string_tuple(item.get("artifact_provides")),
                phase_in=_string_tuple(item.get("phase_in")),
                phase_out=_string_tuple(item.get("phase_out")),
                use_when=_string_tuple(item.get("use_when")),
                minimal_wiring=_string_tuple(item.get("minimal_wiring")),
                required_companions=_string_tuple(item.get("required_companions")),
                config_keys=_string_tuple(item.get("config_keys")),
                example_entry=str(item.get("example_entry", "")).strip(),
                detail_ref=detail_ref,
                contract=dict(item.get("contract", {}) or {}),
                metadata=dict(item.get("metadata", {}) or {}),
            )
        )
    return tuple(entries)


def load_catalog_paths(paths: Iterable[Path | str]) -> tuple[CatalogEntry, ...]:
    """Load catalog files/directories in deterministic order."""

    files: list[Path] = []
    for value in paths:
        path = Path(value).resolve()
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.glob("*.toml")))
    entries: list[CatalogEntry] = []
    for path in files:
        entries.extend(load_catalog_toml(path))
    keys = tuple(entry.key for entry in entries)
    if len(keys) != len(set(keys)):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(f"duplicate catalog entry keys across TOML sources: {duplicates}")
    return tuple(entries)


def render_catalog_toml(entries: Iterable[CatalogEntry]) -> str:
    """Render deterministic ``[[entry]]`` records without a third-party TOML writer."""

    values = tuple(entries)
    if any(not isinstance(entry, CatalogEntry) for entry in values):
        raise TypeError("catalog entries must be CatalogEntry values")
    keys = tuple(entry.key for entry in values)
    if len(keys) != len(set(keys)):
        raise ValueError("catalog entry keys must be unique")

    blocks: list[str] = []
    for entry in sorted(values, key=lambda item: item.key):
        payload = entry.as_dict()
        lines = ["[[entry]]"]
        for name in (
            "key",
            "title",
            "kind",
            "import_path",
            "tags",
            "summary",
            "companions",
            "context_requires",
            "context_provides",
            "context_mutates",
            "context_cache",
            "context_notes",
            "artifact_requires",
            "artifact_provides",
            "phase_in",
            "phase_out",
            "use_when",
            "minimal_wiring",
            "required_companions",
            "config_keys",
            "example_entry",
            "detail_ref",
            "contract",
            "metadata",
        ):
            value = _without_none(payload[name])
            if name not in {"key", "title", "kind", "import_path"} and not value:
                continue
            lines.append(f"{name} = {_toml_value(value)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_catalog_shards(
    entries: Iterable[CatalogEntry],
    directory: Path | str,
    *,
    replace: bool = False,
) -> tuple[Path, ...]:
    """Write one deterministic TOML file per Catalog kind."""

    values = tuple(entries)
    keys = tuple(entry.key for entry in values)
    if len(keys) != len(set(keys)):
        raise ValueError("catalog entry keys must be unique")
    target_dir = Path(directory).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[CatalogEntry]] = {}
    for entry in values:
        grouped.setdefault(entry.kind, []).append(entry)

    written: list[Path] = []
    for kind, kind_entries in sorted(grouped.items()):
        target = target_dir / f"{kind}.toml"
        if target.exists() and not replace:
            raise FileExistsError(f"catalog shard already exists: {target}")
        temporary = target.with_suffix(".toml.tmp")
        temporary.write_text(render_catalog_toml(kind_entries), encoding="utf-8")
        temporary.replace(target)
        written.append(target)
    return tuple(written)


def _without_none(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _without_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, tuple):
        return tuple(_without_none(item) for item in value if item is not None)
    if isinstance(value, list):
        return [_without_none(item) for item in value if item is not None]
    return value


def _toml_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "+inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, Mapping):
        parts = [
            f"{json.dumps(str(key), ensure_ascii=False)} = {_toml_value(item)}"
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        ]
        return "{ " + ", ".join(parts) + " }"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"catalog TOML cannot encode {type(value).__name__}")


def _entry_search_text(entry: CatalogEntry, *, field: str) -> str:
    if field == "name":
        values = (entry.key, entry.title)
    elif field == "tag":
        values = entry.tags
    elif field == "context":
        values = (
            *entry.context_requires,
            *entry.context_provides,
            *entry.context_mutates,
            *entry.context_cache,
            *entry.context_notes,
        )
    elif field == "usage":
        values = (
            *entry.use_when,
            *entry.minimal_wiring,
            *entry.required_companions,
            *entry.config_keys,
            entry.example_entry,
        )
    else:
        values = (
            entry.key,
            entry.title,
            entry.kind,
            entry.summary,
            *entry.tags,
            *entry.context_requires,
            *entry.context_provides,
            *entry.context_mutates,
            *entry.use_when,
            *entry.minimal_wiring,
        )
    return " ".join(str(value).lower() for value in values)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Mapping):
        return tuple(str(key).strip() for key in value if str(key).strip())
    if isinstance(value, Iterable):
        return tuple(str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    return (text,) if text else ()


__all__ = [
    "Catalog",
    "CatalogEntry",
    "load_catalog_paths",
    "load_catalog_toml",
    "render_catalog_toml",
    "write_catalog_shards",
]
