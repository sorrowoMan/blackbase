"""Durable artifact publication owned by the shared Project substrate.

The publication plane is deliberately semantic-neutral: blackbase stores bytes,
checks the Project lease fence and returns a :class:`DataRef`.  Frameworks such
as mlblack remain responsible for choosing model-specific serializers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from uuid import uuid4

import numpy as np

from .lease_store import RedisLeaseStore, SQLiteLeaseStore
from .model import DataRef


ARTIFACT_AUTHORITY_SCHEMA_VERSION = 1


class ArtifactPublicationError(RuntimeError):
    """Raised when a value cannot be published under the current authority."""


class ArtifactFenceError(ArtifactPublicationError):
    """Raised when the publishing Case no longer owns its Project lease fence."""


ArtifactDump = Callable[[Any, Path], None]
ArtifactPredicate = Callable[[Any], bool]


@dataclass(frozen=True)
class ArtifactSerializer:
    """One local value-to-file codec used before durable publication."""

    name: str
    extension: str
    media_type: str
    dump: ArtifactDump = field(repr=False, compare=False)
    predicate: ArtifactPredicate | None = field(default=None, repr=False, compare=False)
    unsafe: bool = False

    def __post_init__(self) -> None:
        name = str(self.name or "").strip().lower()
        if not name:
            raise ValueError("artifact serializer name must be non-empty")
        extension = str(self.extension or "").strip()
        if extension and not extension.startswith("."):
            extension = f".{extension}"
        if not callable(self.dump):
            raise TypeError("artifact serializer dump must be callable")
        if self.predicate is not None and not callable(self.predicate):
            raise TypeError("artifact serializer predicate must be callable")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "extension", extension)
        object.__setattr__(self, "media_type", str(self.media_type or "application/octet-stream"))
        object.__setattr__(self, "unsafe", bool(self.unsafe))


class ArtifactSerializerRegistry:
    """Ordered serializer registry with safe built-ins and provider extension."""

    def __init__(self, *, include_defaults: bool = True) -> None:
        self._serializers: dict[str, ArtifactSerializer] = {}
        if include_defaults:
            for serializer in _default_serializers():
                self.register(serializer)

    def register(self, serializer: ArtifactSerializer, *, replace: bool = False) -> None:
        if not isinstance(serializer, ArtifactSerializer):
            raise TypeError("serializer must be an ArtifactSerializer")
        if serializer.name in self._serializers and not replace:
            raise ValueError(f"artifact serializer '{serializer.name}' is already registered")
        self._serializers[serializer.name] = serializer

    def get(self, name: str) -> ArtifactSerializer:
        key = str(name or "").strip().lower()
        try:
            return self._serializers[key]
        except KeyError as exc:
            raise ArtifactPublicationError(
                f"unknown artifact serializer '{key}'; register it in the Case/provider"
            ) from exc

    def select(self, value: Any, *, allow_unsafe: bool) -> ArtifactSerializer:
        for serializer in self._serializers.values():
            if serializer.unsafe and not allow_unsafe:
                continue
            predicate = serializer.predicate
            if predicate is not None and predicate(value):
                return serializer
        raise ArtifactPublicationError(
            f"no safe artifact serializer accepts {type(value).__name__}; "
            "supply a framework/provider serializer or explicitly enable an unsafe one"
        )

    def names(self) -> tuple[str, ...]:
        return tuple(self._serializers)


@dataclass(frozen=True)
class ArtifactAuthority:
    """Transport-safe Project authority describing the durable artifact store."""

    backend: str = "filesystem"
    root: str = ".blackbase/artifacts"
    namespace: str = "project"
    allow_unsafe_serializers: bool = False
    schema_version: int = ARTIFACT_AUTHORITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.schema_version) != ARTIFACT_AUTHORITY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported artifact authority schema_version={self.schema_version}"
            )
        backend = str(self.backend or "filesystem").strip().lower()
        if backend != "filesystem":
            raise ValueError(
                f"unsupported artifact authority backend '{backend}'; "
                "external object stores require a formal provider backend"
            )
        root = str(self.root or "").strip()
        if not root:
            raise ValueError("filesystem artifact authority requires a root path")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "namespace", str(self.namespace or "project"))
        object.__setattr__(self, "allow_unsafe_serializers", bool(self.allow_unsafe_serializers))
        object.__setattr__(self, "schema_version", ARTIFACT_AUTHORITY_SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactAuthority":
        return cls(
            backend=str(payload.get("backend", "filesystem")),
            root=str(payload.get("root", payload.get("path", ".blackbase/artifacts"))),
            namespace=str(payload.get("namespace", "project")),
            allow_unsafe_serializers=bool(payload.get("allow_unsafe_serializers", False)),
            schema_version=int(
                payload.get("schema_version", ARTIFACT_AUTHORITY_SCHEMA_VERSION)
                or ARTIFACT_AUTHORITY_SCHEMA_VERSION
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_AUTHORITY_SCHEMA_VERSION,
            "backend": self.backend,
            "root": self.root,
            "namespace": self.namespace,
            "allow_unsafe_serializers": self.allow_unsafe_serializers,
        }


class ArtifactStore(Protocol):
    def publish_file(
        self,
        source: Path,
        *,
        relative_path: Path,
        kind: str,
        media_type: str,
        checksum: str,
        metadata: Mapping[str, Any],
        fence: Callable[[], None] | None = None,
    ) -> DataRef: ...


class FilesystemArtifactStore:
    """Atomic filesystem store constrained to one Project-owned root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def publish_file(
        self,
        source: Path,
        *,
        relative_path: Path,
        kind: str,
        media_type: str,
        checksum: str,
        metadata: Mapping[str, Any],
        fence: Callable[[], None] | None = None,
    ) -> DataRef:
        destination = (self.root / relative_path).resolve()
        if not _is_relative_to(destination, self.root):
            raise ArtifactPublicationError(
                f"artifact destination '{destination}' escapes store root '{self.root}'"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            if fence is not None:
                fence()
            with source.open("rb") as reader, temporary.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            if fence is not None:
                fence()
            os.replace(temporary, destination)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return DataRef(
            uri=str(destination),
            kind=str(kind or "artifact"),
            backend="filesystem",
            media_type=str(media_type or "application/octet-stream"),
            checksum=checksum,
            size_bytes=destination.stat().st_size,
            metadata=dict(metadata or {}),
        )


class ArtifactPublisher:
    """Serialize a Case value, publish it atomically and return a DataRef."""

    def __init__(
        self,
        authority: ArtifactAuthority,
        *,
        project_run_id: str,
        case_run_id: str,
        lease: Mapping[str, Any] | None = None,
        lease_authority: Mapping[str, Any] | None = None,
        registry: ArtifactSerializerRegistry | None = None,
        redis_client: Any = None,
    ) -> None:
        self.authority = authority
        self.project_run_id = str(project_run_id or "project-run")
        self.case_run_id = str(case_run_id or "case-run")
        self.lease = dict(lease or {})
        self.lease_authority = dict(lease_authority or {})
        self.registry = registry or ArtifactSerializerRegistry()
        self.store: ArtifactStore = FilesystemArtifactStore(authority.root)
        self._fence = _build_fence_validator(
            self.lease,
            self.lease_authority,
            redis_client=redis_client,
        )

    @classmethod
    def from_resource_context(
        cls,
        resource_context: Mapping[str, Any],
        *,
        project_run_id: str,
        case_run_id: str,
        redis_client: Any = None,
    ) -> "ArtifactPublisher | None":
        context = dict(resource_context or {})
        metadata = dict(context.get("metadata", {}) or {})
        raw_authority = metadata.get("artifact_authority")
        if not isinstance(raw_authority, Mapping):
            return None
        return cls(
            ArtifactAuthority.from_dict(raw_authority),
            project_run_id=project_run_id,
            case_run_id=case_run_id,
            lease=dict(context.get("lease", {}) or {}),
            lease_authority=dict(metadata.get("lease_authority", {}) or {}),
            redis_client=redis_client,
        )

    def register_serializer(
        self,
        serializer: ArtifactSerializer,
        *,
        replace: bool = False,
    ) -> None:
        self.registry.register(serializer, replace=replace)

    def publish(
        self,
        name: str,
        value: Any,
        *,
        serializer: str | ArtifactSerializer = "auto",
        kind: str = "artifact",
        media_type: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> DataRef:
        artifact_name = str(name or "").strip()
        if not artifact_name:
            raise ValueError("artifact name must be non-empty")
        selected = (
            self.registry.select(
                value,
                allow_unsafe=self.authority.allow_unsafe_serializers,
            )
            if isinstance(serializer, str) and str(serializer).strip().lower() == "auto"
            else self.registry.get(serializer)
            if isinstance(serializer, str)
            else serializer
        )
        if not isinstance(selected, ArtifactSerializer):
            raise TypeError("serializer must be a registered name or ArtifactSerializer")
        if selected.unsafe and not self.authority.allow_unsafe_serializers:
            raise ArtifactPublicationError(
                f"artifact serializer '{selected.name}' is unsafe and the Project authority "
                "does not allow unsafe serializers"
            )

        with tempfile.TemporaryDirectory(prefix="blackbase-artifact-") as temp_dir:
            source = Path(temp_dir) / f"payload{selected.extension}"
            selected.dump(value, source)
            if not source.is_file():
                raise ArtifactPublicationError(
                    f"artifact serializer '{selected.name}' did not create its output file"
                )
            checksum = _sha256(source)
            filename = (
                f"{_safe_segment(artifact_name)}--{checksum.split(':', 1)[1][:16]}"
                f"--{uuid4().hex[:12]}{selected.extension}"
            )
            relative = Path(
                _safe_segment(self.authority.namespace),
                _safe_segment(self.project_run_id),
                _safe_segment(self.case_run_id),
                filename,
            )
            ref_metadata = {
                "artifact_name": artifact_name,
                "serializer": selected.name,
                "serializer_unsafe": selected.unsafe,
                "project_run_id": self.project_run_id,
                "case_run_id": self.case_run_id,
                "lease_id": str(self.lease.get("lease_id", "")),
                "fencing_token": int(self.lease.get("fencing_token", 0) or 0),
                **_json_mapping(metadata or {}, path="artifact.metadata"),
            }
            return self.store.publish_file(
                source,
                relative_path=relative,
                kind=kind,
                media_type=str(media_type or selected.media_type),
                checksum=checksum,
                metadata=ref_metadata,
                fence=self._fence,
            )


def _default_serializers() -> tuple[ArtifactSerializer, ...]:
    return (
        ArtifactSerializer(
            name="file",
            extension="",
            media_type="application/octet-stream",
            predicate=lambda value: isinstance(value, Path) and value.is_file(),
            dump=lambda value, path: shutil.copyfile(Path(value), path),
        ),
        ArtifactSerializer(
            name="bytes",
            extension=".bin",
            media_type="application/octet-stream",
            predicate=lambda value: isinstance(value, (bytes, bytearray, memoryview)),
            dump=lambda value, path: path.write_bytes(bytes(value)),
        ),
        ArtifactSerializer(
            name="text",
            extension=".txt",
            media_type="text/plain; charset=utf-8",
            predicate=lambda value: isinstance(value, str),
            dump=lambda value, path: path.write_text(str(value), encoding="utf-8"),
        ),
        ArtifactSerializer(
            name="numpy_npz",
            extension=".npz",
            media_type="application/x-npz",
            predicate=lambda value: isinstance(value, np.ndarray),
            dump=_dump_numpy_npz,
        ),
        ArtifactSerializer(
            name="json",
            extension=".json",
            media_type="application/json",
            predicate=_is_json_candidate,
            dump=_dump_json,
        ),
    )


def _dump_numpy_npz(value: Any, path: Path) -> None:
    if isinstance(value, Mapping):
        arrays = {str(key): np.asarray(item) for key, item in value.items()}
        np.savez_compressed(path, **arrays)
        return
    np.savez_compressed(path, value=np.asarray(value))


def _dump_json(value: Any, path: Path) -> None:
    path.write_text(
        json.dumps(
            _json_value(value, path="artifact"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def _is_json_candidate(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, Mapping, list, tuple)):
        try:
            _json_value(value, path="artifact")
            return True
        except TypeError:
            return False
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        try:
            _json_value(as_dict(), path="artifact")
            return True
        except TypeError:
            return False
    return False


def _json_mapping(value: Mapping[str, Any], *, path: str) -> dict[str, Any]:
    converted = _json_value(value, path=path)
    if not isinstance(converted, dict):  # pragma: no cover - guarded by type
        raise TypeError(f"{path} must be a mapping")
    return converted


def _json_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, DataRef):
        return value.as_dict()
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item, path=f"{path}[]") for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_value(tolist(), path=path)
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return _json_value(as_dict(), path=path)
    raise TypeError(f"artifact field '{path}' is not JSON-safe: {type(value).__name__}")


def _build_fence_validator(
    lease: Mapping[str, Any],
    authority: Mapping[str, Any],
    *,
    redis_client: Any = None,
) -> Callable[[], None] | None:
    lease_id = str(lease.get("lease_id", ""))
    fencing_token = int(lease.get("fencing_token", 0) or 0)
    if not lease_id or fencing_token <= 0:
        return None
    backend = str(authority.get("backend", "memory") or "memory").strip().lower()
    store: Any
    if backend == "sqlite":
        store = SQLiteLeaseStore(
            Path(str(authority.get("path", ""))).resolve(),
            namespace=str(authority.get("namespace", "project")),
        )
    elif backend == "redis":
        redis_url_env = str(authority.get("redis_url_env", "BLACKBASE_REDIS_URL"))
        namespace = str(authority.get("namespace", "project"))

        def validate_redis() -> None:
            redis_url = str(os.environ.get(redis_url_env, "") or "").strip()
            if redis_client is None and not redis_url:
                raise ArtifactPublicationError(
                    f"Redis artifact fence requires environment variable {redis_url_env}"
                )
            store = RedisLeaseStore(
                redis_url or "redis://localhost:6379/0",
                namespace=namespace,
                client=redis_client,
            )
            if not store.is_current(lease_id, fencing_token):
                raise ArtifactFenceError(
                    f"artifact publication rejected by stale Project lease fence: "
                    f"lease_id='{lease_id}' token={fencing_token}"
                )

        return validate_redis
    else:
        return None

    def validate() -> None:
        if not store.is_current(lease_id, fencing_token):
            raise ArtifactFenceError(
                f"artifact publication rejected by stale Project lease fence: "
                f"lease_id='{lease_id}' token={fencing_token}"
            )

    return validate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _safe_segment(value: str) -> str:
    raw = str(value or "item").strip()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._") or "item"
    if cleaned == raw and len(cleaned) <= 96:
        return cleaned
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned[:72]}--{digest}"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


__all__ = [
    "ARTIFACT_AUTHORITY_SCHEMA_VERSION",
    "ArtifactAuthority",
    "ArtifactFenceError",
    "ArtifactPublicationError",
    "ArtifactPublisher",
    "ArtifactSerializer",
    "ArtifactSerializerRegistry",
    "ArtifactStore",
    "FilesystemArtifactStore",
]
