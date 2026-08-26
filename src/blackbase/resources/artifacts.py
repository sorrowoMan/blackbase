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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from uuid import uuid4

import numpy as np

from .lease_store import RedisLeaseStore, SQLiteLeaseStore
from .model import DataRef
from blackbase.wire import freeze_wire_mapping, thaw_wire_mapping


ARTIFACT_AUTHORITY_SCHEMA_VERSION = 1
ARTIFACT_PUBLICATION_RECEIPT_SCHEMA_VERSION = 1
ARTIFACT_BINDING_SCHEMA_VERSION = 1


class ArtifactPublicationError(RuntimeError):
    """Raised when a value cannot be published under the current authority."""


class ArtifactFenceError(ArtifactPublicationError):
    """Raised when the publishing Case no longer owns its Project lease fence."""


@dataclass(frozen=True)
class ArtifactPublicationReceipt:
    """Authority-issued proof that one named Artifact became visible.

    ``DataRef`` intentionally remains a plain location/value reference.  This
    receipt is the Project-registry capability: consumers must validate it
    against the authority ledger before promoting the enclosed ref to a
    cross-Case input.
    """

    publication_id: str
    artifact_name: str
    ref: DataRef
    project_run_id: str
    case_run_id: str
    transaction_id: str
    authority_namespace: str
    lease_id: str = ""
    fencing_token: int = 0
    committed_at: float = 0.0
    receipt_digest: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = ARTIFACT_PUBLICATION_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.schema_version) != ARTIFACT_PUBLICATION_RECEIPT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported ArtifactPublicationReceipt schema_version="
                f"{self.schema_version}"
            )
        for field_name in (
            "publication_id",
            "artifact_name",
            "project_run_id",
            "case_run_id",
            "transaction_id",
            "authority_namespace",
        ):
            value = str(getattr(self, field_name, "") or "").strip()
            if not value:
                raise ValueError(
                    f"ArtifactPublicationReceipt.{field_name} must not be empty"
                )
            object.__setattr__(self, field_name, value)
        ref = self.ref if isinstance(self.ref, DataRef) else DataRef.from_dict(self.ref)
        if not ref.checksum:
            raise ValueError("authoritative Artifact publication requires a checksum")
        object.__setattr__(self, "ref", ref)
        object.__setattr__(self, "lease_id", str(self.lease_id or ""))
        object.__setattr__(self, "fencing_token", max(0, int(self.fencing_token or 0)))
        committed_at = float(self.committed_at or 0.0)
        if committed_at <= 0:
            raise ValueError("Artifact publication committed_at must be positive")
        object.__setattr__(self, "committed_at", committed_at)
        object.__setattr__(
            self,
            "metadata",
            freeze_wire_mapping(
                _json_mapping(self.metadata or {}, path="artifact_publication.metadata"),
                path="artifact_publication.metadata",
            ),
        )
        expected = _artifact_publication_digest(self._unsigned_dict())
        supplied = str(self.receipt_digest or expected).strip().lower()
        if supplied != expected:
            raise ValueError("Artifact publication receipt digest mismatch")
        object.__setattr__(self, "receipt_digest", expected)
        object.__setattr__(
            self,
            "schema_version",
            ARTIFACT_PUBLICATION_RECEIPT_SCHEMA_VERSION,
        )

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_PUBLICATION_RECEIPT_SCHEMA_VERSION,
            "publication_id": self.publication_id,
            "artifact_name": self.artifact_name,
            "ref": self.ref.as_dict(),
            "project_run_id": self.project_run_id,
            "case_run_id": self.case_run_id,
            "transaction_id": self.transaction_id,
            "authority_namespace": self.authority_namespace,
            "lease_id": self.lease_id,
            "fencing_token": self.fencing_token,
            "committed_at": self.committed_at,
            "metadata": thaw_wire_mapping(self.metadata),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._unsigned_dict(), "receipt_digest": self.receipt_digest}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactPublicationReceipt":
        data = dict(payload or {})
        return cls(
            publication_id=str(data.get("publication_id", "")),
            artifact_name=str(data.get("artifact_name", "")),
            ref=DataRef.from_dict(dict(data.get("ref", {}) or {})),
            project_run_id=str(data.get("project_run_id", "")),
            case_run_id=str(data.get("case_run_id", "")),
            transaction_id=str(data.get("transaction_id", "")),
            authority_namespace=str(data.get("authority_namespace", "")),
            lease_id=str(data.get("lease_id", "")),
            fencing_token=int(data.get("fencing_token", 0) or 0),
            committed_at=float(data.get("committed_at", 0.0) or 0.0),
            receipt_digest=str(data.get("receipt_digest", "")),
            metadata=dict(data.get("metadata", {}) or {}),
            schema_version=int(
                data.get(
                    "schema_version",
                    ARTIFACT_PUBLICATION_RECEIPT_SCHEMA_VERSION,
                )
                or ARTIFACT_PUBLICATION_RECEIPT_SCHEMA_VERSION
            ),
        )


@dataclass(frozen=True)
class ArtifactBinding:
    """Versioned authority capability for one cross-Case Artifact input.

    A :class:`DataRef` only describes data. It does not prove that a successful
    Case finalized that data under the Project authority. This envelope keeps
    the ref and its authority receipt inseparable across Case/process boundaries.
    """

    ref: DataRef
    publication: ArtifactPublicationReceipt
    binding_digest: str = ""
    schema_version: int = ARTIFACT_BINDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.schema_version) != ARTIFACT_BINDING_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported ArtifactBinding schema_version={self.schema_version}"
            )
        ref = self.ref if isinstance(self.ref, DataRef) else DataRef.from_dict(self.ref)
        publication = (
            self.publication
            if isinstance(self.publication, ArtifactPublicationReceipt)
            else ArtifactPublicationReceipt.from_dict(self.publication)
        )
        if publication.ref != ref:
            raise ValueError("ArtifactBinding ref does not match its publication receipt")
        if publication.metadata.get("case_finalization_sealed") is not True:
            raise ValueError(
                "ArtifactBinding requires a Case-finalization-sealed publication receipt"
            )
        unsigned = {
            "schema_version": ARTIFACT_BINDING_SCHEMA_VERSION,
            "ref": ref.as_dict(),
            "publication": publication.as_dict(),
        }
        expected = _artifact_publication_digest(unsigned)
        supplied = str(self.binding_digest or expected).strip().lower()
        if supplied != expected:
            raise ValueError("ArtifactBinding digest mismatch")
        object.__setattr__(self, "ref", ref)
        object.__setattr__(self, "publication", publication)
        object.__setattr__(self, "binding_digest", expected)
        object.__setattr__(self, "schema_version", ARTIFACT_BINDING_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_BINDING_SCHEMA_VERSION,
            "ref": self.ref.as_dict(),
            "publication": self.publication.as_dict(),
            "binding_digest": self.binding_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactBinding":
        data = dict(payload or {})
        return cls(
            ref=DataRef.from_dict(dict(data.get("ref", {}) or {})),
            publication=ArtifactPublicationReceipt.from_dict(
                dict(data.get("publication", {}) or {})
            ),
            binding_digest=str(data.get("binding_digest", "")),
            schema_version=int(
                data.get("schema_version", ARTIFACT_BINDING_SCHEMA_VERSION)
                or ARTIFACT_BINDING_SCHEMA_VERSION
            ),
        )


def _artifact_publication_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


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


class FilesystemArtifactPublicationLedger:
    """Durable name reservations and commit receipts for one Case run.

    Payload bytes are provisional until a transaction commit record exists.
    The commit record is the authority point used by Project registries; name
    reservations prevent two concurrent transactions from both publishing the
    same logical Artifact before either reaches commit.
    """

    _LEDGER_SCHEMA = "blackbase.artifact_publication_ledger/v1"

    def __init__(
        self,
        authority: ArtifactAuthority,
        *,
        project_run_id: str,
        case_run_id: str,
    ) -> None:
        self.authority = authority
        self.project_run_id = str(project_run_id or "").strip()
        self.case_run_id = str(case_run_id or "").strip()
        if not self.project_run_id or not self.case_run_id:
            raise ValueError("Artifact publication ledger requires Project and Case run IDs")
        self.root = (
            Path(authority.root).resolve()
            / _safe_segment(authority.namespace)
            / _safe_segment(self.project_run_id)
            / ".publication-ledger"
        )
        self.root.mkdir(parents=True, exist_ok=True)

    def reserve(
        self,
        artifact_name: str,
        transaction_id: str,
        *,
        lease_id: str = "",
        fencing_token: int = 0,
    ) -> None:
        name = str(artifact_name or "").strip()
        transaction = str(transaction_id or "").strip()
        if not name or not transaction:
            raise ValueError("Artifact reservation requires name and transaction_id")
        path = self._reservation_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": self._LEDGER_SCHEMA,
            "status": "reserved",
            "artifact_name": name,
            "project_run_id": self.project_run_id,
            "case_run_id": self.case_run_id,
            "transaction_id": transaction,
            "lease_id": str(lease_id or ""),
            "fencing_token": max(0, int(fencing_token or 0)),
            "reserved_at": time.time(),
        }
        try:
            self._write_json_create(path, payload)
        except FileExistsError as exc:
            current = self._read_json(path)
            if (
                str(current.get("status", "")) == "reserved"
                and str(current.get("transaction_id", "")) == transaction
                and str(current.get("artifact_name", "")) == name
            ):
                return
            raise ArtifactPublicationError(
                f"Artifact name '{name}' is already reserved by another transaction"
            ) from exc

    def attach_provisional(
        self,
        artifact_name: str,
        transaction_id: str,
        ref: DataRef,
    ) -> None:
        """Bind one physical provisional object to its durable reservation."""

        name = str(artifact_name or "").strip()
        transaction = str(transaction_id or "").strip()
        path = self._reservation_path(name)
        current = self._read_json(path)
        if (
            current.get("status") != "reserved"
            or current.get("transaction_id") != transaction
            or current.get("artifact_name") != name
        ):
            raise ArtifactPublicationError(
                f"Artifact '{name}' has no matching provisional reservation"
            )
        existing = current.get("provisional_ref")
        if isinstance(existing, Mapping):
            if DataRef.from_dict(existing) != ref:
                raise ArtifactPublicationError(
                    f"Artifact '{name}' reservation changed provisional content"
                )
            return
        current["provisional_ref"] = ref.as_dict()
        current["provisional_recorded_at"] = time.time()
        self._write_json_replace(path, current)

    def commit(
        self,
        transaction_id: str,
        refs: Mapping[str, DataRef],
        *,
        lease_id: str = "",
        fencing_token: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, ArtifactPublicationReceipt]:
        transaction = str(transaction_id or "").strip()
        if not transaction:
            raise ValueError("Artifact commit requires transaction_id")
        normalized = {str(name): ref for name, ref in dict(refs or {}).items()}
        commit_path = self._commit_path(transaction)
        existing = self._read_json(commit_path)
        if existing:
            if existing.get("status") != "committed":
                raise ArtifactPublicationError(
                    f"Artifact transaction '{transaction}' has a non-commit record"
                )
            raw_receipts = dict(existing.get("receipts", {}) or {})
            if set(raw_receipts) != set(normalized):
                raise ArtifactPublicationError(
                    f"Artifact transaction '{transaction}' retry changed Artifact names"
                )
            restored = {
                name: ArtifactPublicationReceipt.from_dict(raw)
                for name, raw in raw_receipts.items()
            }
            if any(restored[name].ref != normalized[name] for name in normalized):
                raise ArtifactPublicationError(
                    f"Artifact transaction '{transaction}' retry changed a DataRef"
                )
            for name, ref in normalized.items():
                self._validate_physical_ref(name, ref)
            return restored
        if self._tombstone_path(transaction).exists():
            raise ArtifactPublicationError(
                f"Artifact transaction '{transaction}' was already aborted"
            )
        for name, ref in normalized.items():
            reservation = self._read_json(self._reservation_path(name))
            if (
                reservation.get("status") != "reserved"
                or reservation.get("transaction_id") != transaction
                or reservation.get("artifact_name") != name
            ):
                raise ArtifactPublicationError(
                    f"Artifact name '{name}' is not reserved by transaction '{transaction}'"
                )
            if not isinstance(ref, DataRef) or not ref.checksum:
                raise ArtifactPublicationError(
                    f"Artifact '{name}' has no checksummed DataRef"
                )
            self._validate_physical_ref(name, ref)

        committed_at = time.time()
        receipts: dict[str, ArtifactPublicationReceipt] = {}
        for name, ref in normalized.items():
            publication_seed = "\x00".join(
                (self.project_run_id, self.case_run_id, transaction, name)
            ).encode("utf-8")
            publication_id = "publication-" + hashlib.sha256(publication_seed).hexdigest()[:32]
            receipts[name] = ArtifactPublicationReceipt(
                publication_id=publication_id,
                artifact_name=name,
                ref=ref,
                project_run_id=self.project_run_id,
                case_run_id=self.case_run_id,
                transaction_id=transaction,
                authority_namespace=self.authority.namespace,
                lease_id=str(lease_id or ""),
                fencing_token=int(fencing_token or 0),
                committed_at=committed_at,
                metadata=dict(metadata or {}),
            )

        record = {
            "schema": self._LEDGER_SCHEMA,
            "status": "committed",
            "project_run_id": self.project_run_id,
            "case_run_id": self.case_run_id,
            "transaction_id": transaction,
            "committed_at": committed_at,
            "receipts": {name: receipt.as_dict() for name, receipt in receipts.items()},
        }
        commit_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._write_json_create(commit_path, record)
        except FileExistsError:
            existing = self._read_json(commit_path)
            if existing != record:
                raise ArtifactPublicationError(
                    f"Artifact transaction '{transaction}' commit record conflicts"
                )

        # The create-only commit record above is the authority seal point.
        # Name records are a recoverable lookup index; an I/O failure while
        # compacting one must not revoke receipts that are already durable.
        for name in normalized:
            reservation_path = self._reservation_path(name)
            finalized = {
                "schema": self._LEDGER_SCHEMA,
                "status": "committed",
                "artifact_name": name,
                "project_run_id": self.project_run_id,
                "case_run_id": self.case_run_id,
                "transaction_id": transaction,
                "receipt_digest": receipts[name].receipt_digest,
                "committed_at": committed_at,
            }
            try:
                self._write_json_replace(reservation_path, finalized)
            except OSError:
                # A later reserve still fails closed on the existing reserved
                # record, while verify() reads the authoritative commit record.
                continue
        return receipts

    def _validate_physical_ref(self, artifact_name: str, ref: DataRef) -> None:
        """Fail closed unless a staged ref still names its exact authority bytes."""

        name = str(artifact_name or "").strip()
        if ref.backend != "filesystem":
            raise ArtifactPublicationError(
                "filesystem Artifact ledger cannot seal backend "
                f"'{ref.backend}' for '{name}'"
            )
        target = Path(ref.uri).resolve()
        case_root = (
            Path(self.authority.root).resolve()
            / _safe_segment(self.authority.namespace)
            / _safe_segment(self.project_run_id)
            / _safe_segment(self.case_run_id)
        ).resolve()
        if not _is_relative_to(target, case_root):
            raise ArtifactPublicationError(
                f"Artifact '{name}' escapes its authorized Case directory"
            )
        if not target.is_file():
            raise ArtifactPublicationError(
                f"Artifact '{name}' is missing before authority seal"
            )
        try:
            observed_size = int(target.stat().st_size)
            observed_checksum = _sha256(target)
        except OSError as exc:
            raise ArtifactPublicationError(
                f"Artifact '{name}' cannot be read before authority seal"
            ) from exc
        if ref.size_bytes is None:
            raise ArtifactPublicationError(
                f"Artifact '{name}' omitted its physical size before authority seal"
            )
        if observed_size != int(ref.size_bytes):
            raise ArtifactPublicationError(
                f"Artifact '{name}' size changed before authority seal"
            )
        if observed_checksum != ref.checksum:
            raise ArtifactPublicationError(
                f"Artifact '{name}' checksum changed before authority seal"
            )
        metadata = dict(ref.metadata or {})
        expected_metadata = {
            "artifact_name": name,
            "project_run_id": self.project_run_id,
            "case_run_id": self.case_run_id,
        }
        for key, expected in expected_metadata.items():
            if str(metadata.get(key, "")) != str(expected):
                raise ArtifactPublicationError(
                    f"Artifact '{name}' metadata does not match {key} authority"
                )

    def verify(self, receipt: ArtifactPublicationReceipt) -> bool:
        if not isinstance(receipt, ArtifactPublicationReceipt):
            return False
        if (
            receipt.project_run_id != self.project_run_id
            or receipt.case_run_id != self.case_run_id
            or receipt.authority_namespace != self.authority.namespace
        ):
            return False
        record = self._read_json(self._commit_path(receipt.transaction_id))
        if record.get("status") != "committed":
            return False
        raw = dict(record.get("receipts", {}) or {}).get(receipt.artifact_name)
        if not isinstance(raw, Mapping):
            return False
        try:
            authoritative = ArtifactPublicationReceipt.from_dict(raw)
        except Exception:
            return False
        if authoritative.as_dict() != receipt.as_dict():
            return False
        ref = authoritative.ref
        if ref.backend == "filesystem":
            try:
                target = Path(ref.uri).resolve()
                authority_root = Path(self.authority.root).resolve()
                if not _is_relative_to(target, authority_root) or not target.is_file():
                    return False
                if ref.size_bytes and target.stat().st_size != int(ref.size_bytes):
                    return False
                if not ref.checksum or _sha256(target) != ref.checksum:
                    return False
            except OSError:
                return False
        return True

    def abort(
        self,
        transaction_id: str,
        refs: Mapping[str, DataRef],
        *,
        reason: str,
        artifact_names: tuple[str, ...] = (),
    ) -> None:
        transaction = str(transaction_id or "").strip()
        if not transaction:
            return
        if self._commit_path(transaction).exists():
            raise ArtifactPublicationError(
                f"cannot abort committed Artifact transaction '{transaction}'"
            )
        tombstone = {
            "schema": self._LEDGER_SCHEMA,
            "status": "aborted",
            "project_run_id": self.project_run_id,
            "case_run_id": self.case_run_id,
            "transaction_id": transaction,
            "reason": str(reason or "aborted"),
            "aborted_at": time.time(),
            "artifact_names": sorted(
                set(str(name) for name in dict(refs or {})).union(
                    str(name) for name in artifact_names
                )
            ),
        }
        tombstone_path = self._tombstone_path(transaction)
        tombstone_path.parent.mkdir(parents=True, exist_ok=True)
        if not tombstone_path.exists():
            self._write_json_create(tombstone_path, tombstone)
        all_names = set(str(name) for name in dict(refs or {}))
        all_names.update(str(name) for name in artifact_names)
        for name in all_names:
            reservation_path = self._reservation_path(str(name))
            reservation = self._read_json(reservation_path)
            if reservation.get("transaction_id") == transaction:
                try:
                    reservation_path.unlink()
                except FileNotFoundError:
                    pass
            ref = dict(refs or {}).get(name)
            if ref is not None:
                self._delete_provisional_ref(ref)

    def scavenge_stale_reservations(
        self,
        *,
        current_lease_id: str = "",
        current_fencing_token: int = 0,
        stale_after_seconds: float = 86400.0,
        now: float | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Abort orphaned transactions without touching a current lease owner."""

        names_root = self.root / "names" / _safe_segment(self.case_run_id)
        if not names_root.is_dir():
            return ()
        observed_at = float(time.time() if now is None else now)
        max_age = max(1.0, float(stale_after_seconds))
        current_lease = str(current_lease_id or "")
        current_token = max(0, int(current_fencing_token or 0))
        grouped: dict[str, list[tuple[Path, dict[str, Any], bool]]] = {}
        for path in names_root.glob("*.json"):
            record = self._read_json(path)
            if str(record.get("status", "")) != "reserved":
                continue
            transaction = str(record.get("transaction_id", "") or "")
            if not transaction:
                continue
            lease_id = str(record.get("lease_id", "") or "")
            token = max(0, int(record.get("fencing_token", 0) or 0))
            age = max(0.0, observed_at - float(record.get("reserved_at", 0.0) or 0.0))
            fenced = bool(
                current_lease
                and lease_id
                and (lease_id != current_lease or token < current_token)
            )
            same_current_owner = bool(
                current_lease and lease_id == current_lease and token == current_token
            )
            stale = fenced or (not same_current_owner and age >= max_age)
            grouped.setdefault(transaction, []).append((path, record, stale))

        evidence: list[dict[str, Any]] = []
        for transaction, entries in grouped.items():
            if not entries or not all(item[2] for item in entries):
                continue
            if self._commit_path(transaction).exists():
                continue
            refs: dict[str, DataRef] = {}
            names: list[str] = []
            malformed = False
            for _path, record, _stale in entries:
                name = str(record.get("artifact_name", "") or "")
                if not name:
                    malformed = True
                    break
                names.append(name)
                raw_ref = record.get("provisional_ref")
                if isinstance(raw_ref, Mapping):
                    try:
                        refs[name] = DataRef.from_dict(raw_ref)
                    except Exception:
                        malformed = True
                        break
            if malformed:
                evidence.append(
                    {
                        "transaction_id": transaction,
                        "status": "invalid_reservation",
                    }
                )
                continue
            self.abort(
                transaction,
                refs,
                reason="stale_reservation_scavenged",
                artifact_names=tuple(names),
            )
            evidence.append(
                {
                    "transaction_id": transaction,
                    "status": "scavenged",
                    "artifact_names": sorted(names),
                }
            )
        return tuple(evidence)

    def _delete_provisional_ref(self, ref: DataRef) -> None:
        if not isinstance(ref, DataRef) or ref.backend != "filesystem":
            return
        target = Path(ref.uri).resolve()
        case_root = (
            Path(self.authority.root).resolve()
            / _safe_segment(self.authority.namespace)
            / _safe_segment(self.project_run_id)
            / _safe_segment(self.case_run_id)
        ).resolve()
        if not _is_relative_to(target, case_root):
            raise ArtifactPublicationError(
                f"provisional Artifact '{target}' escapes authorized Case directory"
            )
        try:
            target.unlink()
        except FileNotFoundError:
            pass

    def _reservation_path(self, artifact_name: str) -> Path:
        exact = hashlib.sha256(str(artifact_name).encode("utf-8")).hexdigest()[:16]
        return (
            self.root
            / "names"
            / _safe_segment(self.case_run_id)
            / f"{_safe_segment(artifact_name)}--{exact}.json"
        )

    def _commit_path(self, transaction_id: str) -> Path:
        return self.root / "commits" / f"{_safe_segment(transaction_id)}.json"

    def _tombstone_path(self, transaction_id: str) -> Path:
        return self.root / "tombstones" / f"{_safe_segment(transaction_id)}.json"

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except Exception as exc:
            raise ArtifactPublicationError(
                f"invalid Artifact publication ledger record '{path}'"
            ) from exc
        return dict(payload) if isinstance(payload, Mapping) else {}

    @staticmethod
    def _write_json_create(path: Path, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        with path.open("xb") as writer:
            writer.write(encoded)
            writer.flush()
            os.fsync(writer.fileno())

    @staticmethod
    def _write_json_replace(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            encoded = json.dumps(
                dict(payload),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            with temporary.open("xb") as writer:
                writer.write(encoded)
                writer.flush()
                os.fsync(writer.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


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
        reservation_stale_after_seconds: float = 86400.0,
    ) -> None:
        self.authority = authority
        self.project_run_id = str(project_run_id or "project-run")
        self.case_run_id = str(case_run_id or "case-run")
        self.lease = dict(lease or {})
        self.lease_authority = dict(lease_authority or {})
        self.registry = registry or ArtifactSerializerRegistry()
        self.store: ArtifactStore = FilesystemArtifactStore(authority.root)
        self.publication_ledger = FilesystemArtifactPublicationLedger(
            authority,
            project_run_id=self.project_run_id,
            case_run_id=self.case_run_id,
        )
        self._fence = _build_fence_validator(
            self.lease,
            self.lease_authority,
            redis_client=redis_client,
        )
        self.reservation_recovery = self.publication_ledger.scavenge_stale_reservations(
            current_lease_id=str(self.lease.get("lease_id", "")),
            current_fencing_token=int(self.lease.get("fencing_token", 0) or 0),
            stale_after_seconds=reservation_stale_after_seconds,
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
            reservation_stale_after_seconds=float(
                metadata.get("artifact_reservation_stale_after_seconds", 86400.0)
                or 86400.0
            ),
        )

    def register_serializer(
        self,
        serializer: ArtifactSerializer,
        *,
        replace: bool = False,
    ) -> None:
        self.registry.register(serializer, replace=replace)

    def reserve_publication(self, name: str, transaction_id: str) -> None:
        if self._fence is not None:
            self._fence()
        self.publication_ledger.reserve(
            name,
            transaction_id,
            lease_id=str(self.lease.get("lease_id", "")),
            fencing_token=int(self.lease.get("fencing_token", 0) or 0),
        )

    def record_provisional_publication(
        self,
        name: str,
        transaction_id: str,
        ref: DataRef,
    ) -> None:
        if self._fence is not None:
            self._fence()
        self.publication_ledger.attach_provisional(name, transaction_id, ref)

    def commit_publications(
        self,
        transaction_id: str,
        refs: Mapping[str, DataRef],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, ArtifactPublicationReceipt]:
        if self._fence is not None:
            self._fence()
        return self.publication_ledger.commit(
            transaction_id,
            refs,
            lease_id=str(self.lease.get("lease_id", "")),
            fencing_token=int(self.lease.get("fencing_token", 0) or 0),
            metadata=metadata,
        )

    def abort_publications(
        self,
        transaction_id: str,
        refs: Mapping[str, DataRef],
        *,
        reason: str,
        artifact_names: tuple[str, ...] = (),
    ) -> None:
        self.publication_ledger.abort(
            transaction_id,
            refs,
            reason=reason,
            artifact_names=artifact_names,
        )

    def verify_publication(self, receipt: ArtifactPublicationReceipt) -> bool:
        return self.publication_ledger.verify(receipt)

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
    "ARTIFACT_BINDING_SCHEMA_VERSION",
    "ARTIFACT_AUTHORITY_SCHEMA_VERSION",
    "ARTIFACT_PUBLICATION_RECEIPT_SCHEMA_VERSION",
    "ArtifactBinding",
    "ArtifactAuthority",
    "ArtifactFenceError",
    "ArtifactPublicationError",
    "ArtifactPublicationReceipt",
    "ArtifactPublisher",
    "ArtifactSerializer",
    "ArtifactSerializerRegistry",
    "ArtifactStore",
    "FilesystemArtifactStore",
]
