"""Versioned wire envelope for semantic evaluation evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from blackbase.wire import freeze_wire_mapping, thaw_wire_mapping


EVALUATION_EVENT_SCHEMA_V1 = "blackbase.evaluation_event/v1"
EVALUATION_DISPOSITION_SCHEMA_V1 = "blackbase.evaluation_disposition/v1"
EVALUATION_DISPOSITION_VERIFICATION_SCHEMA_V1 = (
    "blackbase.evaluation_disposition_verification/v1"
)
EVALUATION_DISPOSITION_VERIFICATION_SCHEMA_V2 = (
    "blackbase.evaluation_disposition_verification/v2"
)


@dataclass(frozen=True)
class EvaluationEventEnvelope:
    """Transport-safe evidence for one completed evaluation batch.

    BlackBase owns only the envelope.  Semantic frameworks own the candidate,
    feedback, and provenance codecs named by the envelope.
    """

    event_id: str
    candidate_codec: str
    candidate_payload: Mapping[str, Any]
    feedback_codec: str
    feedback_payload: Mapping[str, Any]
    provenance: Sequence[Mapping[str, Any]] = ()
    identity: Mapping[str, Any] = field(default_factory=dict)
    evaluation_count: int = 0
    semantic_complete: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        event_id = str(self.event_id or "").strip()
        candidate_codec = str(self.candidate_codec or "").strip()
        feedback_codec = str(self.feedback_codec or "").strip()
        if not event_id:
            raise ValueError("evaluation event_id must not be empty")
        if not candidate_codec or not feedback_codec:
            raise ValueError("evaluation event candidate/feedback codecs are required")
        evaluation_count = int(self.evaluation_count)
        if evaluation_count < 0:
            raise ValueError("evaluation_count must be non-negative")
        provenance = tuple(
            freeze_wire_mapping(item, path=f"evaluation_event.provenance[{index}]")
            for index, item in enumerate(tuple(self.provenance or ()))
        )
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "candidate_codec", candidate_codec)
        object.__setattr__(self, "feedback_codec", feedback_codec)
        object.__setattr__(self, "evaluation_count", evaluation_count)
        object.__setattr__(
            self,
            "candidate_payload",
            freeze_wire_mapping(
                self.candidate_payload,
                path="evaluation_event.candidate_payload",
            ),
        )
        object.__setattr__(
            self,
            "feedback_payload",
            freeze_wire_mapping(
                self.feedback_payload,
                path="evaluation_event.feedback_payload",
            ),
        )
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(
            self,
            "identity",
            freeze_wire_mapping(self.identity, path="evaluation_event.identity"),
        )
        object.__setattr__(self, "semantic_complete", bool(self.semantic_complete))
        object.__setattr__(
            self,
            "metadata",
            freeze_wire_mapping(self.metadata, path="evaluation_event.metadata"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": EVALUATION_EVENT_SCHEMA_V1,
            "event_id": self.event_id,
            "candidate_codec": self.candidate_codec,
            "candidate_payload": thaw_wire_mapping(self.candidate_payload),
            "feedback_codec": self.feedback_codec,
            "feedback_payload": thaw_wire_mapping(self.feedback_payload),
            "provenance": [thaw_wire_mapping(item) for item in self.provenance],
            "identity": thaw_wire_mapping(self.identity),
            "evaluation_count": self.evaluation_count,
            "semantic_complete": self.semantic_complete,
            "metadata": thaw_wire_mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvaluationEventEnvelope":
        data = dict(payload or {})
        schema = str(data.get("schema", "") or "")
        if schema != EVALUATION_EVENT_SCHEMA_V1:
            raise ValueError(
                f"unsupported evaluation event schema: {schema or '<missing>'}"
            )
        return cls(
            event_id=str(data.get("event_id", "")),
            candidate_codec=str(data.get("candidate_codec", "")),
            candidate_payload=dict(data.get("candidate_payload", {}) or {}),
            feedback_codec=str(data.get("feedback_codec", "")),
            feedback_payload=dict(data.get("feedback_payload", {}) or {}),
            provenance=tuple(data.get("provenance", ()) or ()),
            identity=dict(data.get("identity", {}) or {}),
            evaluation_count=int(data.get("evaluation_count", 0) or 0),
            semantic_complete=bool(data.get("semantic_complete", True)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class EvaluationDispositionEnvelope:
    """Durable edge from one Evaluation Event to its control disposition.

    ``authority_snapshot_key`` identifies the state made authoritative by a
    committed disposition, or the unchanged predecessor authority for a
    rejected/failed disposition.  The semantic framework owns the disposition
    codec while BlackBase guarantees a stable, wire-safe evidence envelope.
    """

    event_id: str
    status: str
    disposition_codec: str
    disposition_payload: Mapping[str, Any]
    event_snapshot_key: str = ""
    authority_snapshot_key: str = ""
    identity: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        event_id = str(self.event_id or "").strip()
        status = str(self.status or "").strip().lower()
        codec = str(self.disposition_codec or "").strip()
        event_snapshot_key = str(self.event_snapshot_key or "").strip()
        authority_snapshot_key = str(self.authority_snapshot_key or "").strip()
        if not event_id:
            raise ValueError("evaluation disposition event_id must not be empty")
        if status not in {"committed", "rejected", "failed"}:
            raise ValueError(
                "evaluation disposition status must be committed, rejected, or failed"
            )
        if not codec:
            raise ValueError("evaluation disposition codec is required")
        # Every disposition is an edge *from* one durable Evaluation Event.
        # Rejected and failed attempts are not exempt: without the Event
        # Snapshot key the journal can accept an intent that no verifier can
        # ever bind back to its source evidence.
        if not event_snapshot_key:
            raise ValueError(
                "evaluation disposition requires event_snapshot_key"
            )
        if status == "committed" and not authority_snapshot_key:
            raise ValueError(
                "committed evaluation disposition requires authority_snapshot_key"
            )
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "disposition_codec", codec)
        object.__setattr__(
            self,
            "disposition_payload",
            freeze_wire_mapping(
                self.disposition_payload,
                path="evaluation_disposition.disposition_payload",
            ),
        )
        object.__setattr__(self, "event_snapshot_key", event_snapshot_key)
        object.__setattr__(self, "authority_snapshot_key", authority_snapshot_key)
        object.__setattr__(
            self,
            "identity",
            freeze_wire_mapping(
                self.identity,
                path="evaluation_disposition.identity",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_wire_mapping(
                self.metadata,
                path="evaluation_disposition.metadata",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": EVALUATION_DISPOSITION_SCHEMA_V1,
            "event_id": self.event_id,
            "status": self.status,
            "disposition_codec": self.disposition_codec,
            "disposition_payload": thaw_wire_mapping(self.disposition_payload),
            "event_snapshot_key": self.event_snapshot_key,
            "authority_snapshot_key": self.authority_snapshot_key,
            "identity": thaw_wire_mapping(self.identity),
            "metadata": thaw_wire_mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "EvaluationDispositionEnvelope":
        data = dict(payload or {})
        schema = str(data.get("schema", "") or "")
        if schema != EVALUATION_DISPOSITION_SCHEMA_V1:
            raise ValueError(
                "unsupported evaluation disposition schema: "
                f"{schema or '<missing>'}"
            )
        return cls(
            event_id=str(data.get("event_id", "")),
            status=str(data.get("status", "")),
            disposition_codec=str(data.get("disposition_codec", "")),
            disposition_payload=dict(data.get("disposition_payload", {}) or {}),
            event_snapshot_key=str(data.get("event_snapshot_key", "") or ""),
            authority_snapshot_key=str(
                data.get("authority_snapshot_key", "") or ""
            ),
            identity=dict(data.get("identity", {}) or {}),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def evaluation_disposition_digest(
    envelope: EvaluationDispositionEnvelope | Mapping[str, Any],
) -> str:
    """Return the canonical digest bound into a verification receipt."""

    item = (
        envelope
        if isinstance(envelope, EvaluationDispositionEnvelope)
        else EvaluationDispositionEnvelope.from_dict(envelope)
    )
    encoded = json.dumps(
        item.as_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EvaluationDispositionVerificationReceipt:
    """Transport-safe proof that the disposition destination was inspected.

    The semantic framework performs the Snapshot-specific read and comparison.
    The shared journal verifies that this receipt is bound to the exact intent
    and destination before it permits a terminal transition.
    """

    event_id: str
    event_snapshot_key: str
    event_snapshot_revision: int
    event_snapshot_digest: str
    event_snapshot_schema: str
    destination_snapshot_key: str
    destination_snapshot_revision: int
    destination_snapshot_digest: str
    destination_snapshot_schema: str
    disposition_digest: str
    verifier: str
    verified_at: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        event_id = str(self.event_id or "").strip()
        event_key = str(self.event_snapshot_key or "").strip()
        destination_key = str(self.destination_snapshot_key or "").strip()
        event_revision = int(self.event_snapshot_revision or 0)
        destination_revision = int(self.destination_snapshot_revision or 0)
        event_snapshot_digest = str(self.event_snapshot_digest or "").strip().lower()
        destination_snapshot_digest = str(
            self.destination_snapshot_digest or ""
        ).strip().lower()
        event_snapshot_schema = str(self.event_snapshot_schema or "").strip()
        destination_snapshot_schema = str(
            self.destination_snapshot_schema or ""
        ).strip()
        digest = str(self.disposition_digest or "").strip().lower()
        verifier = str(self.verifier or "").strip()
        verified_at = float(self.verified_at)
        if not event_id or not event_key or not destination_key:
            raise ValueError(
                "evaluation disposition verification requires event and Snapshot keys"
            )
        if event_revision <= 0 or destination_revision <= 0:
            raise ValueError(
                "evaluation disposition verification requires positive Snapshot revisions"
            )
        for value in (event_snapshot_digest, destination_snapshot_digest):
            if not value.startswith("sha256:") or len(value) != 71:
                raise ValueError(
                    "evaluation disposition verification Snapshot digest is invalid"
                )
        if not event_snapshot_schema or not destination_snapshot_schema:
            raise ValueError(
                "evaluation disposition verification Snapshot schema is required"
            )
        if not digest.startswith("sha256:") or len(digest) != 71:
            raise ValueError("evaluation disposition verification digest is invalid")
        if not verifier:
            raise ValueError("evaluation disposition verification verifier is required")
        if verified_at <= 0:
            raise ValueError("evaluation disposition verification time is invalid")
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "event_snapshot_key", event_key)
        object.__setattr__(self, "event_snapshot_revision", event_revision)
        object.__setattr__(self, "event_snapshot_digest", event_snapshot_digest)
        object.__setattr__(self, "event_snapshot_schema", event_snapshot_schema)
        object.__setattr__(self, "destination_snapshot_key", destination_key)
        object.__setattr__(
            self,
            "destination_snapshot_revision",
            destination_revision,
        )
        object.__setattr__(
            self,
            "destination_snapshot_digest",
            destination_snapshot_digest,
        )
        object.__setattr__(
            self,
            "destination_snapshot_schema",
            destination_snapshot_schema,
        )
        object.__setattr__(self, "disposition_digest", digest)
        object.__setattr__(self, "verifier", verifier)
        object.__setattr__(self, "verified_at", verified_at)
        object.__setattr__(
            self,
            "metadata",
            freeze_wire_mapping(
                self.metadata,
                path="evaluation_disposition_verification.metadata",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": EVALUATION_DISPOSITION_VERIFICATION_SCHEMA_V2,
            "event_id": self.event_id,
            "event_snapshot_key": self.event_snapshot_key,
            "event_snapshot_revision": self.event_snapshot_revision,
            "event_snapshot_digest": self.event_snapshot_digest,
            "event_snapshot_schema": self.event_snapshot_schema,
            "destination_snapshot_key": self.destination_snapshot_key,
            "destination_snapshot_revision": self.destination_snapshot_revision,
            "destination_snapshot_digest": self.destination_snapshot_digest,
            "destination_snapshot_schema": self.destination_snapshot_schema,
            "disposition_digest": self.disposition_digest,
            "verifier": self.verifier,
            "verified_at": self.verified_at,
            "metadata": thaw_wire_mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "EvaluationDispositionVerificationReceipt":
        data = dict(payload or {})
        schema = str(data.get("schema", "") or "")
        if schema != EVALUATION_DISPOSITION_VERIFICATION_SCHEMA_V2:
            raise ValueError(
                "unsupported evaluation disposition verification schema: "
                f"{schema or '<missing>'}"
            )
        return cls(
            event_id=str(data.get("event_id", "")),
            event_snapshot_key=str(data.get("event_snapshot_key", "")),
            event_snapshot_revision=int(
                data.get("event_snapshot_revision", 0) or 0
            ),
            event_snapshot_digest=str(data.get("event_snapshot_digest", "")),
            event_snapshot_schema=str(data.get("event_snapshot_schema", "")),
            destination_snapshot_key=str(
                data.get("destination_snapshot_key", "")
            ),
            destination_snapshot_revision=int(
                data.get("destination_snapshot_revision", 0) or 0
            ),
            destination_snapshot_digest=str(
                data.get("destination_snapshot_digest", "")
            ),
            destination_snapshot_schema=str(
                data.get("destination_snapshot_schema", "")
            ),
            disposition_digest=str(data.get("disposition_digest", "")),
            verifier=str(data.get("verifier", "")),
            verified_at=float(data.get("verified_at", 0.0) or 0.0),
            metadata=dict(data.get("metadata", {}) or {}),
        )


__all__ = [
    "EVALUATION_DISPOSITION_SCHEMA_V1",
    "EVALUATION_DISPOSITION_VERIFICATION_SCHEMA_V1",
    "EVALUATION_DISPOSITION_VERIFICATION_SCHEMA_V2",
    "EVALUATION_EVENT_SCHEMA_V1",
    "EvaluationDispositionEnvelope",
    "EvaluationDispositionVerificationReceipt",
    "EvaluationEventEnvelope",
    "evaluation_disposition_digest",
]
