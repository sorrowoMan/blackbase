"""Formal, self-validating envelope for composable runtime projections."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import Any, Callable, Optional, Sequence


RUNTIME_PROJECTION_SCHEMA = "blackbase.runtime_projection.v1"
RUNTIME_PROJECTION_STATUSES = frozenset({"ok", "degraded", "error"})
RUNTIME_PROJECTION_MAX_COMPONENTS = 2_147_483_647
RUNTIME_PROJECTION_MAX_ISSUE_SAMPLES = 16
RUNTIME_PROJECTION_AUDIT_MAX_BYTES = 4_096
RUNTIME_PROJECTION_COMPONENT_MAX_BYTES = 128
RUNTIME_PROJECTION_REASON_MAX_BYTES = 64
RUNTIME_PROJECTION_ERROR_TYPE_MAX_BYTES = 64
RUNTIME_PROJECTION_MESSAGE_MAX_BYTES = 256


def _update_digest(digest: Any, *parts: Any) -> None:
    for part in parts:
        payload = str(part).encode("utf-8", errors="replace")
        digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)


def _bounded_text(value: str, *, max_bytes: int) -> tuple[str, bool]:
    if not isinstance(value, str):
        raise TypeError("runtime projection issue fields must be strings")
    payload = value.encode("utf-8", errors="replace")
    limit = max(0, int(max_bytes))
    if len(payload) <= limit:
        return value, False
    full_digest = hashlib.sha256(payload).hexdigest()
    suffix = f"…#{full_digest[:16]}".encode("utf-8")
    if limit <= len(suffix):
        bounded = suffix[-limit:] if limit else b""
    else:
        prefix = payload[: limit - len(suffix)].decode("utf-8", errors="ignore")
        bounded = prefix.encode("utf-8") + suffix
    return bounded.decode("utf-8", errors="ignore"), True


def _validated_count(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < 0 or normalized > RUNTIME_PROJECTION_MAX_COMPONENTS:
        raise ValueError(
            f"{name} must be between 0 and {RUNTIME_PROJECTION_MAX_COMPONENTS}"
        )
    return normalized


def _validated_digest(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{name} must be a 64-character SHA-256 digest")
    return normalized


@dataclass(frozen=True)
class RuntimeProjectionIssue:
    """One bounded, typed component-projection issue."""

    component: str
    reason: str
    error_type: str = ""
    message: str = ""
    cause_digest: str = ""
    digest: str = field(init=False)
    message_hash: str = field(init=False)
    truncated: bool = field(init=False)

    def __post_init__(self) -> None:
        raw_values = (self.component, self.reason, self.error_type, self.message)
        if not all(isinstance(value, str) for value in raw_values):
            raise TypeError("runtime projection issue fields must be strings")
        if not isinstance(self.cause_digest, str):
            raise TypeError("cause_digest must be a string")
        cause_digest = (
            _validated_digest(self.cause_digest, name="cause_digest")
            if self.cause_digest
            else ""
        )
        digest = hashlib.sha256()
        _update_digest(digest, RUNTIME_PROJECTION_SCHEMA, *raw_values)
        if cause_digest:
            _update_digest(digest, "cause_digest", cause_digest)
        component, component_truncated = _bounded_text(
            self.component,
            max_bytes=RUNTIME_PROJECTION_COMPONENT_MAX_BYTES,
        )
        reason, reason_truncated = _bounded_text(
            self.reason,
            max_bytes=RUNTIME_PROJECTION_REASON_MAX_BYTES,
        )
        error_type, error_type_truncated = _bounded_text(
            self.error_type,
            max_bytes=RUNTIME_PROJECTION_ERROR_TYPE_MAX_BYTES,
        )
        message, message_truncated = _bounded_text(
            self.message,
            max_bytes=RUNTIME_PROJECTION_MESSAGE_MAX_BYTES,
        )
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "error_type", error_type)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "cause_digest", cause_digest)
        object.__setattr__(self, "digest", digest.hexdigest())
        object.__setattr__(
            self,
            "message_hash",
            hashlib.sha256(raw_values[3].encode("utf-8", errors="replace")).hexdigest(),
        )
        object.__setattr__(
            self,
            "truncated",
            bool(
                component_truncated
                or reason_truncated
                or error_type_truncated
                or message_truncated
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "component": self.component,
            "reason": self.reason,
            "digest": self.digest,
            "truncated": self.truncated,
        }
        if self.error_type:
            out["error_type"] = self.error_type
        if self.message:
            out["message"] = self.message
            out["message_hash"] = self.message_hash
        if self.cause_digest:
            out["cause_digest"] = self.cause_digest
        return out


def _issues_digest(issues: Sequence[RuntimeProjectionIssue]) -> str:
    digest = hashlib.sha256()
    for issue in issues:
        _update_digest(digest, issue.digest)
    return digest.hexdigest()


class RuntimeProjectionIssueAccumulator:
    """Bounded sampler plus full stable digest for component issues."""

    def __init__(self, *, sample_limit: int = RUNTIME_PROJECTION_MAX_ISSUE_SAMPLES) -> None:
        self._sample_limit = _validated_count(sample_limit, name="sample_limit")
        if self._sample_limit > RUNTIME_PROJECTION_MAX_ISSUE_SAMPLES:
            raise ValueError(
                "sample_limit cannot exceed "
                f"{RUNTIME_PROJECTION_MAX_ISSUE_SAMPLES}"
            )
        self._count = 0
        self._samples: list[RuntimeProjectionIssue] = []
        self._digest = hashlib.sha256()

    def add(self, issue: RuntimeProjectionIssue) -> None:
        if not isinstance(issue, RuntimeProjectionIssue):
            raise TypeError("runtime projection issues must use RuntimeProjectionIssue")
        if self._count >= RUNTIME_PROJECTION_MAX_COMPONENTS:
            raise OverflowError("runtime projection issue count exceeds protocol limit")
        self._count += 1
        _update_digest(self._digest, issue.digest)
        if len(self._samples) < self._sample_limit:
            self._samples.append(issue)

    @property
    def count(self) -> int:
        return self._count

    @property
    def samples(self) -> tuple[RuntimeProjectionIssue, ...]:
        return tuple(self._samples)

    @property
    def digest(self) -> str:
        return self._digest.hexdigest()

    @property
    def sample_limit(self) -> int:
        return self._sample_limit

    @property
    def truncated(self) -> bool:
        return self._count > len(self._samples)


@dataclass(frozen=True)
class RuntimeContextProjection(Mapping[str, Any]):
    """A mapping plus bounded, internally consistent component-health evidence."""

    fields: Mapping[str, Any] = field(default_factory=dict)
    field_sources: Mapping[str, str] = field(default_factory=dict)
    status: str = "ok"
    component_count: int = 0
    successful_component_count: int = 0
    degraded_component_count: int = 0
    failed_component_count: int = 0
    invalid_component_count: int = 0
    unavailable_component_count: int = 0
    issue_samples: tuple[RuntimeProjectionIssue, ...] = ()
    issue_sample_limit: int = RUNTIME_PROJECTION_MAX_ISSUE_SAMPLES
    issue_count: Optional[int] = None
    issue_digest: str = ""
    audit_truncated: bool = False
    audit_digest: str = field(init=False)
    field_source_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.fields, Mapping):
            raise TypeError("runtime projection fields must be a Mapping")
        normalized_fields = dict(self.fields)
        if not isinstance(self.field_sources, Mapping):
            raise TypeError("runtime projection field_sources must be a Mapping")
        normalized_sources: dict[str, str] = {}
        for key, source in self.field_sources.items():
            if not isinstance(key, str) or not isinstance(source, str):
                raise TypeError("runtime projection field sources must be strings")
            if key not in normalized_fields:
                raise ValueError(
                    "runtime projection field_sources cannot name an absent field: "
                    f"{key!r}"
                )
            bounded_source, _ = _bounded_text(
                source,
                max_bytes=RUNTIME_PROJECTION_COMPONENT_MAX_BYTES,
            )
            normalized_sources[key] = bounded_source

        if not isinstance(self.status, str):
            raise TypeError("runtime projection status must be a string")
        status = self.status.strip().lower() or "ok"
        if status not in RUNTIME_PROJECTION_STATUSES:
            raise ValueError(
                "runtime projection status must be one of "
                f"{sorted(RUNTIME_PROJECTION_STATUSES)}, got {status!r}"
            )

        count_names = (
            "component_count",
            "successful_component_count",
            "degraded_component_count",
            "failed_component_count",
            "invalid_component_count",
            "unavailable_component_count",
        )
        counts = {
            name: _validated_count(getattr(self, name), name=name)
            for name in count_names
        }
        classified_count = sum(
            counts[name] for name in count_names if name != "component_count"
        )
        if classified_count != counts["component_count"]:
            raise ValueError(
                "runtime projection component classifications must sum to "
                "component_count"
            )

        unhealthy_count = (
            counts["degraded_component_count"]
            + counts["failed_component_count"]
            + counts["invalid_component_count"]
        )
        if unhealthy_count == 0:
            expected_status = "ok"
        elif (
            counts["successful_component_count"] > 0
            or counts["degraded_component_count"] > 0
        ):
            expected_status = "degraded"
        else:
            expected_status = "error"
        if status != expected_status:
            raise ValueError(
                "runtime projection status is inconsistent with component counts: "
                f"expected {expected_status!r}, got {status!r}"
            )

        sample_limit = _validated_count(
            self.issue_sample_limit,
            name="issue_sample_limit",
        )
        if sample_limit > RUNTIME_PROJECTION_MAX_ISSUE_SAMPLES:
            raise ValueError(
                "issue_sample_limit cannot exceed "
                f"{RUNTIME_PROJECTION_MAX_ISSUE_SAMPLES}"
            )
        if not isinstance(self.issue_samples, tuple):
            raise TypeError("issue_samples must be a tuple")
        if len(self.issue_samples) > RUNTIME_PROJECTION_MAX_ISSUE_SAMPLES:
            raise ValueError(
                "issue_samples cannot exceed "
                f"{RUNTIME_PROJECTION_MAX_ISSUE_SAMPLES}; use issue_count and "
                "issue_digest for omitted evidence"
            )
        raw_samples = self.issue_samples
        if not all(isinstance(issue, RuntimeProjectionIssue) for issue in raw_samples):
            raise TypeError("issue_samples must contain RuntimeProjectionIssue values")
        if len(raw_samples) > sample_limit:
            raise ValueError("issue_samples cannot exceed issue_sample_limit")

        issue_count = (
            len(raw_samples)
            if self.issue_count is None
            else _validated_count(self.issue_count, name="issue_count")
        )
        if issue_count != unhealthy_count:
            raise ValueError(
                "issue_count must equal degraded + failed + invalid component counts"
            )
        if issue_count < len(raw_samples):
            raise ValueError("issue_count cannot be smaller than issue_samples")

        computed_issue_digest = _issues_digest(raw_samples)
        if issue_count == len(raw_samples):
            if self.issue_digest:
                issue_digest = _validated_digest(
                    self.issue_digest,
                    name="issue_digest",
                )
                if issue_digest != computed_issue_digest:
                    raise ValueError("issue_digest does not match issue_samples")
            else:
                issue_digest = computed_issue_digest
        else:
            issue_digest = _validated_digest(
                self.issue_digest,
                name="issue_digest",
            )

        samples = raw_samples[:sample_limit]
        for name, value in counts.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "fields", MappingProxyType(normalized_fields))
        object.__setattr__(
            self,
            "field_sources",
            MappingProxyType(normalized_sources),
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "issue_sample_limit", sample_limit)
        object.__setattr__(self, "issue_samples", samples)
        object.__setattr__(self, "issue_count", issue_count)
        object.__setattr__(self, "issue_digest", issue_digest)
        if not isinstance(self.audit_truncated, bool):
            raise TypeError("audit_truncated must be a bool")
        object.__setattr__(
            self,
            "audit_truncated",
            bool(self.audit_truncated or issue_count > len(samples)),
        )

        audit_digest = hashlib.sha256()
        _update_digest(
            audit_digest,
            RUNTIME_PROJECTION_SCHEMA,
            status,
            *(counts[name] for name in count_names),
            issue_count,
            issue_digest,
        )
        object.__setattr__(self, "audit_digest", audit_digest.hexdigest())

        field_source_digest = hashlib.sha256()
        _update_digest(
            field_source_digest,
            "blackbase.runtime_projection.field_sources.v1",
        )
        for key, source in sorted(normalized_sources.items()):
            _update_digest(field_source_digest, key, source)
        object.__setattr__(
            self,
            "field_source_digest",
            field_source_digest.hexdigest(),
        )

    def __getitem__(self, key: str) -> Any:
        return self.fields[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    @staticmethod
    def _audit_size(payload: Mapping[str, Any]) -> int:
        return len(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )

    def as_audit(self) -> dict[str, Any]:
        """Return component evidence within the protocol's hard byte budget."""

        audit: dict[str, Any] = {
            "schema": RUNTIME_PROJECTION_SCHEMA,
            "status": self.status,
            "component_count": self.component_count,
            "successful_component_count": self.successful_component_count,
            "degraded_component_count": self.degraded_component_count,
            "failed_component_count": self.failed_component_count,
            "invalid_component_count": self.invalid_component_count,
            "unavailable_component_count": self.unavailable_component_count,
            "issue_count": self.issue_count,
            "issue_digest": self.issue_digest,
            "issue_samples": [],
            "issue_sample_count": 0,
            "issue_sample_limit": self.issue_sample_limit,
            "field_source_count": len(self.field_sources),
            "field_source_digest": self.field_source_digest,
            "audit_max_bytes": RUNTIME_PROJECTION_AUDIT_MAX_BYTES,
            "audit_truncated": bool(self.audit_truncated),
            "audit_digest": self.audit_digest,
        }
        for issue in self.issue_samples:
            candidate = dict(audit)
            candidate["issue_samples"] = [
                *audit["issue_samples"],
                issue.as_dict(),
            ]
            candidate["issue_sample_count"] = len(candidate["issue_samples"])
            candidate["audit_truncated"] = bool(
                self.audit_truncated
                or self.issue_count > candidate["issue_sample_count"]
            )
            if self._audit_size(candidate) > RUNTIME_PROJECTION_AUDIT_MAX_BYTES:
                audit["audit_truncated"] = True
                break
            audit = candidate
        audit["issue_sample_count"] = len(audit["issue_samples"])
        audit["audit_truncated"] = bool(
            audit["audit_truncated"]
            or self.issue_count > audit["issue_sample_count"]
        )
        while (
            self._audit_size(audit) > RUNTIME_PROJECTION_AUDIT_MAX_BYTES
            and audit["issue_samples"]
        ):
            audit["issue_samples"].pop()
            audit["issue_sample_count"] = len(audit["issue_samples"])
            audit["audit_truncated"] = True
        if self._audit_size(audit) > RUNTIME_PROJECTION_AUDIT_MAX_BYTES:
            raise RuntimeError("runtime projection base audit exceeds its hard budget")
        return audit


@dataclass(frozen=True)
class RuntimeProjectionComponent:
    """One explicitly declared child in a runtime-projection composition."""

    component: str
    projector: Optional[Callable[[Any], Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.component, str):
            raise TypeError("runtime projection component must be a string")
        component = self.component.strip()
        if not component:
            raise ValueError("runtime projection component cannot be empty")
        component, _ = _bounded_text(
            component,
            max_bytes=RUNTIME_PROJECTION_COMPONENT_MAX_BYTES,
        )
        if self.projector is not None and not callable(self.projector):
            raise TypeError("runtime projection component projector must be callable")
        object.__setattr__(self, "component", component)


@dataclass(frozen=True)
class RuntimeProjectionAggregation:
    """Compatibility view over one projection envelope and its writer evidence."""

    projection: RuntimeContextProjection

    def __post_init__(self) -> None:
        if not isinstance(self.projection, RuntimeContextProjection):
            raise TypeError("projection must be a RuntimeContextProjection")

    @property
    def field_sources(self) -> Mapping[str, str]:
        return self.projection.field_sources


def _projection_issue_from_error(
    component: str,
    reason: str,
    error: BaseException,
    *,
    cause_digest: str = "",
) -> RuntimeProjectionIssue:
    try:
        message = str(error)
    except Exception as stringify_error:
        message = f"<unprintable error: {type(stringify_error).__name__}>"
    return RuntimeProjectionIssue(
        component=component,
        reason=reason,
        error_type=type(error).__name__,
        message=message,
        cause_digest=cause_digest,
    )


def aggregate_runtime_projections(
    control: Any,
    components: Sequence[RuntimeProjectionComponent],
    *,
    fields: Optional[Mapping[Any, Any]] = None,
    field_sources: Optional[Mapping[str, str]] = None,
) -> RuntimeProjectionAggregation:
    """Invoke declared child projectors once and aggregate bounded health evidence.

    The substrate owns invocation, validation, status derivation, nested causal
    digest propagation, and first-writer field merging. Semantic frameworks own
    the child topology and decide which components are active.
    """

    if isinstance(components, (str, bytes)) or not isinstance(components, Sequence):
        raise TypeError("components must be a sequence of RuntimeProjectionComponent")
    if not all(isinstance(item, RuntimeProjectionComponent) for item in components):
        raise TypeError("components must contain RuntimeProjectionComponent values")
    if fields is not None and not isinstance(fields, Mapping):
        raise TypeError("fields must be a Mapping")
    if field_sources is not None and not isinstance(field_sources, Mapping):
        raise TypeError("field_sources must be a Mapping")

    merged: dict[str, Any] = {}
    writers: dict[str, str] = {}
    declared_sources = dict(field_sources) if field_sources is not None else {}
    initial_fields = fields.items() if fields is not None else ()
    for key, value in initial_fields:
        if key is None or value is None:
            continue
        key_str = str(key)
        merged[key_str] = value
        source = declared_sources.get(key_str)
        if source is not None:
            if not isinstance(source, str):
                raise TypeError("runtime projection field sources must be strings")
            writers[key_str] = source

    successful_count = 0
    degraded_count = 0
    failed_count = 0
    invalid_count = 0
    unavailable_count = 0
    issues = RuntimeProjectionIssueAccumulator()

    for component in components:
        projector = component.projector
        if projector is None:
            unavailable_count += 1
            continue

        try:
            child_projection = projector(control)
        except Exception as exc:
            failed_count += 1
            issues.add(
                _projection_issue_from_error(component.component, "error", exc)
            )
            continue

        if not isinstance(child_projection, Mapping):
            invalid_count += 1
            issues.add(
                _projection_issue_from_error(
                    component.component,
                    "invalid_result",
                    TypeError(
                        "child runtime projection must return a Mapping, got "
                        f"{type(child_projection).__name__}"
                    ),
                )
            )
            continue

        cause_digest = (
            child_projection.audit_digest
            if isinstance(child_projection, RuntimeContextProjection)
            else ""
        )
        child_sources = (
            child_projection.field_sources
            if isinstance(child_projection, RuntimeContextProjection)
            else {}
        )
        try:
            child_fields = [
                (str(key), value)
                for key, value in child_projection.items()
                if key is not None and value is not None
            ]
        except Exception as exc:
            failed_count += 1
            issues.add(
                _projection_issue_from_error(
                    component.component,
                    "error",
                    exc,
                    cause_digest=cause_digest,
                )
            )
            continue

        if isinstance(child_projection, RuntimeContextProjection):
            if child_projection.status == "ok":
                successful_count += 1
            elif child_projection.status == "degraded":
                degraded_count += 1
                issues.add(
                    RuntimeProjectionIssue(
                        component=component.component,
                        reason="nested_degraded",
                        cause_digest=cause_digest,
                    )
                )
            else:
                failed_count += 1
                issues.add(
                    RuntimeProjectionIssue(
                        component=component.component,
                        reason="nested_error",
                        cause_digest=cause_digest,
                    )
                )
        else:
            successful_count += 1

        for key, value in child_fields:
            if key in merged:
                continue
            merged[key] = value
            writers[key] = child_sources.get(key, component.component)

    unhealthy_count = degraded_count + failed_count + invalid_count
    if unhealthy_count == 0:
        status = "ok"
    elif successful_count > 0 or degraded_count > 0:
        status = "degraded"
    else:
        status = "error"

    projection = RuntimeContextProjection(
        fields=merged,
        field_sources=writers,
        status=status,
        component_count=len(components),
        successful_component_count=successful_count,
        degraded_component_count=degraded_count,
        failed_component_count=failed_count,
        invalid_component_count=invalid_count,
        unavailable_component_count=unavailable_count,
        issue_samples=issues.samples,
        issue_sample_limit=issues.sample_limit,
        issue_count=issues.count,
        issue_digest=issues.digest,
        audit_truncated=issues.truncated,
    )
    return RuntimeProjectionAggregation(projection=projection)


__all__ = [
    "RUNTIME_PROJECTION_SCHEMA",
    "RUNTIME_PROJECTION_STATUSES",
    "RUNTIME_PROJECTION_MAX_COMPONENTS",
    "RUNTIME_PROJECTION_MAX_ISSUE_SAMPLES",
    "RUNTIME_PROJECTION_AUDIT_MAX_BYTES",
    "RUNTIME_PROJECTION_COMPONENT_MAX_BYTES",
    "RUNTIME_PROJECTION_REASON_MAX_BYTES",
    "RUNTIME_PROJECTION_ERROR_TYPE_MAX_BYTES",
    "RUNTIME_PROJECTION_MESSAGE_MAX_BYTES",
    "RuntimeProjectionIssue",
    "RuntimeProjectionIssueAccumulator",
    "RuntimeContextProjection",
    "RuntimeProjectionComponent",
    "RuntimeProjectionAggregation",
    "aggregate_runtime_projections",
]
