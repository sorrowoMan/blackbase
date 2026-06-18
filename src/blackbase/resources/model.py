"""
Protocol objects for the L0 task/resource orchestration plane.

This module provides core data structures for resource management and task scheduling,
shared between NSGABlack and MLBlack frameworks.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Mapping, Optional, Sequence
from uuid import uuid4


def _now_unix() -> float:
    """Get current Unix timestamp."""
    return float(time.time())


def _as_tuple(value: Any) -> tuple:
    """Convert value to tuple."""
    if value is None:
        return tuple()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _as_float_or_none(value: Any) -> Optional[float]:
    """Convert value to float or None."""
    if value is None:
        return None
    out = float(value)
    if out <= 0:
        return None
    return out


# ============================================================================
# Data Reference
# ============================================================================

@dataclass(frozen=True)
class DataRef:
    """
    Reference to data or artifact payload that should not be inlined.
    
    Immutable data reference for tracking large objects, models, and artifacts
    across the resource orchestration layer.
    """
    
    uri: str                           # Resource locator
    kind: str = "artifact"             # Type: artifact/model/data
    backend: str = "filesystem"       # Backend: filesystem/s3/redis
    media_type: str = ""               # MIME type
    checksum: str = ""                 # Integrity checksum
    size_bytes: Optional[int] = None   # Size in bytes
    metadata: Mapping[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", str(self.uri))
        object.__setattr__(self, "kind", str(self.kind or "artifact"))
        object.__setattr__(self, "backend", str(self.backend or "filesystem"))
        object.__setattr__(self, "media_type", str(self.media_type or ""))
        object.__setattr__(self, "checksum", str(self.checksum or ""))
        size = None if self.size_bytes is None else max(0, int(self.size_bytes))
        object.__setattr__(self, "size_bytes", size)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
    
    @classmethod
    def from_path(cls, path: str | Path, *, kind: str = "artifact", media_type: str = "") -> "DataRef":
        """Create DataRef from file path."""
        p = Path(path)
        size = p.stat().st_size if p.exists() and p.is_file() else None
        return cls(uri=str(p), kind=kind, backend="filesystem", media_type=media_type, size_bytes=size)
    
    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DataRef":
        """Create DataRef from dictionary."""
        return cls(
            uri=str(payload.get("uri", "")),
            kind=str(payload.get("kind", "artifact")),
            backend=str(payload.get("backend", "filesystem")),
            media_type=str(payload.get("media_type", "")),
            checksum=str(payload.get("checksum", "")),
            size_bytes=payload.get("size_bytes"),
            metadata=dict(payload.get("metadata", {}) or {}),
        )
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "uri": str(self.uri),
            "kind": str(self.kind),
            "backend": str(self.backend),
            "media_type": str(self.media_type),
            "checksum": str(self.checksum),
            "size_bytes": self.size_bytes,
            "metadata": dict(self.metadata),
        }


# ============================================================================
# Resource Offer (for worker capabilities)
# ============================================================================

@dataclass(frozen=True)
class ResourceOffer:
    """
    Resource offer from a worker node.
    
    Represents the available resources on a worker that can be allocated to tasks.
    """
    
    threads: int = 1
    gpus: int = 0
    backend: str = "local"
    device_tokens: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        object.__setattr__(self, "threads", max(1, int(self.threads)))
        object.__setattr__(self, "gpus", max(0, int(self.gpus)))
        object.__setattr__(self, "backend", str(self.backend or "local"))
        object.__setattr__(self, "device_tokens", tuple(str(x) for x in _as_tuple(self.device_tokens)))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "threads": int(self.threads),
            "gpus": int(self.gpus),
            "backend": str(self.backend),
            "device_tokens": [str(x) for x in self.device_tokens],
            "metadata": dict(self.metadata),
        }


# ============================================================================
# Resource Requirement
# ============================================================================

@dataclass(frozen=True)
class ResourceRequirement:
    """
    Resource request attached to a task.
    
    Specifies the resources needed to execute a task, including compute,
    memory, and capability requirements.
    """
    
    threads: int = 1
    gpus: int = 0
    resource_backend: str = "local"
    device_tokens: tuple[str, ...] = ()
    memory_mb: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    capabilities: tuple[str, ...] = ()
    timeout_seconds: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        object.__setattr__(self, "threads", max(1, int(self.threads)))
        object.__setattr__(self, "gpus", max(0, int(self.gpus)))
        object.__setattr__(self, "resource_backend", str(self.resource_backend or "local"))
        object.__setattr__(self, "device_tokens", tuple(str(x) for x in _as_tuple(self.device_tokens)))
        object.__setattr__(self, "memory_mb", _as_float_or_none(self.memory_mb))
        object.__setattr__(self, "gpu_memory_mb", _as_float_or_none(self.gpu_memory_mb))
        object.__setattr__(self, "capabilities", tuple(str(x) for x in _as_tuple(self.capabilities) if str(x)))
        object.__setattr__(self, "timeout_seconds", _as_float_or_none(self.timeout_seconds))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
    
    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResourceRequirement":
        """Create from dictionary."""
        return cls(
            threads=int(payload.get("threads", 1) or 1),
            gpus=int(payload.get("gpus", 0) or 0),
            resource_backend=str(payload.get("resource_backend", payload.get("backend", "local"))),
            device_tokens=tuple(payload.get("device_tokens", ()) or ()),
            memory_mb=payload.get("memory_mb"),
            gpu_memory_mb=payload.get("gpu_memory_mb"),
            capabilities=tuple(payload.get("capabilities", ()) or ()),
            timeout_seconds=payload.get("timeout_seconds"),
            metadata=dict(payload.get("metadata", {}) or {}),
        )
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "threads": int(self.threads),
            "gpus": int(self.gpus),
            "resource_backend": str(self.resource_backend),
            "device_tokens": [str(x) for x in self.device_tokens],
            "memory_mb": self.memory_mb,
            "gpu_memory_mb": self.gpu_memory_mb,
            "capabilities": [str(x) for x in self.capabilities],
            "timeout_seconds": self.timeout_seconds,
            "metadata": dict(self.metadata),
        }
    
    def satisfies(self, offer: ResourceOffer) -> bool:
        """Check if this requirement can be satisfied by an offer."""
        if int(self.threads) > int(offer.threads):
            return False
        if int(self.gpus) > int(offer.gpus):
            return False
        if self.device_tokens:
            offered = set(str(x) for x in offer.device_tokens)
            concrete = {x for x in self.device_tokens if ":" in str(x) or str(x) == "mps"}
            if not concrete.issubset(offered):
                return False
        if self.memory_mb is not None:
            available = dict(offer.metadata or {}).get("memory_mb")
            if available is not None and float(self.memory_mb) > float(available):
                return False
        return True


# ============================================================================
# Worker Descriptor
# ============================================================================

@dataclass(frozen=True)
class WorkerDescriptor:
    """
    Execution unit that can receive tasks and owns/declares resources.
    
    Represents a worker node in the resource orchestration cluster.
    """
    
    worker_id: str
    executor_backend: str = "thread"   # thread/process/ray
    resource_backend: str = "local"
    host: str = ""
    capabilities: tuple[str, ...] = ()
    offer: Optional[ResourceOffer] = None
    max_inflight: int = 1
    status: str = "online"
    last_heartbeat_at: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        worker_id = str(self.worker_id or f"worker_{uuid4().hex[:12]}")
        executor_backend = str(self.executor_backend or "thread")
        resource_backend = str(self.resource_backend or "local")
        
        # Coerce offer
        offer = self.offer
        if offer is None:
            offer = ResourceOffer()
        elif isinstance(offer, Mapping):
            offer = ResourceOffer(**dict(offer))
        
        object.__setattr__(self, "worker_id", worker_id)
        object.__setattr__(self, "executor_backend", executor_backend)
        object.__setattr__(self, "resource_backend", resource_backend)
        object.__setattr__(self, "host", str(self.host or socket.gethostname()))
        object.__setattr__(self, "capabilities", tuple(str(x) for x in _as_tuple(self.capabilities) if str(x)))
        object.__setattr__(self, "offer", offer)
        object.__setattr__(self, "max_inflight", max(1, int(self.max_inflight)))
        object.__setattr__(self, "status", str(self.status or "online"))
        object.__setattr__(self, "last_heartbeat_at", float(self.last_heartbeat_at or _now_unix()))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
    
    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkerDescriptor":
        """Create from dictionary."""
        offer = payload.get("offer")
        if offer is not None and not isinstance(offer, ResourceOffer):
            offer = ResourceOffer(**dict(offer))
        return cls(
            worker_id=str(payload.get("worker_id", "")),
            executor_backend=str(payload.get("executor_backend", "thread")),
            resource_backend=str(payload.get("resource_backend", "local")),
            host=str(payload.get("host", "")),
            capabilities=tuple(payload.get("capabilities", ()) or ()),
            offer=offer,
            max_inflight=int(payload.get("max_inflight", 1) or 1),
            status=str(payload.get("status", "online")),
            last_heartbeat_at=float(payload.get("last_heartbeat_at", 0.0) or 0.0),
            metadata=dict(payload.get("metadata", {}) or {}),
        )
    
    def heartbeat(self, *, status: str = "online", at: Optional[float] = None) -> "WorkerDescriptor":
        """Create new descriptor with updated heartbeat."""
        return WorkerDescriptor(
            worker_id=self.worker_id,
            executor_backend=self.executor_backend,
            resource_backend=self.resource_backend,
            host=self.host,
            capabilities=self.capabilities,
            offer=self.offer,
            max_inflight=self.max_inflight,
            status=status,
            last_heartbeat_at=float(at if at is not None else _now_unix()),
            metadata=self.metadata,
        )
    
    def can_run(self, requirement: ResourceRequirement, *, active_count: int = 0) -> bool:
        """Check if worker can run a task with given requirement."""
        if str(self.status).lower() not in {"online", "idle", "ready"}:
            return False
        if int(active_count) >= int(self.max_inflight):
            return False
        if str(requirement.resource_backend).lower() != str(self.resource_backend).lower():
            return False
        caps = set(str(x) for x in self.capabilities)
        if not set(str(x) for x in requirement.capabilities).issubset(caps):
            return False
        return self.offer.satisfies(requirement) if self.offer else False
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "worker_id": str(self.worker_id),
            "executor_backend": str(self.executor_backend),
            "resource_backend": str(self.resource_backend),
            "host": str(self.host),
            "capabilities": [str(x) for x in self.capabilities],
            "offer": self.offer.as_dict() if self.offer else None,
            "max_inflight": int(self.max_inflight),
            "status": str(self.status),
            "last_heartbeat_at": float(self.last_heartbeat_at),
            "metadata": dict(self.metadata),
        }


# ============================================================================
# Task Envelope
# ============================================================================

@dataclass(frozen=True)
class TaskEnvelope:
    """
    Serializable L0 task packet.
    
    Represents a task to be executed by a worker in the resource orchestration layer.
    """
    
    task_id: str
    task_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    requirement: ResourceRequirement = field(default_factory=ResourceRequirement)
    executor_backend: str = "auto"
    input_refs: tuple[DataRef, ...] = ()
    output_refs: tuple[DataRef, ...] = ()
    parent_task_id: Optional[str] = None
    trace_id: str = ""
    namespace: str = ""
    max_retries: int = 0
    created_at: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        req = self.requirement if isinstance(self.requirement, ResourceRequirement) else ResourceRequirement.from_dict(self.requirement)
        object.__setattr__(self, "task_id", str(self.task_id or f"task_{uuid4().hex[:16]}"))
        object.__setattr__(self, "task_type", str(self.task_type or "task"))
        object.__setattr__(self, "payload", dict(self.payload or {}))
        object.__setattr__(self, "requirement", req)
        object.__setattr__(self, "executor_backend", str(self.executor_backend or "auto"))
        object.__setattr__(self, "input_refs", tuple(_coerce_data_ref(x) for x in _as_tuple(self.input_refs)))
        object.__setattr__(self, "output_refs", tuple(_coerce_data_ref(x) for x in _as_tuple(self.output_refs)))
        object.__setattr__(self, "parent_task_id", None if self.parent_task_id is None else str(self.parent_task_id))
        object.__setattr__(self, "trace_id", str(self.trace_id or self.task_id))
        object.__setattr__(self, "namespace", str(self.namespace or "default"))
        object.__setattr__(self, "max_retries", max(0, int(self.max_retries)))
        object.__setattr__(self, "created_at", float(self.created_at or _now_unix()))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
    
    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskEnvelope":
        """Create from dictionary."""
        return cls(
            task_id=str(payload.get("task_id", "")),
            task_type=str(payload.get("task_type", "task")),
            payload=dict(payload.get("payload", {}) or {}),
            requirement=ResourceRequirement.from_dict(dict(payload.get("requirement", {}) or {})),
            executor_backend=str(payload.get("executor_backend", "auto")),
            input_refs=tuple(DataRef.from_dict(x) for x in payload.get("input_refs", ()) or ()),
            output_refs=tuple(DataRef.from_dict(x) for x in payload.get("output_refs", ()) or ()),
            parent_task_id=payload.get("parent_task_id"),
            trace_id=str(payload.get("trace_id", "")),
            namespace=str(payload.get("namespace", "default")),
            max_retries=int(payload.get("max_retries", 0) or 0),
            created_at=float(payload.get("created_at", 0.0) or 0.0),
            metadata=dict(payload.get("metadata", {}) or {}),
        )
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": str(self.task_id),
            "task_type": str(self.task_type),
            "payload": dict(self.payload),
            "requirement": self.requirement.as_dict(),
            "executor_backend": str(self.executor_backend),
            "input_refs": [x.as_dict() for x in self.input_refs],
            "output_refs": [x.as_dict() for x in self.output_refs],
            "parent_task_id": self.parent_task_id,
            "trace_id": str(self.trace_id),
            "namespace": str(self.namespace),
            "max_retries": int(self.max_retries),
            "created_at": float(self.created_at),
            "metadata": dict(self.metadata),
        }


# ============================================================================
# Task Result
# ============================================================================

@dataclass(frozen=True)
class TaskResult:
    """
    Serializable task execution result.
    
    Contains the outcome of task execution including objectives, metrics, and artifacts.
    """
    
    task_id: str
    status: str
    objectives: tuple[float, ...] = ()
    violations: tuple[float, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifact_refs: tuple[DataRef, ...] = ()
    worker_id: str = ""
    lease_id: str = ""
    resource_context: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        started = float(self.started_at or 0.0)
        finished = float(self.finished_at or (_now_unix() if started else 0.0))
        object.__setattr__(self, "task_id", str(self.task_id))
        object.__setattr__(self, "status", str(self.status or "ok"))
        object.__setattr__(self, "objectives", tuple(float(x) for x in _as_tuple(self.objectives)))
        object.__setattr__(self, "violations", tuple(float(x) for x in _as_tuple(self.violations)))
        object.__setattr__(self, "metrics", dict(self.metrics or {}))
        object.__setattr__(self, "artifact_refs", tuple(_coerce_data_ref(x) for x in _as_tuple(self.artifact_refs)))
        object.__setattr__(self, "worker_id", str(self.worker_id or ""))
        object.__setattr__(self, "lease_id", str(self.lease_id or ""))
        object.__setattr__(self, "resource_context", dict(self.resource_context or {}))
        object.__setattr__(self, "error", str(self.error or ""))
        object.__setattr__(self, "started_at", started)
        object.__setattr__(self, "finished_at", finished)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
    
    @property
    def ok(self) -> bool:
        """Check if task completed successfully."""
        return str(self.status).lower() in {"ok", "success", "completed"}
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.started_at <= 0 or self.finished_at <= 0:
            return None
        return max(0.0, float(self.finished_at) - float(self.started_at))
    
    @classmethod
    def success(
        cls,
        *,
        task_id: str,
        objectives: Sequence[float] = (),
        violations: Sequence[float] = (),
        worker_id: str = "",
        lease_id: str = "",
        resource_context: Optional[Mapping[str, Any]] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        artifact_refs: Sequence[DataRef | Mapping[str, Any]] = (),
    ) -> "TaskResult":
        """Create a successful task result."""
        return cls(
            task_id=task_id,
            status="ok",
            objectives=tuple(objectives),
            violations=tuple(violations),
            worker_id=worker_id,
            lease_id=lease_id,
            resource_context=dict(resource_context or {}),
            metrics=dict(metrics or {}),
            artifact_refs=tuple(_coerce_data_ref(x) for x in artifact_refs),
            finished_at=_now_unix(),
        )
    
    @classmethod
    def failure(cls, *, task_id: str, error: str, status: str = "failed", worker_id: str = "") -> "TaskResult":
        """Create a failed task result."""
        return cls(task_id=task_id, status=status, error=error, worker_id=worker_id, finished_at=_now_unix())
    
    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskResult":
        """Create from dictionary."""
        return cls(
            task_id=str(payload.get("task_id", "")),
            status=str(payload.get("status", "ok")),
            objectives=tuple(payload.get("objectives", ()) or ()),
            violations=tuple(payload.get("violations", ()) or ()),
            metrics=dict(payload.get("metrics", {}) or {}),
            artifact_refs=tuple(DataRef.from_dict(x) for x in payload.get("artifact_refs", ()) or ()),
            worker_id=str(payload.get("worker_id", "")),
            lease_id=str(payload.get("lease_id", "")),
            resource_context=dict(payload.get("resource_context", {}) or {}),
            error=str(payload.get("error", "")),
            started_at=float(payload.get("started_at", 0.0) or 0.0),
            finished_at=float(payload.get("finished_at", 0.0) or 0.0),
            metadata=dict(payload.get("metadata", {}) or {}),
        )
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": str(self.task_id),
            "status": str(self.status),
            "ok": bool(self.ok),
            "objectives": [float(x) for x in self.objectives],
            "violations": [float(x) for x in self.violations],
            "metrics": dict(self.metrics),
            "artifact_refs": [x.as_dict() for x in self.artifact_refs],
            "worker_id": str(self.worker_id),
            "lease_id": str(self.lease_id),
            "resource_context": dict(self.resource_context),
            "error": str(self.error),
            "started_at": float(self.started_at),
            "finished_at": float(self.finished_at),
            "duration_seconds": self.duration_seconds,
            "metadata": dict(self.metadata),
        }


# ============================================================================
# Scheduled Task
# ============================================================================

@dataclass(frozen=True)
class ScheduledTask:
    """
    Task that has been scheduled to a worker.
    
    Combines task envelope, assigned worker, and lease information.
    """
    
    task: TaskEnvelope
    worker: WorkerDescriptor
    lease_id: str = ""
    resource_context: Mapping[str, Any] = field(default_factory=dict)
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task": self.task.as_dict(),
            "worker": self.worker.as_dict(),
            "lease_id": str(self.lease_id),
            "resource_context": dict(self.resource_context),
        }


# ============================================================================
# Project L0 lease primitives
# ============================================================================

@dataclass(frozen=True)
class ResourceRequest:
    """Project-level resource request for one Case run."""

    workers: int = 1
    threads: int | None = None
    gpus: int = 0
    memory_mb: int | float | None = 512
    gpu_memory_mb: int | float | None = None
    backend: str = "local"
    compute_backend: str = "auto"
    device: str = "cpu"
    device_tokens: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        workers = max(1, int(self.workers or 1))
        threads = workers if self.threads is None else max(1, int(self.threads or 1))
        object.__setattr__(self, "workers", workers)
        object.__setattr__(self, "threads", threads)
        object.__setattr__(self, "gpus", max(0, int(self.gpus or 0)))
        object.__setattr__(self, "memory_mb", _as_float_or_none(self.memory_mb))
        object.__setattr__(self, "gpu_memory_mb", _as_float_or_none(self.gpu_memory_mb))
        object.__setattr__(self, "backend", str(self.backend or "local"))
        object.__setattr__(self, "compute_backend", str(self.compute_backend or "auto"))
        object.__setattr__(self, "device", str(self.device or "cpu"))
        object.__setattr__(self, "device_tokens", tuple(str(x) for x in _as_tuple(self.device_tokens)))
        object.__setattr__(self, "capabilities", tuple(str(x) for x in _as_tuple(self.capabilities) if str(x)))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ResourceRequest":
        data = dict(payload or {})
        return cls(
            workers=int(data.get("workers", data.get("threads", 1)) or 1),
            threads=data.get("threads"),
            gpus=int(data.get("gpus", 0) or 0),
            memory_mb=data.get("memory_mb", 512),
            gpu_memory_mb=data.get("gpu_memory_mb"),
            backend=str(data.get("backend", data.get("resource_backend", "local"))),
            compute_backend=str(data.get("compute_backend", "auto")),
            device=str(data.get("device", "cpu")),
            device_tokens=tuple(data.get("device_tokens", ()) or ()),
            capabilities=tuple(data.get("capabilities", ()) or ()),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    def requirement(self) -> ResourceRequirement:
        return ResourceRequirement(
            threads=int(self.threads),
            gpus=int(self.gpus),
            resource_backend=str(self.backend),
            device_tokens=tuple(self.device_tokens),
            memory_mb=self.memory_mb,
            gpu_memory_mb=self.gpu_memory_mb,
            capabilities=tuple(self.capabilities),
            metadata=dict(self.metadata),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "workers": int(self.workers),
            "threads": int(self.threads),
            "gpus": int(self.gpus),
            "memory_mb": self.memory_mb,
            "gpu_memory_mb": self.gpu_memory_mb,
            "backend": str(self.backend),
            "compute_backend": str(self.compute_backend),
            "device": str(self.device),
            "device_tokens": [str(x) for x in self.device_tokens],
            "capabilities": [str(x) for x in self.capabilities],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ResourcePolicy:
    """Project-level resource policy for local L0 allocation."""

    max_workers: int = 4
    max_threads: int | None = None
    max_memory_mb: int | float | None = 4096
    max_gpus: int | None = None
    mode: str = "strict"
    gpu_sharing: str = "exclusive"
    cpu_oversubscribe: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        workers = max(1, int(self.max_workers or 1))
        max_threads = workers if self.max_threads is None else max(1, int(self.max_threads or 1))
        object.__setattr__(self, "max_workers", workers)
        object.__setattr__(self, "max_threads", max_threads)
        object.__setattr__(self, "max_memory_mb", _as_float_or_none(self.max_memory_mb))
        object.__setattr__(self, "max_gpus", None if self.max_gpus is None else max(0, int(self.max_gpus)))
        object.__setattr__(self, "mode", str(self.mode or "strict"))
        object.__setattr__(self, "gpu_sharing", str(self.gpu_sharing or "exclusive"))
        object.__setattr__(self, "cpu_oversubscribe", bool(self.cpu_oversubscribe))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ResourcePolicy":
        data = dict(payload or {})
        return cls(
            max_workers=int(data.get("max_workers", 4) or 4),
            max_threads=data.get("max_threads"),
            max_memory_mb=data.get("max_memory_mb", 4096),
            max_gpus=data.get("max_gpus"),
            mode=str(data.get("mode", "strict")),
            gpu_sharing=str(data.get("gpu_sharing", "exclusive")),
            cpu_oversubscribe=bool(data.get("cpu_oversubscribe", False)),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_workers": int(self.max_workers),
            "max_threads": int(self.max_threads),
            "max_memory_mb": self.max_memory_mb,
            "max_gpus": self.max_gpus,
            "mode": str(self.mode),
            "gpu_sharing": str(self.gpu_sharing),
            "cpu_oversubscribe": bool(self.cpu_oversubscribe),
            "metadata": dict(self.metadata),
        }


@dataclass
class ResourceLease:
    """Lease issued by the Project L0 substrate."""

    lease_id: str
    owner_id: str = ""
    scope: str = ""
    resources: Mapping[str, Any] = field(default_factory=dict)
    status: str = "active"
    created_at: float = field(default_factory=_now_unix)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def resource_context(self, **kwargs: Any) -> Dict[str, Any]:
        resources = dict(self.resources or {})
        grant = {
            "backend": resources.get("backend", "local"),
            "threads": int(resources.get("threads", resources.get("workers", 1)) or 1),
            "workers": int(resources.get("workers", resources.get("threads", 1)) or 1),
            "gpus": int(resources.get("gpus", 0) or 0),
            "device": resources.get("device", "cpu"),
            "device_tokens": list(resources.get("device_tokens", ()) or ()),
            "compute_backend": resources.get("compute_backend", kwargs.get("compute_backend", "auto")),
        }
        payload = {
            "scope": str(kwargs.get("scope", self.scope or "project_case")),
            "execution_backend": str(kwargs.get("execution_backend", grant["backend"])),
            "compute_backend": str(kwargs.get("compute_backend", grant["compute_backend"])),
            "device": str(kwargs.get("device", grant["device"])),
            "threads": int(kwargs.get("threads", grant["threads"]) or 1),
            "nested": bool(kwargs.get("nested", True)),
            "namespace": str(kwargs.get("namespace", "")),
            "grant": grant,
            "resources": dict(grant),
            "lease": self.as_dict(),
            "metadata": dict(kwargs.get("metadata", {}) or {}),
        }
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value
        return payload

    def as_dict(self) -> Dict[str, Any]:
        return {
            "lease_id": str(self.lease_id),
            "owner_id": str(self.owner_id),
            "scope": str(self.scope),
            "resources": dict(self.resources),
            "status": str(self.status),
            "created_at": float(self.created_at),
            "metadata": dict(self.metadata or {}),
        }


class InMemoryLeaseStore:
    """Small lease store used by local Project L0 runs."""

    def __init__(self) -> None:
        self._leases: Dict[str, ResourceLease] = {}

    def create(self, lease: ResourceLease) -> None:
        self._leases[str(lease.lease_id)] = lease

    def get(self, lease_id: str) -> Optional[ResourceLease]:
        return self._leases.get(str(lease_id))

    def update(self, lease: ResourceLease) -> None:
        self._leases[str(lease.lease_id)] = lease

    def delete(self, lease_id: str) -> None:
        self._leases.pop(str(lease_id), None)

    def list(self) -> tuple[ResourceLease, ...]:
        return tuple(self._leases.values())


class ResourceAllocator:
    """Project-level local resource allocator."""

    def __init__(
        self,
        *,
        policy: ResourcePolicy | Mapping[str, Any] | None = None,
        offer: ResourceOffer | Mapping[str, Any] | None = None,
        lease_store: InMemoryLeaseStore | None = None,
    ) -> None:
        self.policy = policy if isinstance(policy, ResourcePolicy) else ResourcePolicy.from_dict(policy)
        self.offer = offer if isinstance(offer, ResourceOffer) else ResourceOffer(**dict(offer or {}))
        self.lease_store = lease_store if lease_store is not None else InMemoryLeaseStore()

    def allocate(self, request: ResourceRequest | Mapping[str, Any]) -> Optional[ResourceLease]:
        return self.acquire(request)

    def acquire(
        self,
        request: ResourceRequest | Mapping[str, Any],
        owner_id: str = "",
        scope: str = "",
    ) -> ResourceLease:
        req = request if isinstance(request, ResourceRequest) else ResourceRequest.from_dict(request)
        if str(self.policy.mode).lower() == "strict":
            if int(req.workers) > int(self.policy.max_workers):
                raise RuntimeError(f"Resource request workers={req.workers} exceeds policy max_workers={self.policy.max_workers}")
            if int(req.threads) > int(self.policy.max_threads) and not bool(self.policy.cpu_oversubscribe):
                raise RuntimeError(f"Resource request threads={req.threads} exceeds policy max_threads={self.policy.max_threads}")
            if self.policy.max_gpus is not None and int(req.gpus) > int(self.policy.max_gpus):
                raise RuntimeError(f"Resource request gpus={req.gpus} exceeds policy max_gpus={self.policy.max_gpus}")
            if self.policy.max_memory_mb is not None and req.memory_mb is not None and float(req.memory_mb) > float(self.policy.max_memory_mb):
                raise RuntimeError("Resource request memory_mb exceeds policy max_memory_mb")
            if not req.requirement().satisfies(self.offer):
                raise RuntimeError("Resource request is not satisfied by Project ResourceOffer")
        lease_id = f"lease-{owner_id or 'case'}-{scope or 'project'}-{uuid4().hex[:12]}"
        lease = ResourceLease(
            lease_id=lease_id,
            owner_id=str(owner_id or ""),
            scope=str(scope or ""),
            resources=req.as_dict(),
            metadata={"source": "blackbase.project.l0"},
        )
        self.lease_store.create(lease)
        return lease

    def release(self, lease: ResourceLease | Mapping[str, Any] | str) -> None:
        if isinstance(lease, str):
            lease_id = lease
        elif isinstance(lease, ResourceLease):
            lease_id = lease.lease_id
        else:
            lease_id = str(dict(lease).get("lease_id", ""))
        if lease_id:
            self.lease_store.delete(lease_id)


# ============================================================================
# Helper Functions
# ============================================================================

def _coerce_data_ref(value: DataRef | Mapping[str, Any]) -> DataRef:
    """Coerce value to DataRef."""
    if isinstance(value, DataRef):
        return value
    if isinstance(value, Mapping):
        return DataRef.from_dict(value)
    return DataRef(uri=str(value))
