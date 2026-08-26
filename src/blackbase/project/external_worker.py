"""External worker runtime for standard Project Case tasks."""

from __future__ import annotations

import argparse
import os
import socket
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from blackbase.resources import (
    DataRef,
    RedisLeaseStore,
    RedisTaskTransport,
    ResourceOffer,
    SQLiteTaskTransport,
    SQLiteLeaseStore,
    TaskLeaseError,
    TaskResult,
    TaskTransport,
    WorkerDescriptor,
)

from .case_execution import execute_case_payload
from .execution import CaseRunRequest, CaseRunResult


class ExternalCaseWorker:
    """Claims and executes standard Case tasks outside the Project process."""

    def __init__(
        self,
        transport: TaskTransport,
        worker: WorkerDescriptor,
        *,
        allowed_project_root: Path | str,
        lease_seconds: float = 30.0,
        heartbeat_interval_seconds: float | None = None,
        lease_redis_url: str | None = None,
        lease_redis_client: Any = None,
    ) -> None:
        self.transport = transport
        self.worker = worker
        self.allowed_project_root = Path(allowed_project_root).resolve()
        self.lease_seconds = max(0.5, float(lease_seconds))
        self.lease_redis_url = str(lease_redis_url or "").strip()
        self.lease_redis_client = lease_redis_client
        default_heartbeat = max(0.1, self.lease_seconds / 3.0)
        self.heartbeat_interval_seconds = max(
            0.05,
            float(default_heartbeat if heartbeat_interval_seconds is None else heartbeat_interval_seconds),
        )
        if self.heartbeat_interval_seconds >= self.lease_seconds:
            raise ValueError("heartbeat_interval_seconds must be smaller than lease_seconds")
        self.worker = self.transport.register_worker(self.worker)

    def run_once(self) -> bool:
        """Claim at most one Case task; return whether a task was processed."""

        self.transport.heartbeat_worker(self.worker.worker_id, status="online")
        claim = self.transport.claim(
            self.worker,
            lease_seconds=self.lease_seconds,
            task_types=("project_case",),
        )
        if claim is None:
            return False

        payload = dict(claim.task.payload)
        task_root = Path(str(payload.get("project_root", ""))).resolve()
        resource_context: dict[str, Any] = {}
        project_lease_store = None
        project_lease: dict[str, Any] = {}
        try:
            if task_root != self.allowed_project_root:
                raise PermissionError(
                    f"Task project_root '{task_root}' is outside worker root "
                    f"'{self.allowed_project_root}'"
                )
            request = CaseRunRequest.from_dict(dict(payload.get("request", {}) or {}))
            attempt_identity = request.identity.for_attempt(claim.attempt)
            attempt_resource_context = dict(request.resource_context)
            attempt_metadata = dict(attempt_resource_context.get("metadata", {}) or {})
            attempt_metadata["run_identity"] = attempt_identity.as_dict()
            attempt_resource_context["metadata"] = attempt_metadata
            request = replace(
                request,
                identity=attempt_identity,
                resource_context=attempt_resource_context,
            )
            payload["request"] = request.as_dict()
            _validate_request_authorities(request, self.allowed_project_root)
            resource_context = dict(request.resource_context)
            project_lease = dict(resource_context.get("lease", {}) or {})
            authority = dict(
                dict(resource_context.get("metadata", {}) or {}).get(
                    "lease_authority",
                    {},
                )
                or {}
            )
            authority_backend = str(authority.get("backend", "memory")).lower()
            if authority_backend == "sqlite":
                authority_path = Path(str(authority.get("path", ""))).resolve()
                if not _is_relative_to(authority_path, self.allowed_project_root):
                    raise PermissionError(
                        f"Task lease authority '{authority_path}' is outside worker root "
                        f"'{self.allowed_project_root}'"
                    )
                project_lease_store = SQLiteLeaseStore(
                    authority_path,
                    namespace=str(authority.get("namespace", "project")),
                )
                if not project_lease_store.is_current(
                    str(project_lease.get("lease_id", "")),
                    int(project_lease.get("fencing_token", 0) or 0),
                ):
                    raise TaskLeaseError(
                        f"Project L0 fence is not current for task_id="
                        f"'{claim.task.task_id}'"
                    )
            elif authority_backend == "redis":
                redis_url_env = str(
                    authority.get("redis_url_env", "BLACKBASE_REDIS_URL")
                    or "BLACKBASE_REDIS_URL"
                )
                redis_url = self.lease_redis_url or str(
                    os.environ.get(redis_url_env, "") or ""
                ).strip()
                if self.lease_redis_client is None and not redis_url:
                    raise RuntimeError(
                        "Redis Project L0 fence requires worker --lease-redis-url "
                        f"or environment variable {redis_url_env}"
                    )
                project_lease_store = RedisLeaseStore(
                    redis_url or "redis://localhost:6379/0",
                    namespace=str(authority.get("namespace", "project")),
                    client=self.lease_redis_client,
                )
                if not project_lease_store.is_current(
                    str(project_lease.get("lease_id", "")),
                    int(project_lease.get("fencing_token", 0) or 0),
                ):
                    raise TaskLeaseError(
                        f"Project L0 fence is not current for task_id="
                        f"'{claim.task.task_id}'"
                    )
            elif authority_backend != "memory":
                raise RuntimeError(
                    f"Unsupported Project L0 lease authority '{authority_backend}'"
                )
        except Exception as exc:
            self.transport.fail(
                claim,
                TaskResult.failure(
                    task_id=claim.task.task_id,
                    error=f"{type(exc).__name__}: {exc}",
                    worker_id=self.worker.worker_id,
                ),
            )
            self.transport.heartbeat_worker(self.worker.worker_id, status="idle")
            return True

        stop_heartbeat = threading.Event()
        lease_lost = threading.Event()

        def heartbeat() -> None:
            while not stop_heartbeat.wait(self.heartbeat_interval_seconds):
                task_ok = self.transport.heartbeat_task(
                    claim,
                    lease_seconds=self.lease_seconds,
                )
                worker_ok = self.transport.heartbeat_worker(self.worker.worker_id, status="busy")
                project_lease_ok = True
                if project_lease_store is not None:
                    renewed = project_lease_store.renew_lease(
                        str(project_lease.get("lease_id", "")),
                        int(project_lease.get("fencing_token", 0) or 0),
                        ttl_seconds=(
                            float(
                                dict(resource_context.get("metadata", {}) or {})
                                .get("lease_authority", {})
                                .get("ttl_seconds", self.lease_seconds)
                            )
                        ),
                    )
                    project_lease_ok = renewed is not None
                if not task_ok or not worker_ok or not project_lease_ok:
                    lease_lost.set()
                    return

        heartbeat_thread = threading.Thread(
            target=heartbeat,
            name=f"blackbase-heartbeat-{claim.task.task_id}",
            daemon=True,
        )
        heartbeat_thread.start()
        started_at = time.time()
        try:
            worker_output = execute_case_payload(payload)
            case_result = CaseRunResult.from_dict(worker_output)
            if lease_lost.is_set():
                raise TaskLeaseError(
                    f"Worker lost the task lease for task_id='{claim.task.task_id}'"
                )
            named_refs = {
                name: ref.as_dict() for name, ref in case_result.artifact_refs.items()
            }
            lease = dict(resource_context.get("lease", {}) or {})
            if project_lease_store is not None and not project_lease_store.is_current(
                str(lease.get("lease_id", "")),
                int(lease.get("fencing_token", 0) or 0),
            ):
                raise TaskLeaseError(
                    f"Project L0 fence is no longer current for task_id='{claim.task.task_id}'"
                )
            result = TaskResult(
                task_id=claim.task.task_id,
                status="ok" if case_result.ok else "failed",
                output=worker_output,
                artifact_refs=tuple(DataRef.from_dict(value) for value in named_refs.values()),
                worker_id=self.worker.worker_id,
                lease_id=str(lease.get("lease_id", "")),
                resource_context=resource_context,
                started_at=started_at,
                finished_at=time.time(),
                error="" if case_result.ok else case_result.error,
                metadata={
                    "attempt": claim.attempt,
                    "task_type": claim.task.task_type,
                    "lease_fence_validated": project_lease_store is not None,
                    "fencing_token": int(lease.get("fencing_token", 0) or 0),
                },
            )
            if case_result.ok:
                self.transport.complete(claim, result)
            else:
                self.transport.fail(claim, result)
        except Exception as exc:
            result = TaskResult(
                task_id=claim.task.task_id,
                status="failed",
                worker_id=self.worker.worker_id,
                error=f"{type(exc).__name__}: {exc}",
                started_at=started_at,
                finished_at=time.time(),
                metadata={"attempt": claim.attempt, "task_type": claim.task.task_type},
            )
            try:
                self.transport.fail(claim, result)
            except TaskLeaseError:
                if not lease_lost.is_set():
                    raise
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=self.heartbeat_interval_seconds * 2.0)
            self.transport.heartbeat_worker(self.worker.worker_id, status="idle")
        return True

    def run_forever(
        self,
        *,
        poll_interval_seconds: float = 0.1,
        stop_event: threading.Event | None = None,
        max_tasks: int | None = None,
    ) -> int:
        processed = 0
        poll = max(0.01, float(poll_interval_seconds))
        try:
            while stop_event is None or not stop_event.is_set():
                if max_tasks is not None and processed >= int(max_tasks):
                    break
                if self.run_once():
                    processed += 1
                    continue
                time.sleep(poll)
        finally:
            self.transport.heartbeat_worker(self.worker.worker_id, status="offline")
        return processed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an external blackbase Project Case worker.")
    parser.add_argument("--project-root", required=True, help="Only Project root this worker may execute")
    parser.add_argument("--backend", choices=("sqlite", "redis"), default="sqlite")
    parser.add_argument("--transport", help="SQLite task transport path")
    parser.add_argument("--redis-url", help="Redis connection URL")
    parser.add_argument(
        "--lease-redis-url",
        help="Redis URL for Project L0 leases; defaults to --redis-url for Redis workers",
    )
    parser.add_argument("--namespace", help="Redis task namespace")
    parser.add_argument("--worker-id", default=f"{socket.gethostname()}-{Path.cwd().name}")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--device-token", action="append", default=[])
    parser.add_argument("--capability", action="append", default=[])
    parser.add_argument("--lease-seconds", type=float, default=30.0)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.backend == "sqlite" and not args.transport:
        parser.error("--transport is required when --backend=sqlite")
    if args.backend == "redis" and (not args.redis_url or not args.namespace):
        parser.error("--redis-url and --namespace are required when --backend=redis")
    capabilities = tuple(dict.fromkeys(("project_case", *args.capability)))
    worker = WorkerDescriptor(
        worker_id=args.worker_id,
        executor_backend="external",
        resource_backend="local",
        capabilities=capabilities,
        offer=ResourceOffer(
            threads=max(1, int(args.threads)),
            gpus=max(0, int(args.gpus)),
            backend="local",
            device_tokens=tuple(args.device_token),
        ),
        max_inflight=1,
    )
    transport: TaskTransport
    if args.backend == "redis":
        transport = RedisTaskTransport(args.redis_url, namespace=args.namespace)
    else:
        transport = SQLiteTaskTransport(args.transport)
    runtime = ExternalCaseWorker(
        transport,
        worker,
        allowed_project_root=args.project_root,
        lease_seconds=args.lease_seconds,
        lease_redis_url=(args.lease_redis_url or args.redis_url),
    )
    if args.once:
        return 0 if runtime.run_once() else 3
    runtime.run_forever(
        poll_interval_seconds=args.poll_interval,
        max_tasks=args.max_tasks,
    )
    return 0


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_request_authorities(request: CaseRunRequest, project_root: Path) -> None:
    """Reject worker payloads that redirect durable authorities outside the Project."""

    for ref in (request.control.cancellation,):
        if ref.process_local:
            raise PermissionError(
                "External Case worker requires SQLite or Redis cancellation "
                "authority; memory is process-local"
            )
        if ref.backend == "sqlite":
            _require_project_path(Path(ref.path), project_root, label="cancellation authority")
    metadata = dict(request.resource_context.get("metadata", {}) or {})
    descriptors = [
        dict(metadata.get("lease_authority", {}) or {}),
        dict(metadata.get("budget_authority", {}) or {}),
        *(dict(handle.authority) for handle in request.budget_handles.values()),
    ]
    for descriptor in descriptors:
        if str(descriptor.get("backend", "")).lower() != "sqlite":
            continue
        path = str(descriptor.get("path", "") or "").strip()
        if not path:
            raise PermissionError("SQLite authority omitted its database path")
        _require_project_path(Path(path), project_root, label="SQLite authority")
    artifact_authority = dict(metadata.get("artifact_authority", {}) or {})
    if artifact_authority:
        backend = str(artifact_authority.get("backend", "")).strip().lower()
        if backend != "filesystem":
            raise PermissionError(
                f"External Case worker does not support artifact authority '{backend}'"
            )
        root = str(artifact_authority.get("root", "") or "").strip()
        if not root:
            raise PermissionError("Filesystem artifact authority omitted its root")
        _require_project_path(Path(root), project_root, label="artifact authority")


def _require_project_path(path: Path, project_root: Path, *, label: str) -> None:
    resolved = path.resolve()
    if not _is_relative_to(resolved, project_root):
        raise PermissionError(
            f"Task {label} '{resolved}' is outside worker root '{project_root}'"
        )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ExternalCaseWorker", "build_parser", "main"]
