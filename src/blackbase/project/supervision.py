"""Isolated Case supervision for hard deadline/cancellation enforcement."""

from __future__ import annotations

import multiprocessing
import time
from multiprocessing.connection import Connection
from typing import Any, Mapping

from blackbase.resources import CancellationToken

from .execution import CaseFailure, CaseRunRequest, CaseRunResult, ProjectConfigurationError


def execute_case_payload_supervised(
    payload: Mapping[str, Any],
    *,
    redis_client: Any = None,
) -> dict[str, Any]:
    """Execute a payload directly or inside a terminable process boundary."""

    request_payload = payload.get("request")
    if not isinstance(request_payload, Mapping):
        raise ProjectConfigurationError("Case execution payload omitted versioned request")
    request = CaseRunRequest.from_dict(request_payload)
    policy = request.control.termination
    if not policy.requires_isolation:
        from .case_execution import execute_case_payload

        return execute_case_payload(payload)

    tokens = tuple(
        CancellationToken(ref, redis_client=redis_client)
        for ref in (
            *request.control.ancestor_cancellations,
            request.control.cancellation,
        )
    )
    started_at = time.time()
    signal = _control_signal(tokens)
    if signal is not None:
        return _terminated_result(
            request,
            signal=signal,
            started_at=started_at,
            process_id=0,
            terminated=False,
            killed=False,
        ).as_dict()

    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    isolated_payload = dict(payload)
    isolated_payload["_blackbase_isolated_worker"] = True
    process = context.Process(
        target=_isolated_case_entry,
        args=(send, isolated_payload),
        name=f"blackbase-case-{request.identity.case_run_id}",
    )
    process.daemon = False
    process.start()
    send.close()
    signalled_at = 0.0
    signal = None
    try:
        while True:
            message = _receive_if_ready(receive, policy.poll_interval_seconds)
            if message is not None:
                process.join(timeout=policy.kill_grace_seconds)
                if message.get("kind") == "result":
                    result_payload = message.get("payload")
                    if isinstance(result_payload, Mapping):
                        return dict(result_payload)
                return _worker_failure_result(
                    request,
                    started_at=started_at,
                    message=str(message.get("error", "isolated Case worker failed")),
                    process_id=int(process.pid or 0),
                    exit_code=process.exitcode,
                ).as_dict()

            if not process.is_alive():
                message = _receive_if_ready(receive, 0.0)
                if message is not None and message.get("kind") == "result":
                    result_payload = message.get("payload")
                    if isinstance(result_payload, Mapping):
                        return dict(result_payload)
                return _worker_failure_result(
                    request,
                    started_at=started_at,
                    message=(
                        "isolated Case process exited without a result envelope "
                        f"(exit_code={process.exitcode})"
                    ),
                    process_id=int(process.pid or 0),
                    exit_code=process.exitcode,
                ).as_dict()

            if signal is None:
                signal = _control_signal(tokens)
                if signal is not None:
                    signalled_at = time.monotonic()
            if signal is None:
                continue
            if time.monotonic() - signalled_at < policy.grace_seconds:
                continue

            process.terminate()
            process.join(timeout=policy.kill_grace_seconds)
            killed = False
            if process.is_alive():
                kill = getattr(process, "kill", None)
                if callable(kill):
                    kill()
                    killed = True
                else:  # pragma: no cover - supported on current Python targets
                    process.terminate()
                process.join(timeout=max(0.1, policy.kill_grace_seconds))
            return _terminated_result(
                request,
                signal=signal,
                started_at=started_at,
                process_id=int(process.pid or 0),
                terminated=True,
                killed=killed,
            ).as_dict()
    finally:
        receive.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=max(0.1, policy.kill_grace_seconds))
            if process.is_alive():
                kill = getattr(process, "kill", None)
                if callable(kill):
                    kill()
                process.join(timeout=0.5)


def _isolated_case_entry(send: Connection, payload: Mapping[str, Any]) -> None:
    try:
        from .case_execution import execute_case_payload

        send.send({"kind": "result", "payload": execute_case_payload(payload)})
    except BaseException as exc:
        try:
            send.send(
                {
                    "kind": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        send.close()


def _receive_if_ready(connection: Connection, timeout: float) -> dict[str, Any] | None:
    try:
        if not connection.poll(max(0.0, float(timeout))):
            return None
        message = connection.recv()
    except (EOFError, OSError):
        return None
    return dict(message) if isinstance(message, Mapping) else {
        "kind": "error",
        "error": "isolated Case worker returned an invalid supervisor message",
    }


def _control_signal(
    tokens: tuple[CancellationToken, ...],
) -> tuple[str, str] | None:
    for token in tokens:
        if token.deadline_exceeded:
            token.cancel("case deadline exceeded")
            return "timed_out", "case deadline exceeded"
    for token in tokens:
        state = token.state
        if state.requested:
            return "cancelled", state.reason or "case cancellation requested"
    return None


def _terminated_result(
    request: CaseRunRequest,
    *,
    signal: tuple[str, str],
    started_at: float,
    process_id: int,
    terminated: bool,
    killed: bool,
) -> CaseRunResult:
    status, reason = signal
    details = {
        "termination_mode": request.control.termination.mode,
        "grace_seconds": request.control.termination.grace_seconds,
        "process_id": int(process_id),
        "terminated": bool(terminated),
        "killed": bool(killed),
    }
    return CaseRunResult(
        request=request,
        status=status,
        started_at=started_at,
        finished_at=time.time(),
        exit_code=1,
        failure=CaseFailure(
            kind=("CaseDeadlineExceeded" if status == "timed_out" else "CancellationRequested"),
            message=reason,
            phase="terminate" if terminated else "control",
            details=details,
        ),
        metadata={"termination": details},
    )


def _worker_failure_result(
    request: CaseRunRequest,
    *,
    started_at: float,
    message: str,
    process_id: int,
    exit_code: int | None,
) -> CaseRunResult:
    details = {
        "termination_mode": request.control.termination.mode,
        "process_id": int(process_id),
        "worker_exit_code": exit_code,
    }
    return CaseRunResult(
        request=request,
        status="failed",
        started_at=started_at,
        finished_at=time.time(),
        exit_code=1,
        failure=CaseFailure(
            kind="CaseWorkerExitError",
            message=message,
            phase="supervise",
            details=details,
        ),
        metadata={"termination": details},
    )


__all__ = ["execute_case_payload_supervised"]
