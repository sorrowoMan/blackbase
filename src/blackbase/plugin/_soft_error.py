"""Shared soft-error reporting with optional ContextStore audit evidence."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from blackbase.context.context_keys import (
    KEY_METRICS,
    KEY_METRICS_SOFT_ERROR_COUNT,
    KEY_METRICS_SOFT_ERROR_LAST,
)

# Per-key rate limiting state (module-level, lightweight)
_SOFT_ERROR_LAST_EMIT_AT: Dict[str, float] = {}


def _record_context_audit(
    context_store: Any,
    *,
    component: str,
    event: str,
    error_type: str,
    message: str,
) -> None:
    if context_store is None:
        return
    try:
        metrics = context_store.get(KEY_METRICS, {})
    except TypeError:
        metrics = context_store.get(KEY_METRICS)
    except Exception:
        return
    metrics = dict(metrics) if isinstance(metrics, dict) else {}
    counts = metrics.get(KEY_METRICS_SOFT_ERROR_COUNT)
    counts = dict(counts) if isinstance(counts, dict) else {}
    bucket = f"{component}.{event}"
    counts[bucket] = int(counts.get(bucket, 0) or 0) + 1
    metrics[KEY_METRICS_SOFT_ERROR_COUNT] = counts
    metrics[KEY_METRICS_SOFT_ERROR_LAST] = {
        "component": str(component),
        "event": str(event),
        "error_type": str(error_type),
        "message": str(message),
        "ts": float(time.time()),
    }
    try:
        setter = getattr(context_store, "set", None)
        if callable(setter):
            setter(KEY_METRICS, metrics)
        else:
            context_store[KEY_METRICS] = metrics
    except Exception:
        return


def report_soft_error(
    *,
    component: str,
    event: str,
    exc: Exception,
    logger: Optional[logging.Logger] = None,
    context_store: Any = None,
    strict: bool = False,
    level: str = "warning",
    min_interval_seconds: float = 30.0,
) -> None:
    """Record a soft error via logging/context audit, or raise in strict mode.

    Args:
        component: Originating component name (e.g. ``"Plugin"``).
        event: Event or method where the error occurred.
        exc: The caught exception.
        logger: Optional logger; defaults to a logger named after *component*.
        context_store: Optional shared ContextStore receiving bounded audit metrics.
        strict: If ``True``, re-raise *exc* instead of logging.
        level: Log level string (``"debug"``, ``"info"``, ``"warning"``,
            ``"error"``).
        min_interval_seconds: Minimum seconds between repeated log emissions
            for the same component/event/error_type key (rate limiting).
    """
    if strict:
        raise exc

    log = logger or logging.getLogger(str(component))
    error_type = exc.__class__.__name__
    message = str(exc)
    emit_key = f"{component}|{event}|{error_type}"
    now = time.time()
    last = _SOFT_ERROR_LAST_EMIT_AT.get(emit_key, 0.0)
    should_emit = (now - last) >= max(0.0, float(min_interval_seconds))
    if should_emit:
        _SOFT_ERROR_LAST_EMIT_AT[emit_key] = now
        text = f"[soft-error] {component}.{event}: {error_type}: {message}"
        if level == "debug":
            log.debug(text)
        elif level == "info":
            log.info(text)
        elif level == "error":
            log.error(text)
        else:
            log.warning(text)

    _record_context_audit(
        context_store,
        component=str(component),
        event=str(event),
        error_type=error_type,
        message=message,
    )
