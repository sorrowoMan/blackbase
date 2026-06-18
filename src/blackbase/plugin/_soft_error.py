"""Lightweight soft-error reporting for blackbase plugins.

This is a simplified version of nsgablack's ``report_soft_error`` that only
does logging + optional strict raise.  It does *not* update any context_store
metrics, because context_store is a framework-specific concern that belongs
to the upper layers (nsgablack / mlblack), not to the shared substrate.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

# Per-key rate limiting state (module-level, lightweight)
_SOFT_ERROR_LAST_EMIT_AT: Dict[str, float] = {}


def report_soft_error(
    *,
    component: str,
    event: str,
    exc: Exception,
    logger: Optional[logging.Logger] = None,
    strict: bool = False,
    level: str = "warning",
    min_interval_seconds: float = 30.0,
) -> None:
    """Record a soft error via logging, or raise in strict mode.

    Args:
        component: Originating component name (e.g. ``"Plugin"``).
        event: Event or method where the error occurred.
        exc: The caught exception.
        logger: Optional logger; defaults to a logger named after *component*.
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
