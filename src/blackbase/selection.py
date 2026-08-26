"""Shared row-selector normalization for aligned protocol batches."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalize_row_selector(selector: Any, row_count: int) -> np.ndarray:
    """Normalize a slice, boolean mask, or integer sequence to row indices.

    Tuple selectors are converted before indexing so NumPy cannot reinterpret
    them as multidimensional coordinates.  Empty sequences always mean an
    empty subset, never the special ``array[()]`` full-array operation.
    """

    count = int(row_count)
    if count < 0:
        raise ValueError("row_count must be non-negative")
    if isinstance(selector, slice):
        return np.arange(count, dtype=np.intp)[selector]

    raw = np.asarray(selector)
    if raw.ndim != 1:
        raise ValueError("row selector must be one-dimensional")
    if raw.dtype.kind == "b":
        if raw.size != count:
            raise ValueError(
                "boolean row selector length must match row_count: "
                f"selector={raw.size}, row_count={count}"
            )
        return np.flatnonzero(raw).astype(np.intp, copy=False)
    if raw.size == 0:
        return np.empty((0,), dtype=np.intp)
    if raw.dtype.kind not in {"i", "u"}:
        raise TypeError("row selector must contain integers or booleans")
    indices = raw.astype(np.intp, copy=False)
    if np.any(indices < 0) or np.any(indices >= count):
        raise IndexError(
            f"row selector is outside [0, {count}): {indices.tolist()}"
        )
    return np.array(indices, dtype=np.intp, copy=True)


__all__ = ["normalize_row_selector"]
