"""Shared batch-disposition contracts for solver and trainer adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class BatchDisposition:
    """Describe which proposal items survive a control-plane decision.

    ``accepted_indices`` are indices into the original proposal and must be
    strictly increasing. The accepted batch passed to the adapter's later
    ``update`` call must preserve this order. The contract is intentionally
    independent of why items were removed: budget admission, filtering,
    cancellation and backend partial failure can all use the same payload.
    """

    proposed_count: int
    accepted_indices: tuple[int, ...] = ()
    reason: str = "unspecified"
    reservation_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        proposed = int(self.proposed_count)
        if proposed < 0:
            raise ValueError("proposed_count must be >= 0")
        accepted = tuple(int(index) for index in self.accepted_indices)
        if accepted != tuple(sorted(set(accepted))):
            raise ValueError("accepted_indices must be unique and strictly increasing")
        if any(index < 0 or index >= proposed for index in accepted):
            raise ValueError(
                "accepted_indices must reference the original proposal: "
                f"proposed_count={proposed}, accepted_indices={accepted}"
            )
        object.__setattr__(self, "proposed_count", proposed)
        object.__setattr__(self, "accepted_indices", accepted)
        object.__setattr__(self, "reason", str(self.reason or "unspecified"))
        object.__setattr__(self, "reservation_id", str(self.reservation_id or ""))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def accepted_count(self) -> int:
        return int(len(self.accepted_indices))

    @property
    def rejected_indices(self) -> tuple[int, ...]:
        accepted = set(self.accepted_indices)
        return tuple(
            index for index in range(self.proposed_count) if index not in accepted
        )

    @property
    def changed(self) -> bool:
        return self.accepted_count != self.proposed_count

    @classmethod
    def prefix(
        cls,
        *,
        proposed_count: int,
        accepted_count: int,
        reason: str = "unspecified",
        reservation_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> "BatchDisposition":
        proposed = int(proposed_count)
        accepted = int(accepted_count)
        if proposed < 0:
            raise ValueError("proposed_count must be >= 0")
        if accepted < 0 or accepted > proposed:
            raise ValueError(
                "accepted_count must be between zero and proposed_count: "
                f"proposed_count={proposed}, accepted_count={accepted}"
            )
        return cls(
            proposed_count=proposed,
            accepted_indices=tuple(range(accepted)),
            reason=reason,
            reservation_id=reservation_id,
            metadata=dict(metadata or {}),
        )

    def for_range(self, start: int, end: int) -> "BatchDisposition":
        """Project this disposition onto one contiguous child proposal range."""

        range_start = int(start)
        range_end = int(end)
        if range_start < 0 or range_end < range_start or range_end > self.proposed_count:
            raise ValueError(
                "child range must be contained in the proposal: "
                f"range=({range_start}, {range_end}), proposed_count={self.proposed_count}"
            )
        accepted = tuple(
            index - range_start
            for index in self.accepted_indices
            if range_start <= index < range_end
        )
        return BatchDisposition(
            proposed_count=range_end - range_start,
            accepted_indices=accepted,
            reason=self.reason,
            reservation_id=self.reservation_id,
            metadata={
                **dict(self.metadata),
                "parent_range": (range_start, range_end),
            },
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "proposed_count": int(self.proposed_count),
            "accepted_indices": list(self.accepted_indices),
            "accepted_count": int(self.accepted_count),
            "rejected_indices": list(self.rejected_indices),
            "reason": self.reason,
            "reservation_id": self.reservation_id,
            "metadata": dict(self.metadata),
        }


__all__ = ["BatchDisposition"]
