"""Case-side budget reservation lifecycle built on Project L0 authorities."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Iterator, Mapping
from uuid import uuid4

from .budget import (
    BudgetReservation,
    BudgetSnapshot,
    RedisBudgetAuthority,
    SQLiteBudgetAuthority,
    SharedBudgetConfigurationError,
    build_budget_authority_from_resource_context,
)
from .context import ResourceContext, coerce_resource_context


BudgetAuthority = SQLiteBudgetAuthority | RedisBudgetAuthority


@dataclass(frozen=True)
class BudgetHandle:
    """Serializable bounded delegation of one semantic budget."""

    handle_id: str
    budget: str
    authority_budget: str
    limit: int
    authority: Mapping[str, Any] = field(default_factory=dict)
    lease_id: str = ""
    fencing_token: int = 0
    parent_reservation_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        handle_id = str(self.handle_id or "").strip()
        budget = str(self.budget or "").strip()
        authority_budget = str(self.authority_budget or budget).strip()
        limit = int(self.limit)
        if not handle_id:
            raise ValueError("budget handle_id must be non-empty")
        if not budget or not authority_budget:
            raise ValueError("budget handle names must be non-empty")
        if limit < 0:
            raise ValueError("budget handle limit must be non-negative")
        object.__setattr__(self, "handle_id", handle_id)
        object.__setattr__(self, "budget", budget)
        object.__setattr__(self, "authority_budget", authority_budget)
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "authority", dict(self.authority or {}))
        object.__setattr__(self, "lease_id", str(self.lease_id or ""))
        object.__setattr__(self, "fencing_token", max(0, int(self.fencing_token or 0)))
        object.__setattr__(self, "parent_reservation_id", str(self.parent_reservation_id or ""))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def local(cls, budget: str, limit: int, *, handle_id: str = "") -> "BudgetHandle":
        return cls(
            handle_id=str(handle_id or f"budget-handle-{uuid4().hex}"),
            budget=str(budget),
            authority_budget=str(budget),
            limit=int(limit),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BudgetHandle":
        return cls(
            handle_id=str(payload.get("handle_id", "")),
            budget=str(payload.get("budget", "")),
            authority_budget=str(payload.get("authority_budget", payload.get("budget", ""))),
            limit=int(payload.get("limit", 0) or 0),
            authority=dict(payload.get("authority", {}) or {}),
            lease_id=str(payload.get("lease_id", "")),
            fencing_token=int(payload.get("fencing_token", 0) or 0),
            parent_reservation_id=str(payload.get("parent_reservation_id", "")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "handle_id": self.handle_id,
            "budget": self.budget,
            "authority_budget": self.authority_budget,
            "limit": self.limit,
            "authority": dict(self.authority),
            "lease_id": self.lease_id,
            "fencing_token": self.fencing_token,
            "parent_reservation_id": self.parent_reservation_id,
            "metadata": dict(self.metadata),
        }


@dataclass
class BudgetClaim:
    """One in-process claim backed by an optional Project L0 reservation."""

    claim_id: str
    budget: str
    amount: int
    _account_id: str = field(repr=False)
    _shared: BudgetReservation | None = field(default=None, repr=False)
    _consumed: int = field(default=0, repr=False)
    _status: str = field(default="active", repr=False)

    def __post_init__(self) -> None:
        self.claim_id = str(self.claim_id or "").strip()
        self.budget = str(self.budget or "").strip()
        self.amount = int(self.amount)
        self._account_id = str(self._account_id or "")
        self._consumed = int(self._consumed)
        self._status = str(self._status or "active")
        if not self.claim_id:
            raise ValueError("claim_id must be non-empty")
        if not self.budget:
            raise ValueError("budget must be non-empty")
        if self.amount < 0:
            raise ValueError("claim amount must be non-negative")
        if self._consumed < 0 or self._consumed > self.amount:
            raise ValueError("claim consumption must be within the reserved amount")
        if self._status not in {"active", "completed", "cancelled"}:
            raise ValueError(f"unsupported budget claim status '{self._status}'")

    @property
    def consumed(self) -> int:
        return int(self._consumed)

    @property
    def remaining(self) -> int:
        return max(0, int(self.amount) - int(self._consumed))

    @property
    def active(self) -> bool:
        return self._status == "active"

    @property
    def status(self) -> str:
        return str(self._status)

    @property
    def reservation_id(self) -> str:
        if self._shared is not None:
            return str(self._shared.reservation_id)
        return str(self.claim_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": str(self.claim_id),
            "reservation_id": self.reservation_id,
            "budget": str(self.budget),
            "amount": int(self.amount),
            "consumed": int(self.consumed),
            "remaining": int(self.remaining),
            "status": str(self.status),
            "shared": self._shared is not None,
        }


class BudgetAccount:
    """Thread-safe Case-side client for one named Project budget.

    The account owns local claim bookkeeping and delegates cross-Case atomicity
    to the authority encoded in ``ResourceContext``. Framework layers decide
    what one unit means and when it should be consumed.
    """

    def __init__(
        self,
        budget: str,
        *,
        authority: BudgetAuthority | None = None,
        authority_budget: str = "",
        lease_id: str = "",
        fencing_token: int = 0,
        limit: int | None = None,
        handle_id: str = "",
    ) -> None:
        name = str(budget or "").strip()
        if not name:
            raise ValueError("budget must be non-empty")
        self.budget = name
        self._authority_budget = str(authority_budget or name).strip()
        self._authority = authority
        self._lease_id = str(lease_id or "")
        self._fencing_token = int(fencing_token)
        self._limit = None if limit is None else int(limit)
        if self._limit is not None and self._limit < 0:
            raise ValueError("budget account limit must be non-negative")
        self._handle_id = str(handle_id or "")
        self._spent = 0
        if authority is not None:
            if not self._lease_id:
                raise SharedBudgetConfigurationError(
                    f"shared budget '{name}' requires a Project L0 lease"
                )
            if self._fencing_token <= 0:
                raise SharedBudgetConfigurationError(
                    f"shared budget '{name}' requires a positive fencing token"
                )
        self._account_id = f"budget-account-{uuid4().hex}"
        self._claims: dict[str, BudgetClaim] = {}
        self._lock = RLock()

    @classmethod
    def from_handle(cls, handle: BudgetHandle | Mapping[str, Any]) -> "BudgetAccount":
        item = handle if isinstance(handle, BudgetHandle) else BudgetHandle.from_dict(handle)
        authority = None
        if item.authority:
            authority = build_budget_authority_from_resource_context(
                {
                    "lease": {
                        "lease_id": item.lease_id,
                        "fencing_token": item.fencing_token,
                    },
                    "metadata": {"budget_authority": dict(item.authority)},
                }
            )
        return cls(
            item.budget,
            authority=authority,
            authority_budget=item.authority_budget,
            lease_id=item.lease_id,
            fencing_token=item.fencing_token,
            limit=item.limit,
            handle_id=item.handle_id,
        )

    @classmethod
    def from_resource_context(
        cls,
        budget: str,
        resource_context: Mapping[str, Any] | ResourceContext,
    ) -> "BudgetAccount":
        context = coerce_resource_context(resource_context)
        payload = context.as_dict()
        metadata = dict(payload.get("metadata", {}) or {})
        handles = dict(metadata.get("budget_handles", {}) or {})
        name = str(budget or "").strip()
        if name in handles:
            return cls.from_handle(handles[name])
        declared = dict(metadata.get("budget_authority", {}) or {})
        budgets = dict(declared.get("budgets", {}) or {})
        if name not in budgets:
            return cls(name)
        authority = build_budget_authority_from_resource_context(payload)
        if authority is None:
            raise SharedBudgetConfigurationError(
                f"shared budget '{name}' has no reconstructable authority"
            )
        lease = dict(payload.get("lease", {}) or {})
        return cls(
            name,
            authority=authority,
            lease_id=str(lease.get("lease_id", "") or ""),
            fencing_token=int(lease.get("fencing_token", 0) or 0),
        )

    @property
    def shared(self) -> bool:
        return self._authority is not None

    @property
    def active_reserved(self) -> int:
        with self._lock:
            return sum(claim.remaining for claim in self._claims.values() if claim.active)

    @property
    def active_claim_count(self) -> int:
        with self._lock:
            return sum(1 for claim in self._claims.values() if claim.active)

    @property
    def spent(self) -> int:
        with self._lock:
            return int(self._spent)

    @property
    def remaining_limit(self) -> int | None:
        with self._lock:
            if self._limit is None:
                return None
            return max(0, int(self._limit) - int(self._spent) - int(self.active_reserved))

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Serialize a framework-level allowance-and-reserve transaction."""

        with self._lock:
            yield

    def shared_status(self) -> BudgetSnapshot | None:
        with self._lock:
            if self._authority is None:
                return None
            return self._authority.status(self._authority_budget)

    def allowance(self, requested: int, *, local_limit: int | None = None) -> int:
        request = int(requested)
        if request < 0:
            raise ValueError("requested budget units must be non-negative")
        with self._lock:
            allowed = request
            if local_limit is not None:
                local = int(local_limit)
                if local < 0:
                    raise ValueError("local_limit must be non-negative")
                allowed = min(allowed, local)
            if self._limit is not None:
                allowed = min(allowed, int(self.remaining_limit or 0))
            if self._authority is not None:
                allowed = min(
                    allowed,
                    int(self._authority.status(self._authority_budget).remaining),
                )
            return allowed

    def reserve(
        self,
        amount: int,
        *,
        reservation_id: str | None = None,
    ) -> BudgetClaim:
        requested = int(amount)
        if requested < 0:
            raise ValueError("reservation amount must be non-negative")
        with self._lock:
            if self._limit is not None and requested > int(self.remaining_limit or 0):
                raise ValueError("reservation exceeds the delegated budget handle")
            if requested == 0:
                return BudgetClaim(
                    claim_id=str(reservation_id or f"budget-zero-{uuid4().hex}"),
                    budget=self.budget,
                    amount=0,
                    _account_id=self._account_id,
                    _status="completed",
                )
            shared = None
            if self._authority is not None:
                shared = self._authority.reserve(
                    self._authority_budget,
                    requested,
                    lease_id=self._lease_id,
                    fencing_token=self._fencing_token,
                    reservation_id=reservation_id,
                )
            claim_id = (
                str(shared.reservation_id)
                if shared is not None
                else str(reservation_id or f"budget-local-{uuid4().hex}")
            )
            existing = self._claims.get(claim_id)
            if existing is not None:
                if existing.amount != requested:
                    raise ValueError(
                        "budget reservation_id was reused with a different amount"
                    )
                return existing
            claim = BudgetClaim(
                claim_id=claim_id,
                budget=self.budget,
                amount=requested,
                _account_id=self._account_id,
                _shared=shared,
            )
            self._claims[claim_id] = claim
            return claim

    def consume(self, claim: BudgetClaim, amount: int = 1) -> None:
        delta = int(amount)
        if delta < 0:
            raise ValueError("consumed budget units must be non-negative")
        if delta == 0:
            return
        with self._lock:
            current = self._require_active(claim)
            if delta > current.remaining:
                raise ValueError("consumed budget units cannot exceed the claim")
            if current._shared is not None:
                if self._authority is None:
                    raise RuntimeError("shared budget authority was detached")
                current._shared = self._authority.consume(current._shared, amount=delta)
            current._consumed += delta

    def complete(self, claim: BudgetClaim) -> None:
        with self._lock:
            self._require_owned(claim)
            if claim.status == "completed":
                return
            current = self._require_active(claim)
            if current._shared is not None:
                if self._authority is None:
                    raise RuntimeError("shared budget authority was detached")
                current._shared = self._authority.complete(
                    current._shared,
                    completed=current.consumed,
                )
            self._spent += current.consumed
            current._status = "completed"
            self._claims.pop(current.claim_id, None)

    def cancel(self, claim: BudgetClaim) -> bool:
        with self._lock:
            self._require_owned(claim)
            if not claim.active:
                return False
            current = self._require_active(claim)
            if current._shared is not None:
                if self._authority is None:
                    raise RuntimeError("shared budget authority was detached")
                self._authority.cancel(current._shared)
            self._spent += current.consumed
            current._status = "cancelled"
            self._claims.pop(current.claim_id, None)
            return True

    def cancel_all(self) -> None:
        with self._lock:
            for claim in tuple(self._claims.values()):
                if claim.active:
                    self.cancel(claim)

    def _require_active(self, claim: BudgetClaim) -> BudgetClaim:
        self._require_owned(claim)
        current = self._claims.get(claim.claim_id)
        if current is not claim:
            raise ValueError("budget claim is not active in this account")
        if not current.active:
            raise RuntimeError("budget claim is already closed")
        return current

    def _require_owned(self, claim: BudgetClaim) -> None:
        if not isinstance(claim, BudgetClaim):
            raise TypeError("claim must be a BudgetClaim")
        if claim._account_id != self._account_id or claim.budget != self.budget:
            raise ValueError("budget claim belongs to a different account")


__all__ = ["BudgetAccount", "BudgetClaim", "BudgetHandle"]
