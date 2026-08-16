from __future__ import annotations

import pytest

from blackbase.contracts import BatchDisposition


def test_batch_disposition_projects_parent_indices_into_child_range() -> None:
    parent = BatchDisposition(
        proposed_count=7,
        accepted_indices=(0, 2, 5, 6),
        reason="budget",
    )

    child = parent.for_range(2, 6)

    assert child.proposed_count == 4
    assert child.accepted_indices == (0, 3)
    assert child.rejected_indices == (1, 2)
    assert child.metadata["parent_range"] == (2, 6)


def test_batch_disposition_rejects_ambiguous_index_order() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        BatchDisposition(proposed_count=3, accepted_indices=(2, 1))


def test_batch_disposition_prefix_is_explicit_and_bounded() -> None:
    disposition = BatchDisposition.prefix(
        proposed_count=5,
        accepted_count=3,
        reservation_id="claim-1",
    )

    assert disposition.accepted_indices == (0, 1, 2)
    assert disposition.rejected_indices == (3, 4)
    assert disposition.reservation_id == "claim-1"

    with pytest.raises(ValueError, match="between zero and proposed_count"):
        BatchDisposition.prefix(proposed_count=2, accepted_count=3)
