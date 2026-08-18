from __future__ import annotations

import pytest

from blackbase.call_binding import (
    CallCandidate,
    invoke_bound_once,
    invoke_bound_once_with_outcome,
)


def test_declared_call_candidate_binds_before_executing() -> None:
    calls: list[tuple[int, int]] = []

    def operator(value: int, *, scale: int) -> int:
        calls.append((value, scale))
        return value * scale

    outcome = invoke_bound_once_with_outcome(
        operator,
        (
            CallCandidate(args=(3,), label="missing_scale"),
            CallCandidate(args=(3,), kwargs={"scale": 4}, label="canonical"),
        ),
    )

    assert outcome.value == 12
    assert outcome.candidate_index == 1
    assert outcome.candidate_label == "canonical"
    assert calls == [(3, 4)]


def test_body_type_error_is_not_retried() -> None:
    calls = 0

    def operator(value: int, context: object = None) -> int:
        nonlocal calls
        calls += 1
        raise TypeError("operator body failed")

    with pytest.raises(TypeError, match="operator body failed"):
        invoke_bound_once(
            operator,
            (
                CallCandidate(args=(3, {}), label="with_context"),
                CallCandidate(args=(3,), label="without_context"),
            ),
        )

    assert calls == 1


def test_no_supported_form_fails_without_calling() -> None:
    calls = 0

    def operator(*, required: int) -> int:
        nonlocal calls
        calls += 1
        return required

    with pytest.raises(TypeError, match="cannot bind"):
        invoke_bound_once(operator, (CallCandidate(args=(1,), label="positional"),))

    assert calls == 0
