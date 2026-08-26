from __future__ import annotations

import numpy as np
import pytest

from blackbase.context import (
    detach_context_value,
    record_context_event,
    replay_context,
)
from blackbase.resources import DataRef
from blackbase.state_ref import StateRef


def test_detach_context_value_recursively_breaks_mutable_aliases() -> None:
    array = np.asarray([1.0, 2.0])
    source = {"nested": [{"array": array, "items": [1, 2]}]}

    detached = detach_context_value(source)
    detached["nested"][0]["array"][0] = 99.0
    detached["nested"][0]["items"].append(3)

    assert array.tolist() == [1.0, 2.0]
    assert source["nested"][0]["items"] == [1, 2]


def test_detach_context_value_rebuilds_data_ref_metadata() -> None:
    ref = DataRef(uri="artifact://model", metadata={"nested": {"values": [1]}})

    detached = detach_context_value(ref)
    detached_payload = detached.as_dict()
    detached_payload["metadata"]["nested"]["values"].append(2)

    assert detached is not ref
    assert ref.as_dict()["metadata"] == {"nested": {"values": [1]}}
    assert detached.as_dict()["metadata"] == {"nested": {"values": [1]}}


def test_detach_context_value_rebuilds_immutable_state_ref() -> None:
    ref = StateRef(
        provider_id="provider/v1",
        state_id="parameters-1",
        metadata={"nested": {"candidate_index": 0}},
    )

    detached = detach_context_value({"state_ref": ref})

    assert detached["state_ref"] is not ref
    assert detached["state_ref"].as_dict() == ref.as_dict()


def test_detach_context_value_rejects_unknown_and_cycles() -> None:
    class MutableAuthority:
        pass

    with pytest.raises(TypeError, match="unsupported context value type"):
        detach_context_value(MutableAuthority())

    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(TypeError, match="cyclic context value"):
        detach_context_value(cyclic)


def test_context_event_detaches_recorded_and_replayed_values() -> None:
    source = {"items": [1], "array": np.asarray([2.0])}
    context: dict[str, object] = {}
    record_context_event(
        context,
        kind="set",
        key="payload",
        value=source,
    )

    source["items"].append(9)
    source["array"][0] = 7.0
    event = context["context_events"][0]
    assert event["value"]["items"] == [1]
    assert event["value"]["array"].tolist() == [2.0]

    replayed = replay_context({}, context["context_events"], strict=True)
    replayed["payload"]["items"].append(3)
    replayed["payload"]["array"][0] = 5.0
    assert event["value"]["items"] == [1]
    assert event["value"]["array"].tolist() == [2.0]


def test_context_event_validation_fails_closed_in_strict_replay() -> None:
    context: dict[str, object] = {}
    with pytest.raises(ValueError, match="unsupported context event kind"):
        record_context_event(
            context,
            kind="unknown",
            key="value",
            value=1,
        )
    with pytest.raises(TypeError, match="update event value"):
        record_context_event(
            context,
            kind="update",
            key=None,
            value=[("value", 1)],
        )

    invalid = [{"kind": "unknown", "key": "value", "value": 1}]
    with pytest.raises(ValueError, match="unsupported context event kind"):
        replay_context({}, invalid, strict=True)
    assert replay_context({}, invalid, strict=False) == {}


def test_strict_replay_does_not_mutate_nested_base_before_late_failure() -> None:
    base = {"items": [1]}
    events = [
        {"kind": "append", "key": "items", "value": 2},
        {"kind": "unknown", "key": "items", "value": 3},
    ]

    with pytest.raises(ValueError, match="unsupported context event kind"):
        replay_context(base, events, strict=True)

    assert base == {"items": [1]}
