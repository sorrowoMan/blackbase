from __future__ import annotations

import numpy as np
import pytest

from blackbase.context import detach_context_value
from blackbase.resources import DataRef


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
    detached.metadata["nested"]["values"].append(2)

    assert detached is not ref
    assert ref.metadata == {"nested": {"values": [1]}}


def test_detach_context_value_rejects_unknown_and_cycles() -> None:
    class MutableAuthority:
        pass

    with pytest.raises(TypeError, match="unsupported context value type"):
        detach_context_value(MutableAuthority())

    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(TypeError, match="cyclic context value"):
        detach_context_value(cyclic)
