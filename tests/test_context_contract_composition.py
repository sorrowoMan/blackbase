from __future__ import annotations

from types import SimpleNamespace

from blackbase.context import collect_solver_contracts, detect_context_conflicts


class _Child:
    context_requires = ()
    context_provides = ("shared_metric",)
    context_mutates = ()
    context_cache = ()
    context_notes = "child contract"


class _Composite:
    context_requires = ()
    context_provides = ("shared_metric",)
    context_mutates = ()
    context_cache = ()
    context_notes = "composite publishes the child projection"
    context_contract_encapsulates_children = True

    def __init__(self) -> None:
        self.strategies = (
            SimpleNamespace(name="left", adapter=_Child()),
            SimpleNamespace(name="right", adapter=_Child()),
        )


def test_composite_can_encapsulate_child_context_writers() -> None:
    solver = SimpleNamespace(
        representation_pipeline=None,
        bias_module=None,
        adapter=_Composite(),
        plugin_manager=None,
    )

    contracts = collect_solver_contracts(solver)

    assert [name for name, _contract in contracts] == [
        "adapter",
        "adapter.strategy.left",
        "adapter.strategy.right",
    ]
    assert {contract.metadata["writer_scope"] for _name, contract in contracts} == {
        "adapter"
    }
    assert detect_context_conflicts(contracts) == []


def test_unencapsulated_children_remain_visible_to_conflict_detection() -> None:
    adapter = _Composite()
    adapter.context_contract_encapsulates_children = False
    solver = SimpleNamespace(
        representation_pipeline=None,
        bias_module=None,
        adapter=adapter,
        plugin_manager=None,
    )

    conflicts = detect_context_conflicts(collect_solver_contracts(solver))

    assert conflicts == [
        "shared_metric: adapter, adapter.strategy.left, adapter.strategy.right"
    ]


def test_case_level_contract_is_collected_without_solver_components() -> None:
    case = SimpleNamespace(
        context_requires=(),
        context_provides=(),
        context_mutates=(),
        context_cache=(),
        context_notes="Consumes an external grant without publishing Context fields.",
        representation_pipeline=None,
        bias_module=None,
        adapter=None,
        plugin_manager=None,
    )

    contracts = collect_solver_contracts(case)

    assert [name for name, _contract in contracts] == ["solver"]
    assert contracts[0][1].notes is not None
