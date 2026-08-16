from __future__ import annotations

import importlib
import warnings

import pytest

from blackbase.adapters.mlblack.plugin import Capability, CapabilityPluginAdapter
from blackbase.context import ContextContract, ContextStore, normalize_context_keys
from blackbase.plugin import PluginBase, PluginManager
from blackbase.project.scaffold import _build_entry_template, _nsgablack_plugin_template


def test_nsgablack_adapter_package_does_not_eagerly_warn() -> None:
    import blackbase.adapters.nsgablack as adapter_package

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(adapter_package)

    assert not [item for item in caught if issubclass(item.category, DeprecationWarning)]


def test_explicit_legacy_plugin_adapter_still_warns() -> None:
    with pytest.warns(DeprecationWarning, match="blackbase.plugin"):
        module = importlib.import_module("blackbase.adapters.nsgablack.plugin")
        importlib.reload(module)


def test_new_nsgablack_scaffold_sources_use_canonical_plugin_import() -> None:
    plugin_source = _nsgablack_plugin_template("TracePlugin", "trace", "plugin")
    build_source = _build_entry_template("solver", framework="nsgablack")

    assert "from blackbase.plugin import Plugin" in plugin_source
    assert "from blackbase.plugin import Plugin" in build_source
    assert "nsgablack.plugins.base" not in plugin_source + build_source


def test_context_store_supports_declared_mapping_protocol() -> None:
    store = ContextStore()

    store["value"] = 3
    assert store["value"] == 3
    assert dict(store) == {"value": 3}

    del store["value"]
    assert "value" not in store


def test_context_contract_reports_unknown_keys() -> None:
    contract = ContextContract(
        requires=("generation", "custom.unregistered"),
        provides=("feedback.metrics",),
        requires_metrics=("loss", "custom_metric"),
    )

    assert contract.unknown_keys() == ("custom.unregistered",)
    assert contract.unknown_metric_keys() == ("custom_metric",)


def test_ml_data_key_aliases_normalize_to_registered_canonical_keys() -> None:
    assert normalize_context_keys(
        ("data.x_train", "data.x_valid", "problem.data.x_train", "pipeline.slot_context")
    ) == (
        "data.X_train",
        "data.X_valid",
        "problem.data.X_train",
        "pipeline.slot_context",
    )


def test_context_build_chains_replacement_dicts_and_tracks_writers() -> None:
    class First(PluginBase):
        def __init__(self) -> None:
            super().__init__("first")

        def on_context_build(self, context):
            return {**context, "first": 1}

    class Second(PluginBase):
        def __init__(self) -> None:
            super().__init__("second")

        def on_context_build(self, context):
            assert context["first"] == 1
            return {**context, "second": 2}

    manager = PluginManager()
    manager.register(First())
    manager.register(Second())

    context = manager.on_context_build({"seed": 0})

    assert context == {"seed": 0, "first": 1, "second": 2}
    assert manager._context_build_writers == {
        "first": "plugin.first",
        "second": "plugin.second",
    }


def test_context_build_skips_plugins_with_failed_attach() -> None:
    class Failed(PluginBase):
        def __init__(self) -> None:
            super().__init__("failed")
            self._attach_failed = True

        def on_context_build(self, context):
            raise AssertionError("failed plugin must not be dispatched")

    manager = PluginManager(strict=True)
    manager.register(Failed())

    assert manager.on_context_build({"ok": True}) == {"ok": True}


def test_mlblack_capability_adapter_preserves_trainer_context_and_rows() -> None:
    class LegacyCapability(Capability):
        name = "legacy"
        context_provides = ("legacy.seen",)

        def __init__(self) -> None:
            # Legacy mlblack capabilities commonly did not call PluginBase.__init__.
            self.events = []

        def on_fit_start(self, trainer, context):
            self.events.append(("fit_start", trainer, dict(context)))

        def on_step_start(self, trainer, context, row):
            self.events.append(("step_start", trainer, dict(context), dict(row)))

        def on_evaluate_start(self, trainer, candidate, context):
            self.events.append(("evaluate_start", trainer, candidate, dict(context)))

        def on_evaluate_end(self, trainer, candidate, feedback, context):
            self.events.append(
                ("evaluate_end", trainer, candidate, feedback, dict(context))
            )

        def on_step_end(self, trainer, context, row):
            self.events.append(("step_end", trainer, dict(context), dict(row)))

        def on_fit_end(self, trainer, context, report):
            self.events.append(("fit_end", trainer, dict(context), dict(report)))

        def on_error(self, trainer, error, context):
            self.events.append(("error", trainer, error, dict(context)))

    class Trainer:
        def __init__(self) -> None:
            self.context_store = {"run_name": "demo"}
            self.history = [{"step": 3, "score": 0.5}]

        def build_context(self):
            return {**self.context_store, "built": True}

    trainer = Trainer()
    capability = LegacyCapability()
    adapter = CapabilityPluginAdapter(capability)
    manager = PluginManager(strict=True)
    manager.register(adapter)

    manager.on_solver_init(trainer)
    manager.on_generation_start(3)
    manager.on_evaluate_start("candidate", {"phase": "evaluate"})
    manager.on_evaluate_end("candidate", "feedback", {"phase": "evaluate"})
    manager.on_generation_end(3)
    manager.on_solver_finish({"report": {"status": "ok"}})
    error = RuntimeError("boom")
    manager.on_error(error, {"phase": "error"})

    assert [event[0] for event in capability.events] == [
        "fit_start",
        "step_start",
        "evaluate_start",
        "evaluate_end",
        "step_end",
        "fit_end",
        "error",
    ]
    assert all(event[1] is trainer for event in capability.events)
    assert capability.events[1][3] == {"step": 3}
    assert capability.events[4][3] == {"step": 3, "score": 0.5}
    assert capability.events[5][3] == {"status": "ok"}
    assert capability.events[6][2] is error
    assert adapter.get_context_contract()["provides"] == ("legacy.seen",)
