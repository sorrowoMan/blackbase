from __future__ import annotations

from blackbase.context import ContextContract, ContextStore, normalize_context_keys
from blackbase.plugin import PluginBase, PluginManager, report_soft_error
from blackbase.context.context_keys import (
    KEY_METRICS,
    KEY_METRICS_SOFT_ERROR_COUNT,
    KEY_METRICS_SOFT_ERROR_LAST,
)
from blackbase.project.scaffold import _build_entry_template


def test_base_scaffold_source_has_no_downstream_framework_imports() -> None:
    build_source = _build_entry_template("solver", framework="blackbase")

    assert "from blackbase.plugin import PluginBase" in build_source
    assert "nsgablack" not in build_source
    assert "mlblack" not in build_source


def test_context_store_supports_declared_mapping_protocol() -> None:
    store = ContextStore()

    store["value"] = 3
    assert store["value"] == 3
    assert dict(store) == {"value": 3}

    del store["value"]
    assert "value" not in store


def test_soft_error_reporting_records_shared_context_audit() -> None:
    store = ContextStore()

    report_soft_error(
        component="solver",
        event="projection",
        exc=ValueError("broken"),
        context_store=store,
        min_interval_seconds=0,
    )

    metrics = store[KEY_METRICS]
    assert metrics[KEY_METRICS_SOFT_ERROR_COUNT]["solver.projection"] == 1
    assert metrics[KEY_METRICS_SOFT_ERROR_LAST]["error_type"] == "ValueError"


def test_context_contract_reports_unknown_keys() -> None:
    contract = ContextContract(
        requires=("generation", "custom.unregistered"),
        provides=("feedback.metrics",),
        requires_metrics=("loss", "custom_metric"),
    )

    assert contract.unknown_keys() == ("custom.unregistered",)
    assert contract.unknown_metric_keys() == ("custom_metric",)


def test_ml_data_key_casing_normalizes_to_registered_canonical_keys() -> None:
    assert normalize_context_keys(
        ("data.x_train", "data.x_valid", "problem.data.x_train", "pipeline.slot_context")
    ) == (
        "data.X_train",
        "data.X_valid",
        "problem.data.X_train",
        "pipeline.slot_context",
    )


def test_best_candidate_ref_is_a_canonical_context_key() -> None:
    from blackbase.context import CONTEXT_KEY_SET, normalize_context_key
    from blackbase.context.context_keys import KEY_BEST_CANDIDATE_REF

    assert KEY_BEST_CANDIDATE_REF == "best_candidate_ref"
    assert KEY_BEST_CANDIDATE_REF in CONTEXT_KEY_SET
    assert normalize_context_key(" BEST_CANDIDATE_REF ") == KEY_BEST_CANDIDATE_REF


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
