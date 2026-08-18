from blackbase.context.context_keys import (
    CANONICAL_CONTEXT_KEYS,
    KEY_ADAPTER_BEST_OBJECTIVES,
    KEY_ADAPTER_BEST_SCORE,
    KEY_ADAPTER_BEST_X,
    KEY_RUNTIME_PROJECTION_AUDIT,
    normalize_context_key,
)


def test_adapter_best_projection_keys_are_canonical() -> None:
    assert KEY_ADAPTER_BEST_X == "adapter_best_x"
    assert KEY_ADAPTER_BEST_OBJECTIVES == "adapter_best_objectives"
    assert KEY_ADAPTER_BEST_SCORE == "adapter_best_score"
    assert KEY_ADAPTER_BEST_X in CANONICAL_CONTEXT_KEYS
    assert KEY_ADAPTER_BEST_OBJECTIVES in CANONICAL_CONTEXT_KEYS
    assert KEY_RUNTIME_PROJECTION_AUDIT in CANONICAL_CONTEXT_KEYS
    assert normalize_context_key(" ADAPTER_BEST_X ") == KEY_ADAPTER_BEST_X
    assert normalize_context_key("adapterbestx") == "adapterbestx"
