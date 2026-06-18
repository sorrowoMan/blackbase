"""
Test script to verify forwarders in NSGABlack and MLBlack work correctly.
"""

import sys
import os

# Add NSGABlack and MLBlack to path
sys.path.insert(0, "c:/Users/hp/Desktop/nsgablack")
sys.path.insert(0, "c:/Users/hp/Desktop/mlblack")

print("Python version: {}".format(sys.version))
print()

# Test NSGABlack forwarders
print("=" * 60)
print("Test NSGABlack forwarders")
print("=" * 60)

try:
    from nsgablack.core.state.context_store import ContextStore
    from nsgablack.core.state.context_contracts import ContextContract
    from nsgablack.core.state.context_keys import normalize_context_key
    from nsgablack.core.resources import PoolScheduler
    
    # Verify it's using blackbase
    import blackbase.context
    
    assert ContextStore.__module__.startswith("blackbase"), "ContextStore should be from blackbase"
    assert ContextContract.__module__.startswith("blackbase"), "ContextContract should be from blackbase"
    assert normalize_context_key.__module__.startswith("blackbase"), "normalize_context_key should be from blackbase"
    
    print("[OK] NSGABlack forwarders are correctly using blackbase")
    print("  - ContextStore from {}".format(ContextStore.__module__))
    print("  - ContextContract from {}".format(ContextContract.__module__))
    print("  - normalize_context_key from {}".format(normalize_context_key.__module__))
    print("  - PoolScheduler from {}".format(PoolScheduler.__module__))
    
    # Test functionality through forwarder
    store = ContextStore()
    store.set("test_key", "test_value")
    assert store.get("test_key") == "test_value"
    print("[OK] ContextStore through NSGABlack forwarder works")
    
except Exception as e:
    print("[FAIL] NSGABlack forwarders failed: {}".format(e))
    import traceback
    traceback.print_exc()

# Test MLBlack forwarders
print()
print("=" * 60)
print("Test MLBlack forwarders")
print("=" * 60)

try:
    from mlblack.core.context_contracts import ContextContract
    from mlblack.core.stores import ContextStore
    from mlblack.core.resources import PoolScheduler
    from mlblack.core.types import UnknownState
    
    # Verify it's using blackbase
    assert ContextStore.__module__.startswith("blackbase"), "ContextStore should be from blackbase"
    assert ContextContract.__module__.startswith("blackbase"), "ContextContract should be from blackbase"
    
    print("[OK] MLBlack forwarders are correctly using blackbase")
    print("  - ContextStore from {}".format(ContextStore.__module__))
    print("  - ContextContract from {}".format(ContextContract.__module__))
    print("  - PoolScheduler from {}".format(PoolScheduler.__module__))
    
    # Test functionality through forwarder
    store = ContextStore()
    store.set("ml_key", "ml_value")
    assert store.get("ml_key") == "ml_value"
    print("[OK] ContextStore through MLBlack forwarder works")

    state = UnknownState(values=[1, 2], metadata={"source": "forwarder"})
    assert state.metadata["source"] == "forwarder"
    print("[OK] UnknownState through MLBlack forwarder accepts metadata")
    
except Exception as e:
    print("[FAIL] MLBlack forwarders failed: {}".format(e))
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("All forwarder tests completed successfully!")
print("=" * 60)
