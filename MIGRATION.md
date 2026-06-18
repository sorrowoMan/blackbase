# BlackBase Migration Guide

## Overview

BlackBase is the shared Project / Case / Scaffold / L0 substrate for NSGABlack and MLBlack. It owns shared runtime contracts such as context, snapshot, resource grants, local L0 pools, pipeline slot kernels, and cross-case protocol types.

Architectural boundary:

- `blackbase` owns shared substrate contracts.
- `nsgablack` owns optimization/search semantics.
- `mlblack` owns machine-learning semantics.
- orchestration and resource grants belong to the shared substrate, not to either semantic layer privately.

## Migration Timeline

| Phase | Description | Target |
|-------|-------------|--------|
| Phase 1 | Use adapters (backward compatible) | Current |
| Phase 2 | Direct imports from blackbase | v0.2.0 |
| Phase 3 | Remove legacy wrappers | v1.0.0 |

---

## Phase 1: Using Adapters (Current)

The adapter layer provides full backward compatibility with deprecation warnings.

### NSGABlack

#### Context Module

**Before (Legacy):**
```python
from nsgablack.core.state.context_keys import normalize_context_key
from nsgablack.core.state.context_contracts import ContextContract
from nsgablack.core.state.context_store import ContextStore, InMemoryContextStore
```

**After (Adapter):**
```python
# Option 1: Use adapter (with deprecation warning)
from blackbase.adapters.nsgablack.context import (
    normalize_context_key,
    ContextContract,
    ContextStore,
    InMemoryContextStore,
)

# Option 2: Direct import from blackbase (recommended for new code)
from blackbase.context import (
    normalize_context_key,
    ContextContract,
    ContextStore,
)
```

#### Resources Module

**Before (Legacy):**
```python
from nsgablack.core.resources.model import (
    DataRef,
    ResourceRequirement,
    WorkerDescriptor,
    TaskEnvelope,
)
```

**After (Adapter):**
```python
# Option 1: Use adapter
from blackbase.adapters.nsgablack.resources import (
    DataRef,
    ResourceRequirement,
    WorkerDescriptor,
    TaskEnvelope,
)

# Option 2: Direct import from blackbase
from blackbase.resources import (
    DataRef,
    ResourceRequirement,
    WorkerDescriptor,
    TaskEnvelope,
)
```

#### Kernel/Representation Module

**Before (Legacy):**
```python
from nsgablack.representation.base import RepresentationPipeline
```

**After (Adapter):**
```python
# Use adapter with backward-compatible wrapper
from blackbase.adapters.nsgablack.kernel import RepresentationPipeline
```

---

### MLBlack

#### Context Module

**Before (Legacy):**
```python
from mlblack.core.context_contracts import ContextContract
from mlblack.core.stores import InMemoryContextStore
```

**After (Adapter):**
```python
# Option 1: Use adapter
from blackbase.adapters.mlblack.context import (
    ContextContract,
    InMemoryContextStore,
)

# Option 2: Direct import from blackbase
from blackbase.context import ContextContract
```

#### Resources Module

**Before (Legacy):**
```python
from mlblack.core.resources._resources import ResourceContext
```

**After (Adapter):**
```python
# Option 1: Use adapter
from blackbase.adapters.mlblack.resources import ResourceContext

# Option 2: Direct import from blackbase
from blackbase.resources import ResourceContext
```

---

## Phase 2: Direct Imports (v0.2.0)

After adapter deprecation warnings are resolved, migrate to direct imports:

```python
# Replace
from blackbase.adapters.nsgablack.context import ContextStore

# With
from blackbase.context import ContextStore
```

---

## API Mapping Reference

### Context Keys

| Legacy (NSGABlack) | Legacy (MLBlack) | BlackBase |
|-------------------|-------------------|-----------|
| `requires` | `context_requires` | `requires` |
| `provides` | `context_provides` | `provides` |
| `mutates` | `context_mutates` | `mutates` |
| `cache` | `context_cache` | `cache` |

Both naming conventions are supported in BlackBase.

### Context Store

| Legacy | BlackBase |
|--------|-----------|
| `InMemoryContextStore` | `ContextStore` (default backend) |
| `RedisContextStore` | `RedisContextStore` |
| `create_context_store()` | `create_context_store()` |

### Resources

| Legacy (NSGABlack) | BlackBase |
|-------------------|-----------|
| `DataRef` | `DataRef` |
| `ResourceRequirement` | `ResourceRequirement` |
| `WorkerDescriptor` | `WorkerDescriptor` |
| `TaskEnvelope` | `TaskEnvelope` |
| `TaskResult` | `TaskResult` |
| `PoolScheduler` | `PoolScheduler` |
| `PoolTask` | `PoolTask` |
| `PoolResult` | `PoolResult` |

### Shared Types

| Legacy (MLBlack) | BlackBase |
|------------------|-----------|
| `UnknownState` | `UnknownState` |
| `Feedback` | `Feedback` |
| `PopulationSnapshot` | `PopulationSnapshot` |
| `TrainerResult` | `TrainerResult` |

### Kernel

| Legacy (NSGABlack) | BlackBase |
|-------------------|-----------|
| `RepresentationPipeline` | `PipelineKernelBuild` |
| N/A | `PipelineSlotSpec` |
| N/A | `PipelineSpec` |
| N/A | `OrchestrationPolicy` |

---

## Key Changes

### 1. Unified Context Keys

All context keys are now in a single registry with both naming styles supported:

```python
from blackbase.context import normalize_context_key

# Both work now
key1 = normalize_context_key("candidate.model")      # MLBlack style
key2 = normalize_context_key("candidate.unknown_state")  # Canonical
```

### 2. Factory Functions for Stores

Use factory functions instead of direct class instantiation:

```python
# Before
store = InMemoryContextStore()

# After
from blackbase.context import create_context_store
store = create_context_store(backend="memory")
```

### 3. Immutable Protocol Objects

All protocol objects are now frozen (immutable):

```python
from blackbase.resources import DataRef

ref = DataRef(uri="s3://bucket/model.pt")
# ref.uri = "new-uri"  # Raises FrozenInstanceError
```

### 4. Pipeline Specs

Pipeline definitions use declarative specs:

```python
from blackbase.kernel import PipelineSpec, PipelineSlotSpec, build_pipeline_kernel

spec = PipelineSpec(
    key="my_pipeline",
    slots=[
        PipelineSlotSpec(slot="init", operators=["init_op"]),
        PipelineSlotSpec(slot="mutate", mode="parallel", operators=["mut1", "mut2"]),
    ],
)

kernel = build_pipeline_kernel(spec, operator_registry={...})
```

### 5. Shared Candidate State

`UnknownState` accepts both old metadata spellings:

```python
from blackbase.types import UnknownState

state_a = UnknownState(values=[1, 2], meta={"source": "old"})
state_b = UnknownState(values=[1, 2], metadata={"source": "new"})
state_c = state_b.with_values([3, 4], stage="repair")
```

Use `blackbase.types.UnknownState` for shared Case payloads. Keep framework-specific model, codec, trainer, or solver semantics in `nsgablack` / `mlblack`.

---

## Deprecation Warnings

When using adapters, you'll see warnings like:

```
DeprecationWarning: nsgablack.core.state.context_keys is deprecated. 
Use blackbase.context instead. This will be removed in 1.0.0.
```

To see all deprecation warnings:

```python
import warnings
warnings.filterwarnings("always", category=DeprecationWarning)
```

---

## Testing Migration

### Before Migration

```bash
# Run existing tests
pytest tests/
```

### During Migration

```bash
# Enable all warnings
python -W always::DeprecationWarning your_script.py

# Check for adapter usage
grep -r "blackbase.adapters" your_code/
```

### After Migration

```bash
# Ensure no adapter imports remain
grep -r "blackbase.adapters" your_code/
# Should return nothing

# Run tests with direct imports
pytest tests/
```

---

## Getting Help

- **Issues**: https://github.com/your-org/blackbase/issues
- **Discussions**: https://github.com/your-org/blackbase/discussions
- **Documentation**: https://blackbase.readthedocs.io/
