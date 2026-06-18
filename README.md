# BlackBase

`blackbase` 是 `nsgablack` 与 `mlblack` 共享的底座包。

它不是新的优化框架，也不是新的机器学习框架；它负责两边共同依赖的 Project / Case / Scaffold / L0 substrate，以及轻量协议对象。

## 架构口径

- `blackbase`：共享 substrate。提供 context、snapshot、resource grant、L0 pool、pipeline slot kernel、跨 Case payload types。
- `nsgablack`：优化搜索语义层。负责 solver lifecycle、candidate search、multi-objective/Pareto、adapter strategy。
- `mlblack`：机器学习语义层。负责 DataView、Spec、Codec、Head、Problem、Trainer、Provider、Artifact。

编排和资源授权属于 `blackbase` substrate，不属于 `nsgablack` 或 `mlblack` 任一侧的私有能力。

## 当前组件

### Context / Snapshot

- `ContextStore` / `create_context_store`
- `SnapshotStore` / `create_snapshot_store`
- `ContextContract`
- canonical context key registry
- context schema / replay helpers

### L0 Resources

- `ResourceContext`
- `ResourceRequirement`
- `WorkerDescriptor`
- `TaskEnvelope` / `TaskResult`
- `PoolScheduler` / `PoolTask` / `PoolResult`
- local resource probe helpers

### Kernel

- `PipelineSpec`
- `PipelineSlotSpec`
- `PipelineOrchestrator`
- `build_pipeline_kernel`
- legacy `RepresentationPipeline` wrapper

### Shared Types

- `UnknownState`
- `Feedback`
- `PopulationSnapshot`
- `TrainerResult`

这些类型用于跨 `nsgablack` / `mlblack` Case surface 传递轻量候选状态和反馈，不承载任一语义层的私有逻辑。

## 安装

```bash
pip install -e .

# 可选 Redis 支持
pip install -e ".[redis]"

# 开发依赖
pip install -e ".[dev]"
```

## 最小用法

```python
from blackbase.context import build_minimal_context, create_snapshot_store
from blackbase.kernel import PipelineSpec, PipelineSlotSpec, build_pipeline_kernel
from blackbase.resources import PoolScheduler, ResourceContext
from blackbase.types import UnknownState

ctx = build_minimal_context(
    generation=1,
    individual_id=0,
    constraints=[],
    constraint_violation=0.0,
)

state = UnknownState(values=[1.0, 2.0], metadata={"source": "demo"})

pool = PoolScheduler(total_threads=2)
result = pool.submit(lambda x: x + 1, 41)
assert result.result() == 42

snapshot_store = create_snapshot_store(backend="memory")
handle = snapshot_store.write({"state": state.as_array()}, key="demo")
```

## 迁移阶段

当前阶段是兼容迁移：

1. `nsgablack` / `mlblack` 的旧 import path 可以通过 forwarder 继续工作。
2. 新代码优先直接从 `blackbase.context`、`blackbase.resources`、`blackbase.kernel`、`blackbase.types` 导入。
3. 未来两边重复的 runtime/resource/helper/type 实现应逐步收敛到 blackbase。

