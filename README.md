# BlackBase

`blackbase` 是 `nsgablack` 与 `mlblack` 共享的底座包。

它不是新的优化框架，也不是新的机器学习框架；它负责两边共同依赖的 Project / Case / Scaffold / L0 substrate，以及轻量协议对象。

## 架构口径

- `blackbase`：共享 substrate。提供 context、snapshot、resource grant、L0 pool、pipeline slot kernel、跨 Case payload types。
- `nsgablack`：优化搜索语义层。负责 solver lifecycle、candidate search、multi-objective/Pareto、adapter strategy。
- `mlblack`：机器学习语义层。负责 DataView、Spec、Codec、Head、Problem、Trainer、Provider、Artifact。

编排和资源授权属于 `blackbase` substrate，不属于 `nsgablack` 或 `mlblack` 任一侧的私有能力。

## 标准 Case 契约

Solver 与 Trainer 使用同一种 Case 目录形状。`.case` 中的 `kind=solver|trainer`
只表达语义与 Catalog 分类，不改变装配入口：

- `build_solver.py` 是唯一规范装配入口，必须定义 `build_solver()`。
- `run_solver.py` 是唯一规范 CLI/debug 入口。
- `build_trainer.py` 只能薄别名到 `build_solver()`。
- `run_trainer.py` 只能薄别名到 `run_solver.main`。
- 规范 builder 接受 `resource_context` 与 `component_overrides`，供 Project L0
  授权和嵌套 Case 组合使用。

Project 运行时和 Doctor 始终加载、校验规范入口；`kind=trainer` 只影响默认执行
语义（例如优先 `fit()`）和审计信息。

## 当前组件

### Context / Snapshot

- `ContextStore` / `create_context_store`
- built-in ContextStore backends expose atomic `apply_patch(set/delete)` updates
- `SnapshotStore` / `create_snapshot_store`
- `ContextContract`
- canonical context key registry
- context schema / replay helpers
- canonical `best_candidate_ref` for keeping oversized incumbent candidates in SnapshotStore
- canonical Adapter-local best projection keys: `adapter_best_x`, `adapter_best_objectives`, `adapter_best_score`

### L0 Resources

- `ResourceContext`
- `ResourceRequirement`
- `BudgetAccount` / `BudgetClaim`
- `BudgetReservation` / `BudgetSnapshot`
- `WorkerDescriptor`
- `TaskEnvelope` / `TaskResult`
- `PoolScheduler` / `PoolTask` / `PoolResult`
- local resource probe helpers

### Kernel

- `PipelineSpec`
- `PipelineSlotSpec`
- `PipelineOrchestrator`
- `build_pipeline_kernel`

### 递归 Case 调用协议

- `CaseRunRequest` / `CaseRunResult`：串行、进程池、外部 worker 与父子调用共用的版本化信封。
- `CaseRunIdentity`：记录 Project run、root run、父 Case、调用 ID、尝试次数与嵌套深度。
- `ExecutionControl` / `CancellationRef`：传递绝对 deadline 和完整取消祖先链。
- `ChildResourceGrant` / `BudgetHandle`：从父 Case 已获授权的 L0 grant 与预算中原子划分子授权。
- `CaseExecutor` / `CaseInvoker`：唯一的标准 Case 装配、执行与递归调用边界。

父 Case 不加载子 Case 的 builder，也不自行拼接资源上下文。标准执行器会把
`case_runtime` 注入父 Case；父 Case 只需提交完整子请求：

```python
from blackbase.project import CaseRunRequest


def run(self):
    child = self.case_runtime.invoke(
        CaseRunRequest(
            project_name="forecast_system",
            stage_name="inner_search",
            case_name="fit_surrogate",
            resource_request={"workers": 1, "threads": 2},
            budget_request={"evaluations": 100},
            inputs={"dataset_ref": "artifact://prepared/train"},
        )
    )
    if not child.ok:
        raise RuntimeError(child.error)
    return {"child_result": child.as_dict()}
```

子 Case 仍是一个可独立运行、具有规范 `build_solver.py` 的完整 Case。并行子调用
会竞争父 grant 内的子资源池；超出父授权的请求会被拒绝，等待中的调用会持续检查
deadline/cancellation。预算通过可序列化 handle 委派，子 Case 未使用的额度会返还父预算。

### Plugin / Capability 生命周期

- `PluginBase` / `PluginManager` 提供两边共享的注册、优先级、严格模式、事件调度与评估 hook。
- `on_context_build` 按插件顺序链式传递 context；插件返回的新字典会成为下一个插件的输入。
- `CapabilityPluginAdapter` 把旧 mlblack `on_fit_* / on_step_*` Capability 映射到统一 Plugin 生命周期，同时保留 Trainer、context、step row 与 report。
- population snapshot、Pareto 等搜索专用 helper 仍留在 nsgablack；DataView、Trainer、Artifact 等 ML 语义仍留在 mlblack。

### Shared Types

- `UnknownState`
- `Feedback`
- `PopulationSnapshot`
- `TrainerResult`
- `SolverResult`

`SolverResult` separates Case execution success from optimization semantics.
Its typed terminal fields are `solve_status`, `termination_reason`,
`feasibility`, and `SolveQuality` (approximation, gaps, bound, and metrics).
`SolveQuality.approximate` is tri-state: `None` means no quality evidence,
while `True`/`False` are explicit claims. Contradictory terminal states and
quality claims are rejected by one centralized status-rule matrix. Feasibility
evidence is checked in both directions: feasible semantics cannot attach a
positive best violation, while `infeasible`/`no_solution` cannot attach a
feasible authoritative best; `unbounded + infeasible` is also rejected.
Best fields are optional and must be declared by the Solver; a Pareto front can
be delivered inline or through a real `DataRef` published by the Project
artifact authority.

这些类型用于跨 `nsgablack` / `mlblack` Case surface 传递轻量候选状态和反馈，不承载任一语义层的私有逻辑。

## 共享协议版本

当前三仓共享底座基线为 `blackbase 0.3.x`。`nsgablack` 与 `mlblack`
必须声明 `blackbase>=0.3.0,<0.4.0`。

0.3 完成了共享底座的干净边界：删除两侧资源、Context 与 Adapter 转发树；
公共调用绑定、Pipeline 编排、Catalog、Case stage、任务运行后端均由 BlackBase
直接提供。Adapter 的唯一运行投影入口是
`get_runtime_context_projection(control)`，组合投影的健康、因果摘要和叶子 writer
来源均进入有界正式信封。具体破坏性变更见 [MIGRATION.md](MIGRATION.md)。

包版本、共享类型 wire schema 与 Context schema 分别演进，不能互相代替。

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

## 公共 API 归属

Project / Case / Scaffold / L0、递归调用、资源与预算授权均直接从 `blackbase`
导入。`nsgablack` 和 `mlblack` 只保留带各自 framework 参数的 Project 语义入口，
不再维护私有 runtime 转发层。
