# BlackBase 0.3 迁移说明

## 版本与边界

BlackBase 0.3 是一次有意的边界收口，不再提供旧仓库路径的转发层。

- `blackbase==0.3.x`：Project / Case / Scaffold、L0 资源、Context / Snapshot、公共调用绑定、Pipeline 编排、Catalog 与运行协议。
- `nsgablack==0.3.x`：Solver、搜索 Adapter、Representation、Pareto、目标与约束等优化语义。
- `mlblack==0.4.x`：LearningSolver facade、DataView、Codec、Head、Problem、Provider、Artifact 等 ML 语义。

下游依赖应声明：

```toml
blackbase = ">=0.3.6,<0.4.0"
```

0.3.6 固化 canonical Case 的无兼容层规则，Doctor 会拒绝私有 run/fit wrapper 与兼容装配源码；0.3.5 固化 Feedback 的递归不可变边界，删除共享层默认标量化，给 state transition 增加 operand/slot state-kind 契约，并要求 build-check 关闭已构建 Case；0.3.4 固化 UnknownState/CandidateBatch 的不可变语义身份；0.3.3 增加 CandidateBatch 语义/数值双视图、StateRef trajectory 与正式 StateRelease；0.3.2 增加 Problem Evaluation Provider 的正式状态协议：`StateRef` 只表示
Provider 进程内活状态；Adapter 通过版本栅栏 `StateTransitionRequest` 选择更新机制，
再通过 `StateMaterializationRequest` 导出 `UnknownState` 或 `DataRef`。旧进程活引用
不能作为 checkpoint 或 Artifact 恢复。

## 需要修改的导入

共享底座类型必须直接从 BlackBase 导入：

```python
from blackbase.context import ContextContract, ContextStore, SnapshotStore
from blackbase.resources import PoolScheduler, ResourceContext
from blackbase.project import CaseRunRequest, CaseRunResult, CaseStageRunner
from blackbase.call_binding import CallCandidate, invoke_bound_once
```

以下转发路径已删除：

- `blackbase.adapters.*`
- `nsgablack.core.resources.*`
- `nsgablack.utils.context.*`
- `mlblack.core.resources.*`
- `mlblack.core.context_contracts`
- `mlblack.core.context_keys`
- `mlblack.core.stores`
- `blackbase.compat`

`nsgablack.core.state` 仍是优化层的正式状态 API；它包含 incumbent 与 candidate provenance 等优化语义，不是 BlackBase 转发兼容层。

## 公共调用绑定

旧的 `CallForm` / `invoke_compatible_once()` 已删除。需要兼容多个明确签名时，使用公共绑定协议：

```python
from blackbase.call_binding import CallCandidate, invoke_bound_once

result = invoke_bound_once(
    operator,
    (
        CallCandidate(args=(control, value), label="control,value"),
        CallCandidate(args=(value,), label="value"),
    ),
)
```

该协议先用函数签名绑定，再执行唯一匹配项。函数体内部抛出的 `TypeError` 不会触发第二次调用。

## Pipeline 与 Plugin

- 使用 `PipelineOrchestrator.call_operator()` 与 `run_policy()`；旧私有别名已删除。
- Plugin 的公共基类是 `PluginBase`；`Plugin` 别名已删除。
- ML `Capability` 直接继承 `PluginBase`，不再通过 `CapabilityPluginAdapter` 包装。
- Adapter 运行投影只使用 `get_runtime_context_projection(control)`；旧 `get_context_projection()` 钩子已删除。

## ContextContract

构造参数只接受紧凑字段：

```python
ContextContract(
    requires=("candidate",),
    optional=("metrics",),
    provides=("feedback",),
    mutates=(),
    cache=(),
)
```

组件类仍以 `context_requires`、`context_optional`、`context_provides`、`context_mutates`、`context_cache` 声明静态契约。旧的实例属性别名与构造参数别名已删除。

## PoolScheduler

`submit()` 只返回任务函数的原始结果：

```python
handle = pool.submit(fn, value, resource_permits=1, task_id="job-1")
result = handle.result()
```

旧 `PoolTaskResult` 包装已删除；`task_id` 只属于任务句柄元数据。

## Representation 与 incumbent

- nsgablack Representation 的正式初始化入口为 `init(context)`；`context` 必须包含正式 Problem。
- `initialize()` / `transform()` 原型入口已删除。
- `IncumbentState` 是最佳候选、目标、约束、分数、策略和来源的唯一原子权威状态。
- `set_best_snapshot()` 与零散 best 字段回退已删除；不完整的旧 checkpoint 不再被伪造成可验证 incumbent。
- checkpoint v2 保存选择策略、退化审计、投影审计和 run lineage；v1 仅通过显式迁移读取。

## 完整子 Case

父 Case 调用子 Case 必须使用公共 `CaseRunRequest` / `CaseRunResult` 信封。BlackBase 负责：

- 父子 lineage 与 attempt 身份；
- deadline / cancellation 链；
- `ChildResourceGrant` 与预算委托结算；
- Artifact/DataRef 注入；
- 串行、进程和外部 Worker 的统一执行信封；
- 结构化失败与 Manifest 记录。

`TaskInnerRuntimeEvaluator` 一类低层组件调用器不能替代完整子 Case；当子任务需要独立身份、资源、预算、Artifact 或失败信封时，必须升级到 Case 协议。

## 验证

```powershell
python -m pytest -q
python -m nsgablack project doctor --path . --strict --format problem
python -m nsgablack catalog list --profile framework-core --kind adapter
python -m nsgablack catalog list --profile default --kind example
```

升级后应再搜索已删除路径，不能依赖导入失败时的隐式回退。
