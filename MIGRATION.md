# BlackBase 0.3 迁移说明

## 版本与边界

BlackBase 0.3 是一次有意的边界收口，不再提供旧仓库路径的转发层。

- `blackbase==0.3.x`：Project / Case / Scaffold、L0 资源、Context / Snapshot、公共调用绑定、Pipeline 编排、Catalog 与运行协议。
- `nsgablack==0.3.x`：Solver、搜索 Adapter、Representation、Pareto、目标与约束等优化语义。
- `mlblack==0.4.x`：LearningSolver facade、DataView、Codec、Head、Problem、Provider、Artifact 等 ML 语义。

下游依赖应声明：

```toml
blackbase = ">=0.3.24,<0.4.0"
```

0.3.24 将 Case 最终发布拆成严格 prepare、cleanup/serialization、Ledger atomic seal 与
不可否决 finalized observation。PublicationReceipt 发放前会重新核对授权目录、文件存在性、
大小与 checksum；预留记录可通过 lease/fencing 回收崩溃遗留 provisional Artifact。
子 Case 预算委托改为 write-ahead intent，settlement journal 不再保存完整 ResourceContext。

0.3.23 新增 `policy="dag"` Project Stage。`depends_on` 声明普通依赖；同一 Stage 内
`producer.artifact` / `stage.producer.artifact` 形式的 `input_artifacts` 自动形成依赖边。
旧 `serial`、`parallel` 与 `external` Stage 行为不变。DAG Case 必须使用 canonical
`mode="build"`；环、未知节点、自依赖和失败依赖都会 fail-closed，而不会退化成声明顺序。

0.3.22 将跨 Case 输入升级为 `ArtifactBinding`：正式 `input_artifacts` 必须同时携带
`DataRef` 与 Case-finalization-sealed publication receipt，CaseExecutor 在构建业务对象前
回查 ledger 并重新校验物理文件摘要。所有普通 Artifact、最终 checkpoint 与 ML Result
统一进入 Case-wide prepare/seal 事务，只有结果序列化、teardown、scheduler/control
cleanup 全部成功后才生成可进入 Registry 的收据。CaseRunRequest/Result 信封升级为 v3；
旧 v1/v2 只允许作为已完成 Manifest 证据迁移，旧裸输入引用不会恢复成权威能力。

0.3.21 将 Artifact Registry 改为 publication receipt 权威边界：普通 `DataRef`
只能作为诊断引用，成功 Case 只有在 durable ledger 验证通过后才能向下游注入产物。
Snapshot authority 增加 write-once/CAS、revision、content digest 与 evidence pin；
Evaluation disposition 回执升级为 v2 并同时绑定 Event/终态 Snapshot 的不可变内容身份。
父子预算委托新增独立 SQLite settlement journal，可在预算后端暂时不可用后幂等补结算。

0.3.20 新增 `EvaluationDispositionVerificationReceipt`：Journal 的 terminal decision
必须绑定 Event Snapshot、目标 Snapshot 和 disposition digest，旧版无回执 terminal 会由
恢复器重新验证。Case runtime 新增 Artifact publication transaction，使一组最终产物只在
统一 commit 后进入正式 registry；`attach_failure_evidence()` 则把评估、回滚和清理证据
纳入 `CaseFailure.details` 的稳定传输信封。heartbeat close 现在会拒绝仍存活的工作线程。

0.3.19 让并行 Stage 正式拥有中间 cancellation control：执行期间续租，全部分支 join 后
在 `finally` 中停止 heartbeat 并 retire，清理异常保留为结构化 evidence。committed
`EvaluationDispositionEnvelope` 现在强制携带 Event 与 authority Snapshot key；内存、
SQLite、Redis Evidence Journal 对 `statuses=()` 统一返回空集合。

0.3.18 将 Redis Context、Redis Snapshot 与文件 Snapshot extras 统一到版本化值信封。
安全模式不再执行旧裸 pickle；signed pickle 的 HMAC 在反序列化前验证；文件 key 必须
解析在 `base_dir` 内。旧 pickle 只允许在隔离迁移模式下显式读取。

0.3.17 新增共享 `EvaluationEvidenceJournal`，以
`preparing -> pending -> deciding -> terminal` 状态机索引 Event、Disposition 与
Authority Snapshot。内存、SQLite 和 Redis 后端都提供原子幂等转移；恢复时可以补结算
已经落盘的处置，或把没有确定裁决的 Event 明确归档为 `abandoned`，而不是猜测重放。
同版补齐 durable cancellation authority 的 active TTL、heartbeat 与 retire 协议：
Project 拥有根控制回收，父 Case 拥有派生控制回收，崩溃遗留由 Redis/SQLite TTL
兜底。L0 可通过 `control_active_ttl_seconds`、`control_heartbeat_seconds` 与
`control_retention_seconds` 调整保活和完成后保留窗口。

后续安全收口将 Redis Context 与 Snapshot 统一到版本化值信封：Context 默认不再读取
旧裸 pickle，Snapshot 的 signed pickle 改为 JSON 外信封并在反序列化前验签；文件
Snapshot extras 同样不再默认读取 `.extras.pkl`，且 snapshot key 不能逃逸 `base_dir`。
旧 pickle 只能在断开不可信网络、确认数据来源后的独立迁移进程中显式开启，不能作为
正常 Case 的兼容回退。

0.3.16 将运行级 Plugin 生命周期也纳入精确参与者 receipt，错误/结束/最终发布会独立
扇出且观察器失败不再替换主异常。Evaluation Gateway 会在每个 Provider/method 首次
有状态 slot 更新时自动认证 copy-on-write 前驱仍可物化，并公开认证审计；新增
`evaluation_disposition/v1` 作为 Event、裁决与权威 Snapshot 的共享证据信封。定向
`StateReleaseRequest` 的结果还必须完整覆盖每个请求 ID（released 或 not-found）；
部分回执会被 Gateway 拒绝，调用方必须保留清理债务。

0.3.15 让严格 Context replay 从基础 Context 开始递归脱离；Plugin start 返回精确
lifecycle receipt，并由匹配 end 独立关闭全部已启动参与者。Redis Snapshot 写入失败
不再返回未落盘 handle；第三方状态 Provider 可通过
`verify_copy_on_write_predecessors()` 认证 optimizer slot 前驱仍可恢复。

0.3.14 将 Context Event 固化为入队即脱离、严格校验的不可变信封；
`StateReleaseRequest.state_ids` 支持按尝试产生的 Provider 状态精确回滚，且 applied
optimizer slot transition 必须 copy-on-write 并返回新 state ID。已完成的历史
manifest v1 结果会迁移为有界 v2 lineage 证据；仍在执行链上的 live v1 request/result
不会被猜测迁移，混合版本 worker 必须先统一升级。

0.3.13 将在线 cancellation lineage 收敛为固定尺寸的 current/root/parent/digest
引用，由 SQLite/Redis authority 递归解析祖先取消；process/external 后端不再接受
`memory` authority。Case 运行传输 schema 因此升级为 v2。

0.3.12 增加共享 attempt/generation 生命周期槽、`evaluation_event/v1`、统一行选择器与 disposition 组合协议，并让 executor view 复用持久 L0 工作池。

0.3.11 将 Case scheduler shutdown 纳入正式结果冻结事务：`runtime_audit` 只在清理后生成，`finished_at` 覆盖完整清理阶段，清理失败形成结构化 `cleanup` evidence 且不会覆盖已有主失败。0.3.10 将并行 Case stage 纳入 Case grant 限制的 L0 scheduler，线程 stage 在全部已启动子调用终结前不得返回，协作取消超时只形成 `cancellation_overdue_calls` 审计，不再伪造永久 `running` 结果；同时增加 attempt/committed Plugin 生命周期与 `CandidateBatch.subset()`。0.3.9 增加共享 `prepare_restore` 阶段和中间 cancellation lineage；0.3.8 增加递归不可变 Case/L0 wire contract、父 grant 并发分账、逻辑设备 token 到物理设备的权威解析、严格 StateRelease scope、TrainerResult 大字段 Artifact 引用，以及结构化嵌套 Case 失败传播；0.3.6 固化 canonical Case 的无兼容层规则，Doctor 会拒绝私有 run/fit wrapper 与兼容装配源码；0.3.5 固化 Feedback 的递归不可变边界，删除共享层默认标量化，给 state transition 增加 operand/slot state-kind 契约，并要求 build-check 关闭已构建 Case；0.3.4 固化 UnknownState/CandidateBatch 的不可变语义身份；0.3.3 增加 CandidateBatch 语义/数值双视图、StateRef trajectory 与正式 StateRelease；0.3.2 增加 Problem Evaluation Provider 的正式状态协议：`StateRef` 只表示
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
