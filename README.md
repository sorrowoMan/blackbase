# BlackBase

`blackbase` 是 `nsgablack` 与 `mlblack` 共享的运行底座。它不实现优化算法、模型训练方法或领域目标，只提供两种语义层共同依赖的协议与工程闭环。

当前版本：`0.3.24`。

0.3.24 收紧成功证据的发布边界：Case 的严格 finalization prepare、cleanup、结果序列化
和 Artifact 物理内容复核全部通过后，Ledger 才原子 seal；`on_solver_finalized` 现在是
seal 后的不可否决观察钩子。Artifact 名称预留携带 lease/fencing 与 provisional ref，
可回收崩溃遗留对象；子 Case 预算采用先写 intent、再 reserve 的结算日志，日志只保存
最小 authority reference。Evaluation Evidence 的上层可据此只接纳真正权威的产物。

0.3.23 新增 Project DAG Stage：显式控制依赖与同 Stage 权威 Artifact 输入统一进入
`DagStagePlan`，运行前完成缺失节点/自依赖/环检测，运行时由 L0 在资源授权范围内执行
就绪波次，并为失败后代、Manifest 恢复和每个 Case 请求保留结构化依赖证据。

本版把“已提交”从状态值提升为可验证证据：Evidence Journal 只有收到绑定 Event
Snapshot、目标 Snapshot 与 disposition 摘要的验证回执后，才能结算为 decision terminal；
旧版未验证 terminal 会重新进入恢复对账。Case runtime 同时增加 Artifact 发布事务与
正式异常 evidence carrier，批量最终产物可以先暂存、再原子公开，跨 Case 失败信封也
不再丢失评估、回滚和清理证据。Cancellation heartbeat 只有在线程确认终止后才可报告
关闭成功。

本版补齐并行 Stage 中间 cancellation control 的所有权闭环：Stage runner 在执行期间
持续 heartbeat，在所有分支 join 后无论成功或失败都会关闭 heartbeat 并 retire control，
清理失败进入结构化 evidence。`evaluation_disposition/v1` 的 committed 信封现在必须同时
携带 Event 与 authority Snapshot key；三个 Evidence Journal 后端对空状态过滤也统一返回
空集合。

本版把运行级 Plugin 初始化、完成、最终发布与错误通知也纳入 process-local receipt，
严格模式会独立通知所有实际启动参与者而不让观察器异常覆盖主错误。状态 Provider 在
首次含前驱 slot 的 applied transition 上自动执行 copy-on-write 认证，认证通过后按
provider/method 缓存审计；第三方 Provider 不再能只声明 COW 而跳过前驱可物化证明。
共享层新增 `evaluation_disposition/v1`，用于把 Evaluation Event、裁决结果与权威
Snapshot 连成稳定的传输信封；定向 State release 必须回报每个请求 ID 已释放或不存在，
不能用部分成功回执清空调用方债务。Project L0 lease 的 TTL 必须至少为 1 秒，心跳间隔不得大于 TTL
的三分之一；运行时与 Project Doctor 都会 fail-closed 拒绝无法覆盖 Python
调度抖动的不安全配置。子秒 lease 仍可用于底层 authority 的过期与围栏测试，
但不能作为完整 Case 执行的授权时钟。

`EvaluationEvidenceJournal` 进一步把这条证据边变成可枚举状态机：Event 快照写入前
先登记 `preparing`，落盘后转为 `pending`，处置发布前进入 `deciding`，最后才结算为
`committed/rejected/failed`。内存、SQLite 与 Redis 实现都提供原子、幂等转移；恢复器
可以定位崩溃窗口，而不需要扫描和猜测任意 Snapshot，也不会擅自重放有副作用的评估。

durable cancellation authority 也采用明确的所有权生命周期。Project 回收自己发放的
根控制记录，父 Case 回收派生的子控制记录；执行期间 Redis/SQLite 控制记录按
`control_active_ttl_seconds` 与 `control_heartbeat_seconds` 续租。正常完成时按
`control_retention_seconds` 立即删除或短期保留，进程崩溃后则由 TTL 自动回收，
不会再让 lineage 控制键无限增长。

## 职责

- Project / Stage / DAG / Case / Scaffold
- L0 资源池、lease、fanout、设备 token 与 `ResourceContext`
- ContextStore / SnapshotStore / Artifact 引用
- 标准 Case builder 绑定与跨 Case 调用
- lineage、deadline、cancellation、budget handle
- 进程、external worker 与统一 `CaseRunResult` 信封
- `UnknownState`、`CandidateBatch`、Feedback、`SolverResult`、`TrainerResult`
- Problem / Evaluation Provider 公共协议
- pipeline slot kernel、Plugin 生命周期协议与 Catalog 原语

## 不属于 BlackBase

- NSGA-II、Adam、MCMC 等求解策略
- loss、objective、constraint 和模型输出语义
- Torch、CUDA、sklearn estimator 的具体实现
- 业务数据处理、模型结构和部署逻辑

这些能力分别由 nsgablack、mlblack 或外部 Provider 实现。

## 标准 Case

Solver 与 Trainer 使用相同目录和运行协议：

```text
case/
  .case
  README.md
  build_solver.py
  run_solver.py
  config.py
  problem/
  pipeline/
  adapter/
  bias/
  plugins/
  evaluation/
  runtime/
  solver/
```

`build_solver.py` 是唯一规范装配入口，接受 `resource_context` 与 `component_overrides`。`kind=solver|trainer` 只表达语义分类；Trainer 命名入口只能是薄别名。

## 跨 Case 调用

父 Case 通过公共请求调用完整子 Case。底座统一派生：

- root / parent / child lineage
- 绝对 deadline 与 cancellation 链
- 子资源 grant 和并发分账
- 子预算委托与实际结算
- Artifact / DataRef 注入
- 结构化失败和可传输结果

协作式取消适用于可检查 checkpoint 的代码；不可中断的原生调用必须放进可监督的隔离进程，才能升级为 terminate / kill。

并行 Stage 的 `fail_fast` 会建立独立的 stage cancellation lineage：首个失败出现后停止未开始任务并通知运行中的兄弟 Case。线程模式遵守结构化并发，所有已启动调用终结前父 Case 不会返回；超过协作取消 grace 的调用记录为 `cancellation_overdue_calls`。需要有界强制终止的原生调用必须使用隔离执行，不能用无句柄的 `running` 结果脱离父级资源生命周期。

DAG Stage 使用 `policy="dag"`。`depends_on` 可声明普通控制依赖；同一 Stage 内形如
`producer.artifact` 或 `stage.producer.artifact` 的 `input_artifacts` 会自动形成 Artifact
依赖边。运行前统一拒绝未知节点、自依赖与环；运行时只提交依赖已成功的就绪 Case，
实际 fanout 仍受 Project L0 的 worker、线程和设备授权限制。`failure_policy="continue"`
只跳过失败节点的后代并继续独立分支，默认 `fail_fast` 则终止剩余 DAG。DAG 描述运行前
已知的 Case 关系；Case 在运行中动态调用子 Case 仍使用 `CaseRuntimeContext.invoke()`，
两者不是同一层协议。

共享 Plugin 生命周期在普通初始化前提供 `prepare_restore`。该阶段只负责加载、验证并排队 restore envelope；语义框架在 setup 后原子应用恢复状态，之后才触发 `on_solver_init`，从而保证普通插件不会观察到“setup 已完成但 checkpoint 尚未恢复”的中间态。

## 状态边界

- Context：轻量、可更新、可审计。
- Snapshot：运行内大型状态。
- Artifact / DataRef：跨 Case、跨进程或长期保存的正式对象。
- StateRef：Provider 进程内活状态，不是持久引用。

Redis Context 与 Snapshot 默认使用 `blackbase.redis.value/v1` 安全信封。安全模式只恢复
正式协议类型，不执行 Redis 中的 pickle，也不把未知对象静默转换成 `repr`。签名
pickle 使用 JSON 外信封并在 `pickle.loads` 前校验 HMAC；裸 pickle 与旧外层 pickle
只能在显式隔离迁移模式下读取。文件 Snapshot 的 extras 使用同一安全信封，且所有
snapshot key 必须解析在配置的 `base_dir` 内。

协议不会为不可序列化对象伪造引用。模型、tensor 或 optimizer slot 必须由对应 Provider 正式发布、释放或持久化。

Context Event 是不可变审计信封：记录后修改原始 list/dict 不会改变待提交事件。异步尝试只以严格模式重放完整事件批次，损坏批次不会部分写入 Context。Plugin start 返回的 receipt 是匹配 end 的唯一参与者集合。Provider 状态回滚必须优先使用 `StateReleaseRequest.state_ids` 定向释放本次尝试产生的槽位；applied optimizer slot transition 必须 copy-on-write，旧 slot 在调用方提交前保持可恢复，并通过 conformance 检查完成第三方 Provider 认证。

## 文档

- [Project 运行时](PROJECT_RUNTIME_CN.md)
- [强制终止与 Artifact 发布](EXECUTION_ARTIFACT_CLOSURE_CN.md)
- [Redis transport 与恢复](REDIS_TASK_TRANSPORT_CN.md)
- [Problem Evaluation Layer](docs/architecture/PROBLEM_EVALUATION_LAYER.md)
- [0.3 迁移说明](MIGRATION.md)
- [三仓发布协议](STACK_RELEASE.md)

## 验证

```powershell
python -m pip install -e .[dev]
python -m pytest -q
```
