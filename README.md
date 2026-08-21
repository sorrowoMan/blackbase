# BlackBase

`blackbase` 是 `nsgablack` 与 `mlblack` 共享的运行底座。它不实现优化算法、模型训练方法或领域目标，只提供两种语义层共同依赖的协议与工程闭环。

当前版本：`0.3.9`。

## 职责

- Project / Stage / Case / Scaffold
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

并行 Stage 的 `fail_fast` 会建立独立的 stage cancellation lineage：首个失败出现后停止未开始任务、通知运行中的兄弟 Case，并在有界 grace 后把仍未退出的调用记录为 `still_running_calls`，不会无限等待后才声称 fail-fast。

共享 Plugin 生命周期在普通初始化前提供 `prepare_restore`。该阶段只负责加载、验证并排队 restore envelope；语义框架在 setup 后原子应用恢复状态，之后才触发 `on_solver_init`，从而保证普通插件不会观察到“setup 已完成但 checkpoint 尚未恢复”的中间态。

## 状态边界

- Context：轻量、可更新、可审计。
- Snapshot：运行内大型状态。
- Artifact / DataRef：跨 Case、跨进程或长期保存的正式对象。
- StateRef：Provider 进程内活状态，不是持久引用。

协议不会为不可序列化对象伪造引用。模型、tensor 或 optimizer slot 必须由对应 Provider 正式发布、释放或持久化。

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
