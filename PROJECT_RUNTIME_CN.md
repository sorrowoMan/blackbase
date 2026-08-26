# Project 运行时契约

blackbase 是 `nsgablack + mlblack` 统一框架栈的 Project / Case / Scaffold / L0 substrate。它负责跨 Case 编排、资源授权、执行隔离、artifact 传递和恢复记录；优化搜索语义仍属于 nsgablack，ML 训练语义仍属于 mlblack。

## Stage 执行

`project_config.py` 中的 Stage 支持四种策略：

- `policy="serial"`：按声明顺序在主进程执行。
- `policy="parallel"`（兼容名 `run_all_in_parallel`）：在相互隔离的子进程中执行 Case。
- `policy="external"`（兼容名 `external_workers`）：通过耐久 transport 交给独立 worker 进程执行。
- `policy="dag"`（兼容名 `dependency_graph`）：根据 Case 依赖自动选择就绪节点并行执行。

并行 Stage 的 Case 必须使用标准 `build_solver.py` 装配入口和 `mode="build"`。CLI fanout 没有可靠的进程内对象与 artifact 注入契约，因此运行时会明确拒绝并行 `mode="cli"`，不会静默退化成串行。

普通并行 Stage 内的 Case 被视为互相独立。它们可以消费先前 Stage 注册的 `DataRef`，
但不能依赖同一 Stage 中尚未完成的 Case。同一 Stage 需要依赖关系时使用 DAG。

```python
STAGES = [
    {
        "name": "search_and_fit",
        "policy": "parallel",
        "failure_policy": "fail_fast",
        "max_workers": 2,
        "cases": ["search_case", "fit_case"],
        "resource_requests": {
            "search_case": {"workers": 1, "threads": 2},
            "fit_case": {"workers": 1, "threads": 2},
        },
    },
]
```

Project L0 在父进程中先发放 lease，再提交子进程。所有活动 lease 的 workers、threads、GPU、memory 和独占 device token 会做聚合校验；资源不足时，调度器形成资源允许的执行波次，而不是超额授权。

## DAG Stage

DAG 只描述运行前已知的 Case 依赖。普通控制依赖写入 `depends_on`；权威 Artifact 输入
会自动推导同 Stage 依赖边：

```python
STAGES = [{
    "name": "workflow",
    "policy": "dag",
    "max_workers": 4,
    "cases": ["prepare", "train", "baseline", "evaluate"],
    "depends_on": {
        "train": ["prepare"],
    },
    "input_artifacts": {
        "evaluate": {
            "model": "train.model",
            "baseline": "baseline.report",
        },
    },
}]
```

由此得到 `prepare → train → evaluate` 与 `baseline → evaluate`。`prepare` 和
`baseline` 首先成为就绪节点并在 L0 授权范围内并行；两个上游成功且 Artifact 发布回执
验证完成后，`evaluate` 才会被提交。

规则如下：

- `depends_on` 只能引用同一 DAG Stage 中的 Case；未知节点、自依赖和环会在运行前失败。
- `producer.artifact` 与 `stage.producer.artifact` 会推导依赖；无生产者信息的短名不会猜测。
- Case 结果保留 DAG schema、显式依赖、Artifact 推导依赖和拓扑位置审计。
- 默认 `failure_policy="fail_fast"`；`continue` 会继续独立分支，并用
  `DependencyFailed` 明确跳过失败节点的全部后代。
- 恢复成功的节点视为已完成，但其 Artifact 必须通过原有 publication receipt 验证。

DAG 与动态嵌套不是替代关系。DAG 负责 Project 中预先可知的 Case 图；外层 Solver 在
`evaluate()` 中按候选动态创建 Trainer、Trainer 再创建 Solver，仍通过
`CaseRuntimeContext.invoke()` 形成运行时调用树。

## 失败语义

`failure_policy="fail_fast"` 会在观察到第一个失败后停止启动新 Case；已经运行的 Case 会完成并归还 lease，尚未启动的 Case 会产生明确的 `skipped` 结果。`failure_policy="continue"` 则继续执行可以运行的 Case。

Case 输出跨进程时必须是 JSON 兼容数据、`DataRef`、Path、支持 `tolist()` 的数组，或实现 `as_dict()` 的对象。不可安全传输的对象会让该 Case 明确失败。

## Artifact 传递

Case 在结果的 `artifact_refs` 或 `artifacts` 字段返回命名引用：

```python
return {
    "artifact_refs": {
        "model": {"uri": "s3://bucket/model.bin", "kind": "model"},
    },
}
```

后续 Stage 通过正式 key 注入：

```python
{
    "name": "evaluate",
    "cases": ["evaluator"],
    "input_artifacts": {
        "evaluator": {"model": "trainer.model"},
    },
}
```

消费方 Case 必须实现 `set_input_artifacts(refs)`。运行时只传引用，不把模型、population、数据集等大对象塞进 context。

## Manifest 与恢复

实际运行默认原子写入：

```text
.blackbase/runs/<run-id>/manifest.json
```

Manifest 包含 Project、group、framework、运行状态、每个 Case 的状态/耗时/错误、artifact registry，以及配置和 Case 源码指纹。它不持久化任意运行对象或完整大输出。

```powershell
python run_project.py --run-id first-attempt
python run_project.py --resume-from first-attempt
```

恢复时，先前成功的 Case 标记为 `resumed` 并跳过执行；它们的 artifact 引用会重新注册，失败或未启动的 Case 会重新运行。只要 `project_config.py`、Case Python 源码、`.case`、group 或 framework 发生变化，指纹校验就会拒绝恢复，避免用旧运行状态驱动新代码。

调试时可用 `--no-record` 关闭记录；`--check` 本身不会创建运行记录，也不能与 `--resume-from` 同时使用。

## 外部 worker

外部执行仍由 Project L0 先发放 lease。Project 将标准 Case payload、资源 requirement、`ResourceContext`、输入 `DataRef` 和重试上限写入 transport；worker 只能 claim 自己的 `WorkerDescriptor` 能满足的任务。

当前提供 SQLite 参考后端，适用于同机多进程或共享文件系统上的 worker。它具备：

- 原子 claim，同一个任务不会同时授权给两个 worker；
- worker/task 心跳和 lease token；
- worker 崩溃后的 lease 过期回收；
- `max_retries` 控制的 at-least-once 重试；
- 幂等 task id、耐久结果和严格 JSON payload；
- 无兼容 worker 时的明确 queue timeout，不会退化成本地执行。

Project 配置：

```python
{
    "name": "external_fit",
    "policy": "external",
    "cases": ["fit_case"],
    "external": {
        "backend": "sqlite",
        "transport_path": ".blackbase/external_tasks.sqlite",
        "queue_timeout_seconds": 30,
        "poll_interval_seconds": 0.05,
        "max_retries": 1,
    },
}
```

先启动 worker，再运行 Project：

```powershell
python -m blackbase.project.external_worker `
  --project-root . `
  --transport .blackbase/external_tasks.sqlite `
  --worker-id worker-1 `
  --threads 4

python run_project.py
```

worker 会拒绝 payload 中不等于 `--project-root` 的路径，防止 transport 中的任务越权执行其他本地项目。SQLite 后端不伪装成网络集群；真正跨机器且没有共享文件系统时，应实现同一个 `TaskTransport` 契约的 Redis、数据库或云队列 provider。

## 耐久 L0 lease 与 fencing

新建 Project 默认使用 SQLite 作为 L0 lease authority：

```python
L0 = {
    "lease_backend": "sqlite",
    "lease_path": ".blackbase/l0_leases.sqlite",
    "lease_ttl_seconds": 30,
    "lease_heartbeat_seconds": 10,
}
```

每次授权都获得 namespace 内单调递增的 `fencing_token`。资源预算校验、过期 lease 回收、token 发放和 lease 写入在一个 SQLite 写事务中完成，因此多个 Project 进程不能同时抢到同一份最后资源。活动 Case 由 Project 或 external worker 续租；只有 `lease_id + fencing_token` 仍为当前授权时才能续租、释放或提交结果。

这条约束解决三类运行时遗留问题：

- Project 崩溃后，旧 lease 到期自动释放资源，不永久占用预算；
- 旧 worker 即使晚到，也不能用已失效 token 覆盖新一轮运行结果；
- 恢复运行遇到仍为 `leased` 的外部任务时，会接管原 task 和原 lease，不重复申请资源或重复执行 Case。

worker 在提交 `TaskResult` 前必须再次校验 Project fence，并在结果 metadata 中写入验证标记和 token。Project 对 durable authority 下缺少该证明、token 不一致或本地 Case 已失去 fence 的结果一律拒收。

为兼容已有 Project，未声明 `lease_backend` 的旧配置暂时保持进程内 `memory` authority；需要崩溃恢复、外部 worker 或多个 Project 进程共享预算时，应显式迁移到上述 SQLite 配置。`project doctor --strict` 会检查 backend、path、TTL 与 heartbeat 的静态配置。

### 跨机器 Redis lease authority

SQLite authority 适合同机进程或共享文件系统。没有共享文件系统的多机 worker 应把 Project L0 authority 切换到 Redis：

```python
L0 = {
    "namespace": "my_project",
    "lease_backend": "redis",
    "lease_redis_url_env": "BLACKBASE_REDIS_URL",
    "lease_ttl_seconds": 30,
    "lease_heartbeat_seconds": 10,
}
```

Project 与 worker 都通过声明的环境变量读取 Redis URL。URL 不会写入 `ResourceContext`、check 输出或运行 manifest；上下文只传 authority backend、namespace、环境变量名、TTL 与 heartbeat。也可以在受控环境中使用 `L0.lease_redis_url`，但不建议把带凭据 URL 提交到仓库。

Redis authority 在 namespace 级分布式锁内完成过期回收、活动租约聚合校验和 fencing token 发放，再用 Redis transaction 一次提交 lease、审计索引与下一个 token。因此多个 Project 进程共享同一 namespace 时，不会同时拿到最后一份资源。

worker 示例：

```powershell
$env:BLACKBASE_REDIS_URL = "redis://redis-host:6379/0"
python -m blackbase.project.external_worker `
  --project-root . `
  --backend redis `
  --redis-url $env:BLACKBASE_REDIS_URL `
  --namespace my_project:external_tasks
```

如果 task transport 与 lease authority 使用不同 Redis，可额外传 `--lease-redis-url`。生产环境应由 secret manager 或进程环境注入连接信息。
