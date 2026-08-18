# Redis TaskTransport 与 Project 恢复

`RedisTaskTransport` 是 blackbase 的跨进程、跨机器任务传输实现。它和
`SQLiteTaskTransport` 遵循同一个 `TaskTransport` 契约，负责原子认领、任务租约、
worker 心跳、失败重试、过期租约回收、取消、结果等待和状态审计。Solver、Trainer
以及业务 Case 不应直接依赖 Redis 命令。

## Project 外部 Stage

```python
{
    "name": "external_fit",
    "policy": "external",
    "cases": ["fit_case"],
    "external": {
        "backend": "redis",
        "redis_url": "redis://127.0.0.1:6379/0",
        "namespace": "my_project:external_tasks",
        "queue_timeout_seconds": 30,
        "poll_interval_seconds": 0.05,
        "max_retries": 1,
    },
}
```

worker 使用同一 URL 与 namespace：

```powershell
python -m blackbase.project.external_worker `
  --project-root . `
  --backend redis `
  --redis-url redis://127.0.0.1:6379/0 `
  --namespace my_project:external_tasks `
  --worker-id worker-1 `
  --threads 4
```

`redis_url` 可能包含凭据，因此 Project 的 check 输出和 manifest 只记录 backend 与
namespace，不回显 URL。部署时应通过受控配置注入 URL，不要把生产凭据提交到仓库。

## Redis Project L0 authority

`RedisTaskTransport` 只负责任务 broker；跨机器资源预算与结果 fencing 由独立的
`RedisLeaseStore` 负责。多机 Project 应同时配置：

```python
L0 = {
    "namespace": "my_project",
    "lease_backend": "redis",
    "lease_redis_url_env": "BLACKBASE_REDIS_URL",
    "lease_ttl_seconds": 30,
    "lease_heartbeat_seconds": 10,
}
```

task transport namespace 与 lease namespace 是两种不同职责：前者定位队列，后者定义
共享资源预算和 fencing 序列。它们可以使用同一个 Redis 服务，但不能把 task lease
误当作 Project resource lease。worker 必须同时续租两者，并在提交结果前再次验证
Project fencing token。

## 真实 Redis 集成测试

默认测试套件不会依赖外部 Redis；设置专用测试 URL 后会启用真实服务测试：

```powershell
$env:BLACKBASE_TEST_REDIS_URL = "redis://127.0.0.1:6379/15"
pytest tests/test_redis_live_integration.py -q
```

测试使用随机 namespace，覆盖并发 admission、TTL/fencing、完整 Project external
worker 往返、task lease 崩溃恢复和 stale owner 拒绝，并只清理自身 namespace，
不会执行 `FLUSHDB`。

## 崩溃恢复边界

Project 为每个外部 Case 使用确定性 task id：

```text
project:<run-id>:<stage-name>:<case-name>
```

任务提交成功后，Project 会立即把 task id 和 broker 状态写入 manifest。即使进程恰好
在“broker 已提交、manifest 尚未写入”的窗口崩溃，恢复运行仍可根据旧 run id 推导 task
id 并与 transport 对账：

- `succeeded`：直接恢复 worker 结果与 artifact，不重复执行；
- `leased`：复用原任务的 `ResourceContext` 与 L0 fencing lease，等待原任务结束，避免重复授权和并发重复执行；
- `queued`：取消旧排队任务，再用新一轮 Project L0 grant 提交；
- `failed/cancelled/missing`：按新运行重新提交。

恢复得到的 Case 状态为 `resumed`。如果结果来自旧任务，审计中的 `ResourceContext`
使用 worker 实际执行时的上下文，而不是伪装成恢复进程新申请的上下文。

## nsgablack 兼容面

`blackbase.resources.RedisL0RuntimeBackend` 是统一 Redis
`TaskTransport` 的兼容 facade。旧的 `submit/claim/complete/get_result` 调用外形仍然保留，
但 claim token 只能由同一个 worker 进程持有和完成。新的 worker 代码应优先使用
`task_transport`、`claim_task()` 和 `complete_claim()`，以便显式处理 lease 与心跳。
