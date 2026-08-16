# Case 强制终止与 Artifact 发布闭环

## 1. 两层取消语义

`ExecutionControl` 默认采用 `cooperative`：Case 在 generation、step、评估和快照边界调用
`case_runtime.checkpoint()`，能够保留正常清理、插件收尾和一致性提交。

对不可中断的原生调用，可显式声明隔离升级：

```python
L0 = {
    "termination": {
        "mode": "cooperative_then_terminate",
        "grace_seconds": 2.0,
        "kill_grace_seconds": 1.0,
        "poll_interval_seconds": 0.05,
    },
}
```

也可在 Stage 的 `termination` 或 `case_termination[case_name]` 中覆盖。执行顺序为：

1. deadline 或 cancellation 写入共享控制权威；
2. 给 Case 留出 `grace_seconds` 完成协作式退出；
3. 仍未退出时终止隔离进程；
4. `kill_grace_seconds` 后仍存活则强杀。

串行 Project、并行进程、完整嵌套子 Case 和 external worker 都使用同一个监督协议。
硬终止只能作用于隔离边界，因此不会尝试强杀 Python 线程。

## 2. Artifact 发布权威

Project L0 在 `ResourceContext.metadata.artifact_authority` 中注入文件存储权威：

```python
L0 = {
    "artifacts": {
        "path": ".blackbase/artifacts",
        "allow_unsafe_serializers": False,
    },
}
```

Case 通过运行时发布真实对象：

```python
model_ref = self.case_runtime.publish_artifact(
    "best_model",
    model_payload,
    kind="model",
)
```

发布器会先序列化到临时文件，计算 SHA-256，再校验 Project lease fencing，最后原子替换到
Project artifact 根目录。只有上述步骤全部成功才返回 `DataRef`；协议不会根据名称或 URI 字符串
伪造一个无法读取的引用。

安全默认 codec 包括 JSON、文本、bytes、NumPy NPZ 和已有文件。模型专用 codec 属于
`mlblack` ArtifactProvider 或其他正式 provider/plugin。`mlblack` 的默认 Case provider 会在
Trainer 结果封装前发布 best model，并把成功返回的 `DataRef` 交给 Trainer。

任意 Python 对象的 pickle 只在 `allow_unsafe_serializers=True` 时允许；默认关闭，因为加载不可信
pickle 会执行代码。external worker 还会验证 artifact 根目录必须位于其获准的 Project 根目录内。
