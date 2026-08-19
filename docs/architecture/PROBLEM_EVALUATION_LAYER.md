# Problem Evaluation Layer：问题评估公共层

## 1. 为什么这一层属于 BlackBase

统一框架中的一次计算可以拆成三个正交问题：

1. **求解策略**：下一步提出什么候选、怎样根据反馈更新，由 nsgablack Adapter 负责。
2. **问题评估**：候选代表什么、怎样计算 loss/objective/constraint/gradient，由 Problem 与 Evaluation Provider 负责。
3. **资源授权**：可以使用多少线程、GPU、显存和执行槽位，只由 Project L0 发放 `ResourceContext`。

BlackBase 只提供三者之间的公共协议和绑定机制，不实现 Adam、NSGA-II、交叉熵、Torch 或 CUDA。这样既允许 ML 使用 nsgablack 的优化 Adapter，也不会把 ML 语义或设备细节塞入 Adapter。

## 2. 核心对象

### `EvaluationRequest`

Problem 构造的 provider-neutral 请求。它包含：

- `problem_id` 与运行模式；
- 一批候选状态；
- required / preferred / optional 语义能力；
- 小型、可序列化的 payload 与 metadata。

候选可以是：

- `UnknownState`：可直接传输的数值状态；
- `StateRef`：provider 持有的活状态；
- `DataRef`：已经正式发布的持久对象；
- 其他可由 BlackBase 共享 codec 编码的轻量结构。

不可序列化的模型、tensor 或设备指针不能伪装成引用。

### `EvaluationProviderSpec`

Provider 的静态声明，包括：

- 可服务的 `problem_ids`；空集合只用于真正通用的 Provider；
- 提供的语义能力，例如 `loss.forward`、`autograd.backward`；
- 硬资源下限 `ResourceRequirement`；
- 支持和偏好的设备；
- compute backend，例如 `torch`、`jax`、`numpy`；
- 支持的 evaluate/train/validate/predict/sample 模式。

Registry 会先匹配 `problem_id`，再匹配 mode、capability、资源和设备，避免两个 capability 相同但绑定了不同 Problem 语义的 Provider 被错误选择。`resource_requirement` 是硬条件；`preferred_devices` 只是偏好。Provider 不能因为偏好 GPU 就自行申请或伪造 GPU。

### `EvaluationBinding`

Registry 在现有 `ResourceContext` 内选择 Provider 后产生的审计证据。它记录：

- 选择了哪个 Provider；
- 实际设备与 compute backend；
- 已匹配和缺失的偏好能力；
- 是否发生 CPU fallback；
- 使用的是哪个 L0 grant namespace；
- `allocation_performed=False`，明确绑定层没有分配新资源。

### `EvaluationResult`

统一返回一批 `Feedback`，并可附带：

- provider-owned successor `StateRef`；
- 已发布的 `DataRef` artifacts；
- 完整 `EvaluationBinding`；
- 有界、可序列化的结果 metadata。

结果数量必须与请求候选数量一致。Provider 内部异常只执行一次，不使用“捕获 `TypeError` 后猜签名并重试”的兼容行为。

## 3. `StateRef` 与 `DataRef` 的边界

`StateRef` 表示活状态，例如 Torch Provider 持有的参数与 optimizer state。它只有 provider ID、state ID、scope、device、version 与 transport scope，不包含真实 Python 对象。

`DataRef` 表示已经通过 artifact provider 发布的持久对象，例如模型文件、数据集或 Pareto front。

因此：

- 同进程零拷贝训练可以传递 `StateRef(transport_scope="process")`；
- Registry 只会把 `StateRef` 绑定回它声明的 owner Provider；跨 Provider 交接必须显式转换或发布；
- 跨主机运行只有 provider 明确实现 host/cluster 解析能力时才使用对应 scope；
- 需要进入 manifest、长期保存或交给另一个系统的模型必须发布成 `DataRef`；
- 协议不会把一个无法解析的 tensor 或 model object 自动改写成假 URI。

### Provider-owned 状态怎样被通用 Adapter 更新

对于大型神经网络，先把参数和梯度搬回 NumPy 再执行 Adam 会破坏 GPU 零拷贝。共享层因此提供独立的状态迁移协议：

- `Feedback.gradient_ref` 可以返回 Provider 持有的梯度 `StateRef`；小型数值问题仍可直接返回 `Feedback.gradients`。
- Adapter 构造 `StateTransitionRequest`，明确选择 `method_id`、step、超参数、gradient/direction operand 和已有 optimizer slot refs。
- Provider 必须在 `EvaluationProviderSpec.transition_methods` 中声明它实现的设备计算内核，例如 `gradient.sgd`、`gradient.adam`。正式 Provider 应使用 `StateTransitionMethodSpec` 明确 required/optional operands、parameters、input slots 和 result slots；字符串只是无 shape 约束的简写。
- `EvaluationGateway.transition()` 只把请求绑定回原 `StateRef` 的 owner Provider，并继续服从同一个 L0 `ResourceContext`。
- Provider 返回 `StateTransitionResult`，包含新参数引用和新的 moment/slot refs。

这里的职责不是“Provider 决定使用 Adam”。Adapter 决定算法、参数和何时执行；Provider 只是像 BLAS/CUDA kernel 一样执行已经选定的方法。

Provider 必须用 `state_id + version` 做原子 compare-and-swap；旧引用应抛出共享 `StateVersionConflict`，不能覆盖较新的状态。每次原地提交都必须把参数和复用 slot 的 `StateRef.version` 精确增加 1。`skipped` 必须原样返回旧状态；scope、device 和 transport scope 不能在一次普通更新中隐式改变。设备迁移必须走未来独立的显式 transfer 协议，不能伪装成优化 step。

## 4. 神经网络路径

推荐装配关系如下：

```text
NeuralLearningProblem
  定义：交叉熵、约束、验证指标、输出语义
        │
        ▼
EvaluationRequest
  requires: loss.forward
  preferred: autograd.backward
        │
        ▼
BlackBase EvaluationProviderRegistry
  只在 Project L0 ResourceContext 内绑定
        │
        ├─ GPU grant → TorchEvaluationProvider / cuda:0
        └─ CPU grant → 同一 Provider 的 CPU 路径或另一个等价 Provider
        │
        ▼
Feedback(loss, objectives, gradients, metrics)
        │
        ▼
nsgablack Adapter
  选择 Adam / NSGA / DE / 混合策略，不直接依赖 Torch 或 CUDA
        │
        ├─ 小状态：直接计算下一组 UnknownState
        └─ Provider 状态：提交 StateTransitionRequest(method_id="gradient.adam")
                             │
                             ▼
                    Torch Provider device kernel
                    返回新 StateRef + optimizer slot refs
```

这里 GPU 属于评估执行资源，不属于优化 Adapter。Autograd 是 Provider 能力，不是 L0 资源；L0 只授权 GPU、线程、内存和 worker capability。

## 5. 后续 mlblack 重构边界

BlackBase 基础设施完成后，mlblack 应优先迁移为以下角色：

- ML Problem：loss、metric、数据与输出语义；
- Evaluation Provider：Torch/JAX/NumPy、autograd、闭式拟合、第三方 estimator；
- DataSchedule：split、batch、epoch 数据策略；
- Codec/Head/Artifact：模型表示、预测输出、持久化与统计验证。

通用梯度更新、随机搜索、进化搜索与组合策略应由 nsgablack Adapter 提供。用户仍可通过 ML 词汇装配：

```python
TrainingConfig(optimizer="adam", epochs=20, batch_size=64)
```

Catalog 再把稳定方法标识解析为 nsgablack Adapter + mlblack Evaluation Provider + DataSchedule，而不是把仓库边界暴露给用户。

## 6. 当前刻意没有放入 BlackBase 的内容

- 具体 loss、metric、神经网络结构；
- Torch/JAX/NumPy 实现；
- epoch、batch、数据划分策略；
- Adam、NSGA、DE 等算法；
- 活状态实际存储和释放策略。

这些分别属于 mlblack、nsgablack 或具体 Provider。BlackBase 只固定它们可以可靠组合的协议。
