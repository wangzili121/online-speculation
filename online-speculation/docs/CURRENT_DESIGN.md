# DSV4-Flash online-speculation 接入说明

> **当前状态：核心代码已经接通，真实 8 卡正确性和性能验收尚未完成。**

## 1. 我们要解决什么问题

普通大模型一次只生成一枚 token。`online-speculation` 增加一个起草分支：

```text
起草 head 一次猜 31 枚 token
            ↓
主模型一次验证整段
            ↓
提交正确前缀，错误位置由主模型校正
```

起草分支不复制完整 DeepSeek-V4，只保存单独训练的 attention 参数；embedding、mHC、
MoE/MLP、LM head 和历史 KV 都复用主模型。

正确实现时，投机解码不会改变主模型的输出分布。候选接受率只决定能否加速。

## 2. 原本为什么想做成普通插件

上游 vLLM 已经支持：

```json
{"method": "custom_class"}
```

它可以加载一个 Python proposer，并负责候选验证和 rejection sampling。因此最初看起来，
只需要在 `online-speculation` 中实现一个 custom class。

但原始 `custom_class` 只是轻量 token 回调：

```text
输入 token 历史 -> 返回候选 token
```

它适合 n-gram、检索和规则算法。这里的 proposer 却是一个真正的 NPU 神经网络，还需要
模型加载、8 卡 TP、attention metadata、KV cache 和显存规划。

## 3. vLLM-Ascend 缺少什么，我们增加了什么

### 问题一：custom proposer 会在 Ascend 初始化时丢失

上游 vLLM 已经创建 custom proposer，但 vLLM-Ascend 初始化自己的 drafter 时会重新赋值，
只创建 EAGLE、DFlash、DSpark 等内置 proposer。

**解决：**保留上游已经创建的 custom proposer，并在 NPU decode 路径调用它。

这一步已经由此前投机执行接入使用的
[token-only custom class 分支](https://github.com/wangzili121/vllm-ascend/tree/custom-class-proposer-npu)
完成。当前分支的 `a09abd67` 是同一改动在更新 vLLM-Ascend main 上的重放，不属于本次
model-aware 扩展的新功能。它只解决普通 token-only custom class，还不能运行神经网络 head。

### 问题二：插件拿不到 NPU 运行环境

普通插件不知道当前 NPU、TP 配置、target model、attention backend 和 runner 缓冲区。

**解决：**增加可选的 model-aware 接口：

```python
class MyProposer:
    npu_model_runner_api_version = 1

    def bind_runtime(self, runtime):
        ...
```

只有声明该标记的插件才会获得当前 `NPUModelRunner`、device 和配置。普通 custom class
行为不变。

### 问题三：框架不会为 custom class 加载模型

原框架把 custom class 当作普通 Python 函数，因此会跳过：

- draft model 加载；
- attention backend 初始化；
- dummy run 和显存 profiling；
- model proposer 的 padding、输入准备和拒绝处理。

**解决：**让 model-aware custom proposer 进入已有模型 proposer 生命周期：

```text
bind_runtime
    ↓
load_model
    ↓
initialize_attn_backend
    ↓
dummy_run / memory profiling
    ↓
prepare_inputs / _propose
```

插件由此复用成熟的 NPU runner，而不是再复制一套。

### 问题四：没有通用方式复用主模型骨干

Orthrus head 只应保存独立 attention，不能再加载一份完整 DSV4。

**解决：**增加 `bind_shared_backbone(target_model)` hook。target 加载后，draft 绑定并复用
target 的 embedding、mHC、norm、MoE/MLP 和 LM head。

### 问题五：cache planner 不认识插件 attention

原 cache planner 只识别框架内置 `Attention` 类型。插件即使想共享 target KV，也可能被
分配第二份历史 cache。

**解决：**支持通用属性：

```python
kv_sharing_target_layer_name = "目标 cache 层名"
```

cache planner 在服务启动时记录别名，不再为 draft 分配重复历史 KV。

这一点必须由框架完成，因为显存和 cache 在第一次 `propose()` 之前就已经规划好了。

## 4. 哪些内容仍然属于算法插件

vLLM-Ascend 只提供通用生命周期，不包含 Orthrus 或 DSV4 专属逻辑。

`online-speculation` 插件负责：

- `1 个 anchor + 31 个 mask` 的 K=32 起草；
- 只加载 diffusion attention head；
- 读取 DSV4 的 SWA、C4、C128 三类历史 cache；
- C4 diffusion-side indexer；
- 当前块双向 attention 和历史只读；
- 使用 LSE 精确合并历史、当前块和 attention sink；
- head v2 契约、训练、量化检查、部署和 benchmark。

不能直接把它伪装成 DSpark。两者可以复用 NPU 缓冲区、TP 和调度基础设施，但生成语义和
cache 所有权不同。

## 5. 最终代码边界

| 仓库 | 负责什么 | 是否修改 |
| --- | --- | --- |
| vLLM | 调度、验证、rejection sampling、OpenAI API | 不修改 |
| vLLM-Ascend | model-aware custom proposer 通用生命周期 | 小型通用改动 |
| `online-speculation` | Orthrus/online-speculation 和 DSV4 专属实现 | 独立插件 |

因此不是给 vLLM-Ascend 增加一个专属 `orthrus` method，而是补一个其他神经网络 proposer
也能复用的“模型型插件插座”。

当前通用 feature 分支基于较新的 vLLM-Ascend main，不能直接覆盖指定的 v0.26 镜像。
内网 v0.26 镜像需要使用版本锁定 runtime overlay，或把相同接口定向 backport 到
v0.26 release 分支。

## 6. 当前做到什么程度

**已经完成：**

- model-aware custom proposer 接口；
- K=32 起草和共享 target backbone；
- SWA/C4/C128 历史读取、C4 indexer 和 LSE 合并；
- cache 只读检查、head 契约和 preflight；
- BF16 head 训练、恢复、异步续训入口；
- 服务、回滚和 benchmark 脚本；
- 单元级、数学和小模型反向验证。

**尚未完成：**

- 真实 DSV4-Flash 的 8 卡 API 闭环；
- 外部训练后 head 及其独立 ModelSlim W8A8 校准；
- 并发采样、连续拒绝、分页边界和长上下文验证；
- trained head 的真实接受率和 paired goodput；
- graph、DSA CP 和 FlashComm1 路径。

## 7. 下一步

1. 先启动 8 卡 target-only，确认 DSV4 OpenAI API 正常。
2. 使用 initialized head 启动 Orthrus eager，完成请求闭环和 cache 检查。
3. 验证 greedy token 一致、并发拒绝和分页边界。
4. 接收 trained BF16 head，独立校准为 W8A8。
5. 在真实流量上比较 goodput；提升至少 5% 才默认启用。

当前 8 张实验卡已经释放，不会自动恢复任务。

## 8. 进一步阅读

- [完整算法](ALGORITHM.md)
- [插件边界](CUSTOM_PROPOSER_INTEGRATION.md)
- [Head v2 契约](DSV4_HEAD_CONTRACT.md)
- [部署与验收](DSV4_DEPLOYMENT.md)
- [训练与续训](DSV4_TRAINING.md)
