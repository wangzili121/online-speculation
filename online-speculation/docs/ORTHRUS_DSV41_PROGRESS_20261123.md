# Orthrus-DSV4.1 进展简报（2026-11-23）

## 遇到的问题与解决方式

双机整模验证最初在进入投机前的 prefill 就出现 token 分叉。逐层保存并比较张量后确认：权重、KV、Attention 以及四个 TP rank 的局部输出完全一致，第一处差异发生在 BF16 TP 归并。不同执行次序造成的舍入误差经过后续层放大，最终改变 argmax token。

修复方式是让 TP partial 以固定 rank 顺序在 FP32 中累加，最后只转一次 BF16。修复后，单机重复测试与双机 16 卡 target/Orthrus 整模测试均通过，greedy token 一致；Orthrus 确实执行了 draft、verify、accept 和拒绝缓存恢复。

## 与 PyPTO 新分支的关系

最新的 DSV4.1 专用开发分支是：

```text
Adelaide999/pypto-lib
deepseek_v4.1_flash_1122
13f75e54adb82bf6118258f368fd4182f7a91b07
```

该分支也采用 FP32 顺序累加，因此数值处理方向一致。但它同时重写了整图、权重驻留和 Attention/TP 编排，并删除或替换了当前 resident serving 依赖的 `network.py`、`state.py`、`prefill_layer.py` 等接口，不是可直接替换的补丁。

当前继续使用已经通过双机验收的固定 serving 基线，并保留等价的确定性归并修复。等 `1122` 或后续版本提供匹配的 serving 接口后，再迁移 Orthrus 的独立契约、kernel 和测试，而不是现在冒险更换整个底座。

## 已完成

- DSV4.1 W8A8 双机 16 卡 target 与 Orthrus 推理一致性通过。
- K32 双向 draft、共享历史、整块验证、接受/拒绝缓存生命周期通过。
- last-N 计划、真实后缀边界与 Attention/mHC/MoE 精确状态捕获完成。
- 双机全局只选一个在线反馈请求，其他 DP 参加零梯度同步。
- TV/KL、C1A Attention、mHC、固定路由 MoE、AdamW 和发布所需 PyPTO 叶子已实现并独立验证。
- router 的 sigmoid、Top-6 归一化和稀疏回填已经合并为一个 PyPTO VJP；生产 `K32×384` 形状编译通过，小形状实卡数值与 autograd golden 一致。
- EP16 反向路由已固定为 `token/slot/global-expert/owner/local-expert` 顺序：梯度按专家 owner 分发，原顺序回收后再对每个 token 的 6 条路径求和，不会在反向重新执行 Top-K。
- 冻结 W8A8 MoE 权重的输入 VJP 已补齐：设备端按原生分块布局反量化，真实 `2304×5120` 与 `5120×4608` 尺寸编译通过，小形状数值在 NPU 上逐元素一致。
- 原生 MoE 前向在在线 draft 模式下会保存 Top-6 每个选中专家的未加权输出；普通 target/draft 推理不增加这份状态。EP16、K32、hidden=5120 的生产形状编译通过，小形状实卡结果与 Torch golden 逐元素一致。
- 冻结专家的 clipped-SwiGLU 反向已经实现：生产 `K32×2304` 形状编译通过，小形状实卡与 Torch 参考逐元素一致。它与 down/gate-up 两个冻结 W8A8 输入 VJP 共同构成 selected-expert 的完整输入反向，不更新专家权重。
- resident 反向的调用顺序和参数 ABI 已固化，last-1/2/4/8 均在支持的 C1A 后缀内。当前完整生产规格共 51 个 PyPTO 程序编译通过；MoE/router 新增测试 9 项通过。

## 下一步

剩余工作集中在 serving 运行时：把上述 VJP、EP16 通信、FP32 optimizer 状态和双槽发布绑定到真实 resident session，完成一次 16 卡真实更新。之后在 GSM8K 上先做小规模正确性，再扫描 last-N、学习率和更新频率；Qwen3.8 的设置只作为起点，不直接当作 DSV4.1 的最终参数。
