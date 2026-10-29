# Orthrus-DSV4 暂停交接

记录日期：2026-10-29。按当前工作安排暂停 DSV4 的训练、在线更新及推理接入实验，下一阶段评估 Qwen/Qwen3.8-27B。不删除 DSV4 代码、权重、初始化 head 或失败证据，不继续抢占八卡或启动监控。

## 恢复基线

- 仓库：`wangzili121/online-speculation`，分支 `feature/model-aware-custom-proposer`。
- 暂停前最后实现提交：`1e7d805`；原生 C4 独立复现提交：`426a86b`；逐层抓取：`abe9bdf`；旧概率残留修复：`e9ef9f5`。
- 镜像：`quay.io/ascend/vllm-ascend:nightly-releases-v0.26.0rc`。
- 固定镜像 digest：`sha256:0bc946247f2ae008fdf73838e597404fbbfebac4692c703eba531c05a9b5c58a`。不要用更新后的同名 tag 直接覆盖复现环境。
- 111 主模型：`/data/disk/models/DeepSeek-V4-Flash-0731-w8a8`。
- 完整初始化 head：`/data/disk/models/orthrus-dsv4-head-v2-w8a8-init-fa99ed1`。这是链路测试 head，不是训练合格的加速 head。
- 111 调试项目：`/data/disk/wangzili/orthrus-debug-20261029/online-speculation`；容器内为 `/workspace/online-speculation/validation-20261029`。
- 本地证据：工作目录下 `debug-20261029/evidence/`，其中 `c4-layout-rootcause/` 有约 12MB 的独立重放输入，`c4-isolation/` 有四层/12 层配对报告。运行时激活和模型权重不上传公共仓库。

2026-10-29 再次确认自有调试容器 `wangzili-orthrus-debug-1029-npu01-priv` 已退出。本次转向不启动新 NPU 任务；此前 6/7 卡验证任务已释放。该容器名中的 `npu01` 是历史名称，最后一轮实际使用的是物理 6/7。

## 已完成与未完成

已实现固定版本的专用 proposer、共享主干、SWA/C4/C128 只读历史、K32 双向块、原生目标验证及概率传递。算子组合、分页、状态协作、请求换序、取消恢复和受控采样已有分层测试。跨批次旧 draft 概率残留已修复；不要与下面的 C4 数值问题混为一谈。

四层和 12 层真实截取模型、初始化 head、TP2、eager 已做相同候选/调度下的严格对照。仅在测试中规范目标 C4 压缩页布局后，参考重复、参考与真实 draft、真实 draft 与跳过 head 三组对照，每 rank 各 74 个事件的完整 logits 全部逐位相同。每轮每 rank 449 个有效状态验证事件、306 个真实 draft 事件通过。恢复原生布局，两种模型均重新出现差异。CPU readiness 为 171 passed、3 skipped（需要 CUDA graph 硬件），head/评估契约检查通过。

原生 C4 的一个具体问题已有独立因果复现：`CopyInKv` 合并搬运会按物理地址排列一对检索 KV，物理页变化改变装载顺序，产生低精度数值差异；统一布局后差异消失。未修改生产 kernel，也未将诊断页复制加入生产路径。原始失败退出码与报告保留。

完整 43 层曾在 TP8 跑通启动、重启、32K/128K 输入及部分服务生命周期，但跨服务、并发输出仍有未完成验收的差异。双卡隔离通过不等于完整模型验收通过，也不能证明所有差异同源。正式 graph、1M、持续负载、训练 head 接受率及加速收益均未验收。DSV4 全尺寸离线训练和在线更新没有完成，不能记录为“只差替换训练 head 即可生产使用”。

## 恢复顺序

1. 校验固定镜像、运行时补丁及模型/head 契约，先确认初始化 head 和训练 head 的区别。
2. 用 `DSV4_C4_LAYOUT_REPRO.md` 的独立输入复查原生 C4，确认算子侧是否已有保持逻辑顺序的修复。
3. 在完整可用的八卡节点先跑 target-only，再做 Orthrus 同配置对照；原生布局与诊断布局结果分开保存。
4. 若还有差异，按请求 ID、验证位置和逐层记录定位，不用已知 C4 问题豁免其他错误。
5. 推理正确性完成后，再单独安排训练/在线更新、graph、长上下文和业务性能验收。

详细报告分别见 `ORTHRUS_DSV4_NPU_HANDOFF.md`、`DSV4_VALIDATION_20261028.md`、`DSV4_C4_LAYOUT_REPRO.md` 和 `DSV4_VALIDATION_RUNBOOK.md`。它们保留完整公式、实现说明和复现入口，本文件只固定暂停状态与恢复顺序。
