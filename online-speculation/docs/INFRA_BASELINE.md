# Orthrus 单卡 Infra 基线

本阶段冻结算法，只调系统。唯一固定配置见
`configs/orthrus_single_npu.toml`。其中 `block_size=32` 在 vLLM 接口中对应
`num_speculative_tokens=31`：一个已提交 anchor 加 31 个候选组成 32 行
verify，不是 32 个候选。

## 基线边界

- B0 是上游独立 PyTorch 实现，用于记录原始执行形态。
- B1 是共享物理 backbone 的 vLLM-Ascend 0.23.0 实现。
- 普通 AR 只做 greedy、随机分布、rejection correction 和 KV crop
  正确性 oracle，不参与吞吐选参。
- `runtime/vllm_ascend_v0_23` 只补齐 custom proposer、概率校正和 K32
  attention fallback，属于兼容接入，不计优化收益。
- 独立加载第二份完整 draft/backbone 的实现只能作为接线 smoke，不能命名为 B1。

## 当前完成度

- [x] 导入并记录上游 SHA。
- [x] 支持单文件和 sharded safetensors 权重。
- [x] PyTorch runner 支持 CPU/CUDA/NPU 设备抽象。
- [x] NPU tiny 模型 fixed-draft exact correction smoke。
- [x] NPU tiny 模型 online TV backward/publish smoke。
- [x] vLLM-Ascend 0.23 custom proposer 与概率校正兼容补丁。
- [x] 官方 Orthrus-Qwen3-8B B0 fixed、greedy 和 online TV NPU smoke。
- [x] vLLM shared-backbone Orthrus 模型和 paged-KV draft view。
- [x] PyTorch AR 与 vLLM AR 在相同模板下连续 64 个 greedy token 一致。
- [x] B1 eager 与 `FULL_AND_PIECEWISE` 单请求 smoke，后者确认 ACL Graph replay。
- [x] 256 条 6K--40K 混合编码 workload 和流式 TTFT/TPOT 测量入口。
- [ ] B1 系统参数搜索和正式 profiling。
- [ ] Draft/Verify 协同调度、异步续训、sparse draft KV、DDTree。

## 验收纪律

所有性能实验使用同一模型、请求 manifest、seed、算法参数和系统配置。
系统配置先按 SLO goodput 选 B1；后续每项优化只和 B1 A/B。单项收益不足
5% 不保留。自定义算子仅在 trace 同时满足计划中的 busy-time、现有算子
不可消除和预期端到端收益门槛后启动。

## Smoke 结论

- B0 K32 fixed draft、一次 online TV update、参数发布与恢复均已在 8B
  checkpoint 上通过。
- B1 使用一份 18.1 GiB 模型权重；drafter 只持有 runner 引用，没有加载第二份
  backbone。
- `FULL_AND_PIECEWISE` 可编译并 replay；短请求 smoke 的时间受编译缓存和预热
  影响，只用于 capability 判断，不作为收益数字。
- 第一条 7.7K prompt、1024-token 服务请求成功；它验证 streaming 指标采集，
  并非并发性能结论。
- vLLM 0.23 当前会关闭 `custom_class` speculative decoding 的 async scheduling。
  该项在兼容与正确性验证前不能写成“已启用”。
