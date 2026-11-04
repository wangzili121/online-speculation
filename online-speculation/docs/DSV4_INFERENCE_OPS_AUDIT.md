# DSV4 推理算子复核

日期：2026-11-03。范围仅为推理，未启用训练或在线更新。

## 执行路径

主模型仍使用固定镜像的原生 DSV4。正式 draft 位于 `src/blockspec/vllm_ascend/dsv4.py`，共享 mHC/norm/MoE 等主干；独立投影产生 Q、临时 KV 以及 C4 indexer query。

| 部分 | 当前选择 | 复核结果 |
| --- | --- | --- |
| Q/KV 投影 | 共用输入动态量化，原生 quant matmul、RMSNorm/量化和 partial RoPE | 已有原生融合方式保留，没有换成逐算子 FP32 路径。 |
| SWA + 临时块 | 一次 FIA，mask 仅屏蔽历史 padding | 36 组对照通过，含空/部分/完整窗口、batch 1/2/4、8/16/32 heads。 |
| C4 | 量化 Lightning Indexer + 原生分页 sparse Attention | 6 组 Attention 数值、LSE 和 cache 不变检查通过；含 262144 压缩行。Indexer 本轮未重测。 |
| C128 | 大页直接 FIA，小页 sparse Attention，保留 gather 回退 | 8 组通过，含 0 至 8192 压缩行及多请求，不以 Top-K 代替全部历史。 |
| 最终归约 | FP32 sink + `npu_attention_update` | 合并专项通过；原生约 229 μs，Torch 组合约 560 μs。 |
| 单层展示代码 | SWA/临时块一次 FIA，原生归约 | 本轮将旧拆分调用对齐正式路径，K=2/32、SWA/HCA/CSA 共 6 组 FP32 Attention 对照与只读检查通过。 |

固定镜像的 FIA v2 sink 降精度已有失败记录，不因接口较新就启用。以上为明确测试范围内的选择，不是所有形状下的全局最优声明。

## 证据与复现

原始结果位于 [本轮目录](validation/dsv4/ops-20261103/)，保留命令、退出码、日志与 JUnit。单卡在 111 的 NPU 0 执行，用后释放；未停止其他用户任务。

- 镜像：`quay.io/ascend/vllm-ascend@sha256:0bc946247f2ae008fdf73838e597404fbbfebac4692c703eba531c05a9b5c58a`。
- CPU：620 passed、33 skipped。CPU 临时容器内移除 torch-npu，NPU 镜像不修改。
- Attention 合并与独立 FP32 拼接参考的最大绝对误差约 0.00358。
- 单层对照保留同一套投影与参数，将 Attention 替换为 FP32 参考，不是完整模型误差界。
- 合成历史与 anchor 对齐后重新通过 6 组单层测试，记录在 `layer-anchor/`；它们是同一组测试的复验，不重复累计用例数。

```bash
python scripts/run_in_runtime.py scripts/smoke_dsv4_hybrid_npu.py --iterations 20
python scripts/run_in_runtime.py scripts/benchmark_dsv4_attention_paths.py --case joined --iterations 10
python scripts/run_in_runtime.py scripts/benchmark_dsv4_attention_paths.py --case sparse --iterations 10
python scripts/run_in_runtime.py scripts/benchmark_dsv4_attention_paths.py --case c128-sparse --iterations 10
ORTHRUS_RUN_NPU_TESTS=1 PYTHONPATH=src python -m pytest -q tests/test_dsv4_layer_npu.py
```

本轮未重跑完整 43 层 TP8，不能替代端到端验收。此前原生 C4 布局舍入问题仍独立归档。PyPTO 的接口、公式与组合代码见 [需求报告](../../orthrus_dsv4_pypto_requirements.md)。
