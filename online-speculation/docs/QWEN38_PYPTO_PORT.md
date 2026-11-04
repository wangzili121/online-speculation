# Orthrus-Qwen3.8 PyPTO Serving 迁移

本分支基于 OneServing 的 Qwen3.8 独立交付包接入 Orthrus。原生模型保持不变，训练仍由现有 PyTorch 路径完成，推理侧读取相同的 draft head。

当前已实现 head 读取与布局转换、K32 神经 draft 的 PyPTO 前向编排、greedy 候选校正、KV 页回收和 GDN 状态恢复适配。五个 draft 入口已通过固定版本的前端编译和代码生成；**尚未通过 NPU 端到端验收，不能视为已可部署的 Orthrus PyPTO 服务。** 原有 vLLM-Ascend 路径不受影响。

## 固定基线

| 组件 | 固定版本 |
| --- | --- |
| OneServing | `7aa40ba5238e3a83bb5f25e18817497c181e3e9b`，子目录 `pypto-serving-qwen3.8-27B` |
| PyPTO | `98752784548b777b2377a9084411c3716eb341d6` |
| Simpler | `c1f89ea431f5306f78c0e4ab9573c829756d7f1e`，由 PyPTO gitlink 固定 |
| PTO-ISA | `418de18dde8fa33773dc5c384fdfe5506463b628`，由 Simpler 固定 |
| PTOAS | `v0.65`，wheel 哈希见 PyPTO `toolchain/versions.env` |

参考 [OneServing 交付说明](https://github.com/ndleslx/OneServing/tree/7aa40ba5238e3a83bb5f25e18817497c181e3e9b/pypto-serving-qwen3.8-27B) 和 [PyPTO 官方容器说明](https://github.com/hw-native-sys/pypto/blob/98752784548b777b2377a9084411c3716eb341d6/docs/en/user/04-dev-container.md)。已有容器可以作为构建底座，但其内置 PyPTO/Simpler/PTOAS 不一定符合交付包，必须逐项核对，不关闭 ABI/pin 检查。

这份基线的 TP2 仅分片 LM head，其他权重仍在各 rank 上复制，不能据此估算完整 backbone 分片后的显存。原生上限为 512 个位置、greedy；其交付目录的短 smoke 不能替代完整数值和缓存验收。

## 代码与接口

代码位于 `src/blockspec/pypto_qwen38/`：

- `baseline.py`：检查 OneServing 与运行时版本，在独立模块命名空间导入原生代码。
- `head.py`：读取 `orthrus-qwen38-native-tp-v1`，检查基模哈希、K、mask、混合层策略和 tensor schema，按层合并 TP1/2/4 head 分片。
- `greedy.py`：比较整块候选；第一次拒绝时输出目标 token，全接受时输出 bonus；输出上限和停止 token 同样限制提交长度。
- `target.py`：在原生 runner 上执行验证；需要拒绝后缀时，恢复验证前的 GDN 状态和页表长度，再仅重放接受前缀。它不是通过截断长度来恢复递推矩阵。
- `programs.py`：组合原生投影、norm、GDN、Full Attention 和 MLP。K32 分为两个 16 行 tile；Full Attention 先准备整个块的 K/V，再计算两个 tile 的 query，读取主模型历史但不写目标缓存。
- `draft.py`：将 draft 与目标程序提前注册到同一 worker，共享 embedding、norm、MLP 和 LM head；按层加载 draft attention 参数，控制额外设备内存。

Q/K/V-packed 的 GDN 投影与卷积按 Q、K、V 三段分别合并；O projection 按列合并；复制 norm 必须在各 rank 一致。真实 vLLM head 的 `A_log` 是 FP32，转换时保持原精度；decay/beta 编排使用相同公式，去掉原生辅助函数对该参数的重复转换。其他参数按实际 dtype 校验。

Full Attention 的整个临时块双向可见；GDN 延续 head 契约的 `orthrus_qwen35_reference` 因果递推，在目标状态的临时副本上计算。逐层加载 head 的路径用于控制首次联调的内存，仍有大量 CPU 到 NPU 的权重搬运，尚无吞吐或加速结论。

初版事务适配仅针对 TP1、一个活动请求。原生 scheduler 每次消费一个 token，适配器缓存已验证的输出前缀，逐个交付；后续交付不再次运行模型，且检查请求位置和 anchor，拒绝错位的旧缓冲。真实 draft 应提供 `propose(request, anchor, width)`，读取已提交历史，不写入目标缓存。

## 运行入口

在 `online-speculation/` 内安装本项目，模型和 head 均作为只读外部目录挂载。原生服务入口只启动主模型基线，**不启用 Orthrus**：

```bash
python scripts/serve_qwen38_pypto_base.py \
  --source /sources/OneServing --model /models/Qwen3.8-27B \
  --devices 0 --tp 1 --port 8000
```

检查真实 head 的基模契约、64 层参数、布局和 dtype，不执行 NPU 计算：

```bash
python scripts/audit_qwen38_pypto.py \
  --source /sources/OneServing --model /models/Qwen3.8-27B \
  --head-dir /heads/version4 --mode head --output /results/head.json
```

将 `--mode` 改为 `environment` 可检查固定 PyPTO/Simpler/PTOAS；改为 `compile` 可编译五个 draft 入口并检查 PTOAS 产物，不分配 NPU。所有审计输出必须使用新文件，保留失败证据。初始化 head 可省略 `--head-dir`，仅用于参数与链路检查，不代表训练后的接受率。

`validate_qwen38_pypto_flow.py` 是原生目标模型事务验证入口：先运行 target-only，再使用显式 oracle/拒绝候选检查验证和恢复。**候选来自测试 fixture，不运行神经 draft，不作为接受率或加速证据。**

真实神经 draft 的实验验证入口如下。默认生成 3 个 token，用于首次检查完整 64 层前向；后续增加 `--tokens 33` 检查 K32 请求续写和恢复，不以初始化 head 的接受率验收性能：

```bash
python scripts/validate_qwen38_pypto_neural.py \
  --source /sources/OneServing --model /models/Qwen3.8-27B \
  --head-dir /heads/version4 --device 0 --output /results/neural.json
```

`serve_qwen38_pypto.py` 接受相同的模型、head 和设备参数，保留原生 OpenAI API、SSE、停止与取消处理；目前是待实机验收的实验入口。

## 环境复现

官方开发镜像可作为构建底座。本次使用的镜像 digest 为 `swr.cn-east-3.myhuaweicloud.com/cloud-pypto/pypto-dev@sha256:7712c0bae0f8a3c46cae127be4802324682e33ad1c49436e17db7bd60975dbee`，内置版本与交付包不同，需要重建固定依赖。

先将上表固定版本的 OneServing、PyPTO 及其子模块准备好；PTO-ISA 放在 `PyPTO/runtime/build/pto-isa`，PTOAS wheel 从 `hw-native-sys/ptoas` 的 `v0.65` release 获取。容器内执行：

```bash
/opt/pypto/venv/bin/python scripts/build_qwen38_pypto_environment.py \
  --pypto-source /sources/pypto --oneserving-source /sources/OneServing \
  --ptoas-wheel /wheels/ptoas-0.65-cp310-cp310-manylinux_2_34_aarch64.whl \
  --output-dir /results/build-1 --jobs 4
```

构建脚本检查源码 revision 和 wheel SHA256，保留各阶段日志。依赖采用 OneServing 自身的 `requirements.txt`，其中 Transformers 固定为 `5.17.0`。构建后使用该 Python，将 `/opt/pypto/venv/bin` 放到 `PATH` 首位，并将 `/sources/pypto/python` 与本项目 `src` 放入 `PYTHONPATH`；清除旧镜像的 `PTOAS_ROOT`，避免重新选中旧工具链。

先运行 `probe_qwen38_pypto_environment.py --device 0 --output /results/probe.json`，通过后再加载真实模型。它检查版本并实际执行向量加法，输出文件必须使用新路径。

## 验证与后续

已通过 273 项 CPU 用例，覆盖 K32 每个拒绝位置、输出截断、停止 token、异常恢复、跨页回收、递推状态恢复、缓冲边界与 TP 参数合并。另读取真实 27B 离线训练导出的 TP4 head，检查完整 64 层的布局转换和 dtype，保持输入只读。构建脚本已实际执行通过；五个 draft 入口通过前端编译和 PTOAS 代码生成，向量计算通过模拟器执行（误差 0）。证据见 `validation/pypto-qwen38/port-20261105/`。

实机目前停在模型执行之前：服务器的 Ascend Docker Runtime 无法创建容器；显式映射空闲设备后，向量加法仍在设备枚举/初始化阶段报 `drvGetDevNum=87`、`507899`。这些记录不能证明模型或 draft 数值正确，也不能仅据错误码确定驱动与 CANN 不兼容。详细原因需要结合主机的容器运行时日志排查，该日志目录仅管理员可读。

下一步先恢复可用的 NPU 容器运行环境，再验证原生模型、真实 draft 的共享历史与缓存不变性、候选验证与拒绝恢复，以及 head 重载。随后扩展 TP、采样和性能优化；训练和在线更新不在本轮迁移范围内。
