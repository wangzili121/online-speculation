# Orthrus-Qwen3.8 PyPTO Serving 迁移

本分支基于 OneServing 的 Qwen3.8 独立交付包接入 Orthrus。原生模型保持不变，训练仍由现有 PyTorch 路径完成，推理侧读取相同的 draft head。

Orthrus-Qwen3.8-27B 已在单卡 A2 上完成真实 head 的 K32 前向、目标缓存只读、拒绝恢复、短请求 greedy 对照及 HTTP 检查。测试覆盖完整 64 层，主模型与 draft 在同一 worker 中共享 backbone。**目前是 TP1、一个活动请求的功能联调版本；逐层数值对照和性能优化仍待完成。**

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

真实神经 draft 的实验验证入口如下。`--audit-block` 单独执行一次完整 K32 起草，逐字节检查起草前后的目标 KV 和 GDN 状态，随后检查已提交历史与接受前缀重放是否一致。默认另生成 3 个 token，比较服务调度路径与 target-only 的续写；增加 `--tokens 33` 可延长请求验证，不以初始化 head 的接受率验收性能：

```bash
python scripts/validate_qwen38_pypto_neural.py \
  --source /sources/OneServing --model /models/Qwen3.8-27B \
  --head-dir /heads/version4 --device 0 --audit-block --output /results/neural.json
```

`serve_qwen38_pypto.py` 保留原生 OpenAI API、SSE、停止与取消处理：

```bash
python scripts/serve_qwen38_pypto.py \
  --source /sources/OneServing --model /models/Qwen3.8-27B \
  --head-dir /heads/version4 --device 0 --port 8000
```

`/metrics` 在原生指标之外返回 `orthrus` 字段，包括投机轮数、接受数、恢复次数和真实神经 draft 调用次数。使用 `smoke_qwen38_pypto_base.py` 时添加 `--head-dir /heads/version4 --tokens 3`，可检查 completion、chat、SSE、draft 调用计数及服务关闭。

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

本次真实 NPU 验证采用宿主机原生用户进程，使用 Python 3.10.21、Torch 2.10.0+cpu、CANN 9.1.1 和驱动 26.1.1；PyPTO、Simpler、PTOAS、PTO-ISA 仍为上表固定版本。Python/Torch 负责权重与主机侧数据，模型计算由 PyPTO 程序在 NPU 执行。

在可写的用户源码目录中，使用宿主机 CANN 重建设备运行时：

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.1
python -m simpler_setup.build_runtimes --platforms a2a3
```

依赖准备好、分配到空闲设备后，可执行下面的原生服务检查。`ASCEND_RT_VISIBLE_DEVICES` 指定物理卡，脚本中的 `--device 0` 对应可见设备的第一个逻辑编号：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python scripts/smoke_qwen38_pypto_base.py \
  --source /sources/OneServing --model /models/Qwen3.8-27B \
  --device 0 --output-dir /results/native-smoke-1
```

脚本启动独立服务，检查 completion、chat、SSE 和 metrics，随后正常关闭；保留 `server.log` 和 `result.json`。输出目录必须为新目录。上面的命令不加载 Orthrus head。

模型从其他下载工具复制过来时，可能缺少 OneServing 检查的 Hugging Face 下载记录。本次在 61 上逐一核对了 18 个权重分片和推理配置的 SHA256，确认与已验证副本一致，再在独立模型视图中复制原始下载记录。共享模型目录未修改，版本校验保持开启；哈希记录见 `validation/pypto-qwen38/http-20261105/checkpoint.json`。

## 验证与后续

已通过 275 项 CPU 用例，覆盖 K32 每个拒绝位置、输出截断、停止 token、异常恢复、跨页回收、递推状态恢复、缓冲边界、TP 参数合并及原始设备错误保留。另读取真实 27B 离线训练导出的 TP4 head，检查完整 64 层的布局转换和 dtype，保持输入只读。构建脚本已实际执行通过；五个 draft 入口通过前端编译和 PTOAS 代码生成，向量计算通过模拟器执行（误差 0）。初次验证证据见 `validation/pypto-qwen38/port-20261105/`，缓存布局修正后的记录见 `neural-20261105/`。

真实 NPU 已完成以下检查，证据见 `validation/pypto-qwen38/target-only-20261105/`：

- 固定软件栈的向量加法实际在 NPU 执行，最大绝对误差为 0。
- 完整 64 层 BF16 主模型在单张 64 GiB A2 上加载，以 TP1、一个活动请求运行。
- completion 返回文本，chat 对 `1+1` 返回 `2`；SSE 返回 token 及 `[DONE]`。
- 三次请求完成、请求错误数为 0；应用正常关闭，服务和 NPU worker 进程均退出，测试卡恢复无用户进程状态。

此次原生服务的 HBM 峰值约 57.3 GiB，包含当前运行时配置；这是短请求检查，不是吞吐、长上下文或完整数值一致性验收。

K32 验证与拒绝恢复已在同一真实模型上通过，证据见 `validation/pypto-qwen38/transactions-20261105/`。全接受、首位拒绝、第 17 个候选拒绝、最后一个候选拒绝四类用例均生成 33 个 token，与 target-only 逐个一致；首位拒绝用例连续执行 31 次恢复。验证跨越 16 行 tile 和 KV 页边界，结束后请求均已释放。这里的候选由测试 fixture 提供，只证明目标验证与缓存事务可工作，不证明神经 draft 前向、接受率或加速效果。

真实 head 检查使用 TP4 离线更新 4 步导出的测试 head，按层合并并在 TP1 上执行，证据见 `validation/pypto-qwen38/neural-20261105/`：完整 K32 起草返回 31 个候选；起草前后全部目标 KV、卷积状态、递推矩阵及页归属保持不变。验证接受 1 个候选后发生拒绝，提交后的历史与仅重放接受前缀逐字节一致，随后 3 步 AR 续写一致。另通过真实 scheduler 生成 3 个 token，与 target-only 完全相同，确实调用了神经 draft；请求释放、worker 与验证进程退出均已确认。

首次神经 draft 执行暴露了运行时对缓存视图连续性的要求：选中 slot/layer 后须先去掉相应长度为 1 的维度，再 reshape。GDN 和 Full Attention 读取均已按此修正。本次检查证明了前向与验证链路可运行，尚未比较 PyPTO 与 PyTorch draft 的逐层数值，也不代表长请求、并发或生产验收。测试 head 只经历少量更新，且逐层搬运参数的路径尚未优化，不据此评价接受率或加速。

Orthrus HTTP 检查在 61 的单张 A2 上通过，加载的是 TP2 在线 last-4 更新导出的 version298 测试 head，部署仍为 TP1。completion 返回 `, I'm`，chat 对 `1+1` 返回 `2`，SSE 正常返回 token 和 `[DONE]`。三次请求全部完成，错误与取消数为 0；指标记录 3 次神经 draft、3 轮验证和 1 次恢复。服务正常关闭，验证进程、服务进程及 NPU worker 均已退出，证据见 `validation/pypto-qwen38/http-20261105/`。

当前逐层合并、校验和搬运 head 的路径仍然很慢，实测单轮起草约 59–147 秒。设备执行日志表明，大量时间位于设备调用之间，后续需拆分测量权重准备与传输开销，并设计参数驻留路径；这份结果用于功能验证，不作为投机加速部署配置。

早期容器中的设备枚举错误记录保留在 `port-20261105/`。本次通过用户目录下的原生环境完成验证，没有修改 OneServing 模型源码；原容器启动失败的具体原因仍未确认。

下一步补充逐层数值对照、更多请求边界和 head 重载，并优化 head 的准备与驻留方式。随后扩展 TP 和采样；训练和在线更新不在本轮迁移范围内。
