# Orthrus-DSV4.1 调度对照

## 结论

借鉴 DSpark，把起草、验证和有效前缀提交串在设备侧。Lib 中的 Draft、Verify 和 Commit 继续作为各自可验证的组件，Serving 按请求调度一轮执行。在线反向与发布有独立的触发频率。

这个选择来自 111 的 910B3 实卡对照。四卡上，紧凑回传之后再合并调度，中位耗时仍由 39.34 ms 降至 19.90 ms。C2A 层的实卡用例也验证了暂存压缩状态、按通过长度提交的正确性。整模实现仍需将该提交规则扩展到所有缓存和 Engram 状态，并在 EP16 下验证逐 token 一致性和吞吐。

## Draft、Verify 和验收交接

Lib 为 `c2396a5`，使用兼容 PyPTO 镜像 `orthrus-dsv41-pypto:1122-native-manifest-audit`。实验重用 A3 的全词表 greedy sampler，词表 129280，Verify 宽度 8，固定输入为接受 0、3、7 个候选的三组 FP32 logits。每个配置预热 9 轮，计时 120 轮，轮换三个方案的执行顺序。四卡版通过 L3 worker 同时提交四份相同的输入，计时轮次读取一个 owner 的输出。串联路径另逐 rank 核对了三组输入。

| 路径 | 单卡 L2 中位 ms | 四卡 L3 中位 ms | 四卡 L3 P95 ms | 每轮从请求 owner 回传 |
| --- | ---: | ---: | ---: | ---: |
| 分图提交，CPU 读取全词表 logits 并复核 | 64.67 | 85.48 | 135.15 | 7,757,280 B |
| 分图提交，设备检查有限值与验收，紧凑回传 | 35.12 | 39.34 | 88.00 | 288 B |
| 设备侧串联，回传已验收的结果 | 17.79 | 19.90 | 44.10 | 64 B |

三组在同一组输入上返回相同的 token 和接受长度，覆盖跨词表分块的 top-1 并列。设备端还检测了 Draft 和 Verify 中的 NaN、正无穷和负无穷共 6 例。第三组的目标 sampler 通过设备上的数据依赖等待 Draft，避免用两组独立 logits 的并行执行夸大收益。

表中是提交到拿到输出的同步墙钟时间。原始 JSON 分项记录阻塞提交、回传和 CPU 检查。其中提交时间包含设备执行。实验回放已经准备好的 logits，范围是调度和验收交接。整模前向、EP16 通信和在线反向均需单独测量。

- [单卡原始数据](validation/dsv41/pypto-a3-schedule-l2.json)
- [四卡原始数据](validation/dsv41/pypto-a3-schedule-l3.json)

## C2A 有效前缀提交

复用真实 W8A8 checkpoint 的第 2 层压缩投影和 A3 C2A kernel。在 5 组 K8/K16 用例中，直接保留完整 Verify 的环形状态会覆盖已通过前缀需要的前驱。4 组用例进一步改变了后续 latent、FP4 数据和 scale。

原型让 Verify 只读已提交环形状态，保留当轮每行投影，到验收后用设备算子发布有效前缀的尾部状态。跨逻辑页和非连续物理页的 5 组用例均与逐 token Decode 逐位一致。首个用例预热 3 轮后计时 30 轮，现有备份、恢复和重放路径中位数 **37.25 ms**，暂存后设备提交 **34.79 ms**。这个对照保留了独立的提交调用，范围为单层 C2A。

- [C2A 原始数据及差异](validation/dsv41/pypto-a3-c2a-commit.json)

## 复现

在已分配设备的容器里，激活 PyPTO 环境，再把配套 Lib 放到 `PYTHONPATH` 的最前面。从实验仓根目录执行。复现表中原始数据使用 Lib `c2396a5` 和实验仓 `f98f8e5`，当前脚本已适配新版 sampler 的行索引参数。

```bash
export PYTHONPATH="$LIB_ROOT${PYTHONPATH:+:$PYTHONPATH}"
SCRIPTS=online-speculation/scripts
python "$SCRIPTS/benchmark_dsv41_a3_speculation_schedule.py" \
  --devices 0,1,2,3 --output schedule-l3.json
python "$SCRIPTS/validate_dsv41_pypto_verify_state.py" \
  --checkpoint "$CHECKPOINT" --device 0 --resident-commit --output commit.json
```

把第一个命令的 `--devices 0,1,2,3` 换成 `--device 0` 可复现单卡 L2 对照。
