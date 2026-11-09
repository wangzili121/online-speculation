# PyPTO 权重加载验证

固定 OneServing/PyPTO 环境，Ascend A2 单卡，真实 Qwen3.8-27B 与 TP4 训练导出的测试 head。服务仍为 TP1、K32、512 总位置及 greedy；这里验证加载和推理链路，不评估 head 的训练质量。

## 加载路径

原始 safetensors 由主机读取；head 的分片重组、卷积转置、补零、norm 加一、类型转换、有限值和复制参数检查由 PyPTO 在 NPU 执行。主模型的大矩阵按层直接 DMA 到最终设备 bank，不在 CPU 上拼接整块矩阵。设备分配和传输使用 PyPTO runtime。

转换后的 head 可以保存为 `orthrus-qwen38-pypto-banks-v1`。重载时校验基模、K、mask、运行 TP/rank、PyPTO 版本、长度和 SHA256，然后读取字节并建立 bank 视图。文件 I/O、JSON、SHA256 和主机内存视图仍使用主机接口；权重的 CPU 数值转换仅用于本次独立对照或显式 `--weight-backend host`。

## 原始记录

- `kernels.json`：11 项独立 NPU 检查，覆盖真实分片尺寸的行/列复制、补零、转置、norm、A_log 类型和 norm 副本；包含主模型卷积的归约尾块，以及拒绝 NaN、Inf 或不一致副本的用例。
- `head.json`：64 层、528 个 head bank 与独立 CPU 转换结果逐字节一致，约 14.56 GB 的转换结果保存并重载后逐字节一致。
- `model.json`：主模型全部 23 个 bank 与原生 CPU 加载结果逐字节一致。真实 K32 起草的完整 logits 哈希及候选与旧加载路径一致；目标缓存不变，接受前缀恢复、后续续写和四 token 调度输出均通过。
- `http.json`：completion、chat、SSE 实际执行 neural draft，服务报告 `weight_backend=pypto`、`head_preparation_backend=pypto`，正常退出。
- `cpu-tests.txt`：430 项 CPU 回归通过，包含 TP 区间、缓存契约、数据损坏和越界拒绝。
- `environment.json`、`source-sha256.json`：固定环境与测试源码哈希。

head 的原始 CPU 准备约 41.3 s，PyPTO 准备约 155.0 s（含编译），已保存结果重载约 31.8 s。该组记录用于说明加载成本，不能据此声称 NPU 首次转换更快，也不是 token 吞吐对照。

## 复现

在固定环境、已分配的设备上，从 `online-speculation` 目录执行；输出目录应为新目录。

```bash
python scripts/validate_qwen38_pypto_loading.py \
  --device 0 --output /results/loading-kernels.json

python scripts/validate_qwen38_pypto_head_loading.py \
  --source /opt/OneServing --model /models/Qwen3.8-27B \
  --head-dir /models/trained-head --device 0 \
  --prepared-dir /results/prepared-head --output /results/loading-head.json

python scripts/validate_qwen38_pypto_loading_model.py \
  --source /opt/OneServing --model /models/Qwen3.8-27B \
  --head-dir /results/prepared-head --device 0 \
  --output /results/loading-model.json

python scripts/smoke_qwen38_pypto_base.py \
  --source /opt/OneServing --model /models/Qwen3.8-27B \
  --head-dir /results/prepared-head --device 0 --tokens 4 \
  --attention-backend paged --gdn-backend vector \
  --draft-group-size 4 --head-transfer packed \
  --output-dir /results/loading-http
```

`validate_qwen38_pypto_loading_model.py` 可附加 `--reference-json`，与 `compare_qwen38_pypto_serving.py` 保存的 `result.json` 比较完整 draft logits 哈希和候选。CPU 权重转换与大矩阵回读只用于验收，不进入正常服务的每轮起草。
