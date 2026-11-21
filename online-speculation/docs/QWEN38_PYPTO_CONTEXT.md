# Qwen3.8 TP2 的 1,024 上下文实验路径

新增可选的 `--max-model-len 1024`，用于容纳较长的 Pseudo2code 输入。AR、固定起草和持续在线使用同一个总上下文上限；默认仍为 512。head 的参数、K32 存储契约、逻辑块和接受规则不变。

分页 Attention 增加 1,024 容量分支，每个 query 读取 anchor 前的真实历史和整个有效 draft 块。只在总有效长度超过 512 时进入该分支；1,024 上下文使用 16 行 query tile 控制片上空间。目标 KV 保持只读，临时 draft KV 仍不提交到目标 cache。

服务侧页表、mask 和内存预算，在线反馈的历史缓冲、分页采集和训练图都随上下文一起扩展，避免服务看到了后半段历史而学习器只收到前 512 个位置。训练行数仍可独立选择 32 或 last-1 的 16；扩展历史不改变权重分片。

已通过 572 项 CPU 检查，包含 1,024 历史末端的独立 autograd 核对。实机通过两种页尺寸的 38 个 Attention 用例，覆盖非整页边界、无效位置填 NaN 和输入只读检查；小形状的 6 个 TV / KL 前反向用例分别核对 77 个中间结果，最大相对 L2 为 0.00211。真实 27B 权重的 6 组 last-1 前反向中，16 / 32 行的有效 logits 和六类参数梯度均逐位相同，输入与历史不变。

本次真实权重对照中，32 行约 37 ms、16 行约 30 ms；这是前反向计时，不含优化器、发布和整模型推理。首次真实权重检查在工作区执行前遇到设备内存分配失败，保留原始记录；换空卡后的重试未修改 kernel 或输入。单卡检查结束后已释放设备。

首次两卡整模型检查在完成 AR 预热、开启起草后暴露出接入错误：外层页表为 8 页，Full Attention 内部仍只取 4 页，超过 512 历史时发生越界。已将内部 view 改为随上下文计算完整页数，并在 Attention 入口增加固定元素数的 view；外部传入短页表或内部截短到 4 页，都会在编译期失败。两条负例、512 / 1,024 两条正常编译路径、572 项 CPU 检查和 TP2 编排编译通过，原失败完整保留。

修正后的两卡整模型验证已通过：六条旧题补齐到 600-token 输入后，32 行和 16 行两种训练布局均完成生成、教师反馈、11 次梯度更新、11 次发布、最终 head 重跑及导出；greedy 输出与 AR 逐 token 相同。两种布局的更新轨迹和实际发布的两个权重分片均逐位一致，反馈重放最大 TV 为 0.00420、相对 L2 为 0.00284。这里只验证长历史链路，不作为新题吞吐达标证据。GSM8K 三次确认继续使用其冻结的 512 版本，不涉及本次改动。

独立检查入口：

```bash
python scripts/validate_qwen38_pypto_attention.py \
  --context 1024 --backend paged --query-tile 16 \
  --q-heads 12 --kv-heads 2 --page-size 128 \
  --device 0 --output attention1024.json

python scripts/validate_qwen38_pypto_training_rows.py \
  --context 1024 --model /path/to/model-view --head /path/to/tp2-head \
  --device 0 --output training1024.json

python scripts/validate_qwen38_pypto_page_contract.py --output page-contract.json
```

`--compile-only` 只生成程序；`--assemble-only` 进一步编译设备二进制，不创建 NPU worker，也不提供性能结论。后者调用当前固定 PyPTO 版本的内部编译入口，升级运行时后应重新核对。缺少 Git 元数据的导出环境可通过 `--runtime-artifacts` 提供已核对的文件清单，校验源码、原生绑定、实际加载的 Simpler 扩展和工具链；不伪造 Git 信息，也不关闭运行时检查。

LongProc 官方题池按问题 ID 和三台机器实际执行文本排重后，固定划为 30 题探索、24 题确认和 8 题独立留出。尚未执行的新确认题只在探索完整达标、配方冻结后使用。源码、编译、失败记录与分配清单保存在 `validation/pypto-qwen38/online-datasets-20261119/context273/`。
