# Orthrus-DSV4.1 训练交接

部署和训练使用同一份 `Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8` 配置、权重索引和 tokenizer。服务读取 `orthrus-dsv41-head-v1`，块长为 32，包含 anchor 和 31 个待预测位置。当前使用的 MASK token ID 是 128799，初始化脚本从基模配置读取该值。

## 数据和参数

- 教师对真实序列做因果前向，保存全词表 logits。Draft 读取 anchor 前的目标历史，输入 anchor 和 31 个 MASK。块内第 `j` 行对应教师位置 `a+j` 的 logits，两边预测位置 `a+j+1`。离线训练应覆盖每个样本所有有效预测位置，用 forward KL 比较两边的概率分布。
- 基模及共享主干冻结。训练时仅更新选中层的 diffusion Attention 参数。若更新 diffusion indexer，需单独提供监督，语言模型的 KL 梯度无法穿过离散 Top-K。单层交接用第 39 层，沿用第 36 层的检索结果。
- 运行中的教师使用实际 W4A8 前向。单层参考训练采用 CPU FP32 重放，临时反量化用到的冻结专家权重，梯度经过共享计算回到 draft Attention。更新后的权重以 BF16 保存，Attention sink 保留 FP32。

## 复现单层更新

以下命令在仓库的 `online-speculation/` 下执行。单机八卡 smoke 会自动初始化 head、采集一次实际 verifier 反馈、运行一步 KL 更新，再启动服务并检查新 head 可以制造和验证候选。

```bash
HCCL_DETERMINISTIC=strict python3 scripts/watch_dsv41_single_a2.py \
  --model "$MODEL_HOST" --download-marker "$CHECKPOINT_READY" \
  --local-ip "$LOCAL_IP" --nic "$NIC_NAME" --handoff-only
```

`CHECKPOINT_READY` 指向下载流程留下的完成标记，脚本也会校验所有模型分片长度。服务日志和反馈位于 `local/dsv41-deploy/`。交接模式单独验收训练产物和重载。逐 token 严格对照在独立的原生验证器实验里记录。[八卡回放和原生数值差异](ORTHRUS_DSV41_REFERENCE.md#已更新-head-的原生-verifier-对照)

已有反馈时，可以在固定镜像的 CPU 环境单独执行下列命令。一次更新后的 loss、实际变化的张量和重放误差会打印到标准输出。采集样本只包含验证器实际到达的位置，适用于这一步联调。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 TORCHINDUCTOR_CACHE_DIR=/tmp/dsv41-torchinductor \
  OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONPATH=src \
  python3 scripts/train_dsv41_last_layer_smoke.py \
  --model "$MODEL" --head "$INITIAL_HEAD" --feedback "$FEEDBACK" --output "$UPDATED_HEAD"
PYTHONPATH=src python3 scripts/preflight_dsv41_orthrus.py \
  --model "$MODEL" --head "$UPDATED_HEAD" --require-trained
```

## 交付正式 head

训练端导出一份包含全部 40 层和 336 个规范张量的 safetensors，包含保持原值的 draft 层。记录基模版本、初始化权重、训练的层号、MASK ID 和目标数据。下面两条命令生成部署目录并核对基模哈希、张量名、形状和精度。把目录交给服务的 `HEAD_HOST`，通过重启加载。

```bash
PYTHONPATH=src python3 scripts/export_dsv41_orthrus_head.py \
  --model "$MODEL" --mask-token-id "$MASK_ID" \
  --from-trained-head "$TRAINED_SAFETENSORS" --output "$HEAD"
PYTHONPATH=src python3 scripts/preflight_dsv41_orthrus.py \
  --model "$MODEL" --head "$HEAD" --require-trained
```

服务内同步单层更新已有八卡功能验证，CPU 每步约 21 秒。批量后台训练和在线切换 head 仍需验证，交付接口采用独立训练进程与版本化 head 目录。[在线计时记录](validation/dsv41/tp8-online-timings-910b3.json)
