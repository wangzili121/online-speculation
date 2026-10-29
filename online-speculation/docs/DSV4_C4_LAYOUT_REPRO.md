# DSV4 C4 物理页布局敏感性定位

## 定位结果

在四层 DSV4、TP2、K32 的受控投机验证中，已将一个输出差异独立复现到主模型的原生 `npu_sparse_attn_sharedkv`：保持查询、逻辑 KV、检索索引、长度和 sink 不变，仅调整物理页位置并同步修改页表，就能改变一个输出元素。该复现不需要加载主模型或执行 draft。

这解释了已抓取的反例，不能直接代替完整 43 层模型的并发验收。旧概率残留是另外一个已修复的问题，与本节的 Attention 差异分开处理。

## 对照证据

输入来自第 3 层 C4，整个 batch 有 128 行查询。差异位于 `[81, 2, 231]`，对应输入长 511 的请求、绝对位置 529。两次原生目标路径的该元素分别为 `0.478515625` 和 `0.48046875`，相差一个 BF16 间隔 `0.001953125`。此前选中请求的逻辑历史及整个 batch 的查询和索引相同。

| 对照 | 结果 |
| --- | --- |
| 两个输入包各独立重放 5 次 | 每次均逐位复现各自原始输出 |
| 交叉替换两个输入包的 metadata | 输出不变；这些 metadata 差异没有解释该反例 |
| 反转两个 KV 池的物理页，并同步修改页表 | 两个输入包均改变同一个输出元素，最大差 `0.001953125` |
| 将两个输入包重排为相同规范物理布局 | 两个完整 Attention 输出逐位相同 |

这些操作不改逻辑 KV 内容、注意力可见范围、检索集合或模型参数。逐元素 FP64 参考值为 `0.47971800896161576`，其 BF16 舍入结果为 `0.48046875`；它用于描述数值位置，不替代全算子正确性测试。

## 对应实现

容器源码 `csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_scfa_block_vector.h` 的 `CopyInKv` 为合并搬运，计算两个物理地址的正向间隔，并从较小地址开始一次搬运两块：

```cpp
keySrcStride = (abs(keyOffset1 - keyOffset2) - blockWidth) * sizeof(KV_T);
// 可合并搬运时，以较小物理地址作为起点。
startGmOffset = min(validKeyOffset1, validKeyOffset2);
DataCopyPad(destination, cmpKvGm_[startGmOffset], copyTwoBlocks, padding);
```

因此，同一对逻辑 KV 在不同物理布局下会以不同顺序进入后续计算。实测该请求的 C4 页表分别为 `[71,72]` 和 `[66,45]`；同一对逻辑索引 `[0,129]` 的装载顺序随之从 `[0,129]` 变成 `[129,0]`。精确 Attention 对 KV 排列不敏感，但浮点/低精度实现不保证换序后逐位相同；差异又会经动态量化投影传播到后续层。

该源码文件 SHA256：`e0580ca8c6f84ac25875532c25412120b06001d692d002af2522f2852f65bcd9`。

## 复现入口

`check_dsv4_engine_state.py` 的四层诊断命令追加以下参数可保存整组 C4 输入。文件包括完整物理 KV 池，体积较大，仅作诊断，不用于性能统计。

```bash
--shadow-batch 4 --shadow-logit-errors \
--trace-prompt-length 511 --trace-block-start 512 --trace-c4-inputs
```

不加载模型的独立重放：

```bash
python scripts/replay_dsv4_c4_operator.py \
  /path/shadow-token-only-1-rank0.operator.pt \
  /path/shadow-token-only-2-rank0.operator.pt \
  --repeats 5 --probe-layout --reference-element 81 2 231 \
  --output /path/replay-report.json
```

`--cpu-only --compact-output /path/compact` 可导出只包含页表引用页的输入包，保留原物理地址的相对顺序。重放必须使用相同的原生算子版本；输入包不含模型权重，但包含运行时激活，不上传公共仓库。

本次原始两包各约 2GB，精简后各约 12MB。精简包各重放 5 次仍精确复现原输出，metadata 交叉替换、反转物理页及规范布局的三组结果均与原包一致，因而后续无需加载 DSV4 权重即可复查该问题。

## 处理边界

本轮不修改生产 Attention，也不在 draft 路径复制或重排目标历史。若要求严格的跨物理布局逐位一致，修复方向是在原生 C4 算子的搬运/计算路径保持稳定的逻辑 KV 顺序，例如物理地址顺序与逻辑顺序相反时，使用已有单块搬运分支按逻辑顺序装载，再验证精度与性能。这是待验证的原生算子修复方向，不是已应用的补丁。

规范页布局仅是因果诊断，不作为生产解决方案。Orthrus 状态检查和原生算子数值确定性分别记录，严格 logits 验收未通过的结果继续保留；完整 43 层仍需八卡复核。
