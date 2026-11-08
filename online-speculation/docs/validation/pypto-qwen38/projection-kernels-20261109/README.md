# K32 draft 投影与归一化

基于 `6239d15`，在固定 PyPTO/Simpler/PTOAS 与真实 A2 上执行。范围是独立 K32 投影、输入 RMSNorm、GDN 辅助计算及递推分块复测；真实模型接回结果另存，不由这些独立检查替代。

`projection32-release.json` 的 42 项检查覆盖六种真实投影形状、两组输入、原生与 N64/N128 实现，以及原生/每任务一行/每任务两行的 RMSNorm。投影与原生逐字节相同，全部输出通过固定 Torch 误差门槛，参数和输入不变。`preparation32-aux.json` 的六项检查验证 32 行 gate、gated norm 和输出投影。

`projection32-trial.json` 和 `projection32-aligned.json` 保留探索记录，`rejected_compile` 明确记录未采用的编译配置；正式 `projection32-release.json` 不允许以跳过编译失败换取通过。`gdn-retile16/32/64.json` 分别包含十二项有效宽度、零/非零初态和污染无效行检查，最终保留 64 分块。

`source.zip` 是正式独立验收源码。`device-profiles.zip` 保存 345 份计时外采集的时间线，`manifest.json` 将原始路径、归档项和 SHA256 对应起来，并记录四个源文件的哈希。归档不含模型、head 权重或真实激活。
