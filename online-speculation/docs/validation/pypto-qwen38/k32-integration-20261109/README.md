# K32 执行与 NUMA 对照

固定 Qwen3.8-27B、同一测试 head、A2 及配套 PyPTO/Simpler/PTOAS。完整服务范围为 TP1、一个活动请求、512 个总位置和 greedy；不以测试 head 的接受率判断业务加速。

## 各组比较什么

| 记录前缀 | 左侧/基线 | 右侧/候选 | 固定不变的部分 |
| --- | --- | --- | --- |
| `preparation32-*` | GDN 准备与 O projection 按 16+16 执行 | GDN 准备与 O projection 按 K32 执行 | Vector 递推、Full Attention、共享 MLP16 |
| `full-preparation32-*` | Full Attention 准备与 O projection 按 16+16 执行 | Full Attention 准备与 O projection 按 K32 执行 | GDN 准备32、共享 MLP16 |
| `shared-preparation32-*` | 共享 MLP16 | 共享 MLP32 | GDN 与 Full 准备32、Vector 递推 |
| `full-preparation32-isolation` | `full16`：Full 准备16 | `full32`：Full 准备32 | 共享 MLP32、GDN32、O projection32；同一个真实层输入 |
| `k32-release-numa-dual` | `remote`：N6 主机内存 | `local`：N0 主机内存 | 同一 NPU、计算程序、设备缓冲和完全相同的 head |

前三组 JSON 的 `prepare16/prepare32` 分别表示该行的基线和候选。`layered` 是逐层编排，`group4` 是四层编排，不跨组混用时间。`device_seconds` 是同步调用墙钟；设备时间线在计时外单独采集，窗口不累加并行 block。

共享 MLP32 的四层编排将同步计算约 0.199 s 降至 0.136 s，完整起草约 1.501 s 降至 1.434 s。对应 128 项逐层 HF 检查均通过，最大相对 L2 为 0.321%，固定门槛为 2%。Full 准备隔离实验中，K32 在 50 对调用中有 49 次更快，调用墙钟约减少 4.4%；两边的设备窗口都约 721 μs。

NUMA 对照中，仅切换参数主机来源，完整起草约 1.740 s / 1.094 s，计算调用均约 0.130 s，logits 逐字节相同。`transfer-numa-c4.json` 是独立 512 MiB H2D 对照，`numa_maps` 验证实际页面位置；两个实验没有在生产进程中保留第二份 head。

`projection32-mlp.json` 的十二项检查验证 MLP 两类投影及输入不变；`conv-parallel128.json` 虽通过数值检查，速度慢于当前只读卷积，未采用。`k32-release-native-attention.json` 和 `k32-release-fused-attention.json` 使用正式包，检查新 GDN 在另外两种 Attention 后端中的起草、缓存和调度续写。

## 源码与记录

`manifest.json` 保存正式包 24 个模块的 SHA256。`experiment-sources.zip` 按各自不可变快照存放实验脚本及依赖模块，`device-profiles.zip` 保存按原路径映射的时间线。归档不含模型、head 权重或激活；复现时传入匹配的外部模型和 head。

`cpu-k32-release.xml` 对应 387 项 CPU 回归。名称以 `k32-release-` 开头的整包复验记录使用同一份正式模块，SHA256 与提交 `a44b01d` 对应：

- `k32-release-layered/group4.json`：两种编排的候选及 logits 逐字节相同，目标缓存不变，六个历史边界、八组 verifier、接受前缀回放、slot 0→1 及换页均通过。
- `k32-release-hf.json`：64 层、原生和优化两条路径共 128 项真实输入对照通过，最大相对 L2 为 0.321%，门槛保持 2%。CPU 使用 16 线程，只用于参考复验。
- `k32-release-http-r2.json`：completion、chat、SSE、后端配置及正常退出通过；另两种 Attention 后端的服务记录见前述文件。

正式包对照固定分页 Full Attention，仅切换 `native/vector` GDN。逐层同步计算约 0.281/0.160 s，四层同步计算约 0.255/0.131 s，四层完整起草约 1.221/1.097 s；首个完整 draft GDN 层的设备窗口约 4.169/1.210 ms，包含共享 MLP。这组基线与共享 MLP 分块对照不同。

`device-profiles-release.zip` 补充正式包六份重复设备时间线；`release-source.zip` 包含正式模块、验证脚本和 CPU 用例，并保留早期投影候选以复现编译限制。`cann-install.txt` 固定 CANN 9.1.1，`npu-release.txt` 记录验收完成后的资源状态；本次所有 NPU 实验进程已退出。
