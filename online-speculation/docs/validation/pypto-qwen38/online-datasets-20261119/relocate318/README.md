# 尚未启动试验的主机迁移准备

SPoC312 的第三次确认仍在 A 机等待空卡。B 机在原三卡辅助实验结束后，先跑两卡旧合成输入诊断；只有诊断通过且确认 A 的原任务从未启动模型，才允许迁移。已于 2026-11-21 17:48:39 UTC 完成交接：原 A 等待程序从未启动模型，安全停止后，B 启动同一个 trial745。前两次确认继续在 A 运行。

A / B 的18个权重分片及8个加载文件均完成 SHA256 全文件比较，26个文件完全一致。两机使用同一个实际 Docker digest；PyPTO Python / 二进制、PTOAS 和实际解析到的运行库也逐字节一致。检查容器不挂载 NPU，未影响在运行的模型。模型路径仍可通过公共权重及原软链接读取。

B 的模型包、benchmark、native 源码与冻结的 A source292 完全相同；仅启动脚本把镜像标签锁为相同 digest、结果文件属主改为 B 的登录 UID / GID。数据、初始 head、重置优化器、两卡 TP2 / rank0 和训练参数不变。原 trial ID 保留；若 A 已经开始模型计算，取消迁移，绝不重复运行来挑选更有利的结果。

`stage.json` 和 `queue-spoc318B.json` 保存 B 的准备清单和诊断 / 确认门槛；`model-runtime-equivalence.json`、`extended-runtime-equivalence.json` 保存环境证明。`source292_A_bundle.json.gz` 是冻结计算源码快照，不含权重。`audit_boundary318.py` 只读检查 B 诊断，`watch_spoc320.py` 仅在存在成功交接记录后按主机收集结果，不启动测试。迁移状态和实际主机将独立记录，保留原冻结文件。

本目录的 staging 脚本为一次性操作，目标目录或文件已存在时拒绝重建；不要在已运行的实验上重复执行。环境检查也不代替实机诊断，不表示预留题已获准启动。

## 启动环境诊断

首次 B 诊断在模型加载前失败：A 启动脚本将项目内 CANN 目录挂到容器固定路径，B 的这个目录为空，保护进程找不到 `libascendcl.so`。原始错误保留在 `failed-preNN318/`。仅停止本任务等待激活的 B 控制器后，`recovery322-stage.json` / `queue-spoc322B.json` 把挂载源改为 B 已安装的 CANN；NN 包、benchmark、模型和训练参数未变。当时 A 的第三次确认仍未移动，诊断通过后才执行文末记录的交接。

CPU 上的模型 / PyPTO / PTOAS 比较不能代替 CANN 实机初始化。单独库清单发现三个差异：`mstx.so`、`libmspti.so`、`libms_tools_ext.so`，其余所列库相同。因此不声称两个 CANN 完整目录逐字节一致，差异保存在 `cann-library-equivalence.json`。恢复后的 B 仍须通过完整的两卡旧输入、更新、replay、greedy 和实际权重检查，再决定是否迁移。

恢复诊断的只读审核命令加 `--stage-file recovery322-stage.json --plan-file queue-spoc322B.json`。`handoff_unstarted322.py` 在诊断通过后仍会原子检查 A 从未启动模型；失败则保留 A 任务，不重复测量。

第二次诊断完成 NPU 初始化和编译后，在原生权重加载器的版本校验处退出：公共权重目录缺少 `.cache/huggingface/download/config.json.metadata`。`failed-preweight322/` 保存完整错误，没有生成或训练结果。原有私有 model-view 保留这份真实下载标识，已在核对配置 / index / tokenizer 哈希后将其原文补回公共目录；不是生成新的版本声明，18个权重文件均未改动。复制由不挂载 NPU 的容器完成，标识 SHA256 记录在 `recovery323-stage.json`。

当前恢复诊断改用 `--stage-file recovery323-stage.json --plan-file queue-spoc323B.json`。模型计算和启动脚本保持第二次诊断的版本，只修复输入目录缺失文件。原 B 控制器在诊断失败后自然结束，当时 A745 仍在等待、B 预留题仍被拦住；最终激活见下节。完整审核额外核对这份版本标识。

## 实机诊断与交接结果

恢复后的真实 27B 两卡诊断通过：两个旧合成输入分别为992 /1000 token、各生成24 token，greedy 各组一致；3次真实更新 /3次发布。首次反馈重放的 TV 为0.00297629、相对L2为0.00266825，Top-1一致率1；实际导出分片与已经发布的两 rank 权重一致。它是功能诊断，不作为性能达标结果。完整记录为 `boundary-audit.json` 与同名 raw /source /log.gz。

交接前再次暂停并核对 A 的等待控制器1820622及 launcher2397650：UID、完整命令、父子关系匹配，原 trial 的模型报告和模型日志均不存在；子进程只允许空卡查询，包括状态Z、已经退出的本任务 npu-smi。第一轮保守检查因这种查询僵尸拒绝交接，原进程未停止，记录在 `handoff-precheck-aborted.json`；修正只读白名单后重新检查，仍未发生模型启动。最终只停止这两个等待 PID，不涉及其他用户或已在计算的模型，见 `A-stopped-unstarted322.json`。

`placement.json` 在激活 B 前持久化，保留原 trial ID、全部数值参数、输入顺序、初始 head 和 fresh Adam。B 控制器3869442已在空闲0 /1卡启动原确认745；A 的743 /744继续原流。只读观察器80840替换原观察器，按实际主机分别核对源码、配置、计时与实际发布分片。没有重跑任何已经开始的预留题试验，也没有根据结果调整配方。

两机并非所有系统软件完全相同：驱动包 A为25.5.1、B为26.1.1（HAL版本字段均为7.35.23），CANN三项库差异另存。模型 /PyPTO /PTOAS及NN计算源码相同；每次确认内部的AR、固定、在线始终使用同一主机和同两张卡。跨节点重复的放置差异会写入最终报告，不把三次说成只改变请求顺序。
