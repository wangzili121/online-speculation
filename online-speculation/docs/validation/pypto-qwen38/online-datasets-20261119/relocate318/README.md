# 尚未启动试验的主机迁移准备

SPoC312 的第三次确认仍在 A 机等待空卡。B 机在原三卡辅助实验结束后，先跑两卡旧合成输入诊断；只有诊断通过且确认 A 的原任务从未启动模型，才允许迁移。当前只是准备，未停止 A 等待任务，未在 B 派发预留题。

A / B 的18个权重分片及8个加载文件均完成 SHA256 全文件比较，26个文件完全一致。两机使用同一个实际 Docker digest；PyPTO Python / 二进制、PTOAS 和实际解析到的运行库也逐字节一致。检查容器不挂载 NPU，未影响在运行的模型。模型路径仍可通过公共权重及原软链接读取。

B 的模型包、benchmark、native 源码与冻结的 A source292 完全相同；仅启动脚本把镜像标签锁为相同 digest、结果文件属主改为 B 的登录 UID / GID。数据、初始 head、重置优化器、两卡 TP2 / rank0 和训练参数不变。原 trial ID 保留；若 A 已经开始模型计算，取消迁移，绝不重复运行来挑选更有利的结果。

`stage.json` 和 `queue-spoc318B.json` 保存 B 的准备清单和诊断 / 确认门槛；`model-runtime-equivalence.json`、`extended-runtime-equivalence.json` 保存环境证明。`source292_A_bundle.json.gz` 是冻结计算源码快照，不含权重。`audit_boundary318.py` 只读检查 B 诊断，`watch_spoc320.py` 仅在存在成功交接记录后按主机收集结果，不启动测试。迁移状态和实际主机将独立记录，保留原冻结文件。

本目录的 staging 脚本为一次性操作，目标目录或文件已存在时拒绝重建；不要在已运行的实验上重复执行。环境检查也不代替实机诊断，不表示预留题已获准启动。
