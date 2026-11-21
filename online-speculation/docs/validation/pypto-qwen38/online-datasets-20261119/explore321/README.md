# 相同主机上的更新频率对照

沿用 C 机 source303、step128 head、fresh Adam、两个 A2 / TP2、rank0 学习器及 explore314 的旧 LongProc30 / SPoC64。只把持续每16轮更新改成前64次每16轮、之后每64轮持续更新。KL、3e-5、warmup8、pub4、teacher7、训练行16、输出预算和完整对照保持一致。

两种频率在同一主机依次运行，每组内逐请求交错 AR、固定、在线计时，包含所有更新和发布成本。不是同时执行，仍需结合各组 AR / 固定对照判断运行漂移。无新预留题、无自动确认，不修改312冻结配方。

`queue-explore321.json` 是完整命令参数，`launch.json` 绑定沿用的源码和权重。复用 `../explore314/study_queue_explore314.py` 执行；`watch_explore321.py` 只读审核原始记录及实际发布权重。等待时不持有设备，运行仅使用确认空闲的2 /3卡。
