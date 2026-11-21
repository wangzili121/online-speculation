# 旧题上的更新频率探索

本组只用已有 LongProc 30 题和 SPoC 64 题探索集，不新增预留题或修改已冻结的310 /312确认配方。检验：较长代码生成流中，在前64次更新之后仍每16个验证轮次更新，能否比降为每64轮更有效地收回训练成本。学习率、warmup8、last-1 / KL / 3e-5、pub4、teacher7、初始head128保持不变，所有成本仍计入。

C机使用source303的完整计算包及benchmark原文。145个包文件和benchmark逐字节匹配已提交HEAD，未混入本地其他未提交改动；C机native模型源码也匹配。只在既有host launcher中透传上下文、训练行数与输出预算参数，没有修改NN计算或开发新kernel。该主机驱动与A机不同，必须先过自己的实机门槛，结果不与A机探索直接合并。

顺序执行：

1. 两卡1024上下文边界诊断：输入992 /1000，输出预算32 /24，检查AR、固定、在线、最终head输出、首次反馈replay及权重导出。
2. 原LongProc30题：context1024、nominal512按输入长度裁剪、自然EOS，更新间隔持续16轮；完整AR /固定 /在线 /最终发布head已见重跑 /旧开发留出。
3. 原SPoC64题：context1024、输出192、自然EOS，同样持续16轮及完整对照。

后两组只有第一组greedy和replay通过才运行。均为两个物理A2、TP2、学习器共用rank0；只在无其他进程的空闲卡上启动，等待时不保留设备。不会自动派发新题确认，也不会把旧题探索写成新题达标。

`source303_bundle.json.gz`保存原始计算实现，不含权重。`runtime-vs-HEAD-audit.json`、`asset-provenance.json`和source /base /head绑定保留源码与数据依据。`stage_explore314.py`为一次性准备入口，已有目录与清单时拒绝重建；`watch_explore314.py --artifacts ... --validation-root ...`只读核验。
