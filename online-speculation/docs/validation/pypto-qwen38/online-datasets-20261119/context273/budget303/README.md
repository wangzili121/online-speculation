# 输出预算与上下文边界

CPU 全量回归 592 项通过。`budget-v303-cpu` 保留首次准备阶段状态：测试通过，随后数据定位遗漏 `pool283` 而失败；`budget-v304-preflight` 只修正路径并完成真实 tokenizer 的 256 预算检查，`budget-v305-preflight512` 检查 512 预算。均没有 NPU 模型执行，不据此宣称推理已验证。

A 机准备旧题边界输入时，最初沿用了仅 B/C 存在的 `model-view` 路径，在模型启动前停止。改为 A 公共目录后，已验证预先写入的代码逐字节相同，模型 kernel 与 source292 完全一致；错误保留在 `budget303-device-launch-output.txt`。

`device-launch.json` 登记两条 992 / 1000 token 的旧合成输入，三组均申请 512、实际分配 32 / 24，忽略 EOS 以检验总长度 1024。带教师重放、连续发布及导出检查；只有 `budget303-device-validation.json` 成功后才放行新的完整探索。该诊断没有新预留题，也不用于速度结论。

`queue-length307-longproc.json` 使用已完成三种旧配方中双基线较小增益最大的 head128 / KL 配方，保持训练及模型配置，只将三组共用输出预算改为最多 512、按剩余上下文截限，并使用自然 EOS。保留全部 30 道探索题；它是新的完整流测量，不与旧 128-token 流混作同条件加速。预留题没有派发。

边界诊断已完成并正常关闭：3 次实际更新和发布，AR / 固定 / 在线 / 最终 head 重跑一致，完整输入加输出长度均为 1,024，实际发布与导出权重一致；`budget303-device-validation.json` 校验模型及 native kernel 沿用 source292。更长输出的完整探索因此获准启动，尚无性能结论。
