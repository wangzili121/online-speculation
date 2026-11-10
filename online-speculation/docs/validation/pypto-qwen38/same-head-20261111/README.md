# 同一份 last-1 head 的真实推理验证

[结果与方法](../../../QWEN38_PYPTO_SAME_HEAD_EVALUATION.md)；[精简指标](summary.json)；[完整原始结果](fixed128-real-r1.json)；[实机命令与退出码](fixed128-real-r1.supervisor.json)。

入口是 `scripts/evaluate_qwen38_pypto_fixed_head.py`，适配代码版本 `faa7cb2`。使用真实 Qwen3.8-27B、两张 A2、相同主模型与冻结 draft 前缀、K8、greedy。初始及最终 master 来自同一次 128 步固定反馈训练；脚本不执行任何更新，源 master 文件 SHA256 前后不变。

逐值核对六组执行参数。按原采集顺序重放 8 个 anchor，训练输入、teacher 和推理/采集 logits 的差值全部为零。6 个回答 anchor 的真实接受数为 `7/42 → 42/42`；另 2 个 prompt anchor 均只接受 1 个候选，两组不混合解释。

两个 head 各完成 2 条训练题、2 条 dev、8 条 test，输出上限 32 token，全部与配对 AR 逐 token 一致。训练题自由续写只覆盖前 32 个输出位置，不覆盖原 48-token 回答中的全部训练 anchor；精确 anchor 验证则覆盖全部 6 个回答块。

8 条 test 的实际接受率为 `16.69% → 18.48%`，平均接受候选数 `1.05 → 1.17`；2 条 dev 为 `18.08% → 17.88%`。这些是小规模诊断数据，不用于宣称稳定收益。候选接受率为实际接受数除以实际提出数；接受数不含修正 token。

吞吐为单次诊断计时，包含正常起草、验证和拒绝恢复，不包含训练或加载。没有强制接受或候选替换。

总计 36 个服务请求正常完成，错误计数、队列与 KV 占用最终归零。监督脚本正常退出，`returncode=0`。大权重及输入 tensor 留在服务器，不提交 Git。[资源检查](final-resource-check.json)确认本轮 NPU 进程已退出，其他任务保持运行。
