# Orthrus-DSV4 / Ascend

分支：`feature/orthrus-dsv4`。本分支维护 `Eco-Tech/DeepSeek-V4-Flash-0731-w8a8` 的 Orthrus 推理适配；训练和在线更新暂不推进。

**算子需求报告：[Orthrus-DSV4 NPU 推理适配与算子设计](ORTHRUS_DSV4_PYPTO_REQUIREMENTS.md)**

- 模型：主模型保持原生 vLLM-Ascend 路径；draft 共享冻结 backbone，读取 SWA、C4、C128 历史，生成临时双向块。
- 接口：提供专属 proposer、head 契约、缓存适配及部署脚本；部署方式为插件加固定版本兼容适配，并非任意版本的纯配置插件。
- 验证：已完成局部算子、缓存与真实模型启动检查；后续在完整八卡环境继续验证端到端推理。
- 资源：完整真实模型端到端验收按 8 卡 A2 准备；初始化 head 只证明接入，不代表已有训练效果。

部署入口：进入 `online-speculation`，按 [部署说明](online-speculation/docs/DSV4_DEPLOYMENT.md) 固定镜像、模型路径与 head；保留 target-only 启动脚本作为基线和回退。

接口契约见 [head v2](online-speculation/docs/DSV4_HEAD_CONTRACT.md)，当前验收队列见 [验证状态](online-speculation/docs/VALIDATION_QUEUE.md)。

历史中保留其他模型和公共代码；后续 DSV4 改动仅在此分支迭代。
