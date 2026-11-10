# PyPTO 多层后段反向验证

环境沿用 [last-1 记录](../online-last1-20261110/README.md) 中固定的 PyPTO、Simpler、CANN 和 A2 运行时。训练继续使用因果 GDN、块内双向 Full Attention；没有改变混合模型策略。

## 已验证

- `gdn-lastn-core-{small-r8,real-r9}.json`：递推前向、保存状态和解析反向。真实规格为 48 个 value heads、16 个 Q/K heads、head dimension 128；另有较小规格。每种规格覆盖有效宽度 1/17/32、零与非零历史，未使用行填入 NaN，检查输入逐字节不变。
- `gdn-lastn-aux-{small-r11,real-r11}.json`：四 tap 因果卷积及 SiLU、decay/beta 门控、gated RMSNorm 的前向和反向，对照独立 Torch autograd；覆盖宽度 1/17/32 及只读历史。
- `suffix-last2-r11.json`：小尺寸完整 GDN + Full Attention 后段，包含冻结 MLP/norm/LM head 上所需的输入梯度。两种历史/块长配置下，全部 15 组 draft 参数与独立 autograd 对照通过，输入逐字节不变。
- CPU 回归 465 项通过，包含 last-2/4 整段解析梯度与 autograd 对照。新增的运行时接入另行验证。

## 入口

```bash
python scripts/validate_qwen38_pypto_gdn_training.py --device 0 --real-shape --output core.json
python scripts/validate_qwen38_pypto_gdn_training_aux.py --device 0 --real-shape --output aux.json
python scripts/validate_qwen38_pypto_suffix_training.py --device 0 --last-layers 2 --output suffix.json
```

组件通过不能替代真实 27B 多层在线更新验收。last-4 的设备数值对照、优化器、整组发布、checkpoint 和真实服务闭环仍在继续验证；原 last-1 的验收记录不变。
