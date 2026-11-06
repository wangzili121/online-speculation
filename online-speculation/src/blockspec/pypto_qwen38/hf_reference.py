"""CPU HF 单层参考，复用已有双视图层；只转换缓存接口。"""

from types import SimpleNamespace

import torch


class ReferenceCache:
    def __init__(self, layer, count, kind, history):
        self.layers = [SimpleNamespace(keys=None, values=None, conv_states=None,
                                      recurrent_states=None, record_past=False) for _ in range(count)]
        row = self.layers[layer]
        if kind == "full_attention":
            row.keys, row.values = [x.clone() for x in history]
        else:
            conv, recurrent = history
            # PyPTO 保存 [tap, channel] / [head, V, K]；HF 需要 B,C,T / B,H,K,V。
            row.conv_states = (conv.T.unsqueeze(0).contiguous().clone(),)
            row.recurrent_states = (recurrent.transpose(-1, -2).unsqueeze(0).contiguous().clone(),)

    def has_previous_state(self, layer, state_idx=0):
        return self.layers[layer].recurrent_states is not None

    def update(self, key, value, layer, *args, **kwargs):
        row = self.layers[layer]
        row.keys, row.values = (torch.cat((row.keys, key), dim=2), torch.cat((row.values, value), dim=2))
        return row.keys, row.values

    def update_conv_state(self, value, layer, conv_kernel_size=4, **kwargs):
        row = self.layers[layer]
        full = torch.cat((row.conv_states[0], value), dim=-1)
        row.conv_states = (full[..., -conv_kernel_size:].clone(),)
        return full

    def update_recurrent_state(self, value, layer, **kwargs):
        self.layers[layer].recurrent_states = (value,)


@torch.inference_mode()
def layer_reference(config, layer, hidden, shared, mixer, history, rotary):
    """只构造一个 decoder layer，不加载第二份整网或复制 embedding。"""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    from ..parallel.qwen38 import _HybridLayer

    kind = config.layer_types[layer]
    with torch.device("meta"):
        module = _HybridLayer(Qwen3_5DecoderLayer(config, layer), kind)
    module.attention.ar = None
    state = {"input_norm.weight": shared["input_layernorm.weight"],
             "post_norm.weight": shared["post_attention_layernorm.weight"]}
    state.update({name: value for name, value in shared.items() if name.startswith("mlp.")})
    state.update({"attention.draft." + name: value for name, value in mixer.items()})
    module.load_state_dict(state, strict=True, assign=True)
    module.eval().requires_grad_(False)
    cache = ReferenceCache(layer, config.num_hidden_layers, kind, history)
    # HF RotaryEmbedding 按 hidden dtype 返回 cos/sin；native 的表保留 FP32。
    rotary = tuple(x.to(hidden.dtype).unsqueeze(0) for x in rotary)
    result = module(hidden.unsqueeze(0), "draft", rotary, None, cache)
    return result[0]
