"""原生主模型 bank 的设备加载；大矩阵按层 DMA 到最终位置。"""

from dataclasses import dataclass

import torch

from .loading import DeviceLoader, LoadingPrograms


@dataclass
class BaseBank:
    name: str
    tail: str
    layers: tuple
    shape: tuple
    dtype: object
    operation: str = "copy"
    source_shape: tuple | None = None


def base_plans():
    all_layers, gdn, full = tuple(range(64)), tuple(i for i in range(64) if i % 4 != 3), tuple(range(3, 64, 4))
    bf, fp = torch.bfloat16, torch.float32
    plans = [
        BaseBank("embedding", "model.language_model.embed_tokens.weight", (), (248320, 5120), bf),
        BaseBank("final_gamma", "model.language_model.norm.weight", (), (1, 5120), fp, "gamma", (1, 5120)),
        BaseBank("lm_head", "lm_head.weight", (), (248320, 5120), bf),
        BaseBank("input_gamma", "input_layernorm.weight", all_layers, (64, 5120), fp, "gamma", (1, 5120)),
        BaseBank("post_gamma", "post_attention_layernorm.weight", all_layers, (64, 5120), fp, "gamma", (1, 5120)),
    ]
    for name, tail, rows, cols in (("mlp_gate", "gate_proj", 17408, 5120), ("mlp_up", "up_proj", 17408, 5120), ("mlp_down", "down_proj", 5120, 17408)):
        plans.append(BaseBank(name, f"mlp.{tail}.weight", all_layers, (64 * rows, cols), bf))
    for name, tail, shape, op, source_shape in (
        ("delta_qkv", "in_proj_qkv.weight", (48 * 10240, 5120), "copy", None),
        ("delta_z", "in_proj_z.weight", (48 * 6144, 5120), "copy", None),
        ("delta_a", "in_proj_a.weight", (48 * 128, 5120), "pad", (48, 5120)),
        ("delta_b", "in_proj_b.weight", (48 * 128, 5120), "pad", (48, 5120)),
        ("delta_conv", "conv1d.weight", (48 * 4, 10240), "conv", (10240, 4)),
        ("delta_a_log", "A_log", (48, 64), "constant", (1, 48)),
        ("delta_dt_bias", "dt_bias", (48, 64), "constant", (1, 48)),
        ("delta_norm", "norm.weight", (48, 128), "copy", None),
        ("delta_out", "out_proj.weight", (48 * 5120, 6144), "copy", None),
    ):
        plans.append(BaseBank(name, f"linear_attn.{tail}", gdn, shape, bf, op, source_shape))
    for name, tail, shape, op, source_shape in (
        ("full_q", "q_proj.weight", (16 * 12288, 5120), "copy", None),
        ("full_k", "k_proj.weight", (16 * 1024, 5120), "copy", None),
        ("full_v", "v_proj.weight", (16 * 1024, 5120), "copy", None),
        ("full_q_gamma", "q_norm.weight", (16, 256), "gamma", (1, 256)),
        ("full_k_gamma", "k_norm.weight", (16, 256), "gamma", (1, 256)),
        ("full_out", "o_proj.weight", (16 * 5120, 6144), "copy", None),
    ):
        plans.append(BaseBank(name, f"self_attn.{tail}", full, shape, fp if op == "gamma" else bf, op, source_shape))
    return plans


class BaseLoading:
    def __init__(self, runtime, model_root, config):
        self.weights = runtime.TextWeights(model_root)
        self.plans, self.keys, self.registry = base_plans(), {}, LoadingPrograms(config)
        samples = runtime.single_meta_samples()[1:24]
        if [(p.shape, p.dtype) for p in self.plans] != [(tuple(p.shape), p.dtype) for p in samples]:
            raise ValueError("native target bank ABI changed")
        for plan in self.plans:
            if plan.operation in {"pad", "constant"}:
                self.registry.zero(plan.shape, plan.dtype)
            if plan.operation in {"gamma", "conv"}:
                self.keys[plan.name] = self.registry.copy(plan.source_shape, plan.shape, torch.bfloat16, plan.dtype,
                                                         conv=plan.operation == "conv", gamma=plan.operation == "gamma")

    def banks(self, worker):
        loader = DeviceLoader(self.registry, worker)
        try:
            for plan in self.plans:
                target = loader.allocate(plan.shape, plan.dtype)
                if plan.operation in {"pad", "constant"}:
                    loader.zero(target)
                for index, layer in enumerate(plan.layers or (None,)):
                    value = self.weights.read(plan.tail) if layer is None else self.weights.layer(layer, plan.tail)
                    if not value.is_contiguous():
                        raise ValueError("checkpoint storage must be contiguous")
                    row = index * (plan.shape[0] // max(1, len(plan.layers)))
                    if plan.operation in {"gamma", "conv"}:
                        if value.numel() != plan.source_shape[0] * plan.source_shape[1]:
                            raise ValueError(f"checkpoint tensor shape changed: {plan.name}")
                        source = loader.allocate(plan.source_shape, torch.bfloat16, init=value.reshape(plan.source_shape))
                        try:
                            loader.copy(self.keys[plan.name], source, target, target_offset=(row, 0))
                        except ValueError as error:
                            raise ValueError(f"target bank {plan.name}, layer {layer}: {error}") from error
                        finally:
                            loader.free(source)
                    else:
                        expected = plan.source_shape or (plan.shape[0] // max(1, len(plan.layers)), plan.shape[1])
                        if value.numel() != expected[0] * expected[1]:
                            raise ValueError(f"checkpoint tensor shape changed: {plan.name}")
                        worker.copy_to(target.data_ptr, value.data_ptr(), value.numel() * value.element_size(),
                                       dst_offset=row * plan.shape[1] * value.element_size())
                print("loaded target bank on device", plan.name, flush=True)
                yield loader.detach(target)
        finally:
            loader.close()
