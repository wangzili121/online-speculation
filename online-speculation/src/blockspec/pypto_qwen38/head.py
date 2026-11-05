"""Read native vLLM head shards without changing their model or TP semantics."""

import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
from safetensors import safe_open

from .baseline import read_manifest


HIDDEN, KEY_WIDTH, VALUE_WIDTH = 5120, 2048, 6144
POLICY = "orthrus_qwen35_reference"


def tensor_schema(layer, tp=1):
    if layer % 4 == 3:
        return {"q_proj.weight": (12288 // tp, HIDDEN),
                "k_proj.weight": (1024 // tp, HIDDEN), "v_proj.weight": (1024 // tp, HIDDEN),
                "o_proj.weight": (HIDDEN, 6144 // tp),
                "q_norm.weight": (256,), "k_norm.weight": (256,)}
    return {"in_proj_qkv.weight": (10240 // tp, HIDDEN),
            "in_proj_z.weight": (6144 // tp, HIDDEN),
            "in_proj_a.weight": (48 // tp, HIDDEN), "in_proj_b.weight": (48 // tp, HIDDEN),
            "out_proj.weight": (HIDDEN, 6144 // tp), "conv1d.weight": (10240 // tp, 1, 4),
            "A_log": (48 // tp,), "dt_bias": (48 // tp,), "norm.weight": (128,)}


def merge_parameter(layer, name, shards):
    """Q/K/V-packed rows need section-wise gathering, not plain rank concatenation."""
    tp = len(shards)
    expected = tensor_schema(layer, tp)[name]
    dtype = shards[0].dtype
    permitted = {torch.bfloat16, torch.float32} if name == "A_log" else {torch.bfloat16}
    for value in shards:
        if (value.shape != expected or value.dtype not in permitted or value.dtype != dtype
                or not torch.isfinite(value).all()):
            raise ValueError(f"invalid draft parameter {layer}.{name}: expected finite BF16/FP32 {expected}")
    if name in {"q_norm.weight", "k_norm.weight", "norm.weight"}:
        if any(not torch.equal(shards[0], other) for other in shards[1:]):
            raise ValueError(f"replicated draft parameter differs across ranks: {layer}.{name}")
        return shards[0].contiguous()
    if name in {"in_proj_qkv.weight", "conv1d.weight"}:
        parts = [value.split((KEY_WIDTH // tp, KEY_WIDTH // tp, VALUE_WIDTH // tp), dim=0)
                 for value in shards]
        return torch.cat([torch.cat([row[index] for row in parts], dim=0) for index in range(3)], dim=0)
    axis = 1 if name in {"o_proj.weight", "out_proj.weight"} else 0
    return torch.cat(shards, dim=axis).contiguous()


class HeadSource:
    """Validate head shards and prepare an immutable runtime layout once."""

    def __init__(self, model_root, head_dir=None, *, block_size=32, mask_token_id=1):
        self.model_root = Path(model_root)
        self.directory = None if head_dir is None else Path(head_dir)
        self._prepared = None
        self.shared_weights = None
        self.preparation_seconds = 0.0
        self.block_size, self.mask_token_id = block_size, mask_token_id
        config = json.loads((self.model_root / "config.json").read_text())["text_config"]
        expected = dict(hidden_size=HIDDEN, num_hidden_layers=64, num_attention_heads=24,
                        num_key_value_heads=4, head_dim=256, linear_num_key_heads=16,
                        linear_num_value_heads=48, linear_key_head_dim=128, linear_value_head_dim=128)
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError("the pinned dense Qwen3.8-27B architecture is required")
        if config["layer_types"] != (["linear_attention"] * 3 + ["full_attention"]) * 16:
            raise ValueError("incompatible hybrid layer order")
        if type(block_size) is not int or not 2 <= block_size <= 32:
            raise ValueError("block_size must be in 2..32")
        if type(mask_token_id) is not int or not 0 <= mask_token_id < config["vocab_size"]:
            raise ValueError("explicit valid mask token required")
        self.index = json.loads((self.model_root / "model.safetensors.index.json").read_text())["weight_map"]
        self.tp = 1
        if self.directory:
            manifest = read_manifest(self.directory)
            self.tp = manifest.get("tp_size")
            if type(self.tp) is not int or self.tp not in {1, 2, 4}:
                raise ValueError("head must use native TP1/TP2/TP4 shards")
            hashes = [hashlib.sha256((self.model_root / name).read_bytes()).hexdigest()
                      for name in ("config.json", "model.safetensors.index.json")]
            expected_manifest = dict(format="orthrus-qwen38-native-tp-v1", base_config_sha256=hashes[0],
                                     base_index_sha256=hashes[1], tp_size=self.tp, block_size=block_size,
                                     mask_token_id=mask_token_id, hybrid_policy=POLICY)
            if manifest != expected_manifest:
                raise ValueError("head must match the base checkpoint, mask, block and hybrid policy")
            names = {f"layers.{layer}.attention.draft.{name}": shape
                     for layer in range(64) for name, shape in tensor_schema(layer, self.tp).items()}
            for rank in range(self.tp):
                with safe_open(self.directory / f"rank{rank}.safetensors", framework="pt", device="cpu") as shard:
                    if set(shard.keys()) != names.keys():
                        raise ValueError(f"head shard {rank} has missing or extra draft tensors")
                    if any(tuple(shard.get_slice(key).get_shape()) != shape for key, shape in names.items()):
                        raise ValueError(f"head shard {rank} has incompatible tensor shapes")

    def layer(self, layer):
        if type(layer) is not int or not 0 <= layer < 64:
            raise ValueError("layer must be in 0..63")
        result = {}
        for name in tensor_schema(layer):
            if self.directory:
                shards = []
                for rank in range(self.tp):
                    with safe_open(self.directory / f"rank{rank}.safetensors", framework="pt", device="cpu") as file:
                        shards.append(file.get_tensor(f"layers.{layer}.attention.draft.{name}"))
                result[name] = merge_parameter(layer, name, shards)
            else:
                kind = "self_attn" if layer % 4 == 3 else "linear_attn"
                key = f"model.language_model.layers.{layer}.{kind}.{name}"
                filename = Path(self.index[key])
                if filename.is_absolute() or len(filename.parts) != 1:
                    raise ValueError("checkpoint shard names must be local basenames")
                with safe_open(self.model_root / filename, framework="pt", device="cpu") as file:
                    result[name] = merge_parameter(layer, name, [file.get_tensor(key)])
        return result

    def banks(self, layer):
        if self._prepared is not None:
            if type(layer) is not int or not 0 <= layer < 64:
                raise ValueError("layer must be in 0..63")
            return list(self._prepared[layer])
        weights = self.layer(layer)
        if layer % 4 == 3:
            return [weights[name] for name in ("q_proj.weight", "k_proj.weight", "v_proj.weight")] + [
                (1 + weights[name].float()).reshape(1, 256) for name in ("q_norm.weight", "k_norm.weight")] + [
                weights["o_proj.weight"]]
        padded = []
        for name in ("in_proj_a.weight", "in_proj_b.weight"):
            padded.append(torch.nn.functional.pad(weights[name], (0, 0, 0, 80)))
        constants = []
        for name in ("A_log", "dt_bias"):
            value = torch.zeros((1, 64), dtype=weights[name].dtype)
            value[0, :48] = weights[name]
            constants.append(value)
        return [weights["in_proj_qkv.weight"], weights["in_proj_z.weight"], *padded,
                weights["conv1d.weight"].squeeze(1).T.contiguous(), *constants,
                weights["norm.weight"].reshape(1, 128), weights["out_proj.weight"]]

    def prepare(self):
        """一次转换并校验，再放入同一块共享内存，供 worker 直接读取。"""
        if self._prepared is not None:
            return
        start = time.monotonic()
        layers = [self.banks(layer) for layer in range(64)]
        for layer, values in enumerate(layers):
            if layer % 4 != 3:
                values[5] = values[5].float()
        size, layouts = 0, []
        for values in layers:
            current = []
            for value in values:
                size = (size + 63) // 64 * 64
                count = value.numel() * value.element_size()
                current.append((size, count))
                size += count
            layouts.append(current)
        if Path("/dev/shm").is_dir() and shutil.disk_usage("/dev/shm").free < size + (64 << 20):
            raise RuntimeError(f"head preparation requires at least {size + (64 << 20)} free bytes in /dev/shm")
        storage = torch.empty(size, dtype=torch.uint8).share_memory_()
        prepared = []
        for values, layout in zip(layers, layouts, strict=True):
            current = []
            for value, (offset, count) in zip(values, layout, strict=True):
                target = storage[offset:offset + count].view(value.dtype).reshape(value.shape)
                target.copy_(value)
                current.append(target)
            prepared.append(tuple(current))
        self.shared_weights, self._prepared = storage, tuple(prepared)
        self.preparation_seconds = time.monotonic() - start
