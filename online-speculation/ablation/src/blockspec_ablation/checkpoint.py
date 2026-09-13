"""Own-format checkpoints and a narrow, explicit safetensors base-weight bridge."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from .model import Decoder, ModelConfig, is_adapter


FORMAT = "blockspec-v1"


def implementation_fingerprint():
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _weight_fingerprint(model, *, adapters):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if is_adapter(name) != adapters:
            continue
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}\n".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def base_fingerprint(model):
    return _weight_fingerprint(model, adapters=False)


def adapter_fingerprint(model):
    """Fingerprint the actual execution-dtype adapter tensors, including names."""
    return _weight_fingerprint(model, adapters=True)


def adapter_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.state_dict().items() if is_adapter(n)}


def _validate_state(expected, actual):
    if set(expected) != set(actual):
        raise ValueError(f"checkpoint keys differ: missing={set(expected) - set(actual)}, "
                         f"unexpected={set(actual) - set(expected)}")
    for name, value in actual.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape:
            raise ValueError(f"checkpoint shape mismatch: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite checkpoint tensor: {name}")


def save_checkpoint(path, model, *, adapter_only=False, metadata=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = adapter_state(model) if adapter_only else {
        n: p.detach().cpu() for n, p in model.state_dict().items()
    }
    payload = {"format": FORMAT, "config": model.config.to_dict(), "adapter_only": adapter_only,
               "base_fingerprint": base_fingerprint(model), "state": state,
               "metadata": metadata or {}}
    # Exclusive creation also protects against another process racing the check.
    with path.open("xb") as handle:
        torch.save(payload, handle)


def load_checkpoint(path, *, model=None, device="cpu", dtype=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError("unknown checkpoint format")
    config = ModelConfig.from_dict(payload["config"])
    if payload["adapter_only"]:
        if model is None or model.config != config:
            raise ValueError("adapter checkpoint needs an identically configured base model")
        if payload["base_fingerprint"] != base_fingerprint(model):
            raise ValueError("adapter belongs to different base weights or base dtype")
        expected = {n: p for n, p in model.state_dict().items() if is_adapter(n)}
        _validate_state(expected, payload["state"])
        # Do not round saved FP32 master adapters through the base's BF16 dtype.
        for name, parameter in model.named_parameters():
            if is_adapter(name):
                parameter.data = parameter.data.to(payload["state"][name].dtype)
        model.load_state_dict(payload["state"], strict=False)
    else:
        if model is not None:
            raise ValueError("full checkpoint constructs its own model")
        model = Decoder(config)
        source_dtype = payload["state"]["model.embed_tokens.weight"].dtype
        model.to(dtype=source_dtype)
        # Preserve FP32 low-rank master weights in an otherwise BF16 model.
        for name, parameter in model.named_parameters():
            parameter.data = parameter.data.to(payload["state"][name].dtype)
        _validate_state(model.state_dict(), payload["state"])
        model.load_state_dict(payload["state"], strict=True)
        if base_fingerprint(model) != payload["base_fingerprint"]:
            raise ValueError("full checkpoint base fingerprint mismatch")
    if dtype is not None:
        model.to(dtype=dtype)
    return model.to(device), payload["metadata"]


def config_from_hf(raw, *, rank=8, alpha=None):
    """Support explicit dense rotary/GQA families; reject unimplemented features.

    Architecture identifiers select validated tensor layouts and normalization
    rules. Weight locations and dimensions come from the caller's checkpoint.
    """
    if raw.get("model_type") not in ("k2_horizon", "qwen3"):
        raise ValueError("unsupported architecture; implement and validate a bridge first")
    if (raw.get("num_experts", 0) or raw.get("quantization_config")
            or raw.get("use_sliding_window", False) or raw.get("sliding_window")
            or raw.get("mova_num_experts", 0)):
        raise ValueError("MoE, quantization and sliding-window attention are not implemented")
    if raw.get("hidden_act", "silu") != "silu" or raw.get("partial_rotary_factor", 1) != 1:
        raise ValueError("only full-head rotary and SwiGLU are implemented")
    rope = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    if rope_type not in ("default", "yarn"):
        raise ValueError(f"unsupported rotary scheme: {rope_type}")
    factor = rope.get("factor", 1.0) if rope_type == "yarn" else 1.0
    hidden, heads = raw["hidden_size"], raw["num_attention_heads"]
    head_dim = raw.get("head_dim", hidden // heads)
    if (raw.get("attention_gate_func") or raw.get("mlp_bias", False)
            or raw.get("rope_head_dim", head_dim) not in (None, head_dim)):
        raise ValueError("attention gating, MLP bias and partial rotary need a separate bridge")
    if raw["model_type"] == "k2_horizon" and raw.get("query_key_norm", False):
        raise ValueError("head-specific K2 query/key normalization is not implemented")
    import math
    return ModelConfig(
        vocab_size=raw["vocab_size"], hidden_size=hidden,
        intermediate_size=raw["intermediate_size"], num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=heads, num_key_value_heads=raw.get("num_key_value_heads", heads),
        head_dim=head_dim, rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
        norm_groups=raw.get("layernorm_num_groups", raw.get("norm_groups", 1)),
        norm_weight_in_fp32=raw["model_type"] == "k2_horizon",
        query_key_norm=raw.get("query_key_norm", raw["model_type"] == "qwen3"),
        attention_bias=raw.get("attention_bias", False),
        attention_output_bias=raw.get("attention_bias", False),
        tie_word_embeddings=raw.get("tie_word_embeddings", False),
        rope_theta=rope.get("rope_theta", raw.get("rope_theta", 10000.0)),
        rope_factor=factor,
        rope_original_length=rope.get("original_max_position_embeddings",
                                      raw.get("original_max_position_embeddings", 8192)),
        rope_beta_fast=rope.get("beta_fast", 32), rope_beta_slow=rope.get("beta_slow", 1),
        rope_attention_factor=rope.get("attention_factor", 1 + 0.1 * math.log(factor)),
        rope_truncate=rope.get("truncate", True), adapter_rank=rank,
        adapter_alpha=float(rank or 1) if alpha is None else alpha,
    )


def load_hf_base(directory, *, rank=8, alpha=None, device="cpu", dtype=torch.float32):
    from safetensors.torch import load_file

    directory = Path(directory)
    raw = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    config = config_from_hf(raw, rank=rank, alpha=alpha)
    model = Decoder(config).to(dtype=dtype)
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        names = sorted(set(index.values()))
        if any(Path(n).name != n for n in names):
            raise ValueError("weight shards must be direct children of the model directory")
        files = [directory / n for n in names]
    else:
        files = sorted(directory.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError("no base model safetensors files found")
    state = {}
    for file in files:
        shard = load_file(str(file))
        if set(state).intersection(shard):
            raise ValueError("duplicate weights across shards")
        state.update(shard)
    if config.tie_word_embeddings and "lm_head.weight" not in state:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    expected = {n: p for n, p in model.state_dict().items() if not is_adapter(n)}
    _validate_state(expected, state)
    model.load_state_dict(state, strict=False)
    return model.to(device)


def with_adapter_rank(config, rank):
    return replace(config, adapter_rank=rank, adapter_alpha=float(rank or 1))
