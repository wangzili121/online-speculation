"""Strict tensor-name bridge and resumable own-format dual-view checkpoints."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from .backbone import DualViewConfig, DualViewDecoder


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(source):
    """Checkpoint binding by content, independent of its storage location."""
    if not isinstance(source, dict):
        raise ValueError("checkpoint source metadata must be a mapping")
    return {key: value for key, value in source.items() if key in ("weight_sha256", "config_sha256")}


def public_key_map(config, *, include_draft=True):
    """Own key -> public key. Both the output projection and Q/K norms are routed."""
    mapping = {"embedding.weight": "model.embed_tokens.weight", "norm.weight": "model.norm.weight"}
    if not config.tie_word_embeddings:
        mapping["head.weight"] = "lm_head.weight"
    for index in range(config.num_hidden_layers):
        own, source = f"layers.{index}.", f"model.layers.{index}."
        for name, original in (("input_norm", "input_layernorm"),
                               ("post_norm", "post_attention_layernorm")):
            mapping[own + name + ".weight"] = source + original + ".weight"
        for name in ("gate", "up", "down"):
            mapping[own + name + ".weight"] = source + "mlp." + name + "_proj.weight"
        views = (("ar", ""), ("draft", "_diff")) if include_draft else (("ar", ""),)
        for view, suffix in views:
            for name in ("q", "k", "v", "o"):
                for kind in (("weight", "bias") if config.attention_bias else ("weight",)):
                    mapping[own + f"attention.{view}.{name}.{kind}"] = (
                        source + f"self_attn.{name}_proj{suffix}.{kind}")
            for name in ("q_norm", "k_norm"):
                mapping[own + f"attention.{view}.{name}.weight"] = (
                    source + f"self_attn.{name}{suffix}.weight")
    return mapping


def load_ar_base(path, *, block_size, mask_token_id, device="cpu"):
    """Initialize FP32 draft projections from an ordinary local Qwen3 checkpoint.

    Single-file and indexed safetensors share the same strict name/shape bridge.
    Each draft tensor owns separate storage; the shared/AR tensors remain frozen.
    """
    path = Path(path).resolve()
    data = json.loads((path / "config.json").read_text())
    if data.get("model_type") != "qwen3":
        raise ValueError("a plain Qwen3 AR checkpoint is required")
    config = DualViewConfig.from_public(data | {"block_size": block_size, "mask_token_id": mask_token_id})
    mapping = public_key_map(config, include_draft=False)
    index_path = path / "model.safetensors.index.json"
    index = None
    if index_path.exists():
        index = json.loads(index_path.read_text()).get("weight_map")
        if not isinstance(index, dict) or not index or any(
                not isinstance(k, str) or not isinstance(v, str) for k, v in index.items()):
            raise ValueError("nonempty safetensors weight_map required")
        filenames = sorted(set(index.values()))
    else:
        filenames = ["model.safetensors"]
    with torch.device("meta"):
        model = DualViewDecoder(config)
    shapes = model.state_dict()
    reverse = {source: own for own, source in mapping.items()}
    state, seen, fingerprints = {}, {}, {}
    extra_head = None
    for filename in filenames:
        file = (path / filename).resolve()
        if file.parent != path or file.suffix != ".safetensors":
            raise ValueError("checkpoint shards must be local safetensors files")
        fingerprints[filename] = file_sha256(file)
        with safe_open(file, framework="pt", device="cpu") as handle:
            for source in handle.keys():
                if source in seen or (index is not None and index.get(source) != filename):
                    raise ValueError("duplicate tensor or inconsistent shard index")
                seen[source] = filename
                tensor = handle.get_tensor(source)
                if config.tie_word_embeddings and source == "lm_head.weight":
                    extra_head = tensor.clone()
                    continue
                if source not in reverse:
                    raise ValueError(f"unexpected AR tensor: {source}")
                own = reverse[source]
                if tensor.shape != shapes[own].shape or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                    raise ValueError(f"invalid AR tensor: {source}")
                state[own] = tensor.to(device=device, dtype=torch.float32)
    if (set(state) != set(mapping) or (index is not None and set(index) != set(seen))):
        raise ValueError("missing AR tensors or inconsistent shard index")
    if extra_head is not None and not torch.equal(extra_head.float().to(device), state["embedding.weight"]):
        raise ValueError("tied AR output head differs from embeddings")
    for own in public_key_map(config):
        if ".attention.draft." in own:
            state[own] = state[own.replace(".attention.draft.", ".attention.ar.")].clone()
    if config.tie_word_embeddings:
        state["head.weight"] = state["embedding.weight"]
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    model.frequencies = model._frequencies(model.embedding.weight.device)
    model.source = {"kind": "ar-initialization",
                    "config_sha256": file_sha256(path / "config.json"), "weight_sha256": fingerprints}
    return model.eval().requires_grad_(False)


def load_public(path, *, device="cpu", dtype=torch.float32, expected_sha256=None):
    path = Path(path)
    config = DualViewConfig.from_public(json.loads((path / "config.json").read_text()))
    weights_path = path / "model.safetensors"
    fingerprint = file_sha256(weights_path)
    if expected_sha256 is not None and fingerprint != expected_sha256:
        raise ValueError("public weight SHA256 mismatch")
    with torch.device("meta"):
        model = DualViewDecoder(config)
    expected = model.state_dict()
    mapping = public_key_map(config)
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        actual_keys = set(handle.keys())
        extra_tied_head = config.tie_word_embeddings and "lm_head.weight" in actual_keys
        required = set(mapping.values())
        if extra_tied_head:
            required.add("lm_head.weight")
        if actual_keys != required:
            raise ValueError(f"public tensor keys differ: missing={required - actual_keys}, "
                             f"unexpected={actual_keys - required}")
        if extra_tied_head and not torch.equal(handle.get_tensor("lm_head.weight"),
                                              handle.get_tensor("model.embed_tokens.weight")):
            raise ValueError("tied output head differs from token embeddings")
        state = {}
        for own, original in mapping.items():
            tensor = handle.get_tensor(original)
            if tensor.shape != expected[own].shape:
                raise ValueError(f"shape mismatch for {original}: {tensor.shape} != {expected[own].shape}")
            state[own] = tensor.to(device=device, dtype=dtype)
    if config.tie_word_embeddings:
        state["head.weight"] = state["embedding.weight"]
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    model.frequencies = model._frequencies(model.embedding.weight.device)
    model.source = {"weight_sha256": fingerprint, "config_sha256": file_sha256(path / "config.json")}
    return model.eval().requires_grad_(False)


def save_checkpoint(path, model, *, optimizer=None, step=0, metadata=None, training_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "blockspec-dual-view-v1", "config": model.config.to_dict(),
               "state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
               "trainable": [key for key, value in model.named_parameters() if value.requires_grad],
               "optimizer": None if optimizer is None else optimizer.state_dict(),
               "step": step, "metadata": metadata or {}, "source": source_identity(getattr(model, "source", {})),
               "training_state": training_state}
    with path.open("xb") as handle:
        torch.save(payload, handle)


def load_checkpoint(path, *, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "blockspec-dual-view-v1":
        raise ValueError("unknown dual-view checkpoint format")
    config = DualViewConfig(**payload["config"])
    with torch.device("meta"):
        model = DualViewDecoder(config)
    expected, state = model.state_dict(), payload["state"]
    if set(expected) != set(state):
        raise ValueError("checkpoint tensor keys differ")
    for key, value in state.items():
        if value.shape != expected[key].shape or not torch.isfinite(value).all():
            raise ValueError(f"invalid checkpoint tensor: {key}")
    if config.tie_word_embeddings and not torch.equal(state["embedding.weight"], state["head.weight"]):
        raise ValueError("checkpoint tied weights differ")
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    model.frequencies = model._frequencies(model.embedding.weight.device)
    model.requires_grad_(False)
    parameters = dict(model.named_parameters())
    if set(payload["trainable"]) - parameters.keys():
        raise ValueError("unknown trainable checkpoint parameters")
    for name in payload["trainable"]:
        parameters[name].requires_grad_(True)
    model.source = source_identity(payload["source"])
    return model, {key: payload.get(key) for key in ("optimizer", "step", "metadata", "training_state")}
