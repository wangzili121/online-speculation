"""Topology and BF16 head contract for the V4.1 diffusion view."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .head_contract import ContractError, find_weight_index, read_json, sha256_file

HEAD_FORMAT = "orthrus-dsv41-head-v1"
BLOCK_SIZE = 32


def _positive(config, name):
    value = config.get(name)
    if type(value) is not int or value < 1:
        raise ContractError(f"{name} must be a positive integer")
    return value


def _layer_ids(config, name, count):
    values = config.get(name)
    if (
        not isinstance(values, list)
        or any(type(i) is not int or not 0 <= i < count for i in values)
        or len(set(values)) != len(values)
    ):
        raise ContractError(f"invalid {name}")
    return set(values)


@dataclass(frozen=True)
class LayerRole:
    layer: int
    ratio: int
    kv_source: int | None = None
    index_source: int | None = None
    candidate_source: int | None = None

    @property
    def mode(self):
        if self.ratio == 0:
            return "swa"
        if self.layer == self.kv_source:
            return "full"
        return "reindex" if self.layer == self.index_source else "reuse"


@dataclass(frozen=True)
class V41Topology:
    layers: tuple[LayerRole, ...]
    window: int
    topk: int
    candidate_blocks: int
    candidate_block_size: int
    engram_layers: tuple[int, ...]

    @classmethod
    def from_config(cls, config):
        text = text_config(config)
        count = _positive(text, "num_hidden_layers")
        ratios = text.get("compress_ratios")
        # Released configs include three DSpark layers after the text backbone.
        extra = text.get("num_nextn_predict_layers", 0)
        if (
            not isinstance(ratios, list)
            or type(extra) is not int
            or extra < 0
            or len(ratios) not in (count, count + extra)
            or any(type(ratio) is not int or ratio not in (0, 1, 2) for ratio in ratios[:count])
        ):
            raise ContractError("invalid V4.1 backbone compression ratios")
        ratios = ratios[:count]
        kv_sources = _layer_ids(text, "kv_source_layer_ids", count)
        index_sources = _layer_ids(text, "index_source_layer_ids", count)
        engram = _layer_ids(text, "engram_layer_ids", count)
        if not kv_sources <= index_sources or any(ratios[i] == 0 for i in index_sources):
            raise ContractError("a KV source must also produce indices and SWA has no indexer")
        candidate = text.get("candidate_source_layer_id")
        if 1 in ratios and (
            type(candidate) is not int or candidate not in kv_sources or ratios[candidate] != 1
        ):
            raise ContractError("the candidate pool must originate at a C1A Full layer")

        kv_owner, index_owner = {}, {}
        roles = []
        for layer, ratio in enumerate(ratios):
            if ratio == 0:
                roles.append(LayerRole(layer, ratio))
                continue
            if layer in kv_sources:
                kv_owner[ratio] = layer
            if layer in index_sources:
                index_owner[ratio] = layer
            if ratio not in kv_owner:
                raise ContractError(f"layer {layer} has no preceding KV/indexer source")
            if ratio == 1 and kv_owner[ratio] != candidate:
                raise ContractError(f"C1A layer {layer} does not share the candidate source's KV")
            role = LayerRole(layer, ratio, kv_owner[ratio], index_owner[ratio], candidate if ratio == 1 else None)
            if ratio == 2 and role.mode == "reindex":
                raise ContractError("C2A Reindex is outside the V4.1 contract")
            roles.append(role)
        return cls(
            tuple(roles),
            _positive(text, "sliding_window"),
            _positive(text, "index_topk"),
            _positive(text, "candidate_topk_blocks") if 1 in ratios else 0,
            _positive(text, "candidate_block_size") if 1 in ratios else 0,
            tuple(sorted(engram)),
        )


def text_config(config):
    if not isinstance(config, dict) or config.get("model_type") != "deepseek_v41":
        raise ContractError("expected a DeepSeek-V4.1 model config")
    text = config.get("text_config")
    if not isinstance(text, dict) or text.get("model_type") != "deepseek_v41_text":
        raise ContractError("expected the V4.1 text configuration")
    return text


def head_shapes(config, topology):
    text = text_config(config)
    hidden = _positive(text, "hidden_size")
    heads = _positive(text, "num_attention_heads")
    dim = _positive(text, "head_dim")
    q_rank = _positive(text, "q_lora_rank")
    o_rank = _positive(text, "o_lora_rank")
    groups = _positive(text, "o_groups")
    if heads % groups or _positive(text, "num_key_value_heads") != 1:
        raise ContractError("expected grouped output projection and a single latent KV head")
    attention = {
        "attn_sink": (heads,),
        "wq_a.weight": (q_rank, hidden),
        "q_norm.weight": (q_rank,),
        "wq_b.weight": (heads * dim, q_rank),
        "wkv.weight": (dim, hidden),
        "kv_norm.weight": (dim,),
        "wo_a.weight": (groups * o_rank, heads * dim // groups),
        "wo_b.weight": (hidden, groups * o_rank),
    }
    indexer = {}
    if any(role.ratio for role in topology.layers):
        index_heads = _positive(text, "index_n_heads")
        indexer = {
            "indexer.wq_b.weight": (index_heads * _positive(text, "index_head_dim"), q_rank),
            "indexer.weights_proj.weight": (index_heads, hidden),
        }
    shapes = {}
    for role in topology.layers:
        weights = attention | indexer if role.layer == role.index_source else attention
        shapes.update({f"layers.{role.layer}.attn.{key}": shape for key, shape in weights.items()})
    return shapes


def head_manifest(model: Path, *, mask_token_id: int, artifact_role: str):
    config_path = model / "config.json"
    config = read_json(config_path)
    topology = V41Topology.from_config(config)
    if type(mask_token_id) is not int or not 0 <= mask_token_id < _positive(text_config(config), "vocab_size"):
        raise ContractError("mask_token_id must identify a token in the target vocabulary")
    for key in ("bos_token_id", "eos_token_id", "pad_token_id", "image_token_id"):
        token_ids = config.get(key)
        token_ids = token_ids if isinstance(token_ids, list) else [token_ids]
        if mask_token_id in token_ids:
            raise ContractError(f"mask_token_id overlaps {key}")
    if artifact_role not in ("initialized-plumbing", "trained"):
        raise ContractError("unknown V4.1 head artifact role")
    tokenizer = model / "tokenizer.json"
    return {
        "format": HEAD_FORMAT,
        "artifact_role": artifact_role,
        "precision": "bf16-exchange",
        "base_config_sha256": sha256_file(config_path),
        "base_index_sha256": sha256_file(find_weight_index(model)),
        "base_tokenizer_sha256": sha256_file(tokenizer),
        "mask_token_id": mask_token_id,
        "block_size": BLOCK_SIZE,
        "logit_alignment": "next-token",
        "engram_policy": "committed-prefix-anchor-mask",
        "layers": [asdict(role) for role in topology.layers],
    }


def validate_head(model: Path, head: Path, *, mask_token_id: int, require_trained: bool = False):
    """Check the exchange-format manifest and safetensors headers before loading weights."""
    from safetensors import safe_open

    manifest = read_json(head / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("format") != HEAD_FORMAT:
        raise ContractError(f"expected {HEAD_FORMAT}")
    role = manifest.get("artifact_role")
    if require_trained and role != "trained":
        raise ContractError("a trained V4.1 head is required")
    if manifest != head_manifest(model, mask_token_id=mask_token_id, artifact_role=role):
        raise ContractError("head manifest disagrees with the target or draft policy")
    config = read_json(model / "config.json")
    shapes = head_shapes(config, V41Topology.from_config(config))
    index = read_json(find_weight_index(head))
    mapping = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(mapping, dict) or mapping.keys() != shapes.keys():
        raise ContractError("V4.1 head tensor names do not match the topology")
    shards = {}
    for name, filename in mapping.items():
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ContractError(f"invalid head shard for {name}")
        shards.setdefault(filename, set()).add(name)
    for filename, names in shards.items():
        with safe_open(head / filename, framework="pt") as weights:
            if set(weights.keys()) != names:
                raise ContractError(f"head index and shard header disagree in {filename}")
            for name in names:
                tensor = weights.get_slice(name)
                dtype = "F32" if name.endswith(".attn_sink") else "BF16"
                if tuple(tensor.get_shape()) != shapes[name] or tensor.get_dtype() != dtype:
                    raise ContractError(f"wrong shape or dtype for {name}")
    return manifest
