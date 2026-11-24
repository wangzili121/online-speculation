"""Prepare BF16 DSV4.1 diffusion projections for a native PyPTO TP rank."""

from .pypto_dsv41_swa import BLOCK, HEAD_DIM

QKV_TENSORS = ("wq_a.weight", "q_norm.weight", "wq_b.weight", "wkv.weight", "kv_norm.weight", "attn_sink")
OUTPUT_TENSORS = ("wo_a.weight", "wo_b.weight")
INDEX_TENSORS = ("indexer.wq_b.weight", "indexer.weights_proj.weight")


def load_draft_layer(model, head, layer, *, tp_rank, tp=4):
    """Validate the exchange artifact and load one layer's attention projections."""
    config, manifest, files = read_draft_artifact(model, head)
    weights = read_draft_weights(head, manifest, files, layer)
    return config, manifest, prepare_draft_operands(weights, config, tp_rank=tp_rank, tp=tp)


def read_draft_artifact(model, head):
    from .vllm_ascend.dsv41_contract import text_config, validate_head
    from .vllm_ascend.head_contract import find_weight_index, read_json

    manifest = read_json(head / "manifest.json")
    manifest = validate_head(model, head, mask_token_id=manifest["mask_token_id"])
    config = text_config(read_json(model / "config.json"))
    return config, manifest, read_json(find_weight_index(head))["weight_map"]


def read_draft_weights(head, manifest, files, layer):
    from contextlib import ExitStack

    from safetensors import safe_open

    if not 0 <= layer < len(manifest["layers"]):
        raise ValueError("Draft layer is outside the checkpoint")
    owns_index = manifest["layers"][layer]["index_source"] == layer
    names = QKV_TENSORS + OUTPUT_TENSORS + (INDEX_TENSORS if owns_index else ())
    with ExitStack() as stack:
        weights, shards = {}, {}
        for name in names:
            key = f"layers.{layer}.attn.{name}"
            file = files[key]
            if file not in shards:
                shards[file] = stack.enter_context(safe_open(head / file, framework="pt", device="cpu"))
            weights[name] = shards[file].get_tensor(key)
        return weights


def prepare_draft_operands(weights, config, *, tp_rank, tp=4):
    operands = prepare_qkv(weights, config, tp_rank=tp_rank, tp=tp)
    operands.update(prepare_output(weights, config, tp_rank=tp_rank, tp=tp))
    if INDEX_TENSORS[0] in weights:
        operands.update(idx_wq=weights[INDEX_TENSORS[0]].T.contiguous(),
                        idx_weight=weights[INDEX_TENSORS[1]].T.contiguous())
    return operands


def prepare_qkv(weights, config, *, tp_rank, tp=4):
    """Keep the exchanged head precision and shard only the query heads."""
    import torch

    hidden, latent = config["hidden_size"], config["q_lora_rank"]
    heads, dim = config["num_attention_heads"], config["head_dim"]
    if not 0 <= tp_rank < tp or heads % tp or dim != HEAD_DIM:
        raise ValueError("Invalid DSV4.1 draft head partition")
    required = {
        "wq_a.weight": (latent, hidden),
        "q_norm.weight": (latent,),
        "wq_b.weight": (heads * dim, latent),
        "wkv.weight": (dim, hidden),
        "kv_norm.weight": (dim,),
        "attn_sink": (heads,),
    }
    for name, shape in required.items():
        weight = weights[name]
        dtype = torch.float32 if name == "attn_sink" else torch.bfloat16
        if weight.shape != shape or weight.dtype != dtype or weight.device.type != "cpu":
            raise ValueError(f"Invalid DSV4.1 draft operand {name}")
    local_heads = heads // tp
    low, high = tp_rank * local_heads, (tp_rank + 1) * local_heads
    return dict(
        wq_a=weights["wq_a.weight"].T.contiguous(),
        q_norm=weights["q_norm.weight"].contiguous(),
        wq_b=weights["wq_b.weight"][low * dim:high * dim].T.contiguous(),
        wkv=weights["wkv.weight"].T.contiguous(),
        kv_norm=weights["kv_norm.weight"].contiguous(),
        sink=weights["attn_sink"][low:high].contiguous(),
    )


def prepare_output(weights, config, *, tp_rank, tp=4):
    """Assign complete Wo-A groups and their matching Wo-B columns to this rank."""
    import torch

    groups = config["o_groups"]
    o_rank = config["o_lora_rank"]
    width = config["num_attention_heads"] * config["head_dim"]
    hidden = config["hidden_size"]
    if not 0 <= tp_rank < tp or groups % tp or width % groups:
        raise ValueError("Invalid DSV4.1 grouped output partition")
    wo_a, wo_b = (weights[name] for name in OUTPUT_TENSORS)
    if any(value.dtype != torch.bfloat16 or value.device.type != "cpu" for value in (wo_a, wo_b)):
        raise ValueError("Grouped output requires BF16 CPU weights")
    if wo_a.shape != (groups * o_rank, width // groups) or wo_b.shape != (hidden, groups * o_rank):
        raise ValueError("Invalid DSV4.1 grouped output shape")
    local_groups = groups // tp
    low, high = tp_rank * local_groups, (tp_rank + 1) * local_groups
    return dict(wo_a=wo_a.view(groups, o_rank, width // groups)[low:high].contiguous(),
                wo_b=wo_b[:, low * o_rank:high * o_rank].T.contiguous())


def make_qkv_program(native_module, config, *, tp=4, block_size=BLOCK):
    """Compose the native BF16 matmul, weighted RMSNorm and adjacent RoPE leaves.

    Cosine and sine are supplied by the model's layer-specific RoPE path. Q
    keeps its token-major layout so SWA and compressed Attention can read it.
    """
    import pypto.language as pl

    hidden, latent = config["hidden_size"], config["q_lora_rank"]
    heads, dim = config["num_attention_heads"], config["head_dim"]
    if tp <= 0 or heads % tp or dim != HEAD_DIM:
        raise ValueError("Invalid DSV4.1 draft QKV geometry")
    heads //= tp
    rope_width, eps = config["qk_rope_head_dim"], config["rms_norm_eps"]
    projection = native_module("qkv_proj_rope").make_bf16_projection_staged
    norm = native_module("rmsnorm").make_blocked_rmsnorm
    rope = native_module("rope").make_adjacent_rope
    definitions = dict(
        qa=projection(hidden, latent, capacity=block_size),
        qnorm=pl.jit(norm(block_size, latent, eps)._func),
        qb=projection(latent, heads * dim, capacity=block_size),
        qrope=pl.jit(rope(block_size, heads, dim, rope_width)._func),
        kv=projection(hidden, dim, capacity=block_size),
        kvnorm=pl.jit(norm(block_size, dim, eps)._func),
        kvrope=pl.jit(rope(block_size, 1, dim, rope_width)._func),
    )
    steps = (
        ("qa", ("hidden", "wq_a", "qa", "active")),
        ("qnorm", ("qa", "q_norm", "latent", "active")),
        ("qb", ("latent", "wq_b", "qb", "active")),
        ("qrope", ("qb", "cos", "sin", "query", "active")),
        ("kv", ("hidden", "wkv", "kv", "active")),
        ("kvnorm", ("kv", "kv_norm", "normalized_kv", "active")),
        ("kvrope", ("normalized_kv", "cos", "sin", "temporary", "active")),
    )
    return native_module("prefill_layer").compose_l2("orthrus_qkv", definitions, steps, block_size)


def make_draft_attention_reduce_pack(capacity, hidden, tp):
    """Pack BF16 draft contributions into FP32 collective buffers."""
    import pypto.language as pl

    @pl.jit
    def pack(
        value: pl.Tensor[[capacity, hidden], pl.BF16],
        packed: pl.Out[pl.Tensor[[tp * capacity, hidden], pl.FP32]],
        physical: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        for row in pl.spmd(tp * capacity, name_hint="orthrus_attn_forward_reduce_pack"):
            peer, local = row // capacity, row % capacity
            source = peer * physical + local
            if local < physical and source < active:
                picked = pl.slice(value, [1, hidden], [source, 0])
                packed[row:row + 1, :] = pl.cast(picked, pl.FP32)
            else:
                packed[row:row + 1, :] = pl.full([1, hidden], dtype=pl.FP32, value=0.)
        return packed

    return pack


def make_output_program(native_module, config, *, tp=4, head_major_input=False, block_size=BLOCK):
    """Apply inverse RoPE and grouped Wo, yielding the local TP contribution."""
    import pypto.language as pl

    heads, dim = config["num_attention_heads"] // tp, config["head_dim"]
    width, local_out = heads * dim, config["o_groups"] // tp * config["o_lora_rank"]
    rope_width = config["qk_rope_head_dim"]
    projection = native_module("qkv_proj_rope")
    head_stride, token_stride = (block_size, 1) if head_major_input else (1, heads)

    @pl.jit
    def narrow(
        attention: pl.Tensor[[block_size * heads, dim], pl.FP32],
        context: pl.Out[pl.Tensor[[block_size, width], pl.BF16]],
    ):
        for task in pl.spmd(block_size * heads, name_hint="orthrus_output_layout"):
            token, head = task // heads, task % heads
            row = head * head_stride + token * token_stride
            value = pl.slice(attention, [1, dim], [row, 0])
            context[token:token + 1, head * dim:(head + 1) * dim] = pl.cast(value, pl.BF16, mode="rint")
        return context

    definitions = dict(
        narrow=narrow,
        inverse=pl.jit(projection.make_rope(heads, inverse=True, head_dim=dim, rope_dim=rope_width)._func),
        grouped=pl.jit(projection.make_prefill_grouped_output(tp)._func),
        down=projection.make_bf16_projection_staged(local_out, config["hidden_size"], capacity=block_size),
    )
    steps = (
        ("narrow", ("attention", "context")),
        ("inverse", ("context", "cos", "sin", "unrotated", "active")),
        ("grouped", ("unrotated", "wo_a", "grouped", "active")),
        ("down", ("grouped", "wo_b", "contribution", "active")),
    )
    return native_module("prefill_layer").compose_l2("orthrus_output", definitions, steps, block_size)
