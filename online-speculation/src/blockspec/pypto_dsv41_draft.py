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


def pack_draft_input_projection(operands):
    """Lay out the Q-A and KV channels consecutively for one resident GEMM."""
    import torch

    return torch.cat((operands["wq_a"], operands["wkv"]), dim=1)


def make_packed_draft_qkv(config, *, tp=4, capacity=16):
    """Project a BF16 draft head using the 1122 A3 GEMM, RMSNorm and RoPE."""
    import pypto.language as pl
    from models.deepseek_v4_1_flash_a3.gemm import gemm_bf16_to_bf16 as gemm
    from models.deepseek_v4_1_flash_a3.qkv_proj_rope import _tile_rsqrt_for_norm
    from models.deepseek_v4_1_flash_a3.rope import make_pipelined_rope

    hidden, latent = config["hidden_size"], config["q_lora_rank"]
    heads, dim = config["num_attention_heads"] // tp, config["head_dim"]
    q_width, joint_width = heads * dim, latent + dim
    eps = config["rms_norm_eps"]
    q_rope = make_pipelined_rope(
        capacity, heads, dim, inverse=False, name_hint="orthrus_q_rope", allow_early_resolve=False,
    )
    kv_rope = make_pipelined_rope(
        capacity, 1, dim, inverse=False, name_hint="orthrus_kv_rope", allow_early_resolve=False,
    )

    @pl.jit.inline
    def norm_projection(
        joint: pl.Tensor[[capacity, joint_width], pl.BF16],
        gamma: pl.Tensor,
        output: pl.Tensor,
        active: pl.Scalar[pl.INT32],
        columns: pl.constexpr,
        first_col: pl.constexpr,
    ):
        # Load directly from the joined GEMM's row stride. A view with a
        # narrower declared width would require repacking before a native norm.
        for task in pl.spmd((active + 7) // 8, name_hint="orthrus_qkv_norm"):
            row = task * 8
            rows = pl.min(8, active - row)
            raw = pl.load(joint, [row, first_col], [8, columns], valid_shape=[rows, columns])
            padded = pl.set_validshape(pl.fillpad(raw, pad_value=pl.PadValue.zero), 8, columns)
            value = pl.cast(padded, pl.FP32)
            weight_row = pl.tensor.view(gamma, [1, columns])
            gamma_row = pl.cast(pl.load(weight_row, [0, 0], [1, columns]), pl.FP32)
            sum_scratch = pl.create_tile([8, columns], dtype=pl.FP32)
            mean = pl.mul(pl.row_sum(pl.mul(value, value), tmp_tile=sum_scratch), 1.0 / columns)
            inverse_scratch = pl.create_tile([8, 1], dtype=pl.FP32)
            inv = _tile_rsqrt_for_norm(pl.add(mean, eps), inverse_scratch)
            normalized = pl.col_expand_mul(pl.row_expand_mul(value, inv), gamma_row)
            narrowed = pl.cast(normalized, pl.BF16, mode="rint")
            pl.store(pl.set_validshape(narrowed, rows, columns), [row, 0], output)
        return output

    @pl.jit
    def project(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a_kv: pl.Tensor[[hidden, joint_width], pl.BF16],
        q_gamma: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent, q_width], pl.BF16],
        kv_gamma: pl.Tensor[[dim], pl.BF16],
        cosine: pl.Tensor[[capacity, 32], pl.FP32],
        sine: pl.Tensor[[capacity, 32], pl.FP32],
        low_rank: pl.Out[pl.Tensor[[capacity, latent], pl.BF16]],
        query: pl.Out[pl.Tensor[[capacity, q_width], pl.BF16]],
        temporary: pl.Out[pl.Tensor[[capacity, dim], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        qa_kv = pl.create_tensor([capacity, joint_width], dtype=pl.BF16)
        q_raw = pl.create_tensor([capacity, q_width], dtype=pl.BF16)
        kv_normalized = pl.create_tensor([capacity, dim], dtype=pl.BF16)
        start = pl.system.task_dummy(deps=[])
        qa_kv, _ = gemm(x, wq_a_kv, qa_kv, active, start, hidden, joint_width,
                        128, 256, 32, "orthrus_qa_kv", 0, 0, False, 16, 2)
        low_rank = norm_projection(qa_kv, q_gamma, low_rank, active, latent, 0)
        kv_normalized = norm_projection(qa_kv, kv_gamma, kv_normalized, active, dim, latent)
        q_raw, _ = gemm(low_rank, wq_b, q_raw, active, start, latent, q_width,
                        128, 256, 32, "orthrus_qb", 0, 0, False, 16, 2)
        query = q_rope(q_raw, cosine, sine, query, active)
        temporary = kv_rope(kv_normalized, cosine, sine, temporary, active)
        if active < capacity:
            # K8 uses a 16-row KV carrier. Cube multiplies the masked lanes
            # as well, so those values must be zero even with reused buffers.
            for task in pl.spmd((capacity - active + 7) // 8, name_hint="orthrus_zero_kv_padding"):
                row = active + task * 8
                rows = pl.min(8, capacity - row)
                zero = pl.tile.full([8, dim], dtype=pl.BF16, value=0.0)
                pl.store(pl.set_validshape(zero, rows, dim), [row, 0], temporary)
        return low_rank, query, temporary

    return project


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


def pack_draft_output_projection(operands):
    """Use the same resident Cube layout as the 1122 target Wo projection."""
    from models.deepseek_v4_1_flash_a3.weights import pack_bf16_nz

    return dict(wo_a=pack_bf16_nz(operands["wo_a"].transpose(-2, -1).contiguous()),
                wo_b=pack_bf16_nz(operands["wo_b"]))


def make_packed_draft_output(*, block_size, head_major=False):
    """Rotate and project the local Attention result with the native A3 Wo."""
    import pypto.language as pl
    from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE
    from models.deepseek_v4_1_flash_a3.o_proj import decode_o_proj_nz_from_attention

    dim = FLASH.head_dim
    heads = FLASH.num_attention_heads // TP_SIZE
    groups = FLASH.o_groups // TP_SIZE
    group_width = heads * dim // groups
    output_rank, hidden = FLASH.o_lora_rank, FLASH.hidden_size
    width = heads * dim

    @pl.jit
    def project(
        attention: pl.Tensor[[block_size * heads, dim], pl.FP32],
        wo_a: pl.Tensor[[groups, group_width, output_rank], pl.BF16, pl.NZ],
        wo_b: pl.Tensor[[groups * output_rank, hidden], pl.BF16, pl.NZ],
        cosine: pl.Tensor[[block_size, 32], pl.FP32],
        sine: pl.Tensor[[block_size, 32], pl.FP32],
        contribution: pl.Out[pl.Tensor[[block_size, hidden], pl.BF16]],
    ):
        attended = pl.create_tensor([block_size, width], dtype=pl.BF16)
        if head_major:
            for task in pl.spmd(block_size * heads, name_hint="orthrus_swa_to_wo"):
                token, head = task // heads, task % heads
                row = head * block_size + token
                values = pl.load(attention, [row, 0], [1, dim])
                pl.store(pl.cast(values, pl.BF16, mode="rint"), [token, head * dim], attended)
        else:
            flat = pl.reshape(attention, [block_size, width])
            for token in pl.spmd(block_size, name_hint="orthrus_context_to_wo"):
                for column in pl.range(0, width, dim * 4):
                    values = pl.load(flat, [token, column], [1, dim * 4])
                    pl.store(pl.cast(values, pl.BF16, mode="rint"), [token, column], attended)
        return decode_o_proj_nz_from_attention(attended, wo_a, wo_b, cosine, sine, contribution, block_size)

    return project



def make_packed_draft_output_tape(block, last_layers):
    """Direct the original Wo intermediates to the requested suffix slot."""
    import pypto.language as pl
    from models.deepseek_v4_1_flash_a3.config import FLASH as model, TP_SIZE as tp
    from models.deepseek_v4_1_flash_a3.o_proj import decode_o_proj_nz_capture

    heads, dim = model.num_attention_heads // tp, model.head_dim
    groups, rank, hidden = model.o_groups // tp, model.o_lora_rank, model.hidden_size
    width, middle, phase = heads * dim, groups * rank, model.qk_rope_head_dim // 2
    tape_rows = max(1, last_layers * block)

    @pl.jit.inline(auto_scope=False)
    def project(
        attention: pl.Tensor[[block * heads, dim], pl.FP32],
        wo_a: pl.Tensor[[groups, width // groups, rank], pl.BF16, pl.NZ],
        wo_b: pl.Tensor[[middle, hidden], pl.BF16, pl.NZ],
        cosine: pl.Tensor[[block, phase], pl.FP32],
        sine: pl.Tensor[[block, phase], pl.FP32],
        contribution: pl.Tensor[[block, hidden], pl.BF16],
        tape_input: pl.Tensor[[tape_rows, width], pl.BF16],
        tape_latent: pl.Tensor[[tape_rows, middle], pl.BF16],
        head_major: pl.Scalar[pl.INT32],
        slot: pl.Scalar[pl.INT32],
    ):
        attended = pl.create_tensor([block, width], dtype=pl.BF16)
        if head_major != 0:
            for task in pl.spmd(block * heads, name_hint="orthrus_swa_to_wo"):
                token, head = task // heads, task % heads
                head_values = pl.load(attention, [head * block + token, 0], [1, dim])
                pl.store(pl.cast(head_values, pl.BF16, mode="rint"), [token, head * dim], attended)
        else:
            flat = pl.reshape(attention, [block, width])
            for token in pl.spmd(block, name_hint="orthrus_context_to_wo"):
                for column in pl.range(0, width, dim * 4):
                    quartet = pl.load(flat, [token, column], [1, dim * 4])
                    pl.store(pl.cast(quartet, pl.BF16, mode="rint"), [token, column], attended)
        if last_layers > 0:
            if slot >= 0:
                saved_input = pl.slice(tape_input, [block, width], [slot * block, 0])
                saved_latent = pl.slice(tape_latent, [block, middle], [slot * block, 0])
                decode_o_proj_nz_capture(attended, wo_a, wo_b, cosine, sine, contribution, block,
                                          saved_input, saved_latent)
            else:
                rotated = pl.create_tensor([block, width], dtype=pl.BF16)
                projected = pl.create_tensor([block, middle], dtype=pl.BF16)
                decode_o_proj_nz_capture(attended, wo_a, wo_b, cosine, sine, contribution, block, rotated, projected)
        else:
            unused_rotation = pl.create_tensor([block, width], dtype=pl.BF16)
            unused_latent = pl.create_tensor([block, middle], dtype=pl.BF16)
            decode_o_proj_nz_capture(attended, wo_a, wo_b, cosine, sine, contribution, block,
                                      unused_rotation, unused_latent)
        return contribution

    return project


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


def make_draft_attention_reduce_pack(capacity, hidden, tp, *, packet_rows=None):
    """Pack BF16 draft contributions into FP32 collective buffers."""
    import pypto.language as pl

    rows = capacity if packet_rows is None else packet_rows

    @pl.jit
    def pack(
        value: pl.Tensor[[capacity, hidden], pl.BF16],
        packed: pl.Out[pl.Tensor[[tp * rows, hidden], pl.FP32]],
        physical: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        for row in pl.spmd(tp * rows, name_hint="orthrus_attn_forward_reduce_pack"):
            peer, local = row // rows, row % rows
            source = peer * physical + local
            if local < physical and source < active:
                picked = pl.slice(value, [1, hidden], [source, 0])
                packed[row:row + 1, :] = pl.cast(picked, pl.FP32)
            else:
                packed[row:row + 1, :] = pl.full([1, hidden], dtype=pl.FP32, value=0.)
        return packed

    return pack


def make_draft_attention_reduce_narrow(packet_rows, capacity, hidden):
    """Round received rows once and write the padded FFN boundary directly."""
    import pypto.language as pl

    @pl.jit
    def narrow(
        value: pl.Tensor[[packet_rows, hidden], pl.FP32],
        output: pl.Out[pl.Tensor[[capacity, hidden], pl.BF16]],
    ):
        for row in pl.spmd(capacity, name_hint="orthrus_draft_reduce_narrow"):
            if row < packet_rows:
                values = pl.cast(pl.slice(value, [1, hidden], [row, 0]), pl.BF16, mode="rint")
                output[row:row + 1, :] = values
            else:
                output[row:row + 1, :] = pl.full([1, hidden], dtype=pl.BF16, value=0.)
        return output

    return narrow


def make_output_program(native_module, config, *, tp=4, head_major_input=False, block_size=BLOCK):
    """Apply inverse RoPE and grouped Wo, yielding the local TP contribution."""
    import pypto.language as pl

    heads, dim = config["num_attention_heads"] // tp, config["head_dim"]
    width, local_out = heads * dim, config["o_groups"] // tp * config["o_lora_rank"]
    rope_width = config["qk_rope_head_dim"]
    projection = native_module("qkv_proj_rope")
    head_stride, token_stride = (block_size, 1) if head_major_input else (1, heads)

    plain_width = dim - rope_width

    @pl.jit
    def inverse(
        attention: pl.Tensor[[block_size * heads, dim], pl.FP32],
        cos: pl.Tensor[[block_size, rope_width // 2], pl.FP32],
        sin: pl.Tensor[[block_size, rope_width // 2], pl.FP32],
        unrotated: pl.Out[pl.Tensor[[block_size, width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for task in pl.spmd(active * heads, name_hint="orthrus_output_inverse_rope"):
            token, head = task // heads, task % heads
            row = head * head_stride + token * token_stride
            column = head * dim
            plain = pl.slice(attention, [1, plain_width], [row, 0])
            unrotated[token:token + 1, column:column + plain_width] = pl.cast(plain, pl.BF16, mode="rint")
            tail = pl.slice(attention, [1, rope_width], [row, plain_width])
            # Preserve the intermediate BF16 rounding of the serving output.
            tail = pl.cast(pl.cast(tail, pl.BF16, mode="rint"), pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            c = cos[token:token + 1, :]
            s = pl.mul(sin[token:token + 1, :], -1.0)
            re = pl.sub(pl.mul(even, c), pl.mul(odd, s))
            im = pl.add(pl.mul(even, s), pl.mul(odd, c))
            rotated = pl.full([1, rope_width], dtype=pl.FP32, value=0.0)
            rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            unrotated[token:token + 1, column + plain_width:column + dim] = pl.cast(rotated, pl.BF16, mode="rint")
        return unrotated

    definitions = dict(
        inverse=inverse,
        grouped=pl.jit(projection.make_prefill_grouped_output(tp)._func),
        down=projection.make_bf16_projection_staged(local_out, config["hidden_size"], capacity=block_size),
    )
    steps = (
        ("inverse", ("attention", "cos", "sin", "unrotated", "active")),
        ("grouped", ("unrotated", "wo_a", "grouped", "active")),
        ("down", ("grouped", "wo_b", "contribution", "active")),
    )
    return native_module("prefill_layer").compose_l2("orthrus_output", definitions, steps, block_size)
