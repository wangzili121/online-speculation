"""Draft query projection and read-only A3 QLI over the committed prefix."""

from dataclasses import dataclass

from .pypto_dsv41_sparse import TOPK
from .pypto_dsv41_swa import BLOCK, HEAD_DIM, WINDOW

INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_CAPACITIES = (512, 1024, 2048, 4096, 8192, 16384)


@dataclass(frozen=True)
class IndexGeometry:
    ratio: int
    bucket: int
    code_rows: int
    scale_rows: int
    code_stride: int
    scale_stride: int
    kv_stride: int
    code_offset: int
    scale_offset: int
    kv_offset: int


def index_controls(history, *, candidate_blocks, candidate_block_size, max_context):
    """Bind the three planes returned by GenerationSession.draft_history."""
    import torch

    ratio = history["role"]["ratio"]
    anchor = history["committed_tokens"]
    if ratio not in (1, 2) or not 0 <= anchor <= max_context <= INDEX_CAPACITIES[-1]:
        raise ValueError("Draft indexer requires a supported A3 anchor and compression ratio")
    visible = anchor // ratio
    if ratio == 1 and (candidate_block_size != 8 or (visible + 7) // 8 > candidate_blocks):
        raise ValueError("This A3 entry requires that the C1A candidate pool covers the visible prefix")
    pages_per_index = WINDOW // ratio
    bucket = next(size for size in INDEX_CAPACITIES if size >= visible)
    codes, scales = history["planes"]["index"]
    kv, = history["planes"]["long"]
    pages = codes["pages"]
    page_count = (visible + pages_per_index - 1) // pages_per_index
    if len(pages) < page_count or any(pages != plane["pages"] for plane in (scales, kv)):
        raise ValueError("Index keys, scales and KV require the same complete history page table")

    def plane_geometry(plane, width, dtype):
        shape, stride, offset = plane["shape"], plane["stride_rows"], plane["offset_rows"]
        if (shape[1:] != (pages_per_index, 1, width)
                or not 0 <= offset <= stride - pages_per_index
                or plane["handle"].shape != (shape[0] * stride, width)
                or plane["handle"].dtype != dtype
                or any(not 0 < page < shape[0] for page in pages[:page_count])):
            raise ValueError("Draft indexer received incompatible physical page geometry")
        return shape[0] * stride, stride, offset

    codes_rows, code_stride, code_offset = plane_geometry(codes, INDEX_DIM, torch.int8)
    scales_rows, scale_stride, scale_offset = plane_geometry(scales, 1, torch.float16)
    _, kv_stride, kv_offset = plane_geometry(kv, HEAD_DIM, torch.bfloat16)
    geometry = IndexGeometry(ratio, bucket, codes_rows, scales_rows, code_stride, scale_stride, kv_stride,
                             code_offset, scale_offset, kv_offset)
    page_table = torch.zeros(bucket // pages_per_index, dtype=torch.int32)
    page_table[:page_count] = torch.tensor(pages[:page_count], dtype=torch.int32)
    positions = torch.full((BLOCK,), anchor - 1, dtype=torch.int32)
    return geometry, dict(page_table=page_table, positions=positions, visible=visible)


def make_indexer_program(native_module, config, geometry):
    """Project the draft index queries, reuse A3 QLI and emit physical attention rows."""
    import pypto.language as pl

    if (config["index_n_heads"], config["index_head_dim"], config["index_topk"]) != (INDEX_HEADS, INDEX_DIM, TOPK):
        raise ValueError("A3 QLI requires the native 32x128 and Top-512 geometry")
    latent, hidden = config["q_lora_rank"], config["hidden_size"]
    rope_dim = config["qk_rope_head_dim"]
    page_rows, bucket = WINDOW // geometry.ratio, geometry.bucket
    code_rows, scale_rows = geometry.code_rows, geometry.scale_rows
    code_stride, scale_stride, kv_stride = geometry.code_stride, geometry.scale_stride, geometry.kv_stride
    code_offset, scale_offset, kv_offset = geometry.code_offset, geometry.scale_offset, geometry.kv_offset
    index_width = INDEX_HEADS * INDEX_DIM
    scale_factor = index_width ** -0.5
    projection = native_module("qkv_proj_rope").make_bf16_projection_staged
    rope = native_module("rope").make_adjacent_rope(BLOCK, INDEX_HEADS, INDEX_DIM, rope_dim)
    indexer = native_module("indexer")

    @pl.jit
    def rotate(
        projected: pl.Tensor[[BLOCK, index_width], pl.BF16],
        cos: pl.Tensor[[BLOCK, rope_dim // 2], pl.FP32],
        sin: pl.Tensor[[BLOCK, rope_dim // 2], pl.FP32],
        query: pl.InOut[pl.Tensor[[BLOCK, INDEX_HEADS, INDEX_DIM], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        rotated = pl.reshape(query, [BLOCK, index_width])
        rope(projected, cos, sin, rotated, active)
        return query

    @pl.jit
    def scale_weights(
        weights: pl.Tensor[[BLOCK, INDEX_HEADS], pl.BF16],
        scaled: pl.Out[pl.Tensor[[BLOCK, INDEX_HEADS], pl.FP16]],
    ):
        for token in pl.spmd(BLOCK, name_hint="orth_index_weight_scale"):
            value = pl.cast(weights[token:token + 1, :], pl.FP32)
            scaled[token:token + 1, :] = pl.cast(pl.mul(value, scale_factor), pl.FP16, mode="rint")
        return scaled

    @pl.jit
    def gather_pages(
        codes: pl.Tensor[[code_rows, INDEX_DIM], pl.INT8],
        scales: pl.Tensor[[scale_rows, 1], pl.FP16],
        table: pl.Tensor[[bucket // page_rows], pl.INT32],
        zero_codes: pl.Tensor[[page_rows, INDEX_DIM], pl.INT8],
        keys: pl.Out[pl.Tensor[[bucket, INDEX_DIM], pl.INT8]],
        key_scales: pl.Out[pl.Tensor[[bucket], pl.FP16]],
        visible: pl.Scalar[pl.INT32],
    ):
        scale_plane = pl.reshape(scales, [1, scale_rows])
        scale_output = pl.reshape(key_scales, [1, bucket])
        for logical in pl.spmd(bucket // page_rows, name_hint="orth_gather_index_pages"):
            start = logical * page_rows
            rows = pl.min(page_rows, pl.max(0, visible - start))
            if rows > 0:
                page = pl.cast(pl.read(table, [logical]), pl.INDEX)
                code = pl.slice(codes, [page_rows, INDEX_DIM], [page * code_stride + code_offset, 0],
                                valid_shape=[rows, INDEX_DIM])
                scale = pl.slice(scale_plane, [1, page_rows], [0, page * scale_stride + scale_offset],
                                 valid_shape=[1, rows])
                keys[start:start + page_rows, :] = pl.fillpad(code, pad_value=pl.PadValue.zero)
                scale_output[:, start:start + page_rows] = pl.fillpad(scale, pad_value=pl.PadValue.zero)
            else:
                keys[start:start + page_rows, :] = zero_codes[:, :]
                scale_output[:, start:start + page_rows] = pl.full([1, page_rows], pl.FP16, value=0.0)
        return keys, key_scales

    @pl.jit
    def map_rows(
        indices: pl.Tensor[[BLOCK, TOPK], pl.INT32],
        table: pl.Tensor[[bucket // page_rows], pl.INT32],
        physical_rows: pl.Out[pl.Tensor[[BLOCK, TOPK], pl.INT32]],
        lengths: pl.Out[pl.Tensor[[BLOCK], pl.INT32]],
        bias: pl.Out[pl.Tensor[[BLOCK, TOPK], pl.FP32]],
        visible: pl.Scalar[pl.INT32],
    ):
        count = pl.min(TOPK, visible)
        # Separate writers by a full 32-byte row before publishing packed lengths.
        counts = pl.create_tensor([BLOCK, 8], dtype=pl.INT32)
        with pl.spmd(BLOCK, name_hint="orth_qli_physical_rows") as mapping:
            token = pl.tile.get_block_idx()
            rows = pl.tile.full([1, TOPK], pl.INT32, value=-1)
            mask = pl.tile.full([1, TOPK], pl.FP32, value=-1e30)
            used = 0
            for lane in pl.range(count):
                index = pl.read(indices, [token, lane])
                if index >= 0:
                    if index < visible:
                        page = pl.read(table, [index // page_rows])
                        offset = index % page_rows
                        pl.tile.write(rows, [0, used], pl.cast(page * kv_stride + kv_offset + offset, pl.INT32))
                        pl.tile.write(mask, [0, used], 0.0)
                        used = used + 1
            pl.store(rows, [token, 0], physical_rows)
            pl.store(mask, [token, 0], bias)
            padded = pl.tile.full([1, 8], pl.INT32, value=0)
            pl.tile.write(padded, [0, 0], pl.cast(used, pl.INT32))
            pl.store(padded, [token, 0], counts)
        packed = pl.reshape(lengths, [1, BLOCK])
        with pl.spmd(BLOCK // 8, name_hint="orth_qli_lengths", deps=[mapping]):
            start = pl.tile.get_block_idx() * 8
            values = pl.tile.full([1, 8], pl.INT32, value=0)
            for lane in pl.range(8):
                pl.tile.write(values, [0, lane], pl.read(counts, [start + lane, 0]))
            pl.store(values, [0, start], packed)
        return physical_rows, lengths, bias

    leaves = dict(
        project=projection(latent, index_width, capacity=BLOCK),
        weight=projection(hidden, INDEX_HEADS, capacity=BLOCK),
        rotate=rotate,
        scale=scale_weights,
        gather=gather_pages,
        quantize=indexer.make_indexer_query_quantize_fp16_scale(BLOCK),
        select=indexer.make_prefill_indexer_select(BLOCK, bucket, compress_ratio=geometry.ratio),
        lower=map_rows,
    )
    steps = (
        ("project", ("latent", "idx_wq", "idx_projected", "active")),
        ("weight", ("hidden", "idx_weight", "idx_raw_weights", "active")),
        ("rotate", ("idx_projected", "cos", "sin", "idx_query", "active")),
        ("scale", ("idx_raw_weights", "idx_scaled_weights")),
        ("gather", ("index_codes", "index_scales", "page_table", "zero_codes", "index_keys", "key_scales", "visible")),
        ("quantize", ("idx_query", "query_codes", "query_scales", "active")),
        ("select", ("query_codes", "query_scales", "index_keys", "key_scales", "idx_scaled_weights",
                    "positions", "index_scores", "logical_rows", "active")),
        ("lower", ("logical_rows", "page_table", "physical_rows", "lengths", "bias", "visible")),
    )
    return native_module("prefill_layer").compose_l2("orthrus_indexer", leaves, steps, BLOCK)
