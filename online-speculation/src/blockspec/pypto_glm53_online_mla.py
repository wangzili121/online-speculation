"""Train the final GLM draft MLA on the same resident PyPTO Serving worker."""

from math import prod

from .pypto_glm53_draft_attention import make_resident_mla_replay
from .pypto_glm53_head import head_layout
from .pypto_glm53_mla import HEADS, HEAD_TILE, LATENT, QUERY_ROWS, TILE, VALUE
from .pypto_glm53_online_attention import make_mla_input_projection_backward, make_mla_local_backward
from .pypto_glm53_online_optimizer import (
    make_mla_head_adamw, make_mla_head_optimizer_init, make_mla_head_publish,
)


PROGRAMS = ("orthrus_mla_replay", "orthrus_mla_local_vjp", "orthrus_mla_input_vjp",
            "orthrus_mla_init_adamw", "orthrus_mla_adamw", "orthrus_mla_publish", "orthrus_mla_finite",
            "orthrus_mla_input_norm")


def prepare_mla_update(api, compiler):
    from models.glm5_3_flash.config import TOPK_INDEX_WIDTH

    ranks = api.RANKS
    capacity = (TOPK_INDEX_WIDTH + TILE - 1) // TILE * TILE
    entries = (make_resident_mla_replay(api), make_mla_local_backward(capacity, ranks=ranks),
               make_mla_input_projection_backward(ranks), make_mla_head_optimizer_init(ranks),
               make_mla_head_adamw(ranks), make_mla_head_publish(ranks), make_gradient_check(ranks),
               make_mla_input_norm(ranks, api.D))
    return {name: compiler.compile(name, entry) for name, entry in zip(PROGRAMS, entries, strict=True)}


class ResidentMLAUpdate:
    """Own one layer's replay, gradients, and optimizer slots on device."""

    def __init__(self, session, suffix, rows, learning_rate, weight_decay):
        import torch
        from models.glm5_3_flash.config import FLASH, Q_LORA, TOPK_INDEX_WIDTH

        self.session = session
        self.suffix = suffix
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.step_count = 0
        self.runner = session.runner
        self.layer = FLASH.dsa_layer_ids[-1]
        self.ordinal = FLASH.dsa_layer_ids.index(self.layer)
        layout, packed_width = head_layout()
        trainable = layout["index_q"][0]
        width, local = self.runner._api.D, HEADS * VALUE
        capacity = (TOPK_INDEX_WIDTH + TILE - 1) // TILE * TILE
        score_rows = QUERY_ROWS * HEAD_TILE

        def fp32(*shape):
            return session.allocate(shape, torch.float32)

        def bf16(*shape):
            return session.allocate(shape, torch.bfloat16)

        whole = self.runner.stacked["orthrus_head"]
        self.params = {name: self._view(whole, shape, self.ordinal * packed_width + start)
                       for name, (start, _count, shape) in layout.items()}
        self.live = self._view(whole, (1, packed_width), self.ordinal * packed_width)
        self.master, self.first, self.second = (fp32(1, trainable) for _ in range(3))
        self.staged = bf16(1, trainable)
        self.gradient = fp32(1, trainable)
        self.finite_flags = fp32(48, 8)
        self.gradients = {
            name: self._view(self.gradient, (1, *shape) if len(shape) == 1 else shape, start)
            for name, (start, _count, shape) in layout.items() if start < trainable
        }
        self.x = bf16(rows, width)
        self.raw_q, self.normed_q = bf16(rows, Q_LORA), bf16(rows, Q_LORA)
        self.query, self.raw_kv = bf16(rows, local), bf16(rows, LATENT)
        self.query_history, self.query_block = bf16(score_rows, LATENT), bf16(score_rows, LATENT)
        self.block, self.selected = bf16(TILE, LATENT), bf16(QUERY_ROWS * capacity, LATENT)
        self.history_bias, self.context = fp32(QUERY_ROWS, capacity), bf16(rows, local)
        self.dquery, self.dblock = fp32(rows, local), fp32(TILE, LATENT)
        self.dlatent = self._view(self.dblock, (rows, LATENT))
        self.dx = fp32(rows, width)
        self.projection_scratch = (
            fp32(rows, Q_LORA), fp32(rows, Q_LORA), fp32(rows, LATENT), fp32(rows, Q_LORA),
            fp32(rows, LATENT), fp32(rows, width), fp32(rows, width),
        )
        bias = torch.full((1, TILE), -3.4e38, dtype=torch.float32)
        bias[:, :QUERY_ROWS] = 0
        self.block_bias = fp32(1, TILE)
        self._upload(self.block_bias, bias)
        self.table = session.allocate((self.runner._api.TABLE_MAX,), torch.int32)
        self.pool_blocks = session.allocate((self.runner._api.POOL_PAGE_ROWS,), torch.int32)
        self.query_metadata = {name: session.allocate((QUERY_ROWS,), torch.int32) for name in
                               ("seg_start", "pool_count_tok", "tail_start", "tail_count", "kv_len_tok")}
        self._launch(PROGRAMS[3], self.live, self.master, self.first, self.second)

    def _view(self, tensor, shape, offset=0):
        from pypto.runtime import StackedDeviceTensor

        runtime = self.session.runtime
        stride = tensor.shards[0].nbytes // prod(tensor.shape[1:])
        handles = [runtime.view_tensor(shard, shape, byte_offset=offset * stride, worker_id=rank)
                   for rank, shard in zip(tensor.worker_ids, tensor.shards, strict=True)]
        return StackedDeviceTensor(handles, (len(handles), *shape), tensor.worker_ids)

    def _upload(self, tensor, payload):
        # Only page/segment coordinates and the constant block mask come from the host.
        data = payload.contiguous().share_memory_()
        for rank, shard in zip(tensor.worker_ids, tensor.shards, strict=True):
            self.session.runtime.copy_to(shard.data_ptr, data.data_ptr(), data.nbytes, worker_id=rank)

    def _launch(self, name, *args):
        self.runner._run_l3(self.runner.auxiliary_programs[name], *args, config=self.runner.config)

    def replay(self, inputs, slot):
        norm = self.suffix._weight("norm_input_norm", self.layer)
        self._upload(self.table, inputs["block_table"][slot])
        self._upload(self.pool_blocks, inputs["pool_blocks"])
        for name, tensor in self.query_metadata.items():
            self._upload(tensor, inputs[name][:QUERY_ROWS])
        key = self.suffix._weight("mla_w_k", self.ordinal)
        value = self.suffix._weight("mla_w_v", self.ordinal)
        self.ar_key = self._view(key, (HEADS * VALUE, LATENT))
        self.ar_value = self._view(value, (HEADS * VALUE, LATENT))
        cache = self.suffix._weight("mla_latent_cache", self.ordinal)
        pool = self.suffix._weight("index_pool_cache", self.ordinal)
        w = self.params
        self._launch(PROGRAMS[7], self.suffix.collapse, norm, self.x)
        self._launch(PROGRAMS[0], self.x, w["q_a"], w["q_norm"], w["q_b"],
                     w["kv_a"], w["kv_norm"], self.ar_key, w["key"], self.ar_value, w["value"],
                     w["index_q"], w["index_gate"], cache, self.table, pool, self.pool_blocks,
                     *self.query_metadata.values(), self.block_bias, self.raw_q, self.normed_q,
                     self.query, self.raw_kv, self.query_history, self.query_block, self.block, self.selected,
                     self.history_bias, self.context)

    def _check_gradients(self):
        import torch

        self._launch(PROGRAMS[6], self.gradient, self.finite_flags)
        flags = torch.empty((48, 8), dtype=torch.float32).share_memory_()
        for rank, shard in zip(self.finite_flags.worker_ids, self.finite_flags.shards, strict=True):
            self.session.runtime.copy_from(flags.data_ptr(), shard.data_ptr, flags.nbytes, worker_id=rank)
            if flags.count_nonzero():
                raise FloatingPointError(f"rank {rank} produced a nonfinite GLM MLA gradient")

    def step(self, upstream, inputs, slot):
        self.replay(inputs, slot)
        w, g = self.params, self.gradients
        self._launch(PROGRAMS[1], upstream, self.context, w["out"], self.query, self.ar_key, w["key"],
                     self.ar_value, w["value"], self.query_history, self.query_block, self.selected, self.block,
                     self.history_bias, self.block_bias, g["out"], g["key"], g["value"], self.dquery, self.dblock)
        step = getattr(self.runner, "_orthrus_mla_epoch", 0) + 1
        self._launch(PROGRAMS[2], self.x, self.raw_q, self.normed_q, self.raw_kv, self.dquery, self.dlatent,
                     w["q_a"], w["q_norm"], w["q_b"], w["kv_a"], w["kv_norm"],
                     g["q_a"], g["q_norm"], g["q_b"], g["kv_a"], g["kv_norm"], self.dx,
                     *self.projection_scratch, step)
        self.runner._orthrus_mla_epoch = step
        self._check_gradients()
        optimizer_step = self.step_count + 1
        self._launch(PROGRAMS[4], self.gradient, self.master, self.first, self.second, self.staged,
                     self.learning_rate, self.weight_decay, 1 / (1 - 0.9 ** optimizer_step),
                     1 / (1 - 0.999 ** optimizer_step))
        self._launch(PROGRAMS[5], self.staged, self.live)
        self.step_count = optimizer_step


def make_gradient_check(ranks):
    """Check all trainable fields before allowing a staged update."""
    import pypto.language as pl

    layout, _ = head_layout()
    n = layout["index_q"][0]
    cores, tile = 48, 128

    @pl.jit
    def check(
        gradient: pl.Tensor[[1, n], pl.FP32],
        flags: pl.Out[pl.Tensor[[cores, 8], pl.FP32]],
    ):
        for core in pl.spmd(cores, name_hint="glm_mla_adamw_finite"):
            invalid = pl.full([1, tile], dtype=pl.FP32, value=0.0)
            for col in pl.range(core * tile, n, cores * tile):
                g = gradient[:, col:col + tile]
                good = pl.cast(pl.cast(pl.cmp(pl.abs(g), 3.4028234663852886e38, cmp_type=3), pl.INT32), pl.FP32)
                invalid = pl.maximum(invalid, pl.add(pl.neg(good), 1.0))
            full = pl.col_expand(pl.full([8, tile], dtype=pl.FP32, value=0.0), invalid)
            flag = pl.read(pl.row_sum(full), [0, 0])
            flags[core:core + 1, :] = pl.add(pl.full([1, 8], dtype=pl.FP32, value=0.0), flag)

    @pl.jit.host
    def entry(gradient: pl.Tensor[[ranks, 1, n], pl.FP32], flags: pl.Out[pl.Tensor[[ranks, cores, 8], pl.FP32]]):
        for rank in pl.range(ranks):
            check(gradient[rank], flags[rank], device=rank)

    return entry



def make_mla_input_norm(ranks, width):
    import pypto.language as pl
    import pypto.language.distributed as pld
    from models.glm5_3_flash.rmsnorm import rmsnorm

    rows = 16

    @pl.jit
    def normalize(x: pl.Tensor[[rows, width], pl.BF16], scale: pl.Tensor[[width], pl.BF16],
                  output: pl.Out[pl.Tensor[[rows, width], pl.BF16]]):
        rmsnorm(x, scale, output)

    @pl.jit.host
    def entry(x: pl.Tensor[[ranks, rows, width], pl.BF16], scale: pl.Tensor[[ranks, width], pl.BF16],
              output: pl.Out[pl.Tensor[[ranks, rows, width], pl.BF16]]):
        for rank in pl.range(pld.world_size()):
            normalize(x[rank], scale[rank], output[rank], device=rank)

    return entry
