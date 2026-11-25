"""Bind a resident Orthrus draft graph to a native A3 generation session."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from .pypto_dsv41_draft import (
    pack_draft_input_projection,
    pack_draft_output_projection,
    prepare_draft_operands,
    read_draft_artifact,
    read_draft_weights,
)
from .pypto_dsv41_indexer import INDEX_CAPACITIES
from .pypto_dsv41_packed_history import packed_history_lookup
from .pypto_dsv41_packed_indexer import packed_index_controls
from .pypto_dsv41_packed_swa import packed_swa_controls


_CACHE_VIEWS = {
    "window_cache_pool": "window_cache",
    "window_cache_scale_pool": "window_scales",
    "compressed_cache_pool": "compressed_cache",
    "compressed_cache_scale_pool": "compressed_scales",
    "index_cache_pool": "index_cache",
    "index_cache_scale_pool": "index_scales",
}


def draft_geometry(context):
    if not 1 <= context <= INDEX_CAPACITIES[-1]:
        raise ValueError("Packed draft context capacity must be in 1..16384")
    from models.deepseek_v4_1_flash_a3.config import BLOCK_SIZE

    pages = (context + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
    buckets = tuple(next(size for size in INDEX_CAPACITIES if size >= context // ratio) for ratio in (1, 2))
    return pages, *buckets


def draft_graph_manifest(context, block):
    from models.deepseek_v4_1_flash_a3 import config as native

    sources = ("pypto_dsv41_packed_graph.py", "pypto_dsv41_packed_draft_attention.py",
               "pypto_dsv41_packed_swa.py", "pypto_dsv41_packed_history.py", "pypto_dsv41_packed_indexer.py",
               "pypto_dsv41_draft.py", "pypto_dsv41_sparse.py", "pypto_dsv41_swa.py")
    native_root = Path(native.__file__).parent
    native_sources = ("config.py", "decode_fwd.py", "decode_indexer.py", "engram.py",
                      "gemm.py", "hc.py", "lm_head.py", "moe.py", "o_proj.py", "rope.py")
    return dict(
        schema=1, context=context, block=block, geometry=list(draft_geometry(context)),
        weight_layout=native.WEIGHT_LAYOUT_VERSION, cache_layout=native.CACHE_LAYOUT_VERSION,
        topology=dict(dp=native.DP_SIZE, tp=native.TP_SIZE, ep=native.EP_SIZE),
        sources={name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources},
        native_sources={name: sha256((native_root / name).read_bytes()).hexdigest() for name in native_sources},
    )


def load_packed_draft_graph(directory, config, *, context):
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram

    manifest = json.loads((directory / "orthrus_draft_manifest.json").read_text())
    if manifest != draft_graph_manifest(context, 32):
        raise ValueError("Compiled K32 draft graph does not match native context, model layout or source")
    return DistributedCompiledProgram.from_dir(
        directory, backend_type=config.arch, execution_mode=config.execution_mode,
        distributed_config=config.distributed_config,
    )


def _resident_head(runtime, checkpoint, head, world, tp, block):
    """Stream the exchange artifact into TP-sharded device banks once."""
    from models.deepseek_v4_1_flash_a3.weights import bf16_nz_shape

    config, manifest, files = read_draft_artifact(checkpoint.root, head)
    if manifest["block_size"] != block:
        raise ValueError("Draft graph width must match the head artifact")
    layers = manifest["layers"]
    index_owners = tuple(layer for layer, role in enumerate(layers) if role["index_source"] == layer)
    banks = [{} for _ in range(world)]
    total_bytes = 0

    def upload(name, payload, slot, slots, ranks):
        row_bytes = payload.nbytes
        for rank in ranks:
            bank = banks[rank].get(name)
            if bank is None:
                shape = (slots * payload.shape[0], *payload.shape[1:])
                bank = banks[rank][name] = runtime.alloc_tensor(shape, payload.dtype, worker_id=rank)
            runtime.copy_to(bank.data_ptr + slot * row_bytes, payload.data_ptr(), row_bytes, worker_id=rank)
        return len(ranks) * row_bytes

    for layer in range(len(layers)):
        tensors = read_draft_weights(head, manifest, files, layer)
        for lane in range(tp):
            operands = prepare_draft_operands(tensors, config, tp_rank=lane, tp=tp)
            wo = pack_draft_output_projection(operands)
            wo_a, wo_b = wo["wo_a"], wo["wo_b"]
            payloads = {
                "draft_q_a_kv": pack_draft_input_projection(operands).unsqueeze(0),
                "draft_q_gamma": operands["q_norm"].unsqueeze(0),
                "draft_q_b": operands["wq_b"].unsqueeze(0),
                "draft_kv_gamma": operands["kv_norm"].unsqueeze(0),
                "draft_sink": operands["sink"].unsqueeze(0),
                "draft_wo_a": wo_a.reshape(bf16_nz_shape(*wo_a.shape)),
                "draft_wo_b": wo_b.reshape(bf16_nz_shape(1, *wo_b.shape)),
            }
            ranks = tuple(range(lane, world, tp))
            for name, payload in payloads.items():
                total_bytes += upload(name, payload, layer, len(layers), ranks)
            if layer in index_owners:
                slot = index_owners.index(layer)
                total_bytes += upload("draft_index_q", operands["idx_wq"].unsqueeze(0), slot, len(index_owners), ranks)
                total_bytes += upload("draft_index_weight", operands["idx_weight"].unsqueeze(0),
                                      slot, len(index_owners), ranks)
    return banks, manifest, total_bytes


@dataclass
class DraftRoundStats:
    calls: int = 0
    prepare_seconds: float = 0.0
    upload_seconds: float = 0.0
    graph_seconds: float = 0.0
    readback_seconds: float = 0.0
    input_bytes: int = 0
    sampled_id_bytes: int = 0


class PackedDraftProposer:
    """Dispatch one request on its TP group while all ranks serve frozen experts."""

    def __init__(self, runtime, graph, config, target_ranks, decode_inputs, weights, engram,
                 rope_tables, counts, *, draft_head, block, context, tp):
        import torch
        from models.deepseek_v4_1_flash_a3.config import BLOCK_SIZE, FLASH
        from models.deepseek_v4_1_flash_a3.device_views import resident_graph_argument
        from models.deepseek_v4_1_flash_a3.engram import ENGRAM_K
        from models.deepseek_v4_1_flash_a3.lm_head import MAX_LOGIT_ROWS, SAMPLED_IDS_PAD
        from models.deepseek_v4_1_flash_a3.replay import DecodeInputs
        from pypto.runtime import DeviceTensor

        self.runtime, self.graph, self.config = runtime, graph, config
        self.weights, self.engram, self.rope_tables = weights, engram, rope_tables
        self.counts, self.tp, self.block = counts, tp, block
        self.world = len(target_ranks)
        self.pages, *self.buckets = draft_geometry(context)
        self.page_tables = tuple(decode_inputs.fields[0][key][0] for key in
                                 ("c1a_index_block_table", "c2a_index_block_table"))
        self.sampled = torch.empty((MAX_LOGIT_ROWS, SAMPLED_IDS_PAD), dtype=torch.int32)
        self.stats = DraftRoundStats()
        self.previous_owner = None
        capacity = max(16, block)
        geometry = (FLASH.hc_mult, FLASH.hidden_size)
        template = {
            "x_hc": torch.zeros((capacity, *geometry), dtype=torch.float32),
            "pre_mix": torch.zeros((capacity, FLASH.hc_mult), dtype=torch.float32),
            "engram_rows": torch.zeros((2, block, ENGRAM_K), dtype=torch.bfloat16),
            "draft_cos": torch.ones((2, capacity, FLASH.qk_rope_head_dim // 2), dtype=torch.float32),
            "draft_sin": torch.zeros((2, capacity, FLASH.qk_rope_head_dim // 2), dtype=torch.float32),
            "window_rows": torch.full((8, 16), -1, dtype=torch.int32),
            "window_starts": torch.full((8,), -2, dtype=torch.int32),
            "window_spans": torch.tensor(((0, 0, BLOCK_SIZE), (0, 0, 0)), dtype=torch.int32),
            "window_bias": torch.full((1, BLOCK_SIZE), -1e30, dtype=torch.float32),
            "c1a_table": torch.full((1, self.buckets[0] // BLOCK_SIZE), -1, dtype=torch.int32),
            "c2a_table": torch.full((1, self.buckets[0] // BLOCK_SIZE), -1, dtype=torch.int32),
            "request_ids": torch.zeros((capacity,), dtype=torch.int32),
            "c1a_visible": torch.zeros((capacity,), dtype=torch.int32),
            "c2a_visible": torch.zeros((capacity,), dtype=torch.int32),
            "logit_row_indices": torch.full((MAX_LOGIT_ROWS,), -1, dtype=torch.int32),
        }
        template["pre_mix"][:, 0] = 1
        self.inputs = DecodeInputs(template)
        device_head, manifest, self.resident_head_bytes = _resident_head(
            runtime, weights.checkpoint, draft_head, self.world, tp, block,
        )
        self.mask_token_id = manifest["mask_token_id"]
        self.head_role = manifest["artifact_role"]
        self.context = context
        static_inputs = dict(window_lookup=packed_history_lookup("fp8"),
                             compressed_lookup=packed_history_lookup("fp4"),
                             zero_kv=torch.zeros((BLOCK_SIZE, FLASH.head_dim), dtype=torch.bfloat16))
        self.ranks = []
        for rank, target in enumerate(target_ranks):
            bound = {**target, **device_head[rank], **self.inputs.bind(runtime, rank)}
            for name, value in static_inputs.items():
                bound[name] = runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=rank)
            for key, name in _CACHE_VIEWS.items():
                value = target[name]
                shape = (value.shape[0] * value.shape[1], *value.shape[2:])
                if value.shape[1] != self.pages:
                    raise ValueError(f"Native cache pages do not match draft graph: {name}")
                bound[key] = DeviceTensor(value.data_ptr, shape, value.dtype,
                                          buffer=value.buffer, byte_offset=value.byte_offset)
            # The draft graph owns outputs so an interrupted dispatch cannot
            # touch the target graph's logits or sampled IDs.
            bound["logits"] = runtime.alloc_tensor((MAX_LOGIT_ROWS, FLASH.vocab_size), torch.float32, worker_id=rank)
            bound["sampled_ids"] = runtime.alloc_tensor(self.sampled.shape, self.sampled.dtype, worker_id=rank)
            self.inputs.upload(runtime, rank, owner=-tp)
            self.ranks.append(bound)
        ranks = tuple(range(self.world))
        self.arguments = []
        for parameter in graph._get_metadata()[0]:
            name = parameter.name.removesuffix("__ssa_v0")
            argument = counts if name == "moe_num_tokens" else resident_graph_argument(
                [rank[name] for rank in self.ranks], ranks, parameter,
            )
            self.arguments.append(argument)

    def _stage(self, anchor_token, position, owner, candidates):
        import torch

        active, selected = self.inputs.fields[1:]
        indices = (anchor_token, self.mask_token_id)
        embeddings = self.weights.lookup_rows(torch.tensor(indices, dtype=torch.int64)).float()
        provisional = torch.full((self.block,), self.mask_token_id, dtype=torch.int64)
        provisional[0] = anchor_token
        history = self.engram.history.get(owner)
        try:
            engram_rows = self.engram.lookup(provisional, self.block, position, request_id=owner)
        finally:
            if history is None:
                self.engram.history.pop(owner, None)
            else:
                self.engram.history[owner] = history
        page_table, c2a_table = self.page_tables
        window = packed_swa_controls(page_table, position, self.pages)
        index = (
            packed_index_controls(page_table, position, 1, self.block, bucket=self.buckets[0]),
            packed_index_controls(c2a_table, position, 2, self.block, bucket=self.buckets[0]),
        )
        for image in (active, selected):
            image["x_hc"][0].copy_(embeddings[0])
            image["x_hc"][1:self.block].copy_(embeddings[1])
            image["pre_mix"].copy_(self.inputs.fields[0]["pre_mix"])
            image["engram_rows"].copy_(engram_rows)
            for name, table in zip(("draft_cos", "draft_sin"), self.rope_tables, strict=True):
                image[name][:, :self.block].copy_(table[:, position:position + self.block])
            for name, value in zip(("window_rows", "window_starts", "window_spans", "window_bias"), window, strict=True):
                image[name].copy_(value)
            for ratio, (_, table, _, visible) in zip(("c1a", "c2a"), index, strict=True):
                image[f"{ratio}_table"].copy_(table)
                image[f"{ratio}_visible"].copy_(visible)
        selected["logit_row_indices"].fill_(-1)
        selected["logit_row_indices"][:candidates].copy_(torch.arange(candidates, dtype=torch.int32))

    def propose(self, anchor_token, *, position, owner, candidates, uploads):
        if not 0 <= owner < self.world or owner % self.tp:
            raise ValueError("Draft owner must be the first TP rank")
        if not 0 < candidates < min(self.block, self.sampled.shape[0] + 1):
            raise ValueError("Draft candidate count must fit the block and LM head")
        if position < 0 or position + self.block > self.context:
            raise ValueError("Draft block exceeds the context capacity")
        start = perf_counter()
        self._stage(anchor_token, position, owner, candidates)
        self.stats.prepare_seconds += perf_counter() - start
        dirty = tuple(range(owner, owner + self.tp))
        if self.previous_owner is not None and self.previous_owner != owner:
            dirty += tuple(range(self.previous_owner, self.previous_owner + self.tp))
        self.previous_owner = owner
        start = perf_counter()
        list(uploads.map(lambda rank: self.inputs.upload(self.runtime, rank, owner), dirty))
        self.stats.upload_seconds += perf_counter() - start
        self.stats.input_bytes += len(dirty) * self.inputs.images[0].nbytes
        self.counts.zero_()
        self.counts[owner:owner + self.tp] = self.block
        start = perf_counter()
        self.runtime.run(self.graph, *self.arguments, config=self.config)
        self.stats.graph_seconds += perf_counter() - start
        start = perf_counter()
        ids = self.sampled[:candidates]
        handle = self.ranks[owner]["sampled_ids"]
        self.runtime.copy_from(ids.data_ptr(), handle.data_ptr, ids.nbytes, worker_id=owner)
        self.stats.sampled_id_bytes += ids.nbytes
        self.stats.readback_seconds += perf_counter() - start
        self.stats.calls += 1
        return ids[:, 0].tolist()
