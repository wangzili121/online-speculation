"""Bind a resident Orthrus draft graph to a native A3 generation session."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from .dsv41_packed_attention_tape import (
    packed_attention_tape_shapes, packed_core_tape_shapes, packed_qkv_tape_shapes, packed_wo_tape_shapes,
)
from .dsv41_packed_ffn_tape import packed_ffn_tape_shapes
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


def draft_geometry(context, request_slots=1):
    if not 1 <= context <= INDEX_CAPACITIES[-1]:
        raise ValueError("Packed draft context capacity must be in 1..16384")
    from models.deepseek_v4_1_flash_a3.config import BLOCK_SIZE

    pages = ((context + BLOCK_SIZE - 1) // BLOCK_SIZE) * request_slots + 1
    buckets = tuple(next(size for size in INDEX_CAPACITIES if size >= context // ratio) for ratio in (1, 2))
    return pages, *buckets


def draft_graph_manifest(context, block, *, last_layers=0, local_ranks=None, request_slots=1):
    from models.deepseek_v4_1_flash_a3 import config as native

    sources = ("pypto_dsv41_packed_graph.py", "pypto_dsv41_packed_draft_attention.py",
               "pypto_dsv41_packed_swa.py", "pypto_dsv41_packed_history.py", "pypto_dsv41_packed_indexer.py",
               "pypto_dsv41_draft.py", "pypto_dsv41_sparse.py", "pypto_dsv41_swa.py", "dsv41_packed_ffn_tape.py",
               "dsv41_packed_attention_tape.py")
    native_root = Path(native.__file__).parent
    native_sources = ("config.py", "decode_fwd.py", "decode_indexer.py", "engram.py",
                      "gemm.py", "hc.py", "lm_head.py", "moe.py", "o_proj.py", "rope.py")
    deployment = {}
    if local_ranks is not None:
        sources += ("dsv41_packed_decode.py", "dsv41_packed_ep_graph.py", "dsv41_packed_ep_hccl.py")
        deployment = dict(local_ranks=local_ranks, request_slots=request_slots, moe_tokens=16)
    return dict(
        schema=4, context=context, block=block, last_layers=last_layers,
        geometry=list(draft_geometry(context, request_slots)),
        weight_layout=native.WEIGHT_LAYOUT_VERSION, cache_layout=native.CACHE_LAYOUT_VERSION,
        topology=dict(dp=native.DP_SIZE, tp=native.TP_SIZE, ep=native.EP_SIZE),
        sources={name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources},
        native_sources={name: sha256((native_root / name).read_bytes()).hexdigest() for name in native_sources},
        **deployment,
    )


def load_packed_draft_graph(directory, config, *, context, block=32, last_layers=0):
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram

    manifest = json.loads((directory / "orthrus_draft_manifest.json").read_text())
    if block not in (16, 32) or manifest != draft_graph_manifest(context, block, last_layers=last_layers):
        raise ValueError(f"Compiled draft graph does not match K{block}, last-{last_layers}, context or model source")
    return DistributedCompiledProgram.from_dir(
        directory, backend_type=config.arch, execution_mode=config.execution_mode,
        distributed_config=config.distributed_config,
    )


_STAGED_ENTRIES = ("attention", "ffn-restore", "head", "ep-prepare", "ep-compute", "ep-restore")


def load_staged_draft_graphs(directory, *, context, block, local_ranks, request_slots):
    from pypto.ir import DistributedCompiledProgram

    manifest = json.loads((directory / "orthrus_draft_manifest.json").read_text())
    expected = draft_graph_manifest(context, block, local_ranks=local_ranks, request_slots=request_slots)
    if manifest != expected:
        raise ValueError("Staged draft bundle does not match the head width, source or resident page geometry")
    return tuple(DistributedCompiledProgram.from_dir(directory / entry) for entry in _STAGED_ENTRIES)


def read_packed_head(checkpoint, head, block):
    """Validate the head and its intended draft width before resident loading."""
    config, manifest, files = read_draft_artifact(checkpoint.root, head)
    if manifest["block_size"] != block and not (
        block == 16 and manifest["artifact_role"] == "initialized-plumbing"
    ):
        raise ValueError("Changing the trained draft head's block width requires a compatible artifact")
    return config, manifest, files


def _resident_head(runtime, head, world, tp, artifact):
    """Stream the exchange artifact into TP-sharded device banks once."""
    from models.deepseek_v4_1_flash_a3.weights import bf16_nz_shape

    config, manifest, files = artifact
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
            runtime.copy_to(bank.data_ptr, payload.data_ptr(), row_bytes,
                            dst_offset=slot * row_bytes, worker_id=rank)
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


@dataclass(frozen=True)
class DraftRequest:
    anchor: int
    position: int
    owner: int
    candidates: int
    history_key: object
    pages: tuple | None = None


@dataclass(frozen=True)
class PackedFinalState:
    """The draft's final in-place residual streams and delayed pre-mix."""

    allocation: object
    streams: object
    premix: object


@dataclass(frozen=True)
class PackedStudentRows:
    """Full-vocabulary rows on the rank that owns one draft request."""

    owner: int
    position: int
    rows: int
    logits: object
    final_state: PackedFinalState | None = None
    ffn_tape: dict | None = None
    pages: tuple[int, ...] | None = None


class PackedDraftProposer:
    """Dispatch one request per TP group while all ranks serve frozen experts."""

    def __init__(self, runtime, graph, config, target_ranks, decode_inputs, weights, engram,
                 rope_tables, counts, *, draft_head, block, context, tp, head_artifact=None,
                 retain_final_state=False, last_layers=0):
        self.graph, self.config, self.counts = graph, config, counts
        self._bind_inputs(runtime, target_ranks, decode_inputs, weights, engram, rope_tables,
                          draft_head=draft_head, block=block, context=context, tp=tp,
                          head_artifact=head_artifact, retain_final_state=retain_final_state,
                          last_layers=last_layers)
        self.arguments = []
        self._input_arguments = {}
        self._tape_arguments = {}
        for index, parameter in enumerate(graph._get_metadata()[0]):
            name = parameter.name.removesuffix("__ssa_v0")
            if name == "ffn_capture_owner":
                self._capture_owner_argument = index
                argument = -1
            else:
                argument = counts if name == "moe_num_tokens" else self._rank_argument(name, parameter)
            self.arguments.append(argument)
            if name == "logits":
                self._logits_argument = index, parameter
            elif name in self.inputs.fields[0]:
                self._input_arguments[name] = index, parameter
            elif name in self.tape_specs:
                self._tape_arguments[name] = index, parameter

    def _bind_inputs(self, runtime, target_ranks, decode_inputs, weights, engram, rope_tables, *,
                     draft_head, block, context, tp, head_artifact=None, retain_final_state=False,
                     last_layers=0, request_slots=1):
        import torch
        from models.deepseek_v4_1_flash_a3.config import BLOCK_SIZE, FLASH
        from models.deepseek_v4_1_flash_a3.engram import ENGRAM_K
        from models.deepseek_v4_1_flash_a3.lm_head import MAX_LOGIT_ROWS, SAMPLED_IDS_PAD
        from models.deepseek_v4_1_flash_a3.replay import DecodeInputs
        from pypto.runtime import DeviceTensor

        self.runtime = runtime
        self.weights, self.engram, self.rope_tables = weights, engram, rope_tables
        self.tp, self.block = tp, block
        self.world = len(target_ranks)
        self.pages, *self.buckets = draft_geometry(context, request_slots)
        self.page_tables = tuple(decode_inputs.fields[0][key][0] for key in
                                 ("c1a_index_block_table", "c2a_index_block_table"))
        self.sampled = torch.empty((MAX_LOGIT_ROWS, SAMPLED_IDS_PAD), dtype=torch.int32)
        self.stats = DraftRoundStats()
        self.previous_owners = set()
        self.student = {}
        self.retain_final_state = retain_final_state
        self.last_layers = last_layers
        self.tape_specs = {
            **packed_ffn_tape_shapes(block, last_layers, tp, FLASH.hidden_size, FLASH.hc_mult),
            **packed_attention_tape_shapes(block, last_layers, FLASH.hidden_size, FLASH.hc_mult),
            **packed_wo_tape_shapes(block, last_layers, FLASH, tp),
            **packed_core_tape_shapes(block, last_layers, FLASH, tp),
            **packed_qkv_tape_shapes(block, last_layers, FLASH),
        }
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
        if head_artifact is None:
            head_artifact = read_packed_head(weights.checkpoint, draft_head, block)
        device_head, manifest, self.resident_head_bytes = _resident_head(runtime, draft_head, self.world, tp, head_artifact)
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
            bound.update(self._alloc_tape(rank))
            self.inputs.upload(runtime, rank, owner=-tp)
            self.ranks.append(bound)

    def _alloc_tape(self, rank):
        import torch

        return {name: self.runtime.alloc_tensor(shape, getattr(torch, dtype), worker_id=rank)
                for name, (shape, dtype) in self.tape_specs.items()}

    def _rank_argument(self, name, parameter, *, replacement=None):
        from models.deepseek_v4_1_flash_a3.device_views import resident_graph_argument

        replacement = replacement or {}
        shards = [replacement.get(index, operands[name]) for index, operands in enumerate(self.ranks)]
        return resident_graph_argument(shards, tuple(range(self.world)), parameter)

    def _stage(self, request, *, student_rows=0):
        import torch

        anchor_token, position = request.anchor, request.position
        active, selected = self.inputs.fields[1:]
        indices = (anchor_token, self.mask_token_id)
        embeddings = self.weights.lookup_rows(torch.tensor(indices, dtype=torch.int64)).float()
        provisional = torch.full((self.block,), self.mask_token_id, dtype=torch.int64)
        provisional[0] = anchor_token
        history = self.engram.history.get(request.history_key)
        try:
            engram_rows = self.engram.lookup(provisional, self.block, position, request_id=request.history_key)
        finally:
            if history is None:
                self.engram.history.pop(request.history_key, None)
            else:
                self.engram.history[request.history_key] = history
        page_table, c2a_table = self.page_tables if request.pages is None else (request.pages, request.pages)
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
        rows = max(request.candidates, student_rows)
        chosen = selected["logit_row_indices"][:rows]
        chosen.copy_(torch.arange(rows, dtype=torch.int32))

    def propose(self, anchor_token, *, position, owner, candidates, uploads):
        request = DraftRequest(anchor_token, position, owner, candidates, history_key=owner)
        return self.propose_batch((request,), uploads=uploads)[owner]

    def propose_batch(self, requests, *, uploads, capture=None):
        """Upload distinct request controls, then dispatch one draft graph for the wave."""
        owners = {request.owner for request in requests}
        if len(owners) != len(requests) or any(not 0 <= owner < self.world or owner % self.tp for owner in owners):
            raise ValueError("Draft batch requires a unique TP owner per request")
        if any(not 0 < row.candidates < min(self.block, self.sampled.shape[0] + 1) for row in requests):
            raise ValueError("Draft candidate count must fit the block and LM head")
        if any(row.position < 0 or row.position + self.block > self.context for row in requests):
            raise ValueError("Draft block exceeds the context capacity")
        capture_request, capture_rows = None, 0
        if capture is not None:
            key, capture_rows = capture
            capture_request = next((row for row in requests if row.history_key == key), None)
            if (capture_request is None or key in self.student
                    or type(capture_rows) is not int
                    or not 1 <= capture_rows <= min(self.block - 1, self.sampled.shape[0])):
                raise ValueError("Student capture requires live draft rows and a free request capture")
        if not requests:
            return {}

        def upload_group(owner, active, allocation=None):
            before = perf_counter()
            def upload(rank):
                if rank == owner and allocation is not None:
                    return self.inputs.upload(self.runtime, rank, active, allocation=allocation)
                return self.inputs.upload(self.runtime, rank, active)

            list(uploads.map(upload, range(owner, owner + self.tp)))
            self.stats.upload_seconds += perf_counter() - before
            self.stats.input_bytes += self.tp * self.inputs.images[0].nbytes

        arguments = self.arguments
        allocation = None
        if capture_request is not None:
            owner = capture_request.owner
            output = self.ranks[owner]["logits"]
            retained = self.runtime.alloc_tensor(output.shape, output.dtype, worker_id=owner)
            state = None
            tape = None
            index, parameter = self._logits_argument
            arguments = list(self.arguments)
            arguments[index] = self._rank_argument("logits", parameter, replacement={owner: retained})
            if self.retain_final_state:
                allocation, views = self.inputs.reserve(self.runtime, owner)
                state = PackedFinalState(allocation, views["x_hc"], views["pre_mix"])
                for name, (index, parameter) in self._input_arguments.items():
                    arguments[index] = self._rank_argument(name, parameter, replacement={owner: views[name]})
            if self.last_layers:
                tape = {rank: self._alloc_tape(rank) for rank in range(owner, owner + self.tp)}
                arguments[self._capture_owner_argument] = owner
                for name, (index, parameter) in self._tape_arguments.items():
                    arguments[index] = self._rank_argument(name, parameter,
                                                           replacement={rank: record[name] for rank, record in tape.items()})
            pages = capture_request.pages
            if pages is None:
                pages = tuple(map(int, self.page_tables[0]))
            self.student[key] = PackedStudentRows(owner, capture_request.position, capture_rows, retained, state, tape, pages)

        for request in requests:
            start = perf_counter()
            self._stage(request, student_rows=capture_rows if request is capture_request else 0)
            self.stats.prepare_seconds += perf_counter() - start
            upload_group(request.owner, request.owner, allocation if request is capture_request else None)
        for owner in sorted(self.previous_owners - owners):
            upload_group(owner, -self.tp)
        self.previous_owners = owners
        saved_counts = self.counts.clone()
        self.counts.zero_()
        for owner in owners:
            self.counts[owner:owner + self.tp] = self.block
        start = perf_counter()
        try:
            self.runtime.run(self.graph, *arguments, config=self.config)
            self.stats.graph_seconds += perf_counter() - start
        finally:
            self.counts.copy_(saved_counts)
        start = perf_counter()
        result = {}
        for request in requests:
            ids = self.sampled[:request.candidates]
            handle = self.ranks[request.owner]["sampled_ids"]
            self.runtime.copy_from(ids.data_ptr(), handle.data_ptr, ids.nbytes, worker_id=request.owner)
            self.stats.sampled_id_bytes += ids.nbytes
            result[request.owner] = ids[:, 0].tolist()
        self.stats.readback_seconds += perf_counter() - start
        self.stats.calls += 1
        return result

    def release_student(self, history_key):
        """Return a consumed or cancelled student capture to the device allocator."""
        value = self.student.get(history_key)
        if value is not None:
            self.runtime.free_tensor(value.logits, worker_id=value.owner)
            if value.final_state is not None:
                self.runtime.free_tensor(value.final_state.allocation, worker_id=value.owner)
            for rank, record in (value.ffn_tape or {}).items():
                for tensor in record.values():
                    self.runtime.free_tensor(tensor, worker_id=rank)
            del self.student[history_key]


class StagedPackedDraftProposer(PackedDraftProposer):
    """Execute local draft Attention and send routed expert rows over EP16."""

    def __init__(self, target, programs, *, draft_head, head_artifact, block):
        import torch
        from models.deepseek_v4_1_flash_a3.config import DP_SIZE, FLASH

        from .dsv41_packed_decode import ResidentDecodeFFN
        from .dsv41_packed_ep_graph import GraphResidentHcclMoE

        self.target = target
        self.dp_size = DP_SIZE
        slots, context, _ = target.geometry
        geometry = target.graphs
        self._bind_inputs(
            target.runtime, target.handles[0], target.inputs[1], geometry.weights, geometry.engram,
            geometry.rope_tables, draft_head=draft_head, block=block, context=context, tp=target.tp,
            head_artifact=head_artifact, request_slots=slots,
        )
        self.attention, restore, self.head, *ep = programs
        transport = target.decode.ffn.moe.transport
        moe = GraphResidentHcclMoE(self.runtime, transport, {"decode": tuple(ep)}, tokens=16, rows=block)
        self.ffn = ResidentDecodeFFN(moe, restore)
        self.layer_weights = target.decode.layer_weights
        scratch = {
            "selected": ((block, FLASH.index_topk), torch.int32),
            "lengths": ((block,), torch.int32),
            "mask": ((block, FLASH.index_topk), torch.float32),
            "h_attention": ((block, FLASH.hc_mult, FLASH.hidden_size), torch.float32),
            "pre_attention": ((block, FLASH.hc_mult), torch.float32),
            "ffn_input": ((block, FLASH.hidden_size), torch.bfloat16),
        }
        for rank, values in enumerate(self.ranks):
            values.update({name: self.runtime.alloc_tensor(shape, dtype, worker_id=rank)
                           for name, (shape, dtype) in scratch.items()})

        self.attention_args = self._bind_stage(self.attention, ("layer_index", "ffn_capture_owner", "execute_layers"),
                                               count_meta=moe.meta)
        self.head_args = self._bind_stage(self.head, ("execute_layers",))
        self.attention_rows = self._stack("h_attention")
        self.ffn_rows = self._stack("ffn_input")
        self.x_hc = self._stack("x_hc")
        self.pre_mix = self._stack("pre_mix")
        self.runtime.run(self.attention, *self.attention_args, 0, -1, 0)
        self.ffn.reserve_windows(self.attention_rows, self.layer_weights[0][2], self.x_hc, self.pre_mix)
        self.runtime.run(self.head, *self.head_args, 0)

    def _stack(self, name):
        from pypto.runtime import StackedDeviceTensor

        shards = tuple(values[name] for values in self.ranks)
        return StackedDeviceTensor(shards, (self.world, *shards[0].shape), tuple(range(self.world)))

    def _bind_stage(self, program, scalars, **supplied):
        arguments, runtime_scalars = [], []
        for parameter in program._get_metadata()[0]:
            name = parameter.name.removesuffix("__ssa_v0")
            if name in scalars:
                runtime_scalars.append(name)
            elif name in supplied:
                arguments.append(supplied[name])
            else:
                arguments.append(self._rank_argument(name, parameter))
        if tuple(runtime_scalars) != scalars:
            raise ValueError(f"Unexpected staged draft scalar arguments: {runtime_scalars}")
        return tuple(arguments)

    def propose_batch(self, requests, *, uploads, capture=None):
        from .dsv41_online_profile import timed_stage

        if capture is not None:
            raise ValueError("Staged draft capture is not bound to an online learner")
        if not requests:
            return {}
        local = {request.owner - self.target.rank_offset: request for request in requests
                 if request.owner in self.target.global_ranks}
        counts = [0] * self.dp_size
        for request in requests:
            counts[request.owner // self.tp] = self.block

        def upload(owner, active):
            start = perf_counter()
            list(uploads.map(lambda rank: self.inputs.upload(self.runtime, rank, active),
                             range(owner, owner + self.tp)))
            self.stats.upload_seconds += perf_counter() - start
            self.stats.input_bytes += self.tp * self.inputs.images[0].nbytes

        for owner, request in local.items():
            start = perf_counter()
            self._stage(request)
            self.stats.prepare_seconds += perf_counter() - start
            upload(owner, owner)
        for owner in sorted(self.previous_owners - local.keys()):
            upload(owner, -self.tp)
        self.previous_owners = set(local)

        stage = f"staged-draft/{self.stats.calls}"
        start = perf_counter()
        self.ffn.moe.bind_batch(counts)
        for layer, (shared, routed, mixes) in enumerate(self.layer_weights):
            with timed_stage("draft_attention"):
                self.runtime.run(self.attention, *self.attention_args, layer, -1, 1)
            self.ffn.run_layer(f"{stage}/layer-{layer}", attention=self.attention_rows, ffn_input=self.ffn_rows,
                               shared_weights=shared, routed_weights=routed, mixes=mixes,
                               output=self.x_hc, pre_mix=self.pre_mix)
        with timed_stage("draft_final_head"):
            self.runtime.run(self.head, *self.head_args, 1)
        self.stats.graph_seconds += perf_counter() - start

        start = perf_counter()
        result = {}
        for owner, request in local.items():
            ids = self.sampled[:request.candidates]
            handle = self.ranks[owner]["sampled_ids"]
            self.runtime.copy_from(ids.data_ptr(), handle.data_ptr, ids.nbytes, worker_id=owner)
            self.stats.sampled_id_bytes += ids.nbytes
            result[str(request.owner)] = ids[:, 0].tolist()
        self.stats.readback_seconds += perf_counter() - start
        peer = self.target.link.exchange(f"{stage}/sampled", result)
        expected = {str(request.owner): request.candidates for request in requests
                    if request.owner not in self.target.global_ranks}
        if {owner: len(tokens) for owner, tokens in peer.items()} != expected:
            raise ValueError("Draft owners returned a different set of candidate rows")
        self.stats.calls += 1
        return {int(owner): ids for owner, ids in (result | peer).items()}
