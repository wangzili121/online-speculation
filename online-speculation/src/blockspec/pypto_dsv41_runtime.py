"""Device-resident DSV4.1 Orthrus attention for the native A3 layer driver."""

import importlib
from pathlib import Path
from threading import Lock

import torch

from .pypto_dsv41_draft import (
    make_output_program,
    make_qkv_program,
    prepare_draft_operands,
    read_draft_artifact,
    read_draft_weights,
)
from .pypto_dsv41_indexer import INDEX_DIM, INDEX_HEADS, index_controls, make_indexer_program
from .pypto_dsv41_sparse import TILE, TOPK, make_joined_attention, make_sparse_attention
from .pypto_dsv41_swa import BLOCK, HEAD_DIM, WINDOW, make_swa_attention, swa_spans


def native(name):
    return importlib.import_module(f"models.deepseek_v4_1_flash_a3.{name}")


class ResidentDraftAttention:
    """Own static diffusion operands and share the native rank workers."""

    def __init__(self, model, head, resident, topology, output, *, native_module=native, local_ranks=None,
                 joined_attention=False):
        if topology["tp"] != 4:
            raise ValueError("DSV4.1 draft attention requires the verified TP4 shard geometry")
        ranks = tuple(range(topology["ep"]) if local_ranks is None else local_ranks)
        if (not ranks or len(ranks) != len(resident.devices) or len(set(ranks)) != len(ranks)
                or any(type(rank) is not int or not 0 <= rank < topology["ep"] for rank in ranks)):
            raise ValueError("Draft ranks must map one-to-one to the node's resident devices")
        self.rank_ids = {rank: local for local, rank in enumerate(ranks)}
        self.config, self.manifest, files = read_draft_artifact(Path(model), Path(head))
        self.resident, self.topology = resident, topology
        self.joined_attention = joined_attention
        self.native = native_module
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.programs = {}
        self.compile_lock = Lock()
        self.operands = {}
        self.max_context = self.native("decode_network").MAX_CONTEXT
        self.heads = self.config["num_attention_heads"] // topology["tp"]
        for layer, declared in enumerate(self.manifest["layers"]):
            role = self.native("cache").resolve_layer_role(layer)
            fields = ("ratio", "kv_source", "index_source")
            if role["index_source"] == layer:
                fields += ("candidate_source",)
            if any(declared[key] != role[key] for key in fields):
                raise ValueError(f"Incompatible diffusion head role at layer {layer}")
        tp_lanes = sorted({rank % topology["tp"] for rank in ranks})
        load_weight = getattr(resident, "draft_weight", resident.weight)
        for layer in range(len(self.manifest["layers"])):
            weights = read_draft_weights(Path(head), self.manifest, files, layer)
            lanes = {lane: prepare_draft_operands(weights, self.config, tp_rank=lane)
                     for lane in tp_lanes}
            for rank, local in self.rank_ids.items():
                self.operands[rank, layer] = {
                    name: load_weight(value, local)
                    for name, value in lanes[rank % topology["tp"]].items()
                }
        constants = {
            "swa_zero": torch.zeros((WINDOW, HEAD_DIM), dtype=torch.bfloat16),
            "sparse_zero": torch.zeros((TILE, HEAD_DIM), dtype=torch.bfloat16),
            **{f"index_zero_{ratio}": torch.zeros((WINDOW // ratio, INDEX_DIM), dtype=torch.int8)
               for ratio in (1, 2)},
        }
        self.constants = {
            rank: {name: load_weight(value, local) for name, value in constants.items()}
            for rank, local in self.rank_ids.items()
        }

    @property
    def mask_token_id(self):
        return self.manifest["mask_token_id"]

    def execute(self, key, rank, factory, inputs):
        from pypto.backend import BackendType
        from pypto.runtime import ExecutionMode, RunConfig

        with self.compile_lock:
            if key not in self.programs:
                kernel = factory()
                options = RunConfig(
                    arch=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD, save_kernels=True,
                    save_kernels_dir=str(self.output / f"entry-{len(self.programs):03d}"),
                )
                # The compiler consumes static shape/dtype metadata here. Physical
                # target aliases are bound only at execution.
                arguments = [
                    value if isinstance(value, torch.Tensor) or not hasattr(value, "shape")
                    else torch.empty(value.shape, dtype=value.dtype, device="meta")
                    for value in (inputs[name] for name in kernel.param_names)
                ]
                compiled = kernel.compile(*arguments, config=options)
                self.programs[key] = compiled, self.native("network").network_parameter_abi(compiled)
        compiled, abi = self.programs[key]
        self.resident.execute(compiled, self.rank_ids[rank], [inputs[name] for name in compiled.param_names], abi)

    def start_round(self, histories):
        if any(rank not in self.rank_ids for rank, _, _ in histories):
            raise ValueError("Draft history includes a rank assigned to a different node")
        return DraftRound(self, histories)


class DraftRound:
    """Hold request-local Top-K only until the last consumer of each source."""

    def __init__(self, attention, histories):
        self.attention = attention
        self.histories = histories
        self.owners = {(owner[0], owner[1]): owner for _, _, owner in histories}
        self.selection = {}
        self.last_use = {
            role["index_source"]: layer for layer in range(attention.config["num_hidden_layers"])
            if (role := attention.native("cache").resolve_layer_role(layer))["index_source"] is not None
        }

    def finish_layer(self, layer):
        for key in tuple(self.selection):
            if self.last_use[key[-1]] == layer:
                del self.selection[key]
        return [
            (self.attention.rank_ids[key[0]], value)
            for key, (_, controls) in self.selection.items()
            for value in controls.values()
        ]

    def project_layer(self, rank, layer, record, hidden, cos, sin, contribution):
        dp = rank // self.attention.topology["tp"]
        for item in record["request_token_map"]:
            begin, end = item["token_begin"], item["token_end"]
            owner = self.owners[dp, item["request_slot"]]
            if end - begin != BLOCK:
                raise ValueError("Orthrus expects a complete K32 block for each scheduler request")
            if self.histories[rank, layer, owner]["committed_tokens"] != item["initial_context_length"]:
                raise ValueError("Draft block and committed history identify different anchors")
            self.project_block(rank, layer, owner, hidden[begin:end], cos[begin:end], sin[begin:end],
                               contribution[begin:end])

    def pack(self, rank, contribution, packed, physical, total):
        capacity, hidden = contribution.shape
        tp = self.attention.topology["tp"]
        program = lambda: self.attention.native("input_pack").make_sp_reduce_pack(capacity, hidden, tp)
        self.attention.execute(("reduce_pack", capacity), rank, program,
                               dict(value=contribution, packed=packed, physical=physical, active=total))

    def project_block(self, rank, layer, owner, hidden, cos, sin, contribution):
        attention = self.attention
        runtime = attention.resident
        history = self.histories[rank, layer, owner]
        anchor, role, planes = history["committed_tokens"], history["role"], history["planes"]
        heads, config = attention.heads, attention.config
        width, latent = heads * HEAD_DIM, config["q_lora_rank"]

        def workspace(shape, dtype=torch.bfloat16):
            return runtime.workspace(shape, dtype, attention.rank_ids[rank])

        operands = attention.operands[rank, layer]
        values = dict(operands, hidden=hidden, cos=cos, sin=sin, active=BLOCK)
        for name, columns in (
            ("qa", latent), ("latent", latent), ("qb", width), ("query", width),
            ("kv", HEAD_DIM), ("normalized_kv", HEAD_DIM), ("temporary", HEAD_DIM),
        ):
            values[name] = workspace((BLOCK, columns))
        attention.execute("qkv", rank, lambda: make_qkv_program(attention.native, config), values)

        swa, = planes["swa"]
        if (swa["shape"][1:] != (WINDOW, 1, HEAD_DIM)
                or swa["handle"].shape != (swa["shape"][0] * swa["stride_rows"], HEAD_DIM)
                or swa["handle"].dtype != torch.bfloat16):
            raise ValueError("Invalid resident SWA page plane")
        bias = torch.full((1, WINDOW), -torch.inf, dtype=torch.float32)
        bias[:, :min(anchor, WINDOW)] = 0
        spans = swa_spans(swa["pages"], anchor, swa["shape"][0], stride_rows=swa["stride_rows"],
                          offset_rows=swa["offset_rows"])
        output_shape = (BLOCK * heads, HEAD_DIM)
        compressed = role["index_source"] is not None
        lse_shape = (BLOCK * heads, 1)
        swa_args = dict(query=values["query"], spans=torch.tensor(spans, dtype=torch.int32),
                        temporary=values["temporary"], history_bias=bias, sink=operands["sink"])
        swa_pages = swa["handle"]
        swa_zeros = attention.constants[rank]["swa_zero"]
        joined = compressed and attention.joined_attention
        if not joined:
            local = workspace(output_shape, torch.float32)
            local_lse = workspace(lse_shape, torch.float32)
            attention.execute(("swa", swa_pages.shape[0]), rank,
                              lambda: make_swa_attention(heads, token_major_query=True),
                              dict(swa_args, pages=swa_pages, zeros=swa_zeros, output=local, lse=local_lse))
            context = local

        if compressed:
            long, = planes["long"]
            selected = self.index_rows(rank, layer, owner, history, values, workspace)
            context = workspace(output_shape, torch.float32)
            sparse_args = dict(pages=long["handle"], rows=selected["physical_rows"],
                               lengths=selected["lengths"], bias=selected["bias"],
                               zeros=attention.constants[rank]["sparse_zero"], output=context,
                               lse=workspace(lse_shape, torch.float32))
            flat_query = values["query"].view(-1, HEAD_DIM)
            if joined:
                attention.execute(("joined_attention", swa_pages.shape[0], long["handle"].shape[0]), rank,
                                  make_joined_attention,
                                  dict(swa_args, **sparse_args, swa_pages=swa_pages, swa_zeros=swa_zeros,
                                       query_flat=flat_query))
            else:
                attention.execute(("sparse", long["handle"].shape[0]), rank,
                                  lambda: make_sparse_attention(head_major_swa=True),
                                  dict(sparse_args, query=flat_query, local_output=local, local_lse=local_lse))

        values.update(attention=context, context=workspace((BLOCK, width)), unrotated=workspace((BLOCK, width)),
                      grouped=workspace((BLOCK, operands["wo_b"].shape[0])), contribution=contribution)
        attention.execute(("output", compressed), rank,
                          lambda: make_output_program(attention.native, config, head_major_input=not compressed), values)

    def index_rows(self, rank, layer, owner, history, values, workspace):
        attention = self.attention
        config, role = attention.config, history["role"]
        kv, = history["planes"]["long"]
        source = role["index_source"]
        key = rank, owner, source
        identity = (role["kv_source"], history["committed_tokens"], kv["pages"],
                    kv["stride_rows"], kv["offset_rows"])
        if source != layer:
            prior = self.selection.get(key)
            if prior is None or prior[0] != identity:
                raise ValueError(f"Draft layer {layer} has no matching Top-K from layer {source}")
            return prior[1]
        geometry, controls = index_controls(
            history, candidate_blocks=config["candidate_topk_blocks"],
            candidate_block_size=config["candidate_block_size"], max_context=attention.max_context,
        )
        codes, scales = history["planes"]["index"]
        outputs = dict(physical_rows=workspace((BLOCK, TOPK), torch.int32),
                       lengths=workspace((BLOCK,), torch.int32),
                       bias=workspace((BLOCK, TOPK), torch.float32))
        shapes = {
            "idx_projected": ((BLOCK, INDEX_HEADS * INDEX_DIM), torch.bfloat16),
            "idx_raw_weights": ((BLOCK, INDEX_HEADS), torch.bfloat16),
            "idx_query": ((BLOCK, INDEX_HEADS, INDEX_DIM), torch.bfloat16),
            "idx_scaled_weights": ((BLOCK, INDEX_HEADS), torch.float16),
            "index_keys": ((geometry.bucket, INDEX_DIM), torch.int8),
            "key_scales": ((geometry.bucket,), torch.float16),
            "query_codes": ((BLOCK, INDEX_HEADS, INDEX_DIM), torch.int8),
            "query_scales": ((BLOCK, INDEX_HEADS), torch.float16),
            "index_scores": ((BLOCK, geometry.bucket), torch.float32),
            "logical_rows": ((BLOCK, TOPK), torch.int32),
        }
        bindings = dict(values, **controls, **outputs, index_codes=codes["handle"], index_scales=scales["handle"],
                        zero_codes=attention.constants[rank][f"index_zero_{geometry.ratio}"])
        bindings.update({name: workspace(shape, dtype) for name, (shape, dtype) in shapes.items()})
        attention.execute(("indexer", geometry), rank, lambda: make_indexer_program(attention.native, config, geometry), bindings)
        if self.last_use[layer] > layer:
            self.selection[key] = identity, outputs
        return outputs
