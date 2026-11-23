"""Device-resident DSV4.1 Orthrus attention for the native A3 layer driver."""

import importlib
import hashlib
from dataclasses import dataclass
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
from .dsv41_publication import AtomicHeadPublisher, PublicationReceipt
from .dsv41_serving_online import DSV41CapturedLayer, DSV41ServingReplay
from .dsv41_suffix import DSV41SuffixPlan


def native(name):
    return importlib.import_module(f"models.deepseek_v4_1_flash_a3.{name}")


def _head_digest(values):
    """Stable digest of the unsharded suffix, identical on both nodes."""
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        if not torch.is_tensor(value) or value.device.type != "cpu" or not value.is_contiguous():
            raise ValueError("published draft tensors must be contiguous CPU tensors")
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ResidentHeadState:
    """Sharded execution weights already resident on their owning ranks.

    ``operands`` uses the same execution ABI as ``ResidentDraftAttention``.
    The optimizer supplies a deterministic digest computed from globally
    reduced update statistics; both hosts compare it before activating the
    inactive bank.  No updated tensor is copied through HOST memory.
    """

    operands: dict
    public_names: tuple[str, ...]
    digest: str

    def __post_init__(self):
        if not self.operands or any(
            not isinstance(key, tuple) or len(key) != 2 or not isinstance(value, dict)
            for key, value in self.operands.items()
        ):
            raise ValueError("resident head state must map rank/layer to execution operands")
        if not self.public_names or len(set(self.public_names)) != len(self.public_names):
            raise ValueError("resident head state requires the complete public parameter ABI")
        if len(self.digest) != 64 or any(character not in "0123456789abcdef" for character in self.digest):
            raise ValueError("resident head state requires a lowercase SHA256 digest")


class MutableDraftHeadBank:
    """Two resident mutable slots switched only at a verifier boundary."""

    def __init__(self, resident, rank_ids, initial, layers):
        required = {(rank, layer) for rank in rank_ids for layer in layers}
        if set(initial) != required:
            raise ValueError("mutable head initialization must cover each local rank and selected layer")
        for method in ("pin", "handle", "tensor"):
            if not hasattr(resident, method):
                raise TypeError(f"resident runtime lacks mutable draft method {method}")
        self.resident, self.rank_ids, self.layers = resident, dict(rank_ids), tuple(layers)
        self.host, self.handles = {}, {}
        for (rank, layer), operands in initial.items():
            local = self.rank_ids[rank]
            for slot in (0, 1):
                for name, value in operands.items():
                    host = value.detach().cpu().contiguous().clone()
                    resident.pin(
                        host, (local,), readonly=False,
                        name=f"online-draft/rank{rank}/layer{layer}/slot{slot}/{name}",
                    )
                    self.host[slot, rank, layer, name] = host
                    self.handles[slot, rank, layer, name] = resident.handle(host, local)
        self.active_slot, self.active_version = 0, 0
        self.staged = None
        self.previous = None

    def operands(self, rank, layer, *, slot=None):
        slot = self.active_slot if slot is None else slot
        names = (key[3] for key in self.host if key[:3] == (slot, rank, layer))
        return {name: self.handles[slot, rank, layer, name] for name in names}

    def stage(self, values, version, prepare):
        if self.staged is not None or self.previous is not None or version != self.active_version + 1:
            raise ValueError("draft publication must stage exactly the next inactive version")
        digest, slot = _head_digest(values), 1 - self.active_slot
        for layer in self.layers:
            prefix = f"layers.{layer}.attn."
            full = {name.removeprefix(prefix): value for name, value in values.items()
                    if name.startswith(prefix)}
            if not full:
                raise ValueError(f"published suffix has no layer {layer} tensors")
            for rank, local in self.rank_ids.items():
                operands = prepare(full, rank)
                expected = {key[3] for key in self.host if key[:3] == (slot, rank, layer)}
                if set(operands) != expected:
                    raise ValueError(f"published layer {layer} has a different execution ABI")
                for name, value in operands.items():
                    host = self.host[slot, rank, layer, name]
                    if value.shape != host.shape or value.dtype != host.dtype:
                        raise ValueError(f"published operand changed shape or dtype: layer {layer} {name}")
                    host.copy_(value)
                    self.handles[slot, rank, layer, name] = self.resident.tensor(host, local)
        self.staged = version, slot, digest
        return PublicationReceipt("prepared", version, digest, tuple(sorted(self.rank_ids)))

    def stage_resident(self, state, version):
        """Copy sharded device weights into the inactive execution bank."""
        if not isinstance(state, ResidentHeadState):
            raise TypeError("resident publication requires ResidentHeadState")
        if self.staged is not None or self.previous is not None or version != self.active_version + 1:
            raise ValueError("draft publication must stage exactly the next inactive version")
        expected_rows = {(rank, layer) for rank in self.rank_ids for layer in self.layers}
        if set(state.operands) != expected_rows:
            raise ValueError("resident publication does not cover every local rank and layer")
        slot = 1 - self.active_slot
        for (rank, layer), operands in state.operands.items():
            expected = {key[3] for key in self.handles if key[:3] == (slot, rank, layer)}
            if set(operands) != expected:
                raise ValueError(f"resident publication changed layer {layer} execution ABI")
            local = self.rank_ids[rank]
            for name, source in operands.items():
                destination = self.handles[slot, rank, layer, name]
                if source.shape != destination.shape or source.dtype != destination.dtype:
                    raise ValueError(f"resident operand changed shape or dtype: layer {layer} {name}")
                self.resident.copy_tensor(destination, source, local)
        self.staged = version, slot, state.digest
        return PublicationReceipt("prepared", version, state.digest, tuple(sorted(self.rank_ids)))

    def activate(self, version):
        if self.staged is None or self.staged[0] != version:
            raise ValueError("cannot activate an unstaged draft version")
        self.previous = self.active_slot, self.active_version
        _, self.active_slot, digest = self.staged
        self.active_version, self.staged = version, None
        return PublicationReceipt("activated", version, digest, tuple(sorted(self.rank_ids)))

    def commit(self, version):
        if self.active_version != version or self.previous is None:
            raise ValueError("cannot commit a draft version that is not reversibly active")
        self.previous = None

    def rollback(self, version):
        if self.staged is not None and self.staged[0] == version:
            self.staged = None
        elif self.active_version == version and self.previous is not None:
            self.active_slot, self.active_version = self.previous
            self.previous = None


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
        self.model_path, self.head_path = Path(model), Path(head)
        self.config, self.manifest, files = read_draft_artifact(self.model_path, self.head_path)
        self.files = files
        self.resident, self.topology = resident, topology
        self.joined_attention = joined_attention
        self.native = native_module
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.programs = {}
        self.compile_lock = Lock()
        self.operands = {}
        self.online_bank = None
        self._capture_request = None
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

    def arm_online_capture(self, requests, plan):
        """Capture exactly the next draft round for the selected request owners."""
        if self._capture_request is not None:
            raise RuntimeError("a draft replay capture is already armed")
        plan = plan if isinstance(plan, DSV41SuffixPlan) else DSV41SuffixPlan(plan)
        requests = {
            tuple(owner): (str(request_id), tuple(int(token) for token in prefix))
            for owner, (request_id, prefix) in requests.items()
        }
        if not requests or any(len(owner) != 3 or not prefix for owner, (_, prefix) in requests.items()):
            raise ValueError("online capture requires owner-to-request prefix mappings")
        self._capture_request = plan, requests

    def cancel_online_capture(self):
        self._capture_request = None

    def start_round(self, histories):
        if any(rank not in self.rank_ids for rank, _, _ in histories):
            raise ValueError("Draft history includes a rank assigned to a different node")
        capture, self._capture_request = self._capture_request, None
        return DraftRound(self, histories, capture=capture)

    def enable_online(self, last_layers):
        """Allocate two device-resident suffix banks without changing values."""
        if self.online_bank is not None:
            raise RuntimeError("online draft bank is already enabled")
        plan = DSV41SuffixPlan(last_layers)
        tp_lanes = sorted({rank % self.topology["tp"] for rank in self.rank_ids})
        initial = {}
        # Reload from the artifact paths retained by the constructor.  Keeping
        # only these selected host tensors bounds HOST memory for last-N.
        for layer in plan.layers:
            weights = read_draft_weights(self.head_path, self.manifest, self.files, layer)
            lanes = {lane: prepare_draft_operands(weights, self.config, tp_rank=lane)
                     for lane in tp_lanes}
            for rank in self.rank_ids:
                initial[rank, layer] = lanes[rank % self.topology["tp"]]
        self.online_bank = MutableDraftHeadBank(self.resident, self.rank_ids, initial, plan.layers)
        for rank in self.rank_ids:
            for layer in plan.layers:
                self.operands[rank, layer] = self.online_bank.operands(rank, layer)
        return plan

    def stage_online(self, learner, version):
        if self.online_bank is None:
            raise RuntimeError("online draft bank is not enabled")
        values = learner.publication_state()
        if isinstance(values, ResidentHeadState):
            return self.online_bank.stage_resident(values, version)
        return self.online_bank.stage(
            values, version,
            lambda weights, rank: prepare_draft_operands(
                weights, self.config, tp_rank=rank % self.topology["tp"],
            ),
        )

    def activate_online(self, version):
        receipt = self.online_bank.activate(version)
        self._refresh_online_operands()
        return receipt

    def _refresh_online_operands(self):
        for rank in self.rank_ids:
            for layer in self.online_bank.layers:
                self.operands[rank, layer] = self.online_bank.operands(rank, layer)

    def commit_online(self, version):
        self.online_bank.commit(version)

    def rollback_online(self, version):
        if self.online_bank is not None:
            self.online_bank.rollback(version)
            self._refresh_online_operands()

    def online_publisher(self, exchange):
        """Bind the generic two-node protocol to this resident draft bank."""
        if self.online_bank is None:
            raise RuntimeError("enable the online draft bank before creating its publisher")
        return AtomicHeadPublisher(
            exchange=exchange,
            stage=self.stage_online,
            activate=self.activate_online,
            rollback=self.rollback_online,
            commit=self.commit_online,
            current_version=self.online_bank.active_version,
            expected_ranks=range(self.topology["ep"]),
        )


class DraftRound:
    """Hold request-local Top-K only until the last consumer of each source."""

    def __init__(self, attention, histories, *, capture=None):
        self.attention = attention
        self.histories = histories
        self.owners = {(owner[0], owner[1]): owner for _, _, owner in histories}
        self.selection = {}
        self.capture = capture
        self.captured_attention = {}
        self.captured_tail = {}
        self.last_use = {
            role["index_source"]: layer for layer in range(attention.config["num_hidden_layers"])
            if (role := attention.native("cache").resolve_layer_role(layer))["index_source"] is not None
        }

    def finish_layer(self, layer):
        for key in tuple(self.selection):
            if self.last_use[key[-1]] == layer:
                del self.selection[key]
        keep = [
            (self.attention.rank_ids[key[0]], value)
            for key, (_, controls) in self.selection.items()
            for value in controls.values()
        ]
        if self.capture is not None and layer in self.capture[0].layers:
            seen = set()
            for value in (*self.captured_attention.values(), *self.captured_tail.values()):
                for tensor in self._tensors(value):
                    rank = value.get("rank")
                    key = rank, tensor.untyped_storage()._cdata
                    if key not in seen:
                        seen.add(key)
                        if rank is not None:
                            keep.append((self.attention.rank_ids[rank], tensor))
        return keep

    @staticmethod
    def _tensors(value):
        if torch.is_tensor(value):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from DraftRound._tensors(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from DraftRound._tensors(item)

    def observe_layer(self, layer, result):
        """Retain native mHC/MoE state before the layer driver releases scratch."""
        if self.capture is None or layer not in self.capture[0].layers:
            return
        requests = self.capture[1]
        for rank, cases in enumerate(result["states"]):
            if rank not in self.attention.rank_ids:
                continue
            for case, state in cases.items():
                record = state["original"]
                for item in record["request_token_map"]:
                    owner = self.owners[rank // self.attention.topology["tp"], item["request_slot"]]
                    if owner in requests:
                        self.captured_tail[rank, layer, owner] = dict(state, rank=rank, case=case)

    def replays(self):
        """Return backend-owned last-N state after all selected layers completed."""
        if self.capture is None:
            return {}
        plan, requests = self.capture
        replays = {}
        for owner, (request_id, prefix) in requests.items():
            layers = []
            for layer in plan.layers:
                attention, ffn = {}, {}
                for rank in self.attention.rank_ids:
                    key = rank, layer, owner
                    if key not in self.captured_attention or key not in self.captured_tail:
                        continue
                    a, state = self.captured_attention[key], self.captured_tail[key]
                    attention[rank] = dict(
                        a,
                        readonly_hidden=state["readonly_hidden"],
                        hidden=state["hidden"], incoming=state["incoming"],
                        pre=state["pre"], post=state["post"], comb=state["comb"],
                        collapsed=state["attn_collapsed"], normed=state["full_norm"],
                        changed=state["changed"],
                    )
                    ffn[rank] = {
                        "changed": state["changed"], "pre": state["ffn_pre"],
                        "post": state["ffn_post"], "comb": state["ffn_comb"],
                        "collapsed": state["ffn_collapsed"], "normed": state["ffn_normed"],
                        "router_input": state["router_input"], "router_logits": state["router_logits"],
                        "route_ids": state["route_ids"], "route_weights": state["weights"],
                        "shared": state["shared"], "expert_outputs": state["expert_outputs"],
                        "combined": state["combined"],
                        "output_hidden": state["padded_hidden"],
                        "output_pre_mix": state["ffn_pre"],
                    }
                if not attention:
                    raise RuntimeError(f"captured request has no local TP state at layer {layer}")
                layers.append(DSV41CapturedLayer(layer, attention, ffn))
            # A last-N replay must start at the real layer boundary.  The
            # normalized Attention input is sufficient only for last-1; an
            # earlier updated layer changes both streams consumed by every
            # downstream frozen Engram/mHC prelude.
            boundary = {
                rank: {
                    "hidden": state["readonly_hidden"],
                    "pre_mix": state["incoming"],
                }
                for (rank, layer, captured_owner), state in self.captured_tail.items()
                if layer == plan.first_layer and captured_owner == owner
            }
            expected_ranks = {
                rank for rank in self.attention.rank_ids
                if rank // self.attention.topology["tp"] == owner[0]
            }
            if set(boundary) != expected_ranks:
                raise RuntimeError("captured suffix boundary does not cover the owner's TP ranks")
            replays[owner] = DSV41ServingReplay(
                request_id, prefix, owner, plan, boundary, tuple(layers),
            )
        return replays

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
                               contribution[begin:end], record=record)

    def pack(self, rank, contribution, packed, physical, total):
        capacity, hidden = contribution.shape
        tp = self.attention.topology["tp"]
        program = lambda: self.attention.native("input_pack").make_sp_reduce_pack(capacity, hidden, tp)
        self.attention.execute(("reduce_pack", capacity), rank, program,
                               dict(value=contribution, packed=packed, physical=physical, active=total))

    def project_block(self, rank, layer, owner, hidden, cos, sin, contribution, *, record):
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
            final_lse = local_lse

        if compressed:
            long, = planes["long"]
            selected = self.index_rows(rank, layer, owner, history, values, workspace)
            context = workspace(output_shape, torch.float32)
            final_lse = workspace(lse_shape, torch.float32)
            sparse_args = dict(pages=long["handle"], rows=selected["physical_rows"],
                               lengths=selected["lengths"], bias=selected["bias"],
                               zeros=attention.constants[rank]["sparse_zero"], output=context,
                               lse=final_lse)
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
        if self.capture is not None and layer in self.capture[0].layers and owner in self.capture[1]:
            self.captured_attention[rank, layer, owner] = {
                "rank": rank, "record": record, "hidden": hidden, "incoming": None,
                "pre": None, "post": None, "comb": None, "collapsed": hidden,
                "cos": cos, "sin": sin, "history": history,
                "selection": selected if compressed else None,
                "qkv": values, "contribution": contribution, "lse": final_lse,
            }

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
