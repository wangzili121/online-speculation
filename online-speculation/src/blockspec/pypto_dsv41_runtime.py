"""Device-resident DSV4.1 Orthrus attention for the native A3 layer driver."""

import importlib
import hashlib
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, get_native_id
from time import monotonic_ns, perf_counter

import torch

from .pypto_dsv41_draft import (
    make_draft_attention_reduce_narrow,
    make_draft_attention_reduce_pack,
    make_output_program,
    make_qkv_program,
    prepare_draft_operands,
    read_draft_artifact,
    read_draft_weights,
)
from .pypto_dsv41_ep_vjp import ROUTES, route_bucket_width
from .pypto_dsv41_indexer import INDEX_DIM, INDEX_HEADS, index_controls, make_indexer_program
from .pypto_dsv41_resident_abi import compile_resident_entry, kernel_cache_options, resident_entry
from .pypto_dsv41_sparse import TILE, TOPK, make_joined_attention, make_sparse_attention
from .pypto_dsv41_swa import BLOCK, HEAD_DIM, WINDOW, make_swa_attention, swa_spans
from .dsv41_publication import AtomicHeadPublisher, PublicationReceipt
from .dsv41_online_profile import active_profile
from .dsv41_serving_online import DSV41CapturedLayer, DSV41SequenceSpan, DSV41ServingReplay
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

    ``operands`` uses the same execution ABI as ``ResidentDraftAttention`` but
    stores the resident runtime's owning Torch identities, not raw device
    pointers.  ``ResidentNetworkRuntime.copy_tensor`` resolves those identities
    to a device-to-device copy.
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
        copies = {local: [] for local in self.rank_ids.values()}
        for (rank, layer), operands in state.operands.items():
            expected = {key[3] for key in self.handles if key[:3] == (slot, rank, layer)}
            if set(operands) != expected:
                raise ValueError(f"resident publication changed layer {layer} execution ABI")
            local = self.rank_ids[rank]
            for name, source in operands.items():
                destination = self.host[slot, rank, layer, name]
                if source.shape != destination.shape or source.dtype != destination.dtype:
                    raise ValueError(f"resident operand changed shape or dtype: layer {layer} {name}")
                copies[local].append((destination, source))

        def copy_rank(local):
            for destination, source in copies.get(local, ()):
                self.resident.copy_tensor(destination, source, local)

        self.resident.map_ranks(copy_rank)
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
                 joined_attention=True, block_size=BLOCK):
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
        if block_size not in (8, 16, BLOCK):
            raise ValueError("Native draft requires block width 8, 16 or 32")
        self.joined_attention, self.block_size = joined_attention, block_size
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
        positions = torch.arange(WINDOW)
        visible = torch.arange(WINDOW + 1)
        mask = torch.zeros((WINDOW + 1, WINDOW), dtype=torch.float32)
        mask.masked_fill_(positions[None, :] >= visible[:, None], -torch.inf)
        resident.pin(mask, self.rank_ids.values(), readonly=True, name="orthrus/swa-history-bias")
        self.swa_history_bias = tuple(mask[i:i + 1] for i in range(WINDOW + 1))
        self.reuse_swa_controls = True
        self.clear_index_scratch = False

    @property
    def mask_token_id(self):
        return self.manifest["mask_token_id"]

    def _compile_program(self, key, factory, inputs, timing):
        from pypto import cache_stats
        from pypto.backend import BackendType
        from pypto.runtime import ExecutionMode, RunConfig

        online = isinstance(key, tuple) and key[0] == "online"
        cache_options = kernel_cache_options(self.output / f"entry-{len(self.programs):03d}")
        assemble = online or "cache_config" in cache_options
        options = RunConfig(arch=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                            save_kernels=True, **cache_options)
        started = perf_counter()
        prior_cache = cache_stats() if timing is not None else None
        kernel = resident_entry(factory())
        # Compilation consumes tensor metadata. Real resident aliases are
        # bound only at dispatch and never enter the cached artifact.
        arguments = [
            value if isinstance(value, torch.Tensor) or not hasattr(value, "shape")
            else torch.empty(value.shape, dtype=value.dtype, device="meta")
            for value in (inputs[name] for name in kernel.param_names)
        ]
        compiled, abi = compile_resident_entry(
            kernel, arguments, options, abi=self.native("network").network_parameter_abi, name=key,
        )
        generated = perf_counter()
        if assemble:
            compiled.load()
        if timing is not None:
            cache = cache_stats()
            timing["compiled"].append({
                "operator": key, "seconds": generated - started,
                "assemble_seconds": perf_counter() - generated if assemble else 0.0,
                "ready_cache_hits": cache.ready_hits - prior_cache.ready_hits,
                "generation_builds": cache.generation_builds - prior_cache.generation_builds,
                "binary_builds": cache.binary_builds - prior_cache.binary_builds,
                "cache_bypasses": cache.bypasses - prior_cache.bypasses,
            })
        return compiled, abi

    def execute(self, key, rank, factory, inputs):
        profile = active_profile()
        timing = None if profile is None else profile.current
        with self.compile_lock:
            if key not in self.programs:
                self.programs[key] = self._compile_program(key, factory, inputs, timing)
        compiled, abi = self.programs[key]
        local_rank = self.rank_ids[rank]
        arguments = [inputs[name] for name in compiled.param_names]
        if timing is None:
            self.resident.execute(compiled, local_rank, arguments, abi)
            return

        with profile.lock:
            first = not profile.dispatch_seen(rank, key)
        transfers = self.resident.transfer_stats[local_rank]
        before = {name: transfers.get(name, 0) for name in ("h2d_bytes", "d2h_bytes", "d2d_bytes")}
        evictions = self.resident.stats.get("program_evictions", 0)
        started = monotonic_ns()
        try:
            self.resident.execute(compiled, local_rank, arguments, abi)
        finally:
            ended = monotonic_ns()
            changes = {name: transfers.get(name, 0) - previous for name, previous in before.items()}
            operator = key[1] if isinstance(key, tuple) and key[0] == "online" else str(key)
            elapsed = (ended - started) / 1e9
            with profile.lock:
                timing["kernel_dispatch_seconds"][operator] += elapsed
                if not profile.parallel_ranks:
                    changes["program_evictions"] = self.resident.stats.get("program_evictions", 0) - evictions
                    timing["counts"]["device_callable_evictions"] += changes["program_evictions"]
                profile.dispatches.append(dict(
                    update=timing["update"], operator=operator, program=key, rank=rank,
                    device=self.resident.devices[local_rank], tid=get_native_id(),
                    start_ns=started, end_ns=ended, first=first, counters=changes,
                ))
                if first:
                    timing["first_dispatch"].append({"operator": key, "rank": rank, "seconds": elapsed})

    def greedy_shard(self, rank, logits, active):
        """Return the finite-checked argmax of each live row on this TP rank."""
        from .pypto_dsv41_greedy import PACKET_COLUMNS, make_local_greedy

        capacity, vocabulary = logits.shape
        device = self.rank_ids[rank]
        packets = self.resident.workspace((capacity, PACKET_COLUMNS), torch.float32, device, fill=None)
        self.execute(
            ("draft-greedy", capacity, vocabulary), rank, lambda: make_local_greedy(capacity, vocabulary),
            dict(logits=logits, packets=packets, active=active),
        )
        return self.resident.read_output(packets[:active], device)

    def arm_online_capture(self, requests, plan):
        """Capture exactly the next draft round for the selected request owners."""
        if self.block_size != BLOCK:
            raise ValueError("Online replay currently requires block width 32")
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
        self.swa_spans = {}
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
        keep.extend((self.attention.rank_ids[rank], tensor) for (rank, _), tensor in self.swa_spans.items())
        if self.capture is not None and layer in self.capture[0].layers:
            self.attention.resident.retain_replay_values(
                (self.attention.rank_ids[rank], tensor)
                for captured in (self.captured_attention, self.captured_tail)
                for (rank, physical_layer, _), value in captured.items()
                if physical_layer == layer
                for tensor in self._tensors(value)
            )
        return keep

    def captures_rank(self, layer, rank, record):
        """Whether this layer must keep routed expert outputs for online replay."""
        if self.capture is None or layer not in self.capture[0].layers:
            return False
        requests = self.capture[1]
        dp = rank // self.attention.topology["tp"]
        return any(self.owners[dp, item["request_slot"]] in requests
                   for item in record["request_token_map"])

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
                        captured = dict(state, rank=rank, case=case)
                        self._snapshot_boundaries(rank, captured)
                        self.captured_tail[rank, layer, owner] = captured

    def _snapshot_boundaries(self, rank, state):
        # The forward alternates two boundary buffers across layers. A lease
        # alone protects their allocation, not the values in those buffers.
        resident = self.attention.resident
        local = self.attention.rank_ids[rank]
        copies = {}
        for name in ("readonly_hidden", "hidden", "incoming", "output_hidden", "ffn_pre"):
            value = state[name]
            if id(value) not in copies:
                snapshot = resident.workspace(value.shape, value.dtype, local, fill=None)
                resident.copy_tensor(snapshot, value, local)
                copies[id(value)] = snapshot
            state[name] = copies[id(value)]

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
                    span = {}
                    record = state.get("original")
                    if record is not None:
                        matches = []
                        for item in record["request_token_map"]:
                            mapped = self.owners[
                                rank // self.attention.topology["tp"], item["request_slot"]
                            ]
                            if mapped == owner:
                                matches.append(item)
                        if len(matches) != 1:
                            raise RuntimeError(
                                "captured suffix state has no unique request token span"
                            )
                        begin, end = (
                            int(matches[0]["token_begin"]), int(matches[0]["token_end"]),
                        )
                        if end - begin != BLOCK:
                            raise RuntimeError("online suffix replay requires one complete K32 row span")
                        span = {
                            "token_begin": begin, "token_end": end,
                            "sp_span": DSV41SequenceSpan.from_record(record, begin, end),
                        }
                    attention[rank] = dict(
                        a,
                        readonly_hidden=state["readonly_hidden"],
                        hidden=state["hidden"], incoming=state["incoming"],
                        pre=state["pre"], post=state["post"], comb=state["comb"],
                        collapsed=state["attn_collapsed"], normed=a["qkv"]["hidden"],
                        changed=state["changed"],
                        attention_scattered=state["attention_scattered"], **span,
                    )
                    ffn[rank] = {
                        "changed": state["changed"], "pre": state["ffn_pre"],
                        "post": state["ffn_post"], "comb": state["ffn_comb"],
                        "collapsed": state["ffn_collapsed"], "normed": state["ffn_normed"],
                        "router_input": state["router_input"], "router_logits": state["router_logits"],
                        "route_ids": state["route_ids"], "route_weights": state["weights"],
                        "shared": state["shared"], "expert_outputs": state["expert_outputs"],
                        "combined": state["combined"],
                        "output_hidden": state["output_hidden"],
                        "output_pre_mix": state["ffn_pre"],
                        **span,
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
            if any(set(layer.attention) != expected_ranks for layer in layers):
                raise RuntimeError("captured suffix layers do not cover the owner's TP ranks")
            replays[owner] = DSV41ServingReplay(
                request_id, prefix, owner, plan, boundary, tuple(layers),
                release=self.attention.resident.release_replay_values,
                ep_bucket_rows=tuple(self._ep_bucket_rows(layer.ffn) for layer in layers),
            )
        return replays

    def _ep_bucket_rows(self, ffn):
        """Size EP packets from the immutable route IDs captured for this request."""
        if any("sp_span" not in row for row in ffn.values()):
            return ROUTES
        spans = [row["sp_span"] for row in ffn.values()]
        if sum(span.rows for span in spans) != BLOCK:
            return ROUTES
        resident = self.attention.resident
        local_ids = []
        for rank, row in ffn.items():
            span = row["sp_span"]
            if span.rows:
                ids = row["route_ids"][span.local_start:span.local_start + span.rows]
                if ids.shape[0] != span.rows:
                    raise ValueError("captured EP route IDs do not cover the source span")
                local_ids.append(resident.read_control(ids, self.attention.rank_ids[rank]))
        return route_bucket_width(local_ids)

    def project_layer(self, rank, layer, record, hidden, cos, sin, contribution):
        block_size = self.attention.block_size
        dp = rank // self.attention.topology["tp"]
        for item in record["request_token_map"]:
            begin, end = item["token_begin"], item["token_end"]
            owner = self.owners[dp, item["request_slot"]]
            if end - begin != block_size:
                raise ValueError("Orthrus expects a complete draft block for each scheduler request")
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

    @staticmethod
    def reduce_rows(physical):
        """Use one aligned TP lane for the physically live sequence rows."""
        return max(16, (physical + 15) // 16 * 16)

    def pack_fp32(self, rank, contribution, physical, total, *, packet_rows=None):
        capacity, hidden = contribution.shape
        tp = self.attention.topology["tp"]
        rows = capacity if packet_rows is None else packet_rows
        local = self.attention.rank_ids[rank]
        packed = self.attention.resident.workspace(
            (tp * rows, hidden), torch.float32, local, fill=None,
        )
        self.attention.execute(
            ("draft_fp32_reduce_pack", capacity, rows), rank,
            lambda: make_draft_attention_reduce_pack(capacity, hidden, tp, packet_rows=rows),
            dict(value=contribution, packed=packed, physical=physical, active=total),
        )
        return packed

    def narrow_fp32(self, rank, value, capacity):
        rows, hidden = value.shape
        local = self.attention.rank_ids[rank]
        output = self.attention.resident.workspace((capacity, hidden), torch.bfloat16, local, fill=None)
        self.attention.execute(
            ("draft_fp32_reduce_narrow", rows, capacity), rank,
            lambda: make_draft_attention_reduce_narrow(rows, capacity, hidden),
            dict(value=value, output=output),
        )
        return output

    def project_block(self, rank, layer, owner, hidden, cos, sin, contribution, *, record):
        attention = self.attention
        block_size = attention.block_size
        runtime = attention.resident
        history = self.histories[rank, layer, owner]
        anchor, role, planes = history["committed_tokens"], history["role"], history["planes"]
        heads, config = attention.heads, attention.config
        width, latent = heads * HEAD_DIM, config["q_lora_rank"]

        local_rank = attention.rank_ids[rank]

        def workspace(shape, dtype=torch.bfloat16, *, fill=None):
            # The draft programs write every element of these output buffers.
            return runtime.workspace(shape, dtype, local_rank, fill=fill)

        operands = attention.operands[rank, layer]
        values = dict(operands, hidden=hidden, cos=cos, sin=sin, active=block_size)
        for name, columns in (
            ("qa", latent), ("latent", latent), ("qb", width), ("query", width),
            ("kv", HEAD_DIM), ("normalized_kv", HEAD_DIM),
        ):
            values[name] = workspace((block_size, columns))
        # SWA masks padded KV rows in the 16-column Cube tile at K8.
        temporary = workspace((max(16, block_size), HEAD_DIM), fill=0 if block_size < 16 else None)
        values["temporary"] = temporary[:block_size] if block_size < 16 else temporary
        attention.execute("qkv", rank,
                          lambda: make_qkv_program(attention.native, config, block_size=block_size), values)

        swa, = planes["swa"]
        if (swa["shape"][1:] != (WINDOW, 1, HEAD_DIM)
                or swa["handle"].shape != (swa["shape"][0] * swa["stride_rows"], HEAD_DIM)
                or swa["handle"].dtype != torch.bfloat16):
            raise ValueError("Invalid resident SWA page plane")
        spans = swa_spans(swa["pages"], anchor, swa["shape"][0], stride_rows=swa["stride_rows"],
                          offset_rows=swa["offset_rows"])
        if attention.reuse_swa_controls:
            bias = attention.swa_history_bias[min(anchor, WINDOW)]
            key = rank, spans
            spans_tensor = self.swa_spans.get(key)
            if spans_tensor is None:
                spans_tensor = self.swa_spans[key] = torch.tensor(spans, dtype=torch.int32)
        else:
            bias = torch.full((1, WINDOW), -torch.inf, dtype=torch.float32)
            bias[:, :min(anchor, WINDOW)] = 0
            spans_tensor = torch.tensor(spans, dtype=torch.int32)
        output_shape = (block_size * heads, HEAD_DIM)
        compressed = role["index_source"] is not None
        lse_shape = (block_size * heads, 1)
        query = values["query"]
        swa_args = dict(query=query, spans=spans_tensor,
                        temporary=temporary, history_bias=bias, sink=operands["sink"])
        swa_pages = swa["handle"]
        swa_zeros = attention.constants[rank]["swa_zero"]
        joined = compressed and attention.joined_attention
        if not joined:
            local = workspace(output_shape, torch.float32)
            local_lse = workspace(lse_shape, torch.float32)
            attention.execute(("swa", swa_pages.shape[0]), rank,
                              lambda: make_swa_attention(heads, token_major_query=True, block_size=block_size),
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
            flat_query = query.view(-1, HEAD_DIM)
            if joined:
                attention.execute(("joined_attention", swa_pages.shape[0], long["handle"].shape[0]), rank,
                                  lambda: make_joined_attention(block_size=block_size),
                                  dict(swa_args, **sparse_args, swa_pages=swa_pages, swa_zeros=swa_zeros,
                                       query_flat=flat_query))
            else:
                attention.execute(("sparse", long["handle"].shape[0]), rank,
                                  lambda: make_sparse_attention(head_major_swa=True, block_size=block_size),
                                  dict(sparse_args, query=flat_query, local_output=local, local_lse=local_lse))

        values.update(attention=context, unrotated=workspace((block_size, width)),
                      grouped=workspace((block_size, operands["wo_b"].shape[0])),
                      contribution=contribution)
        attention.execute(("output", compressed), rank,
                          lambda: make_output_program(attention.native, config, head_major_input=not compressed,
                                                      block_size=block_size), values)
        if self.capture is not None and layer in self.capture[0].layers and owner in self.capture[1]:
            from .pypto_dsv41_attention_vjp import C1AAttentionVJPGeometry, make_joined_history

            if not compressed or role["ratio"] != 1:
                raise ValueError("online draft replay requires C1A history")
            # Accepted tokens may overwrite these cache pages while delayed
            # teacher rows accumulate. Snapshot the exact selected K/V now.
            geometry = C1AAttentionVJPGeometry()
            joined_history = workspace((block_size * geometry.padded_columns, HEAD_DIM))
            attention.execute("online-joined-history", rank,
                              lambda: make_joined_history(geometry),
                              dict(swa_pages=swa_pages, spans=swa_args["spans"],
                                   compressed_pages=long["handle"], selected_rows=selected["physical_rows"],
                                   selected_lengths=selected["lengths"], temporary=values["temporary"],
                                   zeros=swa_zeros, joined=joined_history))
            self.captured_attention[rank, layer, owner] = {
                "rank": rank, "record": record, "hidden": hidden, "incoming": None,
                "pre": None, "post": None, "comb": None, "collapsed": hidden,
                "cos": cos, "sin": sin, "history": history,
                "selection": selected, "joined_history": joined_history,
                "qkv": values, "contribution": contribution, "lse": final_lse,
            }

    def index_rows(self, rank, layer, owner, history, values, workspace):
        attention = self.attention
        block_size = attention.block_size
        projection_rows = max(16, block_size)
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
            block_size=projection_rows,
        )
        codes, scales = history["planes"]["index"]
        outputs = dict(physical_rows=workspace((projection_rows, TOPK), torch.int32),
                       lengths=workspace((projection_rows,), torch.int32),
                       bias=workspace((projection_rows, TOPK), torch.float32))
        shapes = {
            "idx_projected": ((projection_rows, INDEX_HEADS * INDEX_DIM), torch.bfloat16),
            "idx_raw_weights": ((projection_rows, INDEX_HEADS), torch.bfloat16),
            "idx_query": ((projection_rows, INDEX_HEADS, INDEX_DIM), torch.bfloat16),
            "idx_scaled_weights": ((projection_rows, INDEX_HEADS), torch.float16),
            "index_keys": ((geometry.bucket, INDEX_DIM), torch.int8),
            "key_scales": ((geometry.bucket,), torch.float16),
            "query_codes": ((projection_rows, INDEX_HEADS, INDEX_DIM), torch.int8),
            "query_scales": ((projection_rows, INDEX_HEADS), torch.float16),
            "index_scores": ((projection_rows, geometry.bucket), torch.float32),
            "logical_rows": ((projection_rows, TOPK), torch.int32),
        }
        bindings = dict(values, **controls, **outputs, index_codes=codes["handle"], index_scales=scales["handle"],
                        zero_codes=attention.constants[rank][f"index_zero_{geometry.ratio}"])
        # A full block overwrites every intermediate. Keep padding initialized
        # when the indexer has fewer live rows than its aligned extent.
        local = attention.rank_ids[rank]
        if projection_rows != block_size:
            for name in ("latent", "hidden", "cos", "sin"):
                value = values[name]
                padded = attention.resident.workspace((projection_rows, *value.shape[1:]), value.dtype, local)
                attention.resident.copy_tensor(padded[:block_size], value, local)
                bindings[name] = padded
        index_fill = 0 if projection_rows != block_size or attention.clear_index_scratch else None
        bindings.update({name: attention.resident.workspace(shape, dtype, local, fill=index_fill)
                         for name, (shape, dtype) in shapes.items()})
        attention.execute(("indexer", geometry), rank,
                          lambda: make_indexer_program(attention.native, config, geometry,
                                                       block_size=projection_rows), bindings)
        if projection_rows != block_size:
            outputs = {name: value[:block_size] for name, value in outputs.items()}
        if self.last_use[layer] > layer:
            self.selection[key] = identity, outputs
        return outputs
