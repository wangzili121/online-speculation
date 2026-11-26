"""Rank-local resident buffers for the 1122 whole-model chip graphs."""

from ctypes import memmove
from math import prod
from types import SimpleNamespace


def forward_templates(weights, engram, capacities, context, slots):
    from models.deepseek_v4_1_flash_a3.host import prepare_resident_inputs
    from models.deepseek_v4_1_flash_a3.utils import DECODE_CACHE_NAMES, prepare_decode_inputs

    prefill = prepare_resident_inputs(weights, [], capacities[0], max_seq_len=context, engram=engram,
                                      request_slots=slots)
    initial_cache = {name: prefill[name] for name in DECODE_CACHE_NAMES.values()}
    ropes = prefill["freqs_cos"], prefill["freqs_sin"]
    decode = prepare_decode_inputs(prepare_resident_inputs(
        weights, [], capacities[1], max_seq_len=context, cache_state=initial_cache,
        rope_tables=ropes, engram=engram, request_slots=slots,
    ))
    return dict(prefill=prefill, decode=decode), initial_cache, ropes


def forward_inputs(programs, weights, templates, initial_cache):
    from models.deepseek_v4_1_flash_a3.replay import DecodeInputs
    from models.deepseek_v4_1_flash_a3.utils import DECODE_CACHE_NAMES

    constants = {*weights.tensor_layouts, *initial_cache, *DECODE_CACHE_NAMES, "freqs_cos", "freqs_sin"}
    parameters = {
        phase: {parameter.name.removesuffix("__ssa_v0"): parameter
                for parameter in program._get_metadata()[0]}
        for phase, program in programs.items()
    }
    return parameters, {
        phase: DecodeInputs({name: value for name, value in templates[phase].items()
                             if name in params and name not in constants})
        for phase, params in parameters.items()
    }


class NativeRankMemory:
    """Bind PyPTO's tensor views to allocations owned by one Simpler L3."""

    def __init__(self, worker, *, staging_bytes=32 * 1024**2):
        self.worker = worker
        self.staging = worker.create_buffer(staging_bytes)
        self.allocations = {}

    def alloc_tensor(self, shape, dtype, *, worker_id, init=None):
        from pypto.runtime import DeviceTensor
        from simpler_setup.torch_interop import torch_dtype_to_datatype

        shape = tuple(shape)
        handle = self.worker.alloc_child_tensor(worker_id, shape, torch_dtype_to_datatype(dtype))
        ptr = int(handle.base)
        self.allocations[worker_id, ptr] = handle
        if init is not None:
            init = init.contiguous()
            self.copy_to(ptr, init.data_ptr(), init.nbytes, worker_id=worker_id)
        tensor = DeviceTensor(ptr, shape, dtype, buffer=handle)
        return tensor

    def copy_to(self, dst, src, nbytes, *, worker_id, dst_offset=0, src_offset=0):
        self.upload_to_ranks(src + src_offset, nbytes, ((worker_id, dst, dst_offset),))

    def upload_to_ranks(self, src, nbytes, destinations):
        """Stage one host image, with a single host memcpy for all TP peers."""
        destinations = tuple(destinations)
        host = self.staging
        for start in range(0, nbytes, host.nbytes):
            size = min(host.nbytes, nbytes - start)
            memmove(int(host.base), src + start, size)
            for worker, ptr, offset in destinations:
                device = self.allocations[worker, ptr]
                self.worker.copy_to(device, host, dst_offset=offset + start, nbytes=size)

    def copy_from(self, dst, src, nbytes, *, worker_id, dst_offset=0, src_offset=0):
        device = self.allocations[worker_id, src]
        host = self.staging
        for start in range(0, nbytes, host.nbytes):
            size = min(host.nbytes, nbytes - start)
            self.worker.copy_from(host, device, src_offset=src_offset + start, nbytes=size)
            memmove(dst + dst_offset + start, int(host.base), size)


def graph_argument(shards, workers, parameter, world):
    """Bind local shards to the same ND or folded NZ ABI as the original graph."""
    from models.deepseek_v4_1_flash_a3.device_views import resident_graph_argument

    shape = tuple(parameter.shape)
    first = shape[0]
    if parameter.layout == "NZ" and isinstance(first, int) and first != world:
        if first % world:
            raise ValueError(f"NZ bank axis {first} cannot be split across {world} global ranks")
        local_shape = (first // world * len(workers), *shape[1:])
        parameter = SimpleNamespace(shape=local_shape, layout="NZ")
    return resident_graph_argument(shards, workers, parameter)


class NativeResidentForward:
    """Share Prefill and Decode storage while loading only this host's global shards."""

    phases = ("prefill", "decode")
    count_names = ("moe_num_tokens", "attention_num_tokens")

    def __init__(self, memory, programs, graphs, ranks, weights, engram, *, capacities, context, slots,
                 selected_layers=None):
        import torch
        from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE
        from models.deepseek_v4_1_flash_a3.lm_head import MAX_LOGIT_ROWS, SAMPLED_IDS_PAD
        from models.deepseek_v4_1_flash_a3.utils import DECODE_CACHE_NAMES, decode_cache_views
        from pypto.runtime import DeviceTensor
        from simpler.task_interface import DataType

        self.memory = memory
        self.ranks = tuple(ranks)
        self.workers = tuple(range(len(self.ranks)))
        self.tp = TP_SIZE
        self.world = graphs["prefill"].world_size
        self.local_dps = tuple(dict.fromkeys(rank // self.tp for rank in self.ranks))
        for dp in self.local_dps:
            if any(dp * self.tp + lane not in self.ranks for lane in range(self.tp)):
                raise ValueError("A TP group must fit on one host")

        templates, initial_cache, self.rope_tables = forward_templates(weights, engram, capacities, context, slots)
        self.initial_cache = initial_cache
        cache_views = decode_cache_views(initial_cache)
        outputs = {
            "x_hc_out": ((capacities[0], FLASH.hc_mult, FLASH.hidden_size), torch.float32),
            "attn_hc_out": ((capacities[0], FLASH.hc_mult, FLASH.hidden_size), torch.float32),
            "pre_mix_out": ((capacities[0], FLASH.hc_mult), torch.float32),
            "logits": ((MAX_LOGIT_ROWS, FLASH.vocab_size), torch.float32),
            "sampled_ids": ((MAX_LOGIT_ROWS, SAMPLED_IDS_PAD), torch.int32),
        }
        constants = {*weights.tensor_layouts, *initial_cache, *cache_views, "freqs_cos", "freqs_sin"}
        parameters, self.inputs = forward_inputs(programs, weights, templates, initial_cache)
        for phase, params in parameters.items():
            supported = set(templates[phase]) | constants | set(outputs) | set(self.count_names) | {"execute_layers"}
            unknown = set(params) - supported
            if unknown:
                raise ValueError(f"{phase} graph has unbound parameters: {sorted(unknown)}")
            unranked = {p.name for name, p in params.items() if name in self.count_names or name == "execute_layers"}
            if set(p.name for p in params.values()) - unranked != graphs[phase].rank_stacks:
                raise ValueError(f"{phase} graph rank layout differs from the resident parameter layout")
        self.idle = {
            phase: {name: templates[phase][name] for name in inputs.fields[0]}
            for phase, inputs in self.inputs.items()
        }
        self.count_buffers = {
            name: memory.worker.create_buffer(self.world * 4) for name in self.count_names
        }
        self.buffers = {phase: [] for phase in self.phases}

        for worker in self.workers:
            shared = {}
            for phase, params in parameters.items():
                fields = self.inputs[phase].bind(memory, worker)
                bank = {}
                for name in params:
                    if name in self.count_names or name == "execute_layers":
                        continue
                    if name in fields:
                        bank[name] = fields[name]
                        continue
                    if phase == "decode" and name in DECODE_CACHE_NAMES:
                        source = shared[DECODE_CACHE_NAMES[name]]
                        bank[name] = DeviceTensor(source.data_ptr, cache_views[name].shape, source.dtype,
                                                  buffer=source.buffer)
                        continue
                    if name not in shared:
                        if name in weights.tensor_layouts or name in outputs:
                            shape, dtype = (weights.tensor_layouts[name] if name in weights.tensor_layouts
                                            else outputs[name])
                            shared[name] = memory.alloc_tensor(shape, dtype, worker_id=worker)
                        else:
                            value = templates[phase][name]
                            shared[name] = memory.alloc_tensor(value.shape, value.dtype, init=value, worker_id=worker)
                    bank[name] = shared[name]
                self.buffers[phase].append(bank)

        self.arguments = {}
        self.scratch_buffers = []
        for phase, params in parameters.items():
            banks = self.buffers[phase]
            fields = {
                p.name: graph_argument([bank[name] for bank in banks], self.workers, p, self.world)
                for name, p in params.items() if p.name in graphs[phase].rank_stacks
            }
            fields.update({params[name].name: buffer.tensor((self.world,), DataType.INT32)
                           for name, buffer in self.count_buffers.items() if name in params})
            scratch, buffers = graphs[phase].bind_remote_intermediates(memory.worker, self.ranks)
            self.scratch_buffers.extend(buffers)
            fields.update(scratch)
            if "execute_layers" in params:
                fields[params["execute_layers"].name] = 1
            self.arguments[phase] = fields

        # Both phases have bound their real ABI before streaming any checkpoint matrices.
        for worker, rank in enumerate(self.ranks):
            weights.upload(memory, rank, self.buffers["prefill"][worker] | self.buffers["decode"][worker],
                           layers=selected_layers, worker_id=worker)

    def stage(self, phase, dp, values):
        """Upload one packed image to the TP owner and one shared image to its peers."""
        logit_name = "logit_rows" if phase == "prefill" else "logit_row_indices"
        inputs = self.inputs[phase]
        idle = self.idle[phase]
        inputs.stage((idle, {name: values[name] for name in idle}), logit_name)
        self._upload(phase, dp)

    def stage_image(self, phase, dp, payload):
        """Receive one owner image and keep logits disabled on the three TP peers."""
        inputs = self.inputs[phase]
        logit_name = "logit_rows" if phase == "prefill" else "logit_row_indices"
        owner_image = inputs.images[2]
        if len(payload) != owner_image.nbytes:
            raise ValueError(f"Invalid {phase} packed input size on DP {dp}")
        memoryview(owner_image.numpy())[:] = payload
        inputs.images[1].copy_(owner_image)
        inputs.fields[1][logit_name].copy_(self.idle[phase][logit_name])
        self._upload(phase, dp)

    def _workers(self, dp):
        return tuple(self.ranks.index(dp * self.tp + lane) for lane in range(self.tp))

    def _upload(self, phase, dp):
        inputs = self.inputs[phase]
        workers = self._workers(dp)
        for image, destinations in zip(inputs.images[1:], (workers[1:], workers[:1]), strict=True):
            if destinations:
                self.memory.upload_to_ranks(
                    image.data_ptr(), image.nbytes,
                    ((worker, inputs.devices[worker].data_ptr, 0) for worker in destinations),
                )

    def stage_decode_counts(self, rows_by_dp):
        """Mirror the global row counts needed by the device EP schedule."""
        from array import array

        if len(rows_by_dp) * self.tp != self.world:
            raise ValueError("Decode requires one row count for each global DP group")
        payload = array("i", (rows for rows in rows_by_dp for _ in range(self.tp))).tobytes()
        for buffer in self.count_buffers.values():
            with buffer.host_view() as view:
                view[:] = payload

    def read(self, phase, dp, name, out, *, first_row=0):
        worker = self.ranks.index(dp * self.tp)
        tensor = self.buffers[phase][worker][name]
        self.memory.copy_from(out.data_ptr(), tensor.data_ptr, out.nbytes, worker_id=worker,
                              src_offset=first_row * prod(tensor.shape[1:]) * out.element_size())

    def clear(self, dp, slot, pages):
        workers = self._workers(dp)
        prefill = self.buffers["prefill"]
        for name, initial in self.initial_cache.items():
            blocks = (slot,) if name == "state_cache" else pages
            for source in initial:
                for block in blocks:
                    image = source[block]
                    offset = image.data_ptr() - initial.data_ptr()
                    self.memory.upload_to_ranks(
                        image.data_ptr(), image.nbytes,
                        ((worker, prefill[worker][name].data_ptr, offset) for worker in workers),
                    )


def bind_native_draft(target, graph, program, artifacts, head, block, last_layers, head_artifact):
    """Bind a full draft graph to the node's existing Target resident pages."""
    import torch
    from simpler.task_interface import DataType

    from .dsv41_packed_proposer import PackedDraftProposer

    class LocalDraft(PackedDraftProposer):
        def _rank_argument(self, name, parameter, *, replacement=None):
            replacement = replacement or {}
            shards = [replacement.get(worker, bank[name]) for worker, bank in enumerate(self.ranks)]
            return graph_argument(shards, target.workers, parameter, target.world)

    # The rank owner will supply global counts; they remain Host parameters.
    worker = target.memory.worker
    count_buffer = worker.create_buffer(target.world * 4)
    counts = count_buffer.tensor((target.world,), DataType.INT32)
    draft = LocalDraft(
        target.memory, program, None, target.buffers["prefill"], target.inputs["decode"],
        artifacts.weights, artifacts.engram, target.rope_tables, counts, draft_head=head, block=block,
        context=artifacts.context, tp=target.tp, head_artifact=head_artifact, last_layers=last_layers,
    )
    parameters = program._get_metadata()[0]
    arguments = {p.name: value for p, value in zip(parameters, draft.arguments, strict=True)}
    unranked = {p.name for p in parameters
                if p.name.removesuffix("__ssa_v0") in ("ffn_capture_owner", "moe_num_tokens")}
    if set(arguments) - unranked != graph.rank_stacks:
        raise ValueError("Draft forward graph rank layout differs from the resident parameter layout")
    scratch, buffers = graph.bind_remote_intermediates(worker, target.ranks)
    arguments.update(scratch)
    return SimpleNamespace(draft=draft, arguments=arguments, count_buffer=count_buffer, scratch_buffers=buffers)
