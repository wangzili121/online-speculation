"""Bind 1122 Prefill and Decode to one node-local resident weight and KV bank."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

from .dsv41_packed_decode import ResidentPackedDecode
from .dsv41_packed_prefill import ResidentPackedPrefill, prefill_slabs
from .pypto_dsv41_cluster_transport import ClusterCollectives


def load_staged_target_programs(prefill_root, decode_root, capacities, tokens, *, context, slots, local_ranks):
    """Read the named local-rank artifacts emitted by the two compiler commands."""
    from models.deepseek_v4_1_flash_a3.config import TP_SIZE
    from models.deepseek_v4_1_flash_a3.driver import load_manifest
    from pypto.ir import DistributedCompiledProgram

    for mode, root, capacity, slab in zip(
        ("prefill", "decode"), (prefill_root, decode_root), capacities, tokens, strict=True,
    ):
        manifest = load_manifest(root, mode=mode, capacity=capacity, request_slots=slots)
        if (manifest["history_capacity"], manifest["moe_slab_rows"], manifest.get("local_ranks")) != (
            context, slab, local_ranks,
        ):
            raise ValueError(f"Staged {mode} artifact geometry differs from this host's runtime")
    read = DistributedCompiledProgram.from_dir
    widths = dict.fromkeys(width for _, width in prefill_slabs(capacities[0], tokens[0], TP_SIZE))
    operations = ("prepare", "compute", "restore")
    prefill = dict(
        attention=read(prefill_root / "attention"), head=read(prefill_root / "head"),
        restores={width: read(prefill_root / f"ffn-restore-{width}") for width in widths},
        experts={width: tuple(read(prefill_root / f"ep-{operation}-{width}") for operation in operations)
                 for width in widths},
    )
    decode = dict(
        attention=read(decode_root / "attention"), head=read(decode_root / "head"),
        restore=read(decode_root / "ffn-restore"),
        ep_programs={"decode": tuple(read(decode_root / f"ep-{operation}") for operation in operations)},
    )
    return prefill, decode


@contextmanager
def open_staged_target(config, plan, link, weights, engram, programs, *, capacities, context, slots, tokens):
    """Own the graph worker and HCCL lifetimes for both model phases."""
    from pypto.runtime.distributed_runner import DistributedWorker

    prefill, decode = programs
    graphs = [
        prefill["attention"], prefill["head"], *prefill["restores"].values(),
        *(entry for group in prefill["experts"].values() for entry in group),
        decode["attention"], decode["head"], decode["restore"],
        *(entry for group in decode["ep_programs"].values() for entry in group),
    ]
    devices = config.distributed_config.device_ids
    with (
        DistributedWorker(graphs, config, persistent=True, reset_persistent_windows=False) as runtime,
        ThreadPoolExecutor(max_workers=len(devices)) as pool,
    ):
        dispatcher = SimpleNamespace(devices=devices, map_ranks=lambda task: list(pool.map(task, range(len(devices)))))
        with ClusterCollectives(dispatcher, plan, link) as transport:
            yield StagedPackedTarget(runtime, transport, weights, engram, programs, capacities=capacities,
                                     context=context, slots=slots, tokens=tokens)


class StagedPackedTarget:
    """Serve a DP owner's rows locally while all ranks participate in EP."""

    def __init__(self, runtime, transport, weights, engram, programs, *, capacities, context, slots, tokens):
        import torch
        from models.deepseek_v4_1_flash_a3.config import FLASH, TP_SIZE
        from models.deepseek_v4_1_flash_a3.host import prepare_resident_inputs
        from models.deepseek_v4_1_flash_a3.lm_head import MAX_LOGIT_ROWS, SAMPLED_IDS_PAD
        from models.deepseek_v4_1_flash_a3.replay import DecodeInputs
        from models.deepseek_v4_1_flash_a3.utils import (
            DECODE_CACHE_NAMES, decode_cache_views, prepare_decode_inputs,
        )
        from pypto.runtime import StackedDeviceTensor

        self.runtime = runtime
        self.link = transport.link
        self.global_ranks = tuple(transport.plan.local_ranks)
        self.rank_offset = self.global_ranks[0]
        self.tp = TP_SIZE
        self.workers = tuple(range(len(self.global_ranks)))
        prefill_rows, decode_rows = capacities
        prefill = prepare_resident_inputs(weights, [], prefill_rows, max_seq_len=context,
                                          request_slots=slots, engram=engram)
        self.rope_tables = (prefill["freqs_cos"], prefill["freqs_sin"])
        self.initial_cache = {name: prefill.pop(name) for name in DECODE_CACHE_NAMES.values()}
        self.cache_image = {name: tensor[0, 0].clone() for name, tensor in self.initial_cache.items()}
        decode_values = prepare_resident_inputs(
            weights, [], decode_rows, max_seq_len=context,
            cache_state=self.initial_cache, rope_tables=self.rope_tables, request_slots=slots, engram=engram,
        )
        decode = prepare_decode_inputs(decode_values)
        decode_shape = {name: value.shape for name, value in decode_cache_views(self.initial_cache).items()}
        for values in (prefill, decode):
            for name in ("freqs_cos", "freqs_sin"):
                del values[name]
        prefill_programs, decode_programs = programs
        consumed = {parameter.name.removesuffix("__ssa_v0")
                    for parameter in prefill_programs["attention"]._get_metadata()[0]}
        prefill = {name: value for name, value in prefill.items() if name in consumed or name == "logit_rows"}
        self.templates = (prefill, decode)
        self.inputs = (DecodeInputs(prefill), DecodeInputs(decode))
        self.handles = ([], [])
        prefills, decodes = self.handles

        for worker in self.workers:
            bank = {name: runtime.alloc_tensor(shape, dtype, worker_id=worker)
                    for name, (shape, dtype) in weights.tensor_layouts.items()}
            caches = {name: runtime.alloc_tensor(value.shape, value.dtype, worker_id=worker, init=value)
                      for name, value in self.initial_cache.items()}
            ropes = {name: runtime.alloc_tensor(value.shape, value.dtype, worker_id=worker, init=value)
                     for name, value in zip(("freqs_cos", "freqs_sin"), self.rope_tables, strict=True)}
            prefill_handles = dict(bank, **caches, **ropes, **self.inputs[0].bind(runtime, worker))
            decode_caches = {name: runtime.view_tensor(caches[source], decode_shape[name], worker_id=worker)
                             for name, source in DECODE_CACHE_NAMES.items()}
            decode_handles = dict(bank, **decode_caches, **ropes, **self.inputs[1].bind(runtime, worker))
            for handles in (prefill_handles, decode_handles):
                handles["logits"] = runtime.alloc_tensor((MAX_LOGIT_ROWS, FLASH.vocab_size), torch.float32,
                                                          worker_id=worker)
                handles["sampled_ids"] = runtime.alloc_tensor((MAX_LOGIT_ROWS, SAMPLED_IDS_PAD), torch.int32,
                                                               worker_id=worker)
            prefills.append(prefill_handles)
            decodes.append(decode_handles)

        with ThreadPoolExecutor(max_workers=min(len(self.workers), TP_SIZE)) as pool:
            list(pool.map(lambda worker: weights.upload(
                runtime, self.global_ranks[worker], prefills[worker], worker_id=worker,
            ), self.workers))

        def stack(handles):
            return {name: StackedDeviceTensor(
                [rank[name] for rank in handles], (len(handles), *value.shape), self.workers,
            ) for name, value in handles[0].items()}

        self.prefill = ResidentPackedPrefill(
            runtime, transport, capacity=prefill_rows, tokens=tokens[0],
            buffers=stack(prefills), **prefill_programs,
        )
        self.decode = ResidentPackedDecode.bind(
            runtime, transport, tokens=tokens[1], buffers=stack(decodes), **decode_programs,
        )
        self.graphs = SimpleNamespace(staged_target=self, weights=weights, engram=engram,
                                      rope_tables=self.rope_tables, host_io={"idle_inputs": self.templates})
        self.geometry = (slots, context, capacities)

    def session(self):
        """Reuse native admission, paging and request ownership."""
        from models.deepseek_v4_1_flash_a3.packed import PackedSession

        return PackedSession(self.graphs, *self.geometry)

    def owns(self, dp):
        return dp * self.tp in self.global_ranks

    def _owner(self, dp):
        if not self.owns(dp):
            raise ValueError("The DP partition is owned by the other node")
        return dp * self.tp - self.rank_offset

    def stage(self, phase, dp, values, pool):
        """Stage the prepared native rows once per worker with one packed upload."""
        owner = self._owner(dp)
        idle = self.templates[phase]
        logit_name = "logit_row_indices" if phase else "logit_rows"
        inputs = self.inputs[phase]
        inputs.stage((idle, {name: values[name] for name in idle}), logit_name)
        list(pool.map(lambda worker: inputs.upload(self.runtime, worker, owner),
                      range(owner, owner + self.tp)))

    def read(self, phase, dp, name, result, *, row=0):
        """Read logits or greedy IDs from the owning TP lane."""
        owner = self._owner(dp)
        handle = self.handles[phase][owner][name]
        row_bytes = handle.nbytes // handle.shape[0]
        self.runtime.copy_from(result.data_ptr(), handle.data_ptr, result.nbytes,
                               src_offset=row * row_bytes, worker_id=owner)

    def clear(self, dp, slot, pages, pool):
        """Restore only released cache pages and the slot's compressed state."""
        owner = self._owner(dp)

        def reset(worker):
            for name, initial in self.initial_cache.items():
                target = self.handles[0][worker][name]
                blocks = (slot,) if name == "state_cache" else pages
                image = self.cache_image[name]
                for source in range(initial.shape[0]):
                    for block in blocks:
                        offset = (source * initial.shape[1] + block) * image.nbytes
                        self.runtime.copy_to(target.data_ptr, image.data_ptr(), image.nbytes,
                                             dst_offset=offset, worker_id=worker)

        list(pool.map(reset, range(owner, owner + self.tp)))
