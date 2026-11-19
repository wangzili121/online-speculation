"""Online suffix replay for the OneServing TP2 draft.

The target and draft stay on the two serving ranks. The learner can use
rank zero on the same worker or a separate device. Both shards publish together.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from math import isfinite, prod
from pathlib import Path
import time

import numpy as np
import torch
from safetensors import safe_open
from torch.nn.functional import pad

from .head import merge_parameter
from .head_loading import bank_specs
from .online_feedback_programs import make_loss_sum, make_tp2_history
from .online_suffix_checkpoint import restore
from .online_suffix_optimizer import SuffixOptimizer, SuffixOptimizerPrograms
from .oneserving_tp2 import (BLOCK, HIDDEN, LAYERS, SHAPES, TP_VOCAB, checkpoint_manifest,
                             rank_weights)
from .training_executor import TrainingPrograms, TrainingWorkspace
from .training_memory import require_suffix_memory
from .training_suffix_graph import SuffixGraph
from .training_programs import tv_supervision_weight


FULL_BANK_START = 9


def pack_execution(layout, ordinary, special, packed):
    """Write rendered weights directly into the two preallocated serving shards."""
    bf16 = ordinary.view(torch.int16).numpy()[0]
    fp32 = special.numpy()[0]
    banks = {}
    for entry in layout.entries:
        parameter = entry.parameter
        name = bank_specs(parameter.layer)[parameter.bank][0]
        source, start = (bf16, entry.offset) if entry.special is None else (fp32, entry.special)
        banks.setdefault(parameter.layer, {})[name] = source[start:start + prod(parameter.shape)].reshape(parameter.shape)

    def destinations(rank):
        for host in packed:
            if host is None:
                yield None
            else:
                value = host[rank]
                yield value.numpy() if value.dtype == torch.float32 else value.view(torch.int16).numpy()

    def copy_qkv(source, destination, rank, axis):
        # Each rank gets its own slice of Q, K and V, in that order.
        begin = 0
        for width in (1024, 1024, 3072):
            part = slice(2 * begin + rank * width, 2 * begin + (rank + 1) * width)
            local = slice(begin, begin + width)
            if axis == 0:
                np.copyto(destination[local], source[part])
            else:
                np.copyto(destination[:, local], source[:, part])
            begin += width

    for rank in range(2):
        qkv, z, out, a, b, conv, decay, dt, norm, q, k, v, q_norm, k_norm, o = destinations(rank)
        heads = slice(rank * 24, (rank + 1) * 24)
        value_rows = slice(rank * 3072, (rank + 1) * 3072)
        for i, layer in enumerate(LAYERS[:-1]):
            if layer not in banks:
                continue
            params = banks[layer]
            rows = slice(i * 5120, (i + 1) * 5120)
            copy_qkv(params["in_proj_qkv.weight"], qkv[rows], rank, 0)
            copy_qkv(params["conv1d.weight"], conv[i * 4:(i + 1) * 4], rank, 1)
            np.copyto(z[i * 3072:(i + 1) * 3072], params["in_proj_z.weight"][value_rows])
            np.copyto(out[rows], params["out_proj.weight"][:, value_rows])
            for target, name in ((a, "in_proj_a.weight"), (b, "in_proj_b.weight")):
                local = target[i * 128:(i + 1) * 128]
                np.copyto(local[:24], params[name][heads])
                local[24:].fill(0)
            for target, name in ((decay, "A_log"), (dt, "dt_bias")):
                np.copyto(target[i, :24], params[name].reshape(-1)[heads])
                target[i, 24:].fill(0)
            np.copyto(norm[i], params["norm.weight"].reshape(-1))
        params = banks[LAYERS[-1]]
        for target, name in ((q, "q_proj.weight"), (k, "k_proj.weight"), (v, "v_proj.weight")):
            width = target.shape[0]
            np.copyto(target, params[name][rank * width:(rank + 1) * width])
        np.copyto(o, params["o_proj.weight"][:, value_rows])
        np.copyto(q_norm, params["q_norm.weight"].reshape(q_norm.shape))
        np.copyto(k_norm, params["k_norm.weight"].reshape(k_norm.shape))


def master_from_shards(layout, directory):
    """Lift the exact served BF16/FP32 parameters into a fresh FP32 Adam master."""
    master = torch.zeros(layout.shape, dtype=torch.float32)
    with safe_open(directory / "rank0.safetensors", framework="pt", device="cpu") as left, safe_open(
            directory / "rank1.safetensors", framework="pt", device="cpu") as right:
        for entry in layout.entries:
            parameter = entry.parameter
            name = bank_specs(parameter.layer)[parameter.bank][0]
            key = f"layers.{parameter.layer}.attention.draft.{name}"
            raw = merge_parameter(parameter.layer, name, [source.get_tensor(key) for source in (left, right)])
            if name in {"in_proj_a.weight", "in_proj_b.weight"}:
                raw = pad(raw, (0, 0, 0, 128 - raw.shape[0]))
            elif name in {"A_log", "dt_bias"}:
                raw = pad(raw, (0, 64 - raw.numel()))
            elif name == "conv1d.weight":
                raw = raw.squeeze(1).T
            value = raw.reshape(-1)
            master[0, entry.offset:entry.offset + value.numel()].copy_(value)
    return master


class TP2Online:
    def __init__(self, checkpoint, *, device, last_layers=4, learning_rate=1e-5, background=False,
                 replay_blocks=1, publish_every=1, gather_feedback=False, objective="tv", max_updates=None,
                 focus_weight=1):
        if type(device) is not int or device < 0 or not isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("a learner device and positive finite learning rate are required")
        if last_layers not in (1, 4) or (last_layers == 1 and checkpoint is not None):
            raise ValueError("TP2 replay supports last-4 or last-1 with fresh Adam")
        if (type(replay_blocks) is not int or replay_blocks < 1
                or type(publish_every) is not int or publish_every < 1):
            raise ValueError("replay_blocks and publish_every must be positive integers")
        if (objective not in ("tv", "kl")
                or max_updates is not None and (type(max_updates) is not int or max_updates < 1)):
            raise ValueError("expected TV or KL with an optional positive update limit")
        if (type(focus_weight) is not int or focus_weight < 1
                or focus_weight != 1 and objective != "tv"):
            raise ValueError("positive TV focus weight required; KL uses weight one")
        self.focus_weight = focus_weight
        self.replay_blocks = replay_blocks
        self.publish_every = publish_every
        self.gather_feedback = gather_feedback
        self.objective = objective
        self.max_updates = max_updates
        self.replay = []
        self.layers = LAYERS[-last_layers:]
        self.first_bank = FULL_BANK_START if last_layers == 1 else 0
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.device, self.learning_rate = device, learning_rate
        self.worker = None
        self.colocated = False
        self.background = background
        self.executor = self.pending_update = None
        self.completed_version = self.published_version = 0
        self.pending_reset = self.pending_drain = False
        self.pending_head = None
        self.stride = None
        self.rounds = self.version = 0
        self.feedback = None
        self.records = []

    def bind_devices(self, devices):
        self.colocated = self.device in devices
        if self.colocated and self.device != devices[0]:
            raise ValueError("colocated training uses the first rank in the serving worker")
        if self.colocated and self.background:
            # The worker rejects standalone copies while another run owns its device queue.
            raise ValueError("colocated training requires synchronous updates on the current PyPTO runtime")

    def prepare_host_buffers(self, runner):
        # Register shared mappings before the serving worker forks. Publication
        # can then DMA the packed shards without creating temporary IPC buffers.
        self.packed = [None] * self.first_bank + [
            torch.empty((2, *shape), dtype=dtype).share_memory_() for shape, dtype in SHAPES[self.first_bank:]
        ]
        runner._host_slabs.update({f"orthrus_publish_{i}": value
                                   for i, value in enumerate(self.packed) if value is not None})
        if self.gather_feedback:
            page_size = runner.state_specs[2][0][-2]
            self.history_table = torch.empty((2, 512 // page_size), dtype=torch.int32).share_memory_()
            self.history_host = [torch.empty((4, 512, 256), dtype=torch.bfloat16).share_memory_()
                                 for _ in range(2)]
            runner._host_slabs.update(orthrus_history_table=self.history_table,
                                      orthrus_history_key=self.history_host[0], orthrus_history_value=self.history_host[1])

    def compile_history(self, runner):
        _, _, (shape, dtype), _ = runner.state_specs[:4]
        kv = torch.empty((2, *shape), dtype=dtype, device="meta")
        out = torch.empty((2, 2, 512, 256), dtype=dtype, device="meta")
        layers, _, pages, page_size, _ = shape
        self.history_program = make_tp2_history(layers, pages, page_size).compile(
            kv, kv, self.history_table, out, out, 1, config=runner.runconfig)
        return self.history_program

    def compile(self, config):
        self.config = config if self.colocated else replace(config, distributed_config=replace(
            config.distributed_config, device_ids=[self.device]))
        self.graph = SuffixGraph(len(self.layers), objective=self.objective, focus_weight=self.focus_weight)
        self.training = TrainingPrograms(self.graph, self.config, combined=True, chunk_size=24,
                                         external_gradients=True)
        self.optimization = SuffixOptimizerPrograms(self.graph, self.config, clip_norm=1.0)
        partial = torch.empty((BLOCK, 8), dtype=torch.float32, device="meta")
        total = torch.empty((1, 8), dtype=torch.float32, device="meta")
        self.loss_program = make_loss_sum().compile(partial, total, config=self.config)
        return [*self.training.programs.values(), *self.optimization.programs(), self.loss_program]

    def allocate_execution_host(self):
        return (torch.empty(self.optimization.layout.shape, dtype=torch.bfloat16).share_memory_(),
                torch.empty(self.optimization.layout.special_shape, dtype=torch.float32).share_memory_())

    def compile_colocated(self, runner):
        programs = self.compile(runner.runconfig)
        self.ordinary, self.special = self.allocate_execution_host()
        runner._host_slabs.update(orthrus_render_bf16=self.ordinary, orthrus_render_fp32=self.special)
        return programs

    def start(self, runner):
        from pypto.runtime import DistributedWorker
        from serving.weight import TextWeights

        self.runner = runner
        if self.gather_feedback:
            self.history_output = [runner._allocate((2, 512, 256), torch.bfloat16) for _ in range(2)]
        if self.checkpoint is not None:
            manifest = json.loads((self.checkpoint / "manifest.json").read_text())
            self.head_identity = manifest["head"]
            if (manifest["payload_sha256"] != runner.head["source_payload_sha256"]
                    or manifest["step"] != runner.head["step"]
                    or self.head_identity["contract"] != runner.head["source_head_contract"]):
                raise ValueError("the optimizer checkpoint and served head must come from the same update")
        self.initial_step = runner.head["step"] if self.checkpoint is not None else 0
        if self.colocated:
            self.worker = runner.worker
        else:
            programs = self.compile(runner.runconfig)
            self.ordinary, self.special = self.allocate_execution_host()
            self.worker = DistributedWorker(programs, config=self.config,
                                            inherited_host_tensors=[self.ordinary, self.special])
        require_suffix_memory(self.graph, self.optimization.layout, self.replay_blocks, 1 << 60,
                              self.worker.device_memory_info()[0])
        self.optimizer = SuffixOptimizer(self.optimization, self.worker)
        if self.checkpoint is None:
            self.initial_master = master_from_shards(self.optimizer.layout, runner.head_directory)
        self.workspace = TrainingWorkspace(self.training, self.worker)
        self.loss = self.worker.alloc_tensor((1, 8), torch.float32)
        self.loss_host = torch.empty((1, 8), dtype=torch.float32)
        self.original = runner.suffix
        self.updated = [*self.original[:self.first_bank],
                        *(runner._allocate(shape, dtype) for shape, dtype in SHAPES[self.first_bank:])]
        weights = TextWeights(runner.settings.model_root)
        gamma = lambda value: (1 + value.float()).reshape(1, HIDDEN)
        self.shared = {
            "lm_head": self.upload_new(weights.read("lm_head.weight")),
            "final_gamma": self.upload_new(gamma(weights.read("model.language_model.norm.weight"))),
        }
        for layer in self.layers:
            for name, path in (
                ("input_gamma", "input_layernorm.weight"), ("post_gamma", "post_attention_layernorm.weight"),
                ("mlp_gate", "mlp.gate_proj.weight"), ("mlp_up", "mlp.up_proj.weight"),
                ("mlp_down", "mlp.down_proj.weight"),
            ):
                tensor = weights.layer(layer, path)
                self.shared[f"l{layer}_{name}"] = self.upload_new(gamma(tensor) if "gamma" in name else tensor)
        names = {"hidden", "cos", "sin", "teacher"} | {
            value.name for history in self.graph.histories.values() for value in history.values()
        }
        self.pool = []
        for _ in range(self.replay_blocks):
            inputs = {}
            for name in names:
                value = self.graph.inputs[name]
                dtype = torch.float32 if value.dtype == "fp32" else torch.bfloat16
                inputs[name] = self.worker.alloc_tensor(value.shape, dtype)
            self.pool.append(inputs)
        self.restore()
        if self.background:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="orthrus-learner")

    def upload_new(self, value):
        return self.worker.alloc_tensor(tuple(value.shape), value.dtype, init=value.contiguous())

    def upload(self, key, value, offset=0):
        value = value.contiguous()
        self.worker.copy_to(self.inputs[key].data_ptr, value.data_ptr(), value.nbytes, dst_offset=offset)

    def read(self, rank, source, dest, offset=0):
        self.runner._worker_call("copy_from", dest.data_ptr(), source.shards[rank].data_ptr,
                                 dest.nbytes, src_offset=offset, worker_id=rank)

    def restore(self):
        self.runner.suffix = self.original
        self.optimizer.steps = self.version = 0
        if self.checkpoint is not None:
            restore(self.optimizer, self, self.checkpoint, self.head_identity)
        else:
            self.optimizer.index = 0
            self.worker.copy_to(self.optimizer.state[0].data_ptr, self.initial_master.data_ptr(),
                                self.initial_master.nbytes)
            for tensor in (*self.optimizer.state[1:], *self.optimizer.candidate):
                self.optimizer.run(self.optimizer.programs.zero, tensor)
            self.optimizer.render(0, self.optimizer.state[0])
        # Each experiment starts from the original inference banks. The optimizer
        # either restores the corresponding checkpoint or uses their exact values.
        self.runner.suffix = self.original
        self.feedback = None
        self.replay.clear()
        self.rounds = 0
        self.completed_version = self.published_version = self.version

    def prepare_batch(self):
        self.collect_update(wait=self.pending_drain or self.pending_reset or self.pending_head is not None)
        self.pending_drain = False
        if self.pending_head is not None:
            self.load_head(self.pending_head)
            self.pending_head = None
        if self.pending_reset:
            self.restore()
            self.pending_reset = False

    def load_head(self, directory):
        """Reload a tested start between benchmark phases, with no learner enabled."""
        if self.stride is not None or self.checkpoint is not None:
            raise RuntimeError("head switching requires a disabled learner and fresh Adam")
        head = checkpoint_manifest(directory, self.runner.settings.model_root)
        if head["source_head_contract"] != self.runner.head["source_head_contract"]:
            raise ValueError("the new head must use the same draft contract")
        readers = [rank_weights(directory / f"rank{rank}.safetensors") for rank in range(2)]
        for packed, target, first, second in zip(self.packed, self.original, *readers, strict=True):
            for rank, source in enumerate((first, second)):
                if packed is None:
                    value = source.contiguous()
                else:
                    value = packed[rank]
                    value.copy_(source)
                self.runner._worker_call("copy_to", target.shards[rank].data_ptr, value.data_ptr(),
                                         value.nbytes, worker_id=rank)
        self.runner.head_directory = directory
        self.runner.head = head
        self.initial_master = master_from_shards(self.optimizer.layout, directory)
        self.restore()

    def update_limit_reached(self):
        return self.max_updates is not None and self.completed_version - self.initial_step >= self.max_updates

    def feedback_index(self, active):
        if self.stride is None or self.pending_update is not None or self.update_limit_reached():
            return None
        if self.stride < self.replay_blocks * active:
            raise ValueError("space replay blocks at least one draft batch apart")
        start = self.rounds % self.stride
        # Capture at most one verified block from each batch, spread over the stride.
        for block in range(1, self.replay_blocks + 1):
            index = block * self.stride // self.replay_blocks - 1 - start
            if 0 <= index < active:
                return index
        return None

    def capture_history(self, slot, position, width, candidates, table, cos, sin):
        if self.feedback is not None:
            raise RuntimeError("the previous feedback block is still active")
        started = time.monotonic_ns()
        self.inputs = self.pool[len(self.replay)]
        self.feedback = dict(position=position, width=width, candidates=candidates, valid=0,
                             complete=False, inputs=self.inputs)
        self.upload("cos", cos)
        self.upload("sin", sin)
        conv, recurrent, key, value = self.runner.resident[:4]
        for layer in self.layers[:-1]:
            item = slot * 48 + layer - layer // 4
            conv_parts = [torch.empty((4, 5120), dtype=torch.bfloat16) for _ in range(2)]
            for rank, part in enumerate(conv_parts):
                self.read(rank, conv, part, item * part.nbytes)
            # Local GDN layout is [Q, K, V] on each rank. Merge like groups first.
            groups = [part.split((1024, 1024, 3072), dim=1) for part in conv_parts]
            joined = torch.cat([groups[rank][group] for group in range(3) for rank in range(2)], dim=1)
            self.upload(f"l{layer}_conv_history", joined)
            for rank in range(2):
                part = torch.empty((3072, 128), dtype=torch.float32)
                self.read(rank, recurrent, part, item * part.nbytes)
                self.upload(f"l{layer}_recurrent_history", part, rank * part.nbytes)
        page_size, pages = key.shape[-2], key.shape[-3]
        used_pages = table[:(position + page_size - 1) // page_size].tolist()
        if any(not 0 <= page < pages for page in used_pages):
            raise RuntimeError("the feedback history contains an invalid target page")
        if self.gather_feedback:
            self.history_table.copy_(table)
            self.runner._worker_call("run", self.history_program, key, value, self.history_table,
                                     *self.history_output, position, config=self.runner.runconfig)
            for name, source, host in zip(("key", "value"), self.history_output, self.history_host, strict=True):
                for rank in range(2):
                    self.read(rank, source, host[rank * 2:rank * 2 + 2])
                end = (position + 15) // 16 * 16
                host[:, position:end].zero_()
                self.upload(f"l63_history_{name}", host)
        else:
            for name, source in (("key", key), ("value", value)):
                history = torch.zeros((4, 512, 256), dtype=torch.bfloat16)
                for rank in range(2):
                    for head in range(2):
                        for i, page in enumerate(used_pages):
                            start = i * page_size
                            count = min(position - start, page_size)
                            src = ((15 * 2 + head) * pages + page) * page_size * 256 * 2
                            self.read(rank, source, history[rank * 2 + head, start:start + count], src)
                self.upload(f"l63_history_{name}", history)
        self.runner.spec_stats["feedback_ns"] += time.monotonic_ns() - started

    def capture_hidden(self, buffer):
        started = time.monotonic_ns()
        hidden = torch.empty((BLOCK, HIDDEN), dtype=torch.bfloat16)
        self.read(0, buffer, hidden)
        self.upload("hidden", hidden)
        self.runner.spec_stats["feedback_ns"] += time.monotonic_ns() - started

    def capture_teacher(self, batch_slot, offset, count, *, final):
        started = time.monotonic_ns()
        state = self.feedback
        if offset < state["candidates"]:
            if offset != state["valid"] or state["complete"]:
                raise RuntimeError("feedback teacher rows must be a fresh contiguous verified prefix")
            count = min(count, state["candidates"] - offset)
            source = self.runner.outputs[self.runner._last_layout["output_key"]][1]
            local = torch.empty((count, TP_VOCAB), dtype=torch.bfloat16)
            for rank in range(2):
                self.read(rank, source, local, 4 * batch_slot * TP_VOCAB * 2)
                for i, row in enumerate(local):
                    self.upload("teacher", row, ((offset + i) * 2 + rank) * row.nbytes)
            state["valid"] += count
        state["complete"] = final
        self.runner.spec_stats["feedback_ns"] += time.monotonic_ns() - started

    def finish_batch(self, active):
        if self.stride is None or self.update_limit_reached():
            return
        if self.feedback is not None:
            state, self.feedback = self.feedback, None
            if not state["complete"] or not 1 <= state["valid"] <= state["candidates"]:
                raise RuntimeError("an update requires verified feedback")
            self.replay.append(state)
        previous = self.rounds
        self.rounds += active
        if previous // self.stride == self.rounds // self.stride:
            return
        if self.pending_update is not None:
            self.runner.spec_stats["busy_updates"] += 1
            return
        replay, self.replay = self.replay, []
        if not replay:
            raise RuntimeError("no verified feedback for the update")
        if self.executor is None:
            self.commit_update(self.train(replay, self.rounds))
        else:
            self.pending_update = self.executor.submit(self.train, replay, self.rounds)

    def train(self, replay, round_number):
        started = time.monotonic_ns()
        self.optimizer.zero_grad()
        supervised = sum(item["valid"] for item in replay)
        normalizer = sum(tv_supervision_weight(item["valid"], self.focus_weight) for item in replay)
        loss = 0.0
        common = self.shared | self.optimizer.tensors()
        for item in replay:
            self.workspace.run(common | item["inputs"], past=item["position"], width=item["width"],
                               valid=item["valid"], normalizer=normalizer, gradients=self.optimizer.gradients)
            self.optimizer.accumulate()
            self.worker.run(self.loss_program, self.workspace.values[self.graph.losses.name],
                            self.loss, config=self.config)
            self.worker.copy_from(self.loss_host.data_ptr(), self.loss.data_ptr, self.loss_host.nbytes)
            loss += self.loss_host[0, 0].item()
        if not isfinite(loss):
            raise FloatingPointError("nonfinite TP2 online loss")
        result = self.optimizer.step(self, self.learning_rate)
        elapsed = time.monotonic_ns() - started
        return dict(round=round_number, loss=loss, supervised=supervised, feedback_blocks=len(replay),
                    supervision_weight=normalizer, update_ns=elapsed, **result, **self.pack_stats)

    def publish_flat(self, layout, buffers, *, expected_version):
        # The optimizer calls this before advancing its master. Leave the
        # serving workers alone here, including during background training.
        if self.version != expected_version:
            raise RuntimeError("stale TP2 head update")
        next_version = expected_version + 1
        if (next_version - self.initial_step) % self.publish_every:
            self.pack_stats = dict(publish_read_ns=0, publish_pack_ns=0)
            self.version = next_version
            return
        started = time.monotonic_ns()
        for host, device in zip((self.ordinary, self.special), buffers, strict=True):
            self.worker.copy_from(host.data_ptr(), device.data_ptr, host.nbytes)
        read_done = time.monotonic_ns()
        # SuffixOptimizer checked the candidate FP32 master's norm before
        # rendering, so BF16 conversion cannot overflow a checked parameter.
        pack_execution(layout, self.ordinary, self.special, self.packed)
        self.pack_stats = dict(publish_read_ns=read_done - started,
                               publish_pack_ns=time.monotonic_ns() - read_done)
        self.version = next_version

    def collect_update(self, *, wait=False):
        future = self.pending_update
        if future is not None and (wait or future.done()):
            self.pending_update = None
            self.commit_update(future.result())

    def commit_update(self, result):
        if result["version"] != self.completed_version + 1:
            raise RuntimeError("out-of-order TP2 update")
        self.completed_version = result["version"]
        result.update(publish_upload_ns=0, published_at_round=None)
        if (self.completed_version - self.initial_step) % self.publish_every == 0:
            # Both ranks are idle at this boundary and see the new version together.
            started = time.monotonic_ns()
            for target, host in zip(self.updated, self.packed, strict=True):
                if host is None:
                    continue
                for rank in range(2):
                    value = host[rank]
                    self.runner._worker_call("copy_to", target.shards[rank].data_ptr, value.data_ptr(),
                                             value.nbytes, worker_id=rank)
            self.runner.suffix = self.updated
            self.published_version = self.completed_version
            result["published_at_round"] = self.rounds
            result["publish_upload_ns"] = time.monotonic_ns() - started
            self.runner.spec_stats["publications"] += 1
        result["publish_ns"] = sum(result[f"publish_{phase}_ns"] for phase in ("read", "pack", "upload"))
        self.runner.spec_stats["updates"] += 1
        for name in ("update_ns", "publish_ns", "publish_read_ns", "publish_pack_ns", "publish_upload_ns"):
            self.runner.spec_stats[name] += result[name]
        self.records.append(result)

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
        if self.worker is not None:
            if not self.colocated:
                self.worker.close()
            self.worker = None
