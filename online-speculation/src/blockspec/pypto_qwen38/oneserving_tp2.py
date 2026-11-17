"""Read-only Orthrus draft and bounded target verification for OneServing TP2."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

from safetensors import safe_open
import torch
from torch.nn.functional import pad

from serving.runtime import BLOCK_SIZE, HIDDEN, TensorParallelRunner, TP_VOCAB, shared
from serving.weight import TP_BANK_SPECS, TextWeights

from .head import POLICY, tensor_schema
from .oneserving_tp2_programs import GROUP_LAYERS, make_draft_program


BLOCK = 32
VERIFY_ROWS = 4
LAYERS = (60, 61, 62, 63)
DECAY_BANK = next(index for index, (name, _, _) in enumerate(TP_BANK_SPECS) if name == "delta_a_log")
SHAPES = (
    ((3 * 5120, 5120), torch.bfloat16), ((3 * 3072, 5120), torch.bfloat16),
    ((3 * 5120, 3072), torch.bfloat16), ((3 * 128, 5120), torch.bfloat16),
    ((3 * 128, 5120), torch.bfloat16), ((3 * 4, 5120), torch.bfloat16),
    ((3, 32), torch.float32), ((3, 32), torch.bfloat16), ((3, 128), torch.bfloat16),
    ((6144, 5120), torch.bfloat16), ((512, 5120), torch.bfloat16),
    ((512, 5120), torch.bfloat16), ((1, 256), torch.float32),
    ((1, 256), torch.float32), ((5120, 3072), torch.bfloat16),
)
INPUTS = (
    ("orthrus_ids", (BLOCK,), torch.int32), ("orthrus_rows", (BLOCK,), torch.int32),
    ("orthrus_cos", (BLOCK, 64), torch.bfloat16), ("orthrus_sin", (BLOCK, 64), torch.bfloat16),
    ("orthrus_table", (512 // BLOCK_SIZE,), torch.int32), ("orthrus_mask", (1, 512), torch.float32),
    ("orthrus_valid", (BLOCK,), torch.int32),
)


def digest(path):
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            sha256.update(part)
    return sha256.hexdigest()


def checkpoint_manifest(directory, model_root):
    manifest = json.loads((directory / "manifest.json").read_text())
    head = manifest["source_head_contract"]
    if (manifest["tp"] != 2 or manifest["block_size"] != BLOCK
            or manifest["layers"] != list(LAYERS) or head["hybrid_policy"] != POLICY
            or head["base_config_sha256"] != digest(model_root / "config.json")
            or head["base_index_sha256"] != digest(model_root / "model.safetensors.index.json")
            or not 0 <= head["mask_token_id"] < 2 * TP_VOCAB):
        raise ValueError("Orthrus TP2 suffix must match the base model and K32 hybrid policy")
    schema = {f"layers.{layer}.attention.draft.{name}": (name, shape)
              for layer in LAYERS for name, shape in tensor_schema(layer, 2).items()}
    for rank, shard in enumerate(manifest["rank_files"]):
        expected = f"rank{rank}.safetensors"
        if shard["name"] != expected or digest(directory / expected) != shard["sha256"]:
            raise ValueError(f"checksum mismatch in {expected}")
        with safe_open(directory / expected, framework="pt", device="cpu") as weights:
            if set(weights.keys()) != schema.keys():
                raise ValueError(f"tensor schema mismatch in {expected}")
            for key, (name, shape) in schema.items():
                tensor = weights.get_slice(key)
                dtype = "F32" if name == "A_log" else "BF16"
                if tuple(tensor.get_shape()) != shape or tensor.get_dtype() != dtype:
                    raise ValueError(f"invalid {key} in {expected}")
    if len(manifest["rank_files"]) != 2:
        raise ValueError("expected two Orthrus weight shards")
    return manifest


def rank_weights(path):
    """Match the packed worker's local projection layout and constant padding."""
    with safe_open(path, framework="pt", device="cpu") as weights:
        def read(layer, name):
            return weights.get_tensor(f"layers.{layer}.attention.draft.{name}")

        def stack(name, convert=None):
            values = [read(layer, name) for layer in LAYERS[:3]]
            return torch.cat([convert(value) for value in values] if convert else values)

        def projection(value):
            return pad(value, (0, 0, 0, 128 - value.shape[0]))

        def constant(value):
            return pad(value.reshape(1, -1), (0, 8))

        def gamma(name):
            return (1 + read(63, name).float()).reshape(1, -1)

        yield stack("in_proj_qkv.weight")
        yield stack("in_proj_z.weight")
        yield stack("out_proj.weight")
        yield stack("in_proj_a.weight", projection)
        yield stack("in_proj_b.weight", projection)
        yield stack("conv1d.weight", lambda value: value.squeeze(1).T.contiguous())
        yield stack("A_log", constant)
        yield stack("dt_bias", constant)
        yield stack("norm.weight", lambda value: value.reshape(1, -1))
        yield read(63, "q_proj.weight")
        yield read(63, "k_proj.weight")
        yield read(63, "v_proj.weight")
        yield gamma("q_norm.weight")
        yield gamma("k_norm.weight")
        yield read(63, "o_proj.weight")


def check_chunk(sequence, offset, selected):
    """Return outputs, committed inputs, accepted guesses and whether the round ended."""
    for index, token in enumerate(selected):
        next_index = offset + index + 1
        if next_index == len(sequence) or token != sequence[next_index]:
            return selected[:index + 1], index + 1, index, True
    return selected, len(selected), len(selected), False


class OrthrusTP2Runner(TensorParallelRunner):
    def __init__(self, config, head_directory, *, output_candidates=31, on_cache_event=None):
        if (config.speculative_tokens or config.prefix_cache or config.max_model_len != 512
                or config.long_prefill_token_threshold != VERIFY_ROWS
                or config.max_num_batched_tokens < config.max_num_seqs * VERIFY_ROWS):
            raise ValueError("Orthrus TP2 requires capped prefill, 512 context, and native MTP/prefix cache off")
        if not 1 <= output_candidates < BLOCK:
            raise ValueError("candidate budget must be in [1, 31]")
        self.head_directory = Path(head_directory)
        self.head = checkpoint_manifest(self.head_directory, config.model_root)
        self.output_candidates = output_candidates
        super().__init__(config, on_cache_event=on_cache_event)
        self._host_slabs.update({name: shared((2, *shape), dtype) for name, shape, dtype in INPUTS})
        self.spec_stats.update(rounds=0, verify_calls=0, restore_ns=0)

    def compile_programs(self):
        entries = super().compile_programs()
        self.orthrus = self.compile_draft()
        return [*entries, self.orthrus]

    def _target_decode_row_limit(self):
        # Keep native MTP disabled while authorizing short Orthrus verification calls.
        return VERIFY_ROWS if self.output_candidates else 1

    def compile_draft(self):
        def meta(shape, dtype):
            return torch.empty((2, *shape), dtype=dtype, device="meta")

        sample_inputs = {name: meta(shape, dtype) for name, shape, dtype in INPUTS}
        banks = [meta(shape, getattr(torch, dtype)) for _, shape, dtype in TP_BANK_SPECS[:23]]
        banks[DECAY_BANK] = meta(TP_BANK_SPECS[DECAY_BANK][1], torch.float32)
        suffix = [meta(shape, dtype) for shape, dtype in SHAPES]
        state = [meta(shape, dtype) for shape, dtype in self.state_specs[:4]]
        draft = make_draft_program(pool_pages=self.pool_blocks)
        return draft.compile(sample_inputs["orthrus_ids"], sample_inputs["orthrus_rows"],
                             *banks, *suffix, *state,
                             *(sample_inputs[name] for name, _, _ in INPUTS[2:-1]),
                             meta((BLOCK, HIDDEN), torch.bfloat16),
                             meta((BLOCK, TP_VOCAB), torch.bfloat16), sample_inputs["orthrus_valid"],
                             meta((BLOCK, 16), torch.float32), 0, 64, BLOCK, 0, config=self.runconfig)

    def _check_memory_admission(self):
        super()._check_memory_admission()
        extra = sum(torch.empty(shape, dtype=dtype, device="meta").nbytes for shape, dtype in SHAPES)
        extra += BLOCK * ((HIDDEN + TP_VOCAB) * 2 + 16 * 4)
        for rank in self.admission:
            rank["orthrus_extra_bytes"] = extra
            if rank["headroom_bytes"] < extra:
                raise RuntimeError(f"Insufficient NPU memory for resident Orthrus suffix on rank {rank['rank']}")

    def start(self):
        super().start()
        try:
            loaders = [rank_weights(self.head_directory / f"rank{rank}.safetensors") for rank in range(2)]
            self.suffix = []
            for (shape, dtype), first, second in zip(SHAPES, *loaders, strict=True):
                if any(value.shape != shape or value.dtype != dtype for value in (first, second)):
                    raise ValueError("Incorrect resident suffix shape or dtype")
                self.suffix.append(self._allocate(shape, dtype, (first, second)))
            self.draft_hidden = self._allocate((BLOCK, HIDDEN), torch.bfloat16)
            self.draft_logits = self._allocate((BLOCK, TP_VOCAB), torch.bfloat16)
            self.draft_decisions = self._allocate((BLOCK, 16), torch.float32)
            self.host_decisions = torch.empty((2, BLOCK, 16), dtype=torch.float32)
            text = TextWeights(self.settings.model_root)
            decay_shape = TP_BANK_SPECS[DECAY_BANK][1]
            decay = torch.stack([text.layer(layer, "linear_attn.A_log") for layer in text.linear_layers])
            decay = [pad(value.bfloat16().float(), (0, 8)).contiguous() for value in decay.chunk(2, dim=1)]
            self.draft_banks = self.banks[:23]
            self.draft_banks[DECAY_BANK] = self._allocate(decay_shape, torch.float32, decay)
        except BaseException:
            self.close()
            raise

    def propose(self, slot, candidates):
        request = slot.request
        record = self.requests[request.request_id]
        position = slot.position
        if record["pages"].length != position:
            raise RuntimeError("Draft anchor must follow the committed history")
        # Match the trained K32 block. Near the context limit, only in-range rows are visible.
        width = min(BLOCK, self.settings.max_model_len - position)
        if candidates >= width:
            raise RuntimeError("Draft prediction budget exceeds block width")
        start = time.monotonic_ns()
        ids, rows, cos, sin, table, mask, valid = (self._host_slabs[name] for name, _, _ in INPUTS)
        ids.fill_(self.head["source_head_contract"]["mask_token_id"])
        ids[:, 0] = slot.token_ids[0]
        rows.zero_()
        rows[:, width:] = -1
        cos.fill_(1)
        sin.zero_()
        cos[:, :width].copy_(self.cos[position:position + width])
        sin[:, :width].copy_(self.sin[position:position + width])
        table.copy_(self.tables[:, record["slot"]])
        mask.fill_(-10000)
        mask[:, :, :position + width] = 0
        valid.zero_()
        valid[:, :candidates] = 1
        for begin in range(0, 64, GROUP_LAYERS):
            self._worker_call("run", self.orthrus, ids, rows, *self.draft_banks, *self.suffix,
                              *self.resident[:4], cos, sin, table, mask, self.draft_hidden,
                              self.draft_logits, valid, self.draft_decisions, record["slot"],
                              position, width, begin, config=self.runconfig)
        for rank, shard in enumerate(self.draft_decisions.shards):
            value = self.host_decisions[rank]
            self._worker_call("copy_from", value.data_ptr(), shard.data_ptr, value.nbytes, worker_id=rank)
        selected = self._merge_packed_decisions(self.host_decisions, range(candidates), BLOCK)
        self.spec_stats["draft_ns"] += time.monotonic_ns() - start
        self.spec_stats["rounds"] += 1
        self.spec_stats["proposed"] += candidates
        return [selected[i] for i in range(candidates)]

    def run_batch(self, slots):
        if any(slot.request.sampling.temperature != 0 for slot in slots):
            raise ValueError("Orthrus TP2 currently supports greedy requests")
        # Reuse native prefill and single-token fallback when the draft has no work.
        budgets = [min(self.output_candidates, slot.request.max_tokens - len(slot.request.generated) - 1)
                   if slot.phase == "decode" else 0 for slot in slots]
        if not any(budgets):
            return super().run_batch(slots)

        proposed = [list(slot.token_ids) + (self.propose(slot, budget) if budget else [])
                    for slot, budget in zip(slots, budgets, strict=True)]
        results = [None if slot.phase == "prefill" and not slot.completes_prefill else [] for slot in slots]
        offsets = [0] * len(slots)
        pending = list(range(len(slots)))
        while pending:
            chunks = [proposed[i][offsets[i]:offsets[i] + VERIFY_ROWS] for i in pending]
            batch_slots = [replace(slots[i], token_ids=tuple(chunk), position=slots[i].position + offsets[i])
                           for i, chunk in zip(pending, chunks, strict=True)]
            for slot, chunk in zip(batch_slots, chunks, strict=True):
                self._reserve(slot, slot.position + len(chunk))
            can_reject = any(budgets[i] for i in pending)
            start = time.monotonic_ns()
            batch, _ = self._execute("target", batch_slots, chunks, [s.position for s in batch_slots],
                                     snapshot_mode=int(can_reject))
            self.spec_stats["verify_ns"] += time.monotonic_ns() - start
            self.spec_stats["verify_calls"] += 1
            self.control_ids.fill_(-1)
            self.control_values.fill_(-1)
            committed = []
            unfinished = []
            for j, (i, slot, chunk) in enumerate(zip(pending, batch_slots, chunks, strict=True)):
                if slot.phase == "prefill":
                    count = len(chunk)
                    if slot.completes_prefill:
                        results[i].append(self._distribution(batch, j, count - 1, slot.request))
                else:
                    target = [self._distribution(batch, j, row, slot.request) for row in range(len(chunk))]
                    emitted, count, accepted, done = check_chunk(proposed[i], offsets[i], target)
                    results[i].extend(emitted)
                    self.spec_stats["accepted"] += accepted
                    bonus = offsets[i] + count == len(proposed[i])
                    self.spec_stats["rejected_rounds"] += int(done and not bonus)
                    if not done:
                        unfinished.append(i)
                record = self.requests[slot.request.request_id]
                if count < len(chunk):
                    self.control_ids[:, j] = record["slot"]
                    self.control_values[:, j] = count
                committed.append((record["pages"], slot.position + count))
                offsets[i] += count
            if bool((self.control_ids >= 0).any()):
                start = time.monotonic_ns()
                self._worker_call("run", self.restore, self.control_ids, self.control_values,
                                  self.resident[4], self.resident[5], self.resident[0], self.resident[1],
                                  config=self.runconfig)
                self.spec_stats["restore_ns"] += time.monotonic_ns() - start
            for pages, end in committed:
                self.cache.commit(pages, end)
            pending = unfinished
        return results
