#!/usr/bin/env python3
"""Compare the resident A3 draft seed with the original host staging."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import pypto.language as pl
import torch
from models.deepseek_v4_1_flash_a3.config import FLASH, MAX_LOGIT_ROWS
from models.deepseek_v4_1_flash_a3.engram import ENGRAM_K, HostEngram
from models.deepseek_v4_1_flash_a3.orthrus_draft.inputs import draft_engram_prefix_rows, make_draft_seed
from models.deepseek_v4_1_flash_a3.orthrus_draft.resident import DraftGeometry, ResidentDraft, make_draft_input_images
from models.deepseek_v4_1_flash_a3.weights import Checkpoint
from pypto.backend import BackendType
from pypto.runtime import ChipWorker, ExecutionMode, RunConfig
from tokenizers import Tokenizer


def reference_hash(engram, ids, history, start):
    ids = torch.as_tensor(ids, dtype=torch.int64)
    mapped = torch.cat((history[:start], engram.token_map[ids]))
    positions = torch.arange(start, mapped.numel())
    lookback = []
    for shift in range(FLASH.engram_max_ngram_size):
        values = mapped[(positions - shift).clamp_min(0)]
        lookback.append(torch.where(positions >= shift, values, engram.pad_id))
    products = torch.stack(lookback, dim=-1).unsqueeze(0) * engram.multipliers.unsqueeze(1)
    rolling, hashes = products[..., 0], []
    for shift in range(1, FLASH.engram_max_ngram_size):
        rolling = torch.bitwise_xor(rolling, products[..., shift])
        first = (shift - 1) * FLASH.engram_n_heads
        primes = engram.primes[:, first:first + FLASH.engram_n_heads].unsqueeze(1)
        hashes.append(rolling.unsqueeze(-1) % primes)
    return torch.cat(hashes, dim=-1) + engram.offsets.unsqueeze(1), mapped


def check_hashes(engram):
    gen = torch.Generator().manual_seed(31128)
    checked = 0
    for history_len in (0, 1, 2, 3, 4, 127, 128, 512, 8192, 16383):
        history = torch.randint(FLASH.engram_compressed_vocab_size, (history_len,), generator=gen)
        for start in {0, max(0, history_len - 2), history_len}:
            for rows in (0, 1, 3, 8, 16, 32):
                ids = torch.randint(FLASH.vocab_size, (rows,), generator=gen, dtype=torch.int32)
                expected, next_history = reference_hash(engram, ids, history, start)
                for commit in (False, True):
                    engram.history["test"] = history
                    actual = engram.hash_ids(ids, start, request_id="test", commit=commit)
                    assert torch.equal(actual, expected)
                    assert torch.equal(engram.history["test"], next_history if commit else history)
                    if not commit:
                        assert engram.history["test"] is history
                    checked += 1
    del engram.history["test"]
    return checked


def check_seed(block, embeddings, compact, config, worker):
    prefix, capacity = draft_engram_prefix_rows(block), max(16, block)
    hidden, lanes, width = FLASH.hidden_size, FLASH.hc_mult, ENGRAM_K
    expand = make_draft_seed(block, capacity)

    @pl.jit(auto_scope=False)
    def seed(
        words: pl.Tensor[[2, hidden], pl.BF16],
        rows: pl.Tensor[[capacity], pl.INT64],
        packed: pl.Tensor[[2, prefix, width], pl.BF16],
        x: pl.Out[pl.Tensor[[capacity, lanes, hidden], pl.FP32]],
        engram: pl.Out[pl.Tensor[[2, block, width], pl.BF16]],
    ):
        x, engram = expand(words, rows, packed, x, engram)
        return x, engram

    rows = torch.tensor([0] + [1] * (capacity - 1), dtype=torch.int64)
    expected_x = embeddings[rows].float().unsqueeze(1).expand(-1, lanes, -1).contiguous()
    expected_engram = compact[:, torch.arange(block).clamp_max(prefix - 1)].contiguous()
    host_inputs, host_outputs = (embeddings, rows, compact), (expected_x, expected_engram)
    config_binds = [worker.alloc_tensor(x.shape, x.dtype, init=x) for x in host_inputs]
    buffers = [worker.alloc_tensor(x.shape, x.dtype) for x in host_outputs]
    program = seed.compile(*host_inputs, *host_outputs, config=config)

    def upload(buffers, tensors):
        for buffer, tensor in zip(buffers, tensors, strict=True):
            worker.copy_to(buffer.data_ptr, tensor.data_ptr(), tensor.nbytes)

    full_samples, compact_samples = [], []
    for i in range(55):
        for destination, source, samples in (
            (buffers, host_outputs, full_samples),
            ((config_binds[0], config_binds[2]), (embeddings, compact), compact_samples),
        ):
            start = perf_counter()
            upload(destination, source)
            if i >= 5:
                samples.append((perf_counter() - start) * 1000)
    # Repeat with overwritten output buffers so stale seed data cannot pass.
    for _ in range(3):
        for buffer, expected in zip(buffers, host_outputs, strict=True):
            upload((buffer,), (torch.full_like(expected, float("nan")),))
        worker.run(program, *config_binds, *buffers, config=config)
        for expected, buffer in zip(host_outputs, buffers, strict=True):
            actual = torch.empty_like(expected)
            worker.copy_from(actual.data_ptr(), buffer.data_ptr, actual.nbytes)
            assert torch.equal(actual, expected)
    return dict(block=block, exact=True, old_bytes=sum(x.nbytes for x in host_outputs),
                new_bytes=embeddings.nbytes + compact.nbytes, old_upload_ms=median(full_samples),
                new_upload_ms=median(compact_samples))


def check_host_staging(block, engram, embeddings, full, position, anchor, mask):
    prefix = draft_engram_prefix_rows(block)
    geometry = DraftGeometry(block, 512, 1)
    dim = FLASH.qk_rope_head_dim // 2
    rope = torch.arange(2 * geometry.context * dim, dtype=torch.float32).reshape(2, geometry.context, dim)
    draft = ResidentDraft.__new__(ResidentDraft)
    draft.geometry = geometry
    draft.mask_token_id = mask
    draft.inputs = make_draft_input_images(block, geometry.buckets[0])
    draft.graphs = SimpleNamespace(
        weights=SimpleNamespace(lookup_rows=lambda ids: engram.checkpoint.tensor_rows("embed.weight", ids).bfloat16()),
        engram=engram,
        rope_tables=(rope, -rope),
    )
    draft._stage("benchmark", position, [0], anchor, block // 2)
    for fields in draft.inputs.fields[1:]:
        assert torch.equal(fields["draft_embeddings"], embeddings)
        assert torch.equal(fields["engram_rows"], full[:, :prefix])
        assert torch.equal(fields["draft_cos"][:, :block], rope[:, position:position + block])
        assert torch.equal(fields["draft_sin"][:, :block], -rope[:, position:position + block])
    assert engram.history["benchmark"].numel() == position
    assert draft.inputs.fields[2]["logit_row_indices"].tolist() == (
        list(range(block // 2)) + [-1] * (MAX_LOGIT_ROWS - block // 2)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = Checkpoint(args.checkpoint)
    engram = HostEngram(checkpoint, Tokenizer.from_file(str(args.checkpoint / "tokenizer.json")))
    checks = check_hashes(engram)
    tokens = [112, 182, 2013, 785, 55]
    engram.hash_ids(tokens, request_id="benchmark")
    mask, anchor = 128803, 455
    embeddings = checkpoint.tensor_rows("embed.weight", torch.tensor([anchor, mask])).bfloat16().contiguous()
    provisional = [anchor] + [mask] * (draft_engram_prefix_rows(16) - 1)
    prefix = engram.lookup(provisional, len(provisional), len(tokens), request_id="benchmark", commit=False)
    full = engram.lookup([anchor] + [mask] * 31, 32, len(tokens), request_id="benchmark", commit=False)
    assert torch.equal(full, prefix[:, torch.arange(32).clamp_max(prefix.shape[1] - 1)])
    config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                       device_id=args.device)
    widths = (8, 16, 32)
    for width in widths:
        check_host_staging(width, engram, embeddings, full, len(tokens), anchor, mask)
    with ChipWorker(config) as worker:
        cases = [check_seed(block, embeddings, prefix, config, worker) for block in widths]
    report = dict(passed=True, hash_cases=checks, host_staging_widths=widths, seed=cases, weights="real W8A8 checkpoint",
                  timing_scope="single-rank transfer of embedding and Engram payloads")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
