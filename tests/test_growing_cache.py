from copy import deepcopy
from dataclasses import asdict
import gc
import os
import time
from types import SimpleNamespace

import pytest
import torch
from transformers.cache_utils import DynamicCache

from inference_scaling.arllm.backends.growing_cache import block_allocated_cache
from inference_scaling.arllm.backends.transformers_backend import TransformersBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest


@pytest.fixture(autouse=True)
def legacy_cache_api():
    if not {"key_cache", "value_cache", "_seen_tokens"} <= vars(DynamicCache()).keys():
        pytest.skip("optional block cache storage uses the Transformers 4 legacy API")


def test_block_storage_matches_append_crop_repeat_and_reorder():
    expected = DynamicCache()
    actual = block_allocated_cache(DynamicCache(), 8)
    generator = torch.Generator().manual_seed(91)
    with torch.inference_mode():
        for step in range(21):
            for layer in range(2):
                k = torch.randn((2, 3, 1, 4), generator=generator)
                v = torch.randn((2, 3, 1, 6), generator=generator)
                left = expected.update(k, v, layer)
                right = actual.update(k, v, layer)
                assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))
        assert actual.get_seq_length() == 21
        assert actual._key_storage[0].shape[-2] == 24
        assert actual.copied_prefix_elements < 2 * 2 * 3 * 10 * sum(range(21))
        for cache in (expected, actual):
            cache.crop(13)
            cache.batch_repeat_interleave(2)
            cache.reorder_cache(torch.tensor([3, 1, 2, 0]))
        for layer in range(2):
            k = torch.randn((4, 3, 5, 4), generator=generator)
            v = torch.randn((4, 3, 5, 6), generator=generator)
            assert all(torch.equal(a, b) for a, b in zip(expected.update(k, v, layer), actual.update(k, v, layer), strict=True))
        assert actual.get_seq_length() == expected.get_seq_length() == 18
        clone = deepcopy(actual)
        clone.crop(7)
        assert actual.get_seq_length() == 18 and clone.get_seq_length() == 7


def test_existing_prefix_and_unsupported_cache_are_checked():
    original = DynamicCache()
    original.update(torch.ones((1, 1, 11, 4)), torch.ones((1, 1, 11, 4)), 0)
    growing = block_allocated_cache(original, 8)
    assert growing.get_seq_length() == 11
    assert growing.key_cache[0] is original.key_cache[0]
    with pytest.raises(RuntimeError, match="ordinary DynamicCache"):
        block_allocated_cache((), 8)
    with pytest.raises(ValueError, match="positive"):
        block_allocated_cache(DynamicCache(), 0)


@pytest.mark.parametrize("batch", [1, 2])
def test_qwen_cache_layout_preserves_tokens_probabilities_and_model_cost(batch):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(31)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=23, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16))
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2, bos_token_id=1)
    backend = TransformersBackend(model, tokenizer, device="cpu", model_id="fixture", score_chunk_size=7)
    requests = [GenerationRequest((1, 3, 4, 5) * 4, 65, SamplingConfig(temperature=0.6), 71 + i, str(i))
                for i in range(batch)]
    baseline = backend.sample_batch(requests)
    before = backend.snapshot()
    backend.configure_cache_growth(8)
    accelerated = backend.sample_batch(requests)
    after = backend.snapshot()
    assert accelerated == baseline
    assert after.estimated_dense_forward_flops - before.estimated_dense_forward_flops == before.estimated_dense_forward_flops
    assert after.generation_forward_token_slots == 2 * before.generation_forward_token_slots
    backend.configure_cache_growth(0)
    assert backend.sample_batch(requests) == baseline


@pytest.mark.skipif(not os.environ.get("INFERENCE_SCALING_GPU_MODEL"),
                    reason="optional local-model GPU equivalence and timing check")
def test_local_gpu_cache_equivalence_and_timing():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    backend = TransformersBackend.from_pretrained(os.environ["INFERENCE_SCALING_GPU_MODEL"],
        device="cuda", dtype="bfloat16", local_files_only=True, attn_implementation="sdpa",
        max_score_batch_size=1, score_chunk_size=256)
    try:
        unit = backend.encode("Consider the integers from one to ten. Add them in increasing order. ", add_special_tokens=False)
        prefix = (unit * (8192 // len(unit) + 1))[:8192]
        request = GenerationRequest(prefix, 512, SamplingConfig(temperature=0.6), 3187, "cache-equivalence")
        backend.sample_batch([GenerationRequest(prefix[:128], 4, SamplingConfig(), 1, "warmup")])
        expected = expected_cost = None
        times = {0: [], 512: []}
        for block in (0, 512, 512, 0):
            backend.configure_cache_growth(block)
            torch.cuda.synchronize()
            before = asdict(backend.snapshot())
            start = time.perf_counter()
            sample = backend.sample_batch([request])[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            after = asdict(backend.snapshot())
            cost = {key: after[key] - before[key] for key in before}
            if expected is None:
                expected, expected_cost = sample, cost
            else:
                assert sample == expected
                assert cost == expected_cost
            times[block].append(elapsed)
            print(f"cache block={block} prefix={len(prefix)} generated={len(sample.token_ids)} "
                  f"seconds={elapsed:.3f} forward_tokens={cost['generation_forward_token_slots']}", flush=True)
        print(f"cache mean_speedup={sum(times[0]) / sum(times[512]):.4f}", flush=True)
    finally:
        del backend
        gc.collect()
        torch.cuda.empty_cache()
