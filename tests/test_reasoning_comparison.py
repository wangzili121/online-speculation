from dataclasses import replace
from types import SimpleNamespace

import pytest

from experiments.arllm.reasoning_methods import (
    budget_plan, check_budget, compare_sir, compare_mh, majority_index, REWARDS, sampling_policy, reward_temperature,
    single_sample_budget_length, generation_cost,
)
from experiments.arllm.request_reuse import ColdCostRequestReplay
from experiments.arllm.reasoning_benchmark import build_parser
from inference_scaling.arllm.types import GenerationRequest
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.backends.tabular import TabularAutoregressiveBackend
from inference_scaling.arllm.backends.transformers_backend import TransformersBackendSnapshot


def test_cache_growth_is_an_explicit_execution_option():
    assert build_parser().parse_args([]).cache_growth_tokens == 0
    assert build_parser().parse_args(["--cache-growth-tokens", "512"]).cache_growth_tokens == 512


class CountedBackend(TabularAutoregressiveBackend):
    parameter_count = 7
    tokenizer = SimpleNamespace(eos_token_id=2, get_vocab=lambda: {"a": 0, "b": 1, "eos": 2})

    def __init__(self, fallback=(0.6, 0.2, 0.2)):
        super().__init__({}, fallback=fallback)
        self.counts = TransformersBackendSnapshot(*([0] * 10))

    def _charge(self, key, count):
        self.counts = replace(self.counts, **{key: getattr(self.counts, key) + count},
                             estimated_dense_forward_flops=self.counts.estimated_dense_forward_flops + 14 * count)

    def sample_batch(self, requests):
        samples = super().sample_batch(requests)
        self._charge("generation_forward_token_slots", sum(len(r.prefix) + len(s.token_ids) - 1 for r, s in zip(requests, samples, strict=True)))
        self.counts = replace(self.counts, sample_calls=self.counts.sample_calls + 1,
            sampled_sequences=self.counts.sampled_sequences + len(samples),
            generated_tokens=self.counts.generated_tokens + sum(len(sample.token_ids) for sample in samples),
            prefill_tokens=self.counts.prefill_tokens + sum(len(request.prefix) for request in requests))
        return samples

    def score_batch(self, requests):
        self._charge("score_forward_token_slots", sum(len(r.prefix) + len(tokens) - 1 for r in requests for tokens in r.continuations))
        return super().score_batch(requests)

    def score_statistics_batch(self, requests, **kwargs):
        self.score_batch(requests)
        return [SimpleNamespace(token_topk_confidences=tuple(1 + 0.1 * token for token in tokens)) for r in requests for tokens in r.continuations]

    def decode(self, tokens, **kwargs):
        return str(tokens[0]) if tokens else ""

    def encode(self, text, **kwargs):
        return ()

    def snapshot(self):
        return self.counts


class Judge:
    def grade(self, prediction, reference):
        return {"correct": prediction == reference, "parseable": bool(prediction)}

    def equivalent(self, left, right):
        return bool(left and left == right)

    def answer_key(self, text):
        return text or None


@pytest.mark.parametrize("flag,value", [("--proposal-model", "org/other"), ("--mh-iterations", "1"),
                                       ("--thinking-mode", "enabled")])
def test_comparison_cli_rejects_unimplemented_shared_overrides(flag, value):
    with pytest.raises(SystemExit):
        build_parser().parse_args([flag, value])


@pytest.mark.parametrize("flag", ["--limit", "--draws", "--budgets", "--candidate-counts"])
def test_comparison_cli_rejects_nonpositive_work_before_model_loading(flag):
    with pytest.raises(SystemExit):
        build_parser().parse_args([flag, "0"])


def test_comparison_cli_keeps_generic_model_and_output_options():
    args = build_parser().parse_args(["--model", "org/model", "--model-revision", "fixed", "--allow-download",
                                     "--max-new-tokens", "32768", "--thinking-format", "json"])
    assert args.model == "org/model" and args.model_revision == "fixed" and args.allow_download
    assert args.max_new_tokens == 32768 and args.thinking_format == "json"


def test_sampling_configuration_is_applied_and_mh_support_is_checked():
    config = {"sampling": {"temperature": 0.7, "top_p": 0.9, "top_k": 20}}
    assert sampling_policy(config, eos_token_id=2) == SamplingConfig(0.7, 0.9, 20, 2)
    with pytest.raises(ValueError, match="reference policy"):
        sampling_policy(config, require_full_support=True)
    with pytest.raises(ValueError, match="unsupported"):
        sampling_policy({"sampling": {"temprature": 0.6}})


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf"), float("nan")])
def test_reward_temperature_rejects_invalid_values(temperature):
    with pytest.raises(ValueError, match="finite and positive"):
        reward_temperature("consilience", {"comparison": {"consilience_temperature": temperature}})


def test_sir_log_probability_reuses_the_configured_reward_scale():
    backend = CountedBackend()
    samples = [{"token_ids": (0, 2), "token_logprobs": (-0.4, -1.0)},
               {"token_ids": (1, 2), "token_logprobs": (-1.4, -1.0)}]
    result = compare_sir(backend=backend, judge=Judge(), reference="0", prompt=(0,),
        config={"reward": {"logprob_scale": 2.0}}, plan=budget_plan(128, 2, 1, 16),
        samples=samples, pilots=[], source="sequence_log_probability", seed=8,
        render_output=output, score_cache={})
    assert result["rewards"] == [-2.8, -4.8]
    assert backend.snapshot().score_forward_token_slots == 0


def output(backend, prompt, tokens, config):
    return {"content_text": str(tokens[0]), "thinking_status": "complete"}


def test_budget_reserves_all_generation_scoring_and_pilot_tokens():
    plan = budget_plan(32768, 2, 120, 32768)
    assert plan["max_new_tokens"] == 8072
    assert plan["reserved_forward_tokens"] == 32768
    with pytest.raises(ValueError):
        budget_plan(4, 2, 100, 1)
    with pytest.raises(RuntimeError, match="exceeds"):
        check_budget({"generation_forward_token_slots": 10, "score_forward_token_slots": 2}, 11)


def test_single_sample_can_spend_its_budget_without_candidate_reserves():
    length = single_sample_budget_length(32768, 120, 32768)
    assert length == 32649
    assert check_budget(generation_cost(120, length, 7), 32768) == 32768
    assert length > budget_plan(32768, 2, 120, 32768)["max_new_tokens"]
    assert single_sample_budget_length(131072, 120, 32768) == 32768
    with pytest.raises(ValueError, match="insufficient"):
        single_sample_budget_length(10, 11, 32768)


@pytest.mark.parametrize("existing", ["empty", "complete", "partial"])
def test_budget_base_reuses_eos_pools_and_preserves_shorter_resume_artifacts(tmp_path, monkeypatch, existing):
    from copy import deepcopy
    import experiments.arllm.reasoning_benchmark as benchmark
    from experiments.shared.artifacts import json_fingerprint, load_jsonl, write_json_atomic
    from experiments.shared.math_benchmark import MathProblem

    backend, calls = CountedBackend(), []
    config = {"sampling": {"temperature": 0.6}, "generation": {"max_new_tokens": 16}}
    args = SimpleNamespace(output=tmp_path, draws=1, seed=17, reuse_identical_requests=False,
                           budgets=[12, 32], candidate_counts=[2, 4], methods=["budget_base", "base"], rewards=[])
    problem = MathProblem("one", "question", "0", "algebra", 5)
    full_tokens = (0, 0, 0, 0, 2)

    def item(tokens):
        return {"token_ids": tokens, "token_logprobs": [-0.5] * len(tokens), "ended_by_eos": tokens[-1] == 2,
                "correct": tokens[-1] == 2, "cost": {"seconds": 0.0}}

    def generate(backend, judge, problem, config, seed, mode):
        assert config["generation"]["max_new_tokens"] == 16
        calls.append((mode, seed))
        return item(full_tokens)

    def render(backend, prompt, tokens, config):
        disabled = config["output"]["thinking_mode"] == "disabled"
        complete = bool(tokens and tokens[-1] == 2)
        return {"content_text": "0" if disabled or complete else "",
                "thinking_status": "disabled" if disabled else "complete" if complete else "incomplete"}

    monkeypatch.setattr(benchmark, "model_prompt", lambda *args: (0,))
    monkeypatch.setattr(benchmark, "generation_config_for_prompt", lambda config, *args: (deepcopy(config), {"effective_max_new_tokens": 16}))
    monkeypatch.setattr(benchmark, "run_base", generate)
    monkeypatch.setattr(benchmark, "visible_output", render)
    monkeypatch.setattr(benchmark, "summarize", lambda *args: None)
    pool_path = tmp_path / "pools" / (json_fingerprint(["one", 0])[:20] + ".json")
    if existing != "empty":
        tokens = full_tokens if existing == "complete" else full_tokens[:3]
        write_json_atomic(pool_path, {"fingerprint": "same", "samples": {
            f"candidate:0:{mode}": item(tokens) for mode in ("disabled", "enabled")}})
    benchmark.run_comparisons(backend, Judge(), [problem], config, args, "same")
    assert len(calls) == (0 if existing == "complete" else 2)
    records = load_jsonl(tmp_path / "comparisons.jsonl")
    assert len(records) == 8
    assert all(row["correct"] for row in records if row["method"] == "budget_base_enabled")
    assert not any(row["correct"] for row in records if row["method"] == "base_enabled")
    assert all(row["used_forward_tokens"] <= row["budget_forward_tokens"] for row in records)
    if existing == "partial":
        import json
        saved = json.loads(pool_path.read_text(encoding="utf-8"))["samples"]
        assert len(saved["candidate:0:enabled"]["token_ids"]) == 3
        assert saved["candidate:0:enabled:max=16"]["ended_by_eos"]
    benchmark.run_comparisons(backend, Judge(), [problem], config, args, "same")
    assert load_jsonl(tmp_path / "comparisons.jsonl") == records
    assert len(calls) == (0 if existing == "complete" else 2)


def test_majority_ignores_missing_answers_and_uses_stable_ties():
    assert majority_index(["1", "2", "2"], Judge()) == 1
    assert majority_index(["1", "2"], Judge()) == 0
    assert majority_index(["", "", "2"], Judge()) == 2


def test_identical_request_reuse_preserves_samples_and_cold_cost():
    raw, independent = CountedBackend(), CountedBackend()
    replay = ColdCostRequestReplay(raw)
    request = GenerationRequest((0,), 32, SamplingConfig(eos_token_id=2), 81, "first")
    first = replay.sample_batch([request])[0]
    expected_first = independent.sample_batch([request])[0]
    assert first == expected_first
    same = replace(request, request_id="second")
    assert replay.sample_batch([same])[0] == independent.sample_batch([same])[0]
    assert replay.snapshot() == independent.snapshot()
    assert raw.snapshot().generation_forward_token_slots * 2 == replay.snapshot().generation_forward_token_slots
    assert replay.cache_hits == 1


def test_completed_eos_request_can_be_reused_at_a_longer_limit():
    raw, independent = CountedBackend(), CountedBackend()
    replay = ColdCostRequestReplay(raw)
    request = GenerationRequest((0,), 32, SamplingConfig(eos_token_id=2), 81, "first")
    sample = replay.sample_batch([request])[0]
    independent.sample_batch([request])
    assert sample.finish_reason == "eos"
    longer = replace(request, max_new_tokens=64)
    assert replay.sample_batch([longer])[0] == independent.sample_batch([longer])[0]
    assert replay.snapshot() == independent.snapshot()
    assert replay.cache_hits == 1


def test_prefetched_initial_draw_preserves_each_horizon_and_independent_cost():
    raw, independent = CountedBackend(), CountedBackend()
    replay = ColdCostRequestReplay(raw, prefetch_limits={(0,): 32})
    for length in (8, 32, 16, 8):
        request = GenerationRequest((0,), length, SamplingConfig(), 81, str(length))
        assert replay.sample_batch([request]) == independent.sample_batch([request])
        assert replay.snapshot() == independent.snapshot()
    assert raw.snapshot().sample_calls == 1
    assert raw.snapshot().generated_tokens == 32
    assert replay.cache_hits == 3
    with pytest.raises(ValueError, match="positive integers"):
        ColdCostRequestReplay(raw, prefetch_limits={(0,): 0})


@pytest.mark.parametrize("cache_block", [0, 8])
def test_qwen_prefetch_preserves_token_probabilities_and_cost(cache_block):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from inference_scaling.arllm.backends.transformers_backend import TransformersBackend

    with torch.random.fork_rng():
        torch.manual_seed(19)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=23, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16))
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2, bos_token_id=1)
    raw = TransformersBackend(model, tokenizer, device="cpu", model_id="fixture", score_chunk_size=7)
    independent = TransformersBackend(model, tokenizer, device="cpu", model_id="fixture", score_chunk_size=7)
    if cache_block:
        from transformers.cache_utils import DynamicCache
        if not {"key_cache", "value_cache", "_seen_tokens"} <= vars(DynamicCache()).keys():
            pytest.skip("optional cache growth requires the legacy storage API")
        raw.configure_cache_growth(cache_block)
        independent.configure_cache_growth(cache_block)
    prefix = (1, 3, 4, 5) * 4
    replay = ColdCostRequestReplay(raw, prefetch_limits={prefix: 33})
    for length in (9, 33, 17, 9):
        request = GenerationRequest(prefix, length, SamplingConfig(temperature=0.6), 81, str(length))
        assert replay.sample_batch([request]) == independent.sample_batch([request])
        assert replay.snapshot() == independent.snapshot()
    assert raw.snapshot().sample_calls == 1
    assert raw.snapshot().generated_tokens == 33


def test_prefetch_does_not_extend_explicit_random_draws_or_other_prefixes():
    raw, independent = CountedBackend(), CountedBackend()
    replay = ColdCostRequestReplay(raw, prefetch_limits={(0,): 32})
    requests = [GenerationRequest((0,), 8, SamplingConfig(), 81, "explicit", uniforms=(0.1,) * 8),
                GenerationRequest((0, 1), 8, SamplingConfig(), 81, "different-prefix")]
    for request in requests:
        assert replay.sample_batch([request]) == independent.sample_batch([request])
        assert replay.snapshot() == independent.snapshot()
    assert raw.snapshot() == independent.snapshot()


def test_prefetched_eos_keeps_the_stop_reason_at_the_boundary():
    raw, independent = CountedBackend(), CountedBackend()
    replay = ColdCostRequestReplay(raw, prefetch_limits={(0,): 32})
    full_request = GenerationRequest((0,), 32, SamplingConfig(eos_token_id=2), 81, "probe")
    probe = CountedBackend().sample_batch([full_request])[0]
    assert probe.finish_reason == "eos" and len(probe.token_ids) > 1
    for length in (len(probe.token_ids) - 1, len(probe.token_ids), 64):
        request = replace(full_request, max_new_tokens=length, request_id=str(length))
        assert replay.sample_batch([request]) == independent.sample_batch([request])
        assert replay.snapshot() == independent.snapshot()
    assert raw.snapshot().sample_calls == 1


@pytest.mark.parametrize("source", REWARDS)
def test_prefetched_mh_initialization_preserves_both_budget_chains(source):
    # Zero EOS mass forces both horizons to their caps, exercising the
    # prefetched suffix rather than only the already-complete EOS case.
    replay = ColdCostRequestReplay(CountedBackend((0.6, 0.4, 0.0)), prefetch_limits={(0,): 24})
    independent = CountedBackend((0.6, 0.4, 0.0))
    sample = {"token_ids": (0, 2), "token_logprobs": (-0.4, -1.0)}
    for budget, candidates in ((64, 2), (256, 4)):
        options = dict(judge=Judge(), prompt=(0,), reference="0", config={"sampling": {"temperature": 0.6}},
            plan=budget_plan(budget, candidates, 1, 24), pilots=[sample, sample], source=source,
            seed=17, render_output=output)
        expected = compare_mh(backend=independent, **options)
        actual = compare_mh(backend=replay, **options)
        for key in expected.keys() - {"seconds"}:
            assert actual[key] == expected[key], key


@pytest.mark.parametrize("source", REWARDS)
def test_mh_request_reuse_preserves_whole_chain_and_accounting(source):
    backend = CountedBackend()
    replay = ColdCostRequestReplay(backend)
    sample = {"token_ids": (0, 2), "token_logprobs": (-0.4, -1.0)}
    options = dict(judge=Judge(), prompt=(0,), reference="0", config={"sampling": {"temperature": 0.6}},
                   plan=budget_plan(128, 2, 1, 16), pilots=[sample, sample], source=source, seed=17, render_output=output)
    first = compare_mh(backend=replay, **options)
    second = compare_mh(backend=replay, **options)
    assert first["trace"] == second["trace"]
    assert first["content"] == second["content"]
    assert first["cost"] == second["cost"]
    assert replay.cache_hits >= 1


@pytest.mark.parametrize("source", REWARDS)
def test_sir_reward_and_budget_ignore_gold_until_evaluation(source):
    backend = CountedBackend()
    samples = [{"token_ids": (0, 2), "token_logprobs": (-0.4, -1.0)},
               {"token_ids": (1, 2), "token_logprobs": (-1.4, -1.0)}]
    options = dict(backend=backend, judge=Judge(), prompt=(0,), config={"sampling": {"temperature": 0.6}},
                   plan=budget_plan(128, 2, 1, 16), samples=samples, pilots=samples,
                   source=source, seed=8, render_output=output, score_cache={})
    left = compare_sir(reference="0", **options)
    right = compare_sir(reference="1", **options)
    assert left["probabilities"] == right["probabilities"]
    assert left["selected_index"] == right["selected_index"]
    assert left["correct"] != right["correct"]
    assert left["used_forward_tokens"] <= 128


@pytest.mark.parametrize("source", REWARDS)
def test_mh_all_model_rewards_run_with_same_token_budget(source):
    backend = CountedBackend()
    sample = {"token_ids": (0, 2), "token_logprobs": (-0.4, -1.0)}
    result = compare_mh(backend=backend, judge=Judge(), prompt=(0,), reference="0",
        config={"sampling": {"temperature": 0.6}}, plan=budget_plan(128, 2, 1, 16),
        pilots=[sample, sample], source=source, seed=17, render_output=output)
    assert result["updates"] == 1
    assert result["used_forward_tokens"] <= 128
    assert result["cost"]["estimated_dense_forward_flops"] == result["used_forward_tokens"] * 14
