from dataclasses import asdict, replace

import pytest
import torch

from blockspec_ablation.benchmark import (BenchmarkConfig, aggregate, benchmark_offline, benchmark_streams,
                                compare_tokens, comparison_summary, continuation_prompts, stream_trajectory)
from blockspec_ablation.checkpoint import adapter_state, base_fingerprint
from blockspec_ablation.decoding import Generation, generate_ar, generate_speculative
from blockspec_ablation.diffusion import UniformNoise
from blockspec_ablation.model import Decoder, ModelConfig
from blockspec_ablation.online import OnlineConfig, OnlineLearner
from blockspec.sampling import SamplingConfig
from blockspec_ablation.tree import generate_tree


def test_aggregate_counts_all_time_not_mean_tps():
    rows = [Generation([1] * 10, 1, 10, 10).summary(),
            Generation([1] * 30, 9, 30, 30, updates=2, update_seconds=3).summary()]
    result = aggregate(rows, setup_seconds=2)
    assert result["tps"] == 4
    assert result["tps_including_learner_setup"] == 40 / 12
    assert result["updates"] == 2 and result["update_seconds"] == 3
    assert result["tokens_per_decode_forward"] == 1


def test_trajectory_accumulates_in_order_with_one_learner_initialization():
    rows = [Generation([1] * 10, 1, 4, 2, accepted=6, updates=1, update_seconds=.2, feedback_blocks=1,
                       fully_covered_rounds=1, coverage_skips=1).summary(),
            Generation([1] * 30, 9, 12, 6, accepted=18, updates=2, update_seconds=.3, feedback_blocks=2,
                       fully_covered_rounds=4, coverage_skips=1).summary()]
    rows[0].update(adapter_version_start=0, adapter_version=1, last_update_loss=.7)
    rows[1].update(adapter_version_start=1, adapter_version=3, last_update_loss=.4)
    actual = stream_trajectory(rows, setup_seconds=2, engine_setup_seconds=3)
    assert actual[0]["cumulative"]["tps_including_learner_setup"] == 10 / 3
    end = actual[1]["cumulative"]
    assert end["tps"] == 4 and end["tps_including_learner_setup"] == 40 / 12
    assert end["tps_including_all_setup"] == 40 / 15
    assert end["tokens_per_round"] == 5 and end["requests"] == 2
    assert end["tokens_per_decode_forward"] == 2.5
    assert end["updates"] == 3 and end["update_seconds"] == .5
    assert end["feedback_blocks"] == 3
    assert end["fully_covered_rounds"] == 5 and end["coverage_skips"] == 2
    assert [row["adapter_version"] for row in actual] == [1, 3]
    assert actual[0]["cumulative"]["tokens"] == 10
    assert "cumulative" not in rows[0]


def test_trajectory_validates_each_request_and_setup():
    good = Generation([1], 1, 1, 1).summary()
    with pytest.raises(ValueError):
        stream_trajectory([])
    with pytest.raises(ValueError):
        stream_trajectory([good], setup_seconds=-1)
    with pytest.raises(ValueError):
        stream_trajectory([good], engine_setup_seconds=float("nan"))
    with pytest.raises(ValueError):
        stream_trajectory([good, {**good, "seconds": 0}])


def test_continuation_selection_is_explicit_and_copied():
    sequences = [torch.arange(2), torch.arange(6), torch.arange(7)]
    prompts = continuation_prompts(sequences, count=2, length=4)
    assert [p.tolist() for p in prompts] == [[[0, 1, 2, 3]], [[0, 1, 2, 3]]]
    prompts[0][0, 0] = 7
    assert sequences[1][0] == 0
    with pytest.raises(ValueError):
        continuation_prompts(sequences, count=3, length=4)


def test_comparison_does_not_hide_length_or_token_mismatches():
    assert compare_tokens([1, 2, 3], [1, 4, 3])["common_prefix"] == 1
    assert not compare_tokens([1, 2], [1, 2, 3])["identical"]
    assert compare_tokens([], [])["identical"]


@pytest.mark.parametrize("identical", [False, True])
def test_sampled_token_equality_is_reported_as_a_diagnostic(identical):
    rows = [{"identical": identical}]
    assert comparison_summary(rows, SamplingConfig()) == {
        "comparison_mode": "greedy_exact", "greedy_identical": identical}
    assert comparison_summary(rows, SamplingConfig(temperature=1)) == {
        "comparison_mode": "stochastic_diagnostic", "greedy_identical": None}


@pytest.mark.parametrize("sampler", ["linear", "tree"])
@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("sampling", [SamplingConfig(temperature=1), SamplingConfig(.7, 3, .8)])
def test_sampling_settings_reach_every_warmup_and_measured_arm(monkeypatch, sampler, online, sampling):
    import blockspec_ablation.benchmark as module

    torch.manual_seed(517)
    model = Decoder(ModelConfig(vocab_size=8, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=2, adapter_rank=2)).train_adapters_only()
    observed = []
    noise = UniformNoise(1, 6)

    def record(original, name):
        def generate(*args, **kwargs):
            assert kwargs["sampling"] == sampling
            if name != "generate_ar":
                assert kwargs["noise"] == noise
            seed = kwargs["generator"].initial_seed()
            output = original(*args, **kwargs)
            observed.append((name, seed, tuple(output.tokens), kwargs.get("learner") is not None))
            return output
        return generate

    for name in ("generate_ar", "generate_speculative", "generate_tree"):
        monkeypatch.setattr(module, name, record(getattr(module, name), name))
    config = BenchmarkConfig(tokens=8, block_size=3, warmup_tokens=4, sampler=sampler,
                             top_k=2, prefix_budget=5, sampling=sampling, noise=noise)
    prompts = [torch.tensor([[0, 1]]), torch.tensor([[0, 2]])]

    def run():
        if online:
            return benchmark_streams(model, prompts, config, OnlineConfig(stride=1, replay_blocks=1,
                                                                          train_last_layers=1))
        return benchmark_offline(model, prompts, config)

    result = run()
    first = observed.copy()
    observed.clear()
    repeated = run()
    assert observed == first  # Local generators include warmup; adapters are restored between runs.
    assert result["config"]["sampling"] == asdict(sampling)
    assert result["config"]["noise"] == asdict(noise)
    assert result["config"]["top_k"] == 2  # Tree width is separate from the target filter.
    assert result["comparison_mode"] == "stochastic_diagnostic"
    assert result["greedy_identical"] is None
    assert result["comparisons"] == repeated["comparisons"]
    assert {row[0] for row in observed} == {"generate_ar", "generate_tree" if sampler == "tree"
                                                                     else "generate_speculative"}
    assert all(arm["tokens"] == 32 for arm in result["arms"].values())
    if online:
        assert result["arms"]["online"]["updates"] > 0
        assert all(result["online_adapter_changed_per_stream"])
        for trace in result["trajectories"]:
            first, second = trace["arms"]["online"]
            assert second["adapter_version_start"] == first["adapter_version"] > 0
            assert second["adapter_version"] > first["adapter_version"]


@pytest.mark.parametrize("sampling", [None, {"temperature": 1}, 1])
def test_benchmark_requires_a_validated_sampling_configuration(sampling):
    with pytest.raises(ValueError, match="SamplingConfig"):
        BenchmarkConfig(sampling=sampling)


@pytest.mark.parametrize("noise", [None, {"low": 1}, 1])
def test_benchmark_requires_a_validated_noise_configuration(noise):
    with pytest.raises(ValueError, match="UniformNoise"):
        BenchmarkConfig(noise=noise)


@pytest.mark.parametrize("generate", [generate_speculative, generate_tree])
def test_online_feedback_preserves_actual_noise_and_greedy_target(monkeypatch, generate):
    torch.manual_seed(820)
    model = Decoder(ModelConfig(vocab_size=8, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=2, adapter_rank=2)).double()
    learner = OnlineLearner(model, OnlineConfig(stride=1, replay_blocks=1, train_last_layers=1))
    noise = UniformNoise(3, 4)
    observed = []
    original = learner.observe

    def observe(feedback, **kwargs):
        observed.append(feedback.inputs.clone())
        return original(feedback, **kwargs)

    monkeypatch.setattr(learner, "observe", observe)
    prompt = torch.tensor([[0, 1]])
    expected = generate_ar(model, prompt, 18)
    result = generate(model, prompt, 18, block_size=3, learner=learner, noise=noise,
                      generator=torch.Generator().manual_seed(821))
    assert result.tokens == expected.tokens
    assert observed and learner.updates > 0
    assert all(torch.all(inputs[:, 1:] == 3) for inputs in observed)


def test_cli_checks_source_checkpoint_before_execution_cast(tmp_path, monkeypatch, capsys):
    import copy
    import json
    import sys

    from blockspec_ablation import cli
    from blockspec_ablation.checkpoint import save_checkpoint

    model = Decoder(ModelConfig(vocab_size=8, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=1, adapter_rank=2)).train_adapters_only()
    checkpoint, data = tmp_path / "adapter.pt", tmp_path / "validation.jsonl"
    save_checkpoint(checkpoint, model, adapter_only=True)
    data.write_text(json.dumps({"input_ids": [0, 1, 2]}) + "\n")

    def load_base(*args, **kwargs):
        assert kwargs["dtype"] == torch.float32
        return copy.deepcopy(model)

    def run(converted, prompts, config, online_config, **kwargs):
        assert next(converted.parameters()).dtype == torch.bfloat16
        assert all(p.dtype == torch.float32 for p in converted.adapter_parameters())
        assert config.noise == UniformNoise(1)
        return {"checked": True}

    monkeypatch.setattr(cli, "load_hf_base", load_base)
    monkeypatch.setattr("blockspec_ablation.benchmark.benchmark_streams", run)
    monkeypatch.setattr(sys, "argv", ["blockspec", "benchmark", "--base", str(tmp_path),
                                     "--adapter", str(checkpoint), "--data", str(data), "--device", "cpu",
                                     "--prompts", "1", "--prompt-length", "2", "--dtype", "float32",
                                     "--execution-dtype", "bfloat16", "--noise-low", "1"])
    threads = torch.get_num_threads()
    try:
        cli.main()
    finally:
        torch.set_num_threads(threads)
    output = json.loads(capsys.readouterr().out)
    assert output["checked"] and output["dtype"] == "bfloat16"
    assert output["checkpoint_base_dtype"] == "float32"


@pytest.mark.parametrize("loss", ["l1", "forward_kl"])
@pytest.mark.parametrize("sampler", ["linear", "tree"])
@pytest.mark.parametrize("last_layers", [None, 1])
def test_balanced_streams_restore_weights_and_keep_online_across_requests(sampler, last_layers, loss):
    torch.manual_seed(17)
    model = Decoder(ModelConfig(vocab_size=8, hidden_size=16, intermediate_size=32,
                                num_hidden_layers=2, adapter_rank=2)).train_adapters_only()
    before, frozen = adapter_state(model), base_fingerprint(model)
    requires_grad = [p.requires_grad for p in model.parameters()]
    progress = []
    result = benchmark_streams(model, [torch.tensor([[0, 1]]), torch.tensor([[0, 2]])],
                               BenchmarkConfig(tokens=8, block_size=3, warmup_tokens=4,
                                               sampler=sampler, prefix_budget=5),
                               OnlineConfig(stride=1, replay_blocks=1, train_last_layers=last_layers,
                                            loss=loss),
                               progress=progress.append)
    assert result["greedy_identical"]
    assert result["arms"]["ar"]["tokens"] == 32
    assert result["arms"]["online"]["updates"] > 0
    assert all(result["online_adapter_changed_per_stream"])
    online_requests = [p for p in progress if p["arm"] == "online"]
    assert online_requests[1]["adapter_version"] > online_requests[0]["adapter_version"]
    assert [(p["repeat"], p["arm"]) for p in progress[::2]] == [
        (0, "ar"), (0, "static"), (0, "online"), (1, "online"), (1, "static"), (1, "ar")]
    assert all(torch.equal(v, adapter_state(model)[n]) for n, v in before.items())
    assert base_fingerprint(model) == frozen
    assert requires_grad == [p.requires_grad for p in model.parameters()]
    count = sum(p.numel() for p in model.adapter_parameters())
    assert result["online_trainable_parameters"] == count / (2 if last_layers else 1)
    for trace in result["trajectories"]:
        for arm, requests in trace["arms"].items():
            assert len(requests) == 2
            assert requests[-1]["cumulative"] == result["repeats"][trace["repeat"]][arm]
            assert requests[0]["adapter_version_start"] == 0
            assert requests[1]["adapter_version_start"] == requests[0]["adapter_version"]
            if arm == "online":
                assert requests[1]["adapter_version"] > requests[0]["adapter_version"]
                assert requests[1]["last_update_loss"] is not None


def test_benchmark_restores_adapter_on_exception():
    model = Decoder(ModelConfig(vocab_size=8, hidden_size=16, intermediate_size=32,
                                num_hidden_layers=1, adapter_rank=2))
    before = adapter_state(model)
    requires_grad = [p.requires_grad for p in model.parameters()]
    def stop(_):
        raise RuntimeError("test interruption")
    with pytest.raises(RuntimeError, match="test interruption"):
        benchmark_streams(model, [torch.tensor([[0, 1]])],
                          BenchmarkConfig(tokens=4, warmup_tokens=4), progress=stop)
    assert all(torch.equal(v, adapter_state(model)[n]) for n, v in before.items())
    assert requires_grad == [p.requires_grad for p in model.parameters()]


@pytest.mark.parametrize("key,value", [("repeats", 1), ("repeats", 3), ("tokens", 0),
                                      ("sampler", "invalid"), ("block_size", 1)])
def test_invalid_benchmark_configuration(key, value):
    with pytest.raises(ValueError):
        replace(BenchmarkConfig(), **{key: value})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph hardware required")
@pytest.mark.parametrize("temperature", [0, 1])
@pytest.mark.parametrize("loss", ["l1", "forward_kl"])
@pytest.mark.parametrize("sampler", ["linear", "tree"])
@pytest.mark.parametrize("last_layers", [None, 1])
def test_cuda_graph_three_arm_stream_with_live_adapter_updates(sampler, last_layers, loss, temperature):
    torch.manual_seed(28)
    model = Decoder(ModelConfig(vocab_size=8, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                head_dim=8, adapter_rank=2)).cuda().train_adapters_only()
    result = benchmark_streams(model, [torch.tensor([[0, 1, 2]]), torch.tensor([[0, 2, 1]])],
                               BenchmarkConfig(tokens=8, block_size=3, warmup_tokens=12,
                                               sampler=sampler, prefix_budget=5, execution="cuda_graph",
                                               sampling=SamplingConfig(temperature=temperature)),
                               OnlineConfig(stride=1, replay_blocks=2, train_last_layers=last_layers,
                                            loss=loss))
    assert result["online_optimizer"] == "fused"
    assert result["greedy_identical"] is (True if temperature == 0 else None)
    assert result["base_unchanged"] and result["adapter_restored"]
    assert all(result["online_adapter_changed_per_stream"])
    assert result["execution"]["capacity"] >= 3 + 12  # warmup may be longer than a measured request
    for arm in result["arms"].values():
        assert arm["engine_setup_seconds"] > 0
        assert arm["tps_including_all_setup"] < arm["tps_including_learner_setup"] <= arm["tps"]
    for arm in result["arms"]:
        assert sum(repeat[arm]["engine_setup_seconds"] for repeat in result["repeats"]) == (
            result["arms"][arm]["engine_setup_seconds"])
        for trace in result["trajectories"]:
            assert trace["arms"][arm][-1]["cumulative"] == result["repeats"][trace["repeat"]][arm]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph hardware required")
@pytest.mark.parametrize("feedback_execution", ["windowed", "all"])
@pytest.mark.parametrize("last_layers", [None, 1])
def test_windowed_benchmark_charges_both_needed_draft_signatures(monkeypatch, feedback_execution, last_layers):
    from blockspec_ablation.execution import FixedShapeExecutor
    original = FixedShapeExecutor.prepare

    def measured_signatures(self, signatures):
        original(self, signatures)
        self.signature_seconds = {key: 1.0 + (key[2] is not None) for key in self.slots}
        self.setup_seconds = sum(self.signature_seconds.values())

    monkeypatch.setattr(FixedShapeExecutor, "prepare", measured_signatures)
    torch.manual_seed(71)
    model = Decoder(ModelConfig(vocab_size=8, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=2, adapter_rank=2)).cuda()
    result = benchmark_streams(model, [torch.tensor([[0, 1, 2]])],
                               BenchmarkConfig(tokens=20, block_size=3, warmup_tokens=8,
                                               sampler="tree", prefix_budget=5, execution="cuda_graph"),
                               OnlineConfig(stride=4, replay_blocks=1, train_last_layers=last_layers,
                                            feedback_execution=feedback_execution))
    cost = 7 if last_layers is None else 11 if feedback_execution == "windowed" else 9
    assert result["execution"]["setup_seconds_by_arm"] == {"ar": 2, "static": 7, "online": cost}
    for trace in result["trajectories"]:
        assert trace["arms"]["online"][-1]["cumulative"]["feedback_blocks"] == (
            result["repeats"][trace["repeat"]]["online"]["feedback_blocks"])
