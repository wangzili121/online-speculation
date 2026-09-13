"""Paired runner contracts: matching templates, request order and learning state."""

import numpy as np
import torch

from blockspec import DualViewConfig, DualViewDecoder
from blockspec.commands import continue_training as command
from blockspec.parallel.feedback import OnlineFeedback
from blockspec.parallel.online import SuffixLearner


def test_paired_runner_uses_shared_configuration_and_explicit_factory_context(tmp_path, monkeypatch, capsys):
    # Load optimizer machinery before redirecting this runner's CUDA generator.
    torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
    torch.manual_seed(931)
    net = DualViewDecoder(DualViewConfig(
        vocab_size=17, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8)).eval().requires_grad_(False)
    evaluation, learning = tmp_path / "evaluation.jsonl", tmp_path / "learning.jsonl"
    for path, label in ((evaluation, "eval"), (learning, "learn")):
        path.write_text("\n".join('{"question": "' + label + str(i) + '"}' for i in range(3)))
    args = command.argument_parser().parse_args([
        "--model", str(tmp_path), "--prompts", str(evaluation), "--learning-prompts", str(learning),
        "--output", str(tmp_path / "report.json"), "--requests", "3", "--learn-requests", "3",
        "--tokens", "16", "--learn-tokens", "12", "--repeats", "2", "--block-size", "4",
        "--stride", "1", "--learning-rate", ".001", "--top-k", "0", "--top-p", "1",
        "--empty-system", "--shuffle-requests"])
    defaults = command.argument_parser().parse_args([
        "--model", str(tmp_path), "--prompts", str(evaluation), "--learning-prompts", str(learning),
        "--output", str(tmp_path / "defaults.json")])
    assert (defaults.loss, defaults.temperature, defaults.top_k, defaults.top_p) == ("tv", 1., 0, 1.)
    monkeypatch.setattr(command, "load_public", lambda *a, **kw: net)
    templates = []

    def prompts(folder, count, **options):
        templates.append(options)
        return [torch.tensor([[3, 5, 8 + i]]) for i in range(count)]

    monkeypatch.setattr(command, "prompt_ids", prompts)
    generator_type = torch.Generator
    monkeypatch.setattr(torch, "Generator", lambda **kw: generator_type(device="cpu"))
    executor_type = command.SamplingExecutor
    monkeypatch.setattr(command, "SamplingExecutor", lambda *a, **kw: executor_type(*a, **kw, device="cpu"))
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **kw: None)
    for name in ("memory_allocated", "max_memory_allocated"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **kw: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test device")
    learners, budgets = [], []

    def learner_factory(model, config):
        learner = SuffixLearner(model, config)
        learners.append(learner)
        return learner

    def feedback_factory(*, learner, output_budget):
        budgets.append(output_budget)
        return OnlineFeedback(learner=learner)

    result = command.run_experiment(args, learner_factory=learner_factory, feedback_factory=feedback_factory)
    capsys.readouterr()
    assert result["pass"] and len(learners) == 2
    assert len(result["records"]) == 3 * 2 * 5
    assert {12, 16} == set(budgets)
    assert len(templates) == 2 and all(item["empty_system"] for item in templates)
    for repeat in range(2):
        rows = [row for row in result["records"] if row["repeat"] == repeat]
        observed_order = [row["request"] for row in rows[::5]]
        assert observed_order == np.random.default_rng(args.seed + repeat).permutation(3).tolist()
        stream = result["streams"][repeat]
        assert stream["online_vs_static"] == stream["tps"]["online"] / stream["tps"]["static"]
        assert stream["continued_vs_learned"] == stream["tps"]["continued"] / stream["tps"]["learned"]
        assert stream["learners"]["online"]["updates"] > 0
    assert result["original_restored"] and result["ar_logits_and_kv_unchanged"]
    assert (tmp_path / "report.json").exists()
