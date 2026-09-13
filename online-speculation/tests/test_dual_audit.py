
import pytest
import torch

from blockspec.parallel import DualViewConfig, DualViewDecoder, MaskedAttentionBranch, generate
from blockspec.parallel.audit import AuditSampler, METRICS, SuffixAudit, audit_summary, paired_audit_intervals
from blockspec.parallel.feedback import OnlineFeedback
from blockspec.sampling import SamplingConfig


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("temperature", [0., 1.])
@torch.no_grad()
def test_common_prefix_audit_preserves_generation_and_equal_checkpoint(device, temperature):
    torch.manual_seed(860)
    net = DualViewDecoder(DualViewConfig(vocab_size=17, hidden_size=16, intermediate_size=24,
                                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                        head_dim=8)).to(device).eval().requires_grad_(False)
    alternative = {name: p for name, p in net.named_parameters() if name.startswith("layers.1.attention.draft.")}
    sampling = SamplingConfig(temperature, 5, .8)
    saved = {name: p.clone() for name, p in net.state_dict().items()}
    sampler = AuditSampler(sampling)
    inspector = SuffixAudit(net, 1, alternative, 4, sampling, recorded_logits=lambda: sampler.last_logits)
    prompt = torch.tensor([[3, 4, 5]], device=device)
    kwargs = {"sampling": sampling, "block_size": 4}
    expected = generate(MaskedAttentionBranch(net), prompt, 24,
                        generator=torch.Generator(device=device).manual_seed(861), **kwargs)
    actual = generate(MaskedAttentionBranch(net), prompt, 24, feedback=OnlineFeedback(learner=inspector),
                      generator=torch.Generator(device=device).manual_seed(861), sampler=sampler, **kwargs)
    assert actual.tokens == expected.tokens
    assert actual.accepted_per_round == expected.accepted_per_round
    assert actual.updates == 0 and inspector.feedback_blocks > 0
    assert inspector.max_replay_logit_error == 0.
    assert all(torch.equal(value, net.state_dict()[name]) for name, value in saved.items())
    summary = audit_summary(inspector.totals)
    assert summary["positions"] == sum(row["positions"] for row in summary["by_depth"])
    for pair in ((0, 1), (2, 3), (4, 5)):
        assert summary["mean"][METRICS[pair[0]]] == summary["mean"][METRICS[pair[1]]]
    assert 0 <= summary["mean"]["original_sampling_tv"] <= 1
    intervals = paired_audit_intervals([summary, summary], repeats=10)
    assert all(row["difference"] == 0. and row["paired_request_ci95"] == [0., 0.] for row in intervals.values())


def test_prompt_windows_are_disjoint_and_bounded(tmp_path):
    from blockspec.commands import evaluate as module
    data = tmp_path / "questions.jsonl"
    data.write_text('\n'.join('{"question": "question %d"}' % i for i in range(12)))
    assert module.prompt_texts(data, 4, offset=8) == ["question 8", "question 9", "question 10", "question 11"]
    assert not set(module.prompt_texts(data, 8)) & set(module.prompt_texts(data, 4, offset=8))
    with pytest.raises(ValueError):
        module.prompt_texts(data, 5, offset=8)
    with pytest.raises(ValueError):
        module.prompt_texts(data, 1, offset=-1)
    with pytest.raises(ValueError):
        module.prompt_texts(data, 0)


@pytest.mark.parametrize("empty_system,thinking", [(False, False), (True, False), (True, True)])
def test_public_prompt_template_keeps_roles_and_thinking_explicit(empty_system, thinking):
    from blockspec.commands import evaluate as module

    class Tokenizer:
        def apply_chat_template(self, messages, **options):
            expected = ([{"role": "system", "content": ""}] if empty_system else [])
            assert messages == expected + [{"role": "user", "content": "question"}]
            assert options == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": thinking}
            return "rendered"

    assert module.render_prompt(Tokenizer(), "question", empty_system=empty_system, thinking=thinking) == "rendered"


def test_empty_audit_is_explicit_and_bootstrap_keeps_the_callers_random_state():
    empty = audit_summary(torch.zeros(3, len(METRICS) + 1))
    assert empty["positions"] == 0 and all(value is None for value in empty["mean"].values())
    assert all(row["original_raw_tv"] is None for row in empty["by_depth"])
    with pytest.raises(ValueError):
        paired_audit_intervals([empty])
    values = torch.ones(3, len(METRICS) + 1)
    record = audit_summary(values)
    state = torch.get_rng_state()
    paired_audit_intervals([record], repeats=10)
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("temperature", [0., 1.])
def test_operator_profile_preserves_generation_and_removes_hooks(temperature):
    from blockspec.commands import evaluate as module
    model = DualViewDecoder(DualViewConfig(vocab_size=17, hidden_size=16, intermediate_size=24,
                                          num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                          head_dim=8)).eval().requires_grad_(False)
    result = module.profile_generation(model, torch.tensor([[3, 4, 5]]), tokens=8, block_size=4,
                                       sampling=SamplingConfig(temperature, 5, .8), seed=92)
    assert result["pass"] and len(result["profiles"]) == 2
    assert all(row["tokens"] == 8 for row in result["profiles"])
    assert any(event["name"] == "blockspec.draft" for event in result["profiles"][1]["regions"])
    for row in result["profiles"]:
        assert len(row["regions"]) == len({event["name"] for event in row["regions"]})
        assert all(not event["name"].startswith("blockspec.") for event in row["operators"])
    assert not model._forward_pre_hooks and not model._forward_hooks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="paired CUDA sampling execution")
def test_benchmark_pairs_sampling_executors_and_their_ar_controls(monkeypatch):
    from types import SimpleNamespace

    from blockspec.commands import evaluate as module
    model = DualViewDecoder(DualViewConfig(vocab_size=17, hidden_size=16, intermediate_size=24,
                                          num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                          head_dim=8)).cuda().eval().requires_grad_(False)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    prompts = [torch.tensor([[3, 4, 5]], device="cuda"), torch.tensor([[6, 7]], device="cuda")]
    monkeypatch.setattr(module, "prompt_ids", lambda *a, **kw: prompts)
    monkeypatch.setattr(module, "prompt_texts", lambda *a, **kw: ["first", "second"])
    args = SimpleNamespace(model=None, requests=2, prompts=None, thinking=False, empty_system=True,
                           temperature=1., top_k=5, top_p=.8, blocks=[4], repeats=2, tokens=12, seed=73,
                           sampling_executions=["scalar", "tensor", "graph"])
    result = module.benchmark(args, model, None)
    assert result["sampling_execution"] == {f"own_{mode}": mode for mode in args.sampling_executions}
    assert set(result["aggregate"]) == {f"own_{mode}_k{block}" for mode in args.sampling_executions for block in (1, 4)}
    assert len(result["records"]) == 24
    assert all(row["tokens"] == 12 for row in result["records"])
    assert result["sampling_setup_seconds"]["own_scalar_k4"] == 0
    assert len(result["execution_equivalence"]) == 4
    for row in result["execution_equivalence"].values():
        assert row["requests"] == 4
        assert all(row[field + "_identical"] for field in ("token_ids", "accepted_per_round", "decode_forwards"))
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())

    monkeypatch.setattr(module, "reference_generate", lambda *a: {
        "tokens": 12, "seconds": 1., "token_ids": [3] * 12,
        "prefill_forwards": 1, "prefill_output_tokens": 1, "decode_forwards": 11})
    external = module.benchmark(args, model, object())
    assert external["execution_equivalence"]["reference_vs_own_scalar_k4"]["accepted_per_round_identical"] is None
