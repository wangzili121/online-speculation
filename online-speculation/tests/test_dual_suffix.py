import copy

import pytest
import torch

from blockspec.losses import divergence
from blockspec.feedback import Feedback
from blockspec.parallel import DualViewConfig, DualViewDecoder, MaskedAttentionBranch, generate
from blockspec.parallel.feedback import OnlineFeedback
from blockspec.parallel.online import SuffixConfig, SuffixLearner
from blockspec.parallel.sampling import ProposalSampler
from blockspec.sampling import SamplingConfig


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))]


def model(device="cpu", dtype=torch.float32):
    torch.manual_seed(840)
    config = DualViewConfig(vocab_size=17, hidden_size=16, intermediate_size=24, num_hidden_layers=3,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    return DualViewDecoder(config).to(device=device, dtype=dtype).eval().requires_grad_(False)


@torch.no_grad()
def observation(net, start):
    device = net.embedding.weight.device
    prefix = torch.tensor([[4, 5, 7, 3]], device=device)
    cache = net(prefix).cache
    tokens = torch.tensor([[8, 1, 1, 1]], device=device)
    draft = net(tokens, view="draft", cache=cache, capture_layer=start)
    candidates = torch.cat((tokens[:, :1], draft.logits[:, :-1].argmax(-1)), 1)
    teacher = net(candidates, cache=cache).logits[0, :3]
    return Feedback(tokens, cache, teacher, 3, draft.boundary, True), draft


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("last", [1, 2])
def test_suffix_matches_full_logits_and_gradients(device, dtype, last):
    net = model(device, dtype)
    full = copy.deepcopy(net)
    learner = SuffixLearner(net, SuffixConfig(last_layers=last, stride=1))
    item, draft = observation(net, learner.capture_layer)
    with torch.no_grad():
        suffix = net.forward_suffix(item.boundary, cache=item.cache[learner.capture_layer:], logit_range=(0, 3))
        same_rows = net.head(draft.hidden[:, :3])
    assert suffix.cache[0][0].data_ptr() == item.cache[learner.capture_layer][0].data_ptr()
    torch.testing.assert_close(suffix.hidden, draft.hidden, rtol=0, atol=0)
    torch.testing.assert_close(suffix.logits, same_rows, rtol=0, atol=0)
    # Selecting fewer output-head rows can select a different CPU GEMM kernel.
    torch.testing.assert_close(suffix.logits, draft.logits[:, :3], rtol=2e-6, atol=1e-7)
    learner.observe(item, may_update=False)
    loss = learner.backward()
    for name, p in full.named_parameters():
        p.requires_grad_(name in learner.master)
    hidden = full(item.inputs, view="draft", cache=item.cache, compute_logits=False).hidden
    actual = full.head(hidden[:, :3])[0]
    reference_loss = divergence(actual, item.teacher_logits, "forward_kl").mean()
    reference_loss.backward()
    assert loss == pytest.approx(float(reference_loss.detach()), rel=1e-6)
    for name, p in full.named_parameters():
        if name in learner.master:
            torch.testing.assert_close(learner.master[name].grad, p.grad.float(), rtol=0, atol=0)
        else:
            assert p.grad is None
    assert all(p.grad is None and not p.requires_grad for p in net.parameters())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_updates_preserve_ar_and_restore_online_state(device, dtype):
    net = model(device, dtype)
    config = SuffixConfig(stride=2, learning_rate=1e-3)
    learner = SuffixLearner(net, config)
    prompt = torch.tensor([[3, 4, 5]], device=device)
    with torch.no_grad():
        before = net(prompt)
    frozen = {name: p.clone() for name, p in net.named_parameters() if name not in learner.master}
    initial = {name: p.clone() for name, p in learner.master.items()}
    result = generate(MaskedAttentionBranch(net), prompt, 24,
                      sampling=SamplingConfig(1.), generator=torch.Generator(device=device).manual_seed(842),
                      feedback=OnlineFeedback(learner=learner))
    assert result.updates > 0 and not learner.replay
    assert any(not torch.equal(initial[name], p) for name, p in learner.master.items())
    assert all(torch.equal(value, dict(net.named_parameters())[name]) for name, value in frozen.items())
    with torch.no_grad():
        after = net(prompt)
    torch.testing.assert_close(after.logits, before.logits, rtol=0, atol=0)
    for old, new in zip(before.cache, after.cache):
        for a, b in zip(old, new):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    state = learner.state_dict()
    restored_net = model(device, dtype)
    restored = SuffixLearner(restored_net, config)
    restored.load_state_dict(state)
    outputs = []
    for instance, owner in ((net, learner), (restored_net, restored)):
        outputs.append(generate(MaskedAttentionBranch(instance), prompt, 24,
                                sampling=SamplingConfig(1.), generator=torch.Generator(device=device).manual_seed(843),
                                feedback=OnlineFeedback(learner=owner)))
    assert outputs[0].tokens == outputs[1].tokens
    assert outputs[0].accepted_per_round == outputs[1].accepted_per_round
    for name, p in learner.master.items():
        assert torch.equal(p, restored.master[name])
        assert torch.equal(dict(net.named_parameters())[name], dict(restored_net.named_parameters())[name])
    assert learner.updates == restored.updates


def test_only_reached_rows_and_post_correction_versions_are_used():
    net = model()
    learner = SuffixLearner(net, SuffixConfig(stride=1, learning_rate=.001))
    seen, original = [], learner.observe

    def observe(item, **kwargs):
        assert item.valid == sampler.used
        assert torch.equal(item.teacher_logits, sampler.teacher[:item.valid])
        seen.append(item.valid)
        return original(item, **kwargs)

    learner.observe = observe

    class Sampler(ProposalSampler):
        def propose(self, logits, generator, **kwargs):
            output = super().propose(logits, generator, **kwargs)
            self.version = learner.version
            self.saved = output[1].clone()
            return output

        def verify(self, proposal, logits, generator):
            assert learner.version == self.version
            assert torch.equal(proposal.proposal, self.saved)
            result, target = super().verify(proposal, logits, generator)
            self.used = result.supervised
            self.teacher = logits.detach().clone()
            return result, target

    sampler = Sampler(SamplingConfig(1.))
    result = generate(MaskedAttentionBranch(net), torch.tensor([[3, 4]]), 20, sampler=sampler,
                      sampling=sampler.config, feedback=OnlineFeedback(learner=learner),
                      generator=torch.Generator().manual_seed(844))
    assert seen and all(1 <= n <= 3 for n in seen) and result.updates > 0
    assert not learner.replay


def test_suffix_boundary_guards():
    net = model()
    learner = SuffixLearner(net)
    item, _ = observation(net, learner.capture_layer)
    with pytest.raises(ValueError, match="historical cache"):
        net.forward_suffix(item.boundary)
    with pytest.raises(ValueError, match="shapes"):
        net.forward_suffix(item.boundary, cache=item.cache)
    with pytest.raises(ValueError, match="restricted"):
        net.forward_suffix(item.boundary, cache=item.cache[-1:], draft_weights={"norm.weight": net.norm.weight})
    with pytest.raises(ValueError, match="draft capture"):
        net(torch.tensor([[3]]), capture_layer=2)
    with torch.no_grad():
        net.layers[0].attention.draft.q.weight.add_(.1)
    with pytest.raises(RuntimeError, match="frozen representations"):
        learner.observe(item)


def test_suffix_config_and_state_guards():
    with pytest.raises(ValueError):
        SuffixConfig(stride=1, replay_blocks=2)
    net = model()
    with pytest.raises(ValueError, match="frozen"):
        SuffixLearner(net, SuffixConfig(last_layers=4))
    learner = SuffixLearner(net)
    state = learner.state_dict()
    state["master"][next(iter(state["master"]))].fill_(float("nan"))
    with pytest.raises(ValueError, match="finite FP32"):
        learner.load_state_dict(state)
    other = model()
    with torch.no_grad():
        other.norm.weight.add_(.1)
    with pytest.raises(ValueError, match="model, source"):
        SuffixLearner(other).load_state_dict(learner.state_dict())


def test_online_state_resumes_across_storage_locations():
    net = model()
    net.source = {"weight_sha256": "test-content", "directory": "old-location"}
    learner = SuffixLearner(net)
    state = learner.state_dict()
    assert "directory" not in state["source"]
    state["source"]["directory"] = "legacy-location"
    net.source["directory"] = "new-location"
    learner.load_state_dict(state)
    state["source"]["weight_sha256"] = "different-content"
    with pytest.raises(ValueError, match="model, source"):
        learner.load_state_dict(state)


def test_suffix_gradient_finite_difference_and_fixed_feedback_learning():
    net = model()
    learner = SuffixLearner(net, SuffixConfig(stride=1, learning_rate=.003))
    item, _ = observation(net, learner.capture_layer)
    item.teacher_logits = torch.randn(3, 17, generator=torch.Generator().manual_seed(846))
    learner.observe(item, may_update=False)
    initial_loss = learner.backward()
    name = "layers.2.attention.draft.o.weight"
    master = learner.master[name]
    index = master.grad.abs().argmax()
    exact = float(master.grad.flatten()[index])
    saved = master.detach().clone()
    values = []
    epsilon = .002
    for sign in (-1, 1):
        with torch.no_grad():
            master.copy_(saved)
            master.flatten()[index] += sign * epsilon
        values.append(learner.backward())
    with torch.no_grad():
        master.copy_(saved)
    numerical = (values[1] - values[0]) / (2 * epsilon)
    assert numerical == pytest.approx(exact, rel=.003, abs=3e-5)
    for _ in range(20):
        learner.update()
    assert learner.backward() < initial_loss
