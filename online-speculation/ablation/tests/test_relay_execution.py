import pytest
import torch

from blockspec_ablation.decoding import generate_ar
from blockspec_ablation.execution import FixedShapeExecutor
from blockspec_ablation.model import Decoder, ModelConfig
from blockspec_ablation.relay import RelayConfig, RelayHead, RelayLearner, generate_relay
from blockspec_ablation.relay_execution import RelayExecutor, _RelaySlot, _transform, race_sample
from blockspec.sampling import SamplingConfig, probabilities


@pytest.mark.parametrize("sampling", [SamplingConfig(), SamplingConfig(1.), SamplingConfig(.7, 3, .8)])
@pytest.mark.parametrize("threshold", [0., .3, 1.])
def test_graph_computation_matches_explicit_causal_reference(sampling, threshold):
    torch.manual_seed(19)
    head = RelayHead(RelayConfig(7, 16, 3))
    with torch.no_grad():
        head.projection.weight.normal_(0, .5)
    slot = _RelaySlot(head, 4, sampling, threshold, False)
    slot.logits.normal_()
    slot.hidden.normal_()
    slot.exponential.exponential_()
    with torch.no_grad():
        slot.run()
        q0 = probabilities(slot.logits[0], sampling)
        tokens = [q0.argmax(-1) if sampling.temperature == 0 else race_sample(q0, slot.exponential[0])]
        rows, score = [], 1.
        for i in range(1, 4):
            if threshold:
                score *= float(head.confidence_logits(slot.hidden[i], tokens[-1]).sigmoid())
                if score < threshold:
                    break
            q = probabilities(head(slot.logits[i], tokens[-1]), sampling)
            rows.append(q)
            tokens.append(q.argmax(-1) if sampling.temperature == 0 else race_sample(q, slot.exponential[i]))
        count = int(slot.count)
        assert count == len(rows)
        torch.testing.assert_close(slot.tokens[:count + 1], torch.stack(tokens))
        if rows:
            torch.testing.assert_close(slot.q[:count], torch.stack(rows), rtol=0, atol=0)


@pytest.mark.parametrize("sampling", [SamplingConfig(), SamplingConfig(.7), SamplingConfig(1., 4), SamplingConfig(.5, 3, .8)])
def test_captured_probability_transform_equals_reference(sampling):
    logits = torch.randn(17)
    torch.testing.assert_close(_transform(logits, sampling), probabilities(logits, sampling), rtol=0, atol=0)


def test_exponential_race_recovers_law_with_zero_support():
    q = torch.tensor([.1, .2, 0., .7])
    exponential = torch.empty(80000, 4).exponential_(generator=torch.Generator().manual_seed(89))
    counts = race_sample(q, exponential).bincount(minlength=4).float() / 80000
    torch.testing.assert_close(counts, q, rtol=0, atol=.005)


def test_graph_executor_state_ownership_and_contracts():
    head = RelayHead(RelayConfig(7, 16, 3))
    engine = RelayExecutor(head, block_size=4, use_cuda_graph=False)
    logits, hidden = torch.randn(4, 7), torch.randn(4, 16)
    with pytest.raises(RuntimeError, match="inference-only"):
        engine(logits, hidden)
    with torch.no_grad():
        result = engine(logits, hidden)
        saved = result.q.clone()
        head.projection.weight.normal_(0, 3)
        engine.validate(head, SamplingConfig(), 0.)
        engine(logits, hidden)
        torch.testing.assert_close(result.q, saved, rtol=0, atol=0)
    with pytest.raises(ValueError, match="policy"):
        engine.validate(head, SamplingConfig(1.), 0.)
    head.embedding.weight.data = head.embedding.weight.data.clone()
    with pytest.raises(RuntimeError, match="storage"):
        engine.validate(head, SamplingConfig(), 0.)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA integration")
@pytest.mark.parametrize("threshold", [0., .3, 1.])
def test_cuda_graph_online_causal_kv_and_rng(threshold):
    torch.manual_seed(17)
    model = Decoder(ModelConfig(vocab_size=7, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                head_dim=8, adapter_rank=2)).cuda().eval().requires_grad_(False)
    head = RelayHead(RelayConfig(7, 16, 3)).cuda()
    backbone = FixedShapeExecutor(model, capacity=64, max_query=4)
    backbone.prepare([(n, False, None) for n in range(1, 5)] + [(n, True, 2) for n in range(2, 5)])
    engine = RelayExecutor(head, block_size=4, threshold=threshold)
    prompt = torch.tensor([[1, 3, 4]], device="cuda")
    learner = RelayLearner(head, interval=1)
    result = generate_relay(model, head, prompt, 32, block_size=4, threshold=threshold, executor=backbone,
                            proposal_executor=engine, learner=learner)
    assert result.tokens == generate_ar(model, prompt, 32, executor=backbone).tokens
    stochastic = RelayExecutor(head, block_size=4, threshold=threshold, sampling=SamplingConfig(.8, 4, .9))
    logits, hidden = torch.randn(4, 7, device="cuda"), torch.randn(4, 16, device="cuda")
    with torch.no_grad():
        a = stochastic(logits, hidden, generator=torch.Generator(device="cuda").manual_seed(82))
        b = stochastic(logits, hidden, generator=torch.Generator(device="cuda").manual_seed(82))
    torch.testing.assert_close(a.tokens, b.tokens)
    torch.testing.assert_close(a.q, b.q, rtol=0, atol=0)
