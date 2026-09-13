import pytest
import torch

from blockspec_ablation.model import Decoder, ModelConfig
from blockspec_ablation.decoding import generate_speculative as original_generate
from blockspec.sampling import SamplingConfig
from blockspec_ablation.parallel import CausalLowRankBranch, generate, generate_ar


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_low_rank_branch_uses_existing_decoder_contract(temperature):
    torch.manual_seed(79)
    model = Decoder(ModelConfig()).eval()
    prompt = torch.tensor([[3, 7, 5, 8]])
    config = SamplingConfig(temperature=temperature)
    shared = generate(CausalLowRankBranch(model, initial_ar_token=False), prompt, 19, block_size=4, sampling=config,
                      generator=torch.Generator().manual_seed(14), audit_cache=True)
    original = original_generate(model, prompt, 19, block_size=4, sampling=config,
                                 generator=torch.Generator().manual_seed(14))
    assert shared.tokens == original.tokens
    assert shared.accepted_per_round == original.accepted_per_round


@pytest.mark.parametrize("initial_ar_token", [False, True])
@pytest.mark.parametrize("budget", [1, 2, 7, 19])
def test_causal_branch_bootstrap_conventions(initial_ar_token, budget):
    torch.manual_seed(736)
    model = Decoder(ModelConfig()).eval()
    prompt = torch.tensor([[3, 5, 7]])
    branch = CausalLowRankBranch(model, initial_ar_token=initial_ar_token)
    expected = generate_ar(branch, prompt, budget)
    output = generate(branch, prompt, budget, block_size=4, audit_cache=True)
    assert output.tokens == expected.tokens
    assert output.prefill_output_tokens == int(initial_ar_token)
