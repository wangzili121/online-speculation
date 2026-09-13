import pytest
import torch

from blockspec_ablation.diagnostics import audit_paired_teacher
from blockspec_ablation.model import Decoder, ModelConfig


@pytest.mark.parametrize("attention", ["default", "math", "fp32"])
def test_teacher_audit_restores_flags_hooks_and_weights(attention):
    model = Decoder(ModelConfig(vocab_size=7, hidden_size=16, intermediate_size=24,
                                 num_hidden_layers=1, num_attention_heads=2,
                                 num_key_value_heads=1, head_dim=8, adapter_rank=2))
    before = {name: p.detach().clone() for name, p in model.state_dict().items()}
    old = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    result = audit_paired_teacher(model, torch.tensor([[0, 1, 2, 3, 4]]), attention=attention,
                                  reduced_bf16=not old)
    assert result["logits"]["max_abs"] < 1e-6 and result["argmax_agreement"] == 1
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == old
    assert all(not m._forward_hooks for m in model.modules())
    assert all(torch.equal(before[n], p) for n, p in model.state_dict().items())
