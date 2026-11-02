"""单层 demo 与 FP32 Attention 的对照；投影仍调用同一套 NPU 参数。"""

import os

import pytest
import torch

if os.environ.get("ORTHRUS_RUN_NPU_TESTS") != "1":
    pytest.skip("requires an allocated Ascend device", allow_module_level=True)

from blockspec.vllm_ascend import torch_npu_dsv4_layer as layer_impl
from blockspec.vllm_ascend.dsv4_hybrid import merge_attention_segments


@pytest.mark.parametrize("kind", ["swa", "hca", "csa"])
@pytest.mark.parametrize("block", [2, 32])
def test_layer_matches_dense_attention(kind, block, monkeypatch):
    torch.npu.set_device(0)
    torch.manual_seed(20261103)
    config = layer_impl.NativeDSV4Config.dsv4_flash()
    model = layer_impl.TorchNPUOrthrusDSV4Attention(
        config, device=torch.device("npu:0"), dtype=torch.bfloat16
    )
    history = layer_impl.make_synthetic_history(
        config, batch=1, device=torch.device("npu:0"), dtype=torch.bfloat16
    )
    snapshots = {name: value.clone() for name, value in vars(history).items()}
    hidden = torch.randn(1, block, config.hidden_size, device="npu", dtype=torch.bfloat16) * .02
    positions = torch.arange(1024, 1024 + block, device="npu").view(1, block)

    def dense(query, kv, query_lengths, key_lengths):
        del query_lengths
        scores = torch.einsum("bqhd,bknd->bqhk", query.float(), kv.float()) * config.head_dim**-.5
        lengths = torch.tensor(key_lengths, device=query.device)
        valid = torch.arange(kv.shape[1], device=query.device)[None] < lengths[:, None]
        scores.masked_fill_(~valid[:, None, None], -torch.inf)
        lse = scores.logsumexp(-1, keepdim=True)
        output = torch.einsum("bqhk,bknd->bqhd", scores.softmax(-1), kv.float())
        return output.to(query.dtype), lse

    with torch.inference_mode():
        actual = model(hidden, positions, history, kind)
        monkeypatch.setattr(layer_impl, "_native_fia_segment", dense)
        monkeypatch.setattr(layer_impl, "merge_attention_outputs",
                            lambda outputs, lses: merge_attention_segments(outputs, lses)[0])
        expected = model(hidden, positions, history, kind)
    # O projection 会放大局部 BF16 舍入；同时约束整体 RMS 和绝对误差。
    delta = (actual.output.float() - expected.output.float()).square().mean().sqrt()
    relative = delta / expected.output.float().square().mean().sqrt().clamp_min(1e-12)
    assert relative.item() < .02
    torch.testing.assert_close(actual.output, expected.output, atol=.02, rtol=.05)
    if kind == "csa":
        torch.testing.assert_close(actual.topk_indices, expected.topk_indices, atol=0, rtol=0)
    for name, before in snapshots.items():
        assert torch.equal(before, getattr(history, name)), name
