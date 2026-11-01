"""The readable demo must preserve the production primitive contracts."""

import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch

from blockspec.vllm_ascend.glm53_ops import finite_error


spec = importlib.util.spec_from_file_location(
    "glm53_orthrus_demo", Path(__file__).resolve().parents[1] / "benchmarks" / "glm53_orthrus_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


@pytest.fixture(scope="module")
def case():
    return demo.make_case(block_size=4, prefix_length=19)


@pytest.mark.parametrize("bidirectional", [False, True])
def test_complete_branches_are_finite_and_preserve_all_history(case, bidirectional):
    model, inputs = case
    before = [tensor.clone() for tensor in inputs]
    with torch.inference_mode():
        outputs = model(*inputs, bidirectional=bidirectional)
    assert set(outputs) == {"mla", "kda"}
    for value in outputs.values():
        assert value.shape == (4, 4096)
        assert torch.isfinite(value).all()
    for original, snapshot in zip(inputs, before, strict=True):
        torch.testing.assert_close(original, snapshot, rtol=0, atol=0)


def test_reverse_block_and_mla_allow_later_mask_to_affect_first_row(case):
    model, inputs = case
    changed = [tensor.clone() for tensor in inputs]
    changed[0][-1].add_(.5)
    with torch.inference_mode():
        old = model(*inputs)
        new = model(*changed)
        old_bidir = model(*inputs, bidirectional=True)
        new_bidir = model(*changed, bidirectional=True)
    torch.testing.assert_close(new["kda"][0], old["kda"][0], rtol=0, atol=0)
    assert not torch.equal(new["mla"][0], old["mla"][0])
    assert not torch.equal(new_bidir["kda"][0], old_bidir["kda"][0])


def test_zero_reverse_gate_recovers_causal_policy(case):
    model, inputs = case
    gate = model.backward_gate.clone()
    try:
        with torch.inference_mode():
            model.backward_gate.zero_()
            causal = model(*inputs)["kda"]
            bidir = model(*inputs, bidirectional=True)["kda"]
        torch.testing.assert_close(causal, bidir, rtol=0, atol=0)
    finally:
        model.backward_gate.copy_(gate)


@pytest.mark.parametrize("prefix", [0, 3, 2055])
def test_pool_tail_and_empty_history_use_the_same_demo(case, prefix):
    model, inputs = case
    hidden, _, _, tail, state = inputs
    history = hidden.new_zeros(prefix, 512)
    pools = hidden.new_zeros(prefix // 4, 128)
    with torch.inference_mode():
        output = model(hidden, history, pools, tail, state)
    assert all(torch.isfinite(value).all() for value in output.values())


@pytest.mark.parametrize("width,prefix", [(0, 3), (3, -1)])
def test_invalid_demo_boundaries_are_rejected(width, prefix):
    with pytest.raises(ValueError):
        demo.make_case(width, prefix)


@pytest.fixture(scope="module")
def npu_case():
    import torch_npu  # noqa: F401
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(0)
    if not enable_custom_op():
        pytest.fail("native Ascend operators unavailable")
    model, inputs = demo.make_case(8, 2055, device="npu", dtype=torch.bfloat16)
    reference = copy.deepcopy(model).cpu().float()
    return model, inputs, reference


@pytest.mark.skipif(os.environ.get("GLM53_DEMO_NPU") != "1", reason="explicit single-NPU validation")
@pytest.mark.parametrize("width,prefix", [(2, 0), (8, 3), (8, 2055)])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_npu_complete_branches_match_fp32_reference(npu_case, width, prefix, bidirectional):
    model, (hidden, history, pools, tail, state), reference = npu_case
    inputs = (hidden[:width], history[:prefix], pools[:prefix // 4], tail, state)
    before = [value.cpu().clone() for value in inputs]
    with torch.inference_mode():
        actual = model(*inputs, backend="npu", bidirectional=bidirectional)
        expected = reference(*(value.float() for value in before), bidirectional=bidirectional)
    errors = {name: finite_error(value.cpu(), expected[name]) for name, value in actual.items()}
    assert all(error["relative_rms"] < .04 for error in errors.values()), errors
    for original, snapshot in zip(inputs, before, strict=True):
        torch.testing.assert_close(original.cpu(), snapshot, rtol=0, atol=0)
    print(json.dumps(dict(width=width, prefix=prefix, bidirectional=bidirectional, errors=errors)))
