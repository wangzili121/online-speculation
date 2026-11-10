"""检查全模型参数分片边界及 dtype；共享是参数视图，不共享 Attention 输出。"""

from types import SimpleNamespace

import pytest
import torch

from blockspec.pypto_qwen38 import ar_prefix
from blockspec.pypto_qwen38.head_loading import bank_specs


@pytest.mark.parametrize("stop", (60, 63))
def test_every_borrowed_bank_stays_in_its_native_layer_and_preserves_fp32_decay(tmp_path, monkeypatch, stop):
    plans = ar_prefix.base_plans()
    banks = [torch.empty(p.shape, dtype=p.dtype, device="meta") for p in plans]
    copied, views = [], []

    def view(bank, shape, dtype, offset):
        index = next(i for i, value in enumerate(banks) if value is bank)
        size = shape[0] * shape[1] * dtype.itemsize
        assert 0 <= offset <= bank.numel() * bank.element_size() - size
        assert offset % size == 0
        plan = plans[index]
        layer = plan.layers[offset // size]
        views.append((layer, plan.tail, tuple(shape)))
        return torch.empty(shape, dtype=dtype, device="meta")

    def alloc(shape, dtype, *, init):
        assert shape == (1, 64) and init.dtype == dtype == torch.float32
        value = torch.empty(shape, dtype=dtype, device="meta")
        copied.append(value)
        return value

    monkeypatch.setattr(ar_prefix, "_view", view)
    worker = SimpleNamespace(alloc_tensor=alloc, free_tensor=lambda value: None)
    native = SimpleNamespace(model_root=tmp_path, banks=banks, worker=worker)
    head = SimpleNamespace(directory=None, layout=SimpleNamespace(size=1), model_root=tmp_path,
        banks=lambda layer: [torch.empty(shape, dtype=dtype, device="meta") for _,shape,dtype,_ in bank_specs(layer)])
    layers, owned = ar_prefix.borrow_prefix(native, head, stop)
    assert list(layers) == list(range(stop))
    assert len(owned) == len(copied) == sum(layer % 4 != 3 for layer in range(stop))
    for layer, values in layers.items():
        assert [(tuple(value.shape), value.dtype) for value in values] == [
            (shape, dtype) for _,shape,dtype,_ in bank_specs(layer)]
        actual = [tail for index, tail, shape in views if index == layer]
        expected = [("self_attn." if layer % 4 == 3 else "linear_attn.") + name
                    for name,shape,dtype,_ in bank_specs(layer) if name != "A_log"]
        assert actual == expected
    head.directory = tmp_path / "trained-head"
    with pytest.raises(ValueError, match="AR-copy"):
        ar_prefix.borrow_prefix(native, head, stop)
    head.directory = None
    native.banks = banks[:-1]
    with pytest.raises(ValueError, match="ABI changed"):
        ar_prefix.borrow_prefix(native, head, stop)
