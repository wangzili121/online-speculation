"""参数视图必须从分配基址加偏移读回，不能当成独立设备指针。"""

import ctypes
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("dtype",(torch.float32,torch.bfloat16))
def test_readback_preserves_owner_and_nonzero_byte_offset(monkeypatch,dtype):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/"scripts"))
    module=importlib.import_module("diagnose_qwen38_pypto_training")
    backing=torch.arange(24).to(dtype)
    owner=SimpleNamespace(data_ptr=backing.data_ptr())
    calls=[]
    class Worker:
        def copy_from(self,dst,src,size,*,src_offset):
            assert src==owner.data_ptr
            calls.append((size,src_offset));ctypes.memmove(dst,src+src_offset,size)
    view=module.ReadableSpan(owner,(2,3),dtype,5*backing.element_size())
    actual=module.read_tensor(Worker(),view)
    assert calls==[(6*backing.element_size(),5*backing.element_size())]
    torch.testing.assert_close(actual,backing[5:11].reshape(2,3),atol=0,rtol=0)
