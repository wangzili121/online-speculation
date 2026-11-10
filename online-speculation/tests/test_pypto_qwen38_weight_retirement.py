"""只释放已被完整替代且没有别名引用的初始 head 存储。"""

from types import SimpleNamespace as NS

import pytest

from blockspec.pypto_qwen38.weights import DraftWeights


def weights(mode="stream"):
    result = DraftWeights(NS(), transfer="packed", group_size=4)
    owner = NS(shape=(512,), dtype=NS(itemsize=1), buffer=NS(identity="original"))
    freed = []
    result.worker = NS(free_tensor=lambda value, **kwargs: freed.append(value))
    result.storage, result.handles, result.version = mode, [owner], 7
    result.buffers["group"] = owner
    result.packed[0] = object()
    result.resident[0] = (owner,)
    result.overrides = {layer:(NS(buffer=NS(identity=f"replacement{layer}")),)
                        for layer in range(64)}
    return result, owner, freed


@pytest.mark.parametrize("mode", ("stream", "resident"))
def test_fully_replaced_storage_is_freed_once_and_overrides_keep_working(mode):
    value, owner, freed = weights(mode)
    expected = tuple(value.overrides[layer] for layer in range(4))
    assert value.release_replaced_weights() == 512 and value.released_bytes == 512
    assert value.for_group(0) == expected and value.version == 7
    assert not value.handles and not value.buffers and not value.packed and not value.resident
    assert value.release_replaced_weights() == 0
    value.close()
    assert freed == [owner]


@pytest.mark.parametrize("problem", ("missing", "alias"))
def test_storage_is_retained_if_a_layer_is_missing_or_still_uses_it(problem):
    value, owner, freed = weights()
    if problem == "missing": value.overrides.pop(0)
    else: value.overrides[0] = (NS(buffer=NS(identity="original")),)
    with pytest.raises((ValueError, RuntimeError)): value.release_replaced_weights()
    assert not freed and value.handles == [owner] and value.version == 7


def test_failed_free_keeps_ownership_for_retry():
    value, owner, freed = weights()
    def fail(*args, **kwargs): raise RuntimeError("free failed")
    value.worker.free_tensor = fail
    with pytest.raises(RuntimeError, match="free failed"): value.release_replaced_weights()
    assert value.handles == [owner] and value.released_bytes == 0
    value.worker.free_tensor = lambda item, **kwargs: freed.append(item)
    assert value.release_replaced_weights() == 512
    value.close()
    assert freed == [owner]


def test_real_address_free_tensor_views_use_buffer_identity():
    import torch
    runtime = pytest.importorskip("pypto.runtime.task_interface")
    from pypto.runtime.device_tensor import DeviceTensor
    from simpler.buffer import wrap_fork_inherited
    initial, replacement = torch.zeros(128), torch.zeros(4)
    original = wrap_fork_inherited(initial.data_ptr(),512,b"testview",1)
    different = wrap_fork_inherited(replacement.data_ptr(),16,b"testview",2)
    value = DraftWeights(NS(),transfer="packed",group_size=4)
    freed=[]
    value.worker=NS(free_tensor=lambda item,**kwargs:freed.append(item))
    value.storage="stream"
    owner=DeviceTensor(initial.data_ptr(),(128,),torch.float32,buffer=original)
    value.handles=[owner]
    view=different.tensor((4,),runtime.DataType.FLOAT32)
    assert not hasattr(view,"data_ptr")
    value.overrides={layer:(view,) for layer in range(64)}
    value.overrides[0]=(original.tensor((4,),runtime.DataType.FLOAT32,byte_offset=16),)
    with pytest.raises(ValueError,match="aliases"):value.release_replaced_weights()
    assert not freed
    value.overrides[0]=(view,)
    assert value.release_replaced_weights()==512 and freed==[owner]
    assert value.for_group(0)==((view,),)*4
    value.close();original.close();different.close()
