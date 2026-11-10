"""AR 复制冷启动专用：冻结前段直接借用主模型的参数 bank。"""

from pathlib import Path

from .base_loading import base_plans
from .head_loading import bank_specs
from .loading import tensor_layout


def _view(bank, shape, dtype, offset):
    import torch
    from pypto.runtime.task_interface import DataType
    wire = DataType.FLOAT32 if dtype == torch.float32 else DataType.BFLOAT16
    return bank.buffer.tensor(shape, wire, byte_offset=offset)


def borrow_prefix(native, head, stop):
    """返回 [0, stop) 的只读视图和少量独立常量；不发布、不修改版本号。"""
    if (head.directory is not None or head.layout.size != 1 or type(stop) is not int or not 1 <= stop <= 64
            or Path(native.model_root).resolve() != head.model_root.resolve()):
        raise ValueError("parameter sharing requires an AR-copy head from the same TP1 target and a frozen prefix")
    plans = base_plans()
    if len(native.banks) != len(plans) or any(tensor_layout(bank) != (plan.shape, plan.dtype)
                                            for bank, plan in zip(native.banks, plans, strict=True)):
        raise ValueError("native target bank ABI changed")
    locations = {plan.tail: (index, plan) for index, plan in enumerate(plans)}
    layers, owned = {}, []
    try:
        for layer in range(stop):
            kind = "self_attn" if layer % 4 == 3 else "linear_attn"
            row = []
            for bank_index, (name, shape, dtype, _) in enumerate(bank_specs(layer)):
                index, plan = locations[f"{kind}.{name}"]
                if dtype != plan.dtype:
                    # 主模型 A_log bank 为 BF16；draft 保留原始 FP32，不能用转换后的值代替。
                    value = head.banks(layer)[bank_index]
                    handle = native.worker.alloc_tensor(shape, dtype, init=value)
                    owned.append(handle)
                    row.append(handle)
                    continue
                count = len(plan.layers)
                rows = plan.shape[0] // count
                if shape != (rows, plan.shape[1]):
                    raise ValueError("incompatible draft/target per-layer bank layout")
                offset = plan.layers.index(layer) * rows * plan.shape[1] * dtype.itemsize
                row.append(_view(native.banks[index], shape, dtype, offset))
            layers[layer] = tuple(row)
        return layers, owned
    except BaseException:
        for handle in reversed(owned):
            native.worker.free_tensor(handle)
        raise
