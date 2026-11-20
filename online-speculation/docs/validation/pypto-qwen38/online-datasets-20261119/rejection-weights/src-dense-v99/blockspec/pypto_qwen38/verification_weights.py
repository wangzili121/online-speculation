"""验证结果只影响训练权重；不改变候选、验证或 cache 提交。"""


def rejection_weights(valid, first_rejection, *, block=32):
    """拒绝点及之前保持全权重，之后逐位置减半，最低为前者的 1/16。"""
    if type(valid) is not int or not 1 <= valid < block:
        raise ValueError("a nonempty verified prefix within the block is required")
    if first_rejection is not None and (type(first_rejection) is not int
                                        or not 0 <= first_rejection < block - 1):
        raise ValueError("first rejection must be a zero-based candidate offset")
    # 所有块使用相同的 16 倍整数尺度，避免混合 replay 时改变块之间的相对权重。
    weights = ([16] * valid if first_rejection is None or first_rejection >= valid else
               [16 >> min(4, max(0, row - first_rejection)) for row in range(valid)])
    return weights + [0] * (block - valid), sum(weights)
