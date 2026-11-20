"""验证结果只影响训练权重；不改变候选、验证或 cache 提交。"""


def rejection_weights(valid, first_rejection, *, block=32, mode="rejection_decay"):
    """保留历史衰减策略，或只将首次拒绝位置的训练权重提高到 16。"""
    if type(valid) is not int or not 1 <= valid < block:
        raise ValueError("a nonempty verified prefix within the block is required")
    if first_rejection is not None and (type(first_rejection) is not int
                                        or not 0 <= first_rejection < block - 1):
        raise ValueError("first rejection must be a zero-based candidate offset")
    if mode not in {"rejection_decay", "first_rejection"}:
        raise ValueError("unknown rejection weighting mode")
    if mode == "first_rejection":
        # 普通位置均为 1；只提升已观测且有效的拒绝位置，不从错误前缀补标签。
        weights = [16 if row == first_rejection else 1 for row in range(valid)]
    else:
        # 所有块使用相同的 16 倍整数尺度，避免混合 replay 时改变块之间的相对权重。
        weights = ([16] * valid if first_rejection is None or first_rejection >= valid else
                   [16 >> min(4, max(0, row - first_rejection)) for row in range(valid)])
    return weights + [0] * (block - valid), sum(weights)
