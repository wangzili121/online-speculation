"""Orthrus 的临时任务池预算；不改变模型计算或运行时全局设置。"""


def bounded_ring_policy(native, minimum_mib):
    if type(minimum_mib) is not int or minimum_mib < 256 or minimum_mib & (minimum_mib - 1):
        raise ValueError("ring heap floor must be a power of two, at least 256 MiB")
    rings = native["ring_heap_bytes"]
    if len(rings) != 4 or any(type(n) is not int or n <= 0 for n in rings):
        raise ValueError("expected four positive native ring heap budgets")
    # Worker prepare 先分配这些池，随后沿用 native admission 检查剩余显存。
    # 不减小 native 预算，也不从后续权重、cache 或通信预留中挪用空间。
    return {**native, "ring_heap_bytes": [max(n, minimum_mib << 20) for n in rings],
            "orthrus_ring_floor_mib": minimum_mib, "native_ring_heap_bytes": list(rings)}
