"""一次设备调用完成全局 norm；保留原分级归约和非有限值检查。"""

from math import isfinite, prod


def statistics_source(shape):
    """生成直线调用图，缓冲区由调用者持有，归约顺序与逐次提交相同。"""
    if not shape or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("positive integer statistics shape required")
    elements = prod(shape)
    tile = min(4096, elements)
    if elements % tile:
        raise ValueError("aligned gradient tensor required")
    counts = [elements // tile]
    while counts[-1] > 1:
        counts.append((counts[-1] + 63) // 64)
    buffers = [f"{kind}{i}" for i in range(len(counts)) for kind in ("sums", "flags")]
    args = ["value", *buffers, "output"]
    parameters = ["value: pl.Tensor", *[f"{name}: pl.Out[pl.Tensor]" for name in args[1:]]]
    lines = [f"def statistics_chip({', '.join(parameters)}):",
             "    sums0, flags0 = statistics(value, sums0, flags0)"]
    for i in range(1, len(counts)):
        lines.append(f"    sums{i}, flags{i} = reduce_{i}(sums{i-1}, flags{i-1}, sums{i}, flags{i})")
    last = len(counts) - 1
    lines.extend([f"    output = summarize(sums{last}, flags{last}, output)", "    return output",
                  f"def statistics_host({', '.join(parameters)}):",
                  f"    statistics_chip({', '.join(args)}, device=0)"])
    return "\n".join(lines) + "\n", counts


def make_statistics(shape, max_norm):
    from .training_flat import make_summary
    from .training_optimizer_programs import make_reduce, make_stats
    from .training_pipeline import bind

    if isinstance(max_norm, bool) or not isfinite(max_norm) or max_norm <= 0:
        raise ValueError("positive finite clipping norm required")
    text, counts = statistics_source(shape)
    functions = {"statistics": make_stats(shape, inline=True)[0],
                 "summarize": make_summary(max_norm, inline=True)}
    functions.update({f"reduce_{i}": make_reduce(rows, inline=True)[0]
                      for i, rows in enumerate(counts[:-1], 1)})
    return bind(text, functions, "statistics"), counts
