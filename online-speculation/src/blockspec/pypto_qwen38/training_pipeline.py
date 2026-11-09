"""将显式图编排成一个 CHIP 入口，复用独立验证过的 inline 阶段。"""

import hashlib
import linecache

from .training_layout_programs import make_layout, make_rope, make_softmax
from .training_programs import make_element, make_linear, make_rms, make_tv


def stage(graph, node, *, accumulate=False):
    shape, options = node.inputs[0].shape, {**node.options, "inline": True}
    if node.kind == "linear":
        options["accumulate"] = accumulate and node.options["gradient_weight"]
        return make_linear(**options)
    if node.kind == "element":
        return make_element(shape, **options)
    if node.kind in {"norm", "norm_dx", "norm_dw"}:
        options["accumulate"] = accumulate and node.kind == "norm_dw"
        return make_rms(*shape, mode={"norm": "forward", "norm_dx": "input", "norm_dw": "weight"}[node.kind], **options)
    if node.kind in {"layout", "q_join", "join", "reduce_block"}:
        return make_layout(graph.shape, node.kind, **options)
    if node.kind == "rope":
        return make_rope(graph.shape, shape[0], **options)
    if node.kind in {"softmax", "softmax_grad"}:
        return make_softmax(*shape, backward=node.kind == "softmax_grad", **options)
    if node.kind == "tv":
        return make_tv(*shape, **options)
    raise ValueError(node.kind)


def source(graph, *, accumulate=False):
    values = (*graph.inputs.values(), *(value for node in graph.nodes for value in node.outputs))
    inputs = set(graph.inputs)
    gradients = {v.name for v in graph.gradients.values()} if accumulate else set()
    parameters = [f"{v.name}: pl.Tensor" if v.name in inputs else
                  f"{v.name}: pl.InOut[pl.Tensor]" if v.name in gradients else f"{v.name}: pl.Out[pl.Tensor]" for v in values]
    scalars = ("past", "width", "valid", "normalizer")
    parameters += [f"{name}: pl.Scalar[pl.INT32]" for name in scalars]
    arguments = ", ".join([v.name for v in values]+list(scalars))
    body = [f"def suffix_chip({', '.join(parameters)}):"]
    for i, node in enumerate(graph.nodes):
        extra = (("past", "width") if node.kind in {"join", "reduce_block", "softmax"}
                 else ("valid", "normalizer") if node.kind == "tv" else ())
        result = ", ".join(v.name for v in node.outputs)
        args = ", ".join([v.name for v in (*node.inputs, *node.outputs)]+list(extra))
        body.append(f"    {result} = stage_{i}({args})")
    body.append("    return " + ", ".join(v.name for node in graph.nodes for v in node.outputs))
    body += [f"def suffix_host({', '.join(parameters)}):", f"    suffix_chip({arguments}, device=0)"]
    return "\n".join(body)+"\n", values


def make_pipeline(graph, *, accumulate=False):
    text, values = source(graph, accumulate=accumulate)
    return bind(text, {f"stage_{i}": stage(graph, node, accumulate=accumulate)
                      for i, node in enumerate(graph.nodes)}, "suffix"), values


def bind(text, functions, prefix):
    import pypto.language as pl
    filename = f"<orthrus_suffix_{hashlib.sha256(text.encode()).hexdigest()}>"
    # 仅从受控图节点生成直线调用，linecache 供 JIT 读取同一份可检查的 Python 源码。
    linecache.cache[filename] = (len(text), None, text.splitlines(True), filename)
    namespace = {"pl": pl, **functions}
    exec(compile(text, filename, "exec"), namespace)
    namespace[prefix+"_chip"] = pl.jit(namespace[prefix+"_chip"])
    return pl.jit.host(namespace[prefix+"_host"])
