"""根据实际图和扁平布局估计新增显存；模型层数可选不代表任意 N 都装得下。"""

from math import prod


def suffix_memory(graph, layout, replay_blocks, *, compact_optimizer=False):
    size = lambda value:prod(value.shape)*{"bf16":2,"fp32":4}[value.dtype]
    names = {"hidden","cos","sin","teacher"}|{
        value.name for history in graph.histories.values() for value in history.values()}
    if getattr(graph, "position_weighted", False):
        names.add("position_weights")
    feedback = replay_blocks*sum(size(graph.inputs[name]) for name in names)
    execution = 2*layout.elements+4*layout.special_elements
    gradients = {value.name for value in graph.gradients.values()}
    workspace = sum(size(value) for node in graph.nodes for value in node.outputs if value.name not in gradients)
    frozen = sum(size(graph.inputs[name]) for name in graph.frozen)
    frozen += sum(size(graph.inputs[p.name]) for p in getattr(graph, "frozen_attention", ()))
    rows,statistics = layout.elements//min(4096,layout.elements),0
    while True:
        statistics += rows*64
        if rows == 1:
            break
        rows = (rows+63)//64
    optimizer = (24 if compact_optimizer else 28)*layout.elements+2*execution+32*len(layout.table)+statistics+96
    initialize = 2*layout.elements+4*layout.special_elements+4*graph.shape.dimension+32
    return dict(serving=2*execution+feedback,
                learner=optimizer+workspace+max(initialize,frozen+feedback)+32)


def require_suffix_memory(graph, layout, replay_blocks, serving_free, learner_free, *, reserve=256<<20,
                          compact_optimizer=False):
    required = suffix_memory(graph,layout,replay_blocks,compact_optimizer=compact_optimizer)
    for role,available in (("serving",serving_free),("learner",learner_free)):
        if required[role]+reserve > available:
            raise RuntimeError(f"last-{graph.last_layers} {role} needs at least {required[role]+reserve} "
                               f"additional free bytes (including reserve); available {available}; "
                               "reduce last-layers or use a deployment with more free device memory")
    return required
