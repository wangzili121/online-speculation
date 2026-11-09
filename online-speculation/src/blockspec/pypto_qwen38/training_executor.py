"""执行显式训练图；先编译/注册，再在同一 PyPTO worker 内重放 suffix。"""

import json

from .training_layout_programs import make_layout, make_rope, make_softmax
from .training_programs import make_element, make_linear, make_rms, make_tv


class TrainingPrograms:
    def __init__(self, graph, config):
        import torch
        self.graph, self.config, self.programs, self.sequence = graph, config, {}, []
        def meta(value):
            return torch.empty(value.shape, dtype=getattr(torch, "bfloat16" if value.dtype == "bf16" else "float32"), device="meta")
        for node in graph.nodes:
            key = json.dumps((node.kind, [(v.shape, v.dtype) for v in (*node.inputs, *node.outputs)], node.options), sort_keys=True)
            extra = ()
            if key not in self.programs:
                shape = node.inputs[0].shape
                if node.kind == "linear":
                    entry = make_linear(**node.options)
                elif node.kind == "element":
                    entry = make_element(shape, **node.options)
                elif node.kind in {"norm", "norm_dx", "norm_dw"}:
                    mode = {"norm": "forward", "norm_dx": "input", "norm_dw": "weight"}[node.kind]
                    entry = make_rms(*shape, mode=mode)
                elif node.kind in {"layout", "q_join", "join", "reduce_block"}:
                    entry = make_layout(graph.shape, node.kind, **node.options)
                elif node.kind == "rope":
                    entry = make_rope(graph.shape, shape[0], **node.options)
                elif node.kind in {"softmax", "softmax_grad"}:
                    entry = make_softmax(*shape, backward=node.kind == "softmax_grad")
                elif node.kind == "tv":
                    entry = make_tv(*shape)
                else:
                    raise ValueError(node.kind)
                if node.kind in {"join", "reduce_block", "softmax"}:
                    extra = (0, 32)
                elif node.kind == "tv":
                    extra = (31, 31)
                print("compiling suffix", node.kind, shape, node.options, flush=True)
                self.programs[key] = entry.compile(*map(meta, (*node.inputs, *node.outputs)), *extra, config=config)
            self.sequence.append(self.programs[key])


class TrainingWorkspace:
    def __init__(self, programs, worker):
        import torch
        self.programs, self.worker, self.values = programs, worker, {}
        self.handles = []
        try:
            for node in programs.graph.nodes:
                for value in node.outputs:
                    dtype = torch.bfloat16 if value.dtype == "bf16" else torch.float32
                    tensor = worker.alloc_tensor(value.shape, dtype)
                    self.values[value.name] = tensor
                    self.handles.append(tensor)
        except BaseException:
            self.close()
            raise

    def run(self, inputs, *, past, width, valid, normalizer=None):
        s = self.programs.graph.shape
        normalizer = valid if normalizer is None else normalizer
        if normalizer < valid:
            raise ValueError("normalizer must include all reached positions")
        if not 0 <= past <= s.context - width or not 2 <= width <= s.block or not 1 <= valid < width:
            raise ValueError("history, block and reached-row boundary required")
        for name, value in self.programs.graph.inputs.items():
            tensor = inputs[name]
            if tuple(tensor.shape) != value.shape:
                raise ValueError(f"suffix input shape changed: {name}")
        values = {**inputs, **self.values}
        for node, program in zip(self.programs.graph.nodes, self.programs.sequence, strict=True):
            extra = ((past, width) if node.kind in {"join", "reduce_block", "softmax"}
                     else (valid, normalizer) if node.kind == "tv" else ())
            self.worker.run(program, *(values[v.name] for v in (*node.inputs, *node.outputs)),
                            *extra, config=self.programs.config)
        return {name: values[value.name] for name, value in self.programs.graph.gradients.items()}

    def close(self):
        while self.handles:
            self.worker.free_tensor(self.handles[-1])
            self.handles.pop()
        self.values.clear()
