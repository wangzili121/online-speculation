"""Draft 投影的低秩更新图；推理仍接收合并后的普通 attention 权重。"""

from dataclasses import dataclass
from math import prod

from .training_flat import FlatLayout
from .training_graph import GraphBuilder, Value
from .training_suffix_graph import Parameter, SuffixGraph


@dataclass(frozen=True)
class Adapter:
    projection: Parameter
    a: Parameter
    b: Parameter


def weight_element(graph, a, b, kind, **options):
    """大矩阵用 4096 元素 tile；小矩阵保留足够的独立任务。"""
    rows = prod(a.shape) // 4096
    return graph.element(a, b, kind, tile=4096, task_group=min(16, max(1, rows // 32)), **options)


class LoRASuffixGraph(SuffixGraph):
    """W = W0 + BA；其余 attention 参数保留原来的完整梯度。

    复用已有的 dense dW，再收缩成 dA/dB。前向和 dX 使用已合并的 W，
    与普通 draft 执行相同；本类不负责参数发布，也不自动接入在线服务。
    """

    def __init__(self, last_layers, *args, rank=64, **kwargs):
        if type(rank) is not int or rank < 64 or rank % 64:
            raise ValueError("a positive 64-aligned adapter rank is required")
        if kwargs.get("parameter_scope", "full") != "full":
            raise ValueError("low-rank projection updates retain the full attention parameter scope")
        super().__init__(last_layers, *args, **kwargs)
        self.adapter_rank = rank
        self.original_parameters = tuple(self.parameters)
        self.dense_gradients = dict(self.gradients)
        producers = {v.name: node for node in self.nodes for v in node.outputs}
        self.adapters = {}
        parameters, gradients = [], {}
        for parameter in self.original_parameters:
            dw = self.dense_gradients[parameter.name]
            node = producers[dw.name]
            if node.kind != "linear" or not node.options.get("gradient_weight"):
                parameters.append(parameter)
                gradients[parameter.name] = dw
                continue
            rows, columns = parameter.shape
            if rank > min(rows, columns):
                raise ValueError("adapter rank exceeds projection dimensions: " + parameter.name)
            a = Parameter(parameter.name + "_lora_a", parameter.layer, 2 * parameter.bank,
                          (rank, columns))
            b = Parameter(parameter.name + "_lora_b", parameter.layer, 2 * parameter.bank + 1,
                          (rows, rank))
            av, bv = self.input(a.name, a.shape), self.input(b.name, b.shape)
            # 原 dW 已按 BF16 梯度语义舍入；Cube 的输入显式转回 BF16。
            rounded = weight_element(self, dw, dw, "scale", scale=1.)
            gradients[a.name] = self.adapter_gradient(bv, rounded, transpose_a=True, transpose_b=False)
            gradients[b.name] = self.adapter_gradient(rounded, av, transpose_b=True)
            parameters.extend((a, b))
            self.adapters[parameter.name] = Adapter(parameter, a, b)
        self.parameters = sorted(parameters, key=lambda p: (p.layer, p.bank, p.name))
        self.gradients = gradients
        assert self.adapters and set(gradients) == {p.name for p in self.parameters}

    def adapter_gradient(self, a, b, *, transpose_a=False, transpose_b=True):
        # 低秩梯度的输出 tile 少；每组最多 16 个，同时尽量铺满 32 个任务块。
        rows = a.shape[1] if transpose_a else a.shape[0]
        columns = b.shape[0] if transpose_b else b.shape[1]
        tasks = rows // (64 if rows % 64 == 0 else 32) * (columns // 64)
        group = min(16, max(1, tasks // 32))
        return self.mm(a, b, transpose_a=transpose_a, transpose_b=transpose_b,
                       gradient_weight=True, task_group=group)

    @property
    def dense_parameter_count(self):
        return sum(prod(p.shape) for p in self.original_parameters)

    @property
    def trainable_parameter_count(self):
        return sum(prod(p.shape) for p in self.parameters)

    def initial_master(self, layout, read_parameter, *, seed=743):
        """仅初始化/加载时在 CPU 整理参数；B=0 保持初始推理 head 不变。"""
        import torch

        if [e.parameter for e in layout.entries] != self.parameters:
            raise ValueError("adapter optimizer layout differs from its graph")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        a_names = {entry.a.name for entry in self.adapters.values()}
        b_names = {entry.b.name for entry in self.adapters.values()}
        master = torch.zeros(layout.shape, dtype=torch.float32)
        for entry in layout.entries:
            parameter = entry.parameter
            if parameter.name in a_names:
                bound = parameter.shape[1] ** -.5
                value = torch.empty(parameter.shape).uniform_(-bound, bound, generator=generator)
            elif parameter.name in b_names:
                value = torch.zeros(parameter.shape)
            else:
                value = read_parameter(parameter)
                expected = torch.float32 if parameter.raw_dtype == "fp32" else torch.bfloat16
                if tuple(value.shape) != parameter.shape or value.dtype != expected:
                    raise ValueError("initial attention parameter schema differs: " + parameter.name)
            master[0, entry.offset:entry.offset + value.numel()].copy_(value.reshape(-1).float())
        if not bool(torch.isfinite(master).all()):
            raise ValueError("nonfinite initial adapter/attention parameters")
        return master


class LoRAMergeGraph(GraphBuilder):
    """将增量合并为普通 BF16 head，可选融合 BA 与 W0 相加。

    分开执行时 scratch 借用已完成反向的 dense dW；融合路径不输出 delta。
    FP32 norm
    等参数与适配器优化器共享 special buffer，保持原来的发布精度和布局。
    """

    def __init__(self, graph, *, fused=False):
        if type(fused) is not bool:
            raise ValueError("fused must be a boolean")
        self.shape = graph.shape
        self.inputs, self.nodes, self.gradients = {}, [], {}
        self.targets, self.scratch = {}, {}
        self.base_inputs, self.optimizer_inputs = {}, {}
        self.layout = FlatLayout(graph.original_parameters)
        low = FlatLayout(graph.parameters)
        specials = lambda layout: [(e.parameter.name, e.special, e.parameter.shape)
                                   for e in layout.entries if e.special is not None]
        if specials(self.layout) != specials(low):
            raise ValueError("merged head and adapter FP32 special layouts differ")
        for parameter in graph.original_parameters:
            name = parameter.name
            if name in graph.adapters:
                adapter = graph.adapters[name]
                base = self.input("base_" + name, parameter.shape)
                self.base_inputs[base.name] = name
                av = self.input(adapter.a.name, adapter.a.shape)
                bv = self.input(adapter.b.name, adapter.b.shape)
                self.optimizer_inputs.update({av.name: av.name, bv.name: bv.name})
                if fused:
                    self.targets[name] = self.op("lora_merge", (base, bv, av),
                        ((parameter.shape, "bf16"),), rows=parameter.shape[0], columns=parameter.shape[1],
                        rank=graph.adapter_rank)[0]
                else:
                    delta = self.mm(bv, av, transpose_b=False, gradient_weight=True)
                    self.scratch[name] = delta
                    self.targets[name] = weight_element(self, base, delta, "add")
            elif parameter.execution_dtype != "fp32":
                current = self.input("current_" + name, parameter.shape)
                self.optimizer_inputs[current.name] = name
                self.targets[name] = self.op("parameter_copy", (current,), ((parameter.shape, "bf16"),))[0]

    def input(self, name, shape, dtype="bf16"):
        value = Value(name, tuple(shape), dtype)
        self.inputs[name] = value
        return value
