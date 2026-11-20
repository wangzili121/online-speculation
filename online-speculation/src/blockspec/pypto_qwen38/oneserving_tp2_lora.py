"""可选低秩在线更新；推理和 TP2 发布始终使用合并后的普通 draft 权重。"""

from math import isfinite

import torch
from safetensors import safe_open

from .oneserving_tp2_delayed import TP2DelayedOnline
from .oneserving_tp2_online import master_from_shards, read_raw_parameter
from .training_executor import TrainingPrograms
from .training_flat import FlatLayout
from .training_lora_graph import LoRAMergeGraph, LoRASuffixGraph
from .training_optimizer_programs import make_reduce, make_stats


class TP2LoRAOnline(TP2DelayedOnline):
    def __init__(self, checkpoint, *args, adapter_rank=64, adapter_seed=743,
                 adapter_merge_backend="chained", **kwargs):
        if checkpoint is not None or kwargs.get("parameter_scope", "full") != "full":
            raise ValueError("merged LoRA currently requires a full parameter scope and fresh Adam")
        if type(adapter_rank) is not int or adapter_rank < 64 or adapter_rank % 64:
            raise ValueError("adapter rank must be a positive multiple of 64")
        if type(adapter_seed) is not int or adapter_seed < 0:
            raise ValueError("adapter seed must be a nonnegative integer")
        if adapter_merge_backend not in ("chained", "fused"):
            raise ValueError("adapter merge backend must be chained or fused")
        self.adapter_rank, self.adapter_seed = adapter_rank, adapter_seed
        self.adapter_merge_backend = adapter_merge_backend
        super().__init__(checkpoint, *args, **kwargs)

    def build_training_graph(self):
        graph = LoRASuffixGraph(len(self.layers), rank=self.adapter_rank, objective=self.objective,
            focus_weight=self.focus_weight, position_weighted=self.position_weighting != "uniform")
        self.full_layout = FlatLayout(graph.original_parameters)
        return graph

    def publication_layout(self):
        return self.full_layout

    def load_head(self, directory):
        raise ValueError("merged LoRA requires a new learner to replace its immutable starting head")

    def compile_training_views(self):
        self.merge_graph = LoRAMergeGraph(self.graph, fused=self.adapter_merge_backend == "fused")
        self.merge_programs = TrainingPrograms(self.merge_graph, self.config, combined=True)
        meta = lambda shape: torch.empty(shape, dtype=torch.float32, device="meta")
        entry, rows = make_stats(self.full_layout.shape, input_bf16=True)
        self.merged_stat = entry.compile(torch.empty(self.full_layout.shape, dtype=torch.bfloat16, device="meta"),
                                         meta((rows, 8)), meta((rows, 8)), config=self.config)
        self.merged_reductions, self.merged_counts = [], [rows]
        while rows > 1:
            entry, count = make_reduce(rows)
            self.merged_reductions.append(entry.compile(meta((rows, 8)), meta((rows, 8)),
                                                        meta((count, 8)), meta((count, 8)), config=self.config))
            self.merged_counts.append(count)
            rows = count
        return [*self.merge_programs.programs.values(), self.merged_stat, *self.merged_reductions]

    def training_view_bytes(self):
        # W0 和两个完整执行视图；合并的 FP32 delta 借用已退休的 dW。
        return 6 * self.full_layout.elements + 64 * sum(self.merged_counts)

    def can_reuse_publication_execution(self, compact):
        # 小适配器执行缓冲不能充当完整 head 的打包区。
        return False

    def initial_parameters(self, directory):
        with safe_open(directory / "rank0.safetensors", framework="pt", device="cpu") as left, safe_open(
                directory / "rank1.safetensors", framework="pt", device="cpu") as right:
            return self.graph.initial_master(self.optimizer.layout,
                lambda parameter: read_raw_parameter(parameter, (left, right)), seed=self.adapter_seed)

    def initialize_training_views(self):
        base = master_from_shards(self.full_layout, self.runner.head_directory).bfloat16()
        self.merged_base = self.upload_new(base)
        self.merged_execution = [self.upload_new(base) for _ in range(2)]
        self.merged_stats = [[self.worker.alloc_tensor((count, 8), torch.float32) for _ in range(2)]
                             for count in self.merged_counts]
        # base 只提供投影矩阵；special 参数直接来自适配器优化器的同布局视图。
        base_views = self.full_layout.views(self.merged_base, self.optimizer.execution[0][1])
        self.merge_base_inputs = {alias: base_views[name] for alias, name in self.merge_graph.base_inputs.items()}

    def render_merged(self, index, adapter_buffers):
        adapters = self.optimizer.layout.views(*adapter_buffers)
        current = self.full_layout.views(self.merged_execution[index], adapter_buffers[1])
        values = dict(self.merge_base_inputs)
        values.update({alias: adapters[name] for alias, name in self.merge_graph.optimizer_inputs.items()})
        values.update({value.name: current[name] for name, value in self.merge_graph.targets.items()})
        # dW 已用于求 dA/dB，Adam 之后没有读者；仅覆盖该工作区，不覆盖 W0。
        values.update({value.name: self.workspace.values[self.graph.dense_gradients[name].name]
                       for name, value in self.merge_graph.scratch.items()})
        entry = self.merge_programs.programs["suffix"]
        self.worker.run(entry, *(values[value.name] for value in self.merge_programs.arguments),
                        0, 32, 31, 31, config=self.config)
        self.worker.run(self.merged_stat, self.merged_execution[index], *self.merged_stats[0], config=self.config)
        for i, program in enumerate(self.merged_reductions):
            self.worker.run(program, *self.merged_stats[i], *self.merged_stats[i + 1], config=self.config)
        self.optimizer.run(self.optimizer.programs.summary, *self.merged_stats[-1], self.optimizer.summary)
        self.worker.copy_from(self.optimizer.summary_host.data_ptr(), self.optimizer.summary.data_ptr, 96)
        norm, scale, valid = self.optimizer.summary_host[:, 0].tolist()
        if valid != 1 or not isfinite(norm) or not isfinite(scale):
            raise FloatingPointError("nonfinite merged LoRA head; serving version remains unchanged")
        return self.full_layout, (self.merged_execution[index], adapter_buffers[1])

    def restore_training_views(self):
        self.render_merged(self.optimizer.index, self.optimizer.execution[self.optimizer.index])

    def training_tensors(self):
        return self.optimizer.tensors() | self.full_layout.views(
            self.merged_execution[self.optimizer.index], self.optimizer.execution[self.optimizer.index][1])

    def publication_inputs(self, layout, buffers):
        if layout is not self.optimizer.layout:
            raise ValueError("unexpected adapter optimizer layout")
        # 每次更新都刷新 learner 的完整 W；只有到发布间隔才传给两个 serving rank。
        # 合并和有限性检查发生在 optimizer master/index 或 serving version 切换之前。
        return self.render_merged(1 - self.optimizer.index, buffers)
