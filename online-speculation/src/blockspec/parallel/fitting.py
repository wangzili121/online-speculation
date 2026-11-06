# 中文学习版｜完整阅读路线见 docs/QWEN38_CODE_WALKTHROUGH.md
# 文件职责：训练数据及独立训练工具：Qwen vLLM 离线入口主要复用 TokenDataset / BatchStream，文件内其他 fitting 入口不是该服务的必经调用。
# 约定：B=batch，K=块长（含 anchor），H=hidden_size，V=词表大小；通常 B=1。
# 注释不改变执行代码；续行与同一逻辑语句一起阅读。

"""Single-device full-distribution fitting with deterministic data and resume state."""

# 依赖：从 array 导入 array；供本模块调用。
from array import array
# 依赖：从 contextlib 导入 nullcontext；供本模块调用。
from contextlib import nullcontext
# 依赖：从 dataclasses 导入 asdict, dataclass；供本模块调用。
from dataclasses import asdict, dataclass
# 依赖：导入 hashlib。
import hashlib
# 依赖：导入 json。
import json
# 依赖：导入 math。
import math
# 依赖：从 pathlib 导入 Path；供本模块调用。
from pathlib import Path
# 依赖：导入 time。
import time

# 依赖：导入 torch。
import torch

# 依赖：从 ..accelerator 导入 autocast, name, rng_state, set_rng_state, synchronize；供本模块调用。
from ..accelerator import (autocast, name as device_name, rng_state, set_rng_state,
                           synchronize)
# 依赖：从 .training 导入 distillation_loss, distillation_update, sample_anchors；供本模块调用。
from .training import distillation_loss, distillation_update, sample_anchors
# 依赖：从 .weights 导入 file_sha256, load_checkpoint, save_checkpoint；供本模块调用。
from .weights import file_sha256, load_checkpoint, save_checkpoint


# ── TokenDataset ──
# 职责：类型 `TokenDataset`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class TokenDataset:
    """Indexed tokenized JSONL with optional valid draft-block spans."""

    # ── TokenDataset.__init__ ──
    # 职责：扫描 JSONL，验证 token 和合法区间，记录足够长样本的文件偏移及数据指纹，避免把所有文本常驻内存。
    def __init__(self, path, vocab_size, sequence_length):
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：positive vocabulary and sequence window required。
        if any(type(n) is not int or n < 1 for n in (vocab_size, sequence_length)):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("positive vocabulary and sequence window required")
        self.path = Path(path).resolve()
        self.vocab_size, self.sequence_length = vocab_size, sequence_length
        self.offsets, self.skipped = array("Q"), 0
        with self.path.open("rb") as handle:
            # 循环：重复处理直到条件不再成立；留意循环中的 break 和状态推进。
            while True:
                # 数据：从原生卷积工作缓存选择接受前缀尾部的起点。
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    # 结束当前循环，继续循环之后的流程。
                    break
                if not line.strip():
                    # 跳过本次迭代的后续步骤，进入下一项。
                    continue
                ids, _ = self._decode_record(line)
                if len(ids) >= sequence_length:
                    # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
                    self.offsets.append(offset)
                else:
                    # 状态推进：在原有值上累加/更新；若目标是张量，要注意这里的原地语义。
                    self.skipped += 1
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：training data needs a complete sequence window。
        if not self.offsets:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("training data needs a complete sequence window")
        self.fingerprint = file_sha256(self.path)

    # ── TokenDataset._decode_record ──
    # 职责：解析一行 JSONL，校验 token ID 和合法 anchor 区间，拒绝越界/错误数据。
    def _decode_record(self, line):
        record = json.loads(line)
        ids = record.get("input_ids")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：each JSONL record requires valid input_ids。
        if not isinstance(ids, list) or not ids or any(type(x) is not int or not 0 <= x < self.vocab_size for x in ids):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("each JSONL record requires valid input_ids")
        ranges = record.get("anchor_ranges")
        if ranges is None:
            ranges = [(0, len(ids))]
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：anchor_ranges must be nonempty half-open token spans。
        if (
            not isinstance(ranges, list)
            or not ranges
            or any(
                not isinstance(span, (list, tuple))
                or len(span) != 2
                or type(span[0]) is not int
                or type(span[1]) is not int
                or not 0 <= span[0] < span[1] <= len(ids)
                for span in ranges
            )
        ):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("anchor_ranges must be nonempty half-open token spans")
        ordered = sorted((int(start), int(stop)) for start, stop in ranges)
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：anchor_ranges must not overlap。
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("anchor_ranges must not overlap")
        # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
        return ids, ordered

    # ── TokenDataset._decode ──
    # 职责：取一条记录中的 token 序列，舍弃附加区间信息的简化接口。
    def _decode(self, line):
        return self._decode_record(line)[0]

    # ── TokenDataset.__len__ ──
    # 职责：返回可供训练读取的合格记录数。
    def __len__(self):
        return len(self.offsets)

    # ── TokenDataset.__getitem__ ──
    # 职责：按文件偏移读取指定样本的 token ID；不做模型计算。
    def __getitem__(self, index):
        return self.record(index)[0]

    # ── TokenDataset.record ──
    # 职责：按索引读取一条原始合格记录，保留 token ID 与 anchor_ranges。
    def record(self, index):
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            ids, ranges = self._decode_record(handle.readline())
        # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
        return torch.tensor(ids, dtype=torch.long), ranges


# ── BatchStream ──
# 职责：类型 `BatchStream`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class BatchStream:
    # ── BatchStream.__init__ ──
    # 职责：创建固定种子的 CPU 数据抽样 RNG、样本排列和游标；数据随机数与 NPU 生成采样 RNG 分开。
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.rng = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(dataset), generator=self.rng)
        self.cursor = self.epoch = 0

    # ── BatchStream._next_record ──
    # 职责：按确定性排列取下一条记录，轮次结束后重新洗牌并更新游标。
    def _next_record(self):
        if self.cursor == len(self.order):
            self.order = torch.randperm(len(self.dataset), generator=self.rng)
            self.cursor = 0
            # 状态推进：在原有值上累加/更新；若目标是张量，要注意这里的原地语义。
            self.epoch += 1
        record = self.dataset.record(int(self.order[self.cursor]))
        # 状态推进：在原有值上累加/更新；若目标是张量，要注意这里的原地语义。
        self.cursor += 1
        return record

    # ── BatchStream.batch ──
    # 职责：抽取固定长度 token 窗口，组成 CPU LongTensor batch；适用于不额外限制 anchor 的数据路径。
    # 输入提示：count：本轮有效 token/候选数量；以赋值来源和所在函数为准，不等同于固定 K。
    def batch(self, count):
        rows, length = [], self.dataset.sequence_length
        for _ in range(count):
            sequence, _ = self._next_record()
            start = int(torch.randint(len(sequence) - length + 1, (), generator=self.rng))
            # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
            rows.append(sequence[start:start + length])
        return torch.stack(rows)

    @staticmethod
    # ── BatchStream._valid_crop ──
    # 职责：寻找既容纳完整 token 窗口、又有合法 K-token 预测块的裁剪位置，返回窗口及候选 anchor。
    def _valid_crop(sequence_length, window, ranges, block_size, rng):
        max_start = sequence_length - window
        intervals = []
        for span_start, span_stop in ranges:
            if span_stop - span_start < block_size:
                # 跳过本次迭代的后续步骤，进入下一项。
                continue
            low = max(0, span_start - window + block_size)
            high = min(max_start, span_stop - block_size)
            if low <= high:
                # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
                intervals.append((low, high))
        if not intervals:
            return None
        merged = []
        for low, high in sorted(intervals):
            if merged and low <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], high))
            else:
                # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
                merged.append((low, high))
        choice = int(
            torch.randint(sum(high - low + 1 for low, high in merged), (), generator=rng)
        )
        start = 0
        for low, high in merged:
            # 数据：本轮实际块长，受配置 K 和剩余上下文空间共同限制。
            width = high - low + 1
            if choice < width:
                start = low + choice
                # 结束当前循环，继续循环之后的流程。
                break
            # 状态推进：在原有值上累加/更新；若目标是张量，要注意这里的原地语义。
            choice -= width
        # 数据：待主模型验证的 token ID，一维长度最多 K−1。
        candidates = []
        for span_start, span_stop in ranges:
            low = max(span_start, start) - start
            high = min(span_stop - block_size, start + window - block_size) - start
            if low <= high:
                # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
                candidates.append(torch.arange(low, high + 1, dtype=torch.long))
        # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
        return start, torch.cat(candidates)

    # ── BatchStream.batch_with_anchor_candidates ──
    # 职责：抽取文本窗口并返回能容纳完整预测块的 anchor 候选，避免跨不允许的语料边界。
    # 输入提示：count：本轮有效 token/候选数量；以赋值来源和所在函数为准，不等同于固定 K。
    def batch_with_anchor_candidates(self, count, block_size):
        """Return crops and local anchors whose complete blocks stay in one span."""
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：positive batch and block sizes required。
        if type(count) is not int or count < 1 or type(block_size) is not int or block_size < 1:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("positive batch and block sizes required")
        # 数据：待主模型验证的 token ID，一维长度最多 K−1。
        rows, candidates = [], []
        attempts = 0
        limit = max(len(self.dataset) * 2, count)
        # 循环：重复处理直到条件不再成立；留意循环中的 break 和状态推进。
        while len(rows) < count and attempts < limit:
            sequence, ranges = self._next_record()
            # 状态推进：在原有值上累加/更新；若目标是张量，要注意这里的原地语义。
            attempts += 1
            selected = self._valid_crop(
                len(sequence), self.dataset.sequence_length, ranges, block_size, self.rng
            )
            if selected is None:
                # 跳过本次迭代的后续步骤，进入下一项。
                continue
            start, valid = selected
            # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
            rows.append(sequence[start:start + self.dataset.sequence_length])
            # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
            candidates.append(valid)
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：training data has too few ranges containing a complete draft block。
        if len(rows) != count:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("training data has too few ranges containing a complete draft block")
        # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
        return torch.stack(rows), candidates

    # ── BatchStream.state_dict ──
    # 职责：保存数据 RNG、排列和游标，以便恢复后继续相同的抽样序列。
    def state_dict(self):
        return {"rng": self.rng.get_state(), "order": self.order.clone(),
                "cursor": self.cursor, "epoch": self.epoch}

    # ── BatchStream.load_state_dict ──
    # 职责：校验并恢复数据流状态，不涉及模型权重或 AdamW 状态。
    def load_state_dict(self, state):
        order = state["order"].cpu()
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：invalid data permutation or cursor。
        if (order.dtype != torch.long or order.shape != (len(self.dataset),)
                or not torch.equal(order.sort().values, torch.arange(len(self.dataset)))
                or type(state["cursor"]) is not int or not 0 <= state["cursor"] <= len(order)
                or type(state["epoch"]) is not int or state["epoch"] < 0):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("invalid data permutation or cursor")
        self.rng.set_state(state["rng"].cpu())
        self.order, self.cursor, self.epoch = order, state["cursor"], state["epoch"]


@dataclass(frozen=True)
# ── FitConfig ──
# 职责：类型 `FitConfig`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class FitConfig:
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    steps: int = 1000
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    batch_size: int = 1
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    sequence_length: int = 256
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    anchors_per_sequence: int = 4
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    accumulate: int = 1
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    chunk_rows: int = 32
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    learning_rate: float = 1e-4
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    warmup_steps: int = 50
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    minimum_lr_ratio: float = 0.1
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    weight_decay: float = 0.01
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    clip_grad: float = 1.0
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    precision: str = "fp32"
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    backend: str = "sdpa"
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    seed: int = 731
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    optimizer_impl: str = "single"

    # ── FitConfig.__post_init__ ──
    # 职责：数据类构造后校验配置，拒绝无法支持的参数组合。
    def __post_init__(self):
        counts = (self.steps, self.batch_size, self.sequence_length, self.anchors_per_sequence,
                  self.accumulate, self.chunk_rows)
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：positive integer training dimensions required。
        if any(type(n) is not int or n < 1 for n in counts):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("positive integer training dimensions required")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：warmup must fit within the full training schedule。
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps < self.steps:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("warmup must fit within the full training schedule")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：finite positive learning rate/clip and valid decay required。
        if (any(not math.isfinite(x) for x in (self.learning_rate, self.minimum_lr_ratio,
                                               self.weight_decay, self.clip_grad))
                or self.learning_rate <= 0 or self.clip_grad <= 0 or self.weight_decay < 0
                or not 0 <= self.minimum_lr_ratio <= 1):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("finite positive learning rate/clip and valid decay required")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：training supports FP32/BF16 and eager/SDPA masks。
        if self.precision not in ("fp32", "bf16") or self.backend not in ("eager", "sdpa"):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("training supports FP32/BF16 and eager/SDPA masks")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：AdamW implementation must be single or fused。
        if self.optimizer_impl not in ("single", "fused"):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("AdamW implementation must be single or fused")

    # ── FitConfig.make_optimizer ──
    # 职责：为独立训练工具配置选定优化器；vLLM native learner 自己构造 AdamW，不经过这里。
    # 输入提示：parameters：优化器管理的 FP32 master 参数列表，不是所有冻结 backbone 参数。
    def make_optimizer(self, parameters):
        return torch.optim.AdamW(parameters, lr=self.learning_rate, weight_decay=self.weight_decay,
                                  foreach=False, fused=self.optimizer_impl == "fused")

    # ── FitConfig.rate ──
    # 职责：计算独立训练器当前 step 的学习率调度值。
    def rate(self, step):
        if step < self.warmup_steps:
            return self.learning_rate * (step + 1) / self.warmup_steps
        fraction = (step - self.warmup_steps) / max(1, self.steps - self.warmup_steps - 1)
        multiplier = self.minimum_lr_ratio + (1 - self.minimum_lr_ratio) * (1 + math.cos(math.pi * fraction)) / 2
        return self.learning_rate * multiplier


# ── frozen_fingerprint ──
# 职责：为独立训练器计算冻结参数内容指纹；不是 native learner 的低开销进程内身份记录。
def frozen_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if ".attention.draft." in name:
            # 跳过本次迭代的后续步骤，进入下一项。
            continue
        digest.update((name + str(tuple(value.shape)) + str(value.dtype)).encode())
        # 执行：使张量布局连续，满足算子/通信输入要求；必要时会拷贝；断开旧 autograd 计算图，不自动复制底层存储。
        digest.update(memoryview(value.detach().contiguous().cpu().view(torch.uint8).numpy()))
    return digest.hexdigest()


# ── Trainer ──
# 职责：类型 `Trainer`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class Trainer:
    """Checkpoint boundaries coincide with complete AdamW updates."""

    # ── Trainer.__init__ ──
    # 职责：独立通用离线 Trainer 的初始化；Qwen vLLM 离线 RPC 入口不使用这个 Trainer。
    def __init__(self, model, data, config=FitConfig()):
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：matching data windows and complete draft blocks required。
        if (data.sequence_length != config.sequence_length or data.vocab_size != model.config.vocab_size
                or config.sequence_length < model.config.block_size):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("matching data windows and complete draft blocks required")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：training uses FP32 master parameters with optional BF16 autocast。
        if any(value.dtype != torch.float32 for value in model.parameters()):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("training uses FP32 master parameters with optional BF16 autocast")
        self.model = model.train_draft_only().set_backend(config.backend)
        self.config, self.data = config, data
        self.device = model.embedding.weight.device
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：BF16 training autocast requires CUDA or NPU。
        if config.precision == "bf16" and self.device.type not in ("cuda", "npu"):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("BF16 training autocast requires CUDA or NPU")
        # 数据：优化器管理的 FP32 master 参数列表，不是所有冻结 backbone 参数。
        self.parameters = [value for value in model.parameters() if value.requires_grad]
        self.optimizer = config.make_optimizer(self.parameters)
        self.stream = BatchStream(data, config.seed)
        self.anchors_rng = torch.Generator().manual_seed(config.seed + 1)
        self.step = 0
        self.base_fingerprint = frozen_fingerprint(model)

    # ── Trainer._autocast ──
    # 职责：选择独立 Trainer 的混合精度上下文。
    def _autocast(self):
        return (autocast(self.device, torch.bfloat16) if self.config.precision == "bf16"
                else nullcontext())

    # ── Trainer._runtime ──
    # 职责：收集独立 Trainer 的设备、随机状态等恢复信息。
    def _runtime(self):
        return {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                "device_type": self.device.type,
                "device_name": device_name(self.device),
                "threads": torch.get_num_threads(), "matmul_precision": torch.get_float32_matmul_precision(),
                "deterministic": torch.are_deterministic_algorithms_enabled()}

    # ── Trainer.run ──
    # 职责：独立 Trainer 的训练循环，区别于 train_qwen38_draft.py 的原生 worker RPC 循环。
    def run(self, until=None, progress=None):
        end = self.config.steps if until is None else until
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：stop boundary must lie within the unchanged full schedule。
        if type(end) is not int or not self.step <= end <= self.config.steps:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("stop boundary must lie within the unchanged full schedule")
        records = []
        self.model.train()
        # 循环：重复处理直到条件不再成立；留意循环中的 break 和状态推进。
        while self.step < end:
            synchronize(self.device)
            start = time.perf_counter()
            rate = self.config.rate(self.step)
            batches = []
            for _ in range(self.config.accumulate):
                tokens = self.stream.batch(self.config.batch_size)
                anchors = sample_anchors(tokens, self.model.config.block_size,
                                         self.config.anchors_per_sequence, generator=self.anchors_rng)
                # 执行：对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
                batches.append((tokens.to(self.device), anchors.to(self.device)))
            metrics = distillation_update(self.model, batches, self.optimizer, learning_rate=rate,
                                          clip_grad=self.config.clip_grad, chunk_rows=self.config.chunk_rows,
                                          autocast=self._autocast)
            # 状态推进：在原有值上累加/更新；若目标是张量，要注意这里的原地语义。
            self.step += 1
            synchronize(self.device)
            record = {"step": self.step, **metrics, "seconds": time.perf_counter() - start}
            # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
            records.append(record)
            if progress is not None:
                progress(record)
        return records

    @torch.no_grad()
    # ── Trainer.evaluate ──
    # 职责：在独立 Trainer 中对给定数据估计蒸馏 loss，不做参数更新。
    def evaluate(self, data, batches=4):
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：matching validation window and positive batch count required。
        if data.sequence_length != self.config.sequence_length or batches < 1:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("matching validation window and positive batch count required")
        stream = BatchStream(data, self.config.seed + 2)
        rng = torch.Generator().manual_seed(self.config.seed + 3)
        losses, training = [], self.model.training
        self.model.eval()
        # 保护区：执行可能失败的步骤；异常处理或 finally 负责恢复状态/释放资源。
        try:
            for _ in range(batches):
                tokens = stream.batch(self.config.batch_size)
                anchors = sample_anchors(tokens, self.model.config.block_size,
                                         self.config.anchors_per_sequence, generator=rng)
                with self._autocast():
                    # 数据：学生与教师概率分布的散度；必须有限才允许反向更新。 此处对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
                    loss = distillation_loss(self.model, tokens.to(self.device), anchors.to(self.device),
                                             chunk_rows=self.config.chunk_rows)
                # 执行：转为 FP32 做分数/归一化/梯度等中间计算。
                losses.append(float(loss))
        finally:
            self.model.train(training)
        return sum(losses) / len(losses)

    # ── Trainer.save ──
    # 职责：保存独立 Trainer 的模型、优化器、数据流和运行状态。
    def save(self, path):
        state = {"format": "dual-fit-v1", "config": asdict(self.config), "data_sha256": self.data.fingerprint,
                 "data_stream": self.stream.state_dict(), "anchors_rng": self.anchors_rng.get_state(),
                 "torch_rng": torch.get_rng_state(), "runtime": self._runtime(),
                 "accelerator_rng": rng_state(self.device),
                 "base_fingerprint": self.base_fingerprint}
        save_checkpoint(path, self.model, optimizer=self.optimizer, step=self.step, training_state=state)

    @classmethod
    # ── Trainer.resume ──
    # 职责：从独立 Trainer checkpoint 恢复模型、数据和训练进度；不是 native head 的 _load_head。
    def resume(cls, path, data, *, device="cpu"):
        model, saved = load_checkpoint(path, device=device)
        state = saved["training_state"]
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：a complete training checkpoint is required。
        if not isinstance(state, dict) or state.get("format") != "dual-fit-v1" or saved["optimizer"] is None:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("a complete training checkpoint is required")
        if not isinstance(data, TokenDataset):
            data = TokenDataset(data, model.config.vocab_size, state["config"]["sequence_length"])
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：resume training data SHA256 differs。
        if state["data_sha256"] != data.fingerprint:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("resume training data SHA256 differs")
        trainer = cls(model, data, FitConfig(**state["config"]))
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：resume requires the same runtime and frozen base。
        if trainer._runtime() != state["runtime"] or trainer.base_fingerprint != state["base_fingerprint"]:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("resume requires the same runtime and frozen base")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：invalid training step。
        if type(saved["step"]) is not int or not 0 <= saved["step"] <= trainer.config.steps:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("invalid training step")
        # 调用：校验保存状态与当前训练契约再恢复，避免错模型/错冻结边界续训。
        trainer.optimizer.load_state_dict(saved["optimizer"])
        # 调用：校验保存状态与当前训练契约再恢复，避免错模型/错冻结边界续训。
        trainer.stream.load_state_dict(state["data_stream"])
        trainer.anchors_rng.set_state(state["anchors_rng"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        saved_rng = state.get("accelerator_rng", state.get("cuda_rng"))
        # 设备选路：按 CPU/NPU 和当前形状选择可用实现，不能假定所有设备调用同一个算子。
        if trainer.device.type != "cpu":
            set_rng_state(saved_rng, trainer.device)
        trainer.step = saved["step"]
        return trainer
