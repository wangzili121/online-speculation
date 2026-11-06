# 中文学习版｜完整阅读路线见 docs/QWEN38_CODE_WALKTHROUGH.md
# 文件职责：通用后缀训练器：选中 last-N draft 参数，维护 FP32 master / AdamW / replay，重放反向并发布更新。离线也复用它。
# 约定：B=batch，K=块长（含 anchor），H=hidden_size，V=词表大小；通常 B=1。
# 注释不改变执行代码；续行与同一逻辑语句一起阅读。

"""Online suffix distillation with FP32 masters and read-only AR history."""

# 依赖：从 collections 导入 deque；供本模块调用。
from collections import deque
# 依赖：从 dataclasses 导入 asdict, dataclass；供本模块调用。
from dataclasses import asdict, dataclass
# 依赖：导入 copy。
import copy
# 依赖：导入 hashlib。
import hashlib
# 依赖：导入 math。
import math
# 依赖：导入 time。
import time

# 依赖：导入 torch。
import torch

# 依赖：从 ..accelerator 导入 synchronize；供本模块调用。
from ..accelerator import synchronize
# 依赖：从 ..losses 导入 divergence；供本模块调用。
from ..losses import divergence
# 依赖：从 .backbone 导入 DraftBoundary；供本模块调用。
from .backbone import DraftBoundary
# 依赖：从 .weights 导入 source_identity；供本模块调用。
from .weights import source_identity


@dataclass(frozen=True)
# ── SuffixConfig ──
# 职责：类型 `SuffixConfig`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class SuffixConfig:
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    last_layers: int = 1
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    stride: int = 16
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    replay_blocks: int = 1
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    learning_rate: float = 1e-5
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    clip_norm: float = 1.
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    loss: str = "forward_kl"
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    optimizer_device: str = "model"
    # 字段：显式标注数据类型；字段含义与所在数据结构的函数说明一起阅读。
    checkpoint_layers: bool = False

    # ── SuffixConfig.__post_init__ ──
    # 职责：数据类构造后校验配置，拒绝无法支持的参数组合。
    def __post_init__(self):
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：positive bounded suffix/update windows, optimizer scales and KL or TV required。
        if (any(type(v) is not int or v < 1 for v in (self.last_layers, self.stride, self.replay_blocks))
                or self.replay_blocks > self.stride
                or any(not math.isfinite(v) or v <= 0 for v in (self.learning_rate, self.clip_norm))
                or self.loss not in ("forward_kl", "tv")
                or self.optimizer_device not in ("model", "cpu")
                or type(self.checkpoint_layers) is not bool):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("positive bounded suffix/update windows, optimizer scales and KL or TV required")


# ── portable ──
# 职责：递归复制训练状态到 CPU 以便保存；这是 checkpoint 序列化，不等于训练时必须 CPU offload。
# 输入提示：value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
def portable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(portable(item) for item in value)
    return copy.deepcopy(value)


# ── SuffixLearner ──
# 职责：类型 `SuffixLearner`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class SuffixLearner:
    """Only draft attention in a suffix is published; AR parameters stay frozen.

    Each feedback block stores a detached boundary, clean suffix KV and the
    actual reached teacher rows. The whole masked block is replayed, followed
    by the output projection of its first `valid` rows.
    """

    # ── SuffixLearner.__init__ ──
    # 职责：仅选最后 N 层 .attention.draft. 参数，复制 FP32 master 并创建 AdamW；执行参数保持冻结，训练以函数式替换注入 master。
    def __init__(self, model, config=SuffixConfig()):
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：a frozen dual-view model and a suffix within its layers required。
        if config.last_layers > len(model.layers) or any(p.requires_grad for p in model.parameters()):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("a frozen dual-view model and a suffix within its layers required")
        self.model, self.config = model, config
        self.capture_layer = len(model.layers) - config.last_layers
        # 数据：参与训练的 draft 执行参数映射；publish 的写入目标，通常 BF16。
        self.execution = {name: p for name, p in model.named_parameters()
                          if name.startswith("layers.") and int(name.split(".")[1]) >= self.capture_layer
                          and ".attention.draft." in name}
        dtype = next(model.parameters()).dtype
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：FP32 or BF16 inference weights required。
        if dtype not in (torch.float32, torch.bfloat16):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("FP32 or BF16 inference weights required")
        # 数据：用于训练的 FP32 参数副本；优化器更新它，再 publish 到执行参数。 此处对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
        self.master = {name: torch.nn.Parameter(p.detach().to(
            device="cpu" if config.optimizer_device == "cpu" else p.device, dtype=torch.float32, copy=True))
            for name, p in self.execution.items()}
        # 数据：优化器管理的 FP32 master 参数列表，不是所有冻结 backbone 参数。
        self.parameters = list(self.master.values())
        self.optimizer = torch.optim.AdamW(self.parameters, lr=config.learning_rate, weight_decay=0,
                                           fused=True if self.parameters[0].is_cuda else None)
        self._frozen = [(p, self.parameter_stamp(p)) for name, p in model.named_parameters() if name not in self.execution]
        self.frozen_fingerprint = self.fingerprint()
        self._backend = model.backend
        self.replay = deque(maxlen=config.replay_blocks)
        self.rounds = self.updates = self.version = self.feedback_blocks = self.coverage_skips = 0
        self.update_seconds, self.last_loss = 0., None

    @staticmethod
    # ── SuffixLearner.parameter_stamp ──
    # 职责：记录参数版本/身份供冻结保护；NativeSuffixLearner 覆盖以兼容 inference tensor。
    def parameter_stamp(parameter):
        return parameter._version

    # ── SuffixLearner.fingerprint ──
    # 职责：建立冻结部分身份；native 版本使用进程内参数身份，验证脚本的完整内容哈希是另一种更强的审计。
    def fingerprint(self):
        fingerprint = hashlib.sha256()
        for name, p in self.model.named_parameters():
            if name not in self.execution:
                fingerprint.update(name.encode())
                # 执行：使张量布局连续，满足算子/通信输入要求；必要时会拷贝；断开旧 autograd 计算图，不自动复制底层存储。
                fingerprint.update(p.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
        return fingerprint.hexdigest()

    @property
    # ── SuffixLearner.trainable_parameters ──
    # 职责：求选中 FP32 master 参数元素数；不是整个基模参数量。
    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters)

    @property
    # ── SuffixLearner.needs_decoder_feedback ──
    # 职责：判断当前 stride 是否需要保存训练 boundary；无需更新的轮次避免额外捕获。
    def needs_decoder_feedback(self):
        return self.config.stride - self.rounds % self.config.stride <= self.config.replay_blocks

    # ── SuffixLearner._check ──
    # 职责：检查冻结参数/后端约定没有被外部改变，防止重放所基于的表示失效。
    def _check(self):
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：frozen representations or attention execution changed; rebuild the learner。
        if (self.model.backend != self._backend or any(p.requires_grad for p in self.model.parameters())
                or any(self.parameter_stamp(p) != version for p, version in self._frozen)):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise RuntimeError("frozen representations or attention execution changed; rebuild the learner")

    # ── SuffixLearner._sync ──
    # 职责：同步执行参数所在设备，用于正确计时和更新边界；不是 TP 求和。
    def _sync(self):
        for device in {p.device for p in self.execution.values()}:
            synchronize(device)

    # ── SuffixLearner.clear_replay ──
    # 职责：清除旧请求/旧步骤保存的反馈，避免跨请求训练材料混用或不必要显存占用。
    def clear_replay(self):
        self.replay.clear()

    # ── SuffixLearner._skip_decoder_feedback ──
    # 职责：记录未捕获 boundary 的正常轮次，同时验证它确实不在必须训练的窗口。
    def _skip_decoder_feedback(self, valid):
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：positive decoder feedback outside the replay window required。
        if type(valid) is not int or valid < 1 or self.needs_decoder_feedback:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("positive decoder feedback outside the replay window required")
        # 状态推进：在原有值上累加/更新；若目标是张量，要注意这里的原地语义。
        self.rounds += 1

    # ── SuffixLearner.observe ──
    # 职责：校验 Feedback、detach 后写入有界 replay；到 stride 且允许更新时调用 update。
    # 输入提示：feedback：训练样本包：inputs/history/teacher_logits/valid/boundary，之后重放不依赖旧计算图。
    # 调用者：在线 proposer.observe_verified 或离线 worker。
    # 输入：Feedback 包，teacher_logits [valid,V] 与 boundary 对齐。
    # 输出：未达到更新间隔时 None；执行更新时返回 update 的指标字典。
    def observe(self, feedback, *, may_update=True):
        # 调用：检查冻结参数/后端约定没有被外部改变，防止重放所基于的表示失效。
        self._check()
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：aligned masked-block boundary and reached teacher rows required。
        if (feedback.inputs.ndim != 2 or feedback.inputs.shape[0] != 1
                or not 0 < feedback.valid < feedback.inputs.shape[1]
                or feedback.teacher_logits.shape != (feedback.valid, self.model.config.vocab_size)
                or not isinstance(feedback.boundary, DraftBoundary)
                or feedback.boundary.start_layer != self.capture_layer
                or feedback.boundary.hidden.shape[:2] != feedback.inputs.shape):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("aligned masked-block boundary and reached teacher rows required")
        # 状态推进：在原有值上累加/更新；若目标是张量，要注意这里的原地语义。
        self.rounds += 1
        # 状态推进：在原有值上累加/更新；若目标是张量，要注意这里的原地语义。
        self.feedback_blocks += 1
        # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
        self.replay.append(feedback.detached(cache_start=self.capture_layer))
        if may_update and self.rounds % self.config.stride == 0:
            return self.update()
        return None

    # ── SuffixLearner.replay_logits ──
    # 职责：用保存的 boundary/history 重算学生后缀，选取有效预测行；native 可覆盖为词表分片实现。
    def replay_logits(self, item):
        options = {"checkpoint_layers": True} if self.config.checkpoint_layers else {}
        return self.model.forward_suffix(item.boundary, cache=item.cache, logit_range=(0, item.valid),
                                         draft_weights=self.master, **options).logits[0]

    # ── SuffixLearner.replay_divergence ──
    # 职责：按配置比较 student 与 teacher 概率；不是与 token ID 做均方差。
    # 输入提示：student：需要梯度的 draft logits（可为本 rank 的词表分片）；teacher：冻结主模型的监督 logits；它不承担参数更新。
    def replay_divergence(self, student, teacher):
        return divergence(student, teacher, self.config.loss)

    # ── SuffixLearner.backward ──
    # 职责：清梯度，开启 autograd，逐个反馈重放后缀，按有效位置归一化 loss 并反向到 master。
    # 输出：本次 replay 的标量 loss；副作用：master 的 .grad 得到梯度。
    # 权重虽然冻结，后缀共享 MLP/norm/head 仍需输入梯度；不能在整个后缀套 no_grad。
    def backward(self):
        """Full-soft-target gradients through a frozen-prefix suffix replay."""
        # 数据：每个输入位置的绝对 token 位置，包含历史长度偏移，用于 RoPE。
        total, positions = 0., sum(item.valid for item in self.replay)
        # 执行：清除上次梯度，避免误累积。
        self.optimizer.zero_grad(set_to_none=True)
        # 梯度上下文：torch.enable_grad()；此块是否保存计算图由该设置决定。
        with torch.enable_grad():
            for item in self.replay:
                # CPU masters are staged inside each layer, not copied as a second full head.
                # 数据：词表未归一化分数；最后一维 V，起草有效行通常是前 K−1 行。
                logits = self.replay_logits(item)
                # 数据：学生与教师概率分布的散度；必须有限才允许反向更新。
                loss = self.replay_divergence(logits, item.teacher_logits).sum() / positions
                if not torch.isfinite(loss):
                    student_ok = bool(torch.isfinite(logits).all())
                    teacher_ok = bool(torch.isfinite(item.teacher_logits).all())
                    teacher_nan = int(item.teacher_logits.isnan().sum())
                    # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
                    raise FloatingPointError(f"nonfinite online suffix loss: student_finite={student_ok}, "
                                             f"teacher_finite={teacher_ok}, teacher_nan={teacher_nan}")
                # 执行：从 loss 建立的图反向累积梯度。
                loss.backward()
                # 状态推进：在原有值上累加/更新；若目标是张量，要注意这里的原地语义。
                total += float(loss.detach())
        return total

    @torch.no_grad()
    # ── SuffixLearner.publish ──
    # 职责：将 FP32 master 更新值复制回 draft 执行参数（通常 BF16），让下一次起草生效；不修改 AR 参数。
    # 调用者：update 在 AdamW.step 之后执行。
    # 写入方向：FP32 master → draft 执行参数；通常转换成 BF16，不写入主模型 AR 权重。
    def publish(self):
        for name, master in self.master.items():
            # 执行：把值写入既有目标存储；是否可写需看目标是 draft、master 还是工作 cache。
            self.execution[name].copy_(master)

    # ── SuffixLearner.update ──
    # 职责：串联 backward→梯度裁剪→AdamW.step→publish；记录版本/耗时/loss。
    # 输出：loss、gradient_norm、seconds、version、gradient_layers；无 replay 时返回 None。
    # 注意：gradient_layers 说明哪些参数实际有梯度，不代表它们对业务接受率已经有效。
    def update(self):
        if not self.replay:
            return None
        # 调用：检查冻结参数/后端约定没有被外部改变，防止重放所基于的表示失效。
        self._check()
        # 调用：同步执行参数所在设备，用于正确计时和更新边界；不是 TP 求和。
        self._sync()
        start = time.perf_counter()
        # 数据：学生与教师概率分布的散度；必须有限才允许反向更新。 此处从 loss 建立的图反向累积梯度。
        loss = self.backward()
        # 数据：本次确实获得梯度的 draft 层索引，供全层训练覆盖检查。
        gradient_layers = sorted({int(name.split(".")[1]) for name, value in self.master.items()
                                  if value.grad is not None})
        norm = self.clip_gradients()
        # 执行：执行优化器参数更新。
        self.optimizer.step()
        # 执行：把更新后的 FP32 master 复制回 draft 执行权重。
        self.publish()
        # 调用：同步执行参数所在设备，用于正确计时和更新边界；不是 TP 求和。
        self._sync()
        elapsed = time.perf_counter() - start
        # 状态推进：在原有值上累加/更新；若目标是张量，要注意这里的原地语义。
        self.updates += 1
        # 状态推进：在原有值上累加/更新；若目标是张量，要注意这里的原地语义。
        self.version += 1
        # 状态推进：在原有值上累加/更新；若目标是张量，要注意这里的原地语义。
        self.update_seconds += elapsed
        self.last_loss = loss
        return {"loss": loss, "gradient_norm": float(norm), "seconds": elapsed, "version": self.version,
                "gradient_layers": gradient_layers}

    # ── SuffixLearner.clip_gradients ──
    # 职责：计算全局梯度范数并按阈值缩放；TP 版本排除复制参数的重复计数后跨 rank 归并。
    def clip_gradients(self):
        return torch.nn.utils.clip_grad_norm_(self.parameters, self.config.clip_norm, error_if_nonfinite=True)

    # ── SuffixLearner.state_dict ──
    # 职责：导出 learner 的配置、版本、优化器和参数等恢复状态；不是重新执行模型。
    def state_dict(self):
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：save online state at a request boundary after clearing replay。
        if self.replay:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("save online state at a request boundary after clearing replay")
        return {"format": "blockspec-dual-suffix-v1", "config": asdict(self.config),
                "model_config": self.model.config.to_dict(), "source": source_identity(getattr(self.model, "source", {})),
                "dtype": str(next(self.model.parameters()).dtype), "backend": self._backend,
                "frozen_fingerprint": self.frozen_fingerprint,
                "master": portable(self.master), "optimizer": portable(self.optimizer.state_dict()),
                "rounds": self.rounds, "updates": self.updates, "version": self.version,
                "feedback_blocks": self.feedback_blocks, "update_seconds": self.update_seconds,
                "last_loss": self.last_loss}

    # ── SuffixLearner.load_state_dict ──
    # 职责：校验保存状态与当前训练契约再恢复，避免错模型/错冻结边界续训。
    def load_state_dict(self, state):
        # 调用：检查冻结参数/后端约定没有被外部改变，防止重放所基于的表示失效。
        self._check()
        saved_config = {"optimizer_device": "model", "checkpoint_layers": False, **state.get("config", {})}
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：online state must match its model, source, precision and update policy。
        if (state.get("format") != "blockspec-dual-suffix-v1" or saved_config != asdict(self.config)
                or state.get("model_config") != self.model.config.to_dict()
                or source_identity(state.get("source", {})) != source_identity(getattr(self.model, "source", {}))
                or state.get("dtype") != str(next(self.model.parameters()).dtype)
                or state.get("frozen_fingerprint") != self.frozen_fingerprint
                or state.get("backend") != self._backend or state.get("master", {}).keys() != self.master.keys()):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("online state must match its model, source, precision and update policy")
        for name, value in state["master"].items():
            # 约束检查：下方条件表示不支持的输入/状态；触发时报错：finite FP32 suffix master tensors with matching shapes required。
            if value.shape != self.master[name].shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
                raise ValueError("finite FP32 suffix master tensors with matching shapes required")
        for key in ("rounds", "updates", "version", "feedback_blocks"):
            # 约束检查：下方条件表示不支持的输入/状态；触发时报错：nonnegative integer online counters required。
            if type(state.get(key)) is not int or state[key] < 0:
                # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
                raise ValueError("nonnegative integer online counters required")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：finite nonnegative update time required。
        if not math.isfinite(state.get("update_seconds", -1)) or state["update_seconds"] < 0:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("finite nonnegative update time required")
        # 梯度上下文：torch.no_grad()；此块是否保存计算图由该设置决定。
        with torch.no_grad():
            for name, value in state["master"].items():
                # 执行：把值写入既有目标存储；是否可写需看目标是 draft、master 还是工作 cache。
                self.master[name].copy_(value)
        # 调用：校验保存状态与当前训练契约再恢复，避免错模型/错冻结边界续训。
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        # 执行：把更新后的 FP32 master 复制回 draft 执行权重。
        self.publish()
        # 调用：清除旧请求/旧步骤保存的反馈，避免跨请求训练材料混用或不必要显存占用。
        self.clear_replay()
        for key in ("rounds", "updates", "version", "feedback_blocks", "update_seconds", "last_loss"):
            setattr(self, key, state[key])
