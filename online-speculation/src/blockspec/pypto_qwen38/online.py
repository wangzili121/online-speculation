"""串行 last-1 在线重放：只更新 draft，AR 历史和共享 backbone 保持只读。"""

from collections import deque
from dataclasses import dataclass, replace
import math

from .online_feedback_programs import make_copy, make_history, make_loss_sum, make_teacher
from .online_optimizer import DeviceOptimizer, OptimizerPrograms
from .training_graph import LastLayerGraph
from .training_executor import TrainingPrograms, TrainingWorkspace


@dataclass(frozen=True)
class OnlineConfig:
    last_layers: int = 1
    stride: int = 16
    replay_blocks: int = 2
    learning_rate: float = 1e-5
    clip_norm: float = 1.
    raw_head_dir: str | None = None
    checkpoint_dir: str | None = None
    resume_dir: str | None = None
    learner_device: int | None = None
    objective: str = "tv"
    reset_optimizer: bool = False

    def __post_init__(self):
        if type(self.reset_optimizer) is not bool or (self.reset_optimizer and self.resume_dir is None):
            raise ValueError("reset_optimizer requires a checkpoint initialization")
        if self.objective not in {"tv", "kl"}:
            raise ValueError("objective must be tv or kl")
        if (type(self.last_layers) is not int or not 1 <= self.last_layers <= 64
                or type(self.stride) is not int or self.stride < 1
                or type(self.replay_blocks) is not int or not 1 <= self.replay_blocks <= self.stride
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not math.isfinite(self.clip_norm) or self.clip_norm <= 0):
            raise ValueError("last-1..64, bounded replay and positive finite optimizer scales required")
        if self.learner_device is not None and (type(self.learner_device) is not int or self.learner_device < 0):
            raise ValueError("learner device must be a nonnegative visible device")
        if self.last_layers > 1 and self.learner_device is None:
            raise ValueError("multi-layer updates require a separate learner device")


class FeedbackBlock:
    def __init__(self, owner, inputs):
        self.owner, self.inputs = owner, inputs
        self.training_inputs = inputs
        self.rows = self.valid = self.past = self.width = 0

    def transfer(self):
        """只传受监督行的教师分布；其余激活保持完整，避免尾部影响反向。"""
        owner = self.owner
        for name, value in self.inputs.items():
            kwargs = ({"nbytes": self.valid * value.shape[1] * value.dtype.itemsize}
                      if name == "teacher" else {})
            owner.bridge.copy(owner.native.worker, value, owner.trainer,
                              self.training_inputs[name], **kwargs)

    def teacher(self, native, slot, count):
        if self.rows >= self.width:
            return  # verifier 的回滚重放不再次作为教师数据。
        if self.rows + count > self.width:
            raise RuntimeError("verifier teacher rows exceed the proposed block")
        self.owner.run(self.owner.teacher_program, native.logits, self.inputs["teacher"],
                       slot * 16, self.rows, count)
        self.rows += count


class OnlineLearner:
    def __init__(self, head, options):
        self.head, self.options = head, options
        self.graph = LastLayerGraph(objective=options.objective)
        self.rounds = self.updates = 0
        self.last_result = None
        self.current = None
        self.replay, self.pool, self.handles = deque(), [], []
        self.workspace = self.optimizer = self.native = None
        self.trainer = None
        self.training_handles = []

    def compile(self, samples, config):
        import torch
        self.config = config
        self.dedicated = self.options.learner_device is not None
        if self.dedicated and self.options.learner_device in config.distributed_config.device_ids:
            raise ValueError("dedicated learner must use a different visible device")
        self.training_config = (replace(config, distributed_config=replace(config.distributed_config,
                                  device_ids=[self.options.learner_device])) if self.dedicated else config)
        self.training = TrainingPrograms(self.graph, self.training_config, combined=True, accumulate_gradients=True)
        self.optimization = OptimizerPrograms(self.graph, self.training_config, clip_norm=self.options.clip_norm, combined=True)
        meta = lambda shape, dtype=torch.bfloat16: torch.empty(shape, dtype=dtype, device="meta")
        self.history_program = make_history(samples[31].shape[2]).compile(samples[31], samples[32],
            meta((32,), torch.int32), meta((2048, 256)), meta((2048, 256)), 0, config=config)
        self.teacher_program = make_teacher().compile(meta((8, 16, 248320)), meta((32, 248320)), 0, 0, 16, config=config)
        self.rope_program = make_copy(32, 64).compile(meta((32, 64)), meta((32, 64)), config=config)
        self.loss_program = make_loss_sum().compile(meta((32, 8), torch.float32), meta((1, 8), torch.float32), config=self.training_config)
        self.training_programs = [*self.training.programs.values(), *self.optimization.programs(), self.loss_program]
        if self.dedicated:
            from .online_bridge import TensorBridge
            self.bridge = TensorBridge()
        return [*(() if self.dedicated else self.training_programs),
                self.history_program, self.teacher_program, self.rope_program]

    def alloc(self, shape, dtype, *, training=False):
        separate = training and self.dedicated
        worker = self.trainer if separate else self.native.worker
        value = worker.alloc_tensor(shape, dtype)
        (self.training_handles if separate else self.handles).append(value)
        return value

    def run(self, program, *args):
        self.native.worker.run(program, *args, config=self.config)

    def start(self, native, draft):
        import torch
        from pypto.runtime.task_interface import DataType
        from .head import HeadSource
        from .head_loading import raw_location
        from safetensors import safe_open
        self.native, self.draft = native, draft
        from .online_checkpoint import identity, restore
        self.head_identity = identity(self.head)
        raw = self.head
        if self.options.raw_head_dir is not None and self.options.resume_dir is None:
            raw = HeadSource(self.head.model_root, self.options.raw_head_dir,
                             block_size=self.head.block_size, mask_token_id=self.head.mask_token_id)
        if raw.prepared_manifest is not None and self.options.resume_dir is None:
            raise ValueError("online initialization needs raw BF16 norms: pass --learner-head-dir")
        raw_norms = {}
        if self.options.resume_dir is None:
            for name, tensor in (("q_gamma", "q_norm.weight"), ("k_gamma", "k_norm.weight")):
                path, key = raw_location(raw, 63, tensor, 0)
                with safe_open(path, framework="pt", device="cpu") as file:
                    raw_norms[name] = file.get_tensor(key)
        try:
            if self.dedicated:
                from pypto.runtime import DistributedWorker
                from .online_bridge import HeadPublisher
                self.trainer = DistributedWorker(self.training_programs, config=self.training_config,
                    inherited_host_tensors=[self.bridge.scratch])
            else:
                self.trainer = native.worker
            print("online training free bytes before workspace", self.trainer.device_memory_info()[0], flush=True)
            self.workspace = TrainingWorkspace(self.training, self.trainer)
            self.optimizer = DeviceOptimizer(self.optimization, self.trainer)
            self.loss_device = self.alloc((1, 8), torch.float32, training=True)
            self.loss_host = torch.empty((1, 8), dtype=torch.float32)
            self.publisher = draft.weights
            if self.dedicated:
                self.serving_banks = [tuple(self.alloc(value.shape, torch.float32 if "gamma" in name else torch.bfloat16)
                    for name, value in ((name, self.graph.inputs[name]) for name in
                    ("q_weight", "k_weight", "v_weight", "q_gamma", "k_gamma", "o_weight"))) for _ in range(2)]
                self.publisher = HeadPublisher(self, draft.weights)
            if self.options.resume_dir is None:
                self.optimizer.initialize(self.head, raw_norms,
                    device_banks=None if self.dedicated else draft.weights.for_group(60)[3])
            else:
                restore(self.optimizer, self.publisher, self.options.resume_dir, self.head_identity,
                        reset_optimizer=self.options.reset_optimizer)
                self.updates = self.optimizer.steps
                self.rounds = self.updates * self.options.stride
            print("online training free bytes after optimizer", self.trainer.device_memory_info()[0], flush=True)
            for _ in range(self.options.replay_blocks):
                values = {}
                for name in ("hidden", "history_key", "history_value", "cos", "sin", "teacher"):
                    values[name] = self.alloc(self.graph.inputs[name].shape, torch.bfloat16)
                block = FeedbackBlock(self, values)
                if self.dedicated:
                    block.training_inputs = {name: self.alloc(value.shape, value.dtype, training=True) for name, value in values.items()}
                self.pool.append(block)
            def view(bank, shape, offset, dtype=DataType.BFLOAT16):
                return bank.buffer.tensor(shape, dtype, byte_offset=offset)
            H, I = 5120, 17408
            self.shared = dict(input_gamma=view(native.banks[3], (1, H), 63*H*4, DataType.FLOAT32),
                post_gamma=view(native.banks[4], (1, H), 63*H*4, DataType.FLOAT32), final_gamma=native.banks[1],
                lm_head=native.banks[2], mlp_gate=view(native.banks[5], (I, H), 63*I*H*2),
                mlp_up=view(native.banks[6], (I, H), 63*I*H*2), mlp_down=view(native.banks[7], (H, I), 63*I*H*2))
            if self.dedicated:
                offsets = dict(input_gamma=(3, 63*H*4), post_gamma=(4, 63*H*4), final_gamma=(1, 0),
                    lm_head=(2, 0), mlp_gate=(5, 63*I*H*2), mlp_up=(6, 63*I*H*2), mlp_down=(7, 63*I*H*2))
                for name, (bank, offset) in offsets.items():
                    # view 的 shapes/DataType 与 worker 句柄接口不同；以训练图契约分配。
                    spec = self.graph.inputs[name]
                    target = self.alloc(spec.shape, torch.float32 if spec.dtype == "fp32" else torch.bfloat16,
                                        training=True)
                    self.bridge.copy(native.worker, native.banks[bank], self.trainer, target, source_offset=offset)
                    self.shared[name] = target
            print("online learner ready", flush=True)
        except BaseException:
            try:
                self.close()
            except BaseException:
                import logging
                logging.getLogger(__name__).exception("online learner cleanup failed after startup error")
            raise

    def begin(self, draft, past, width, *, slot=None):
        if self.current is not None:
            raise RuntimeError("previous online feedback has not been committed or aborted")
        if self.options.stride - self.rounds % self.options.stride > self.options.replay_blocks:
            return None
        used = {id(item) for item in self.replay}
        block = next((item for item in self.pool if id(item) not in used), None)
        if block is None:
            block = self.replay.popleft()
        block.past, block.width, block.rows, block.valid = past, width, 0, 0
        self.current = block
        self.run(self.history_program, *self.native.resident[2:], draft.table,
                 block.inputs["history_key"], block.inputs["history_value"], past)
        self.run(self.rope_program, draft.cos, block.inputs["cos"])
        self.run(self.rope_program, draft.sin, block.inputs["sin"])
        return block

    def observe(self, result):
        block, self.current = self.current, None
        self.rounds += 1
        if block is not None:
            if block.rows != block.width:
                raise RuntimeError("incomplete teacher distribution capture")
            block.valid = min(result.accepted + 1, block.width - 1, len(result.tokens))
            self.replay.append(block)
        if self.rounds % self.options.stride == 0 and self.replay:
            try:
                return self.update()
            except FloatingPointError as error:
                # 数值不合法的候选没有发布；继续交付本轮已经验证过的 token。
                self.replay.clear()
                self.last_result = dict(skipped=True, reason=str(error), version=self.draft.weights.version)
                return self.last_result
        return None

    def update(self):
        import ctypes
        self.optimizer.zero_grad()
        positions = sum(item.valid for item in self.replay)
        loss = 0.
        for item in self.replay:
            if self.dedicated:
                item.transfer()
            inputs = {**self.shared, **item.training_inputs, **self.optimizer.tensors()}
            self.workspace.run(inputs, past=item.past, width=item.width,
                               valid=item.valid, normalizer=positions, gradients=self.optimizer.accum)
            handle = self.workspace.values[self.graph.losses.name]
            self.trainer.run(self.loss_program, handle, self.loss_device, config=self.training_config)
            self.trainer.copy_from(self.loss_host.data_ptr(), self.loss_device.data_ptr, 32)
            loss += ctypes.c_float.from_address(self.loss_host.data_ptr()).value
        if not math.isfinite(loss):
            raise FloatingPointError("nonfinite online loss; head not published")
        result = self.optimizer.step(self.publisher, self.options.learning_rate)
        self.updates += 1
        self.last_result = dict(loss=loss, reached_positions=positions, **result)
        if self.options.checkpoint_dir is not None:
            from pathlib import Path
            from .online_checkpoint import save
            directory = Path(self.options.checkpoint_dir)/f"step-{self.optimizer.steps:08d}"
            save(self.optimizer, directory, self.head_identity)
            self.last_result["checkpoint"] = str(directory)
        return self.last_result

    def observe_clean_block(self):
        """仅供干净序列重放：教师已用真实输入计算整块，与投机接受长度无关。"""
        block, self.current = self.current, None
        if block is None or block.rows != block.width:
            raise RuntimeError("complete clean teacher block required")
        block.valid = block.width - 1
        self.replay.append(block)
        self.rounds += 1
        if self.rounds % self.options.stride == 0:
            return self.update()
        return None

    def abort(self):
        self.current = None
        self.replay.clear()

    def close(self):
        self.current = None
        self.replay.clear()
        self.pool.clear()
        try:
            if self.trainer is not None and getattr(self, "dedicated", False):
                # 独立 worker 拥有全部训练分配；整棵运行时关闭也覆盖设备故障后的清理。
                self.trainer.close()
                self.trainer = None
                self.workspace = self.optimizer = None
                self.training_handles.clear()
            else:
                if self.workspace is not None:
                    self.workspace.close()
                    self.workspace = None
                if self.optimizer is not None:
                    self.optimizer.close()
                    self.optimizer = None
        finally:
            if self.native is not None:
                while self.handles:
                    self.native.worker.free_tensor(self.handles[-1])
                    self.handles.pop()
