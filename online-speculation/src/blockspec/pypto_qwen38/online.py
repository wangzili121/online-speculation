"""串行 last-1 在线重放：只更新 draft，AR 历史和共享 backbone 保持只读。"""

from collections import deque
from dataclasses import dataclass
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

    def __post_init__(self):
        if (self.last_layers != 1 or type(self.stride) is not int or self.stride < 1
                or type(self.replay_blocks) is not int or not 1 <= self.replay_blocks <= self.stride
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not math.isfinite(self.clip_norm) or self.clip_norm <= 0):
            raise ValueError("last-1, bounded replay and positive finite optimizer scales required")


class FeedbackBlock:
    def __init__(self, owner, inputs):
        self.owner, self.inputs = owner, inputs
        self.rows = self.valid = self.past = self.width = 0

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
        self.graph = LastLayerGraph()
        self.rounds = self.updates = 0
        self.last_result = None
        self.current = None
        self.replay, self.pool, self.handles = deque(), [], []
        self.workspace = self.optimizer = self.native = None

    def compile(self, samples, config):
        import torch
        self.config = config
        self.training = TrainingPrograms(self.graph, config)
        self.optimization = OptimizerPrograms(self.graph, config, clip_norm=self.options.clip_norm)
        meta = lambda shape, dtype=torch.bfloat16: torch.empty(shape, dtype=dtype, device="meta")
        self.history_program = make_history(samples[31].shape[2]).compile(samples[31], samples[32],
            meta((32,), torch.int32), meta((2048, 256)), meta((2048, 256)), 0, config=config)
        self.teacher_program = make_teacher().compile(meta((8, 16, 248320)), meta((32, 248320)), 0, 0, 16, config=config)
        self.rope_program = make_copy(32, 64).compile(meta((32, 64)), meta((32, 64)), config=config)
        self.loss_program = make_loss_sum().compile(meta((32, 8), torch.float32), meta((1, 8), torch.float32), config=config)
        return [*self.training.programs.values(), *self.optimization.programs(),
                self.history_program, self.teacher_program, self.rope_program, self.loss_program]

    def alloc(self, shape, dtype):
        value = self.native.worker.alloc_tensor(shape, dtype)
        self.handles.append(value)
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
        if self.options.raw_head_dir is not None:
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
            self.workspace = TrainingWorkspace(self.training, native.worker)
            self.optimizer = DeviceOptimizer(self.optimization, native.worker)
            self.loss_device = self.alloc((1, 8), torch.float32)
            self.loss_host = torch.empty((1, 8), dtype=torch.float32)
            if self.options.resume_dir is None:
                self.optimizer.initialize(self.head, raw_norms)
            else:
                restore(self.optimizer, draft.weights, self.options.resume_dir, self.head_identity)
                self.updates = self.optimizer.steps
                self.rounds = self.updates * self.options.stride
            for _ in range(self.options.replay_blocks):
                values = {}
                for name in ("hidden", "history_key", "history_value", "cos", "sin", "teacher"):
                    values[name] = self.alloc(self.graph.inputs[name].shape, torch.bfloat16)
                self.pool.append(FeedbackBlock(self, values))
            def view(bank, shape, offset, dtype=DataType.BFLOAT16):
                return bank.buffer.tensor(shape, dtype, byte_offset=offset)
            H, I = 5120, 17408
            self.shared = dict(input_gamma=view(native.banks[3], (1, H), 63*H*4, DataType.FLOAT32),
                post_gamma=view(native.banks[4], (1, H), 63*H*4, DataType.FLOAT32), final_gamma=native.banks[1],
                lm_head=native.banks[2], mlp_gate=view(native.banks[5], (I, H), 63*I*H*2),
                mlp_up=view(native.banks[6], (I, H), 63*I*H*2), mlp_down=view(native.banks[7], (H, I), 63*I*H*2))
        except BaseException:
            self.close()
            raise

    def begin(self, draft, past, width):
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
            return self.update()
        return None

    def update(self):
        import ctypes
        self.optimizer.zero_grad()
        positions = sum(item.valid for item in self.replay)
        loss = 0.
        for item in self.replay:
            inputs = {**self.shared, **item.inputs, **self.optimizer.tensors()}
            gradients = self.workspace.run(inputs, past=item.past, width=item.width,
                                            valid=item.valid, normalizer=positions)
            self.optimizer.accumulate(gradients)
            handle = self.workspace.values[self.graph.losses.name]
            self.run(self.loss_program, handle, self.loss_device)
            self.native.worker.copy_from(self.loss_host.data_ptr(), self.loss_device.data_ptr, 32)
            loss += ctypes.c_float.from_address(self.loss_host.data_ptr()).value
        if not math.isfinite(loss):
            raise FloatingPointError("nonfinite online loss; head not published")
        result = self.optimizer.step(self.draft.weights, self.options.learning_rate)
        self.updates += 1
        self.last_result = dict(loss=loss, reached_positions=positions, **result)
        if self.options.checkpoint_dir is not None:
            from pathlib import Path
            from .online_checkpoint import save
            directory = Path(self.options.checkpoint_dir)/f"step-{self.optimizer.steps:08d}"
            save(self.optimizer, directory, self.head_identity)
            self.last_result["checkpoint"] = str(directory)
        return self.last_result

    def abort(self):
        self.current = None
        self.replay.clear()

    def close(self):
        self.current = None
        self.replay.clear()
        self.pool.clear()
        if self.workspace is not None:
            self.workspace.close()
            self.workspace = None
        if self.optimizer is not None:
            self.optimizer.close()
            self.optimizer = None
        if self.native is not None:
            while self.handles:
                self.native.worker.free_tensor(self.handles[-1])
                self.handles.pop()
