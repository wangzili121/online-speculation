"""Bind the Serving forward outputs directly to the PyPTO distillation program."""

from .pypto_glm53_online_suffix import make_distillation_suffix
from .pypto_glm53_serving import _clean_replay, _draft_inputs, _teacher_inputs


ROWS = 16
PROGRAM = "orthrus_distillation"


def prepare_distillation(api, compiler):
    from models.glm5_3_flash.config import FLASH

    if not api.VERIFY_QUERY_WIDTH <= ROWS <= api.LOGIT_ROWS:
        raise ValueError("Serving replay rows do not match the GLM distillation program")
    entry = make_distillation_suffix(
        api.RANKS, ROWS, api.LOCAL_VOCAB, api.D, FLASH.rms_norm_eps, FLASH.hc_mult,
    )
    return compiler.compile(PROGRAM, entry)


class ResidentDistillation:
    """Own the replay banks and device gradients for one Serving runner."""

    def __init__(self, runner):
        self.runner = runner
        self.runtime = runner.runtime
        self.owned = []
        self.suffix_tape = None

    def allocate(self, shape, dtype):
        from pypto.runtime import StackedDeviceTensor

        ranks = tuple(range(self.runner._api.RANKS))
        shards = []
        for rank in ranks:
            shard = self.runtime.alloc_tensor(shape, dtype, worker_id=rank)
            self.owned.append((rank, shard))
            shards.append(shard)
        return StackedDeviceTensor(shards, (len(ranks), *shape), ranks)

    def first_rows(self, tensor):
        from pypto.runtime import StackedDeviceTensor

        shape = (ROWS, *tensor.shape[2:])
        shards = [
            self.runtime.view_tensor(shard, shape, worker_id=rank)
            for rank, shard in zip(tensor.worker_ids, tensor.shards, strict=True)
        ]
        return StackedDeviceTensor(shards, (len(shards), *shape), tensor.worker_ids)

    def __enter__(self):
        import torch

        runner = self.runner
        api = runner._api
        if PROGRAM not in runner.auxiliary_programs:
            raise RuntimeError("the resident distillation callable was not registered before opening Serving")
        with runner._state_lock:
            if self.owned or self.runtime is not runner.runtime:
                raise RuntimeError("resident distillation requires the current Serving worker")
            runner._drain_forwards()
            try:
                logits_shape = (api.LOGIT_ROWS, api.HEAD_VOCAB_PAD)
                hidden_shape = (api.LOGIT_ROWS, api.D)
                self.student = self.allocate(logits_shape, torch.float32)
                self.teacher = self.allocate(logits_shape, torch.float32)
                self.hidden = self.allocate(hidden_shape, torch.bfloat16)
                self.teacher_hidden = self.allocate(hidden_shape, torch.bfloat16)
                self.dlogit = self.allocate((ROWS, api.HEAD_VOCAB_PAD), torch.float32)
                self.local = self.allocate((ROWS, api.D), torch.float32)
                self.combined = self.allocate((ROWS, api.D), torch.float32)
                self.loss = self.allocate((ROWS, 8), torch.float32)
                self.gradient = self.allocate((ROWS * api.HC_MULT, api.D), torch.float32)
                if hasattr(api, "ORTHRUS_TAPE_ROWS"):
                    self.suffix_tape = self.allocate((api.ORTHRUS_TAPE_ROWS, api.HC_MULT, api.D), torch.bfloat16)
            except BaseException:
                self.close()
                raise
        return self

    def close(self):
        while self.owned:
            rank, handle = self.owned[-1]
            self.runtime.free_tensor(handle, worker_id=rank)
            self.owned.pop()
        self.suffix_tape = None

    def __exit__(self, exc_type, _error, _traceback):
        if exc_type is None or self.runner._dispatch_error is None:
            with self.runner._state_lock:
                self.close()

    def capture(self, inputs, phase, logits, hidden):
        from pypto_serving.model.glm53.task_args import forward_task_args

        runner = self.runner
        tensors = dict(runner._phase_tensors[0][phase], logits=logits, head_hidden=hidden)
        if phase == "orthrus" and self.suffix_tape is not None:
            inputs = dict(inputs, execution_mode=inputs["execution_mode"].new_full(
                (1,), runner._api.ORTHRUS_REPLAY_MODE,
            ))
            tensors["orthrus_suffix_tape"] = self.suffix_tape
        args = forward_task_args(runner._api, phase, tensors)
        try:
            pending = runner._dispatch_step(inputs, phase=phase, buffer_slot=0, task_args=args)
            pending.dispatch.wait()
            runner._pending_forwards.pop(pending.buffer_slot)
            runner.calls += 1
        except BaseException as error:
            runner._poison_forwards(error)
            raise
        finally:
            args.close()

    def step(self, tokens, anchor, slot, mask_id):
        """Produce device-resident loss and final-stream gradients for a clean block."""
        runner = self.runner
        if not self.owned:
            raise RuntimeError("resident distillation must be used inside its context")
        with _clean_replay(runner, tokens, anchor, slot):
            inputs = _draft_inputs(runner, slot, anchor, tokens[anchor], mask_id)
            self.capture(inputs, "orthrus", self.student, self.hidden)
            inputs = _teacher_inputs(runner, tokens, anchor, slot)
            self.capture(inputs, "verify", self.teacher, self.teacher_hidden)
            valid = runner._api.VERIFY_QUERY_WIDTH - 1
            epoch = runner._orthrus_distillation_epoch + 1
            try:
                runner._run_l3(
                    runner.auxiliary_programs[PROGRAM], self.first_rows(self.student), self.first_rows(self.teacher),
                    runner.stacked["lm_head_weight"], self.first_rows(self.hidden), runner.stacked["final_norm"],
                    self.dlogit, self.local, self.combined, self.loss, self.gradient,
                    valid, epoch, config=runner.config,
                )
            except BaseException as error:
                runner._poison_forwards(error)
                raise
            runner._orthrus_distillation_epoch = epoch
        return self.loss, self.gradient
