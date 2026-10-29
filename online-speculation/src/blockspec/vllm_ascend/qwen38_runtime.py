"""Scoped compatibility hooks for the v0.26 Ascend eager runner.

This is an internal runner adapter, not a claim that upstream custom_class
already exposes neural-model lifecycle and post-verification feedback hooks.
Other proposers keep the original runner behavior.
"""

import functools
import inspect


def _ours(runner):
    return getattr(getattr(runner, "drafter", None), "qwen38_native_proposer", False)


def install_runner_hooks(runner_class=None):
    if runner_class is None:
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
        runner_class = NPUModelRunner
    if getattr(runner_class, "_qwen38_orthrus_hooks", False):
        return
    expected = {
        "_set_up_drafter": {"self"},
        "_get_drafter": {"self"},
        "_prepare_inputs": {"self", "scheduler_output"},
        "propose_draft_token_ids": {"self", "valid_sampled_token_ids", "scheduler_output"},
        "_sample": {"self", "logits", "spec_decode_metadata"},
    }
    for name, required in expected.items():
        method = getattr(runner_class, name, None)
        if method is None or not required <= inspect.signature(method).parameters.keys():
            raise RuntimeError(f"unsupported Ascend runner interface: {name}; expected v0.26 eager runner")
    setup = runner_class._set_up_drafter
    get_drafter = runner_class._get_drafter
    propose = runner_class.propose_draft_token_ids
    sample = runner_class._sample
    prepare_inputs = runner_class._prepare_inputs
    propose_signature = inspect.signature(propose)

    @functools.wraps(setup)
    def retain(self):
        custom = getattr(self, "drafter", None)
        if getattr(custom, "qwen38_native_proposer", False):
            self._qwen38_custom = custom
        setup(self)
        if _ours(self):
            self.drafter.bind_runner(self)

    @functools.wraps(get_drafter)
    def create(self):
        custom = getattr(self, "_qwen38_custom", None)
        if custom is not None:
            return custom
        return get_drafter(self)

    @functools.wraps(propose)
    def draft(self, *args, **kwargs):
        if not _ours(self):
            return propose(self, *args, **kwargs)
        arguments = propose_signature.bind(self, *args, **kwargs).arguments
        return self.drafter.propose_batch(arguments["valid_sampled_token_ids"], arguments["scheduler_output"])

    @functools.wraps(prepare_inputs)
    def prepare(self, *args, **kwargs):
        result = prepare_inputs(self, *args, **kwargs)
        if _ours(self) and result[1] is None:
            # 无候选的 prefill 不能沿用 profiling/上一轮的 GDN 投机计数。
            self.num_decode_draft_tokens.np.fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()
        return result

    @functools.wraps(sample)
    def verify(self, logits, spec_decode_metadata):
        if not _ours(self) or spec_decode_metadata is None:
            return sample(self, logits, spec_decode_metadata)
        # 保留原生 rejection sampler；在采样约束修改 logits 前取出教师分布。
        proposer = self.drafter
        teacher = proposer.teacher_rows(logits, spec_decode_metadata)
        q = proposer.verification_probs(spec_decode_metadata)
        native_sampler = self.rejection_sampler

        class WithDraftProbabilities:
            def __getattr__(self, name):
                return getattr(native_sampler, name)

            def __call__(self, metadata, draft_probs, *args, **kwargs):
                return native_sampler(metadata, q, *args, **kwargs)

        # 同步 worker 内临时注入 q，原生 _sample 的裁剪、prepare_sampling 等步骤不变。
        self.rejection_sampler = WithDraftProbabilities()
        try:
            output = sample(self, logits, spec_decode_metadata)
        finally:
            self.rejection_sampler = native_sampler
        proposer.observe_verified(spec_decode_metadata, teacher, output.sampled_token_ids)
        return output

    runner_class._set_up_drafter = retain
    runner_class._get_drafter = create
    runner_class.propose_draft_token_ids = draft
    runner_class._prepare_inputs = prepare
    runner_class._sample = verify
    runner_class._qwen38_orthrus_hooks = True
