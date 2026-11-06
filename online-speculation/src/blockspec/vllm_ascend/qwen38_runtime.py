# 中文学习版｜完整阅读路线见 docs/QWEN38_CODE_WALKTHROUGH.md
# 文件职责：引擎接入层：包装指定版本 NPUModelRunner 的生命周期、起草和采样入口；不重新实现原生 scheduler 或 rejection sampler。
# 约定：B=batch，K=块长（含 anchor），H=hidden_size，V=词表大小；通常 B=1。
# 注释不改变执行代码；续行与同一逻辑语句一起阅读。

"""Scoped compatibility hooks for the v0.26 Ascend eager runner.

This is an internal runner adapter, not a claim that upstream custom_class
already exposes neural-model lifecycle and post-verification feedback hooks.
Other proposers keep the original runner behavior.
"""

# 依赖：导入 functools。
import functools
# 依赖：导入 inspect。
import inspect


# ── _DraftProbabilities ──
# 职责：类型 `_DraftProbabilities`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class _DraftProbabilities:
    # 局部类的闭包会让整块 q 等待循环 GC；模块级代理可在验证后立即释放引用。
    __slots__ = ("sampler", "probabilities")

    # ── _DraftProbabilities.__init__ ──
    # 职责：保存原生 sampler 和本轮实际 draft 概率，创建短生命周期的调用代理。
    def __init__(self, sampler, probabilities):
        self.sampler, self.probabilities = sampler, probabilities

    # ── _DraftProbabilities.__getattr__ ──
    # 职责：把代理未定义的属性访问转发给原生 sampler，保持原生调用方接口。
    def __getattr__(self, name):
        return getattr(self.sampler, name)

    # ── _DraftProbabilities.__call__ ──
    # 职责：代理原生 rejection sampler，把本轮真实 draft 概率 q 替换进调用，其余参数保持原样。
    # 输入提示：metadata：本轮原生引擎的执行索引/长度/采样信息，具体类型由所在函数决定。
    def __call__(self, metadata, draft_probs, *args, **kwargs):
        return self.sampler(metadata, self.probabilities, *args, **kwargs)


# ── _ours ──
# 职责：判断 runner 当前 drafter 是否属于本套适配，避免影响其他 proposer。
def _ours(runner):
    draft = getattr(runner, "drafter", None)
    return (getattr(draft, "qwen38_native_proposer", False)
            or getattr(draft, "glm53_native_proposer", False))


# ── install_runner_hooks ──
# 职责：检查 runner 内部接口后安装限定版本包装；仅标记为本实现的 proposer 使用这些分支。
def install_runner_hooks(runner_class=None):
    if runner_class is None:
        # 依赖：从 vllm_ascend.worker.model_runner_v1 导入 NPUModelRunner；供本模块调用。
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
        runner_class = NPUModelRunner
    if getattr(runner_class, "_qwen38_orthrus_hooks", False):
        # 返回 None：本路径不提供计算结果（例如不需要训练/概率的分支）。
        return
    expected = {
        "_set_up_drafter": {"self"},
        "_get_drafter": {"self"},
        "_prepare_inputs": {"self", "scheduler_output"},
        "_build_attention_metadata": {"self", "num_tokens", "num_reqs", "max_query_len", "use_spec_decode"},
        "propose_draft_token_ids": {"self", "valid_sampled_token_ids", "scheduler_output"},
        "_sample": {"self", "logits", "spec_decode_metadata"},
    }
    for name, required in expected.items():
        method = getattr(runner_class, name, None)
        if method is None or not required <= inspect.signature(method).parameters.keys():
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise RuntimeError(f"unsupported Ascend runner interface: {name}; expected v0.26 eager runner")
    setup = runner_class._set_up_drafter
    get_drafter = runner_class._get_drafter
    propose = runner_class.propose_draft_token_ids
    sample = runner_class._sample
    prepare_inputs = runner_class._prepare_inputs
    build_metadata = runner_class._build_attention_metadata
    propose_signature = inspect.signature(propose)
    metadata_signature = inspect.signature(build_metadata)

    @functools.wraps(setup)
    # ── install_runner_hooks.retain ──
    # 职责：包装 _set_up_drafter：保留上游建立的 custom 实例，初始化后把 runner 绑定给它。
    def retain(self):
        custom = getattr(self, "drafter", None)
        if (getattr(custom, "qwen38_native_proposer", False)
                or getattr(custom, "glm53_native_proposer", False)):
            self._qwen38_custom = custom
        setup(self)
        if _ours(self):
            # 调用：保存 NPU runner 引用并取得 TP 通信组；后续访问当前请求和历史 metadata 都依赖它。
            self.drafter.bind_runner(self)

    @functools.wraps(get_drafter)
    # ── install_runner_hooks.create ──
    # 职责：包装 _get_drafter：优先返回保存的 custom 实例，否则继续原生创建逻辑。
    def create(self):
        custom = getattr(self, "_qwen38_custom", None)
        if custom is not None:
            return custom
        return get_drafter(self)

    @functools.wraps(propose)
    # ── install_runner_hooks.draft ──
    # 职责：包装 propose_draft_token_ids：解析原方法实参并转给本 proposer 的 propose_batch。
    def draft(self, *args, **kwargs):
        if not _ours(self):
            return propose(self, *args, **kwargs)
        arguments = propose_signature.bind(self, *args, **kwargs).arguments
        return self.drafter.propose_batch(arguments["valid_sampled_token_ids"], arguments["scheduler_output"],
                                         verification=arguments.get("spec_decode_metadata"))

    @functools.wraps(build_metadata)
    # ── install_runner_hooks.metadata ──
    # 职责：包装 metadata 构建：保留 GDN 接受边界语义，并把当前请求逐层 metadata 交给 history_reader。
    def metadata(self, *args, **kwargs):
        arguments = metadata_signature.bind(self, *args, **kwargs)
        if (_ours(self) and self.input_batch.num_reqs == 1
                and self.input_batch.num_computed_tokens_cpu[0] >= self.input_batch.num_prompt_tokens[0]):
            # 即使本轮没有候选，也要向 GDN builder 传入上轮接受位置。
            arguments.arguments["use_spec_decode"] = True
        # 数据：被调用步骤返回的结果；更新路径通常含 loss、版本、梯度范数和实际更新层。
        result = build_metadata(*arguments.args, **arguments.kwargs)
        if _ours(self) and self.drafter.history_reader is not None and self.input_batch.num_reqs == 1:
            # 调用：保存当前请求的逐层 metadata；这里只记录索引信息，真正张量读取发生在 read。
            self.drafter.history_reader.capture(result[0], self.input_batch.req_ids[0])
        # 输出：被调用步骤返回的结果；更新路径通常含 loss、版本、梯度范数和实际更新层。
        return result

    @functools.wraps(prepare_inputs)
    # ── install_runner_hooks.prepare ──
    # 职责：包装输入准备：区分 prefill 和没有候选的 decode，让 GDN 正确延续上次接受状态。
    def prepare(self, *args, **kwargs):
        # 数据：被调用步骤返回的结果；更新路径通常含 loss、版本、梯度范数和实际更新层。
        result = prepare_inputs(self, *args, **kwargs)
        if _ours(self) and result[1] is None:
            # Prefill 用 -1；无候选的 decode 仍用 0，让 GDN 从上轮接受状态继续。
            self.num_decode_draft_tokens.np.fill(-1)
            for index in range(self.input_batch.num_reqs):
                if self.input_batch.num_computed_tokens_cpu[index] >= self.input_batch.num_prompt_tokens[index]:
                    self.num_decode_draft_tokens.np[index] = 0
            self.num_decode_draft_tokens.copy_to_gpu()
        # 输出：被调用步骤返回的结果；更新路径通常含 loss、版本、梯度范数和实际更新层。
        return result

    @functools.wraps(sample)
    # ── install_runner_hooks.verify ──
    # 职责：在 runtime 中包装原生采样；在独立 ProposalSampler 中则是参考校正。以所在类/模块为准，不是两套同时调用。
    # 输入提示：logits：词表未归一化分数；最后一维 V，起草有效行通常是前 K−1 行。
    def verify(self, logits, spec_decode_metadata):
        if not _ours(self) or spec_decode_metadata is None:
            return sample(self, logits, spec_decode_metadata)
        # 保留原生 rejection sampler；在采样约束修改 logits 前取出教师分布。
        proposer = self.drafter
        # 数据：冻结主模型的监督 logits；它不承担参数更新。
        teacher = proposer.teacher_rows(logits, spec_decode_metadata)
        # 数据：本轮真实起草概率分布；随机校正需要它，greedy 可为 None。
        q = proposer.verification_probs(spec_decode_metadata)
        native_sampler = self.rejection_sampler

        # 同步 worker 内临时注入 q，原生 _sample 的裁剪、prepare_sampling 等步骤不变。
        self.rejection_sampler = _DraftProbabilities(native_sampler, q)
        # 保护区：执行可能失败的步骤；异常处理或 finally 负责恢复状态/释放资源。
        try:
            # 数据：前向结果包或算子输出；若为 BackboneOutput，可访问 logits/cache/boundary。
            output = sample(self, logits, spec_decode_metadata)
        finally:
            self.rejection_sampler = native_sampler
        # 调用：消费本轮 pending 与原生验证结果。统计接受情况，按实际到达行构造 Feedback；有 learner 才触发 observe。
        proposer.observe_verified(spec_decode_metadata, teacher, output.sampled_token_ids)
        # 输出：前向结果包或算子输出；若为 BackboneOutput，可访问 logits/cache/boundary。
        return output

    runner_class._set_up_drafter = retain
    runner_class._get_drafter = create
    runner_class.propose_draft_token_ids = draft
    runner_class._prepare_inputs = prepare
    runner_class._build_attention_metadata = metadata
    runner_class._sample = verify
    runner_class._qwen38_orthrus_hooks = True
