# 中文学习版｜完整阅读路线见 docs/QWEN38_CODE_WALKTHROUGH.md
# 文件职责：投机流程组织层：绑定原生模型，构造起草输入，保存 pending，提供 q，消费验证反馈并触发训练。
# 约定：B=batch，K=块长（含 anchor），H=hidden_size，V=词表大小；通常 B=1。
# 注释不改变执行代码；续行与同一逻辑语句一起阅读。

"""Qwen3.8 Orthrus on the native vLLM target, with serialized online replay.

Eager, synchronous batch-one text generation, BF16/FP32, TP-sharded target
weights. History is read from the native target cache, never recomputed by AR.
"""

# 依赖：导入 json。
import json
# 依赖：导入 logging。
import logging
# 依赖：从 pathlib 导入 Path；供本模块调用。
from pathlib import Path

# 依赖：导入 torch。
import torch

# 依赖：从 ..feedback 导入 Feedback；供本模块调用。
from ..feedback import Feedback
# 依赖：从 ..parallel.online 导入 SuffixConfig；供本模块调用。
from ..parallel.online import SuffixConfig
# 依赖：从 ..parallel.sampling 导入 ProposalSampler；供本模块调用。
from ..parallel.sampling import ProposalSampler
# 依赖：从 ..sampling 导入 SamplingConfig；供本模块调用。
from ..sampling import SamplingConfig
# 依赖：从 .qwen38_runtime 导入 install_runner_hooks；供本模块调用。
from .qwen38_runtime import install_runner_hooks
# 依赖：从 .qwen38_attention 导入 install_attention_hooks；供本模块调用。
from .qwen38_attention import install_attention_hooks
# 依赖：从 .qwen38_weights 导入 NativeSuffixLearner, TensorParallel, from_native_model；供本模块调用。
from .qwen38_weights import NativeSuffixLearner, TensorParallel, from_native_model
# 依赖：从 .qwen38_history 导入 NativeHybridHistory；供本模块调用。
from .qwen38_history import NativeHybridHistory


logger = logging.getLogger("vllm.blockspec.qwen38")


# ── OrthrusWorkerExtension ──
# 职责：类型 `OrthrusWorkerExtension`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class OrthrusWorkerExtension:
    """Read worker counters through vLLM's named RPC, without pickled callables."""

    # ── OrthrusWorkerExtension.orthrus_use_cann_gdn_prefill ──
    # 职责：为当前 worker 安装选定的 CANN GDN prefill 适配。
    def orthrus_use_cann_gdn_prefill(self):
        # 依赖：从 .qwen38_gdn_backend 导入 install_cann_gdn_prefill；供本模块调用。
        from .qwen38_gdn_backend import install_cann_gdn_prefill
        # 调用：把符合配置的原生 GDN prefill 切换为已有 CANN 路径；不是训练优化器。
        install_cann_gdn_prefill(self.model_runner.vllm_config)
        return "cann"

    # ── OrthrusWorkerExtension.orthrus_stats ──
    # 职责：通过 worker RPC 返回当前 rank 起草、验证、更新计数及配置；不是模型 forward。
    def orthrus_stats(self):
        draft = self.model_runner.drafter
        if not getattr(draft, "qwen38_native_proposer", False):
            return {"orthrus": False}
        return {"orthrus": True, "rank": draft.tp.rank, "proposals": draft.proposals,
                "verified": draft.verified, "accepted": draft.accepted,
                "max_verified_tokens": draft.max_verified_tokens, "full_blocks": draft.full_blocks,
                "updates": 0 if draft.learner is None else draft.learner.updates,
                "loss": None if draft.learner is None else draft.learner.last_loss,
                "update_seconds": 0. if draft.learner is None else draft.learner.update_seconds,
                "updated_layers": 0 if draft.learner is None else draft.learner.config.last_layers,
                "trainable_parameters": 0 if draft.learner is None else draft.learner.trainable_parameters,
                "optimizer_device": None if draft.learner is None else draft.learner.config.optimizer_device,
                "training_gdn_backend": draft.options.get("training_gdn_backend", "torch"),
                "training_loss_backend": draft.options.get("training_loss_backend", "gathered"),
                "head_dir": draft.options.get("head_dir"),
                "history_backend": draft.history_reader.backend, "history_reads": draft.history_reader.reads}

    # ── OrthrusWorkerExtension.orthrus_offline_step ──
    # 职责：worker 上的一次离线训练：允许梯度、创建 token 张量、构造 offline_feedback，再交给同一个 learner。
    # 输入提示：anchor：锚点：起草时指已知 token ID，离线训练函数的 anchor 参数则是文本中的位置索引，注意所在函数。
    # 调用者：train_qwen38_draft.py 用 llm.collective_rpc 调度各 rank 执行。
    # 输入：一条干净文本的 token_ids 与 anchor 位置索引。输出：本 rank 的层覆盖、loss、梯度范数等更新结果。
    def orthrus_offline_step(self, token_ids, anchor):
        # 依赖：从 ..parallel.qwen38_training 导入 offline_feedback；供本模块调用。
        from ..parallel.qwen38_training import offline_feedback
        draft = self.model_runner.drafter
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：offline updates require a separate offline worker, not a serving learner。
        if not getattr(draft, "qwen38_native_proposer", False) or not draft.options.get("offline", False):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise RuntimeError("offline updates require a separate offline worker, not a serving learner")
        # 梯度上下文：torch.inference_mode(False)；False 的 inference_mode 允许普通张量/训练，但 no_grad 仍禁止记录本段前向图。
        with torch.inference_mode(False), torch.enable_grad():
            tokens = torch.tensor([token_ids], device=draft.model.embedding.weight.device, dtype=torch.long)
            # 数据：训练样本包：inputs/history/teacher_logits/valid/boundary，之后重放不依赖旧计算图。
            feedback = offline_feedback(draft.model, tokens, anchor)
            # 数据：被调用步骤返回的结果；更新路径通常含 loss、版本、梯度范数和实际更新层。
            result = draft.learner.observe(feedback)
            # 调用：清除旧请求/旧步骤保存的反馈，避免跨请求训练材料混用或不必要显存占用。
            draft.learner.clear_replay()
        return {"rank": draft.tp.rank, "layers": draft.learner.config.last_layers,
                "trainable_parameters": draft.learner.trainable_parameters, **result}

    # ── OrthrusWorkerExtension.orthrus_checkpoint ──
    # 职责：通过 worker RPC 导出 head；调用方不直接操作各卡参数。
    def orthrus_checkpoint(self):
        draft = self.model_runner.drafter
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：checkpoint requires an updated Orthrus learner and checkpoint_dir。
        if (not getattr(draft, "qwen38_native_proposer", False) or draft.learner is None
                or not draft.learner.updates or not draft.options.get("checkpoint_dir")):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise RuntimeError("checkpoint requires an updated Orthrus learner and checkpoint_dir")
        root = Path(draft.options["checkpoint_dir"])
        # 调用：保存当前 draft 参数/训练状态等交付信息，供后续重载；检查具体输出文件以区分 head 和优化器状态。
        draft._save_head(root)
        return {"rank": draft.tp.rank, "version": draft.learner.version,
                "directory": str(root / f"version{draft.learner.version}")}


# ── Qwen38OrthrusProposer ──
# 职责：类型 `Qwen38OrthrusProposer`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class Qwen38OrthrusProposer:
    qwen38_native_proposer = True

    # ── Qwen38OrthrusProposer.__init__ ──
    # 职责：检查 BF16/FP32、K、单请求/eager 等支持范围，初始化计数与空状态，安装限定版本 hooks；模型绑定稍后完成。
    def __init__(self, vllm_config):
        self.config = vllm_config
        spec = vllm_config.speculative_config
        options = (vllm_config.additional_config or {}).get("qwen38_orthrus", {})
        # 数据：Attention 可见范围；AR 屏蔽未来，单个 draft 块全可见。
        allowed = {"mask_token_id", "online", "last_layers", "update_stride", "learning_rate",
                   "head_dir", "checkpoint_dir", "checkpoint_every", "all_draft_layers", "offline",
                   "optimizer_device", "checkpoint_layers", "training_gdn_backend", "training_loss_backend"}
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：invalid qwen38_orthrus options。
        if not isinstance(options, dict) or set(options) - allowed:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("invalid qwen38_orthrus options")
        self.options = dict(options)
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：training_loss_backend must be gathered or vocab_parallel。
        if options.get("training_loss_backend", "gathered") not in {"gathered", "vocab_parallel"}:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("training_loss_backend must be gathered or vocab_parallel")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：training_gdn_backend must be torch or fla。
        if options.get("training_gdn_backend", "torch") not in {"torch", "fla"}:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("training_gdn_backend must be torch or fla")
        native_backend = (vllm_config.additional_config or {}).get("qwen38_gdn_prefill_backend", "native")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：qwen38_gdn_prefill_backend must be native or cann。
        if native_backend not in {"native", "cann"}:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("qwen38_gdn_prefill_backend must be native or cann")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：CANN GDN prefill requires a BF16 target。
        if native_backend == "cann" and vllm_config.model_config.dtype != torch.bfloat16:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("CANN GDN prefill requires a BF16 target")
        for name in ("online", "offline", "all_draft_layers", "checkpoint_layers"):
            if type(options.get(name, False)) is not bool:
                # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
                raise ValueError(f"{name} must be a boolean")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：offline fitting and online serving use separate workers。
        if options.get("online", False) and options.get("offline", False):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("offline fitting and online serving use separate workers")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：optimizer_device must be model or cpu。
        if options.get("optimizer_device", "model") not in {"model", "cpu"}:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("optimizer_device must be model or cpu")
        for name in ("last_layers", "update_stride", "checkpoint_every"):
            if name in options and (type(options[name]) is not int or options[name] < 1):
                # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
                raise ValueError(f"{name} must be a positive integer")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：explicit qwen38_orthrus.mask_token_id required。
        if type(options.get("mask_token_id")) is not int:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("explicit qwen38_orthrus.mask_token_id required")
        parallel = vllm_config.parallel_config
        schedule = vllm_config.scheduler_config
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：Qwen3.8 Orthrus requires eager sync max_num_seqs=1, PP=DP=DCP=PCP=1, no LoRA, BF16/FP32。
        if (not vllm_config.model_config.enforce_eager or schedule.max_num_seqs != 1
                or getattr(schedule, "async_scheduling", False)
                or parallel.pipeline_parallel_size != 1 or parallel.data_parallel_size != 1
                or getattr(parallel, "decode_context_parallel_size", 1) != 1
                or getattr(parallel, "prefill_context_parallel_size", 1) != 1
                or getattr(vllm_config, "lora_config", None) is not None
                or vllm_config.quant_config is not None
                or vllm_config.model_config.dtype not in {torch.bfloat16, torch.float32}):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("Qwen3.8 Orthrus requires eager sync max_num_seqs=1, PP=DP=DCP=PCP=1, no LoRA, BF16/FP32")
        multimodal = getattr(vllm_config.model_config, "multimodal_config", None)
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：Qwen3.8 Orthrus currently requires language_model_only=True。
        if multimodal is not None and not multimodal.language_model_only:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("Qwen3.8 Orthrus currently requires language_model_only=True")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：custom_class with 1..31 speculative tokens required。
        if spec.method != "custom_class" or not 1 <= spec.num_speculative_tokens <= 31:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("custom_class with 1..31 speculative tokens required")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：greedy or probabilistic draft sampling required。
        if getattr(spec, "draft_sample_method", "greedy") not in {"greedy", "probabilistic"}:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("greedy or probabilistic draft sampling required")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：standard exact rejection sampling required。
        if getattr(spec, "rejection_sample_method", "standard") != "standard":
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("standard exact rejection sampling required")
        self.block_size = spec.num_speculative_tokens + 1
        # 数据：训练器对象；fixed-head 推理时可以为 None。
        self.runner = self.model = self.learner = None
        # 数据：各层已提交 AR 历史；Full Attention 是 K/V，GDN 是卷积/递推状态；保存本轮输入/候选/q/boundary，等待原生 verifier 的对应结果。
        self.history = self.history_reader = self.pending = self.request_id = None
        # 数据：本轮已提交边界对应的位置数；具体包含哪些输入由调用处的 native metadata 定义。
        self.proposals = self.verified = self.accepted = 0
        self.max_verified_tokens = self.full_blocks = 0
        # 调用：检查 runner 内部接口后安装限定版本包装；仅标记为本实现的 proposer 使用这些分支。
        install_runner_hooks()
        # 调用：为指定版本 Ascend Attention 安装 builder 和调用切片适配；不把 verifier 改成双向。
        install_attention_hooks()
        # 依赖：从 .qwen38_gdn_backend 导入 install_recurrent_hook；供本模块调用。
        from .qwen38_gdn_backend import install_recurrent_hook
        # 调用：为原生 GDN 安装递推切片适配，避免短块 query 限制导致调用失败。
        install_recurrent_hook(vllm_config)
        if (vllm_config.additional_config or {}).get("qwen38_gdn_prefill_backend") == "cann":
            # 依赖：从 .qwen38_gdn_backend 导入 install_cann_gdn_prefill；供本模块调用。
            from .qwen38_gdn_backend import install_cann_gdn_prefill
            # 调用：把符合配置的原生 GDN prefill 切换为已有 CANN 路径；不是训练优化器。
            install_cann_gdn_prefill(vllm_config)

    # ── Qwen38OrthrusProposer.bind_runner ──
    # 职责：保存 NPU runner 引用并取得 TP 通信组；后续访问当前请求和历史 metadata 都依赖它。
    def bind_runner(self, runner):
        self.runner = runner
        # 依赖：从 vllm.distributed 导入 get_tp_group；供本模块调用。
        from vllm.distributed import get_tp_group
        group = get_tp_group()
        # 数据：TP 通信适配器，持有通信组与 rank 信息，不是模型本身。
        self.tp = TensorParallel(group.device_group, group.ranks)

    # ── Qwen38OrthrusProposer.load_model ──
    # 职责：基于已加载 target 构建共享权重双视图，初始化历史读取器/随机数生成器，按配置加载 head 和创建 learner。
    # 调用时机：原生 target 已加载后，由 runner 生命周期调用；不要自己再次 from_pretrained。
    # 输出/副作用：建立 self.model/history_reader/generator，按配置建立 learner；无返回模型对象。
    def load_model(self, target_model):
        # 梯度上下文：torch.inference_mode(False)；False 的 inference_mode 允许普通张量/训练，但 no_grad 仍禁止记录本段前向图。
        with torch.inference_mode(False), torch.no_grad():
            text_config = self.config.model_config.hf_text_config
            self.model = from_native_model(target_model, text_config, tp=self.tp,
                                          block_size=self.block_size,
                                          mask_token_id=self.options["mask_token_id"],
                                          training_gdn_backend=self.options.get("training_gdn_backend", "torch"))
            self.history_reader = NativeHybridHistory(target_model, self.runner, self.model.layer_types)
            self.model.source["history_backend"] = self.history_reader.backend
            # 数据：本 rank 的 draft 采样随机数生成器，控制随机候选的可复现性。
            self.generator = torch.Generator(device=self.model.embedding.weight.device)
            # 设置随机种子；相同初始种子与同样调用顺序才能重现采样序列。
            self.generator.manual_seed(self.config.model_config.seed + 1337)
            if self.options.get("head_dir"):
                # 调用：校验交付 head 身份及 tensor 布局，再复制到当前 draft 执行参数；不是加载第二份基模。
                self._load_head(Path(self.options["head_dir"]))
            # 数据：独立离线 worker 标记，禁止与线上生成服务角色混用。
            offline = self.options.get("offline", False)
            if self.options.get("online", False) or offline:
                # 数据：离线默认选择全部 draft 层；在线是否全层由显式配置决定。
                all_layers = offline or self.options.get("all_draft_layers", False)
                # 数据：SuffixConfig：待更新层数、stride、loss、学习率、优化器设备和 checkpoint 策略。
                policy = SuffixConfig(last_layers=len(self.model.layers) if all_layers else self.options.get("last_layers", 1),
                                      stride=1 if offline else self.options.get("update_stride", 16),
                                      loss="forward_kl" if offline else "tv",
                                      learning_rate=self.options.get("learning_rate", 1e-5),
                                      optimizer_device=self.options.get("optimizer_device", "model"),
                                      checkpoint_layers=self.options.get("checkpoint_layers", False))
                # 数据：训练器对象；fixed-head 推理时可以为 None。
                self.learner = NativeSuffixLearner(self.model, policy, self.tp,
                    vocab_parallel_loss=self.options.get("training_loss_backend", "gathered") == "vocab_parallel")
        mode = "offline" if self.options.get("offline", False) else "online" if self.learner is not None else "fixed"
        # 记录运行信息/结果，不改变模型预测公式；文件输出由调用参数指定。
        logger.info("Orthrus native target ready: TP=%d, K=%d, mode=%s, history=shared_target_snapshot",
                    self.tp.size, self.block_size, mode)

    # ── Qwen38OrthrusProposer.propose ──
    # 职责：拒绝走不含原生模型/历史生命周期的通用入口；本实现需要 runtime 的 propose_batch 接入。
    def propose(self, *args, **kwargs):
        # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
        raise RuntimeError("neural Orthrus requires the scoped Ascend runner adapter")

    # ── Qwen38OrthrusProposer.dummy_run ──
    # 职责：此处不执行实际 draft；权重/master 已预先分配，历史快照也不参加图捕获。
    def dummy_run(self, **kwargs):
        # Weights and FP32 optimizer masters are allocated before KV profiling.
        # History snapshots are bounded by max_model_len, not graph-captured.
        # 空实现：保留接口，但本路径有意不执行实际计算。
        pass

    # ── Qwen38OrthrusProposer._clear_request ──
    # 职责：切换请求时清空旧 history/pending/replay，并按请求随机种子重置 draft RNG，避免请求之间串状态。
    # 输入提示：request_id：当前请求身份，用于拒绝使用其他请求遗留的 pending/history。
    def _clear_request(self, request_id):
        # 数据：当前请求身份，用于拒绝使用其他请求遗留的 pending/history。
        self.request_id = request_id
        # 数据：各层已提交 AR 历史；Full Attention 是 K/V，GDN 是卷积/递推状态；保存本轮输入/候选/q/boundary，等待原生 verifier 的对应结果。
        self.history = self.pending = None
        generators = self.runner.input_batch.sampling_metadata.generators
        if 0 in generators:
            # 设置随机种子；相同初始种子与同样调用顺序才能重现采样序列。
            self.generator.manual_seed((generators[0].initial_seed() + 1337) % (2**64))
        # 训练开关：有 learner 时才记录/更新训练状态；fixed-head 推理跳过这部分。
        if self.learner is not None:
            # 调用：清除旧请求/旧步骤保存的反馈，避免跨请求训练材料混用或不必要显存占用。
            self.learner.clear_replay()

    # ── Qwen38OrthrusProposer._draft_sampling ──
    # 职责：把原生请求的 greedy/temperature/top-k/top-p 转成 draft 使用的同一概率策略。
    def _draft_sampling(self):
        # 数据：本轮原生引擎的执行索引/长度/采样信息，具体类型由所在函数决定。
        metadata = self.runner.input_batch.sampling_metadata
        if self.config.speculative_config.draft_sample_method == "greedy" or metadata.all_greedy:
            return SamplingConfig()
        return SamplingConfig(temperature=float(metadata.temperature[0]),
                              top_k=0 if metadata.top_k is None else max(0, int(metadata.top_k[0])),
                              top_p=1. if metadata.top_p is None else float(metadata.top_p[0]))

    # ── Qwen38OrthrusProposer.propose_batch ──
    # 职责：起草入口。输入是原生同步记账后的 sampled token 与调度信息；读取严格历史，构造 [1,K] anchor+MASK，返回候选 ID 列表。
    # 调用者：qwen38_runtime.install_runner_hooks 中的 draft 包装。
    # 输入：sampled 是同步记账后的本轮输出；verification 是上轮验证元数据，可为 None。
    # 输出：batch-one 的候选 ID 列表 [[id1,...]]；不是 logits，也不是已经验证的文本。
    # 数据去向：pending 留给下一次 verify；原生引擎拿候选运行 target，不把 draft KV 当成 AR KV。
    def propose_batch(self, sampled, scheduler_output, *, verification=None):
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：offline workers cannot serve generation requests。
        if self.options.get("offline", False):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise RuntimeError("offline workers cannot serve generation requests")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：draft weights have not been bound to the native target。
        if self.model is None:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise RuntimeError("draft weights have not been bound to the native target")
        runner = self.runner
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：only synchronous batch-one Orthrus is supported。
        if runner.input_batch.num_reqs != 1 or len(sampled) != 1:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise RuntimeError("only synchronous batch-one Orthrus is supported")
        if isinstance(sampled, torch.Tensor):
            sampled = [[v for v in row if v >= 0] for row in sampled.cpu().tolist()]
        # 数据：当前请求身份，用于拒绝使用其他请求遗留的 pending/history。
        request_id = runner.input_batch.req_ids[0]
        if request_id != self.request_id:
            # 调用：切换请求时清空旧 history/pending/replay，并按请求随机种子重置 draft RNG，避免请求之间串状态。
            self._clear_request(request_id)
        if not sampled[0]:
            return [[]]
        # 数据：本轮有效 token/候选数量；以赋值来源和所在函数为准，不等同于固定 K。
        count = int(runner.input_batch.num_tokens_no_spec[0])
        # 数据：锚点：起草时指已知 token ID，离线训练函数的 anchor 参数则是文本中的位置索引，注意所在函数。
        anchor = int(runner.input_batch.token_ids_cpu[0, count - 1]) if count else None
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：draft must run after native synchronous token bookkeeping。
        if anchor is None or anchor != sampled[0][-1]:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise RuntimeError("draft must run after native synchronous token bookkeeping")
        # 数据：anchor 前的严格历史长度；绝对位置与缓存读取边界以它为准。
        past = count - 1
        device = self.model.embedding.weight.device
        # 梯度上下文：torch.inference_mode(False)；False 的 inference_mode 允许普通张量/训练，但 no_grad 仍禁止记录本段前向图。
        with torch.inference_mode(False), torch.no_grad():
            # Anchor 后一次预测整个 MASK 块；所有位置共享同一份已提交的 AR 历史。
            # 数据：本轮实际块长，受配置 K 和剩余上下文空间共同限制。
            width = min(self.block_size, runner.max_model_len - past)
            if width < 2:
                # 数据：保存本轮输入/候选/q/boundary，等待原生 verifier 的对应结果。
                self.pending = None
                return [[]]
            # 数据：原生 metadata 记录的本轮候选个数，用于分辨普通 decode 和投机验证。
            draft_count = 0 if verification is None else verification.num_draft_tokens[0]
            # 数据：各层已提交 AR 历史；Full Attention 是 K/V，GDN 是卷积/递推状态。
            self.history = self.history_reader.read(request_id, past, sampled_count=len(sampled[0]),
                                                    draft_count=draft_count)
            # 数据：当前 draft token 块 [1,K]；第 0 个是已知 anchor，其余是 MASK ID。
            inputs = torch.full((1, width), self.options["mask_token_id"], device=device, dtype=torch.long)
            # 数据：当前 draft token 块 [1,K]；第 0 个是已知 anchor，其余是 MASK ID。
            inputs[0, 0] = anchor
            # 数据：需要训练反馈时的后缀起始层；不需要时为 None。
            capture = (self.learner.capture_layer if self.learner is not None
                       and self.learner.needs_decoder_feedback else None)
            # 数据：前向结果包或算子输出；若为 BackboneOutput，可访问 logits/cache/boundary。
            output = self.model(inputs, view="draft", cache=self.history, capture_layer=capture)
            # 数据：词表未归一化分数；最后一维 V，起草有效行通常是前 K−1 行。 此处转为 FP32 做分数/归一化/梯度等中间计算。
            logits = output.logits[0, :-1].float()
            # 与参考算法使用相同的温度/top-k/top-p；传给 verifier 的 q 必须是真实起草分布。
            # 数据：待主模型验证的 token ID，一维长度最多 K−1；本轮真实起草概率分布；随机校正需要它，greedy 可为 None。
            candidates, q, _ = ProposalSampler(self._draft_sampling()).propose(logits, self.generator)
            # 调用：用首个 TP rank 的内容同步其他 rank，确保候选 token 一致。
            self.tp.broadcast(candidates)
            # 数据：保存本轮输入/候选/q/boundary，等待原生 verifier 的对应结果。
            self.pending = {"request_id": request_id, "inputs": inputs, "cache": self.history,
                            "boundary": output.boundary, "q": q, "candidates": candidates}
            # 状态推进：在原有值上累加/更新；若目标是张量，要注意这里的原地语义。
            self.proposals += 1
            return [candidates.cpu().tolist()]

    # ── Qwen38OrthrusProposer._verification_count ──
    # 职责：检查原生待验证候选与 pending 一致；返回本轮实际验证候选数，不允许错请求或错 token。
    # 输入提示：metadata：本轮原生引擎的执行索引/长度/采样信息，具体类型由所在函数决定。
    def _verification_count(self, metadata):
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：batch-one verification metadata required。
        if len(metadata.num_draft_tokens) != 1:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise RuntimeError("batch-one verification metadata required")
        # 数据：本轮有效 token/候选数量；以赋值来源和所在函数为准，不等同于固定 K。
        count = metadata.num_draft_tokens[0]
        if not count:
            # 数据：保存本轮输入/候选/q/boundary，等待原生 verifier 的对应结果。
            self.pending = None
            return 0
        # 数据：保存本轮输入/候选/q/boundary，等待原生 verifier 的对应结果。
        pending = self.pending
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：missing proposal for native verifier request。
        if pending is None or pending["request_id"] != self.runner.input_batch.req_ids[0]:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise RuntimeError("missing proposal for native verifier request")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：native verifier draft tokens do not match the pending proposal。
        if count > pending["candidates"].numel() or not torch.equal(
                metadata.draft_token_ids.to(pending["candidates"]), pending["candidates"][:count]):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise RuntimeError("native verifier draft tokens do not match the pending proposal")
        # 输出：本轮有效 token/候选数量；以赋值来源和所在函数为准，不等同于固定 K。
        return count

    # ── Qwen38OrthrusProposer.verification_probs ──
    # 职责：从 pending 取实际用于起草的 q，按本轮验证长度截取；greedy 路径允许返回 None。
    # 输入提示：metadata：本轮原生引擎的执行索引/长度/采样信息，具体类型由所在函数决定。
    def verification_probs(self, metadata):
        # 数据：本轮有效 token/候选数量；以赋值来源和所在函数为准，不等同于固定 K。
        count = self._verification_count(metadata)
        if not count:
            return None
        # 数据：本轮真实起草概率分布；随机校正需要它，greedy 可为 None。
        q = self.pending["q"]
        return None if q is None else q[:count].contiguous()

    # ── Qwen38OrthrusProposer.teacher_rows ──
    # 职责：在 sampler 修改 logits 之前摘取教师行；只有在线更新确实捕获 boundary 时才需要。
    # 输入提示：logits：词表未归一化分数；最后一维 V，起草有效行通常是前 K−1 行；metadata：本轮原生引擎的执行索引/长度/采样信息，具体类型由所在函数决定。
    def teacher_rows(self, logits, metadata):
        # 数据：本轮有效 token/候选数量；以赋值来源和所在函数为准，不等同于固定 K。
        count = self._verification_count(metadata)
        # 训练开关：有 learner 时才记录/更新训练状态；fixed-head 推理跳过这部分。
        if not count or self.learner is None or self.pending["boundary"] is None:
            return None
        # 梯度上下文：torch.inference_mode(False)；False 的 inference_mode 允许普通张量/训练，但 no_grad 仍禁止记录本段前向图。
        with torch.inference_mode(False), torch.no_grad():
            # index_select 已拥有独立存储；采样器随后改写原 logits 不会污染教师。
            return logits.index_select(0, metadata.target_logits_indices).detach()

    # ── Qwen38OrthrusProposer.observe_verified ──
    # 职责：消费本轮 pending 与原生验证结果。统计接受情况，按实际到达行构造 Feedback；有 learner 才触发 observe。
    # 输入提示：metadata：本轮原生引擎的执行索引/长度/采样信息，具体类型由所在函数决定；teacher：冻结主模型的监督 logits；它不承担参数更新。
    # 调用者：runtime.verify 在原生 _sample 返回后执行。
    # 输入：metadata 描述本次候选，teacher 是未经采样约束改写的监督 logits，sampled 含有效输出和负数 padding。
    # 输出/副作用：清空本次 pending；累加统计；有 boundary 时将 Feedback 给 learner。更新默认是同步完成。
    def observe_verified(self, metadata, teacher, sampled):
        # 数据：本轮有效 token/候选数量；以赋值来源和所在函数为准，不等同于固定 K。
        count = metadata.num_draft_tokens[0]
        if not count:
            # 返回 None：本路径不提供计算结果（例如不需要训练/概率的分支）。
            return
        # 数据：保存本轮输入/候选/q/boundary，等待原生 verifier 的对应结果。
        pending, self.pending = self.pending, None
        # 数据：原生 sampler 本轮输出的有效 token 个数，包含校正/bonus token 的情况。
        sampled_count = int((sampled[0] >= 0).sum().item())
        # 数据：实际到达且可用于本轮在线监督的位置数；拒绝后的后缀不能直接算作真实到达。
        used = min(sampled_count, count)
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：native verifier produced no reached token。
        if used < 1:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise RuntimeError("native verifier produced no reached token")
        # 状态推进：在原有值上累加/更新；若目标是张量，要注意这里的原地语义。
        self.verified += 1
        self.max_verified_tokens = max(self.max_verified_tokens, count)
        # 状态推进：在原有值上累加/更新；若目标是张量，要注意这里的原地语义。
        self.full_blocks += count == self.block_size - 1
        # 状态推进：在原有值上累加/更新；若目标是张量，要注意这里的原地语义。
        self.accepted += min(sampled_count - 1, count)
        # 训练开关：有 learner 时才记录/更新训练状态；fixed-head 推理跳过这部分。
        if self.learner is None:
            # 返回 None：本路径不提供计算结果（例如不需要训练/概率的分支）。
            return
        # 梯度上下文：torch.inference_mode(False)；False 的 inference_mode 允许普通张量/训练，但 no_grad 仍禁止记录本段前向图。
        with torch.inference_mode(False):
            if pending["boundary"] is None:
                # 调用：记录未捕获 boundary 的正常轮次，同时验证它确实不在必须训练的窗口。
                self.learner._skip_decoder_feedback(used)
                # 返回 None：本路径不提供计算结果（例如不需要训练/概率的分支）。
                return
            # 约束检查：下方条件表示不支持的输入/状态；触发时报错：online update requires raw reached teacher logits。
            if teacher is None:
                # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
                raise RuntimeError("online update requires raw reached teacher logits")
            # 数据：训练样本包：inputs/history/teacher_logits/valid/boundary，之后重放不依赖旧计算图。
            feedback = Feedback(pending["inputs"], pending["cache"], teacher[:used], used, pending["boundary"])
            # 数据：被调用步骤返回的结果；更新路径通常含 loss、版本、梯度范数和实际更新层。
            result = self.learner.observe(feedback)
            if result is not None:
                # 记录运行信息/结果，不改变模型预测公式；文件输出由调用参数指定。
                logger.info("Orthrus online update: rank=%d version=%d reached=%d loss=%.7f grad_norm=%.7f",
                            self.tp.rank, result["version"], used, result["loss"], result["gradient_norm"])
                every = self.options.get("checkpoint_every", 16)
                # 训练开关：有 learner 时才记录/更新训练状态；fixed-head 推理跳过这部分。
                if self.options.get("checkpoint_dir") and self.learner.version % every == 0:
                    # 调用：保存当前 draft 参数/训练状态等交付信息，供后续重载；检查具体输出文件以区分 head 和优化器状态。
                    self._save_head(Path(self.options["checkpoint_dir"]))

    # ── Qwen38OrthrusProposer._head_identity ──
    # 职责：构造 head 与当前模型/TP/算法配置的身份信息，防止误加载不兼容参数。
    def _head_identity(self):
        # 依赖：导入 hashlib。
        import hashlib
        source = Path(self.config.model_config.model)
        files = (source / "config.json", source / "model.safetensors.index.json")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：head exchange requires a local indexed checkpoint and config.json。
        if not all(path.is_file() for path in files):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("head exchange requires a local indexed checkpoint and config.json")
        hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in files]
        return {"format": "orthrus-qwen38-native-tp-v1", "base_config_sha256": hashes[0],
                "base_index_sha256": hashes[1], "tp_size": self.tp.size,
                "block_size": self.block_size, "mask_token_id": self.options["mask_token_id"],
                "hybrid_policy": "orthrus_qwen35_reference"}

    # ── Qwen38OrthrusProposer._draft_parameters ──
    # 职责：仅枚举新增 draft mixer 参数；共享 backbone 不属于 head 交付物。
    def _draft_parameters(self):
        return {name: p for name, p in self.model.named_parameters() if ".attention.draft." in name}

    # ── Qwen38OrthrusProposer._load_head ──
    # 职责：校验交付 head 身份及 tensor 布局，再复制到当前 draft 执行参数；不是加载第二份基模。
    def _load_head(self, directory):
        # 依赖：从 safetensors.torch 导入 load_file；供本模块调用。
        from safetensors.torch import load_file
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：head must match the base model, TP layout, block and hybrid policy。
        if json.loads((directory / "manifest.json").read_text()) != self._head_identity():
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("head must match the base model, TP layout, block and hybrid policy")
        tensors = load_file(str(directory / f"rank{self.tp.rank}.safetensors"))
        # 数据：优化器管理的 FP32 master 参数列表，不是所有冻结 backbone 参数。
        parameters = self._draft_parameters()
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：head tensor names do not match the draft mixers。
        if tensors.keys() != parameters.keys():
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("head tensor names do not match the draft mixers")
        for name, value in tensors.items():
            if (value.shape != parameters[name].shape or value.dtype != parameters[name].dtype
                    or not torch.isfinite(value).all()):
                # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
                raise ValueError(f"invalid head tensor: {name}")
        for name, value in tensors.items():
            # 执行：把值写入既有目标存储；是否可写需看目标是 draft、master 还是工作 cache。
            parameters[name].copy_(value)

    # ── Qwen38OrthrusProposer._save_head ──
    # 职责：保存当前 draft 参数/训练状态等交付信息，供后续重载；检查具体输出文件以区分 head 和优化器状态。
    def _save_head(self, directory):
        # 依赖：从 safetensors.torch 导入 save_file；供本模块调用。
        from safetensors.torch import save_file
        # 每版独立保存；全部 TP 分片写完后，rank 0 才发布 manifest。
        directory = directory / f"version{self.learner.version}"
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "manifest.json").exists():
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError(f"refusing to overwrite a published head: {directory}")
        shard = directory / f"rank{self.tp.rank}.safetensors"
        # 执行：使张量布局连续，满足算子/通信输入要求；必要时会拷贝；断开旧 autograd 计算图，不自动复制底层存储。
        save_file({name: p.detach().cpu().contiguous() for name, p in self._draft_parameters().items()},
                  str(shard.with_suffix(".tmp")))
        shard.with_suffix(".tmp").replace(shard)
        if self.tp.size > 1:
            torch.distributed.barrier(group=self.tp.group)
        if self.tp.rank == 0:
            manifest = directory / "manifest.json"
            # 记录运行信息/结果，不改变模型预测公式；文件输出由调用参数指定。
            manifest.with_suffix(".tmp").write_text(json.dumps(self._head_identity(), indent=2) + "\n")
            manifest.with_suffix(".tmp").replace(manifest)
