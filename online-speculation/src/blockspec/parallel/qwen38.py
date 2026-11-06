# 中文学习版｜完整阅读路线见 docs/QWEN38_CODE_WALKTHROUGH.md
# 文件职责：模型结构层：AR/draft 双视图、Full Attention 与 GDN 混合层、临时 cache，以及可重放的后缀 forward。
# 约定：B=batch，K=块长（含 anchor），H=hidden_size，V=词表大小；通常 B=1。
# 注释不改变执行代码；续行与同一逻辑语句一起阅读。

"""Qwen3.5/3.8 text-only dual view, using existing Transformers mixers.

The GDN extension follows Orthrus' Qwen3.5 reference, not an attention-mask
replacement: its draft recurrence is causal within each masked block. The
explicit policy records this model adaptation rather than silently replacing GDN.
"""

# 依赖：导入 copy。
import copy
# 依赖：从 types 导入 SimpleNamespace；供本模块调用。
from types import SimpleNamespace

# 依赖：导入 torch。
import torch
# 依赖：从 torch 导入 nn；供本模块调用。
from torch import nn
# 依赖：从 torch.utils.checkpoint 导入 checkpoint；供本模块调用。
from torch.utils.checkpoint import checkpoint

# 依赖：从 .backbone 导入 BackboneOutput, DraftBoundary；供本模块调用。
from .backbone import BackboneOutput, DraftBoundary
# 依赖：从 .branches 导入 MaskedAttentionBranch；供本模块调用。
from .branches import MaskedAttentionBranch


@torch.no_grad()
# ── native_reference_logits ──
# 职责：独立 HF 参考前向，用 hook 支持手动按层放置；vLLM 在线起草不靠它重算历史。
def native_reference_logits(target, tokens):
    """Run the unchanged HF forward with hooks for manual layer placement."""
    text = getattr(target.model, "language_model", target.model)

    # ── native_reference_logits.dispatch ──
    # 职责：在手动分层参考模型执行前，把本层输入移动到本层参数设备。
    def dispatch(module, args, kwargs):
        device = next(module.parameters()).device

        # ── native_reference_logits.dispatch.move ──
        # 职责：递归搬移张量/元组到目标层设备；非张量对象保留原样。
        def move(item):
            if isinstance(item, torch.Tensor):
                return item.to(device)
            if isinstance(item, tuple):
                return tuple(move(x) for x in item)
            return item

        # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
        return tuple(move(x) for x in args), {name: move(x) for name, x in kwargs.items()}

    handles = [layer.register_forward_pre_hook(dispatch, with_kwargs=True) for layer in text.layers]
    # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
    handles.append(target.lm_head.register_forward_pre_hook(dispatch, with_kwargs=True))
    # 保护区：执行可能失败的步骤；异常处理或 finally 负责恢复状态/释放资源。
    try:
        return target(input_ids=tokens, use_cache=False).logits
    finally:
        for handle in handles:
            # 移除临时 hook，避免后续 forward 继续受到本次参考检查影响。
            handle.remove()


# ── HybridHistory ──
# 职责：类型 `HybridHistory`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class HybridHistory(tuple):
    """Owned AR snapshots: full-attention KV or GDN convolution/recurrent state."""

    # ── HybridHistory.__new__ ──
    # 职责：创建带 length 属性的 tuple 历史容器，记录每层状态及严格历史长度。
    # 输入提示：states：逐层状态列表；Full Attention 与 GDN 每项的含义不同。
    def __new__(cls, states, length):
        obj = super().__new__(cls, states)
        obj.length = length
        return obj


# ── _ScratchCache ──
# 职责：类型 `_ScratchCache`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class _ScratchCache:
    """Minimal mixer cache; writes never reach the supplied AR snapshots."""

    # ── _ScratchCache.__init__ ──
    # 职责：建立每层临时工作行；Full Attention 引用只读历史后以新张量追加，GDN 先 clone 避免原地算子污染历史。
    # 输入提示：states：逐层状态列表；Full Attention 与 GDN 每项的含义不同。
    def __init__(self, states, layer_types):
        self.layers = []
        for state, kind in zip(states, layer_types, strict=True):
            # 数据：某层工作缓存行，或输入数据记录，具体由所在模块决定。
            row = SimpleNamespace(keys=None, values=None, conv_states=None, recurrent_states=None)
            if state is not None:
                if kind == "full_attention":
                    # 数据：某层工作缓存行，或输入数据记录，具体由所在模块决定。
                    row.keys, row.values = state
                else:
                    # Existing single-token GDN kernels may update these in place.
                    # 数据：某层工作缓存行，或输入数据记录，具体由所在模块决定。 此处新建独立张量存储；与只建立别名不同。
                    row.conv_states = (state[0].clone(),)
                    # 数据：某层工作缓存行，或输入数据记录，具体由所在模块决定。 此处新建独立张量存储；与只建立别名不同。
                    row.recurrent_states = (state[1].clone(),)
            # 把当前项加入列表/缓冲，供后续统一处理；这里尚未执行优化器更新。
            self.layers.append(row)

    # ── _ScratchCache.has_previous_state ──
    # 职责：检查该 GDN 层是否已有递推初态，决定 mixer 是否使用历史。
    def has_previous_state(self, layer_idx):
        return self.layers[layer_idx].recurrent_states is not None

    # ── _ScratchCache.update ──
    # 职责：将当前层临时 K/V 与历史沿序列维拼接，替换工作行引用；不是对主模型持久缓存提交。
    # 输入提示：key：K 特征；历史与当前块可能同时存在，不是字典 key 的统一含义；value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。
    def update(self, key, value, layer_idx, *args, **kwargs):
        # 数据：某层工作缓存行，或输入数据记录，具体由所在模块决定。
        row = self.layers[layer_idx]
        if row.keys is not None:
            # 数据：K 特征；历史与当前块可能同时存在，不是字典 key 的统一含义。 此处沿指定维拼接；是新结果，不等于向原生持久 cache 提交。
            key = torch.cat((row.keys, key), dim=2)
            # 数据：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义。 此处沿指定维拼接；是新结果，不等于向原生持久 cache 提交。
            value = torch.cat((row.values, value), dim=2)
        # 数据：某层工作缓存行，或输入数据记录，具体由所在模块决定。
        row.keys, row.values = key, value
        # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
        return key, value

    # ── _ScratchCache.update_conv_state ──
    # 职责：仅更新 _ScratchCache 中该层临时卷积状态，不回写主模型历史。
    def update_conv_state(self, state, layer_idx, **kwargs):
        self.layers[layer_idx].conv_states = (state,)
        return state

    # ── _ScratchCache.update_recurrent_state ──
    # 职责：仅更新 _ScratchCache 中该层临时递推状态，不回写主模型历史。
    def update_recurrent_state(self, state, layer_idx, **kwargs):
        self.layers[layer_idx].recurrent_states = (state,)
        return state

    # ── _ScratchCache.snapshot ──
    # 职责：将工作 cache 复制/脱离计算图为新的 AR 历史快照；draft forward 不提交其临时 cache。
    def snapshot(self, length, layer_types):
        # 数据：逐层状态列表；Full Attention 与 GDN 每项的含义不同。
        states = []
        for row, kind in zip(self.layers, layer_types, strict=True):
            pair = ((row.keys, row.values) if kind == "full_attention"
                    else (row.conv_states[0], row.recurrent_states[0]))
            # 执行：新建独立张量存储；与只建立别名不同；断开旧 autograd 计算图，不自动复制底层存储。
            states.append(tuple(x.detach().clone() for x in pair))
        return HybridHistory(states, length)


# ── _HybridLayer ──
# 职责：类型 `_HybridLayer`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class _HybridLayer(nn.Module):
    # ── _HybridLayer.__init__ ──
    # 职责：引用共享 norm/MLP，保留 AR mixer；clone mixer 参数生成独立 draft，Full Attention 显式关闭 causal。
    def __init__(self, original, kind):
        super().__init__()
        self.kind = kind
        mixer = original.self_attn if kind == "full_attention" else original.linear_attn
        self.attention = nn.Module()
        self.attention.ar = mixer
        # Draft 只保留普通参数，不复制 vLLM 分片参数类型及其加载器属性。
        # 数据：优化器管理的 FP32 master 参数列表，不是所有冻结 backbone 参数。 此处新建独立张量存储；与只建立别名不同。
        parameters = {id(p): nn.Parameter(p.detach().clone(), requires_grad=False) for p in mixer.parameters()}
        self.attention.draft = copy.deepcopy(mixer, parameters)
        if kind == "full_attention":
            # 无 padding 的 draft 块全可见；None mask 也必须显式关闭 SDPA causal。
            self.attention.draft.is_causal = False
        self.input_norm = original.input_layernorm
        self.post_norm = original.post_attention_layernorm
        self.mlp = original.mlp

    # ── _HybridLayer.forward ──
    # 职责：按 view 选 AR/draft mixer，执行 norm→mixer→残差→norm→共享 MLP→残差；输入/输出 [1,K,H]。
    # 输入提示：hidden：层间浮点特征，常见形状 [batch,sequence,hidden_size]，不是 token ID；allowed：Attention 可见范围；AR 屏蔽未来，单个 draft 块全可见。
    # 调用者：Qwen38DualView.forward / forward_suffix 的逐层执行。
    # 输入：hidden [1,K,H]、view、RoPE、allowed 和临时 cache。
    # 输出：本层特征 [1,K,H]；这里还没有执行最终 LM head。
    def forward(self, hidden, view, rotary, allowed, cache):
        device = self.input_norm.weight.device
        # 数据：层间浮点特征，常见形状 [batch,sequence,hidden_size]，不是 token ID。 此处对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
        hidden = hidden.to(device)
        # 计算：对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
        rotary = tuple(x.to(device) for x in rotary)
        # 数据：Attention 可见范围；AR 屏蔽未来，单个 draft 块全可见。 此处对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
        allowed = None if allowed is None else allowed.to(device)
        # 数据：归一化后的浮点特征，送入 mixer 或 MLP。
        normalized = self.input_norm(hidden)
        mixer = getattr(self.attention, view)
        if self.kind == "full_attention":
            mixed, _ = mixer(normalized, position_embeddings=rotary,
                             attention_mask=allowed, past_key_values=cache)
        else:
            mixed = mixer(normalized, cache_params=cache, attention_mask=None)
        # 数据：层间浮点特征，常见形状 [batch,sequence,hidden_size]，不是 token ID。
        hidden = hidden + mixed
        return hidden + self.mlp(self.post_norm(hidden))


# ── Qwen38DualView ──
# 职责：类型 `Qwen38DualView`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class Qwen38DualView(nn.Module):
    """Shared frozen backbone, independent draft mixers, no second full model.

    This is an eager reference executor, not yet a vLLM paged-cache adapter.
    Mask token and block size are explicit; neither is guessed from the name.
    """

    # ── Qwen38DualView.__init__ ──
    # 职责：解析文本模型配置，构造每层 AR/draft mixer 和共享模块；冻结执行视图参数，训练时由 master 函数式替换。
    def __init__(self, target, *, block_size, mask_token_id, hybrid_policy):
        super().__init__()
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：explicit orthrus_qwen35_reference hybrid policy required。
        if hybrid_policy != "orthrus_qwen35_reference":
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("explicit orthrus_qwen35_reference hybrid policy required")
        text = getattr(target.model, "language_model", target.model)
        config = copy.deepcopy(text.config)
        kinds = tuple(config.layer_types)
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：Qwen3.5/3.8 text hybrid architecture required。
        if (config.model_type != "qwen3_5_text" or len(kinds) != config.num_hidden_layers
                or set(kinds) - {"linear_attention", "full_attention"}):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("Qwen3.5/3.8 text hybrid architecture required")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：explicit valid block size and mask token required。
        if (type(block_size) is not int or block_size < 2 or type(mask_token_id) is not int
                or not 0 <= mask_token_id < config.vocab_size):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("explicit valid block size and mask token required")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：dispatched/offloaded models need a separate device-placement adapter。
        if getattr(target, "hf_device_map", None):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("dispatched/offloaded models need a separate device-placement adapter")
        # 数据：优化器管理的 FP32 master 参数列表，不是所有冻结 backbone 参数。
        parameters = list(text.parameters()) + list(target.lm_head.parameters())
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：single-device BF16/FP32 text weights required; INT8 is inference-only here。
        if (len({p.device for p in parameters}) != 1 or any(
                p.dtype not in (torch.float32, torch.bfloat16) for p in parameters)):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("single-device BF16/FP32 text weights required; INT8 is inference-only here")
        config.block_size, config.mask_token_id = block_size, mask_token_id
        self.config = config
        self.layer_types = kinds
        self.hybrid_policy = hybrid_policy
        self.embedding, self.norm, self.head = text.embed_tokens, text.norm, target.lm_head
        self.rotary = text.rotary_emb
        self.layers = nn.ModuleList(_HybridLayer(layer, kind)
                                    for layer, kind in zip(text.layers, kinds, strict=True))
        self.source = {}
        # 调用：配置双视图模型的 Attention 实现选择，保持显式后端约定。
        self.set_backend("sdpa")
        self.eval().requires_grad_(False)

    # ── Qwen38DualView.set_backend ──
    # 职责：配置双视图模型的 Attention 实现选择，保持显式后端约定。
    def set_backend(self, backend):
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：hybrid reference supports sdpa or eager。
        if backend not in {"sdpa", "eager"}:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("hybrid reference supports sdpa or eager")
        self.backend = backend
        for layer in self.layers:
            if layer.kind == "full_attention":
                for view in ("ar", "draft"):
                    getattr(layer.attention, view).config._attn_implementation = backend

    # ── Qwen38DualView._apply ──
    # 职责：适配 nn.Module 设备/精度变换，同时维护共享对象与源模型引用。
    def _apply(self, fn, recurse=True):
        # RoPE frequencies must not lose precision when execution weights cast.
        # 计算：新建独立张量存储；与只建立别名不同；断开旧 autograd 计算图，不自动复制底层存储。
        frequencies = {name: getattr(self.rotary, name).detach().clone()
                       for name in ("inv_freq", "original_inv_freq")
                       if isinstance(getattr(self.rotary, name, None), torch.Tensor)}
        # 数据：被调用步骤返回的结果；更新路径通常含 loss、版本、梯度范数和实际更新层。
        result = super()._apply(fn, recurse)
        for name, value in frequencies.items():
            # 执行：对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
            setattr(self.rotary, name, value.to(device=self.rotary.inv_freq.device, dtype=torch.float32))
        # 输出：被调用步骤返回的结果；更新路径通常含 loss、版本、梯度范数和实际更新层。
        return result

    # ── Qwen38DualView.place_layers ──
    # 职责：独立执行器的按层设备放置；它不是 vLLM 的 tensor parallel。
    def place_layers(self, devices):
        """Contiguous layer placement for low-throughput reference validation.

        This is not tensor parallelism. Embedding and LM head stay together so
        the existing proposal/correction loop keeps a single token/logit device.
        """
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：one to num_hidden_layers devices required。
        if not devices or len(devices) > len(self.layers):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("one to num_hidden_layers devices required")
        devices = [torch.device(device) for device in devices]
        # 执行：对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
        self.embedding.to(devices[0])
        # 执行：对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
        self.head.to(devices[0])
        # 执行：对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
        self.rotary.to(devices[0])
        for i, layer in enumerate(self.layers):
            # 执行：对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
            layer.to(devices[i * len(devices) // len(self.layers)])
        # 执行：对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
        self.norm.to(devices[-1])
        return self

    # ── Qwen38DualView._logits ──
    # 职责：执行最终 norm 与 LM head；可传局部词表 head 供训练减少通信。
    # 输入提示：hidden：层间浮点特征，常见形状 [batch,sequence,hidden_size]，不是 token ID。
    def _logits(self, hidden, head=None):
        # 数据：归一化后的浮点特征，送入 mixer 或 MLP。 此处对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
        normalized = self.norm(hidden.to(self.norm.weight.device))
        return (self.head if head is None else head)(normalized.to(self.head.weight.device))

    # ── Qwen38DualView._cache ──
    # 职责：按层类型建立临时工作 cache，可从后缀起始层选择对应历史。
    def _cache(self, cache, start=0):
        if cache is None:
            # 数据：逐层状态列表；Full Attention 与 GDN 每项的含义不同。
            states = [None] * len(self.layers)
        else:
            # 约束检查：下方条件表示不支持的输入/状态；触发时报错：one history state per executed layer required。
            if len(cache) != len(self.layers) - start:
                # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
                raise ValueError("one history state per executed layer required")
            # 数据：逐层状态列表；Full Attention 与 GDN 每项的含义不同。
            states = [None] * start + list(cache)
        return _ScratchCache(states, self.layer_types)

    # ── Qwen38DualView.forward ──
    # 职责：输入 [1,K] token 与 HybridHistory；遍历混合层，按 capture_layer 保存 detached boundary，返回 hidden/cache/logits/boundary。
    # 调用者：proposer 的 draft 前向，以及离线教师的 AR 视图。
    # 输入：tokens [1,K]，view=ar/draft，HybridHistory 或 None，可选捕获层索引。
    # 输出：BackboneOutput(hidden,cache,logits,boundary)；draft 的 cache 返回 None，临时状态不提交。
    def forward(self, tokens, *, view="ar", cache=None, capture_layer=None,
                logits_to_keep=0):
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：nonempty batch-one token block and ar/draft view required。
        if (view not in {"ar", "draft"} or tokens.ndim != 2 or tokens.shape[0] != 1
                or tokens.shape[1] < 1):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("nonempty batch-one token block and ar/draft view required")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：hybrid AR history required, not an ordinary KV tuple。
        if cache is not None and not isinstance(cache, HybridHistory):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise TypeError("hybrid AR history required, not an ordinary KV tuple")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：valid draft capture layer required。
        if capture_layer is not None and (view != "draft" or type(capture_layer) is not int
                                          or not 0 <= capture_layer < len(self.layers)):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("valid draft capture layer required")
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：nonnegative logits_to_keep required。
        if type(logits_to_keep) is not int or logits_to_keep < 0:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("nonnegative logits_to_keep required")
        # 数据：anchor 前的严格历史长度；绝对位置与缓存读取边界以它为准。
        past = 0 if cache is None else cache.length
        # 数据：层间浮点特征，常见形状 [batch,sequence,hidden_size]，不是 token ID；Attention 可见范围；AR 屏蔽未来，单个 draft 块全可见。
        hidden, rotary, allowed = self._block_inputs(tokens, past, view)
        # 数据：本次前向的临时 cache；draft 内部更新不允许污染目标历史。
        scratch = self._cache(cache)
        # 数据：冻结前段与待训练后缀的交界，包含 hidden、位置/mask、起始层等。
        boundary = None
        for index, layer in enumerate(self.layers):
            if index == capture_layer:
                # 数据：冻结前段与待训练后缀的交界，包含 hidden、位置/mask、起始层等。
                boundary = DraftBoundary(hidden, rotary, allowed, index, past, self.backend).detached()
            # 数据：层间浮点特征，常见形状 [batch,sequence,hidden_size]，不是 token ID。
            hidden = layer(hidden, view, rotary, allowed, scratch)
        # 数据：词表未归一化分数；最后一维 V，起草有效行通常是前 K−1 行。
        logits = self._logits(hidden[:, -logits_to_keep:] if logits_to_keep else hidden)
        # 数据：各层已提交 AR 历史；Full Attention 是 K/V，GDN 是卷积/递推状态。
        history = scratch.snapshot(past + tokens.shape[1], self.layer_types) if view == "ar" else None
        return BackboneOutput(hidden, history, logits, boundary)

    # ── Qwen38DualView._block_inputs ──
    # 职责：token embedding 和绝对位置/RoPE；AR 构造因果 mask，draft Full Attention 使用全块可见。
    # 输入提示：past：anchor 前的严格历史长度；绝对位置与缓存读取边界以它为准。
    def _block_inputs(self, tokens, past, view):
        # 数据：层间浮点特征，常见形状 [batch,sequence,hidden_size]，不是 token ID。
        hidden = self.embedding(tokens)
        # 数据：每个输入位置的绝对 token 位置，包含历史长度偏移，用于 RoPE。
        positions = torch.arange(past, past + tokens.shape[1], device=tokens.device)[None]
        rotary = self.rotary(hidden, positions[None].expand(3, -1, -1))
        # Orthrus: every draft query sees the strict AR prefix and the whole
        # temporary block. AR instead uses each row's own causal boundary.
        # 数据：Attention 可见范围；AR 屏蔽未来，单个 draft 块全可见。
        allowed = None
        # 视图选路：AR 使用因果教师/历史提交语义，draft 使用独立 mixer 和临时块语义。
        if view == "ar":
            # 数据：Attention 可见范围；AR 屏蔽未来，单个 draft 块全可见。
            allowed = hidden.new_zeros((1, 1, tokens.shape[1], past + tokens.shape[1]))
            future = torch.arange(past + tokens.shape[1], device=tokens.device)[None] > positions[0, :, None]
            # 执行：转为 FP32 做分数/归一化/梯度等中间计算。
            allowed.masked_fill_(future, float("-inf"))
        # 输出按 tuple 顺序对应下方各项；调用方通过解包接收，不能交换次序。
        return hidden, rotary, allowed

    @torch.no_grad()
    # ── Qwen38DualView.draft_boundary ──
    # 职责：只执行冻结前段，得到 capture_layer 入口中间特征及位置/mask，供后缀训练重放。
    # 调用者：offline_feedback；在线路径则通过 forward(capture_layer=...) 捕获边界。
    # 输出：DraftBoundary，保存最后 N 层入口的中间特征/位置/可见性，训练时从这里重放。
    def draft_boundary(self, tokens, cache=None):
        """Offline full-layer replay starts at embeddings; no redundant draft forward."""
        # 数据：anchor 前的严格历史长度；绝对位置与缓存读取边界以它为准。
        past = 0 if cache is None else cache.length
        return DraftBoundary(*self._block_inputs(tokens, past, "draft"), 0, past, self.backend).detached()

    # ── Qwen38DualView.forward_suffix ──
    # 职责：从 boundary.hidden 开始执行待训练后缀；用函数式参数替换注入 master，可选逐层 checkpoint 和局部词表 head。
    # 输入提示：boundary：冻结前段与待训练后缀的交界，包含 hidden、位置/mask、起始层等。
    # 调用者：SuffixLearner.replay_logits；不是正常起草入口。
    # 输入：已 detach 的 boundary、该后缀历史、master 参数字典；不重新做 embedding 或冻结前缀。
    # 输出：后缀学生 logits 及相关输出；梯度通过 functional_call 回到 master，不回到旧 boundary 的前段计算。
    def forward_suffix(self, boundary, *, cache=None, logit_range=None, draft_weights=None,
                       checkpoint_layers=False, logit_head=None):
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：compatible captured draft boundary required。
        if (not isinstance(boundary, DraftBoundary) or boundary.backend != self.backend
                or not 0 <= boundary.start_layer < len(self.layers)):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("compatible captured draft boundary required")
        start = boundary.start_layer
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：one history state per executed layer required。
        if cache is not None and len(cache) != len(self.layers) - start:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("one history state per executed layer required")
        overrides = {} if draft_weights is None else draft_weights
        permitted = {f"layers.{i}.attention.draft.{name}" for i in range(start, len(self.layers))
                     for name, _ in self.layers[i].attention.draft.named_parameters()}
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：only suffix draft parameters may be overridden。
        if set(overrides) - permitted:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("only suffix draft parameters may be overridden")
        # 数据：层间浮点特征，常见形状 [batch,sequence,hidden_size]，不是 token ID。
        hidden = boundary.hidden
        for i in range(start, len(self.layers)):
            prefix = f"layers.{i}."
            weights = {name[len(prefix):]: value for name, value in overrides.items() if name.startswith(prefix)}

            # ── Qwen38DualView.forward_suffix.execute ──
            # 职责：执行单个待训练层；将对应 master 转为计算 dtype/设备，用 functional_call 临时替换本层 draft 参数。
            # 输入提示：value：V/输入浮点特征；与 token ID 不同，具体形状由调用函数定义；parameters：优化器管理的 FP32 master 参数列表，不是所有冻结 backbone 参数。
            def execute(value, index=i, parameters=weights):
                layer = self.layers[index]
                resident = dict(layer.named_parameters())
                # 计算：对齐设备或 dtype；设备变化是数据搬运，类型变化可能引入舍入。
                staged = {name: weight.to(device=resident[name].device, dtype=resident[name].dtype)
                          for name, weight in parameters.items()}
                # Recompute must start from the same clean state, not the first forward's tail.
                # 数据：逐层状态列表；Full Attention 与 GDN 每项的含义不同。
                states = [None] * len(self.layers)
                if cache is not None:
                    # 数据：逐层状态列表；Full Attention 与 GDN 每项的含义不同。
                    states[index] = cache[index - start]
                # 数据：本次前向的临时 cache；draft 内部更新不允许污染目标历史。
                scratch = _ScratchCache(states, self.layer_types)
                args = (value, "draft", boundary.rotary, boundary.allowed, scratch)
                return torch.func.functional_call(layer, staged, args) if staged else layer(*args)

            # 数据：层间浮点特征，常见形状 [batch,sequence,hidden_size]，不是 token ID。 此处前向少存激活，反向时重算，换取显存节省。
            hidden = (checkpoint(execute, hidden, use_reentrant=False)
                      if checkpoint_layers and torch.is_grad_enabled() else execute(hidden))
        projected = hidden if logit_range is None else hidden[:, slice(*logit_range)]
        return BackboneOutput(hidden, None, self._logits(projected, logit_head))


# ── Qwen38Branch ──
# 职责：类型 `Qwen38Branch`：封装下方方法及状态；请先读 __init__，再沿调用者进入 forward/执行方法。
class Qwen38Branch(MaskedAttentionBranch):
    """Use the same proposals, exact correction and reached-row feedback loop.

    A recurrent state cannot be cropped like KV. After rejection we replay only
    accepted AR inputs from the pre-verification snapshot. This adds work, but
    is a correctness baseline for a later vLLM state-checkpoint implementation.
    """

    @staticmethod
    # ── Qwen38Branch.history_length ──
    # 职责：读取混合历史的 token 长度；不能用 cache 张量某一维统一推断 GDN 与 KV。
    def history_length(cache):
        return 0 if cache is None else cache.length

    # ── Qwen38Branch.commit_history ──
    # 职责：独立参考生成器按接受前缀提交教师 AR 历史；vLLM 原生提交路径不调用这个参考分支方法。
    def commit_history(self, proposal, verified_cache, committed):
        previous = self.history_length(proposal.cache)
        # 数据：本轮有效 token/候选数量；以赋值来源和所在函数为准，不等同于固定 K。
        count = committed - previous
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：committed AR inputs must be a nonempty verified prefix。
        if not 0 < count <= proposal.verifier_inputs.shape[1]:
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("committed AR inputs must be a nonempty verified prefix")
        if committed == verified_cache.length:
            return verified_cache
        _, clean = self.ar(proposal.verifier_inputs[:, :count], cache=proposal.cache, logits_to_keep=1)
        return clean

    # ── Qwen38Branch.audit_history ──
    # 职责：独立参考分支的历史正确性审计，比较分段生成与参考历史。
    def audit_history(self, actual, tokens, chunks):
        # BF16 recurrent continuation and full recomputation have different
        # rounding paths. Match AR-only chunk boundaries to isolate corruption.
        # 约束检查：下方条件表示不支持的输入/状态；触发时报错：audit chunks must partition the committed prefix。
        if sum(chunks) != tokens.shape[1] or any(length < 1 for length in chunks):
            # 边界保护：拒绝不满足约定的输入/配置/状态，避免继续执行后静默产生错误结果。
            raise ValueError("audit chunks must partition the committed prefix")
        # 数据：从原生卷积工作缓存选择接受前缀尾部的起点。
        reference, offset = None, 0
        for length in chunks:
            _, reference = self.ar(tokens[:, offset:offset + length], cache=reference, logits_to_keep=1)
            # 状态推进：在原有值上累加/更新；若目标是张量，要注意这里的原地语义。
            offset += length
        for index, (pair, other) in enumerate(zip(actual, reference, strict=True)):
            for component, (value, expected) in enumerate(zip(pair, other, strict=True)):
                torch.testing.assert_close(value, expected, rtol=0, atol=0,
                                           msg=lambda message, i=index, c=component: f"layer {i} state {c}: {message}")
