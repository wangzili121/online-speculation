# Qwen3.8 函数索引

本索引覆盖学习分支中 19 个已注释核心文件。点击函数名可跳到对应源码。参数为真实 Python 签名；调用线索仅为函数本体中直接出现的调用，不展开内部闭包、外部库、hook 或动态方法分派。实际执行顺序请配合[主导读](QWEN38_CODE_WALKTHROUGH.md)。

`self` 是当前实例，`cls` 是当前类型；`*args`/`**kwargs` 是可变参数。相同方法名必须结合所在类判断职责。

## scripts/prepare_qwen38_data.py

数据准备入口：将语料编码为训练脚本读取的 token ID 记录；数据准备与模型参数更新分开。

| 函数入口 | 职责与输入 | 直接调用线索 |
| --- | --- | --- |
| [`sha256`](../scripts/prepare_qwen38_data.py#L21) | 计算文件内容 SHA256，用于记录数据身份；不是模型前向或 loss。<br/>参数：`path` | `Path(path).open()`<br/>`Path()`<br/>`hashlib.file_digest(stream, 'sha256').hexdigest()`<br/>`hashlib.file_digest()` |
| [`rows`](../scripts/prepare_qwen38_data.py#L28) | 按批次读取 Parquet，或逐行解析 JSONL，yield 原始语料记录；不会一次读入全部数据。<br/>参数：`path` | `pq.ParquetFile(path).iter_batches()`<br/>`pq.ParquetFile()`<br/>`batch.to_pylist()`<br/>`path.open()`<br/>`line.strip()`<br/>`json.loads()` |
| [`tokenize`](../scripts/prepare_qwen38_data.py#L46) | 文本使用 tokenizer.encode，对话使用 apply_chat_template；输出 token ID 序列，不是 embedding。<br/>参数：`row, tokenizer` | `isinstance()`<br/>`ValueError()`<br/>`tokenizer.apply_chat_template()`<br/>`row.get()`<br/>`tokenizer.encode()` |
| [`prepare`](../scripts/prepare_qwen38_data.py#L63) | 逐条读取语料、分词、截断并过滤短样本，写入新的 JSONL 和来源摘要。<br/>参数：`source, output, tokenizer, *, min_tokens, max_tokens, max_records` | `output.open()`<br/>`enumerate()`<br/>`rows()`<br/>`tokenize()`<br/>`len()`<br/>`stream.write()`<br/>`json.dumps()`<br/>`dict()`<br/>`selected.append()`<br/>`output.unlink()`；另有 2 项，见源码 |
| [`main`](../scripts/prepare_qwen38_data.py#L97) | 解析数据/模型路径与长度限制，加载本地 tokenizer，调用 prepare 生成训练输入。<br/>参数：`` | `argparse.ArgumentParser()`<br/>`parser.add_argument()`<br/>`parser.parse_args()`<br/>`parser.error()`<br/>`AutoTokenizer.from_pretrained()`<br/>`args.output.parent.mkdir()`<br/>`prepare()`<br/>`manifest.update()`<br/>`str()`<br/>`hashlib.sha256()`；另有 6 项，见源码 |

## scripts/run_qwen38_vllm.py

服务入口：解析命令行，拼出原生模型与 custom_class 配置，再执行 vllm serve。它不实现模型 forward。

| 函数入口 | 职责与输入 | 直接调用线索 |
| --- | --- | --- |
| [`prepare_runtime_environment`](../scripts/run_qwen38_vllm.py#L23) | 补齐 CANN Python 搜索路径，让后续进程能导入运行时绑定。<br/>参数：`` | `os.environ.get()`<br/>`str()`<br/>`Path()`<br/>`Path(site).is_dir()`<br/>`sys.path.append()`<br/>`os.environ.get('PYTHONPATH', '').split()`<br/>`os.pathsep.join()` |
| [`arguments`](../scripts/run_qwen38_vllm.py#L39) | 读取命令行选项并做基础合法性检查；返回 argparse.Namespace。<br/>参数：`parser=None, *, defaults=None` | `argparse.ArgumentParser()`<br/>`parser.add_argument()`<br/>`parser.set_defaults()`<br/>`parser.parse_args()`<br/>`parser.error()` |
| [`native_options`](../scripts/run_qwen38_vllm.py#L72) | 生成原生 vLLM 主模型配置：TP、eager、单请求、KV 预算与 worker 扩展类。<br/>参数：`args` | `dict()`<br/>`getattr()` |
| [`speculative_options`](../scripts/run_qwen38_vllm.py#L86) | 声明 custom_class proposer 路径及 K−1 个候选数，把训练/head 选项放入 additional_config。<br/>参数：`args` | `dict()`<br/>`getattr()` |
| [`main`](../scripts/run_qwen38_vllm.py#L105) | 准备 CANN 环境，解析参数，组合原生/投机配置，构造 vllm serve 命令并 exec 启动服务进程。<br/>参数：`` | `prepare_runtime_environment()`<br/>`arguments()`<br/>`native_options()`<br/>`speculative_options()`<br/>`settings.pop()`<br/>`str()`<br/>`settings.items()`<br/>`name.replace()`<br/>`isinstance()`<br/>`command.append()`；另有 3 项，见源码 |

## scripts/train_qwen38_draft.py

离线全层训练入口：读取 tokenized JSONL，抽样 anchor，通过 worker RPC 驱动蒸馏与导出；不是对话请求训练。

| 函数入口 | 职责与输入 | 直接调用线索 |
| --- | --- | --- |
| [`main`](../scripts/train_qwen38_draft.py#L26) | 读取 tokenized 数据，创建 offline LLM worker，逐步 RPC 全层蒸馏，保存 loss.jsonl 并导出 head。<br/>参数：`` | `prepare_runtime_environment()`<br/>`argparse.ArgumentParser()`<br/>`parser.add_argument()`<br/>`arguments()`<br/>`dict()`<br/>`parser.error()`<br/>`args.output.mkdir()`<br/>`AutoConfig.from_pretrained()`<br/>`getattr()`<br/>`TokenDataset()`；另有 28 项，见源码 |

## src/blockspec/feedback.py

训练反馈数据结构：把 MASK 输入、历史、教师 logits、有效位置与冻结边界打包，供 learner 重放。

| 函数入口 | 职责与输入 | 直接调用线索 |
| --- | --- | --- |
| [`ReplayBoundary.detached`](../src/blockspec/feedback.py#L31) | 断开旧计算图，并按需要保留 cache 后缀和复制反馈张量；之后训练会新建后缀计算图。<br/>参数：`self` | 无直接函数调用 |
| [`Feedback.detached`](../src/blockspec/feedback.py#L53) | 克隆输入/教师 logits 并 detach cache 与 boundary，保留从 cache_start 起的层历史，供后缀重放。<br/>参数：`self, *, cache_start=0` | `tuple()`<br/>`k.detach()`<br/>`v.detach()`<br/>`Feedback()`<br/>`self.inputs.detach().clone()`<br/>`self.inputs.detach()`<br/>`self.teacher_logits.detach().clone()`<br/>`self.teacher_logits.detach()`<br/>`self.boundary.detached()` |

## src/blockspec/parallel/backbone.py

公共模型输出与边界类型，以及普通 Transformer 参考 backbone；Qwen 主要复用 BackboneOutput / DraftBoundary。

| 函数入口 | 职责与输入 | 直接调用线索 |
| --- | --- | --- |
| [`DualViewConfig.__post_init__`](../src/blockspec/parallel/backbone.py#L62) | 数据类构造后校验配置，拒绝无法支持的参数组合。<br/>参数：`self` | `any()`<br/>`ValueError()`<br/>`math.isfinite()` |
| [`DualViewConfig.to_dict`](../src/blockspec/parallel/backbone.py#L93) | 把参考模型配置转为可保存的字典。<br/>参数：`self` | `asdict()` |
| [`DualViewConfig.from_public`](../src/blockspec/parallel/backbone.py#L99) | 从外部模型配置抽取普通参考 backbone 所需的配置字段。<br/>参数：`cls, data` | `data.get()`<br/>`ValueError()`<br/>`any()`<br/>`rope.get()`<br/>`set()`<br/>`values.keys()`<br/>`sorted()`<br/>`cls()` |
| [`RMSNorm.__init__`](../src/blockspec/parallel/backbone.py#L135) | 初始化此类持有的参数、配置或状态；这里只构造对象，是否执行计算看具体语句。<br/>参数：`self, width, epsilon` | `super().__init__()`<br/>`super()`<br/>`nn.Parameter()`<br/>`torch.ones()` |
| [`RMSNorm.forward`](../src/blockspec/parallel/backbone.py#L143) | 普通参考 backbone 的 RMSNorm；Qwen native 使用 qwen38_weights.SharedRMSNorm。<br/>参数：`self, hidden` | `hidden.float()`<br/>`torch.rsqrt()`<br/>`work.square().mean()`<br/>`work.square()`<br/>`normalized.to()` |
| [`AttentionView.__init__`](../src/blockspec/parallel/backbone.py#L158) | 初始化此类持有的参数、配置或状态；这里只构造对象，是否执行计算看具体语句。<br/>参数：`self, config` | `super().__init__()`<br/>`super()`<br/>`nn.Linear()`<br/>`RMSNorm()` |
| [`_rotate`](../src/blockspec/parallel/backbone.py#L173) | 执行 RoPE 所需的向量分半旋转；不改变 token 的序列顺序。<br/>参数：`x, cosine, sine` | `x.chunk()`<br/>`torch.cat()` |
| [`DualAttention.__init__`](../src/blockspec/parallel/backbone.py#L183) | 初始化此类持有的参数、配置或状态；这里只构造对象，是否执行计算看具体语句。<br/>参数：`self, config` | `super().__init__()`<br/>`super()`<br/>`AttentionView()` |
| [`DualAttention.forward`](../src/blockspec/parallel/backbone.py#L192) | 普通全 Attention 参考模型的 AR/draft QKV 与可见性计算；不是 Qwen GDN。<br/>参数：`self, hidden, view, cache, rotary, allowed, backend` | `getattr()`<br/>`projection.q_norm()`<br/>`projection.q(hidden).view()`<br/>`projection.q()`<br/>`projection.k_norm()`<br/>`projection.k(hidden).view()`<br/>`projection.k()`<br/>`projection.v(hidden).view(b, length, -1, config.head_dim).transpose()`<br/>`projection.v(hidden).view()`<br/>`projection.v()`；另有 19 项，见源码 |
| [`DualLayer.__init__`](../src/blockspec/parallel/backbone.py#L249) | 初始化此类持有的参数、配置或状态；这里只构造对象，是否执行计算看具体语句。<br/>参数：`self, config` | `super().__init__()`<br/>`super()`<br/>`DualAttention()`<br/>`RMSNorm()`<br/>`nn.Linear()` |
| [`DualLayer.forward`](../src/blockspec/parallel/backbone.py#L262) | 普通参考 block 的 norm、Attention、MLP 与残差计算。<br/>参数：`self, hidden, view, cache, rotary, allowed, backend` | `self.attention()`<br/>`self.input_norm()`<br/>`self.post_norm()`<br/>`self.down()`<br/>`F.silu()`<br/>`self.gate()`<br/>`self.up()` |
| [`DraftBoundary.detached`](../src/blockspec/parallel/backbone.py#L291) | 断开旧计算图，并按需要保留 cache 后缀和复制反馈张量；之后训练会新建后缀计算图。<br/>参数：`self` | `DraftBoundary()`<br/>`self.hidden.detach().clone()`<br/>`self.hidden.detach()`<br/>`tuple()`<br/>`x.detach().clone()`<br/>`x.detach()`<br/>`self.allowed.detach().clone()`<br/>`self.allowed.detach()` |
| [`DualViewDecoder.__init__`](../src/blockspec/parallel/backbone.py#L318) | 初始化此类持有的参数、配置或状态；这里只构造对象，是否执行计算看具体语句。<br/>参数：`self, config=DualViewConfig()` | `DualViewConfig()`<br/>`super().__init__()`<br/>`super()`<br/>`nn.Embedding()`<br/>`nn.ModuleList()`<br/>`DualLayer()`<br/>`range()`<br/>`RMSNorm()`<br/>`nn.Linear()`<br/>`self.register_buffer()`；另有 7 项，见源码 |
| [`DualViewDecoder.tie_weights`](../src/blockspec/parallel/backbone.py#L337) | 普通参考模型中绑定输入 embedding 与输出词表 head 的权重。<br/>参数：`self` | 无直接函数调用 |
| [`DualViewDecoder._frequencies`](../src/blockspec/parallel/backbone.py#L343) | 为普通参考模型构造 RoPE 所需频率/位置旋转信息。<br/>参数：`self, device` | `torch.arange()`<br/>`(1.0 / self.config.rope_theta ** (indices / self.config.head_dim)).to()` |
| [`DualViewDecoder._apply`](../src/blockspec/parallel/backbone.py#L351) | 适配 nn.Module 设备/精度变换，同时维护共享对象与源模型引用。<br/>参数：`self, fn, recurse=True` | `super()._apply()`<br/>`super()`<br/>`self._frequencies()` |
| [`DualViewDecoder.initialize_draft_from_ar`](../src/blockspec/parallel/backbone.py#L360) | 把普通参考模型的 AR Attention 参数复制到 draft 作为初始化；不代表已经蒸馏训练。<br/>参数：`self` | `layer.attention.draft.load_state_dict()`<br/>`layer.attention.ar.state_dict()`<br/>`torch.no_grad()` |
| [`DualViewDecoder.train_draft_only`](../src/blockspec/parallel/backbone.py#L367) | 普通参考模型设置只训练 draft 的参数开关；native learner 则使用自己的 master 参数。<br/>参数：`self` | `self.requires_grad_()`<br/>`layer.attention.draft.requires_grad_()`<br/>`self.train()` |
| [`DualViewDecoder.set_backend`](../src/blockspec/parallel/backbone.py#L375) | 配置双视图模型的 Attention 实现选择，保持显式后端约定。<br/>参数：`self, backend` | `ValueError()` |
| [`DualViewDecoder.forward`](../src/blockspec/parallel/backbone.py#L386) | 普通参考 backbone 的整网前向；Qwen 使用 Qwen38DualView，二者不要混淆。<br/>参数：`self, tokens, *, view='ar', cache=None, positions=None, allowed=None, logits_to_keep=0, compute_logits=True, capture_layer=None` | `ValueError()`<br/>`type()`<br/>`len()`<br/>`cache_length()`<br/>`any()`<br/>`torch.arange()`<br/>`isinstance()`<br/>`self.embedding()`<br/>`positions.float()`<br/>`torch.cat()`；另有 12 项，见源码 |
| [`DualViewDecoder.forward_suffix`](../src/blockspec/parallel/backbone.py#L471) | 从 boundary.hidden 开始执行待训练后缀；用函数式参数替换注入 master，可选逐层 checkpoint 和局部词表 head。<br/>参数：`self, boundary, *, cache=None, logit_range=None, draft_weights=None, checkpoint_layers=False` | `isinstance()`<br/>`len()`<br/>`cache_length()`<br/>`ValueError()`<br/>`any()`<br/>`range()`<br/>`self.layers[i].attention.draft.named_parameters()`<br/>`overrides.keys()`<br/>`enumerate()`<br/>`overrides.items()`；另有 7 项，见源码 |
| [`DualViewDecoder.forward_suffix.execute`](../src/blockspec/parallel/backbone.py#L510) | 实现 DualViewDecoder.forward_suffix.execute；输入见下方签名，具体处理与返回值逐语句标注。<br/>参数：`value, module=layer, parameters=weights, history=None if cache is None else cache[offset]` | `dict()`<br/>`module.named_parameters()`<br/>`weight.to()`<br/>`parameters.items()`<br/>`torch.func.functional_call()`<br/>`module()` |

## src/blockspec/parallel/fitting.py

训练数据及独立训练工具：Qwen vLLM 离线入口主要复用 TokenDataset / BatchStream，文件内其他 fitting 入口不是该服务的必经调用。

| 函数入口 | 职责与输入 | 直接调用线索 |
| --- | --- | --- |
| [`TokenDataset.__init__`](../src/blockspec/parallel/fitting.py#L44) | 扫描 JSONL，验证 token 和合法区间，记录足够长样本的文件偏移及数据指纹，避免把所有文本常驻内存。<br/>参数：`self, path, vocab_size, sequence_length` | `any()`<br/>`type()`<br/>`ValueError()`<br/>`Path(path).resolve()`<br/>`Path()`<br/>`array()`<br/>`self.path.open()`<br/>`handle.tell()`<br/>`handle.readline()`<br/>`line.strip()`；另有 4 项，见源码 |
| [`TokenDataset._decode_record`](../src/blockspec/parallel/fitting.py#L79) | 解析一行 JSONL，校验 token ID 和合法 anchor 区间，拒绝越界/错误数据。<br/>参数：`self, line` | `json.loads()`<br/>`record.get()`<br/>`isinstance()`<br/>`any()`<br/>`type()`<br/>`ValueError()`<br/>`len()`<br/>`sorted()`<br/>`int()`<br/>`zip()` |
| [`TokenDataset._decode`](../src/blockspec/parallel/fitting.py#L114) | 取一条记录中的 token 序列，舍弃附加区间信息的简化接口。<br/>参数：`self, line` | `self._decode_record()` |
| [`TokenDataset.__len__`](../src/blockspec/parallel/fitting.py#L119) | 返回可供训练读取的合格记录数。<br/>参数：`self` | `len()` |
| [`TokenDataset.__getitem__`](../src/blockspec/parallel/fitting.py#L124) | 按文件偏移读取指定样本的 token ID；不做模型计算。<br/>参数：`self, index` | `self.record()` |
| [`TokenDataset.record`](../src/blockspec/parallel/fitting.py#L129) | 按索引读取一条原始合格记录，保留 token ID 与 anchor_ranges。<br/>参数：`self, index` | `self.path.open()`<br/>`handle.seek()`<br/>`self._decode_record()`<br/>`handle.readline()`<br/>`torch.tensor()` |
| [`BatchStream.__init__`](../src/blockspec/parallel/fitting.py#L142) | 创建固定种子的 CPU 数据抽样 RNG、样本排列和游标；数据随机数与 NPU 生成采样 RNG 分开。<br/>参数：`self, dataset, seed` | `torch.Generator().manual_seed()`<br/>`torch.Generator()`<br/>`torch.randperm()`<br/>`len()` |
| [`BatchStream._next_record`](../src/blockspec/parallel/fitting.py#L150) | 按确定性排列取下一条记录，轮次结束后重新洗牌并更新游标。<br/>参数：`self` | `len()`<br/>`torch.randperm()`<br/>`self.dataset.record()`<br/>`int()` |
| [`BatchStream.batch`](../src/blockspec/parallel/fitting.py#L164) | 抽取固定长度 token 窗口，组成 CPU LongTensor batch；适用于不额外限制 anchor 的数据路径。<br/>参数：`self, count` | `range()`<br/>`self._next_record()`<br/>`int()`<br/>`torch.randint()`<br/>`len()`<br/>`rows.append()`<br/>`torch.stack()` |
| [`BatchStream._valid_crop`](../src/blockspec/parallel/fitting.py#L176) | 寻找既容纳完整 token 窗口、又有合法 K-token 预测块的裁剪位置，返回窗口及候选 anchor。<br/>参数：`sequence_length, window, ranges, block_size, rng` | `max()`<br/>`min()`<br/>`intervals.append()`<br/>`sorted()`<br/>`merged.append()`<br/>`int()`<br/>`torch.randint()`<br/>`sum()`<br/>`candidates.append()`<br/>`torch.arange()`；另有 1 项，见源码 |
| [`BatchStream.batch_with_anchor_candidates`](../src/blockspec/parallel/fitting.py#L224) | 抽取文本窗口并返回能容纳完整预测块的 anchor 候选，避免跨不允许的语料边界。<br/>参数：`self, count, block_size` | `type()`<br/>`ValueError()`<br/>`max()`<br/>`len()`<br/>`self._next_record()`<br/>`self._valid_crop()`<br/>`rows.append()`<br/>`candidates.append()`<br/>`torch.stack()` |
| [`BatchStream.state_dict`](../src/blockspec/parallel/fitting.py#L259) | 保存数据 RNG、排列和游标，以便恢复后继续相同的抽样序列。<br/>参数：`self` | `self.rng.get_state()`<br/>`self.order.clone()` |
| [`BatchStream.load_state_dict`](../src/blockspec/parallel/fitting.py#L265) | 校验并恢复数据流状态，不涉及模型权重或 AdamW 状态。<br/>参数：`self, state` | `state['order'].cpu()`<br/>`len()`<br/>`torch.equal()`<br/>`order.sort()`<br/>`torch.arange()`<br/>`type()`<br/>`ValueError()`<br/>`self.rng.set_state()`<br/>`state['rng'].cpu()` |
| [`FitConfig.__post_init__`](../src/blockspec/parallel/fitting.py#L315) | 数据类构造后校验配置，拒绝无法支持的参数组合。<br/>参数：`self` | `any()`<br/>`type()`<br/>`ValueError()`<br/>`math.isfinite()` |
| [`FitConfig.make_optimizer`](../src/blockspec/parallel/fitting.py#L345) | 为独立训练工具配置选定优化器；vLLM native learner 自己构造 AdamW，不经过这里。<br/>参数：`self, parameters` | `torch.optim.AdamW()` |
| [`FitConfig.rate`](../src/blockspec/parallel/fitting.py#L351) | 计算独立训练器当前 step 的学习率调度值。<br/>参数：`self, step` | `max()`<br/>`math.cos()` |
| [`frozen_fingerprint`](../src/blockspec/parallel/fitting.py#L361) | 为独立训练器计算冻结参数内容指纹；不是 native learner 的低开销进程内身份记录。<br/>参数：`model` | `hashlib.sha256()`<br/>`model.named_parameters()`<br/>`digest.update()`<br/>`(name + str(tuple(value.shape)) + str(value.dtype)).encode()`<br/>`str()`<br/>`tuple()`<br/>`memoryview()`<br/>`value.detach().contiguous().cpu().view(torch.uint8).numpy()`<br/>`value.detach().contiguous().cpu().view()`<br/>`value.detach().contiguous().cpu()`；另有 3 项，见源码 |
| [`Trainer.__init__`](../src/blockspec/parallel/fitting.py#L380) | 独立通用离线 Trainer 的初始化；Qwen vLLM 离线 RPC 入口不使用这个 Trainer。<br/>参数：`self, model, data, config=FitConfig()` | `FitConfig()`<br/>`ValueError()`<br/>`any()`<br/>`model.parameters()`<br/>`model.train_draft_only().set_backend()`<br/>`model.train_draft_only()`<br/>`config.make_optimizer()`<br/>`BatchStream()`<br/>`torch.Generator().manual_seed()`<br/>`torch.Generator()`；另有 1 项，见源码 |
| [`Trainer._autocast`](../src/blockspec/parallel/fitting.py#L407) | 选择独立 Trainer 的混合精度上下文。<br/>参数：`self` | `autocast()`<br/>`nullcontext()` |
| [`Trainer._runtime`](../src/blockspec/parallel/fitting.py#L413) | 收集独立 Trainer 的设备、随机状态等恢复信息。<br/>参数：`self` | `str()`<br/>`device_name()`<br/>`torch.get_num_threads()`<br/>`torch.get_float32_matmul_precision()`<br/>`torch.are_deterministic_algorithms_enabled()` |
| [`Trainer.run`](../src/blockspec/parallel/fitting.py#L422) | 独立 Trainer 的训练循环，区别于 train_qwen38_draft.py 的原生 worker RPC 循环。<br/>参数：`self, until=None, progress=None` | `type()`<br/>`ValueError()`<br/>`self.model.train()`<br/>`synchronize()`<br/>`time.perf_counter()`<br/>`self.config.rate()`<br/>`range()`<br/>`self.stream.batch()`<br/>`sample_anchors()`<br/>`batches.append()`；另有 5 项，见源码 |
| [`Trainer.evaluate`](../src/blockspec/parallel/fitting.py#L458) | 在独立 Trainer 中对给定数据估计蒸馏 loss，不做参数更新。<br/>参数：`self, data, batches=4` | `ValueError()`<br/>`BatchStream()`<br/>`torch.Generator().manual_seed()`<br/>`torch.Generator()`<br/>`self.model.eval()`<br/>`range()`<br/>`stream.batch()`<br/>`sample_anchors()`<br/>`self._autocast()`<br/>`distillation_loss()`；另有 8 项，见源码 |
| [`Trainer.save`](../src/blockspec/parallel/fitting.py#L485) | 保存独立 Trainer 的模型、优化器、数据流和运行状态。<br/>参数：`self, path` | `asdict()`<br/>`self.stream.state_dict()`<br/>`self.anchors_rng.get_state()`<br/>`torch.get_rng_state()`<br/>`self._runtime()`<br/>`rng_state()`<br/>`save_checkpoint()` |
| [`Trainer.resume`](../src/blockspec/parallel/fitting.py#L496) | 从独立 Trainer checkpoint 恢复模型、数据和训练进度；不是 native head 的 _load_head。<br/>参数：`cls, path, data, *, device='cpu'` | `load_checkpoint()`<br/>`isinstance()`<br/>`state.get()`<br/>`ValueError()`<br/>`TokenDataset()`<br/>`cls()`<br/>`FitConfig()`<br/>`trainer._runtime()`<br/>`type()`<br/>`trainer.optimizer.load_state_dict()`；另有 6 项，见源码 |

## src/blockspec/parallel/online.py

通用后缀训练器：选中 last-N draft 参数，维护 FP32 master / AdamW / replay，重放反向并发布更新。离线也复用它。

| 函数入口 | 职责与输入 | 直接调用线索 |
| --- | --- | --- |
| [`SuffixConfig.__post_init__`](../src/blockspec/parallel/online.py#L57) | 数据类构造后校验配置，拒绝无法支持的参数组合。<br/>参数：`self` | `any()`<br/>`type()`<br/>`math.isfinite()`<br/>`ValueError()` |
| [`portable`](../src/blockspec/parallel/online.py#L72) | 递归复制训练状态到 CPU 以便保存；这是 checkpoint 序列化，不等于训练时必须 CPU offload。<br/>参数：`value` | `isinstance()`<br/>`value.detach().cpu().clone()`<br/>`value.detach().cpu()`<br/>`value.detach()`<br/>`portable()`<br/>`value.items()`<br/>`type(value)()`<br/>`type()`<br/>`copy.deepcopy()` |
| [`SuffixLearner.__init__`](../src/blockspec/parallel/online.py#L94) | 仅选最后 N 层 .attention.draft. 参数，复制 FP32 master 并创建 AdamW；执行参数保持冻结，训练以函数式替换注入 master。<br/>参数：`self, model, config=SuffixConfig()` | `SuffixConfig()`<br/>`len()`<br/>`any()`<br/>`model.parameters()`<br/>`ValueError()`<br/>`model.named_parameters()`<br/>`name.startswith()`<br/>`int()`<br/>`name.split()`<br/>`next()`；另有 10 项，见源码 |
| [`SuffixLearner.parameter_stamp`](../src/blockspec/parallel/online.py#L128) | 记录参数版本/身份供冻结保护；NativeSuffixLearner 覆盖以兼容 inference tensor。<br/>参数：`parameter` | 无直接函数调用 |
| [`SuffixLearner.fingerprint`](../src/blockspec/parallel/online.py#L133) | 建立冻结部分身份；native 版本使用进程内参数身份，验证脚本的完整内容哈希是另一种更强的审计。<br/>参数：`self` | `hashlib.sha256()`<br/>`self.model.named_parameters()`<br/>`fingerprint.update()`<br/>`name.encode()`<br/>`p.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()`<br/>`p.detach().contiguous().cpu().view(torch.uint8).numpy()`<br/>`p.detach().contiguous().cpu().view()`<br/>`p.detach().contiguous().cpu()`<br/>`p.detach().contiguous()`<br/>`p.detach()`；另有 1 项，见源码 |
| [`SuffixLearner.trainable_parameters`](../src/blockspec/parallel/online.py#L145) | 求选中 FP32 master 参数元素数；不是整个基模参数量。<br/>参数：`self` | `sum()`<br/>`p.numel()` |
| [`SuffixLearner.needs_decoder_feedback`](../src/blockspec/parallel/online.py#L151) | 判断当前 stride 是否需要保存训练 boundary；无需更新的轮次避免额外捕获。<br/>参数：`self` | 无直接函数调用 |
| [`SuffixLearner._check`](../src/blockspec/parallel/online.py#L156) | 检查冻结参数/后端约定没有被外部改变，防止重放所基于的表示失效。<br/>参数：`self` | `any()`<br/>`self.model.parameters()`<br/>`self.parameter_stamp()`<br/>`RuntimeError()` |
| [`SuffixLearner._sync`](../src/blockspec/parallel/online.py#L165) | 同步执行参数所在设备，用于正确计时和更新边界；不是 TP 求和。<br/>参数：`self` | `self.execution.values()`<br/>`synchronize()` |
| [`SuffixLearner.clear_replay`](../src/blockspec/parallel/online.py#L171) | 清除旧请求/旧步骤保存的反馈，避免跨请求训练材料混用或不必要显存占用。<br/>参数：`self` | `self.replay.clear()` |
| [`SuffixLearner._skip_decoder_feedback`](../src/blockspec/parallel/online.py#L176) | 记录未捕获 boundary 的正常轮次，同时验证它确实不在必须训练的窗口。<br/>参数：`self, valid` | `type()`<br/>`ValueError()` |
| [`SuffixLearner.observe`](../src/blockspec/parallel/online.py#L190) | 校验 Feedback、detach 后写入有界 replay；到 stride 且允许更新时调用 update。<br/>参数：`self, feedback, *, may_update=True` | `self._check()`<br/>`isinstance()`<br/>`ValueError()`<br/>`self.replay.append()`<br/>`feedback.detached()`<br/>`self.update()` |
| [`SuffixLearner.replay_logits`](../src/blockspec/parallel/online.py#L214) | 用保存的 boundary/history 重算学生后缀，选取有效预测行；native 可覆盖为词表分片实现。<br/>参数：`self, item` | `self.model.forward_suffix()` |
| [`SuffixLearner.replay_divergence`](../src/blockspec/parallel/online.py#L222) | 按配置比较 student 与 teacher 概率；不是与 token ID 做均方差。<br/>参数：`self, student, teacher` | `divergence()` |
| [`SuffixLearner.backward`](../src/blockspec/parallel/online.py#L229) | 清梯度，开启 autograd，逐个反馈重放后缀，按有效位置归一化 loss 并反向到 master。<br/>参数：`self` | `sum()`<br/>`self.optimizer.zero_grad()`<br/>`torch.enable_grad()`<br/>`self.replay_logits()`<br/>`self.replay_divergence(logits, item.teacher_logits).sum()`<br/>`self.replay_divergence()`<br/>`torch.isfinite()`<br/>`bool()`<br/>`torch.isfinite(logits).all()`<br/>`torch.isfinite(item.teacher_logits).all()`；另有 7 项，见源码 |
| [`SuffixLearner.publish`](../src/blockspec/parallel/online.py#L261) | 将 FP32 master 更新值复制回 draft 执行参数（通常 BF16），让下一次起草生效；不修改 AR 参数。<br/>参数：`self` | `self.master.items()`<br/>`self.execution[name].copy_()`<br/>`torch.no_grad()` |
| [`SuffixLearner.update`](../src/blockspec/parallel/online.py#L270) | 串联 backward→梯度裁剪→AdamW.step→publish；记录版本/耗时/loss。<br/>参数：`self` | `self._check()`<br/>`self._sync()`<br/>`time.perf_counter()`<br/>`self.backward()`<br/>`sorted()`<br/>`int()`<br/>`name.split()`<br/>`self.master.items()`<br/>`self.clip_gradients()`<br/>`self.optimizer.step()`；另有 2 项，见源码 |
| [`SuffixLearner.clip_gradients`](../src/blockspec/parallel/online.py#L303) | 计算全局梯度范数并按阈值缩放；TP 版本排除复制参数的重复计数后跨 rank 归并。<br/>参数：`self` | `torch.nn.utils.clip_grad_norm_()` |
| [`SuffixLearner.state_dict`](../src/blockspec/parallel/online.py#L308) | 导出 learner 的配置、版本、优化器和参数等恢复状态；不是重新执行模型。<br/>参数：`self` | `ValueError()`<br/>`asdict()`<br/>`self.model.config.to_dict()`<br/>`source_identity()`<br/>`getattr()`<br/>`str()`<br/>`next()`<br/>`self.model.parameters()`<br/>`portable()`<br/>`self.optimizer.state_dict()` |
| [`SuffixLearner.load_state_dict`](../src/blockspec/parallel/online.py#L324) | 校验保存状态与当前训练契约再恢复，避免错模型/错冻结边界续训。<br/>参数：`self, state` | `self._check()`<br/>`state.get()`<br/>`asdict()`<br/>`self.model.config.to_dict()`<br/>`source_identity()`<br/>`getattr()`<br/>`str()`<br/>`next()`<br/>`self.model.parameters()`<br/>`state.get('master', {}).keys()`；另有 14 项，见源码 |

## src/blockspec/parallel/qwen38.py

模型结构层：AR/draft 双视图、Full Attention 与 GDN 混合层、临时 cache，以及可重放的后缀 forward。

| 函数入口 | 职责与输入 | 直接调用线索 |
| --- | --- | --- |
| [`native_reference_logits`](../src/blockspec/parallel/qwen38.py#L34) | 独立 HF 参考前向，用 hook 支持手动按层放置；vLLM 在线起草不靠它重算历史。<br/>参数：`target, tokens` | `getattr()`<br/>`layer.register_forward_pre_hook()`<br/>`handles.append()`<br/>`target.lm_head.register_forward_pre_hook()`<br/>`target()`<br/>`handle.remove()`<br/>`torch.no_grad()` |
| [`native_reference_logits.dispatch`](../src/blockspec/parallel/qwen38.py#L40) | 在手动分层参考模型执行前，把本层输入移动到本层参数设备。<br/>参数：`module, args, kwargs` | `next()`<br/>`module.parameters()`<br/>`tuple()`<br/>`move()`<br/>`kwargs.items()` |
| [`native_reference_logits.dispatch.move`](../src/blockspec/parallel/qwen38.py#L45) | 递归搬移张量/元组到目标层设备；非张量对象保留原样。<br/>参数：`item` | `isinstance()`<br/>`item.to()`<br/>`tuple()`<br/>`move()` |
| [`HybridHistory.__new__`](../src/blockspec/parallel/qwen38.py#L75) | 创建带 length 属性的 tuple 历史容器，记录每层状态及严格历史长度。<br/>参数：`cls, states, length` | `super().__new__()`<br/>`super()` |
| [`_ScratchCache.__init__`](../src/blockspec/parallel/qwen38.py#L89) | 建立每层临时工作行；Full Attention 引用只读历史后以新张量追加，GDN 先 clone 避免原地算子污染历史。<br/>参数：`self, states, layer_types` | `zip()`<br/>`SimpleNamespace()`<br/>`state[0].clone()`<br/>`state[1].clone()`<br/>`self.layers.append()` |
| [`_ScratchCache.has_previous_state`](../src/blockspec/parallel/qwen38.py#L109) | 检查该 GDN 层是否已有递推初态，决定 mixer 是否使用历史。<br/>参数：`self, layer_idx` | 无直接函数调用 |
| [`_ScratchCache.update`](../src/blockspec/parallel/qwen38.py#L115) | 将当前层临时 K/V 与历史沿序列维拼接，替换工作行引用；不是对主模型持久缓存提交。<br/>参数：`self, key, value, layer_idx, *args, **kwargs` | `torch.cat()` |
| [`_ScratchCache.update_conv_state`](../src/blockspec/parallel/qwen38.py#L130) | 仅更新 _ScratchCache 中该层临时卷积状态，不回写主模型历史。<br/>参数：`self, state, layer_idx, **kwargs` | 无直接函数调用 |
| [`_ScratchCache.update_recurrent_state`](../src/blockspec/parallel/qwen38.py#L136) | 仅更新 _ScratchCache 中该层临时递推状态，不回写主模型历史。<br/>参数：`self, state, layer_idx, **kwargs` | 无直接函数调用 |
| [`_ScratchCache.snapshot`](../src/blockspec/parallel/qwen38.py#L142) | 将工作 cache 复制/脱离计算图为新的 AR 历史快照；draft forward 不提交其临时 cache。<br/>参数：`self, length, layer_types` | `zip()`<br/>`states.append()`<br/>`tuple()`<br/>`x.detach().clone()`<br/>`x.detach()`<br/>`HybridHistory()` |
| [`_HybridLayer.__init__`](../src/blockspec/parallel/qwen38.py#L158) | 引用共享 norm/MLP，保留 AR mixer；clone mixer 参数生成独立 draft，Full Attention 显式关闭 causal。<br/>参数：`self, original, kind` | `super().__init__()`<br/>`super()`<br/>`nn.Module()`<br/>`id()`<br/>`nn.Parameter()`<br/>`p.detach().clone()`<br/>`p.detach()`<br/>`mixer.parameters()`<br/>`copy.deepcopy()` |
| [`_HybridLayer.forward`](../src/blockspec/parallel/qwen38.py#L181) | 按 view 选 AR/draft mixer，执行 norm→mixer→残差→norm→共享 MLP→残差；输入/输出 [1,K,H]。<br/>参数：`self, hidden, view, rotary, allowed, cache` | `hidden.to()`<br/>`tuple()`<br/>`x.to()`<br/>`allowed.to()`<br/>`self.input_norm()`<br/>`getattr()`<br/>`mixer()`<br/>`self.mlp()`<br/>`self.post_norm()` |
| [`Qwen38DualView.__init__`](../src/blockspec/parallel/qwen38.py#L213) | 解析文本模型配置，构造每层 AR/draft mixer 和共享模块；冻结执行视图参数，训练时由 master 函数式替换。<br/>参数：`self, target, *, block_size, mask_token_id, hybrid_policy` | `super().__init__()`<br/>`super()`<br/>`ValueError()`<br/>`getattr()`<br/>`copy.deepcopy()`<br/>`tuple()`<br/>`len()`<br/>`set()`<br/>`type()`<br/>`list()`；另有 9 项，见源码 |
| [`Qwen38DualView.set_backend`](../src/blockspec/parallel/qwen38.py#L258) | 配置双视图模型的 Attention 实现选择，保持显式后端约定。<br/>参数：`self, backend` | `ValueError()`<br/>`getattr()` |
| [`Qwen38DualView._apply`](../src/blockspec/parallel/qwen38.py#L271) | 适配 nn.Module 设备/精度变换，同时维护共享对象与源模型引用。<br/>参数：`self, fn, recurse=True` | `getattr(self.rotary, name).detach().clone()`<br/>`getattr(self.rotary, name).detach()`<br/>`getattr()`<br/>`isinstance()`<br/>`super()._apply()`<br/>`super()`<br/>`frequencies.items()`<br/>`setattr()`<br/>`value.to()` |
| [`Qwen38DualView.place_layers`](../src/blockspec/parallel/qwen38.py#L287) | 独立执行器的按层设备放置；它不是 vLLM 的 tensor parallel。<br/>参数：`self, devices` | `len()`<br/>`ValueError()`<br/>`torch.device()`<br/>`self.embedding.to()`<br/>`self.head.to()`<br/>`self.rotary.to()`<br/>`enumerate()`<br/>`layer.to()`<br/>`self.norm.to()` |
| [`Qwen38DualView._logits`](../src/blockspec/parallel/qwen38.py#L314) | 执行最终 norm 与 LM head；可传局部词表 head 供训练减少通信。<br/>参数：`self, hidden, head=None` | `self.norm()`<br/>`hidden.to()`<br/>`self.head if head is None else head()`<br/>`normalized.to()` |
| [`Qwen38DualView._cache`](../src/blockspec/parallel/qwen38.py#L321) | 按层类型建立临时工作 cache，可从后缀起始层选择对应历史。<br/>参数：`self, cache, start=0` | `len()`<br/>`ValueError()`<br/>`list()`<br/>`_ScratchCache()` |
| [`Qwen38DualView.forward`](../src/blockspec/parallel/qwen38.py#L339) | 输入 [1,K] token 与 HybridHistory；遍历混合层，按 capture_layer 保存 detached boundary，返回 hidden/cache/logits/boundary。<br/>参数：`self, tokens, *, view='ar', cache=None, capture_layer=None, logits_to_keep=0` | `ValueError()`<br/>`isinstance()`<br/>`TypeError()`<br/>`type()`<br/>`len()`<br/>`self._block_inputs()`<br/>`self._cache()`<br/>`enumerate()`<br/>`DraftBoundary(hidden, rotary, allowed, index, past, self.backend).detached()`<br/>`DraftBoundary()`；另有 4 项，见源码 |
| [`Qwen38DualView._block_inputs`](../src/blockspec/parallel/qwen38.py#L382) | token embedding 和绝对位置/RoPE；AR 构造因果 mask，draft Full Attention 使用全块可见。<br/>参数：`self, tokens, past, view` | `self.embedding()`<br/>`torch.arange()`<br/>`self.rotary()`<br/>`positions[None].expand()`<br/>`hidden.new_zeros()`<br/>`allowed.masked_fill_()`<br/>`float()` |
| [`Qwen38DualView.draft_boundary`](../src/blockspec/parallel/qwen38.py#L407) | 只执行冻结前段，得到 capture_layer 入口中间特征及位置/mask，供后缀训练重放。<br/>参数：`self, tokens, cache=None` | `DraftBoundary()`<br/>`self._block_inputs()`<br/>`torch.no_grad()` |
| [`Qwen38DualView.forward_suffix`](../src/blockspec/parallel/qwen38.py#L419) | 从 boundary.hidden 开始执行待训练后缀；用函数式参数替换注入 master，可选逐层 checkpoint 和局部词表 head。<br/>参数：`self, boundary, *, cache=None, logit_range=None, draft_weights=None, checkpoint_layers=False, logit_head=None` | `isinstance()`<br/>`len()`<br/>`ValueError()`<br/>`range()`<br/>`self.layers[i].attention.draft.named_parameters()`<br/>`set()`<br/>`overrides.items()`<br/>`name.startswith()`<br/>`torch.is_grad_enabled()`<br/>`checkpoint()`；另有 4 项，见源码 |
| [`Qwen38DualView.forward_suffix.execute`](../src/blockspec/parallel/qwen38.py#L447) | 执行单个待训练层；将对应 master 转为计算 dtype/设备，用 functional_call 临时替换本层 draft 参数。<br/>参数：`value, index=i, parameters=weights` | `dict()`<br/>`layer.named_parameters()`<br/>`weight.to()`<br/>`parameters.items()`<br/>`len()`<br/>`_ScratchCache()`<br/>`torch.func.functional_call()`<br/>`layer()` |
| [`Qwen38Branch.history_length`](../src/blockspec/parallel/qwen38.py#L484) | 读取混合历史的 token 长度；不能用 cache 张量某一维统一推断 GDN 与 KV。<br/>参数：`cache` | 无直接函数调用 |
| [`Qwen38Branch.commit_history`](../src/blockspec/parallel/qwen38.py#L489) | 独立参考生成器按接受前缀提交教师 AR 历史；vLLM 原生提交路径不调用这个参考分支方法。<br/>参数：`self, proposal, verified_cache, committed` | `self.history_length()`<br/>`ValueError()`<br/>`self.ar()` |
| [`Qwen38Branch.audit_history`](../src/blockspec/parallel/qwen38.py#L504) | 独立参考分支的历史正确性审计，比较分段生成与参考历史。<br/>参数：`self, actual, tokens, chunks` | `sum()`<br/>`any()`<br/>`ValueError()`<br/>`self.ar()`<br/>`enumerate()`<br/>`zip()`<br/>`torch.testing.assert_close()` |

## src/blockspec/parallel/qwen38_training.py

离线监督构造：对干净文本执行冻结 AR 教师前向，构造 anchor+MASK，保存待训练后缀入口。

| 函数入口 | 职责与输入 | 直接调用线索 |
| --- | --- | --- |
| [`offline_feedback`](../src/blockspec/parallel/qwen38_training.py#L22) | 教师强制 AR 前向生成完整 K−1 行监督；构造 MASK 输入与冻结前缀边界，返回 Feedback。<br/>参数：`model, tokens, anchor` | `type()`<br/>`bool()`<br/>`((tokens < 0) \| (tokens >= model.config.vocab_size)).any()`<br/>`ValueError()`<br/>`model()`<br/>`model(clean, cache=history).logits[0, :-1].detach()`<br/>`torch.full_like()`<br/>`model.draft_boundary()`<br/>`Feedback()`<br/>`torch.no_grad()` |

## src/blockspec/parallel/sampling.py

采样策略包装：从 draft logits 取候选与真实 q；其独立 verify 方法不等于 vLLM 在线路径的原生 sampler。

| 函数入口 | 职责与输入 | 直接调用线索 |
| --- | --- | --- |
| [`ProposalSampler.__init__`](../src/blockspec/parallel/sampling.py#L18) | 保存采样配置与可选 executor；无 executor 时直接使用概率数学工具。<br/>参数：`self, config=SamplingConfig(), *, executor=None` | `SamplingConfig()` |
| [`ProposalSampler.sample_ar`](../src/blockspec/parallel/sampling.py#L24) | 独立参考路径的 AR token 选择；可交给外部 executor 执行。<br/>参数：`self, logits, generator` | `self.executor.sample_ar()`<br/>`int()`<br/>`sample_logits()` |
| [`ProposalSampler.propose`](../src/blockspec/parallel/sampling.py#L32) | greedy 取最大分数 token 且 q=None；随机模式计算真实温度/截断后 q 并从中抽样。<br/>参数：`self, logits, generator` | `self.executor.draft()`<br/>`greedy_tokens()`<br/>`probabilities()`<br/>`draw()` |
| [`ProposalSampler.verify`](../src/blockspec/parallel/sampling.py#L46) | 在 runtime 中包装原生采样；在独立 ProposalSampler 中则是参考校正。以所在类/模块为准，不是两套同时调用。<br/>参数：`self, proposal, target_logits, generator` | `self.executor.verify()`<br/>`verify_greedy()`<br/>`greedy_tokens()`<br/>`probabilities()`<br/>`verify_linear()` |

## src/blockspec/sampling.py

概率与校正数学工具：温度/top-k/top-p、抽样、greedy 和随机校正参考实现。vLLM 主路径复用的是原生 rejection sampler。

| 函数入口 | 职责与输入 | 直接调用线索 |
| --- | --- | --- |
| [`SamplingConfig.__post_init__`](../src/blockspec/sampling.py#L32) | 数据类构造后校验配置，拒绝无法支持的参数组合。<br/>参数：`self` | `math.isfinite()`<br/>`ValueError()` |
| [`probabilities`](../src/blockspec/sampling.py#L46) | 验证采样配置/输入 logits 后计算 temperature/top-k/top-p 对应的完整概率分布。<br/>参数：`logits: Tensor, config: SamplingConfig` | `torch.isfinite(logits).all()`<br/>`torch.isfinite()`<br/>`ValueError()`<br/>`_probabilities_unchecked()` |
| [`_probabilities_unchecked`](../src/blockspec/sampling.py#L57) | 对 logits 做温度缩放、候选截断和 softmax；内部路径假定输入已由上层检查。<br/>参数：`logits: Tensor, config: SamplingConfig` | `logits.float()`<br/>`torch.zeros_like(work).scatter_()`<br/>`torch.zeros_like()`<br/>`work.argmax()`<br/>`min()`<br/>`work.topk()`<br/>`torch.zeros_like(work, dtype=torch.bool).scatter_()`<br/>`work.masked_fill()`<br/>`work.sort()`<br/>`ordered.softmax()`；另有 3 项，见源码 |
| [`greedy_tokens`](../src/blockspec/sampling.py#L84) | 沿词表维取 argmax，得到每个预测位置的 token ID。<br/>参数：`logits` | `torch.isfinite(logits).all()`<br/>`torch.isfinite()`<br/>`ValueError()`<br/>`logits.argmax()` |
| [`sample_logits`](../src/blockspec/sampling.py#L96) | 把一行 logits 按配置转成概率并抽取一个 token；不是独立执行模型。<br/>参数：`logits, config=SamplingConfig(), generator=None` | `SamplingConfig()`<br/>`greedy_tokens()`<br/>`draw()`<br/>`probabilities()` |
| [`validate_distribution`](../src/blockspec/sampling.py#L104) | 检查概率非负、有限、归一化等要求，防止非法 p/q 进入随机校正。<br/>参数：`p` | `torch.isfinite(p).all()`<br/>`torch.isfinite()`<br/>`(p < 0).any()`<br/>`ValueError()`<br/>`torch.allclose()`<br/>`p.sum()`<br/>`torch.ones_like()` |
| [`draw`](../src/blockspec/sampling.py#L118) | 按每行概率分布抽样，用传入 Generator 控制随机序列。<br/>参数：`p: Tensor, generator=None` | `validate_distribution()`<br/>`torch.multinomial(p.reshape(-1, p.shape[-1]), 1, generator=generator).reshape()`<br/>`torch.multinomial()`<br/>`p.reshape()` |
| [`residual`](../src/blockspec/sampling.py#L127) | 构造首次拒绝后的残差分布：对 p−q 取非负部分再归一化。<br/>参数：`p: Tensor, q: Tensor` | `validate_distribution()`<br/>`ValueError()`<br/>`(p - q).clamp_min()`<br/>`positive.sum()`<br/>`torch.where()`<br/>`mass.clamp_min()`<br/>`torch.finfo()`<br/>`mass.squeeze()` |
| [`verify_greedy`](../src/blockspec/sampling.py#L161) | 参考 greedy 校正：找候选与 AR argmax 的第一个不一致处，保留匹配前缀并补目标 token。<br/>参数：`proposed, target_ids` | `proposed.numel()`<br/>`ValueError()`<br/>`proposed.tolist()`<br/>`target_ids.tolist()`<br/>`any()`<br/>`enumerate()`<br/>`Verification()`<br/>`len()` |
| [`verify_linear`](../src/blockspec/sampling.py#L184) | 参考随机校正：按 min(1,p/q) 接受候选，首次拒绝后从残差分布采样；vLLM 服务路径仍用原生 sampler。<br/>参数：`proposed: Tensor, q: Tensor, p: Tensor, *, generator=None, acceptance_uniforms: Tensor \| None=None` | `proposed.numel()`<br/>`ValueError()`<br/>`validate_distribution()`<br/>`((proposed < 0) \| (proposed >= q.shape[-1])).any()`<br/>`torch.rand()`<br/>`torch.isfinite(acceptance_uniforms).all()`<br/>`torch.isfinite()`<br/>`((acceptance_uniforms < 0) \| (acceptance_uniforms >= 1)).any()`<br/>`range()`<br/>`int()`；另有 7 项，见源码 |

## src/blockspec/vllm_ascend/qwen38_attention.py

主模型 verifier 的分页 Attention 适配：K32 保持不变，内部按 query 上限切片并保持因果可见边界。不是 draft 双向 Attention 的实现文件。

| 函数入口 | 职责与输入 | 直接调用线索 |
| --- | --- | --- |
| [`_enabled`](../src/blockspec/vllm_ascend/qwen38_attention.py#L18) | 限定 hook 仅作用于本 Qwen proposer 配置，避免改变其他模型执行路径。<br/>参数：`config` | 无直接函数调用 |
| [`verification_chunks`](../src/blockspec/vllm_ascend/qwen38_attention.py#L27) | 把原生因果 verifier query 分为最多 16 行的片段；每片 key 末端随 query 末端推进，保持因果语义。<br/>参数：`metadata, limit=16` | `len()`<br/>`ValueError()`<br/>`range()`<br/>`min()`<br/>`copy.copy()`<br/>`metadata.seq_lens.new_tensor()`<br/>`metadata.seq_lens_cpu.new_tensor()`<br/>`metadata.query_start_loc.new_tensor()`<br/>`slice()` |
| [`install_attention_hooks`](../src/blockspec/vllm_ascend/qwen38_attention.py#L57) | 为指定版本 Ascend Attention 安装 builder 和调用切片适配；不把 verifier 改成双向。<br/>参数：`builder_class=None, impl_class=None, spec_state=None` | `getattr()`<br/>`inspect.signature(initialize).parameters.keys()`<br/>`inspect.signature()`<br/>`RuntimeError()`<br/>`inspect.signature(forward).parameters.keys()` |
| [`install_attention_hooks.bounded`](../src/blockspec/vllm_ascend/qwen38_attention.py#L82) | 用受限候选宽度建立兼容 builder，再恢复真实 K32 配置与 decode 阈值。<br/>参数：`self, kv_cache_spec, layer_names, vllm_config, device` | `_enabled()`<br/>`initialize()`<br/>`copy.copy()`<br/>`functools.wraps()` |
| [`install_attention_hooks.sliced`](../src/blockspec/vllm_ascend/qwen38_attention.py#L97) | 对超过设备 query 限制的 verifier 逐片调用已有 FIA；整块 KV 已在外层写入，不重复提交 draft KV。<br/>参数：`self, query, key, value, attn_metadata, output, kv_cache=None` | `_enabled()`<br/>`forward()`<br/>`verification_chunks()`<br/>`functools.wraps()` |

## src/blockspec/vllm_ascend/qwen38_gdn_backend.py

原生 GDN 后端适配：处理递推 query 的分片与 prefill 路径，供原生模型及 draft 算子包装使用。

| 函数入口 | 职责与输入 | 直接调用线索 |
| --- | --- | --- |
| [`recurrent_slices`](../src/blockspec/vllm_ascend/qwen38_gdn_backend.py#L15) | 按设备支持的 query 宽度切分 GDN 递推，保持跨片状态延续和索引对齐。<br/>参数：`operation, *, query, key, value, state, actual_seq_lengths, ssm_state_indices, num_accepted_tokens, g, beta, scale, limit=16` | `actual_seq_lengths.numel()`<br/>`ssm_state_indices.numel()`<br/>`ValueError()`<br/>`ssm_state_indices[num_accepted_tokens.long() - 1].long()`<br/>`num_accepted_tokens.long()`<br/>`state.index_select()`<br/>`tensor.contiguous()`<br/>`g.contiguous()`<br/>`beta.contiguous()`<br/>`range()`；另有 10 项，见源码 |
| [`install_recurrent_hook`](../src/blockspec/vllm_ascend/qwen38_gdn_backend.py#L58) | 为原生 GDN 安装递推切片适配，避免短块 query 限制导致调用失败。<br/>参数：`vllm_config=None` | `getattr()`<br/>`get_current_vllm_config()` |
| [`install_recurrent_hook.recurrent`](../src/blockspec/vllm_ascend/qwen38_gdn_backend.py#L75) | 对原生递推调用做包装，转交保留状态语义的 recurrent_slices。<br/>参数：`**kwargs` | `_enabled()`<br/>`kwargs.get()`<br/>`original()`<br/>`recurrent_slices()`<br/>`functools.wraps()` |
| [`install_cann_gdn_prefill`](../src/blockspec/vllm_ascend/qwen38_gdn_backend.py#L86) | 把符合配置的原生 GDN prefill 切换为已有 CANN 路径；不是训练优化器。<br/>参数：`vllm_config=None` | `getattr()`<br/>`hasattr()`<br/>`RuntimeError()`<br/>`get_current_vllm_config()` |
| [`install_cann_gdn_prefill.prefill`](../src/blockspec/vllm_ascend/qwen38_gdn_backend.py#L113) | 处理 GDN 初始文本前向并按需要输出最终状态；是主模型 prefill 的后端适配。<br/>参数：`q, k, v, g, beta, *, initial_state=None, output_final_state=False, cu_seqlens=None, head_first=False, use_qk_l2norm_in_kernel=False, **kwargs` | `(config.additional_config or {}).get()`<br/>`original()`<br/>`set()`<br/>`ValueError()`<br/>`sorted()`<br/>`l2norm_fwd()`<br/>`initial_state.transpose(-1, -2).contiguous().to()`<br/>`initial_state.transpose(-1, -2).contiguous()`<br/>`initial_state.transpose()`<br/>`torch_npu.npu_chunk_gated_delta_rule()`；另有 12 项，见源码 |

## src/blockspec/vllm_ascend/qwen38_history.py

历史读取层：从原生分页 KV / GDN checkpoint 取得已提交前缀的快照。共享历史来源，不代表所有数据零拷贝。

| 函数入口 | 职责与输入 | 直接调用线索 |
| --- | --- | --- |
| [`paged_history`](../src/blockspec/vllm_ascend/qwen38_history.py#L26) | 把原生 [page,token,KV_head,dim] 按 block_table 收集为 [1,KV_head,history,dim]；发生 gather/重排拷贝。<br/>参数：`cache, block_table, length` | `ValueError()`<br/>`block_table[0, :pages].to()`<br/>`cache.index_select(0, indices).flatten(0, 1)[:length].transpose(0, 1).unsqueeze()`<br/>`cache.index_select(0, indices).flatten(0, 1)[:length].transpose()`<br/>`cache.index_select(0, indices).flatten()`<br/>`cache.index_select()` |
| [`gdn_history`](../src/blockspec/vllm_ascend/qwen38_history.py#L48) | 从原生 speculative checkpoint 选取接受前缀对应的卷积/递推状态，避免把拒绝后缀当历史。<br/>参数：`cache, metadata, *, accepted, query_length, kernel_width` | `ValueError()`<br/>`conv_meta.cache_indices.reshape()`<br/>`conv.index_select()`<br/>`conv_index.to()`<br/>`F.pad()`<br/>`tail.transpose()`<br/>`recurrent.index_select(0, state_index.to(recurrent.device, torch.long)).transpose()`<br/>`recurrent.index_select()`<br/>`state_index.to()` |
| [`NativeHybridHistory.__init__`](../src/blockspec/vllm_ascend/qwen38_history.py#L109) | 把每层原生缓存模块与 static_forward_context 名称对应起来；之后用名称取得本轮 metadata。<br/>参数：`self, target, runner, layer_types` | `getattr()`<br/>`id()`<br/>`runner.compilation_config.static_forward_context.items()`<br/>`zip()`<br/>`ValueError()`<br/>`self.layers.append()`<br/>`any()` |
| [`NativeHybridHistory.capture`](../src/blockspec/vllm_ascend/qwen38_history.py#L133) | 保存当前请求的逐层 metadata；这里只记录索引信息，真正张量读取发生在 read。<br/>参数：`self, metadata, request_id` | `isinstance()`<br/>`ValueError()` |
| [`NativeHybridHistory.read`](../src/blockspec/vllm_ascend/qwen38_history.py#L146) | 校对请求和已提交长度，逐层读取 Full Attention KV 或 GDN 状态，返回 HybridHistory。<br/>参数：`self, request_id, length, *, sampled_count, draft_count` | `RuntimeError()`<br/>`next()`<br/>`len()`<br/>`states.append()`<br/>`paged_history()`<br/>`gdn_history()`<br/>`HybridHistory()` |

## src/blockspec/vllm_ascend/qwen38_loss.py

可选词表并行损失：每卡只保留自己的 student 词表分片，通过 collective 得到全局归一化与散度。

| 函数入口 | 职责与输入 | 直接调用线索 |
| --- | --- | --- |
| [`_VocabDivergence.forward`](../src/blockspec/vllm_ascend/qwen38_loss.py#L19) | 从局部 student logits 和全局 teacher 分布计算 KL/TV；跨 rank 合并归一化量，并保存局部解析梯度。<br/>参数：`ctx, student, teacher, start, tp, kind` | `ValueError()`<br/>`student.to()`<br/>`teacher[..., start:start + student.shape[-1]].detach().to()`<br/>`teacher[..., start:start + student.shape[-1]].detach()`<br/>`tp.gather()`<br/>`torch.stack()`<br/>`value.logsumexp()`<br/>`normalizer.unflatten(-1, (tp.size, 2)).logsumexp()`<br/>`normalizer.unflatten()`<br/>`enumerate()`；另有 9 项，见源码 |
| [`_VocabDivergence.backward`](../src/blockspec/vllm_ascend/qwen38_loss.py#L53) | 将保存的局部 student 梯度乘上游梯度；教师、分片起点、TP 对象和散度类型不更新。<br/>参数：`ctx, gradient` | `(derivative * gradient[..., None]).to()` |
| [`vocab_divergence`](../src/blockspec/vllm_ascend/qwen38_loss.py#L62) | 计算词表分片 KL/TV；输入 student 是局部分片，teacher 与全局词表/分片起点共同决定监督。<br/>参数：`student, teacher, *, start, tp, kind` | `_VocabDivergence.apply()` |

## src/blockspec/vllm_ascend/qwen38_ops.py

计算后端：给双视图模型安装现成 NPU 算子或可微分组合。根据设备、形状和是否启用梯度选路，不等同于自研 kernel。

| 函数入口 | 职责与输入 | 直接调用线索 |
| --- | --- | --- |
| [`swiglu`](../src/blockspec/vllm_ascend/qwen38_ops.py#L23) | 对 gate/up 两半计算 SiLU(gate)*up；NPU 优先使用现成融合算子。<br/>参数：`packed` | `torch_npu.npu_swiglu()`<br/>`packed.float().chunk()`<br/>`packed.float()`<br/>`(F.silu(gate) * up).to()`<br/>`F.silu()` |
| [`l2norm`](../src/blockspec/vllm_ascend/qwen38_ops.py#L37) | 对 Q/K 最后一维做 L2 归一化；无梯度且形状符合时走已有高效实现。<br/>参数：`value` | `torch.is_grad_enabled()`<br/>`l2norm_fwd()`<br/>`value.float()`<br/>`(full * torch.rsqrt(full.square().sum(-1, keepdim=True) + 1e-06)).to()`<br/>`torch.rsqrt()`<br/>`full.square().sum()`<br/>`full.square()` |
| [`causal_conv`](../src/blockspec/vllm_ascend/qwen38_ops.py#L50) | GDN 的因果深度卷积和 SiLU；推理符合限制时走原生算子，训练/其他形状用可微组合。<br/>参数：`x, weight, bias=None, activation='silu', seq_idx=None` | `ValueError()`<br/>`torch.is_grad_enabled()`<br/>`torch.is_inference()`<br/>`weight.clone()`<br/>`F.conv1d()`<br/>`x.float()`<br/>`weight[:, None].float()`<br/>`bias.float()`<br/>`F.silu(out).to()`<br/>`F.silu()`；另有 10 项，见源码 |
| [`causal_conv_update`](../src/blockspec/vllm_ascend/qwen38_ops.py#L95) | 处理单步卷积状态滚动；state 必须是临时工作状态，不能把 draft 更新写进主模型。<br/>参数：`hidden, state, weight, bias=None, activation='silu'` | `torch.cat()`<br/>`hidden.reshape()`<br/>`causal_conv()`<br/>`state.copy_()` |
| [`delta_rule`](../src/blockspec/vllm_ascend/qwen38_ops.py#L111) | 选择 GDN 递推后端：短块推理用原生递推，训练按配置选 torch 或 FLA 可微实现。<br/>参数：`query, key, value, *, fallback, groups, training_fallback=None, **kwargs` | `kwargs.pop()`<br/>`l2norm()`<br/>`torch.is_grad_enabled()`<br/>`kwargs.get()`<br/>`set()`<br/>`training_fallback()`<br/>`(max(1, query.shape[1]) - 1).bit_length()`<br/>`max()`<br/>`fallback()`<br/>`torch.empty()`；另有 11 项，见源码 |
| [`GatedRMSNorm.__init__`](../src/blockspec/vllm_ascend/qwen38_ops.py#L167) | 保存 GDN 的 norm 权重和 epsilon，供前向选择融合或可微分实现。<br/>参数：`self, weight, eps` | `super().__init__()`<br/>`super()` |
| [`GatedRMSNorm.forward`](../src/blockspec/vllm_ascend/qwen38_ops.py#L174) | 对 GDN 输出做 RMSNorm 并乘 SiLU(gate)；无梯度优先融合，训练走可微路径。<br/>参数：`self, value, gate` | `torch.is_grad_enabled()`<br/>`layer_norm_fwd_npu()`<br/>`torch.is_inference()`<br/>`weight.clone()`<br/>`value.float()`<br/>`torch_npu.npu_rms_norm()`<br/>`weight.float()`<br/>`(normalized * F.silu(gate.float())).to()`<br/>`F.silu()`<br/>`gate.float()`；另有 4 项，见源码 |
| [`configure_gdn`](../src/blockspec/vllm_ascend/qwen38_ops.py#L202) | 给 mixer 装配归一化、卷积和 delta rule 实现；这里选择后端，并不执行一次完整推理。<br/>参数：`mixer, training_backend='torch'` | `os.environ.get()`<br/>`sys.path.insert()`<br/>`version()`<br/>`ValueError()`<br/>`RuntimeError()`<br/>`GatedRMSNorm()`<br/>`partial()` |
| [`SharedRotary.__init__`](../src/blockspec/vllm_ascend/qwen38_ops.py#L250) | 引用原生 RoPE cos/sin 缓存及旋转配置；不是再训练一套位置参数。<br/>参数：`self, original` | `tuple()` |
| [`SharedRotary.__deepcopy__`](../src/blockspec/vllm_ascend/qwen38_ops.py#L258) | 控制深复制的共享语义；返回 self 表示对象继续共享，而非创建新的通信组/缓存。<br/>参数：`self, memo` | 无直接函数调用 |
| [`SharedRotary.positions`](../src/blockspec/vllm_ascend/qwen38_ops.py#L263) | 根据严格历史长度创建当前块的绝对位置索引，供 RoPE 读取。<br/>参数：`self, start, length, device` | `torch.arange()` |
| [`SharedRotary.coefficients`](../src/blockspec/vllm_ascend/qwen38_ops.py#L269) | 从共享 RoPE 表按位置取 cos/sin 并调整布局与 dtype。<br/>参数：`self, positions, dtype` | `self.cache.index_select(0, positions).chunk()`<br/>`self.cache.index_select()`<br/>`tuple()`<br/>`torch.cat((value, value), dim=-1)[None, None].to()`<br/>`torch.cat()` |
| [`full_attention`](../src/blockspec/vllm_ascend/qwen38_ops.py#L281) | draft/AR 视图的 Full Attention 执行：投影 Q/K/V 与 gate，norm/RoPE，拼入工作历史，融合 Attention，门控和 O 投影。<br/>参数：`self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs` | `self._reference_forward()`<br/>`projection()`<br/>`projected_q.reshape(batch, length, -1, 2 * self.head_dim).chunk()`<br/>`projected_q.reshape()`<br/>`query.reshape(shape).transpose()`<br/>`query.reshape()`<br/>`projected_k.reshape(shape).transpose()`<br/>`projected_k.reshape()`<br/>`projected_v.reshape(shape).transpose()`<br/>`projected_v.reshape()`；另有 28 项，见源码 |
| [`full_attention.normalize_rotate`](../src/blockspec/vllm_ascend/qwen38_ops.py#L337) | 对 Q 或 K 先做 FP32 中间计算的 RMSNorm，再只旋转 RoPE 对应维度，最后回到执行 dtype。<br/>参数：`value, weight` | `value.float()`<br/>`torch.rsqrt()`<br/>`full.square().mean()`<br/>`full.square()`<br/>`(1 + weight).float()`<br/>`torch_npu.npu_rotary_mul()`<br/>`normalized[..., :rotary_dim].contiguous()`<br/>`cos.float()`<br/>`sin.float()`<br/>`torch.cat((rotated, normalized[..., rotary_dim:]), dim=-1).to(value.dtype).contiguous()`；另有 2 项，见源码 |
| [`configure_attention`](../src/blockspec/vllm_ascend/qwen38_ops.py#L373) | 保留 reference forward，并把已有 NPU 算子包装绑定为该 mixer 的 forward。<br/>参数：`mixer, native_rope=None` | `SharedRotary()`<br/>`MethodType()` |

## src/blockspec/vllm_ascend/qwen38_proposer.py

投机流程组织层：绑定原生模型，构造起草输入，保存 pending，提供 q，消费验证反馈并触发训练。

| 函数入口 | 职责与输入 | 直接调用线索 |
| --- | --- | --- |
| [`OrthrusWorkerExtension.orthrus_use_cann_gdn_prefill`](../src/blockspec/vllm_ascend/qwen38_proposer.py#L50) | 为当前 worker 安装选定的 CANN GDN prefill 适配。<br/>参数：`self` | `install_cann_gdn_prefill()` |
| [`OrthrusWorkerExtension.orthrus_stats`](../src/blockspec/vllm_ascend/qwen38_proposer.py#L59) | 通过 worker RPC 返回当前 rank 起草、验证、更新计数及配置；不是模型 forward。<br/>参数：`self` | `getattr()`<br/>`draft.options.get()` |
| [`OrthrusWorkerExtension.orthrus_offline_step`](../src/blockspec/vllm_ascend/qwen38_proposer.py#L82) | worker 上的一次离线训练：允许梯度、创建 token 张量、构造 offline_feedback，再交给同一个 learner。<br/>参数：`self, token_ids, anchor` | `getattr()`<br/>`draft.options.get()`<br/>`RuntimeError()`<br/>`torch.inference_mode()`<br/>`torch.enable_grad()`<br/>`torch.tensor()`<br/>`offline_feedback()`<br/>`draft.learner.observe()`<br/>`draft.learner.clear_replay()` |
| [`OrthrusWorkerExtension.orthrus_checkpoint`](../src/blockspec/vllm_ascend/qwen38_proposer.py#L104) | 通过 worker RPC 导出 head；调用方不直接操作各卡参数。<br/>参数：`self` | `getattr()`<br/>`draft.options.get()`<br/>`RuntimeError()`<br/>`Path()`<br/>`draft._save_head()`<br/>`str()` |
| [`Qwen38OrthrusProposer.__init__`](../src/blockspec/vllm_ascend/qwen38_proposer.py#L125) | 检查 BF16/FP32、K、单请求/eager 等支持范围，初始化计数与空状态，安装限定版本 hooks；模型绑定稍后完成。<br/>参数：`self, vllm_config` | `(vllm_config.additional_config or {}).get()`<br/>`isinstance()`<br/>`set()`<br/>`ValueError()`<br/>`dict()`<br/>`options.get()`<br/>`type()`<br/>`getattr()`<br/>`install_runner_hooks()`<br/>`install_attention_hooks()`；另有 2 项，见源码 |
| [`Qwen38OrthrusProposer.bind_runner`](../src/blockspec/vllm_ascend/qwen38_proposer.py#L229) | 保存 NPU runner 引用并取得 TP 通信组；后续访问当前请求和历史 metadata 都依赖它。<br/>参数：`self, runner` | `get_tp_group()`<br/>`TensorParallel()` |
| [`Qwen38OrthrusProposer.load_model`](../src/blockspec/vllm_ascend/qwen38_proposer.py#L241) | 基于已加载 target 构建共享权重双视图，初始化历史读取器/随机数生成器，按配置加载 head 和创建 learner。<br/>参数：`self, target_model` | `torch.inference_mode()`<br/>`torch.no_grad()`<br/>`from_native_model()`<br/>`self.options.get()`<br/>`NativeHybridHistory()`<br/>`torch.Generator()`<br/>`self.generator.manual_seed()`<br/>`self._load_head()`<br/>`Path()`<br/>`SuffixConfig()`；另有 3 项，见源码 |
| [`Qwen38OrthrusProposer.propose`](../src/blockspec/vllm_ascend/qwen38_proposer.py#L280) | 拒绝走不含原生模型/历史生命周期的通用入口；本实现需要 runtime 的 propose_batch 接入。<br/>参数：`self, *args, **kwargs` | `RuntimeError()` |
| [`Qwen38OrthrusProposer.dummy_run`](../src/blockspec/vllm_ascend/qwen38_proposer.py#L286) | 此处不执行实际 draft；权重/master 已预先分配，历史快照也不参加图捕获。<br/>参数：`self, **kwargs` | 无直接函数调用 |
| [`Qwen38OrthrusProposer._clear_request`](../src/blockspec/vllm_ascend/qwen38_proposer.py#L295) | 切换请求时清空旧 history/pending/replay，并按请求随机种子重置 draft RNG，避免请求之间串状态。<br/>参数：`self, request_id` | `self.generator.manual_seed()`<br/>`generators[0].initial_seed()`<br/>`self.learner.clear_replay()` |
| [`Qwen38OrthrusProposer._draft_sampling`](../src/blockspec/vllm_ascend/qwen38_proposer.py#L311) | 把原生请求的 greedy/temperature/top-k/top-p 转成 draft 使用的同一概率策略。<br/>参数：`self` | `SamplingConfig()`<br/>`float()`<br/>`max()`<br/>`int()` |
| [`Qwen38OrthrusProposer.propose_batch`](../src/blockspec/vllm_ascend/qwen38_proposer.py#L326) | 起草入口。输入是原生同步记账后的 sampled token 与调度信息；读取严格历史，构造 [1,K] anchor+MASK，返回候选 ID 列表。<br/>参数：`self, sampled, scheduler_output, *, verification=None` | `self.options.get()`<br/>`RuntimeError()`<br/>`len()`<br/>`isinstance()`<br/>`sampled.cpu().tolist()`<br/>`sampled.cpu()`<br/>`self._clear_request()`<br/>`int()`<br/>`torch.inference_mode()`<br/>`torch.no_grad()`；另有 11 项，见源码 |
| [`Qwen38OrthrusProposer._verification_count`](../src/blockspec/vllm_ascend/qwen38_proposer.py#L400) | 检查原生待验证候选与 pending 一致；返回本轮实际验证候选数，不允许错请求或错 token。<br/>参数：`self, metadata` | `len()`<br/>`RuntimeError()`<br/>`pending['candidates'].numel()`<br/>`torch.equal()`<br/>`metadata.draft_token_ids.to()` |
| [`Qwen38OrthrusProposer.verification_probs`](../src/blockspec/vllm_ascend/qwen38_proposer.py#L428) | 从 pending 取实际用于起草的 q，按本轮验证长度截取；greedy 路径允许返回 None。<br/>参数：`self, metadata` | `self._verification_count()`<br/>`q[:count].contiguous()` |
| [`Qwen38OrthrusProposer.teacher_rows`](../src/blockspec/vllm_ascend/qwen38_proposer.py#L440) | 在 sampler 修改 logits 之前摘取教师行；只有在线更新确实捕获 boundary 时才需要。<br/>参数：`self, logits, metadata` | `self._verification_count()`<br/>`torch.inference_mode()`<br/>`torch.no_grad()`<br/>`logits.index_select(0, metadata.target_logits_indices).detach()`<br/>`logits.index_select()` |
| [`Qwen38OrthrusProposer.observe_verified`](../src/blockspec/vllm_ascend/qwen38_proposer.py#L457) | 消费本轮 pending 与原生验证结果。统计接受情况，按实际到达行构造 Feedback；有 learner 才触发 observe。<br/>参数：`self, metadata, teacher, sampled` | `int()`<br/>`(sampled[0] >= 0).sum().item()`<br/>`(sampled[0] >= 0).sum()`<br/>`min()`<br/>`RuntimeError()`<br/>`max()`<br/>`torch.inference_mode()`<br/>`self.learner._skip_decoder_feedback()`<br/>`Feedback()`<br/>`self.learner.observe()`；另有 4 项，见源码 |
| [`Qwen38OrthrusProposer._head_identity`](../src/blockspec/vllm_ascend/qwen38_proposer.py#L511) | 构造 head 与当前模型/TP/算法配置的身份信息，防止误加载不兼容参数。<br/>参数：`self` | `Path()`<br/>`all()`<br/>`path.is_file()`<br/>`ValueError()`<br/>`hashlib.sha256(path.read_bytes()).hexdigest()`<br/>`hashlib.sha256()`<br/>`path.read_bytes()` |
| [`Qwen38OrthrusProposer._draft_parameters`](../src/blockspec/vllm_ascend/qwen38_proposer.py#L528) | 仅枚举新增 draft mixer 参数；共享 backbone 不属于 head 交付物。<br/>参数：`self` | `self.model.named_parameters()` |
| [`Qwen38OrthrusProposer._load_head`](../src/blockspec/vllm_ascend/qwen38_proposer.py#L533) | 校验交付 head 身份及 tensor 布局，再复制到当前 draft 执行参数；不是加载第二份基模。<br/>参数：`self, directory` | `json.loads()`<br/>`(directory / 'manifest.json').read_text()`<br/>`self._head_identity()`<br/>`ValueError()`<br/>`load_file()`<br/>`str()`<br/>`self._draft_parameters()`<br/>`tensors.keys()`<br/>`parameters.keys()`<br/>`tensors.items()`；另有 3 项，见源码 |
| [`Qwen38OrthrusProposer._save_head`](../src/blockspec/vllm_ascend/qwen38_proposer.py#L558) | 保存当前 draft 参数/训练状态等交付信息，供后续重载；检查具体输出文件以区分 head 和优化器状态。<br/>参数：`self, directory` | `directory.mkdir()`<br/>`(directory / 'manifest.json').exists()`<br/>`ValueError()`<br/>`save_file()`<br/>`p.detach().cpu().contiguous()`<br/>`p.detach().cpu()`<br/>`p.detach()`<br/>`self._draft_parameters().items()`<br/>`self._draft_parameters()`<br/>`str()`；另有 8 项，见源码 |

## src/blockspec/vllm_ascend/qwen38_runtime.py

引擎接入层：包装指定版本 NPUModelRunner 的生命周期、起草和采样入口；不重新实现原生 scheduler 或 rejection sampler。

| 函数入口 | 职责与输入 | 直接调用线索 |
| --- | --- | --- |
| [`_DraftProbabilities.__init__`](../src/blockspec/vllm_ascend/qwen38_runtime.py#L27) | 保存原生 sampler 和本轮实际 draft 概率，创建短生命周期的调用代理。<br/>参数：`self, sampler, probabilities` | 无直接函数调用 |
| [`_DraftProbabilities.__getattr__`](../src/blockspec/vllm_ascend/qwen38_runtime.py#L32) | 把代理未定义的属性访问转发给原生 sampler，保持原生调用方接口。<br/>参数：`self, name` | `getattr()` |
| [`_DraftProbabilities.__call__`](../src/blockspec/vllm_ascend/qwen38_runtime.py#L38) | 代理原生 rejection sampler，把本轮真实 draft 概率 q 替换进调用，其余参数保持原样。<br/>参数：`self, metadata, draft_probs, *args, **kwargs` | `self.sampler()` |
| [`_ours`](../src/blockspec/vllm_ascend/qwen38_runtime.py#L44) | 判断 runner 当前 drafter 是否属于本套适配，避免影响其他 proposer。<br/>参数：`runner` | `getattr()` |
| [`install_runner_hooks`](../src/blockspec/vllm_ascend/qwen38_runtime.py#L52) | 检查 runner 内部接口后安装限定版本包装；仅标记为本实现的 proposer 使用这些分支。<br/>参数：`runner_class=None` | `getattr()`<br/>`expected.items()`<br/>`inspect.signature(method).parameters.keys()`<br/>`inspect.signature()`<br/>`RuntimeError()` |
| [`install_runner_hooks.retain`](../src/blockspec/vllm_ascend/qwen38_runtime.py#L85) | 包装 _set_up_drafter：保留上游建立的 custom 实例，初始化后把 runner 绑定给它。<br/>参数：`self` | `getattr()`<br/>`setup()`<br/>`_ours()`<br/>`self.drafter.bind_runner()`<br/>`functools.wraps()` |
| [`install_runner_hooks.create`](../src/blockspec/vllm_ascend/qwen38_runtime.py#L98) | 包装 _get_drafter：优先返回保存的 custom 实例，否则继续原生创建逻辑。<br/>参数：`self` | `getattr()`<br/>`get_drafter()`<br/>`functools.wraps()` |
| [`install_runner_hooks.draft`](../src/blockspec/vllm_ascend/qwen38_runtime.py#L107) | 包装 propose_draft_token_ids：解析原方法实参并转给本 proposer 的 propose_batch。<br/>参数：`self, *args, **kwargs` | `_ours()`<br/>`propose()`<br/>`propose_signature.bind()`<br/>`self.drafter.propose_batch()`<br/>`arguments.get()`<br/>`functools.wraps()` |
| [`install_runner_hooks.metadata`](../src/blockspec/vllm_ascend/qwen38_runtime.py#L117) | 包装 metadata 构建：保留 GDN 接受边界语义，并把当前请求逐层 metadata 交给 history_reader。<br/>参数：`self, *args, **kwargs` | `metadata_signature.bind()`<br/>`_ours()`<br/>`build_metadata()`<br/>`self.drafter.history_reader.capture()`<br/>`functools.wraps()` |
| [`install_runner_hooks.prepare`](../src/blockspec/vllm_ascend/qwen38_runtime.py#L134) | 包装输入准备：区分 prefill 和没有候选的 decode，让 GDN 正确延续上次接受状态。<br/>参数：`self, *args, **kwargs` | `prepare_inputs()`<br/>`_ours()`<br/>`self.num_decode_draft_tokens.np.fill()`<br/>`range()`<br/>`self.num_decode_draft_tokens.copy_to_gpu()`<br/>`functools.wraps()` |
| [`install_runner_hooks.verify`](../src/blockspec/vllm_ascend/qwen38_runtime.py#L151) | 在 runtime 中包装原生采样；在独立 ProposalSampler 中则是参考校正。以所在类/模块为准，不是两套同时调用。<br/>参数：`self, logits, spec_decode_metadata` | `_ours()`<br/>`sample()`<br/>`proposer.teacher_rows()`<br/>`proposer.verification_probs()`<br/>`_DraftProbabilities()`<br/>`proposer.observe_verified()`<br/>`functools.wraps()` |

## src/blockspec/vllm_ascend/qwen38_weights.py

权重与 TP 适配层：引用原生 backbone 参数，复制 draft mixer，并为冻结权重输入梯度、TP collective 和 learner 提供适配。

| 函数入口 | 职责与输入 | 直接调用线索 |
| --- | --- | --- |
| [`TensorParallel.__init__`](../src/blockspec/vllm_ascend/qwen38_weights.py#L41) | 记录设备通信组、全局 rank 列表、组大小和本 rank 的组内编号。<br/>参数：`self, group=None, ranks=(0,)` | `tuple()`<br/>`len()`<br/>`dist.get_rank()` |
| [`TensorParallel.__deepcopy__`](../src/blockspec/vllm_ascend/qwen38_weights.py#L48) | 控制深复制的共享语义；返回 self 表示对象继续共享，而非创建新的通信组/缓存。<br/>参数：`self, memo` | 无直接函数调用 |
| [`TensorParallel.reduce`](../src/blockspec/vllm_ascend/qwen38_weights.py#L54) | TP all-reduce 求和；形状不变，部分结果变成各 rank 一致的完整结果。<br/>参数：`self, value` | `dist.all_reduce()` |
| [`TensorParallel.gather`](../src/blockspec/vllm_ascend/qwen38_weights.py#L64) | 把各 rank 最后一维的分片 all-gather 后拼接；常用于重建完整词表 logits。<br/>参数：`self, value` | `torch.empty_like()`<br/>`dist.all_gather()`<br/>`value.contiguous()`<br/>`torch.cat()` |
| [`TensorParallel.broadcast`](../src/blockspec/vllm_ascend/qwen38_weights.py#L76) | 用首个 TP rank 的内容同步其他 rank，确保候选 token 一致。<br/>参数：`self, value` | `dist.broadcast()` |
| [`_Copy.forward`](../src/blockspec/vllm_ascend/qwen38_weights.py#L91) | column/vocab 并行输入在前向保持不变，记录 TP 组供反向归并。<br/>参数：`ctx, value, tp` | 无直接函数调用 |
| [`_Copy.backward`](../src/blockspec/vllm_ascend/qwen38_weights.py#L100) | 把不同列分片对同一输入的梯度 all-reduce；第二返回值 None 对应无梯度的 tp 对象。<br/>参数：`ctx, gradient` | `ctx.tp.reduce()`<br/>`gradient.contiguous().clone()`<br/>`gradient.contiguous()` |
| [`_Reduce.forward`](../src/blockspec/vllm_ascend/qwen38_weights.py#L112) | row 并行输出在前向 all-reduce 相加；先 clone，避免原地改动输入。<br/>参数：`ctx, value, tp` | `tp.reduce()`<br/>`value.clone()` |
| [`_Reduce.backward`](../src/blockspec/vllm_ascend/qwen38_weights.py#L118) | 把共同输出的梯度直接传回本 rank 的局部投影输入，不再重复求和。<br/>参数：`ctx, gradient` | 无直接函数调用 |
| [`_Gather.forward`](../src/blockspec/vllm_ascend/qwen38_weights.py#L130) | 在词表最后一维 all-gather 局部分数，形成完整词表 logits。<br/>参数：`ctx, value, tp` | `tp.gather()` |
| [`_Gather.backward`](../src/blockspec/vllm_ascend/qwen38_weights.py#L138) | 取出全局梯度属于本 rank 词表片段的部分，不把其他 rank 的分片用于本地反向。<br/>参数：`ctx, gradient` | `gradient.chunk(ctx.tp.size, dim=-1)[ctx.tp.rank].contiguous()`<br/>`gradient.chunk()` |
| [`_FrozenLinear.forward`](../src/blockspec/vllm_ascend/qwen38_weights.py#L150) | 计算 y=xWᵀ，只保存冻结权重的只读引用，避免 inference tensor 的 save_for_backward 限制。<br/>参数：`ctx, value, weight` | `F.linear()` |
| [`_FrozenLinear.backward`](../src/blockspec/vllm_ascend/qwen38_weights.py#L158) | 返回输入梯度 dL/dx=dL/dy·W，权重梯度返回 None；冻结不等于阻断输入梯度。<br/>参数：`ctx, gradient` | 无直接函数调用 |
| [`ParallelLinear.__init__`](../src/blockspec/vllm_ascend/qwen38_weights.py#L169) | 包装已经存在的 TP 权重引用，并记住 column/row/vocab 并行模式，不重新从磁盘加载矩阵。<br/>参数：`self, weight, tp, mode` | `super().__init__()`<br/>`super()`<br/>`isinstance()`<br/>`nn.Parameter()` |
| [`ParallelLinear._project`](../src/blockspec/vllm_ascend/qwen38_weights.py#L178) | 执行本 rank 的线性投影；处理 column/vocab 输入梯度归并与冻结 inference tensor 的输入反向。<br/>参数：`self, value` | `_Copy.apply()`<br/>`torch.is_inference()`<br/>`torch.is_grad_enabled()`<br/>`RuntimeError()`<br/>`_FrozenLinear.apply()`<br/>`F.linear()` |
| [`ParallelLinear.forward`](../src/blockspec/vllm_ascend/qwen38_weights.py#L199) | 本地矩阵乘后按 row/vocab 模式求和或拼接，column 模式保持局部输出。<br/>参数：`self, value` | `self._project()`<br/>`_Reduce.apply()`<br/>`_Gather.apply()` |
| [`SharedRMSNorm.__init__`](../src/blockspec/vllm_ascend/qwen38_weights.py#L217) | 引用冻结 norm 权重，保存 epsilon 与 residual 舍入模式；必要时建立全 1 的辅助 buffer。<br/>参数：`self, weight, eps, *, native_residual=False` | `super().__init__()`<br/>`super()`<br/>`torch.inference_mode()`<br/>`self.register_buffer()`<br/>`torch.ones_like()` |
| [`SharedRMSNorm.forward`](../src/blockspec/vllm_ascend/qwen38_weights.py#L229) | 用共享冻结 norm 参数做归一化，匹配原生 residual RMSNorm 的 BF16 舍入，训练保留输入梯度。<br/>参数：`self, value` | `torch.is_inference()`<br/>`torch.is_grad_enabled()`<br/>`weight.clone()`<br/>`torch_npu.npu_rms_norm()`<br/>`value.float()`<br/>`torch.rsqrt()`<br/>`value.float().square().mean()`<br/>`value.float().square()`<br/>`(normalized * (1 + weight.float())).to()`<br/>`weight.float()` |
| [`SharedMLP.__init__`](../src/blockspec/vllm_ascend/qwen38_weights.py#L259) | 用共享 gate/up 与 down 矩阵构造列并行/行并行线性层。<br/>参数：`self, gate_up, down, tp` | `super().__init__()`<br/>`super()`<br/>`ParallelLinear()` |
| [`SharedMLP.forward`](../src/blockspec/vllm_ascend/qwen38_weights.py#L267) | 共享列并行 gate/up 投影→SwiGLU→行并行 down 投影；共享权重仍执行计算。<br/>参数：`self, value` | `self.down_proj()`<br/>`swiglu()`<br/>`self.gate_up_proj()` |
| [`SharedEmbedding.__init__`](../src/blockspec/vllm_ascend/qwen38_weights.py#L277) | 引用原生词表分片权重，保存该 rank 的 token 起止 ID；当前不接受新增 LoRA 词表。<br/>参数：`self, original, tp, vocab_size` | `super().__init__()`<br/>`super()`<br/>`getattr()`<br/>`ValueError()` |
| [`SharedEmbedding.forward`](../src/blockspec/vllm_ascend/qwen38_weights.py#L291) | 全局 token ID 映射到本 rank 的词表分片；非本地 token 贡献置零，再求和得到 embedding。<br/>参数：`self, tokens` | `(tokens - self.start).masked_fill()`<br/>`F.embedding(local, self.weight).masked_fill()`<br/>`F.embedding()`<br/>`self.tp.reduce()` |
| [`SharedHead.__init__`](../src/blockspec/vllm_ascend/qwen38_weights.py#L306) | 建立共享词表并行 LM head，并记录真实词表宽度以裁掉 padding。<br/>参数：`self, weight, tp, vocab_size` | `super().__init__()`<br/>`super()` |
| [`SharedHead.forward`](../src/blockspec/vllm_ascend/qwen38_weights.py#L313) | 调用词表并行线性层，拼接后裁掉 padding，输出完整有效词表分数。<br/>参数：`self, value` | `super().forward()`<br/>`super()` |
| [`SharedHead.vocab_start`](../src/blockspec/vllm_ascend/qwen38_weights.py#L319) | 返回当前 rank 分片对应的全局词表起点，用于 teacher 切片与全局 token 对齐。<br/>参数：`self` | `min()` |
| [`SharedHead.local_logits`](../src/blockspec/vllm_ascend/qwen38_weights.py#L325) | 只计算本 rank 的有效词表分片，去除 padding；供词表并行 loss 避免完整 logits gather。<br/>参数：`self, value` | `max()`<br/>`min()`<br/>`self._project()` |
| [`_weight`](../src/blockspec/vllm_ascend/qwen38_weights.py#L333) | 确认当前原生线性权重是可直接使用的二维 BF16/FP32；量化 packed 权重不能走此映射。<br/>参数：`module` | `ValueError()` |
| [`from_native_model`](../src/blockspec/vllm_ascend/qwen38_weights.py#L348) | 用 meta 模型骨架承接 vLLM 原生 TP 参数；共享 backbone 后构造 Qwen38DualView，检查没有遗漏 meta 权重。<br/>参数：`target, config, *, tp, block_size, mask_token_id, training_gdn_backend='torch'` | `getattr()`<br/>`ValueError()`<br/>`Qwen3_5TextConfig()`<br/>`config.to_dict()`<br/>`copy.deepcopy()`<br/>`setattr()`<br/>`torch.device()`<br/>`Qwen3_5ForCausalLM()`<br/>`SharedEmbedding()`<br/>`SharedRMSNorm()`；另有 17 项，见源码 |
| [`NativeSuffixLearner.__init__`](../src/blockspec/vllm_ascend/qwen38_weights.py#L441) | 在通用 learner 上增加 TP 组、可选词表分片 loss、复制 norm 参数清单，以及 CPU 分支的融合优化器。<br/>参数：`self, model, config, tp, *, vocab_parallel_loss=False` | `super().__init__()`<br/>`super()`<br/>`torch.optim.AdamW()` |
| [`NativeSuffixLearner.replay_logits`](../src/blockspec/vllm_ascend/qwen38_weights.py#L456) | 用保存的 boundary/history 重算学生后缀，选取有效预测行；native 可覆盖为词表分片实现。<br/>参数：`self, item` | `super().replay_logits()`<br/>`super()`<br/>`self.model.forward_suffix()` |
| [`NativeSuffixLearner.replay_divergence`](../src/blockspec/vllm_ascend/qwen38_weights.py#L466) | 按配置比较 student 与 teacher 概率；不是与 token ID 做均方差。<br/>参数：`self, student, teacher` | `super().replay_divergence()`<br/>`super()`<br/>`vocab_divergence()` |
| [`NativeSuffixLearner.parameter_stamp`](../src/blockspec/vllm_ascend/qwen38_weights.py#L477) | 记录参数版本/身份供冻结保护；NativeSuffixLearner 覆盖以兼容 inference tensor。<br/>参数：`parameter` | `torch.is_inference()`<br/>`parameter.data_ptr()`<br/>`tuple()` |
| [`NativeSuffixLearner.fingerprint`](../src/blockspec/vllm_ascend/qwen38_weights.py#L484) | 建立冻结部分身份；native 版本使用进程内参数身份，验证脚本的完整内容哈希是另一种更强的审计。<br/>参数：`self` | `str()`<br/>`self.parameter_stamp()`<br/>`self.model.named_parameters()`<br/>`hashlib.sha256(json.dumps(identity).encode()).hexdigest()`<br/>`hashlib.sha256()`<br/>`json.dumps(identity).encode()`<br/>`json.dumps()` |
| [`NativeSuffixLearner.backward`](../src/blockspec/vllm_ascend/qwen38_weights.py#L493) | 先调用通用后缀反向，再把 TP 中复制的 norm 参数梯度打包 all-reduce，保持各 rank 更新一致。<br/>参数：`self` | `super().backward()`<br/>`super()`<br/>`sorted()`<br/>`next()`<br/>`iter()`<br/>`self.execution.values()`<br/>`torch.cat([gradient.reshape(-1) for gradient in gradients]).to()`<br/>`torch.cat()`<br/>`gradient.reshape()`<br/>`self.tp.reduce(packed).to()`；另有 5 项，见源码 |
| [`NativeSuffixLearner.clip_gradients`](../src/blockspec/vllm_ascend/qwen38_weights.py#L514) | 计算全局梯度范数并按阈值缩放；TP 版本排除复制参数的重复计数后跨 rank 归并。<br/>参数：`self` | `torch.zeros()`<br/>`value.grad.float()`<br/>`self.master.items()`<br/>`torch.stack(torch._foreach_norm(gradients, 2)).square().sum()`<br/>`torch.stack(torch._foreach_norm(gradients, 2)).square()`<br/>`torch.stack()`<br/>`torch._foreach_norm()`<br/>`torch.linalg.vector_norm(gradient, dim=-1).square().sum()`<br/>`torch.linalg.vector_norm(gradient, dim=-1).square()`<br/>`torch.linalg.vector_norm()`；另有 13 项，见源码 |
