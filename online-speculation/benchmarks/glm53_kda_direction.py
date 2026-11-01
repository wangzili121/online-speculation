#!/usr/bin/env python3
"""KDA direction ablation; synthetic hybrid network, NOT a GLM head trainer."""

import argparse
import copy
import json
from pathlib import Path
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F

from blockspec.vllm_ascend.glm53_ops import finite_error, kda_native, kda_reference


def reverse_core(query, key, value, gate, beta, initial, a_log, bias, *, native=False):
    """只反向遍历临时块，从零状态开始；绝不反向遍历真实历史。"""
    core = kda_native if native else kda_reference
    output, _ = core(query.flip(0), key.flip(0), value.flip(0), gate.flip(0), beta.flip(0),
                     torch.zeros_like(initial), a_log, bias)
    return output.flip(0)


class KDA(nn.Module):
    def __init__(self, width=32, heads=2, dim=16):
        super().__init__()
        self.heads, self.dim = heads, dim
        self.qkvb = nn.Linear(width, 3 * width + heads, bias=False)
        self.f_a = nn.Linear(width, dim, bias=False)
        self.f_b = nn.Linear(dim, width, bias=False)
        self.g_a = nn.Linear(width, dim, bias=False)
        self.g_b = nn.Linear(dim, width, bias=False)
        self.conv = nn.Conv1d(3 * width, 3 * width, 4, groups=3 * width, bias=False)
        self.a_log = nn.Parameter(torch.zeros(heads))
        self.bias = nn.Parameter(torch.full((width,), -3.))
        self.norm = nn.RMSNorm(dim)
        self.out = nn.Linear(width, width, bias=False)
        self.register_parameter("backward_gate", None)

    def forward(self, hidden, history=None, *, causal=True):
        batch, length, width = hidden.shape
        qkv, beta = self.qkvb(hidden).split((3 * width, self.heads), -1)
        tail, initial = (history if history is not None else
                         (hidden.new_zeros(batch, 3, 3 * width),
                          hidden.new_zeros(batch, self.heads, self.dim, self.dim)))
        gate = self.f_b(self.f_a(hidden)).reshape(batch, length, self.heads, self.dim)

        def convolve(rows, left):
            stream = torch.cat((left, rows), 1)
            mixed = F.silu(self.conv(stream.transpose(1, 2))).transpose(1, 2)
            return mixed.reshape(batch, length, 3, self.heads, self.dim).unbind(2)

        def flatten(rows):
            return rows.transpose(0, 1).reshape(length, batch * self.heads, self.dim)

        q, k, v = convolve(qkv, tail)
        log = self.a_log.repeat(batch)
        bias = self.bias.repeat(batch)
        state = initial.reshape(batch * self.heads, self.dim, self.dim)
        beta_flat = beta.transpose(0, 1).reshape(length, batch * self.heads)
        output, checkpoints = kda_reference(flatten(q), flatten(k), flatten(v), flatten(gate),
                                            beta_flat, state, log, bias)
        if self.backward_gate is not None and not causal:
            # 实验候选：共享方向权重，反向卷积不带 AR 左历史，反向递推从零开始。
            rq, rk, rv = convolve(qkv.flip(1), torch.zeros_like(tail))
            reverse, _ = kda_reference(flatten(rq), flatten(rk), flatten(rv), flatten(gate.flip(1)),
                                       beta_flat.flip(0), torch.zeros_like(state), log, bias)
            weight = self.backward_gate.tanh().repeat(batch).reshape(1, -1, 1)
            output = output + weight * reverse.flip(0)
        output = output.reshape(length, batch, self.heads, self.dim).transpose(0, 1)
        out_gate = self.g_b(self.g_a(hidden)).reshape_as(output).sigmoid()
        mixed = self.out((self.norm(output) * out_gate).flatten(2))
        final = checkpoints[-1].reshape_as(initial)
        return mixed, (torch.cat((tail, qkv), 1)[:, -3:], final)


class MLA(nn.Module):
    def __init__(self, width=32, heads=2, dim=16, rank=16):
        super().__init__()
        self.heads, self.dim = heads, dim
        self.qa = nn.Linear(width, rank, bias=False)
        self.qnorm = nn.RMSNorm(rank)
        self.qb = nn.Linear(rank, width, bias=False)
        self.kva = nn.Linear(width, rank, bias=False)
        self.kvnorm = nn.RMSNorm(rank)
        self.kb = nn.Linear(rank, width, bias=False)
        self.vb = nn.Linear(rank, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, hidden, history=None, *, causal=True):
        batch, length, _ = hidden.shape
        def split(rows):
            return rows.reshape(batch, length, self.heads, self.dim).transpose(1, 2)
        q = split(self.qb(self.qnorm(self.qa(hidden))))
        latent = self.kvnorm(self.kva(hidden))
        k, v = split(self.kb(latent)), split(self.vb(latent))
        past = 0 if history is None else history[0].shape[2]
        if history is not None:
            k, v = torch.cat((history[0], k), 2), torch.cat((history[1], v), 2)
        allowed = None
        if causal:
            allowed = torch.arange(past + length)[None] <= past + torch.arange(length)[:, None]
        # NoPE；短历史低于 indexer Top-K，上下文全部可见。
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        return self.out(output.transpose(1, 2).reshape_as(hidden)), (k, v)


class Layer(nn.Module):
    def __init__(self, mixer, width=32):
        super().__init__()
        self.mixer = mixer
        self.input_norm, self.post_norm = nn.RMSNorm(width), nn.RMSNorm(width)
        self.up, self.gate = nn.Linear(width, 64, bias=False), nn.Linear(width, 64, bias=False)
        self.down = nn.Linear(64, width, bias=False)

    def forward(self, hidden, history=None, *, causal=True):
        mixed, cache = self.mixer(self.input_norm(hidden), history, causal=causal)
        hidden = hidden + mixed
        normalized = self.post_norm(hidden)
        return hidden + self.down(F.silu(self.gate(normalized)) * self.up(normalized)), cache


class Hybrid(nn.Module):
    def __init__(self, vocab=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab, 32)
        self.layers = nn.ModuleList(Layer(mixer) for mixer in (KDA(), KDA(), KDA(), MLA(), KDA()))
        self.norm, self.head = nn.RMSNorm(32), nn.Linear(32, vocab, bias=False)

    def forward(self, tokens, history=None, *, causal=True):
        hidden = self.embedding(tokens)
        states = []
        for index, layer in enumerate(self.layers):
            hidden, state = layer(hidden, None if history is None else history[index], causal=causal)
            states.append(state)
        return self.head(self.norm(hidden)), states


def make_student(teacher, bidirectional):
    student = copy.deepcopy(teacher)
    student.embedding, student.norm, student.head = teacher.embedding, teacher.norm, teacher.head
    for layer, original in zip(student.layers, teacher.layers, strict=True):
        for name in ("input_norm", "post_norm", "up", "gate", "down"):
            setattr(layer, name, getattr(original, name))
        if bidirectional and isinstance(layer.mixer, KDA):
            # 零门控使两种学生在第零步完全相同，不用初始化漂移来比较优劣。
            layer.mixer.backward_gate = nn.Parameter(torch.zeros(layer.mixer.heads))
    student.requires_grad_(False)
    for layer in student.layers:
        layer.mixer.requires_grad_(True)
    return student


def dataset(seed, length=24):
    generator = torch.Generator().manual_seed(seed)
    # 自包含的合成序列；不是 GLM 业务数据，不用于宣称真实接受率。
    cases = torch.tensor([(start, step) for start in range(31) for step in (1, 2, 3, 5)])
    cases = cases[torch.randperm(len(cases), generator=generator)]
    rows = 1 + (cases[:, :1] + cases[:, 1:] * torch.arange(length)) % 31
    return rows[:96], rows[96:]


@torch.no_grad()
def prepare(teacher, rows, block_size=8):
    logits, _ = teacher(rows)
    batches = []
    for anchor in (4, 8, 12):
        _, history = teacher(rows[:, :anchor])
        inputs = torch.zeros(rows.shape[0], block_size, dtype=torch.long)
        inputs[:, 0] = rows[:, anchor]
        probabilities = logits[:, anchor:anchor + block_size - 1].softmax(-1)
        batches.append((inputs, history, probabilities))
    return batches


def metrics(logits, expected):
    log = logits.log_softmax(-1)
    return dict(kl=float(F.kl_div(log, expected, reduction="none").sum(-1).mean()),
                tv=float((log.exp() - expected).abs().sum(-1).mean() * .5),
                teacher_forced_top1_agreement=float((log.argmax(-1) == expected.argmax(-1)).float().mean()))


@torch.no_grad()
def evaluate(student, batches):
    outputs, targets = [], []
    for inputs, history, probabilities in batches:
        outputs.append(student(inputs, history, causal=False)[0][:, :-1])
        targets.append(probabilities)
    return metrics(torch.cat(outputs), torch.cat(targets))


def train_trial(seed, teacher_steps, draft_steps):
    torch.manual_seed(seed)
    teacher = Hybrid()
    train, heldout = dataset(seed + 1000)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=3e-3)
    curve = []
    for step in range(teacher_steps):
        indices = torch.randint(len(train), (8,))
        logits, _ = teacher(train[indices])
        loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), train[indices, 1:].flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.)
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == teacher_steps:
            curve.append(dict(step=step + 1, nll=float(loss.detach())))
            print(json.dumps(dict(seed=seed, phase="teacher", **curve[-1])), flush=True)
    teacher.eval().requires_grad_(False)
    before = {name: parameter.detach().clone() for name, parameter in teacher.named_parameters()}
    training, validation = prepare(teacher, train), prepare(teacher, heldout)
    with torch.no_grad():
        logits, _ = teacher(heldout)
        heldout_nll = float(F.cross_entropy(logits[:, :-1].flatten(0, 1), heldout[:, 1:].flatten()))
    results = {}
    for name in ("causal", "residual_bidir"):
        student = make_student(teacher, name != "causal")
        parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=1e-3)
        # 两种模式使用相同抽样顺序、相同批量和步数，验证集不参与更新。
        generator = torch.Generator().manual_seed(seed + 2000)
        initial = evaluate(student, validation)
        draft_curve = []
        start = time.monotonic()
        for step in range(draft_steps):
            inputs, history, probabilities = training[step % len(training)]
            indices = torch.randint(len(inputs), (8,), generator=generator)
            selected = [tuple(state[indices] for state in pair) for pair in history]
            logits, _ = student(inputs[indices], selected, causal=False)
            loss = F.kl_div(logits[:, :-1].log_softmax(-1), probabilities[indices],
                            reduction="none").sum(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            if step == 0 or (step + 1) % 50 == 0 or step + 1 == draft_steps:
                draft_curve.append(dict(step=step + 1, train_kl=float(loss.detach())))
                print(json.dumps(dict(seed=seed, phase=name, **draft_curve[-1])), flush=True)
        gates = [layer.mixer.backward_gate.tanh().detach().tolist() for layer in student.layers
                 if isinstance(layer.mixer, KDA) and layer.mixer.backward_gate is not None]
        results[name] = dict(initial=initial, heldout=evaluate(student, validation),
                             trainable_parameters=sum(p.numel() for p in parameters),
                             train_seconds=time.monotonic() - start, curve=draft_curve, gates=gates)
        assert all(torch.equal(before[name], parameter) for name, parameter in teacher.named_parameters())
    return dict(seed=seed, teacher_curve=curve, teacher_heldout_nll=heldout_nll, students=results)


@torch.no_grad()
def native_probe(repeats):
    import torch_npu  # noqa: F401
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(0)
    if not enable_custom_op():
        raise RuntimeError("Ascend custom operators unavailable")
    torch.manual_seed(91)
    def random(*shape):
        return (torch.randn(*shape, device="npu", dtype=torch.bfloat16) * .15).contiguous()
    q, k, v, g = [random(8, 8, 128) for _ in range(4)]
    beta = random(8, 8)
    initial = torch.randn(8, 128, 128, device="npu") * .02
    a_log, bias = torch.zeros(8, device="npu"), torch.full((1024,), -3., device="npu")
    original = initial.clone()
    def run(reverse, changed=False, native=True):
        values = v.clone()
        if changed:
            values[-1] += .5
        core = kda_native if native else kda_reference
        output, _ = core(q, k, values, g, beta, initial, a_log, bias)
        if reverse:
            output = output + .5 * reverse_core(q, k, values, g, beta, initial, a_log, bias, native=native)
        return output
    reports = {}
    for name, reverse in (("causal", False), ("residual_bidir", True)):
        actual = run(reverse)
        reference = run(reverse, native=False)
        error = finite_error(actual, reference)
        torch.testing.assert_close(actual, reference, rtol=.02, atol=1e-5)
        influence = float((run(reverse, changed=True)[0].float() - actual[0].float()).abs().max())
        assert (influence > 0) == reverse
        for _ in range(5):
            run(reverse)
        torch.npu.synchronize()
        times = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(repeats):
                run(reverse)
            torch.npu.synchronize()
            times.append((time.perf_counter() - start) * 1000 / repeats)
        reports[name] = dict(error=error, last_position_influence_on_first=influence,
                             median_adapter_ms=statistics.median(times))
    torch.testing.assert_close(initial, original, rtol=0, atol=0)
    return dict(results=reports, shape=dict(block_size=8, local_heads=8, head_dim=128),
                initial_state_unchanged=True,
                scope="projected KDA core adapter only, not full layer or end-to-end throughput")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("train", "npu"), default="train")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--teacher-steps", type=int, default=300)
    parser.add_argument("--draft-steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if min(args.teacher_steps, args.draft_steps, args.repeats) <= 0:
        parser.error("training steps and timing repeats must be positive")
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.case == "npu":
        report = native_probe(args.repeats)
    else:
        report = dict(scope="synthetic NoPE KDA/MLA hybrid; no GLM checkpoint, mHC or MoE; not acceptance evidence",
                      settings=vars(args) | {"output": str(args.output)}, trials=[])
        for seed in args.seeds:
            trial = train_trial(seed, args.teacher_steps, args.draft_steps)
            report["trials"].append(trial)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(trial), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), case=args.case)), flush=True)


if __name__ == "__main__":
    main()
