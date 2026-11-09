"""last-1 Full Attention 的显式前向/反向图；不依赖自动微分生成反向。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SuffixShape:
    hidden: int = 5120
    intermediate: int = 17408
    vocabulary: int = 248320
    q_heads: int = 24
    kv_heads: int = 4
    dimension: int = 256
    context: int = 512
    rotary: int = 64
    block: int = 32

    def __post_init__(self):
        if (self.block != 32 or self.q_heads % self.kv_heads or self.rotary % 32
                or self.rotary > self.dimension or any(x % 64 for x in
                (self.hidden, self.intermediate, self.vocabulary, self.dimension, self.context))):
            raise ValueError("aligned K32 suffix and complete GQA/RoPE groups required")


@dataclass(frozen=True)
class Value:
    name: str
    shape: tuple
    dtype: str = "bf16"


@dataclass(frozen=True)
class Node:
    kind: str
    inputs: tuple
    outputs: tuple
    options: dict


class LastLayerGraph:
    """只计算最后一层。冻结前缀在 hidden 处截断，历史 K/V 是常量。"""

    def __init__(self, shape=SuffixShape()):
        s = self.shape = shape
        self.inputs, self.nodes = {}, []
        K, H, D, Q, V = s.block, s.hidden, s.dimension, s.q_heads, s.kv_heads
        def input(name, rows, columns, dtype="bf16"):
            value = Value(name, (rows, columns), dtype)
            self.inputs[name] = value
            return value
        hidden = input("hidden", K, H)
        ng = input("input_gamma", 1, H, "fp32")
        pg = input("post_gamma", 1, H, "fp32")
        fg = input("final_gamma", 1, H, "fp32")
        wg, wu, wd = (input("mlp_gate", s.intermediate, H), input("mlp_up", s.intermediate, H),
                      input("mlp_down", H, s.intermediate))
        lm = input("lm_head", s.vocabulary, H)
        qw, kw, vw, ow = (input("q_weight", 2 * Q * D, H), input("k_weight", V * D, H),
                          input("v_weight", V * D, H), input("o_weight", H, Q * D))
        qg, kg = input("q_gamma", 1, D, "fp32"), input("k_gamma", 1, D, "fp32")
        hk, hv = input("history_key", V * s.context, D), input("history_value", V * s.context, D)
        cos, sin = input("cos", K, s.rotary), input("sin", K, s.rotary)
        teacher = input("teacher", K, s.vocabulary)
        x, _ = self.norm(hidden, ng)
        qp, kp, vp = self.mm(x, qw), self.mm(x, kw), self.mm(x, vw)
        qraw = self.layout(qp, Q, mode="pack", interleaved=True)
        gate = self.layout(qp, Q, mode="pack", interleaved=True, gate=True)
        kraw, value = self.layout(kp, V, mode="pack"), self.layout(vp, V, mode="pack")
        qnorm, qi = self.norm(qraw, qg)
        knorm, ki = self.norm(kraw, kg)
        q, k = self.rope(qnorm, cos, sin), self.rope(knorm, cos, sin)
        allk, allv = self.join(k, hk), self.join(value, hv)
        scores = self.mm(q, allk, batches=Q, b_repeats=Q // V)
        scaled = self.element(scores, scores, "scale", scale=D ** -.5)
        probabilities, rounded = self.op("softmax", (scaled,),
            (((Q * K, s.context), "fp32"), ((Q * K, s.context), "bf16")))
        attended = self.mm(rounded, allv, transpose_b=False, batches=Q, b_repeats=Q // V)
        sigmoid = self.element(gate, gate, "sigmoid")
        gated = self.element(attended, sigmoid, "mul")
        packed = self.layout(gated, Q, mode="unpack")
        output = self.mm(packed, ow)
        skip = self.element(hidden, output, "add")
        post, pi = self.norm(skip, pg)
        pre_gate, up = self.mm(post, wg), self.mm(post, wu)
        silu = self.element(pre_gate, pre_gate, "silu")
        middle = self.element(silu, up, "mul")
        mlp = self.mm(middle, wd)
        final = self.element(skip, mlp, "add")
        normalized, fi = self.norm(final, fg)
        self.logits = logits = self.mm(normalized, lm)
        delta, self.losses = self.op("tv", (logits, teacher),
            (((K, s.vocabulary), "bf16"), ((K, 8), "fp32")))

        # 手工链式求导。共享 MLP/LM head 只算 dX，不产生 dW。
        dfinal = self.norm_dx(final, fg, fi, self.mm(delta, lm, transpose_b=False))
        dmiddle = self.mm(dfinal, wd, transpose_b=False)
        dup = self.element(dmiddle, silu, "mul")
        dsilu = self.element(dmiddle, up, "mul")
        dgate_mlp = self.element(pre_gate, dsilu, "silu_grad")
        dpost = self.element(self.mm(dup, wu, transpose_b=False), self.mm(dgate_mlp, wg, transpose_b=False), "add")
        dskip = self.element(dfinal, self.norm_dx(skip, pg, pi, dpost), "add")
        self.gradients = {"o_weight": self.mm(dskip, packed, transpose_a=True, transpose_b=False, gradient_weight=True)}
        dgated = self.layout(self.mm(dskip, ow, transpose_b=False), Q, mode="pack")
        dattn = self.element(dgated, sigmoid, "mul")
        dactivated = self.element(dgated, attended, "mul")
        dgate = self.element(gate, dactivated, "sigmoid_grad")
        dv_heads = self.mm(rounded, dattn, transpose_a=True, transpose_b=False, batches=Q)
        dp = self.mm(dattn, allv, batches=Q, b_repeats=Q // V)
        ds = self.op("softmax_grad", (probabilities, dp), ((scaled.shape, "bf16"),))[0]
        ds = self.element(ds, ds, "scale", scale=D ** -.5)
        dq = self.mm(ds, allk, transpose_b=False, batches=Q, b_repeats=Q // V)
        dk_heads = self.mm(ds, q, transpose_a=True, transpose_b=False, batches=Q)
        dk, dv = self.reduce_block(dk_heads), self.reduce_block(dv_heads)
        dqnorm, dknorm = self.rope(dq, cos, sin, backward=True), self.rope(dk, cos, sin, backward=True)
        self.gradients["q_gamma"] = self.norm_dw(qraw, qi, dqnorm)
        self.gradients["k_gamma"] = self.norm_dw(kraw, ki, dknorm)
        dqraw = self.norm_dx(qraw, qg, qi, dqnorm)
        dkraw = self.norm_dx(kraw, kg, ki, dknorm)
        dqp = self.op("q_join", (dqraw, dgate), ((qp.shape, "bf16"),))[0]
        dkp, dvp = self.layout(dkraw, V, mode="unpack"), self.layout(dv, V, mode="unpack")
        for name, grad in (("q_weight", dqp), ("k_weight", dkp), ("v_weight", dvp)):
            self.gradients[name] = self.mm(grad, x, transpose_a=True, transpose_b=False, gradient_weight=True)

    def op(self, operation, inputs, outputs, **options):
        values = tuple(Value(f"t{len(self.nodes)}_{i}", shape, dtype) for i, (shape, dtype) in enumerate(outputs))
        self.nodes.append(Node(operation, tuple(inputs), values, options))
        return values

    def mm(self, a, b, *, transpose_a=False, transpose_b=True, gradient_weight=False, batches=1, b_repeats=1):
        ar = a.shape[0] // batches
        br = b.shape[0] // (batches // b_repeats)
        m, k = (a.shape[1], ar) if transpose_a else (ar, a.shape[1])
        kb, n = (b.shape[1], br) if transpose_b else (br, b.shape[1])
        if k != kb:
            raise ValueError((a, b, transpose_a, transpose_b))
        return self.op("linear", (a, b), ((((batches * m, n)), "fp32" if gradient_weight else "bf16"),),
                       a_shape=(ar, a.shape[1]), b_shape=(br, b.shape[1]), transpose_a=transpose_a,
                       transpose_b=transpose_b, gradient_weight=gradient_weight, batches=batches, b_repeats=b_repeats)[0]

    def element(self, a, b, kind, **options):
        return self.op("element", (a, b), ((a.shape, "bf16"),), kind=kind, **options)[0]

    def norm(self, x, gamma):
        return self.op("norm", (x, gamma), ((x.shape, "bf16"), ((x.shape[0], 8), "fp32")))

    def norm_dx(self, x, gamma, inv, dy):
        return self.op("norm_dx", (x, gamma, inv, dy), ((x.shape, "bf16"),))[0]

    def norm_dw(self, x, inv, dy):
        return self.op("norm_dw", (x, inv, dy), (((1, x.shape[1]), "fp32"),))[0]

    def layout(self, x, heads, *, mode, interleaved=False, gate=False):
        s = self.shape
        out = (heads * s.block, s.dimension) if mode == "pack" else (s.block, heads * s.dimension)
        return self.op("layout", (x,), ((out, "bf16"),), heads=heads, mode=mode, interleaved=interleaved, gate=gate)[0]

    def rope(self, x, cos, sin, *, backward=False):
        return self.op("rope", (x, cos, sin), ((x.shape, "bf16"),), backward=backward)[0]

    def join(self, block, history):
        return self.op("join", (block, history), ((history.shape, "bf16"),))[0]

    def reduce_block(self, x):
        return self.op("reduce_block", (x,), (((self.shape.kv_heads * self.shape.block, self.shape.dimension), "bf16"),))[0]
