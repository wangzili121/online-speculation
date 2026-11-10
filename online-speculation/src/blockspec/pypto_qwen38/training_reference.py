"""独立 CPU Torch golden：解释训练图，autograd 只由验证代码调用。"""


def evaluate(graph, inputs, *, past, width, valid, normalizer=None, fp32=False):
    import torch
    from torch.nn import functional as F
    s = graph.shape
    normalizer = valid if normalizer is None else normalizer
    if normalizer < valid:
        raise ValueError("normalizer must include all reached positions")
    if not 0 <= past <= s.context - width or not 2 <= width <= s.block or not 1 <= valid < width:
        raise ValueError("aligned history/block and reached teacher rows required")
    values = dict(inputs)
    cast = (lambda x: x) if fp32 else (lambda x: x.bfloat16())
    for name, spec in graph.inputs.items():
        if tuple(values[name].shape) != spec.shape:
            raise ValueError(name)

    def execute(node, args):
        op, cfg = node.kind, node.options
        if op.startswith("gdn_"):
            return gdn_execute(op,args,graph.gdn_shape,width,cast)
        if op == "linear":
            a, b = args
            batches, repeats = cfg["batches"], cfg["b_repeats"]
            a = a.reshape(batches, *cfg["a_shape"])
            b = b.reshape(batches // repeats, *cfg["b_shape"]).repeat_interleave(repeats, 0)
            if cfg["transpose_a"]:
                a = a.transpose(1, 2)
            if cfg["transpose_b"]:
                b = b.transpose(1, 2)
            result = cast(a @ b).reshape(node.outputs[0].shape)
            return (result.float() if cfg["gradient_weight"] else result,)
        if op == "element":
            a, b = args
            u, v = a.float(), b.float()
            kind = cfg["kind"]
            if kind == "add":
                result = u + v
            elif kind == "mul":
                result = u * v
            elif kind == "silu":
                result = F.silu(u)
            elif kind == "sigmoid":
                result = u.sigmoid()
            elif kind == "silu_grad":
                sig = u.sigmoid()
                result = v * sig * (1 + u * (1 - sig))
            elif kind == "sigmoid_grad":
                sig = u.sigmoid()
                result = v * sig * (1 - sig)
            elif kind == "scale":
                result = u * cfg["scale"]
            else:
                raise ValueError(kind)
            return (cast(result),)
        if op == "norm":
            x, gamma = args
            inv = (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
            return cast(x.float() * inv * gamma), inv.expand(-1, 8)
        if op == "norm_dx":
            x, gamma, inv, dy = args
            z, delta, r = x.float(), dy.float() * gamma, inv[:, :1]
            return (cast(delta * r - z * r.pow(3) * (delta * z).mean(-1, keepdim=True)),)
        if op == "norm_dw":
            x, inv, dy = args
            return (cast((x.float() * inv[:, :1] * dy.float()).sum(0, keepdim=True)).float(),)
        if op == "layout":
            x, = args
            heads = cfg["heads"]
            if cfg["mode"] == "pack":
                x = x.reshape(s.block, heads, -1)
                start = s.dimension if cfg["gate"] else 0
                return (x[:, :, start:start + s.dimension].transpose(0, 1).reshape(node.outputs[0].shape),)
            return (x.reshape(heads, s.block, s.dimension).transpose(0, 1).reshape(node.outputs[0].shape),)
        if op == "q_join":
            q, gate = (x.reshape(s.q_heads, s.block, s.dimension).transpose(0, 1) for x in args)
            return (torch.cat((q, gate), dim=-1).reshape(node.outputs[0].shape),)
        if op == "rope":
            x, cos, sin = args
            x = x.reshape(-1, s.block, s.dimension)
            a, b = x[..., :s.rotary // 2].float(), x[..., s.rotary // 2:s.rotary].float()
            c0, c1 = cos.float().chunk(2, -1)
            s0, s1 = sin.float().chunk(2, -1)
            if cfg["backward"]:
                first, second = a*c0 + b*s1, b*c1 - a*s0
            else:
                first, second = a*c0 - b*s0, b*c1 + a*s1
            return (torch.cat((cast(first), cast(second), x[..., s.rotary:]), -1).reshape(node.outputs[0].shape),)
        if op == "join":
            block, history = args
            old = history.reshape(s.kv_heads, s.context, s.dimension)[:, :past]
            new = block.reshape(s.kv_heads, s.block, s.dimension)[:, :width]
            zeros = block.new_zeros((s.kv_heads, s.context - past - width, s.dimension))
            return (torch.cat((old, new, zeros), 1).reshape(history.shape),)
        if op == "reduce_block":
            x, = args
            x = x.reshape(s.kv_heads, s.q_heads // s.kv_heads, s.context, s.dimension)
            result = cast(x[:, :, past:past+width].float().sum(1))
            result = F.pad(result, (0, 0, 0, s.block-width))
            return (result.reshape(node.outputs[0].shape),)
        if op == "softmax":
            x, = args
            bias = x.new_zeros((1, s.context), dtype=torch.float32)
            bias[:, past + width:] = -10000.
            probability = (x.float() + bias).softmax(-1)
            return probability, cast(probability)
        if op == "softmax_grad":
            p, dy = args
            result = p * (dy.float() - (p * dy.float()).sum(-1, keepdim=True))
            return (cast(result),)
        if op == "tv":
            x, teacher = args
            p, q = x.float().softmax(-1), teacher.float().softmax(-1)
            sign = .5 * (p - q).sign()
            gradient = p * (sign - (p * sign).sum(-1, keepdim=True)) / normalizer
            mask = (torch.arange(s.block) < valid)[:, None]
            losses = .5 * (p-q).abs().sum(-1, keepdim=True) * mask / normalizer
            return cast(gradient * mask), losses.expand(-1, 8)
        raise ValueError(op)

    for node in graph.nodes:
        result = execute(node, [values[v.name] for v in node.inputs])
        for value, tensor in zip(node.outputs, result, strict=True):
            values[value.name] = tensor
    return values


def gdn_execute(op,args,s,width,cast):
    import torch
    from . import training_gdn_reference as ref
    if op == "gdn_conv":
        return ref.convolution_forward(*args,width)
    if op == "gdn_conv_dx":
        return ref.convolution_backward(*args,width)
    if op == "gdn_gates":
        a,b,al,bias = args
        g,beta = ref.gates_forward(a[:width],b[:width],al,bias,s.heads)
        pad = lambda x: torch.nn.functional.pad(x,(0,64-s.heads,0,s.rows-width))
        return pad(g),pad(beta)
    if op == "gdn_gates_dx":
        a,b,al,bias,dg,db = args
        z = a[:width,:s.heads].float()+bias[:,:s.heads].float()
        factor = -al[:,:s.heads].float().exp()*dg[:width,:s.heads]
        pa = factor*torch.nn.functional.softplus(z)
        pb = factor*z.sigmoid()
        sig = b[:width,:s.heads].float().sigmoid()
        delta = db[:width,:s.heads].to(b.dtype).float()*sig*(1-sig)
        pad = lambda x,cols: torch.nn.functional.pad(x,(0,cols-s.heads,0,s.rows-width))
        return cast(pad(pb,128)),cast(pad(delta,128)),pad(pa,64),pad(pb,64)
    if op == "gdn_gates_dw":
        return args[0].sum(0,keepdim=True),cast(args[1].sum(0,keepdim=True)).float()
    if op == "gdn_recurrence":
        packed,g,b,initial = args
        y,saved,_ = ref.recurrence_forward(packed,g[:,:s.heads],b[:,:s.heads],initial.reshape(s.heads,s.dimension,s.dimension),width,s)
        return y.reshape(s.rows,-1),saved.reshape(-1,s.dimension)
    if op == "gdn_recurrence_dx":
        x,g,b,saved,dy = args
        return ref.recurrence_partials(x,g[:,:s.heads],b[:,:s.heads],saved.reshape(s.rows+1,s.heads,s.dimension,s.dimension),
                                       dy.reshape(s.rows,s.heads,s.dimension),width,s)
    if op == "gdn_recurrence_reduce":
        return ref.reduce_partials(*args,width,s)
    if op == "gdn_tail":
        x,z,w = args
        return ref.tail_forward(x.reshape(s.rows,s.heads,s.dimension),z,w)
    if op == "gdn_tail_dx":
        x,z,w,weighted,inv,dy = args
        # d(weight) 在单独的归约节点计算；这里保留逐元素 BF16 舍入。
        cr = x.reshape(s.rows,s.heads,s.dimension)
        normalized = (cr.reshape(-1,s.dimension).float()*inv[:,:1]).to(x.dtype)
        dx,dz,_ = ref.tail_backward(cr,z,w,normalized,weighted,inv,dy)
        dw = (dy.reshape(-1,s.dimension).float()*torch.nn.functional.silu(z.reshape(-1,s.dimension).float())).to(x.dtype)
        return dx.reshape_as(x),dz,dw
    if op == "gdn_tail_dw":
        return (cast((args[0].float()*args[1].float()).sum(0,keepdim=True)).float(),)
    raise ValueError(op)
