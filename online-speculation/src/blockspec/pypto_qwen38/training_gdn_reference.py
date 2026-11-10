"""GDN 训练的独立 Torch golden 与显式 VJP；运行时不调用此参考实现。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GDNShape:
    heads: int = 48
    query_heads: int = 16
    dimension: int = 128
    rows: int = 32

    def __post_init__(self):
        if (not 1 <= self.heads <= 64 or self.query_heads < 1 or self.heads % self.query_heads
                or self.dimension % 64 or self.rows != 32):
            raise ValueError("K32, complete GQA groups and a 64-aligned dimension required")

    @property
    def channels(self):
        return (2 * self.query_heads + self.heads) * self.dimension


def recurrence_forward(packed, decay, beta, history, width, shape=GDNShape()):
    """历史是常量；保存 FP32 临时状态供反向读取，不写回 AR cache。"""
    import torch
    s = shape
    if not 1 <= width <= s.rows:
        raise ValueError("active GDN width required")
    q, k, v = torch.split(packed[:width].float(),
        (s.query_heads*s.dimension, s.query_heads*s.dimension, s.heads*s.dimension), dim=-1)
    q, k = (x.reshape(width, s.query_heads, s.dimension) for x in (q, k))
    qi, ki = ((x.square().sum(-1, keepdim=True)+1e-6).rsqrt() for x in (q, k))
    qn = (q*qi*s.dimension**-.5).repeat_interleave(s.heads//s.query_heads, 1)
    kn = (k*ki).repeat_interleave(s.heads//s.query_heads, 1)
    v = v.reshape(width, s.heads, s.dimension)
    state, states, output = history.float(), [history.float()], []
    for t in range(width):
        decayed = state*decay[t].exp()[:, None, None]
        residual = v[t]-(decayed*kn[t, :, None]).sum(-1)
        state = decayed+(residual*beta[t, :, None])[:, :, None]*kn[t, :, None]
        states.append(state)
        output.append((state*qn[t, :, None]).sum(-1).to(packed.dtype))
    out = torch.stack(output)
    out = torch.cat((out, out.new_zeros(s.rows-width, s.heads, s.dimension)))
    saved = torch.stack(states+[state]*(s.rows-width))
    return out, saved, (q, k, qi, ki, qn, kn, v)


def recurrence_backward(packed, decay, beta, saved, upstream, width, shape=GDNShape()):
    """解析反向，不调用 autograd；不返回历史梯度。"""
    import torch
    s = shape
    _, _, tape = recurrence_forward(packed, decay, beta, saved[0], width, shape)
    q, k, qi, ki, qn, kn, v = tape
    dq, dk, dv = (torch.zeros_like(qn) for _ in range(3))
    dg, db = (torch.zeros_like(decay.float()) for _ in range(2))
    carry = torch.zeros_like(saved[0])
    for t in reversed(range(width)):
        do = upstream[t].float()
        ds = carry+do[:, :, None]*qn[t, :, None]
        dq[t] = (saved[t+1]*do[:, :, None]).sum(1)
        old = saved[t]*decay[t].exp()[:, None, None]
        residual = v[t]-(old*kn[t, :, None]).sum(-1)
        delta = residual*beta[t, :, None]
        dd = (ds*kn[t, :, None]).sum(-1)
        dr = dd*beta[t, :, None]
        dv[t] = dr
        db[t] = (dd*residual).sum(-1)
        dk[t] = (ds*delta[:, :, None]-old*dr[:, :, None]).sum(1)
        dold = ds-dr[:, :, None]*kn[t, :, None]
        dg[t] = (dold*old).sum((1, 2))
        carry = dold*decay[t].exp()[:, None, None]
    group = s.heads//s.query_heads
    dq, dk = (x.reshape(width, s.query_heads, group, s.dimension).sum(2) for x in (dq, dk))
    dq = dq*s.dimension**-.5
    dq = dq*qi-q*qi.pow(3)*(dq*q).sum(-1, keepdim=True)
    dk = dk*ki-k*ki.pow(3)*(dk*k).sum(-1, keepdim=True)
    active = torch.cat((dq.reshape(width, -1), dk.reshape(width, -1), dv.reshape(width, -1)), -1)
    dx = torch.cat((active.to(packed.dtype), packed.new_zeros(s.rows-width, s.channels)))
    return dx, dg, db


def convolution_forward(projected, weight, history, width):
    """四 tap 因果卷积→BF16 舍入→SiLU；历史四行中的最早一行不再使用。"""
    import torch
    x, w, h = projected.float(), weight.float(), history.float()
    rows, raw = [], []
    source = torch.cat((h[1:], x[:width]), 0)
    for t in range(width):
        z = ((source[t]*w[0]+source[t+1]*w[1])
             +(source[t+2]*w[2]+source[t+3]*w[3])).to(projected.dtype)
        raw.append(z)
        rows.append(torch.nn.functional.silu(z.float()).to(projected.dtype))
    padding = projected.new_zeros(projected.shape[0]-width, projected.shape[1])
    return torch.cat((torch.stack(rows), padding)), torch.cat((torch.stack(raw), padding))


def convolution_backward(projected, weight, history, preactivation, upstream, width):
    import torch
    x, w, h = projected.float(), weight.float(), history.float()
    z = preactivation[:width].float()
    sig = z.sigmoid()
    dz = (upstream[:width].float()*sig*(1+z*(1-sig))).to(projected.dtype).float()
    dx, dw = torch.zeros_like(x), torch.zeros_like(w)
    source = torch.cat((h[1:], x[:width]), 0)
    for t in range(width):
        for tap in range(4):
            dw[tap] += dz[t]*source[t+tap]
            position = t+tap-3
            if position >= 0:
                dx[position] += dz[t]*w[tap]
    return dx.to(projected.dtype), dw.to(weight.dtype).float()


def gates_forward(a, b, a_log, dt_bias, heads):
    import torch
    z = a[:, :heads].float()+dt_bias[:, :heads].float()
    g = -torch.exp(a_log[:, :heads].float())*torch.nn.functional.softplus(z)
    beta = b[:, :heads].float().sigmoid().to(b.dtype).float()
    return g, beta


def gates_backward(a, b, a_log, dt_bias, dg, db, heads):
    import torch
    z = a[:, :heads].float()+dt_bias[:, :heads].float()
    factor = -a_log[:, :heads].float().exp()*dg.float()
    da = factor*z.sigmoid()
    sig = b[:, :heads].float().sigmoid()
    delta = db.to(b.dtype).float()*sig*(1-sig)
    pad = lambda x, n: torch.nn.functional.pad(x, (0, n-heads))
    return (pad(da, a.shape[1]).to(a.dtype), pad(delta, b.shape[1]).to(b.dtype),
            pad((factor*torch.nn.functional.softplus(z)).sum(0, keepdim=True), a_log.shape[1]),
            pad(da.sum(0, keepdim=True), dt_bias.shape[1]).to(dt_bias.dtype).float())


def tail_forward(core, z, weight):
    """保留原生 gated RMSNorm 的两次 BF16 舍入，GDN norm 不加 1。"""
    import torch
    x, zr = core.reshape(-1, core.shape[-1]), z.reshape(-1, core.shape[-1])
    inverse = (x.float().square().mean(-1, keepdim=True)+1e-6).rsqrt()
    normed = (x.float()*inverse).to(core.dtype)
    weighted = (normed.float()*weight.float()).to(core.dtype)
    output = (weighted.float()*torch.nn.functional.silu(zr.float())).to(core.dtype)
    return output.reshape(z.shape), normed, weighted, inverse.expand(-1, 8)


def tail_backward(core, z, weight, normed, weighted, inverse, upstream):
    x, zr = core.reshape(-1, core.shape[-1]), z.reshape(-1, core.shape[-1])
    dy = upstream.reshape_as(x).float()
    sig = zr.float().sigmoid()
    dweight_value = (dy*(zr.float()*sig)).to(core.dtype)
    delta = (dweight_value.float()*weight.float()).to(core.dtype).float()
    r = inverse[:, :1]
    dx = (delta*r-x.float()*r.pow(3)*(x.float()*delta).mean(-1, keepdim=True)).to(core.dtype)
    dz = (dy*weighted.float()*sig*(1+zr.float()*(1-sig))).to(z.dtype)
    dw = (normed.float()*dweight_value.float()).sum(0, keepdim=True).to(weight.dtype).float()
    return dx.reshape_as(core), dz.reshape_as(z), dw


def recurrence_partials(packed, decay, beta, saved, upstream, width, shape, tile=16):
    """按 value 分片返回局部梯度；用于逐阶段核对设备归约。"""
    import torch
    s = shape
    _,_,tape = recurrence_forward(packed,decay,beta,saved[0],width,s)
    _,_,_,_,q,k,v = tape
    parts = s.dimension//tile
    dq = torch.zeros((s.rows,s.heads,parts,s.dimension),dtype=torch.float32)
    dk = torch.zeros_like(dq)
    dv = torch.zeros((s.rows,s.heads,s.dimension),dtype=packed.dtype)
    dg = torch.zeros((s.rows,s.heads,parts,8),dtype=torch.float32)
    db = torch.zeros_like(dg)
    carry = torch.zeros_like(saved[0])
    for t in reversed(range(width)):
        do = upstream[t].float()
        ds = carry+do[:,:,None]*q[t,:,None]
        dq[t] = (saved[t+1]*do[:,:,None]).reshape(s.heads,parts,tile,s.dimension).sum(2)
        old = saved[t]*decay[t].exp()[:,None,None]
        residual = v[t]-(old*k[t,:,None]).sum(-1)
        delta = residual*beta[t,:,None]
        dd = (ds*k[t,:,None]).sum(-1)
        dr = dd*beta[t,:,None]
        dk[t] = (ds*delta[:,:,None]-old*dr[:,:,None]).reshape(s.heads,parts,tile,s.dimension).sum(2)
        dv[t] = dr.to(packed.dtype)
        dold = ds-dr[:,:,None]*k[t,:,None]
        dg[t] = (dold*old).reshape(s.heads,parts,tile,8,s.dimension//8).sum((2,4))
        db[t] = (dd*residual).reshape(s.heads,parts,tile//8,8).sum(-2)
        carry = dold*decay[t].exp()[:,None,None]
    return (dq.reshape(-1,s.dimension),dk.reshape(-1,s.dimension),dv.reshape(s.rows,-1),
            dg.reshape(-1,8),db.reshape(-1,8))


def reduce_partials(packed,dq,dk,dv,dg,db,width,shape,tile=16):
    import torch
    s = shape
    q,k,_ = torch.split(packed[:width].float(),(s.query_heads*s.dimension, s.query_heads*s.dimension,
                                             s.heads*s.dimension),-1)
    q,k = q.reshape(width,s.query_heads,s.dimension),k.reshape(width,s.query_heads,s.dimension)
    qi,ki = ((x.square().sum(-1,keepdim=True)+1e-6).rsqrt() for x in (q,k))
    group,parts = s.heads//s.query_heads,s.dimension//tile
    aq,ak = (x.reshape(s.rows,s.query_heads,group,parts,s.dimension)[:width].sum((2,3)) for x in (dq,dk))
    aq = aq*s.dimension**-.5
    aq = aq*qi-q*qi.pow(3)*(q*aq).sum(-1,keepdim=True)
    ak = ak*ki-k*ki.pow(3)*(k*ak).sum(-1,keepdim=True)
    data = torch.cat((aq.reshape(width,-1),ak.reshape(width,-1),dv[:width].float()),-1).to(packed.dtype)
    data = torch.cat((data,packed.new_zeros(s.rows-width,s.channels)))
    grads = [torch.nn.functional.pad(x.reshape(s.rows,s.heads,parts,8).sum((2,3)),(0,64-s.heads)) for x in (dg,db)]
    return data,*grads
