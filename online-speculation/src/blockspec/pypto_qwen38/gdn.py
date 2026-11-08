"""GDN 递推的独立 Torch 参考；状态布局为 [head, value, key]。"""

import torch


@torch.inference_mode()
def convolution_golden(projected, weight, history):
    """宽度 4 的逐通道卷积及 SiLU，保留卷积后的原生舍入。"""
    if projected.ndim != 2 or weight.shape != (4, projected.shape[1]) or history.shape != weight.shape:
        raise ValueError("projected rows and matching four-tap weights/history required")
    state, w = history.float().clone(), weight.float()
    result = torch.empty_like(projected)
    for token, current in enumerate(projected.float()):
        value = (state[1] * w[0] + state[2] * w[1]) + (state[3] * w[2] + current * w[3])
        value = value.to(projected.dtype).float()
        result[token] = torch.nn.functional.silu(value).to(projected.dtype)
        state = torch.cat((state[1:], current[None]), dim=0)
    return result


@torch.inference_mode()
def recurrence_golden(query, key, value, log_decay, beta, initial_state, width):
    """输入为卷积后的 Q/K/V；返回有效输出和最终临时状态，不修改历史。"""
    rows, heads, dim = query.shape
    if (key.shape != query.shape or value.shape != query.shape
            or log_decay.shape != (rows, heads) or beta.shape != (rows, heads)
            or initial_state.shape != (heads, dim, dim)
            or type(width) is not int or not 1 <= width <= rows):
        raise ValueError("matching Q/K/V, gates, state and an active width are required")
    q, k, v = (x[:width].float() for x in (query, key, value))
    # 与当前模型定义一致：在平方和后加 epsilon；Q 另乘 1/sqrt(dim)。
    q = q * (q.square().sum(-1, keepdim=True) + 1e-6).rsqrt() * dim ** -.5
    k = k * (k.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    state = initial_state.float().clone()
    output = torch.zeros_like(query)
    for token in range(width):
        decayed = state * log_decay[token].float().exp()[:, None, None]
        remembered = (decayed * k[token, :, None, :]).sum(-1)
        delta = (v[token] - remembered) * beta[token].float()[:, None]
        state = decayed + delta[:, :, None] * k[token, :, None, :]
        output[token] = (state * q[token, :, None, :]).sum(-1).to(query.dtype)
    return output, state
