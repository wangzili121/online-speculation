"""GDN 递推的独立 Torch 参考；状态布局为 [head, value, key]。"""

import torch


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
