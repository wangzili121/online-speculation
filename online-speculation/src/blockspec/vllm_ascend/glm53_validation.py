"""Explicit worker-side diagnostics, separate from the serving extension."""

from .glm53_proposer import GLM53WorkerExtension


def probe_head_replacement(worker):
    """验证原生投影读取加载后的 head 参数，并在结束前恢复全部参数。"""
    import torch
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context

    proposer = worker.model_runner.drafter
    model = proposer.model
    target = model.target.model.layers
    kda = next(i for i, layer in enumerate(target) if layer.layer_kind == "kda")
    mla = next(i for i, layer in enumerate(target) if layer.layer_kind == "mla")
    pairs = [("kda_qkv", model.layers[kda].in_proj_qkvbfg_a, target[kda].self_attn.in_proj_qkvbfg_a),
             ("kda_o", model.layers[kda].o_proj, target[kda].self_attn.o_proj),
             ("mla_qkv", model.layers[mla].fused_qkv_a_proj, target[mla].self_attn.fused_qkv_a_proj),
             ("mla_o", model.layers[mla].o_proj, target[mla].self_attn.o_proj),
             ("indexer_q", model.layers[mla].index_query, target[mla].self_attn.indexer.wq_b)]
    results = []
    with torch.no_grad(), set_ascend_forward_context(None, model.vllm_config, num_tokens=2,
                                                    model_instance=model.target, skip_compiled=True):
        for name, projection, original in pairs:
            state = {key: value.detach().clone() for key, value in projection.state_dict().items()}
            target_weight = original.weight.detach().clone()
            width = (projection.input_size_per_partition if getattr(projection, "input_is_parallel", False)
                     else projection.input_size)
            generator = torch.Generator(device=model.embedding.weight.device).manual_seed(781)
            inputs = torch.randn(2, width, device=model.embedding.weight.device,
                                 dtype=model.embedding.weight.dtype, generator=generator)
            before = projection(inputs)[0].clone()
            target_before = original(inputs)[0].clone()
            if not torch.isfinite(before).all():
                raise AssertionError(f"nonfinite head projection: {name}")
            try:
                projection.load_state_dict(dict(state, weight=torch.zeros_like(state["weight"])))
                changed = not torch.allclose(before, projection(inputs)[0], rtol=1e-3, atol=1e-3)
                target_unchanged = (torch.equal(original.weight, target_weight)
                                    and torch.allclose(target_before, original(inputs)[0], rtol=1e-3, atol=1e-3))
            finally:
                projection.load_state_dict(state)
            restored = projection(inputs)[0]
            torch.testing.assert_close(restored, before, rtol=1e-3, atol=1e-3)
            if not changed or not target_unchanged:
                raise AssertionError(f"head replacement inactive or aliases target weights: {name}")
            results.append(dict(name=name, projection=type(projection).__name__, dtype=str(projection.weight.dtype),
                                changed=True, restored=True, target_unchanged=True,
                                restore_max_abs=float((restored.float() - before.float()).abs().max())))
    return dict(rank=proposer.tp.rank, projections=results)


class GLM53ValidationWorkerExtension(GLM53WorkerExtension):
    def orthrus_probe_head_replacement(self):
        return probe_head_replacement(self)
