"""Replay the resident suffix above the final trainable GLM MLA layer."""

from .pypto_glm53_online_ep import ROUTES, ROUTE_STRIDE, make_frozen_moe_backward
from .pypto_glm53_online_kda import KDA_PROGRAMS, ResidentKDAReplay, prepare_kda_programs
from .pypto_glm53_online_mhc import make_mhc_sublayer_adjoint


PROGRAMS = ("orthrus_mhc_enter", "orthrus_mhc_leave", "orthrus_moe_router",
            "orthrus_moe_routed", "orthrus_moe_finish")
ROWS = 16
MIX_PAD = 8


def prepare_frozen_suffix(api, compiler):
    from models.glm5_3_flash.config import FLASH

    if getattr(api, "ORTHRUS_TAPE_STREAMS", 0) != api.HC_MULT + 1:
        raise ValueError("frozen suffix replay needs the mHC sublayer-output tape")
    layers = FLASH.num_hidden_layers
    if FLASH.dsa_layer_ids[-1] != layers - 2 or (layers - 1) not in FLASH.kda_layer_ids:
        raise ValueError("frozen suffix replay expects MLA followed by KDA in the last two GLM layers")
    mHC = make_mhc_sublayer_adjoint(api.RANKS, api.VERIFY_QUERY_WIDTH)
    moe = make_frozen_moe_backward(api.RANKS, FLASH.n_routed_experts // api.RANKS)
    programs = {name: compiler.compile(name, entry) for name, entry in zip(PROGRAMS, (*mHC, *moe), strict=True)}
    programs.update(prepare_kda_programs(api, compiler, ROWS))
    return programs


class ResidentFrozenSuffix:
    """Connect the final mHC and MoE adjoints without reading activations back to host."""

    def __init__(self, session):
        import torch

        self.session = session
        self.runner = session.runner
        self.api = self.runner._api
        missing = set((*PROGRAMS, *KDA_PROGRAMS)).difference(self.runner.auxiliary_programs)
        if missing or self.api.ORTHRUS_TAPE_STREAMS != self.api.HC_MULT + 1:
            raise RuntimeError("resident frozen suffix programs or full mHC tape are missing")
        width, streams = self.api.D, self.api.HC_MULT
        hc_rows, mix_rows = ROWS * streams, ROWS * MIX_PAD

        def fp32(*shape):
            return session.allocate(shape, torch.float32)

        def bf16(*shape):
            return session.allocate(shape, torch.bfloat16)

        self.residual = bf16(hc_rows, width)
        self.collapse = bf16(ROWS, width)
        self.pre = fp32(ROWS, MIX_PAD)
        self.post = fp32(ROWS, MIX_PAD)
        self.logits = fp32(mix_rows, MIX_PAD)
        self.dy = fp32(ROWS, width)
        self.direct = fp32(hc_rows, width)
        self.dpost = fp32(ROWS, MIX_PAD)
        self.dmix = fp32(hc_rows, MIX_PAD)
        self.sublayer_scratch = bf16(ROWS, width)
        self.mix_scratch = fp32(hc_rows, MIX_PAD)
        self.via_pre = fp32(hc_rows, width)
        self.dpre = fp32(ROWS, MIX_PAD)
        self.packed_grad = fp32(mix_rows, MIX_PAD)
        self.dlogits = fp32(mix_rows, MIX_PAD)
        self.via_mix = fp32(ROWS, streams * width)
        self.stream_gradient = fp32(hc_rows, width)

        self.selected = session.allocate((ROWS, ROUTES), torch.int32)
        self.coefficients = fp32(ROWS, ROUTES)
        self.sigmoid = fp32(ROWS, ROUTES)
        self.tokens = session.allocate((ROWS, width), torch.int8)
        self.scale = fp32(ROWS, 1)
        self.routed_partial = fp32(ROWS, width + ROUTE_STRIDE)
        self.routed_total = fp32(ROWS, width + ROUTE_STRIDE)
        self.shared_partial = fp32(ROWS, width)
        self.shared_total = fp32(ROWS, width)
        self.route_gradient = fp32(ROWS, ROUTES)
        self.norm_gradient = fp32(ROWS, width)
        self.collapse_gradient = fp32(ROWS, width)
        self.kda = ResidentKDAReplay(session, ROWS)

    def _view_rows(self, tensor, start, count):
        from pypto.runtime import StackedDeviceTensor

        if not 0 <= start < start + count <= tensor.shape[1]:
            raise ValueError("GLM suffix view is outside the resident tensor")
        if count == 1:
            shape = tuple(tensor.shape[2:])
        else:
            shape = (count, *tensor.shape[2:])
        runtime = self.session.runtime
        shards = [runtime.view_tensor(shard, shape, byte_offset=start * (shard.nbytes // tensor.shape[1]), worker_id=rank)
                  for rank, shard in zip(tensor.worker_ids, tensor.shards, strict=True)]
        return StackedDeviceTensor(shards, (len(shards), *shape), tensor.worker_ids)

    def _weight(self, name, ordinal):
        return self._view_rows(self.runner.stacked[name], ordinal, 1)

    def _launch(self, name, *args):
        self.runner._run_l3(self.runner.auxiliary_programs[name], *args, config=self.runner.config)

    def enter_mhc(self, gradient, layer, *, ffn):
        from models.glm5_3_flash.config import FLASH

        index = 2 * (layer - (FLASH.num_hidden_layers - 2)) + int(ffn)
        block_rows = self.api.VERIFY_QUERY_WIDTH
        tape = self._view_rows(self.session.suffix_tape, index * block_rows, block_rows)
        prefix = "mhc_ffn" if ffn else "mhc_attn"
        params = tuple(self._weight(f"{prefix}_{name}", layer) for name in ("fn", "scale", "base"))
        self._launch(PROGRAMS[0], tape, *params, gradient, self.residual, self.collapse, self.pre, self.post,
                     self.logits, self.dy, self.direct, self.dpost, self.dmix, self.sublayer_scratch, self.mix_scratch)
        return params

    def leave_mhc(self, params, dcollapse):
        self._launch(PROGRAMS[1], self.residual, *params[:2], self.pre, self.post, self.logits,
                     self.direct, self.dpost, self.dmix, dcollapse, self.stream_gradient,
                     self.via_pre, self.dpre, self.packed_grad, self.dlogits, self.via_mix)

    def moe_backward(self, layer, epoch):
        from models.glm5_3_flash.config import FLASH

        ordinal = layer - FLASH.first_k_dense_replace
        norm = self._weight("norm_post_norm", layer)
        router = self._weight("router_gate_w", ordinal)
        correction = self._weight("router_gate_bias", ordinal)
        names = ("w_gate_up", "w_gate_up_scale", "w_down", "w_down_scale")
        routed = tuple(self._weight(f"moe_{name}", ordinal) for name in names)
        shared = tuple(self._weight(f"moe_shared_{name}", ordinal) for name in names)
        active = self.api.VERIFY_QUERY_WIDTH
        self._launch(PROGRAMS[2], self.collapse, norm, router, correction, self.selected, self.coefficients,
                     self.sigmoid, self.tokens, self.scale, active)
        self._launch(PROGRAMS[3], self.tokens, self.scale, self.dy, self.selected, self.coefficients,
                     *routed, self.routed_partial, self.routed_total, active, epoch)
        self._launch(PROGRAMS[4], self.collapse, self.dy, norm, router, *shared, self.selected, self.sigmoid,
                     self.tokens, self.scale, self.routed_total, self.shared_partial, self.shared_total,
                     self.route_gradient, self.norm_gradient, self.collapse_gradient, active, epoch)

    def kda_backward(self, layer, history_row, epoch):
        from models.glm5_3_flash.config import FLASH

        ordinal = FLASH.kda_layer_ids.index(layer)
        conv_state = self._weight("kda_conv_state", ordinal)
        recurrent_state = self._weight("kda_recurrent_state", ordinal)
        initial_conv = self._view_rows(conv_state, history_row, 1)
        initial_recurrence = self._view_rows(recurrent_state, history_row, 1)
        weight_keys = ("conv_q", "conv_k", "conv_v", "w_q", "w_k", "w_v", "w_f_a", "w_f_b",
                       "dt_bias", "a_log", "w_b", "w_g_a", "w_g_b", "w_o", "o_norm")
        weights = tuple(self._weight(f"kda_{key}", ordinal) for key in weight_keys)
        input_norm = self._weight("norm_input_norm", layer)
        return self.kda.step(self._launch, self.collapse, self.dy, input_norm, weights,
                             initial_conv, initial_recurrence, epoch)

    def step(self, gradient, history_row):
        """Return the gradient at the output of the final trainable MLA sublayer."""
        from models.glm5_3_flash.config import FLASH

        layer = FLASH.num_hidden_layers - 1
        params = self.enter_mhc(gradient, layer, ffn=True)
        step = getattr(self.runner, "_orthrus_frozen_suffix_epoch", 0) + 1
        self.moe_backward(layer, 2 * step - 1)
        self.leave_mhc(params, self.collapse_gradient)
        params = self.enter_mhc(self.stream_gradient, layer, ffn=False)
        self.leave_mhc(params, self.kda_backward(layer, history_row, step))
        params = self.enter_mhc(self.stream_gradient, layer - 1, ffn=True)
        self.moe_backward(layer - 1, 2 * step)
        self.leave_mhc(params, self.collapse_gradient)
        self.enter_mhc(self.stream_gradient, layer - 1, ffn=False)
        self.runner._orthrus_frozen_suffix_epoch = step
        return self.dy
