# SPDX-License-Identifier: Apache-2.0
"""Step5 packed INT4 experts on the lightop Marlin W4A8 grouped GEMM.

Two routes share one compute core:

* naive / all-gather dispatch: tokens arrive gathered, ``apply_step5_deepgemm``
  quantizes them and runs the experts locally;
* DeepEP high-throughput dispatch: DeepEP only moves tokens, so the experts
  (``Step5MarlinHTExperts``) receive per-token INT8 activations plus an
  ``[M, 1]`` FP32 scale and global expert ids.

The HIPC DeepGEMM contiguous kernel (m_grouped_w4a8_gemm_nt_contiguous_hipc)
fails to launch on gfx936 ("W4A8 WASP kernel launch failed"), so the GEMMs use
``lightop.moe_groupgemm_marlin_w4a8`` instead.  Its contract, verified against
a torch reference on gfx936:

* weights are re-laid out once with ``lightop.awq_marlin_repack_w4a8``
  ([E,N,K/2] high-nibble-first signed INT4 bytes -> int32 [E,K/32,4N]);
* ``b_scale`` must be divided by 16 and be rank 3 ([E,N,1]);
* ``a_scale`` is [M,1]; there is no row padding and no ``m_indices``: rows are
  grouped by expert and described by ``problem_size`` / ``experts_offsets``.

Class and function names keep the ``DeepGemm`` spelling so the Step5 patch
module that imports them does not change.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import current_platform

from vllm_hcu.model_executor.layers.quantization import (
    slimquant_w4a8_deepgemm_runtime as runtime,
)

logger = init_logger(__name__)

_DEFAULT_SWIGLUSTEP_LIMIT = 7.0
_MARLIN_CONFIG = {"MODE": 1000}
# Shapes the Marlin kernel was verified with (128x512x128 tiles).
_MARLIN_N_ALIGN = 512
_MARLIN_K_ALIGN = 128


def _marlin_repack(weight: torch.Tensor) -> torch.Tensor:
    """[E,N,K/2] signed INT4 bytes -> same-size int8 tensor holding Marlin data.

    The Marlin layout is int32 [E,K/32,4N]; it is stored viewed as int8
    [E,N,K/2] so the layer parameter keeps its checkpoint shape and dtype.
    """
    from lightop import awq_marlin_repack_w4a8

    experts, n, k_half = weight.shape
    k = k_half * 2
    packed = torch.empty(
        (experts, k // 32, 4 * n), dtype=torch.int32, device=weight.device
    )
    awq_marlin_repack_w4a8(weight.contiguous(), packed, experts, k, n)
    return packed.view(torch.int8).view(experts, n, k_half)


def _as_marlin(weight: torch.Tensor) -> torch.Tensor:
    experts, n, k_half = weight.shape
    return weight.view(torch.int32).view(experts, (k_half * 2) // 32, 4 * n)


class Step5W4A8DeepGemmExperts(runtime.DeepEPDeepGemmW4A8ContiguousExperts):
    """Marlin W4A8 grouped experts with silu and swiglustep activations."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ) -> None:
        super().__init__(moe_config, quant_config)
        activation = getattr(moe_config.activation, "value", moe_config.activation)
        if activation not in ("silu", "swiglustep"):
            raise ValueError("Step5 W4A8 DeepGEMM supports silu and swiglustep only")
        self._activation_name = activation
        self._marlin_b13: torch.Tensor | None = None
        self._marlin_b2: torch.Tensor | None = None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.SWIGLUSTEP)

    @staticmethod
    def adjust_N_for_activation(N: int, activation: MoEActivation) -> int:
        return N // 2

    def apply(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Step5 W4A8 uses the lightop Marlin path; call apply_step5_deepgemm"
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Repack in place of the checkpoint Parameters; no second copy is kept."""
        runtime._validate_w4a8_channel_weights(layer)
        marker = getattr(layer, "_hcu_step5_hipc_packed_ids", None)
        if marker is not None:
            if marker != (id(layer.w13_weight), id(layer.w2_weight)):
                raise RuntimeError(
                    "Step5 W4A8 weights changed after Marlin packing; "
                    "reload is unsupported"
                )
            self._deepgemm_w13 = layer.w13_weight
            self._deepgemm_w2 = layer.w2_weight
            return
        w13, w2 = layer.w13_weight, layer.w2_weight
        n1, k1 = w13.size(1), w13.size(2) * 2
        n2, k2 = w2.size(1), w2.size(2) * 2
        if (
            n1 % _MARLIN_N_ALIGN
            or n2 % _MARLIN_N_ALIGN
            or k1 % _MARLIN_K_ALIGN
            or k2 % _MARLIN_K_ALIGN
        ):
            raise ValueError(
                "Step5 W4A8 Marlin needs N % "
                f"{_MARLIN_N_ALIGN} == 0 and K % {_MARLIN_K_ALIGN} == 0, "
                f"got w13 N={n1} K={k1}, w2 N={n2} K={k2}"
            )
        with torch.no_grad():
            packed_w13 = _marlin_repack(w13.detach())
            packed_w2 = _marlin_repack(w2.detach())
            self._marlin_b13 = (
                (layer.w13_weight_scale.detach().float() / 16.0)
                .reshape(w13.size(0), n1, 1)
                .contiguous()
            )
            self._marlin_b2 = (
                (layer.w2_weight_scale.detach().float() / 16.0)
                .reshape(w2.size(0), n2, 1)
                .contiguous()
            )
        layer.w13_weight = torch.nn.Parameter(packed_w13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(packed_w2, requires_grad=False)
        self._deepgemm_w13 = layer.w13_weight
        self._deepgemm_w2 = layer.w2_weight
        layer._hcu_step5_hipc_packed_ids = (id(layer.w13_weight), id(layer.w2_weight))

    def _quantize_activation(
        self, gateup_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._activation_name == "swiglustep":
            limit = float(self.quant_config.gemm1_clamp_limit or _DEFAULT_SWIGLUSTEP_LIMIT)
            return runtime.fuse_silu_mul_clamp_quant(gateup_output, limit=limit)
        return runtime.fuse_silu_mul_quant(gateup_output)


def marlin_moe_from_int8(
    experts: Step5W4A8DeepGemmExperts,
    layer: torch.nn.Module,
    xq: torch.Tensor,
    xs: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Marlin W4A8 MoE on per-token INT8 activations; returns the weighted sum."""
    if experts._marlin_b13 is None or experts._marlin_b2 is None:
        raise RuntimeError("Step5 W4A8 Marlin weights were not packed before apply")
    from lightop import moe_groupgemm_marlin_w4a8

    device = xq.device
    num_tokens, hidden = xq.shape
    topk = topk_ids.shape[1]
    local_experts = layer.w13_weight.size(0)
    n1 = layer.w13_weight.size(1)

    # Rows (token, slot) that run on a local expert, grouped by expert.
    flat_ids = topk_ids.reshape(-1).long()
    if expert_map is None:
        local_ids = flat_ids
    else:
        local_ids = expert_map[flat_ids.clamp_min(0)].long()
    valid = (flat_ids >= 0) & (local_ids >= 0) & (local_ids < local_experts)
    key = torch.where(valid, local_ids, torch.full_like(local_ids, local_experts))
    order = torch.argsort(key, stable=True)
    counts = torch.bincount(key, minlength=local_experts + 1)[:local_experts]
    num_rows = int(counts.sum())  # host sync: output buffers need the row count
    if num_rows == 0:
        return torch.zeros((num_tokens, hidden), dtype=out_dtype, device=device)
    problem_size = counts.to(torch.int32)
    expert_offsets = (torch.cumsum(counts, 0) - counts).to(torch.int32)
    selected = order[:num_rows]
    token_index = selected // topk

    rows_q = xq.index_select(0, token_index)
    rows_s = xs.reshape(num_tokens, 1).float().index_select(0, token_index).contiguous()

    gateup = torch.empty((num_rows, n1), dtype=out_dtype, device=device)
    moe_groupgemm_marlin_w4a8(
        rows_q,
        _as_marlin(layer.w13_weight),
        gateup,
        rows_s,
        experts._marlin_b13,
        problem_size,
        expert_offsets,
        _MARLIN_CONFIG,
    )
    q_act, q_act_scale = experts._quantize_activation(gateup)
    del gateup
    down = torch.empty((num_rows, hidden), dtype=out_dtype, device=device)
    moe_groupgemm_marlin_w4a8(
        q_act,
        _as_marlin(layer.w2_weight),
        down,
        q_act_scale.reshape(num_rows, 1).float().contiguous(),
        experts._marlin_b2,
        problem_size,
        expert_offsets,
        _MARLIN_CONFIG,
    )

    # Weighted reduce back to token order, one top-k slot at a time to keep
    # the fp32 temporaries at [num_tokens, hidden].
    inverse = torch.full((num_tokens * topk,), -1, dtype=torch.long, device=device)
    inverse[selected] = torch.arange(num_rows, device=device)
    inverse = inverse.view(num_tokens, topk)
    topk_weights = topk_weights.to(torch.float32)
    output = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=device)
    for slot in range(topk):
        row = inverse[:, slot]
        weight = topk_weights[:, slot] * (row >= 0)
        output += down.index_select(0, row.clamp_min(0)).float() * weight[:, None]
    return output.to(out_dtype)


def apply_step5_deepgemm(
    experts: Step5W4A8DeepGemmExperts,
    layer: torch.nn.Module,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    global_num_experts: int,
) -> torch.Tensor:
    del global_num_experts
    if x.dtype != torch.bfloat16:
        raise ValueError("Step5 W4A8 DeepGEMM requires BF16 activations")
    xq, xs = moe_kernel_quantize_input(x, None, torch.int8, per_act_token_quant=True)
    return marlin_moe_from_int8(
        experts, layer, xq, xs, topk_weights, topk_ids, expert_map, x.dtype
    )


class Step5MarlinHTExperts(mk.FusedMoEExpertsModular):
    """Marlin W4A8 experts behind DeepEP high-throughput dispatch.

    The experts get per-token INT8 activations with an ``[M, 1]`` FP32 scale
    and global expert ids (non-local ids are dropped through ``expert_map``),
    and must return the routing-weighted sum over top-k, which the finalize
    step then reduces across ranks.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        inner: Step5W4A8DeepGemmExperts,
    ) -> None:
        super().__init__(moe_config, quant_config)
        self._inner = inner
        self._layer: torch.nn.Module | None = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Weights were already repacked by Step5W4A8DeepGemmExperts.
        self._layer = layer

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_rocm()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.SWIGLUSTEP)

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return bool(getattr(moe_parallel_config, "use_ep", False))

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        local_experts = int(w1.shape[0])
        two_n = 2 * int(self.moe_config.intermediate_size_per_partition)
        return (
            local_experts,
            int(a1.shape[0]),
            two_n,
            int(a1.shape[1]),
            int(topk_ids.shape[1]),
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return ((0,), (0,), (M, K))

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        del w1, w2, activation, global_num_experts, a2_scale
        del workspace13, workspace2, expert_tokens_meta
        if apply_router_weight_on_input:
            raise RuntimeError("Step5 Marlin W4A8 applies routing weights in the reduce")
        if self._layer is None:
            raise RuntimeError(
                "Step5 Marlin W4A8 experts were used before weight postprocessing"
            )
        if hidden_states.size(0) == 0:
            return
        if a1q_scale is not None:
            if hidden_states.dtype is not torch.int8:
                raise RuntimeError("Step5 Marlin W4A8 scale requires INT8 activations")
            xq = hidden_states
            xs = a1q_scale.reshape(-1, 1).to(torch.float32)
        else:
            xq, xs = moe_kernel_quantize_input(
                hidden_states, None, torch.int8, per_act_token_quant=True
            )
        output.copy_(
            marlin_moe_from_int8(
                self._inner,
                self._layer,
                xq,
                xs,
                topk_weights,
                topk_ids,
                expert_map,
                output.dtype,
            )
        )


def make_step5_marlin_ht_kernel(
    inner: Step5W4A8DeepGemmExperts,
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    """DeepEP high-throughput prepare/finalize around the Marlin W4A8 experts."""

    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )

    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=False,
    )
    if prepare_finalize is None:
        raise RuntimeError("Step5 DeepEP HT prepare/finalize was not constructed")
    experts = Step5MarlinHTExperts(moe_config, moe_quant_config, inner)
    logger.info_once("Using Step5 DeepEP HT dispatch with Marlin W4A8 experts.")
    return mk.FusedMoEKernel(prepare_finalize, experts)
