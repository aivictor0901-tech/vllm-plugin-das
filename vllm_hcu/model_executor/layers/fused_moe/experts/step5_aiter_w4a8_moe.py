# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Step5 SlimQuant W4A8 experts for DeepEP high-throughput dispatch.

DeepEP only moves tokens.  After the HCU HT receiver the experts see the
standard vLLM contract: global expert ids (ids that are not local map to an
in-range non-local expert and are dropped by ``expert_map``), per-token INT8
activations with an ``[M, 1]`` FP32 scale, and an already weighted output that
the finalize step sums across ranks.  The compute is the same AITER MOE_C
W4A8 path the all-gather route uses, so no DeepGEMM kernel is involved.
"""

from __future__ import annotations

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
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import current_platform

logger = init_logger(__name__)


class Step5AiterW4A8Experts(mk.FusedMoEExpertsModular):
    """Run the AITER W4A8 MoE on tokens received through DeepEP HT."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        method: object,
    ) -> None:
        super().__init__(moe_config, quant_config)
        self._method = method
        self._layer: torch.nn.Module | None = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # The quantization method already installed the AITER weight/scale
        # layout on the layer; this class only needs a handle to it.
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
        # AITER may re-lay out w1/w2, so take the logical sizes from the config.
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
        # AITER owns its scratch; only the output buffer is requested.
        return ((0,), (0,), (M, K))

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # AITER already applied the routing weights and reduced over top-k.
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
        if apply_router_weight_on_input:
            raise RuntimeError("Step5 AITER W4A8 applies routing weights in GEMM2")
        if self._layer is None:
            raise RuntimeError(
                "Step5 AITER W4A8 experts were used before weight postprocessing"
            )
        if hidden_states.size(0) == 0:
            return

        from vllm_hcu.model_executor.layers.quantization import (
            compressed_tensors_moe_runtime as moe_runtime,
        )

        if getattr(self._method, "_boltops", False):
            # BoltOps takes BF16 activations.  DeepEP delivers INT8 + per-token
            # scale, so undo the per-token quantisation (re-quantising inside
            # BoltOps reproduces the same integers).
            x = hidden_states
            if a1q_scale is not None:
                if hidden_states.dtype is not torch.int8:
                    raise RuntimeError("Step5 BoltOps W4A8 scale requires INT8 activations")
                x = (
                    hidden_states.float() * a1q_scale.reshape(-1, 1).to(torch.float32)
                ).to(output.dtype)
            output.copy_(
                moe_runtime.apply_step5_boltops_w4a8_moe(
                    self._method, self._layer, x, topk_weights, topk_ids
                )
            )
            return

        extra: dict[str, object] = {}
        if a1q_scale is not None:
            if hidden_states.dtype is not torch.int8:
                raise RuntimeError("Step5 AITER W4A8 scale requires INT8 activations")
            extra = {
                "activation_scale": a1q_scale.reshape(-1, 1)
                .to(torch.float32)
                .contiguous(),
                "output_dtype": output.dtype,
            }
        result = moe_runtime.apply_aiter_w4a8_moe(
            self._method,
            self._layer,
            hidden_states,
            topk_weights,
            topk_ids,
            **extra,
        )
        output.copy_(result)


def make_step5_aiter_w4a8_ht_kernel(
    method: object,
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    """DeepEP high-throughput prepare/finalize around the AITER W4A8 experts."""

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
    experts = Step5AiterW4A8Experts(moe_config, moe_quant_config, method)
    logger.info_once("Using Step5 DeepEP HT dispatch with AITER W4A8 experts.")
    return mk.FusedMoEKernel(prepare_finalize, experts)


__all__ = ["Step5AiterW4A8Experts", "make_step5_aiter_w4a8_ht_kernel"]
