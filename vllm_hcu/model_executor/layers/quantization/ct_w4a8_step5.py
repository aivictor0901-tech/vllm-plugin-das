# SPDX-License-Identifier: Apache-2.0
"""Step5 compressed-tensors INT4 experts with packed GPU weight storage.

CT stores eight offset-binary nibbles per INT32, low K index first. BoltOps
channel-wise W4A8 consumes signed nibble pairs with even K in the HIGH nibble.
XOR 0x88 and a nibble swap re-encode bytes without expanding weights or scales.
"""

from functools import lru_cache
from inspect import signature

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


def ct_int32_to_packed_int4(packed: torch.Tensor) -> torch.Tensor:
    """Re-encode CT INT32 storage as signed nibble pairs, with identical bytesize."""
    if packed.dtype != torch.int32 or packed.ndim not in (2, 3):
        raise ValueError("Expected 2-D or 3-D compressed-tensors INT32 weights")
    signed_pairs = packed.contiguous().view(torch.uint8).bitwise_xor(0x88)
    return ((signed_pairs << 4) | (signed_pairs >> 4)).view(torch.int8)


@lru_cache(maxsize=1)
def _packed_w4a8_experts():
    try:
        from boltops.fused_moe import fused_experts, normalize_moe_activation
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Step5 packed INT4 requires BoltOps W4A8 with swiglustep support; "
            "there is no expanded INT8 fallback"
        ) from exc
    required = {
        "use_int4_w4a8", "per_channel_quant", "activation", "gemm1_limit",
        "expert_map", "global_num_experts", "w1_scale", "w2_scale",
        "routed_scaling_factor", "output_dtype", "a1_scale",
    }
    if not required.issubset(signature(fused_experts).parameters):
        raise RuntimeError("Installed BoltOps lacks the Step5 packed W4A8 interface")
    if normalize_moe_activation("swiglustep") != ("swiglustep", True):
        raise RuntimeError("Installed BoltOps lacks gated swiglustep support")
    return fused_experts


class Step5CompressedW4A8MoEMethod(FusedMoEMethodBase):
    """Keep expert Parameters packed throughout loading, post-load and inference."""

    supports_int8_allgatherv_input = False

    def __init__(self, quant_config, moe):
        super().__init__(moe)
        self.quant_config = quant_config
        activation = getattr(moe.activation, "value", moe.activation)
        if activation not in ("silu", "swiglustep"):
            raise ValueError("Step5 packed INT4 supports silu and swiglustep only")
        if getattr(moe, "has_bias", False):
            raise ValueError("Step5 packed INT4 does not support expert biases")
        parallel = getattr(moe, "moe_parallel_config", None)
        self._use_deepep = (
            getattr(parallel, "all2all_backend", None) == "deepep_low_latency"
            and getattr(moe, "moe_backend", "auto") in ("auto", "triton", "deep_gemm")
        )
        self._use_deepgemm = self._use_deepep and getattr(moe, "moe_backend", "auto") == "deep_gemm"
        if self._use_deepep:
            if not getattr(parallel, "use_deepep_ll_kernels", False):
                raise ValueError("DeepEP LL is not active in the resolved MoE parallel config")
            return
        if getattr(moe, "moe_backend", "auto") not in ("auto", "triton"):
            raise ValueError("Step5 packed INT4 requires --moe-backend auto or triton")
        parallel = getattr(moe, "moe_parallel_config", None)
        if getattr(parallel, "all2all_backend", None) not in (
            None, "naive", "allgather_reducescatter",
        ):
            raise ValueError("Step5 packed INT4 currently requires naive/allgather EP")

    def create_weights(
        self, layer, num_experts, hidden_size, intermediate_size_per_partition,
        params_dtype, **extra_weight_attrs,
    ):
        del params_dtype
        size = intermediate_size_per_partition
        if hidden_size % 8 or size % 8:
            raise ValueError("Step5 CT INT4 requires hidden and shard sizes divisible by 8")
        for name, shape in (
            ("w13_weight", (num_experts, 2 * size, hidden_size // 2)),
            ("w2_weight", (num_experts, hidden_size, size // 2)),
        ):
            param = torch.nn.Parameter(torch.empty(shape, dtype=torch.int8), requires_grad=False)
            set_weight_attrs(param, dict(extra_weight_attrs))
            original = param.weight_loader

            def loader(param, loaded, weight_name, shard_id, expert_id,
                       return_success=False, _original=original):
                if layer._map_global_expert_id_to_local_expert_id(expert_id) == -1:
                    return False if return_success else None
                if not weight_name.endswith("_packed"):
                    raise ValueError("Step5 INT4 weight loader expects a weight_packed tensor")
                loaded = ct_int32_to_packed_int4(loaded)
                layer._step5_ct_load_generation = getattr(layer, "_step5_ct_load_generation", 0) + 1
                return _original(
                    param, loaded, weight_name.removesuffix("_packed"),
                    shard_id=shard_id, expert_id=expert_id,
                    return_success=return_success,
                )

            param.weight_loader = loader
            # The checkpoint name must win named_parameters() deduplication.
            layer.register_parameter(name + "_packed", param)
            layer.register_parameter(name, param)

        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.CHANNEL.value
        for name, shape in (
            ("w13_weight_scale", (num_experts, 2 * size, 1)),
            ("w2_weight_scale", (num_experts, hidden_size, 1)),
        ):
            param = torch.nn.Parameter(torch.ones(shape, dtype=torch.float32), requires_grad=False)
            set_weight_attrs(param, scale_attrs)
            layer.register_parameter(name, param)

    def get_fused_moe_quant_config(self, layer):
        return FusedMoEQuantConfig.make(
            torch.int8, weight_dtype="int4",
            w1_scale=(layer.w13_weight_scale / 16 if self._use_deepgemm else layer.w13_weight_scale),
            w2_scale=(layer.w2_weight_scale / 16 if self._use_deepgemm else layer.w2_weight_scale),
            per_act_token_quant=True, per_out_ch_quant=True,
            block_shape=None,
        )

    def process_weights_after_loading(self, layer):
        if not self._use_deepgemm:
            _packed_w4a8_experts()
        for name in ("w13_weight_packed", "w2_weight_packed"):
            layer._parameters.pop(name, None)
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self._use_deepep:
            from vllm_hcu.model_executor.layers.quantization.ct_w4a8_step5_deepep import make_step5_w4a8_ll_kernel
            self.moe_kernel = make_step5_w4a8_ll_kernel(self.moe, self.moe_quant_config, layer)
            logger.info_once("Step5 CT W4A8: DeepEP LL + %s",
                             "masked ASM DeepGEMM" if self._use_deepgemm else "BoltOps")
            return
        logger.info_once(
            "Step5 CT W4A8: packed INT4 expert storage, BoltOps W4A8, "
            "original channel scales; no INT8 weight expansion"
        )

    def apply(
        self, layer, x, topk_weights, topk_ids,
        shared_experts=None, shared_experts_input=None,
    ):
        if self._use_deepep:
            if self.moe_kernel is None:
                raise RuntimeError("Step5 DeepEP weights/kernel have not been initialized")
            return self.moe_kernel.apply(
                hidden_states=x, w1=layer.w13_weight, w2=layer.w2_weight,
                topk_weights=topk_weights, topk_ids=topk_ids,
                activation=self.moe.activation,
                global_num_experts=layer.global_num_experts,
                expert_map=getattr(layer, "_expert_map", None),
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                shared_experts=shared_experts, shared_experts_input=shared_experts_input,
            )
        # Shared experts and DP/TP reductions remain owned by MoERunner.
        del shared_experts, shared_experts_input
        if getattr(layer, "apply_router_weight_on_input", False):
            raise ValueError("Step5 packed INT4 requires output-side routing weights")
        activation = getattr(self.moe.activation, "value", self.moe.activation)
        if x.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Step5 packed INT4 requires BF16/FP16 input activations")
        w1, w2 = layer.w13_weight, layer.w2_weight
        if (
            w1.dtype != torch.int8 or w2.dtype != torch.int8
            or w1.ndim != 3 or w2.ndim != 3 or w1.shape[0] != w2.shape[0]
            or x.ndim != 2 or x.shape[1] != w1.shape[2] * 2
            or w2.shape[1] != x.shape[1] or w1.shape[1] != w2.shape[2] * 4
        ):
            raise ValueError("Step5 W4A8 requires packed [E, N, K/2] expert weights")
        if (
            topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape
            or topk_ids.shape[0] != x.shape[0]
        ):
            raise ValueError("Step5 W4A8 requires matching token/top-k tensors")
        if x.shape[0] == 0:
            return torch.empty_like(x)
        expert_map = getattr(layer, "_expert_map", None)
        global_num_experts = getattr(layer, "global_num_experts", w1.shape[0])
        return _packed_w4a8_experts()(
            x.contiguous(), w1, w2,
            topk_weights.to(torch.float32), topk_ids.to(torch.int32),
            inplace=False, activation=activation, is_gated=True,
            use_int4_w4a8=True, per_channel_quant=True,
            w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
            expert_map=expert_map, global_num_experts=global_num_experts,
            output_dtype=x.dtype, routed_scaling_factor=1.0,
            gemm1_limit=7.0 if activation == "swiglustep" else None,
        )
