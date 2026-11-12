# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Channel INT4 / dynamic-token INT8 compressed-tensors MoE on HCU.

The checkpoint remains pack-quantized. Only the loaded expert tensors are
converted to LightOp's signed, high-nibble-first INT4 layout. This follows
Alone-wl/sglang-das#1; vLLM owns routing, shared experts and TP/EP reduction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.logger import init_logger

logger = init_logger(__name__)


def convert_packed_weight(weight: torch.Tensor) -> torch.Tensor:
    """Convert [E,N,K/8] INT32 (q+8) into [E,N,K/2] LightOp INT4 bytes."""
    if weight.dtype != torch.int32 or weight.ndim != 3:
        raise ValueError("compressed-tensors W4A8 requires rank-3 INT32 weights")
    packed = weight.contiguous().view(torch.uint8)
    # Swap each pair of nibbles and change offset binary to two's complement.
    return (((packed << 4) | (packed >> 4)) ^ 0x88).view(torch.int8)


def _load_w4a8_tuning_config(
    num_experts: int,
    n1: int,
    n2: int,
    k: int,
    top_k: int,
) -> None:
    """Load the LightOp W4A8 tuning table selected by TRITON_JSON_DIR."""
    from lightop._lmslim_native.vllm_compat.w8a8_get_cache_json import (
        W8a8GetCacheJSON,
    )

    cache = W8a8GetCacheJSON()
    json_path = cache.get_moeint8json_name(
        num_experts,
        n1,
        n2,
        k,
        top_k,
        use_int4_w4a8=True,
    )
    moe_config = cache.get_moeint8_triton_cache(
        json_path, num_experts, n1, n2, k, top_k
    )
    cache.triton_moejson_dict = moe_config or {}
    if moe_config:
        logger.info_once(
            "Using tuned LightOp W4A8 MoE config: %s", json_path
        )
    else:
        logger.warning_once(
            "W4A8 tuning config not found at %s; using LightOp default "
            "MoE config.",
            json_path,
        )


class HCUCompressedTensorsW4A8MoEMethod(FusedMoEMethodBase):
    """Load compressed-tensors directly, without a SlimQuant config override."""

    def __init__(self, weight_quant, input_quant, moe):
        super().__init__(moe)
        if not (
            weight_quant.num_bits == 4
            and weight_quant.type == "int"
            and weight_quant.strategy == "channel"
            and weight_quant.symmetric
            and not weight_quant.dynamic
            and weight_quant.actorder is None
            and weight_quant.group_size in (None, -1)
            and input_quant is not None
            and input_quant.num_bits == 8
            and input_quant.type == "int"
            and input_quant.strategy == "token"
            and input_quant.symmetric
            and input_quant.dynamic is True
        ):
            raise ValueError(
                "HCU compressed-tensors W4A8 requires symmetric static channel "
                "INT4 weights without actorder and dynamic symmetric token INT8"
            )
        activation = getattr(moe.activation, "value", moe.activation)
        if activation not in ("silu", "swiglustep"):
            raise ValueError(f"Unsupported HCU W4A8 activation: {activation}")
        if moe.has_bias or getattr(moe, "apply_router_weight_on_input", False):
            raise ValueError("HCU compressed-tensors W4A8 requires bias-free MoE "
                             "with router weights applied to the output")
        parallel = moe.moe_parallel_config
        if parallel.use_ep and parallel.all2all_backend not in (
            "naive", "allgather_reducescatter", None,
        ):
            raise ValueError(
                "HCU compressed-tensors W4A8 currently supports standard "
                "TP/EP dispatch (naive or allgather_reducescatter)"
            )
        self.activation = activation
        self.swiglu_limit = getattr(moe, "swiglu_limit", None)
        logger.info_once(
            "Using HCU compressed-tensors W4A8 MoE (LightOp INT4 x dynamic "
            "token INT8, activation=%s)", activation
        )

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        inter = intermediate_size_per_partition
        # LightOp's GEMMs read K in unmasked blocks of 64.
        if hidden_size % 64 or inter % 64:
            raise ValueError("HCU W4A8 hidden and intermediate sizes must be "
                             "multiples of 64")
        for name, shape in (
            ("w13_weight_packed", (num_experts, 2 * inter, hidden_size // 8)),
            ("w2_weight_packed", (num_experts, hidden_size, inter // 8)),
        ):
            param = torch.nn.Parameter(
                torch.empty(shape, dtype=torch.int32), requires_grad=False
            )
            # No is_transposed: shard w1/w3 along N and w2 along packed K.
            set_weight_attrs(param, extra_weight_attrs)
            layer.register_parameter(name, param)
        for name, n in (("w13", 2 * inter), ("w2", hidden_size)):
            scale = torch.nn.Parameter(
                torch.ones(num_experts, n, 1, dtype=torch.float32),
                requires_grad=False,
            )
            set_weight_attrs(
                scale, {**extra_weight_attrs, "quant_method": "channel"}
            )
            layer.register_parameter(f"{name}_weight_scale", scale)
            shape = torch.nn.Parameter(
                torch.empty(num_experts, 2, dtype=torch.int64),
                requires_grad=False,
            )
            set_weight_attrs(shape, extra_weight_attrs)
            layer.register_parameter(f"{name}_weight_shape", shape)
            layer.register_parameter(f"{name}_input_scale", None)

    def process_weights_after_loading(self, layer):
        if getattr(layer, "_hcu_compressed_w4a8_converted", False):
            return
        for name in ("w13_weight_packed", "w2_weight_packed"):
            replace_parameter(layer, name, convert_packed_weight(getattr(layer, name)))
        for name in ("w13_weight_scale", "w2_weight_scale"):
            # LightOp expands each signed INT4 into the high half of INT8 (x16).
            replace_parameter(
                layer, name, getattr(layer, name).float().contiguous() / 16,
                prefer_copy=True,
            )
        layer._hcu_compressed_w4a8_converted = True
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

    def get_fused_moe_quant_config(self, layer):
        return FusedMoEQuantConfig.make(
            torch.int8,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            per_act_token_quant=True,
            per_out_ch_quant=True,
            weight_dtype="int4",
        )

    def apply(
        self, layer, x, topk_weights, topk_ids,
        shared_experts=None, shared_experts_input=None,
    ):
        # MoERunner executes and combines shared experts outside this method.
        del shared_experts, shared_experts_input
        if getattr(layer, "apply_router_weight_on_input", False):
            raise ValueError("HCU W4A8 requires output router weighting")
        from lightop._lmslim_native.layers.fused_moe import w4a8
        from lightop._lmslim_native.layers.gemm.int8_utils import per_token_quant_int8
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )
        from vllm.triton_utils import tl

        if not getattr(layer, "_hcu_compressed_w4a8_converted", False):
            raise RuntimeError("HCU W4A8 weights have not been processed")
        if x.ndim != 2 or topk_ids.ndim != 2 or (
            topk_weights.shape != topk_ids.shape or topk_ids.shape[0] != x.shape[0]
        ):
            raise ValueError("HCU W4A8 expects matching rank-2 hidden/top-k tensors")
        w1, w2 = layer.w13_weight_packed, layer.w2_weight_packed
        if x.shape[1] != w1.shape[2] * 2:
            raise ValueError("HCU W4A8 input does not match the logical hidden size")
        output = torch.empty_like(x)
        if x.shape[0] == 0:
            return output
        top_k = topk_ids.shape[1]
        n1, hidden = w1.shape[1], w2.shape[1]
        if not getattr(layer, "_hcu_w4a8_tuning_config_loaded", False):
            _load_w4a8_tuning_config(
                w1.shape[0], n1, hidden, n1 // 2, top_k
            )
            layer._hcu_w4a8_tuning_config_loaded = True
        expert_map = getattr(layer, "_expert_map", None)
        if expert_map is None:
            expert_map = getattr(layer, "expert_map", None)
        compute_type = tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16
        # Bounded scratch space also works for long prefills, with no process-
        # global workspace that could race with another worker or MoE layer.
        chunk_size = 256
        capacity = min(x.shape[0], chunk_size)
        scratch = torch.empty(
            capacity * top_k * max(n1, hidden), device=x.device, dtype=x.dtype
        )
        for begin in range(0, x.shape[0], chunk_size):
            end = min(begin + chunk_size, x.shape[0])
            count = end - begin
            weights = topk_weights[begin:end].float().contiguous()
            ids = topk_ids[begin:end].int().contiguous()
            c1, c2 = w4a8.get_w8a8moe_json(w1.shape[0], count, n1, hidden, n1 // 2)

            def align(config):
                return moe_align_block_size(
                    ids, config["BLOCK_SIZE_M"], layer.global_num_experts, expert_map
                )

            sorted_ids, experts, padded = align(c1)
            gate_up = scratch[:count * top_k * n1].view(count, top_k, n1)
            gate_up.zero_()  # Nonlocal experts must never read stale scratch.
            qx, xs = per_token_quant_int8(x[begin:end].contiguous())
            w4a8.invoke_fused_moe_kernel_w4a8(
                qx, w1, gate_up, xs, layer.w13_weight_scale, None,
                weights, sorted_ids, experts, padded, False, top_k, c1,
                compute_type=compute_type,
            )
            gate, up = gate_up.chunk(2, dim=-1)
            if self.activation == "silu" and self.swiglu_limit is not None:
                gate = gate.clamp(max=self.swiglu_limit)
                up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            # vLLM Step5 clamps AFTER SiLU; preserve SwigluStepAndMul semantics.
            activated_gate = F.silu(gate)
            if self.activation == "swiglustep":
                activated_gate = activated_gate.clamp(max=7.0)
                up = up.clamp(min=-7.0, max=7.0)
            activated = activated_gate * up
            qa, a_scale = per_token_quant_int8(
                activated.reshape(count * top_k, n1 // 2)
            )
            down = scratch[:count * top_k * hidden].view(count, top_k, hidden)
            down.zero_()
            if c2["BLOCK_SIZE_M"] != c1["BLOCK_SIZE_M"]:
                sorted_ids, experts, padded = align(c2)
            w4a8.invoke_fused_moe_kernel_w4a8(
                qa, w2, down, a_scale, layer.w2_weight_scale, None,
                weights, sorted_ids, experts, padded, True, 1, c2,
                compute_type=compute_type,
            )
            # Step5's router already applies routed_scaling_factor to weights.
            torch.sum(down, dim=1, out=output[begin:end])
        return output
