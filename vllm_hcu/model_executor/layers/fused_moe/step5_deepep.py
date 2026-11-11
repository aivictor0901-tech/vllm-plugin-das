# SPDX-License-Identifier: Apache-2.0
"""Step5 routed experts: DeepEP HT prefill / LL decode, shared DeepGEMM weights.

TP token ownership is handled by Step5's FusedMoEBlock before routing. This adapter never
infers a phase from batch size and never caches a previous layer's routing.
"""
from functools import lru_cache
import math
import os

import torch
import triton
import vllm.model_executor.layers.fused_moe  # initialize upstream before HCU
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import TopKWeightAndReduceNoOP
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm_hcu.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import BatchedDeepGemmExperts
from vllm_hcu.model_executor.layers.fused_moe.experts.dpsk_v4_deep_gemm_moe import (
    DeepEPAutoDeepGemmExperts, DeepEPAutoW4A8Experts,
    DeepEPDeepGemmContiguousExperts, DeepEPDeepGemmMaskedExperts,
)
from vllm_hcu.model_executor.layers.fused_moe.step5_w4a8_kernels import activation_masked
from vllm_hcu.model_executor.layers.fused_moe.step5_w4a8_policy import uses_step5_deepep

logger = init_logger("vllm.hcu.step5_deepep")


def _gemm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor,
          a_scale: torch.Tensor, b_scale: torch.Tensor, layout: torch.Tensor,
          expected_m: int, int4: bool, masked: bool) -> None:
    if int4:
        import deepgemm
        fn = (deepgemm.m_grouped_w4a8_gemm_nt_masked_hipc if masked
              else deepgemm.m_grouped_w4a8_gemm_nt_contiguous_hipc)
        args = ((a, a_scale), (b, b_scale), out, layout)
        fn(*args, *([expected_m] if masked else []), config={"MODE": 1002})
    else:
        import deepgemm
        fn = (deepgemm.m_grouped_fp8_gemm_nt_masked if masked
              else deepgemm.m_grouped_fp8_gemm_nt_contiguous)
        args = ((a, a_scale), (b, b_scale), out, layout)
        fn(*args, *([expected_m] if masked else []))


def _gemm_fake(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor,
               a_scale: torch.Tensor, b_scale: torch.Tensor, layout: torch.Tensor,
               expected_m: int, int4: bool, masked: bool) -> None:
    pass


@lru_cache(maxsize=1)
def gemm_op():
    try:
        return torch.ops.vllm.step5_deepep_gemm.default
    except AttributeError:
        from vllm.utils.torch_utils import direct_register_custom_op
        direct_register_custom_op(op_name="step5_deepep_gemm", op_func=_gemm,
                                  mutates_args=["out"], fake_impl=_gemm_fake)
        return torch.ops.vllm.step5_deepep_gemm.default


class Step5FP8HTExperts(DeepEPDeepGemmContiguousExperts):
    _hcu_step5_owner = True

    @staticmethod
    def _supports_activation(activation):
        return activation == MoEActivation.SWIGLUSTEP

    @staticmethod
    def adjust_N_for_activation(n, activation):
        return n // 2

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
              activation, global_num_experts, expert_map, a1q_scale, a2_scale,
              workspace13, workspace2, expert_tokens_meta, apply_router_weight_on_input):
        if apply_router_weight_on_input or activation != MoEActivation.SWIGLUSTEP:
            raise ValueError("Step5 requires output-side weights and SwigluStep7")
        from vllm_hcu.model_executor.layers.fused_moe.deep_gemm_utils import (
            compute_aligned_M, deepgemm_moe_permute, deepgemm_unpermute_and_reduce,
        )
        if getattr(expert_tokens_meta, "_hcu_deepep_ht_local_counts", False):
            expert_tokens_meta._hcu_step5_compact_rows = True
        e, _, n, k, topk = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        rows = compute_aligned_M(topk_ids.shape[0], topk, e, self.ALIGNMENT, expert_tokens_meta)
        fused = getattr(expert_tokens_meta, "_hcu_step5_fused_scatter", None)
        if fused is None:
            gathered = _resize_cache(workspace13.view(hidden_states.dtype), (rows, k))
            aq, scales, indices, inverse, _ = deepgemm_moe_permute(
                aq=hidden_states, aq_scale=self._ensure_2d_scale(a1q_scale),
                topk_ids=topk_ids, local_num_experts=e, expert_map=expert_map,
                expert_tokens_meta=expert_tokens_meta, aq_out=gathered,
            )
        else:
            aq, scales, indices, inverse = fused
            rows = aq.shape[0]
        gateup = _resize_cache(workspace2, (rows, n))
        op = gemm_op()
        op(aq, self._deepgemm_w13, gateup, scales.contiguous(),
           self.w1_scale.contiguous(), indices, 0, False, False)
        # Preserve Step5's post-SiLU clamp and BF16 rounding; LightOp's
        # pre-SiLU clamp API is a different activation and must not replace it.
        q = _resize_cache(workspace13.view(torch.float8_e4m3fn), (rows, n // 2))
        if n == 3072:
            from vllm_hcu.model_executor.layers.fused_moe.ep_fused_moe import ep_swiglustep_quant_deepgemm
            q, qs = ep_swiglustep_quant_deepgemm(gateup, indices, q, limit=7.)
        else:
            counts = torch.full((1,), rows, dtype=torch.int32, device=q.device)
            qs = torch.empty((1, rows, 1), dtype=torch.float32, device=q.device)
            activation_masked[(1, 8)](gateup.view(1, rows, n), q.view(1, rows, n // 2),
                qs, counts, rows, n // 2, False, triton.next_power_of_2(n // 2), num_warps=4)
            qs = qs.view(rows, 1)
        down = _resize_cache(workspace2, (rows, k))
        op(q, self._deepgemm_w2, down, qs.contiguous(), self.w2_scale.contiguous(),
           indices, 0, False, False)
        deepgemm_unpermute_and_reduce(a=down, topk_ids=topk_ids, topk_weights=topk_weights,
            inv_perm=inverse, expert_map=expert_map, output=output)
        finish_scatter(expert_tokens_meta)


class Step5LLExperts(BatchedDeepGemmExperts):
    _hcu_step5_owner = True

    @staticmethod
    def _supports_activation(activation):
        return activation == MoEActivation.SWIGLUSTEP

    @staticmethod
    def adjust_N_for_activation(n, activation):
        return n // 2

    def moe_problem_size(self, a1, w1, w2, topk_ids):
        return (a1.shape[0], a1.shape[1], self._hcu_logical_n,
                self._hcu_logical_k, topk_ids.shape[1])

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
              activation, global_num_experts, expert_map, a1q_scale, a2_scale,
              workspace13, workspace2, expert_tokens_meta, apply_router_weight_on_input):
        if apply_router_weight_on_input or activation != MoEActivation.SWIGLUSTEP:
            raise ValueError("Step5 requires output-side weights and SwigluStep7")
        int4 = self.quant_config.weight_quant_dtype == "int4"
        dtype = torch.int8 if int4 else torch.float8_e4m3fn
        if hidden_states.dtype != dtype or a1q_scale is None or expert_tokens_meta is None:
            raise ValueError("Step5 LL requires quantized input, scales and device expert counts")
        e, capacity, k = hidden_states.shape
        n = self._hcu_logical_n
        if not capacity:
            return
        counts = expert_tokens_meta.expert_num_tokens
        expected = max(1, math.ceil(topk_ids.shape[0] * topk_ids.shape[1]
                                   * self.num_dispatchers / global_num_experts))
        gateup = _resize_cache(workspace13, (e, capacity, n))
        op = gemm_op()
        op(hidden_states, self._deepgemm_w13, gateup, a1q_scale.contiguous(),
           self.w1_scale.contiguous(), counts, expected, int4, True)
        q = _resize_cache(workspace2.view(dtype), (e, capacity, n // 2))
        scales = torch.empty((e, capacity, 1), device=q.device, dtype=torch.float32)
        activation_masked[(e, 8)](gateup, q, scales, counts, capacity, n // 2,
            int4, triton.next_power_of_2(n // 2), num_warps=4)
        op(q, self._deepgemm_w2, output, scales, self.w2_scale.contiguous(),
           counts, expected, int4, True)


class Step5FP8LLExperts(Step5LLExperts):
    # Reuse the library-specific FP8 packers and logical-shape decoder.
    _pack_channel_weights = DeepEPDeepGemmMaskedExperts._pack_channel_weights
    WEIGHT_LAYOUT = "masked"


def finish_scatter(meta):
    fused = getattr(meta, "_hcu_step5_fused_scatter", None)
    guard = getattr(meta, "_hcu_step5_fused_scatter_guard", None)
    if fused is not None and guard is not None:
        stream = torch.cuda.current_stream()
        for tensor in fused:
            tensor.record_stream(stream)
        event = torch.cuda.Event()
        event.record(stream)
        guard["last_use"] = event


@lru_cache(maxsize=2)
def auto_experts_class(int4):
    base = DeepEPAutoW4A8Experts if int4 else DeepEPAutoDeepGemmExperts

    class Step5AutoExperts(base):
        _hcu_step5_owner = True

        def __init__(self, moe_config, quant_config, max_num_tokens, num_dispatchers,
                     fixed_use_low_latency=None, global_ht_token_bound=False):
            self.moe_config = moe_config
            self.quant_config = quant_config
            self.max_num_tokens = max_num_tokens
            self.num_dispatchers = num_dispatchers
            self._fixed_use_low_latency = fixed_use_low_latency
            self._global_ht_token_bound = global_ht_token_bound
            self._use_low_latency_snapshot = bool(fixed_use_low_latency)
            if int4:
                from .step5_w4a8 import _make_experts_class
                ht_type = _make_experts_class("prefill", "deepgemm")
                ll_type = Step5LLExperts
            else:
                ht_type, ll_type = Step5FP8HTExperts, Step5FP8LLExperts
            if fixed_use_low_latency is not True:
                self.ht_experts = ht_type(moe_config, quant_config)
            if fixed_use_low_latency is not False:
                self.ll_experts = ll_type(moe_config, quant_config,
                    max_num_tokens=max_num_tokens, num_dispatchers=num_dispatchers)
            if fixed_use_low_latency is True:
                self.ht_experts = self.ll_experts
            elif fixed_use_low_latency is False:
                self.ll_experts = self.ht_experts

        @staticmethod
        def _supports_activation(activation):
            return activation == MoEActivation.SWIGLUSTEP

        @staticmethod
        def _supports_parallel_config(parallel_config):
            return True

        def process_weights_after_loading(self, layer):
            base.process_weights_after_loading(self, layer)
            if int4:
                from .step5_w4a8 import _generation
                layer._hcu_step5_w4a8_owner = _generation(layer)

    return Step5AutoExperts


def prepare_step5_deepep(method, layer, int4):
    """Build the normal modular kernel, sharing the HT/LL communication handle."""
    from vllm.model_executor.layers.fused_moe.all2all_utils import maybe_make_prepare_finalize
    from vllm_hcu.model_executor.layers.fused_moe.prepare_finalize.deepep_auto import DeepEPAutoPrepareAndFinalize
    if not uses_step5_deepep(method.moe):
        raise ValueError("Step5 DeepEP adapter requires a DeepEP backend")
    if method.moe.moe_parallel_config.enable_eplb:
        raise ValueError("Step5 packed weights require static expert placement")
    gemm_op()
    quant = method.get_fused_moe_quant_config(layer)
    prepare = maybe_make_prepare_finalize(moe=method.moe, quant_config=quant,
        routing_tables=layer._expert_routing_tables(), allow_new_interface=True,
        use_monolithic=False)
    backend = method.moe.moe_parallel_config.all2all_backend
    fixed = None
    local = not method.moe.moe_parallel_config.use_all2all_kernels
    if local:
        # EP1 and ordinary TP need no DeepEP collective. Keep the same
        # contiguous DeepGEMM experts and the framework's TP reduction.
        fixed = False
        ht = ll = prepare
    elif not isinstance(prepare, DeepEPAutoPrepareAndFinalize):
        if backend not in ("deepep_high_throughput", "deepep_low_latency"):
            raise RuntimeError("Step5 DeepEP auto did not construct its prepare/finalize")
        fixed = backend == "deepep_low_latency"
        prepare = DeepEPAutoPrepareAndFinalize(prepare, prepare, fixed_use_low_latency=fixed)
    else:
        fixed = prepare._fixed_use_low_latency
    if not local:
        ht, ll = prepare.ht_prepare_finalize, prepare.ll_prepare_finalize
    # INT8 dispatch is native. Channel FP8 remains FN, so use BF16 wire data
    # and the FN receiver quantizer instead of silently reinterpreting FNUZ.
    if fixed is not False:
        ll.use_int8_dispatch = bool(int4)
        ll.use_fp8_dispatch = False
        ll.use_ue8m0_dispatch = False
    capacity = ll.max_num_tokens_per_rank() if fixed is not False else 1
    experts = auto_experts_class(int4)(method.moe, quant, capacity,
        ht.num_dispatchers(), fixed_use_low_latency=fixed,
        global_ht_token_bound=(method.moe.moe_parallel_config.pcp_size == 1
            or bool(getattr(method.moe, "_hcu_global_ht_token_bound", False))))
    experts.process_weights_after_loading(layer)
    # A single HT scatter cache is reused across layers; last_use fences its
    # lifetime. DBO buffer pairs are deliberately outside this adapter.
    if not local and fixed is not True and int4 and os.environ.get("HCU_STEP5_DEEPEP_FUSED_SCATTER", "0") == "1":
        manager = None
        from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
        manager = get_ep_all2all_manager()
        if not hasattr(manager, "_hcu_step5_scatter_buffers"):
            manager._hcu_step5_scatter_buffers = {}
            manager._hcu_step5_scatter_guards = {}
        ht._hcu_step5_scatter_buffers = manager._hcu_step5_scatter_buffers
        ht._hcu_step5_scatter_guards = manager._hcu_step5_scatter_guards
        ht._hcu_step5_fused_scatter_enabled = True
    method.moe_quant_config = quant
    method.moe_kernel = mk.FusedMoEKernel(prepare, experts)
    layer._hcu_step5_deepep = dict(int4=int4, backend=backend,
        local_experts=layer.w13_weight.shape[0], fixed_use_low_latency=fixed,
        shared_w4_storage=bool(int4))
    logger.info_once("[step5-deepep] HT/LL SwigluStep7, local_experts=%d, INT4=%s, backend=%s",
        layer.w13_weight.shape[0], int4, backend)


def pad_token_rows(x, multiple):
    padding = (-x.shape[0]) % multiple
    return torch.nn.functional.pad(x, (0, 0, 0, padding)) if padding else x


def shared_expert_token_slice(shared, tp_group):
    """Sum TP weight shards and scatter token rows in one collective."""
    if tp_group.world_size == 1:
        return shared
    return tp_group.reduce_scatter(pad_token_rows(shared, tp_group.world_size), dim=0)
