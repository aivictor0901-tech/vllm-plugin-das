# SPDX-License-Identifier: Apache-2.0
"""Step5 CT W4A8 experts on the standard DeepEP LL modular interface.

No TP/DP topology is encoded here. Routed expert dimensions and maps belong
to vLLM; LL combine and MoERunner own routed/shared reduction respectively.
"""
from __future__ import annotations

from dataclasses import replace
from functools import wraps
import torch

from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import TopKWeightAndReduceDelegate
from vllm.model_executor.layers.quantization.utils.quant_utils import kInt4W4A8StaticChannelSym, kInt8DynamicTokenSym
from vllm.platforms import current_platform


def is_step5_ll(moe) -> bool:
    if (getattr(moe, "moe_backend", None) not in ("auto", "triton", "deep_gemm")
            or not getattr(moe.moe_parallel_config, "use_deepep_ll_kernels", False)):
        return False
    config = get_current_vllm_config().model_config.hf_config
    return getattr(config, "model_type", None) in ("step5", "step5_vl")


def enable_bf16_batched_passthrough():
    """Unquantized tensors need no capture query or quantization.

    This also supports DTK Torch builds whose accelerator Stream lacks
    is_capturing(). Quantized paths retain the installed implementation.
    """
    from vllm.model_executor.layers.fused_moe.experts import fused_batched_moe as batched
    quantize = batched.batched_moe_kernel_quantize_input
    if getattr(quantize, "_step5_bf16_passthrough", False):
        return

    @wraps(quantize)
    def passthrough(A, A_scale, num_tokens, E, N, expert_num_tokens,
                    qtype, per_act_token_quant, block_shape=None):
        if qtype is None:
            return A, batched.normalize_batched_scales_shape(A_scale, E)
        return quantize(A, A_scale, num_tokens, E, N, expert_num_tokens,
                        qtype, per_act_token_quant, block_shape)

    passthrough._step5_bf16_passthrough = True
    batched.batched_moe_kernel_quantize_input = passthrough


def bounded_ll_config(moe):
    """Allocate for a decode-sized chunk; larger batches use collective rounds.

    This is a workspace bound, not a limit on input length/batch size. Derive
    it from the configured workload, equally on every rank and every layer.
    """
    config = get_current_vllm_config()
    speculative = getattr(config.speculative_config, "num_speculative_tokens", 0) or 0
    tokens = config.scheduler_config.max_num_seqs * (1 + speculative)
    captures = getattr(config.compilation_config, "cudagraph_capture_sizes", None) or []
    tokens = max(tokens, max(captures, default=1))
    capacity = min(moe.max_num_tokens, 1 << (max(1, tokens) - 1).bit_length())
    return replace(moe, max_num_tokens=capacity)


def collective_token_bound(moe, local_tokens: int) -> int:
    """Use host DP/SP metadata, never a GPU count sync during graph replay.

    LL has no dp_metadata. Its context carries the already synchronized CPU
    bound from the model runner, including graph padding and MTP batches.
    Unknown contexts retain a conservative bound for standalone profiling;
    local token counts alone must never determine cross-rank round counts.
    """
    try:
        ctx = get_forward_context()
    except AssertionError:
        return moe.max_num_tokens
    global_tokens = getattr(ctx, "_hcu_step5_ll_global_tokens", None)
    if global_tokens is not None:
        sp_size = getattr(moe, "sp_size", 1)
        bound = (global_tokens + sp_size - 1) // sp_size
        if local_tokens > bound:
            raise RuntimeError(
                f"Step5 LL synchronized bound underestimates local input: {local_tokens} > {bound}"
            )
        return bound
    metadata = ctx.dp_metadata
    if metadata is None:
        return moe.max_num_tokens
    sizes = getattr(metadata, "local_sizes", None)
    if sizes is None:
        sizes = metadata.num_tokens_across_dp_cpu.tolist()
    bound = max(sizes, default=0)
    descriptor = getattr(ctx, "batch_descriptor", None)
    if descriptor is not None:
        # Graph padding is coordinated across DP by the model runner.
        # The descriptor counts pre-SP tokens; metadata.local_sizes and
        # expert inputs count per-SP-rank tokens. Compare in the same units.
        sp_size = getattr(moe, "sp_size", 1)
        padded_local = (descriptor.num_tokens + sp_size - 1) // sp_size
        bound = max(bound, padded_local)
    if local_tokens > bound:
        raise RuntimeError(
            f"Step5 LL token metadata underestimates the local input: {local_tokens} > {bound}"
        )
    return bound


class ChunkedLLKernel:
    """Delegate the vLLM kernel interface while bounding LL temporary memory."""
    def __init__(self, kernel, moe):
        self.kernel = kernel
        self.original_moe = moe
        self.capacity = kernel.prepare_finalize.max_num_tokens_per_rank()

    def __getattr__(self, name):
        return getattr(self.kernel, name)

    def apply(self, hidden_states, w1, w2, topk_weights, topk_ids, activation,
              global_num_experts, expert_map, apply_router_weight_on_input,
              shared_experts=None, shared_experts_input=None):
        bound = collective_token_bound(self.original_moe, hidden_states.size(0))
        rounds = max(1, (bound + self.capacity - 1) // self.capacity)
        if rounds == 1:
            # Decode hot path: no output allocation, slices or copy kernel.
            # Empty local ranks still participate in dispatch and combine.
            return self.kernel.apply(
                hidden_states=hidden_states, w1=w1, w2=w2,
                topk_weights=topk_weights, topk_ids=topk_ids,
                activation=activation, global_num_experts=global_num_experts,
                expert_map=expert_map, apply_router_weight_on_input=apply_router_weight_on_input,
                shared_experts=shared_experts, shared_experts_input=shared_experts_input,
            )
        output = torch.empty_like(hidden_states)
        for index in range(rounds):
            start = min(index * self.capacity, hidden_states.size(0))
            end = min((index + 1) * self.capacity, hidden_states.size(0))
            chunk = self.kernel.apply(
                hidden_states=hidden_states[start:end], w1=w1, w2=w2,
                topk_weights=topk_weights[start:end], topk_ids=topk_ids[start:end],
                activation=activation, global_num_experts=global_num_experts,
                expert_map=expert_map, apply_router_weight_on_input=apply_router_weight_on_input,
                # SharedExperts owns the full input/output and is invoked once.
                shared_experts=shared_experts if index == 0 else None,
                shared_experts_input=shared_experts_input if index == 0 else None,
            )
            output[start:end].copy_(chunk)
        return output


def pack_step5_masked_weight_(weight):
    """Pack CT high-first signed nibbles into Marlin2/QQQ ASM storage.

    Equivalent to DeepGEMM's weight4bit_nt_kpack2_marlin2_qqq_from_packed.
    Convert one expert at a time; retain the Parameter's INT8 shape/storage.
    Only the native GEMM receives an INT32 view. Channel scales stay separate.
    """
    if weight.ndim != 3 or weight.dtype != torch.int8 or not weight.is_contiguous():
        raise ValueError("Step5 masked needs contiguous INT8 [E,N,K/2] weights")
    E, N, packed_k = weight.shape
    K = packed_k * 2
    if N % 256 or K % 128:
        raise ValueError(f"Step5 masked ASM requires N%256=0 and K%128=0; got N={N}, K={K}")
    for e in range(E):
        source = weight[e].to(torch.uint8)
        q = torch.stack((source >> 4, source & 15), dim=-1)
        q = q.reshape(N // 16, 16, K // 64, 4, 16)
        q = q.permute(2, 0, 3, 1, 4).contiguous().reshape(-1, 8)
        words = torch.zeros(q.shape[0], dtype=torch.int32, device=weight.device)
        for i, index in enumerate((0, 4, 1, 5, 2, 6, 3, 7)):
            words |= q[:, index].to(torch.int32) << (4 * i)
        weight[e].copy_(words.view(torch.int8).reshape(N, packed_k))
    return weight.view(torch.int32)


def _ll_scales(scales, experts, capacity):
    if scales is None or scales.dtype != torch.float32:
        raise ValueError("Step5 LL requires FP32 per-token INT8 scales")
    if scales.numel() != experts * capacity:
        raise ValueError("Step5 LL scale count does not match expert capacity")
    return scales.reshape(experts, capacity, 1).contiguous()


def _ll_boltops_capacity(moe, num_dispatchers, capacity):
    """Static upper bound on received rows per expert, safe for graph replay.

    Each source token visits an expert at most once (the LL buffer contract).
    Reuse synchronized host counts; never read GPU expert counts to compact.
    Keep full capacity when synchronization evidence is unavailable.
    """
    try:
        ctx = get_forward_context()
    except AssertionError:
        return capacity
    tokens = getattr(ctx, "_hcu_step5_ll_global_tokens", None)
    if tokens is None:
        return capacity
    sp_size = getattr(moe, "sp_size", 1)
    per_rank = (tokens + sp_size - 1) // sp_size
    return min(capacity, per_rank * num_dispatchers)


class Step5LLBoltOpsExperts(mk.FusedMoEExpertsModular):
    """Adapt LL batches to BoltOps top-1 local routing without double weighting."""

    def __init__(self, moe_config, quant_config, max_num_tokens, num_dispatchers):
        super().__init__(moe_config, quant_config, max_num_tokens, num_dispatchers)
        self._layer = None

    @staticmethod
    def activation_format():
        return mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def _supports_current_device():
        return current_platform.is_rocm()

    @staticmethod
    def _supports_no_act_and_mul():
        return False

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key):
        return (weight_key, activation_key) == (
            kInt4W4A8StaticChannelSym, kInt8DynamicTokenSym,
        )

    @staticmethod
    def _supports_activation(activation):
        return activation in (MoEActivation.SILU, MoEActivation.SWIGLUSTEP)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config):
        return (
            moe_parallel_config.use_ep
            and moe_parallel_config.all2all_backend == "deepep_low_latency"
        )

    @staticmethod
    def adjust_N_for_activation(N, activation):
        return N // 2

    def process_weights_after_loading(self, layer):
        self._layer = layer
        # Identity physical slots plus one inactive expert for padded rows.
        # The vendor sorter supports inactive expert_map entries, not an
        # undocumented negative top-k ID. Build once, outside graph capture.
        E = layer.w13_weight_scale.size(0)
        self._local_expert_map = torch.arange(
            E + 1, device=layer.w13_weight_scale.device, dtype=torch.int32,
        )
        self._local_expert_map[-1] = -1

    def moe_problem_size(self, a1, w1, w2, topk_ids):
        # Channel scales preserve the logical output dimension.
        if self._layer is None:
            raise RuntimeError("Step5 LL experts used before weight postprocessing")
        return (
            w1.size(0), a1.size(1), self._layer.w13_weight_scale.size(1),
            a1.size(-1), topk_ids.size(-1),
        )

    def workspace_shapes(self, M, N, K, topk, global_num_experts,
                         local_num_experts, expert_tokens_meta, activation):
        return ((0,), (0,), (local_num_experts, M, K))

    def finalize_weight_and_reduce_impl(self):
        return TopKWeightAndReduceDelegate()

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
              activation, global_num_experts, expert_map, a1q_scale, a2_scale,
              workspace13, workspace2, expert_tokens_meta,
              apply_router_weight_on_input):
        if apply_router_weight_on_input:
            raise ValueError("Step5 LL requires output-side routing weights")
        if self._layer is None or expert_tokens_meta is None:
            raise RuntimeError("Step5 LL requires loaded weights and expert counts")
        if hidden_states.ndim != 3 or hidden_states.dtype != torch.int8:
            raise ValueError("Step5 LL BoltOps expects INT8 [E, T, K] activations")
        E, full_T, K = hidden_states.shape
        if E != w1.size(0) or output.shape != hidden_states.shape:
            raise ValueError("Step5 LL expert/output layout mismatch")
        counts = expert_tokens_meta.expert_num_tokens
        if counts.shape != (E,):
            raise ValueError("Step5 LL requires one valid count per local expert")
        scales = _ll_scales(a1q_scale, E, full_T)
        T = _ll_boltops_capacity(self.moe_config, self.num_dispatchers, full_T)
        if T == 0:
            return
        valid = torch.arange(T, device=hidden_states.device)[None, :] < counts[:, None]
        # BoltOps accepts prequantized INT8 with FP32 per-token scales.
        # Do not dequantize/requantize the LL payload. Crop using the global
        # bound before any materialization, rather than processing E*full_T.
        x = hidden_states[:, :T].contiguous().view(E * T, K)
        input_scales = torch.where(valid[..., None], scales[:, :T], 0.0)
        input_scales = input_scales.reshape(E * T, 1).contiguous()
        local_ids = torch.arange(E, device=x.device, dtype=torch.int32)
        local_ids = torch.where(valid, local_ids[:, None], E).reshape(E * T, 1).contiguous()
        unit_weights = torch.ones((E * T, 1), device=x.device, dtype=torch.float32)
        from .ct_w4a8_step5 import _packed_w4a8_experts

        result = _packed_w4a8_experts()(
            x, w1, w2, unit_weights, local_ids,
            inplace=False, activation=activation.value, is_gated=True,
            use_int4_w4a8=True, per_channel_quant=True,
            w1_scale=self._layer.w13_weight_scale,
            w2_scale=self._layer.w2_weight_scale,
            a1_scale=input_scales,
            # Valid IDs already name LOCAL physical slots. The identity map
            # only excludes padding; it does not apply global routing again.
            expert_map=self._local_expert_map, global_num_experts=E + 1,
            output_dtype=output.dtype, routed_scaling_factor=1.0,
            gemm1_limit=7.0 if activation == MoEActivation.SWIGLUSTEP else None,
        ).reshape(E, T, K)
        output[:, :T].copy_(torch.where(valid[..., None], result, 0.0))
        # Slots beyond T are provably invalid; LL combine never reads them.


class Step5W4A8LLExperts(Step5LLBoltOpsExperts):
    """Explicit deep_gemm backend: two ordinary masked ASM GEMMs."""

    def process_weights_after_loading(self, layer):
        # Fail on a missing symbol before modifying any checkpoint storage.
        from deepgemm import m_grouped_w4a8_gemm_nt_masked
        weights = (layer.w13_weight, layer.w2_weight)
        generation = (getattr(layer, "_step5_ct_load_generation", 0), *(id(w) for w in weights))
        if getattr(layer, "_step5_hipc_generation", None) is not None:
            raise RuntimeError("Reload original CT weights before switching HIPC to masked ASM")
        # Validate both projections before packing either one.
        for weight in weights:
            if (weight.ndim != 3 or weight.dtype != torch.int8
                    or not weight.is_contiguous() or weight.size(1) % 256
                    or (weight.size(2) * 2) % 128):
                raise ValueError("Step5 masked ASM needs contiguous INT8 weights with N%256=0, K%128=0")
        if getattr(layer, "_step5_masked_generation", None) != generation:
            with torch.no_grad():
                for weight in weights:
                    pack_step5_masked_weight_(weight.detach())
            layer._step5_masked_generation = generation
        self._w13, self._w2 = (w.detach().view(torch.int32) for w in weights)
        self._gemm = m_grouped_w4a8_gemm_nt_masked
        self._layer = layer

    def workspace_shapes(self, M, N, K, topk, global_num_experts,
                         local_num_experts, expert_tokens_meta, activation):
        return ((local_num_experts, M, N), (0,), (local_num_experts, M, K))

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
              activation, global_num_experts, expert_map, a1q_scale, a2_scale,
              workspace13, workspace2, expert_tokens_meta, apply_router_weight_on_input):
        if apply_router_weight_on_input:
            raise ValueError("Step5 LL requires output-side routing weights")
        if hidden_states.dtype != torch.int8 or output.dtype != torch.bfloat16:
            raise ValueError("Step5 masked ASM requires INT8 activations and BF16 output")
        if not self._supports_activation(activation):
            raise ValueError(f"Unsupported Step5 W4A8 activation: {activation}")
        counts = expert_tokens_meta.expert_num_tokens.to(torch.int32).contiguous()
        experts, capacity, hidden = hidden_states.shape
        gate_up_dim = self._layer.w13_weight_scale.size(1)
        a1q_scale = _ll_scales(a1q_scale, experts, capacity)
        if capacity == 0:
            return
        gate_up = _resize_cache(workspace13, (experts, capacity, gate_up_dim))
        # expected_m tunes launch geometry only. masked_m owns correctness.
        expected_m = max(1, min(capacity, (topk_ids.size(0) * self.num_dispatchers
                                         * topk_ids.size(-1) + global_num_experts - 1)
                                        // global_num_experts))
        self._gemm(
            (hidden_states, a1q_scale), (self._w13, self.quant_config.w1_scale),
            gate_up, counts, expected_m,
        )
        # Fuse activation and quantization, iterating only valid expert rows.
        # Avoid a separate BF16 activated tensor and a launch over E*capacity.
        if activation == MoEActivation.SWIGLUSTEP:
            from lightop import fuse_silu_mul_clamp_quant_ep

            quantized, scales = fuse_silu_mul_clamp_quant_ep(
                gate_up, limit=7.0, mask_m=counts, expect_m=expected_m,
            )
        else:
            from lightop.activation import fuse_silu_mul_quant_ep

            quantized, scales = fuse_silu_mul_quant_ep(
                gate_up, tokens_per_expert=counts, expect_m=expected_m,
            )
        self._gemm(
            (quantized, _ll_scales(scales, experts, capacity)), (self._w2, self.quant_config.w2_scale), output, counts, expected_m,
        )


def make_step5_w4a8_ll_kernel(moe, quant, layer):
    from vllm.model_executor.layers.fused_moe.all2all_utils import maybe_make_prepare_finalize
    bounded = bounded_ll_config(moe)
    tables = getattr(layer, "_expert_routing_tables", None)
    prepare = maybe_make_prepare_finalize(
        moe=bounded, quant_config=quant,
        routing_tables=tables() if callable(tables) else None,
        allow_new_interface=True, use_monolithic=False,
    )
    if prepare is None or prepare.activation_format != mk.FusedMoEActivationFormat.BatchedExperts:
        raise RuntimeError("Step5 W4A8 requires a batched DeepEP LL prepare/finalize")
    experts_cls = (Step5W4A8LLExperts if getattr(moe, "moe_backend", "auto") == "deep_gemm"
                   else Step5LLBoltOpsExperts)
    experts = experts_cls(
        bounded, quant, prepare.max_num_tokens_per_rank(), prepare.num_dispatchers(),
    )
    experts.process_weights_after_loading(layer)
    return ChunkedLLKernel(mk.FusedMoEKernel(prepare, experts), moe)
