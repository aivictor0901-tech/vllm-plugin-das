"""Opt-in Step5 W4A8 backend for the existing modular EP/DP execution path.

Selected by the normal SlimQuant method only for the explicit Step5 profile.
"""

import torch
import triton
from vllm.logger import init_logger

logger = init_logger(__name__)
from functools import lru_cache
from .step5_w4a8_policy import padded_route_rows, w4a8_execution_plan
from .step5_w4a8_kernels import (
    _activation,
    _weighted_output,
    activation_persistent,
    small_permute,
)
from .step5_w4a8_routing import permute64


def _generation(layer):
    return tuple(
        (id(t), t.data_ptr(), t._version, tuple(t.shape))
        for t in (
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
        )
    )


def _no_reload(*args, **kwargs):
    raise RuntimeError("W4A8 packed layout: restart from checkpoint for reload or EPLB")


@lru_cache(maxsize=4)
def _make_experts_class(phase, backend="deepgemm"):
    is_prefill = phase == "prefill"
    # Initialize the public package before HCU replacement modules, avoiding
    # the partial deep_gemm_utils import cycle seen in the original probe.
    import vllm.model_executor.layers.fused_moe  # noqa: F401 - initialize before HCU adapters
    from vllm_hcu.model_executor.layers.quantization import (
        slimquant_w4a8_deepgemm_runtime as dg,
    )
    from vllm.model_executor.layers.fused_moe.utils import _resize_cache
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm_hcu.ops.step5.w4a8 import require_gemm_op

    op = require_gemm_op(backend)

    def expert_gemm(a, b, output, indices, config):
        op(
            a[0],
            b[0],
            output,
            a[1].contiguous(),
            b[1].contiguous(),
            indices.contiguous(),
            config["MODE"],
        )
        return output

    class Step5W4A8Experts(dg.DeepEPDeepGemmW4A8ContiguousExperts):
        _hcu_step5_owner = True
        _hcu_step5_w4a8 = True
        _hcu_step5_w4a8_gemm_backend = backend

        @staticmethod
        def _supports_activation(activation):
            return activation == MoEActivation.SWIGLUSTEP

        def process_weights_after_loading(self, layer):
            # Reuse the common validator and packer, but retain the Parameters
            # as owners here: the base method deliberately packs clones for
            # reload support. Changing that policy for other models would be
            # unsafe, and a second packed owner exceeds this profile's budget.
            dg._validate_w4a8_channel_weights(layer)
            old = getattr(layer, "_hcu_step5_w4a8_owner", None)
            if old is not None:
                if old != _generation(layer):
                    raise RuntimeError(
                        "Packed W4 owner changed; reload/EPLB unsupported"
                    )
            else:
                for parameter in (layer.w13_weight, layer.w2_weight):
                    before = (
                        id(parameter),
                        parameter.data_ptr(),
                        parameter.untyped_storage().nbytes(),
                        tuple(parameter.shape),
                    )
                    if not parameter.is_contiguous() or parameter.storage_offset():
                        raise ValueError(
                            "Single-owner W4 packing requires contiguous complete storage"
                        )
                    with torch.no_grad():
                        packed = dg.pack_w4a8_moe_hipc_weight(parameter)
                    after = (
                        id(parameter),
                        packed.data_ptr(),
                        packed.untyped_storage().nbytes(),
                        tuple(packed.shape),
                    )
                    if before != after:
                        raise RuntimeError(
                            "W4A8 repack allocated a second weight owner"
                        )
                layer._hcu_step5_w4a8_owner = _generation(layer)
                for parameter in (
                    layer.w13_weight,
                    layer.w2_weight,
                    layer.w13_weight_scale,
                    layer.w2_weight_scale,
                ):
                    parameter.requires_grad_(False)
                    if callable(getattr(parameter, "weight_loader", None)):
                        parameter.weight_loader = _no_reload
            self._deepgemm_w13, self._deepgemm_w2 = layer.w13_weight, layer.w2_weight

        def workspace_shapes(
            self,
            M,
            N,
            K,
            topk,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        ):
            # This private adapter stores only INT8 gathered/activation data in
            # workspace13; gateup/down BF16 data lives in workspace2. Keep the
            # framework's BF16 dtype but reserve the exact byte capacity.
            base = super().workspace_shapes(
                M,
                N,
                K,
                topk,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
                activation,
            )
            _, alignment = w4a8_execution_plan(M, phase)
            if (not is_prefill and alignment == 64) or (N, K) != (3072, 4096):
                return base
            if expert_tokens_meta is None:
                aligned = padded_route_rows(M, topk, local_num_experts, alignment)
            else:
                from vllm_hcu.model_executor.layers.quantization import (
                    slimquant_w4a8_deepgemm_runtime as dg,
                )

                aligned = dg.compute_aligned_M(
                    M=M,
                    num_topk=topk,
                    local_num_experts=local_num_experts,
                    alignment=self.ALIGNMENT,
                    expert_tokens_meta=expert_tokens_meta,
                )
            # The final output can alias workspace13 after GEMM2, so also
            # retain M*K BF16 elements when the allocator reuses this buffer.
            first_elements = max(triton.cdiv(aligned * max(K, N // 2), 2), M * K)
            return ((triton.cdiv(first_elements, K), K), (aligned, max(N, K)), (M, K))

        def apply(
            self,
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            a1q_scale,
            a2_scale,
            workspace13,
            workspace2,
            expert_tokens_meta,
            apply_router_weight_on_input,
        ):
            if apply_router_weight_on_input or not self._supports_activation(
                activation
            ):
                raise ValueError(
                    "Step5 W4 requires output-side weighting and SwigluStep7"
                )
            if hidden_states.dtype != torch.int8 or a1q_scale is None:
                raise ValueError(
                    "Modular prepare must dispatch paired INT8 activations and FP32 scales"
                )
            if hidden_states.shape[0] == 0:
                return
            if getattr(expert_tokens_meta, "_hcu_deepep_ht_local_counts", False):
                expert_tokens_meta._hcu_step5_compact_rows = True
            E, _, N, K, topk = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
            use_small = (
                0 < hidden_states.shape[0] <= 256
                and expert_map is not None
                and topk_ids.is_contiguous()
                and expert_map.is_contiguous()
                and a1q_scale.is_contiguous()
                and expert_tokens_meta is None
            )
            use_large64 = (
                is_prefill
                and not use_small
                and expert_tokens_meta is None
                and expert_map is not None
                and a1q_scale.is_contiguous()
            )
            mode, alignment = w4a8_execution_plan(hidden_states.shape[0], phase)
            if not use_small:
                # The 128-row mode is validated only with the small-route
                # producer; other dispatchers retain their existing ABI.
                mode, alignment = 1002, 64
            aligned = dg.compute_aligned_M(
                M=topk_ids.shape[0],
                num_topk=topk,
                local_num_experts=E,
                alignment=self.ALIGNMENT,
                expert_tokens_meta=expert_tokens_meta,
            )
            if use_small:
                aligned = padded_route_rows(hidden_states.shape[0], topk, E, alignment)
            elif use_large64:
                aligned = padded_route_rows(hidden_states.shape[0], topk, E)
            gathered = _resize_cache(workspace13.view(dtype=torch.int8), (aligned, K))
            fused_scatter = getattr(expert_tokens_meta, "_hcu_step5_fused_scatter", None)
            if fused_scatter is not None:
                aq, scales, indices, inverse = fused_scatter
                if aq.dtype != torch.int8 or aq.shape[1] != K:
                    raise ValueError("Step5 W4A8 fused scatter requires matching INT8 expert rows")
                aligned = aq.shape[0]
            elif use_small:
                aq, scales, indices, inverse, _ = small_permute(
                    hidden_states,
                    self._ensure_2d_scale(a1q_scale),
                    topk_ids,
                    expert_map,
                    E,
                    gathered,
                    global_num_experts,
                    alignment,
                )
            elif use_large64:
                aq, scales, indices, inverse, _ = permute64(
                    hidden_states,
                    self._ensure_2d_scale(a1q_scale),
                    topk_ids,
                    expert_map,
                    E,
                    gathered,
                )
            else:
                aq, scales, indices, inverse, _ = dg.deepgemm_moe_permute(
                    aq=hidden_states,
                    aq_scale=self._ensure_2d_scale(a1q_scale),
                    topk_ids=topk_ids,
                    local_num_experts=E,
                    expert_map=expert_map,
                    expert_tokens_meta=expert_tokens_meta,
                    aq_out=gathered,
                )
            indices = indices.to(torch.int32).contiguous()
            gateup = _resize_cache(workspace2, (aligned, N))
            expert_gemm(
                (aq, scales),
                (self._deepgemm_w13, self.w1_scale),
                gateup,
                indices,
                config={"MODE": mode},
            )
            intermediate = _resize_cache(
                workspace13.view(dtype=torch.int8), (aligned, N // 2)
            )
            a_scale = torch.empty(
                (aligned, 1), device=hidden_states.device, dtype=torch.float32
            )
            if hidden_states.shape[0] >= 1024 and N // 2 == 1536:
                activation_persistent[(min(triton.cdiv(aligned, 8), 1024),)](
                    gateup,
                    intermediate,
                    a_scale,
                    indices,
                    aligned,
                    N // 2,
                    8,
                    512,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            else:
                _activation[(aligned,)](
                    gateup,
                    intermediate,
                    a_scale,
                    indices,
                    N // 2,
                    triton.next_power_of_2(N // 2),
                    num_warps=4,
                )
            down = _resize_cache(workspace2, (aligned, K))
            expert_gemm(
                (intermediate, a_scale),
                (self._deepgemm_w2, self.w2_scale),
                down,
                indices,
                config={"MODE": mode},
            )
            # Same provider/state protocol as FP8 fused experts. It declines
            # while DP-pipe owns combine or shared output is not ready. The
            # existing finalize hook then avoids a second shared addition.
            from vllm_hcu.model_executor.layers.fused_moe import ep_fused_moe

            provider = getattr(ep_fused_moe, "shared_add_provider", None)
            addition = provider(output) if provider is not None else None
            shared, start = addition if addition is not None else (output, 0)
            _weighted_output[(output.shape[0], triton.cdiv(K, 2048))](
                down,
                inverse,
                topk_weights,
                expert_map if expert_map is not None else topk_ids,
                topk_ids,
                output,
                shared,
                output.shape[0],
                K,
                topk,
                global_num_experts,
                expert_map is not None,
                addition is not None,
                start,
                shared.shape[0] if addition is not None else 0,
                2048,
                num_warps=4,
                enable_fp_fusion=False,
            )
            if fused_scatter is not None:
                from .step5_deepep import finish_scatter
                finish_scatter(expert_tokens_meta)

    return Step5W4A8Experts


def prepare_step5_w4a8(method, layer, phase):
    """Select modular W4 before model load; no legacy monolithic hook allowed."""
    import vllm.model_executor.layers.fused_moe  # noqa: F401 - initialize before HCU adapters
    from .step5_w4a8_policy import uses_step5_deepep
    if uses_step5_deepep(method.moe):
        from .step5_deepep import prepare_step5_deepep
        prepare_step5_deepep(method, layer, int4=True)
        return
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk

    from vllm_hcu.ops.step5.w4a8 import gemm_backend, require_gemm_op

    backend = gemm_backend()
    Experts = _make_experts_class(phase, backend)

    activation = getattr(method.moe.activation, "value", method.moe.activation)
    if activation != "swiglustep":
        raise ValueError("Step5 W4A8 experts require SwigluStep activation")
    parallel = method.moe.moe_parallel_config
    if parallel.enable_eplb or parallel.all2all_backend != "allgather_reducescatter":
        raise ValueError(
            "This W4 profile supports static EP8 AG/RS, not EPLB or DeepEP"
        )
    if tuple(layer.w13_weight.shape) != (44, 3072, 2048) or tuple(
        layer.w2_weight.shape
    ) != (44, 4096, 768):
        raise ValueError("Expected Step5 EP8 packed W4 dimensions")
    if getattr(layer, "swiglu_limit", 7.0) not in (None, 7.0, 7):
        raise ValueError("Step5 W4 activation clamp must remain7")
    if getattr(layer, "_hcu_step5_w4a8_owner", None) is not None:
        if layer._hcu_step5_w4a8_owner != _generation(layer):
            raise RuntimeError("Packed W4 generation changed")
        if method.moe_kernel is not None:
            previous = getattr(layer, "_hcu_step5_w4a8_backend", {}).get(
                "gemm_backend", "deepgemm"
            )
            if previous != backend:
                raise RuntimeError("Changing Step5 W4A8 GEMM backend requires a server restart")
            return
    quant = method.get_fused_moe_quant_config(layer)
    prepare = maybe_make_prepare_finalize(
        moe=method.moe,
        quant_config=quant,
        routing_tables=None,
        allow_new_interface=True,
        use_monolithic=False,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
        MoEPrepareAndFinalizeNaiveDPEPModular,
    )

    if type(prepare) is not MoEPrepareAndFinalizeNaiveDPEPModular:
        raise RuntimeError(
            "Expected modular naive DP/EP; refuse silent loss of DP pipeline"
        )
    experts = Experts(method.moe, quant)
    experts.process_weights_after_loading(layer)
    method.moe_quant_config = quant
    method.moe_kernel = mk.FusedMoEKernel(prepare, experts)
    decode_mode, decode_alignment = w4a8_execution_plan(128, phase)
    logger.info_once(
        "[step5-w4a8] gemm_backend=%s, decode MODE=%d/pad%d, profile=%s, static EP8",
        backend, decode_mode, decode_alignment, phase,
    )
    # The modular prepare owns quantize/dispatch and the modular kernel
    # owns pipe/DBO/combine. Outer naive dispatch must therefore be off.
    if not method.supports_internal_mk:
        raise RuntimeError("Framework did not recognize internal modular ownership")
    layer._hcu_step5_w4a8_backend = {
        "experts": type(experts).__name__,
        "prepare": type(prepare).__name__,
        "activation_dtype": "int8",
        "weight_bits": 4,
        "gemm_backend": backend,
        "gemm_op": str(require_gemm_op(backend)),
        "gemm_mode": decode_mode,
        "reduction_block": 2048,
        "small_route_alignment": decode_alignment,
        "prefill_activation": "persistent8x512",
        "profile": phase,
        "shared_add_protocol": "EP_COMBINE provider/finalize",
        "weight_scale": "canonical checkpoint s8, no x16 copy",
    }


def validate_loaded_model(model):
    layers = [
        module
        for module in model.modules()
        if getattr(module, "_hcu_step5_w4a8_owner", None) is not None
    ]
    storage = {}
    for layer in layers:
        for weight in (layer.w13_weight, layer.w2_weight):
            storage[(weight.device.index, weight.untyped_storage().data_ptr())] = (
                weight.untyped_storage().nbytes()
            )
    declared = [m for m in model.modules() if hasattr(m, "_hcu_step5_gptq_kind")]
    import os
    if os.environ.get("VLLM_HCU_STEP5_W4A8_PROFILE") == "auto" or any(
        hasattr(m, "_hcu_step5_deepep") for m in declared
    ):
        # PP stages own only their local layers. EP degree and padded physical
        # matrices determine the storage; no eight-rank byte constant applies.
        expected = {id(m) for m in declared if m._hcu_step5_gptq_kind == "w4a8"}
        expected_bytes = sum(w.numel() * w.element_size() for m in layers
                             for w in (m.w13_weight, m.w2_weight))
        if ({id(m) for m in layers} != expected or len(storage) != 2 * len(layers)
                or sum(storage.values()) != expected_bytes):
            raise RuntimeError("Step5 DeepEP requires one packed owner per local W4 expert matrix")
        logger.info("[step5-deepep] verified local W4 layers=%d, mixed layers=%d, packed bytes=%d",
                    len(layers), len(declared) - len(layers), sum(storage.values()))
        return
    expected_layers = 88
    if declared:
        expected = {id(m) for m in declared if m._hcu_step5_gptq_kind == "w4a8"}
        if len(declared) != 88 or not expected or expected != {id(m) for m in layers}:
            raise RuntimeError("Step5 GPTQ packed owners must match all declared W4A8 experts")
        for m in declared:
            if m._hcu_step5_gptq_kind == "bf16" and (
                m.w13_weight.dtype != torch.bfloat16 or m.w2_weight.dtype != torch.bfloat16
            ):
                raise RuntimeError("Step5 GPTQ ignored experts must retain BF16 weights")
            if m._hcu_step5_gptq_kind == "fp8" and (
                m.w13_weight.dtype != torch.float8_e4m3fn
                or m.w2_weight.dtype != torch.float8_e4m3fn
                or m.w13_weight_scale.dtype != torch.float32
                or m.w2_weight_scale.dtype != torch.float32
            ):
                raise RuntimeError("Step5 mixed FP8 experts require FP8 weights and FP32 scales")
        expected_layers = len(expected)
    if (len(layers) != expected_layers or len(storage) != expected_layers * 2
            or sum(storage.values()) != expected_layers * (36_540_776_448 // 88)):
        raise RuntimeError(
            "Step5 W4A8 must have one packed owner per declared EP8 expert layer"
        )
    logger.info(
        "[step5-w4a8] verified %d W4A8 layers, %d FP8 layers, %d BF16 layers, packed bytes=%d",
        expected_layers, sum(m._hcu_step5_gptq_kind == "fp8" for m in declared),
        sum(m._hcu_step5_gptq_kind == "bf16" for m in declared), sum(storage.values())
    )
