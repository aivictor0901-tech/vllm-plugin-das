# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Dependency-light Step5 family predicates and parallelism and W4A8 policies."""

import os
import re


def is_step5_config(config):
    model = getattr(config, "model_config", config)
    hf = getattr(model, "hf_config", model)
    text = getattr(hf, "text_config", hf)
    return getattr(text, "model_type", "") in ("step5", "step5_mtp") or any(
        name.startswith("Step5") for name in (getattr(hf, "architectures", None) or ())
    )


def is_step5_owner(owner=None):
    """Resolve ownership at construction time; unknown owners never opt in."""
    marker = getattr(owner, "_hcu_step5_owner", None)
    if marker is not None:
        return marker is True
    config = getattr(owner, "_hcu_vllm_config", None)
    if config is None:
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
    return config is not None and is_step5_config(config)


def supports_step5_ll_topology(tp: int, dp: int, pp: int = 1) -> bool:
    # EP is TP * DP: 8 ranks for the one-host deployment, otherwise 16.
    return (tp, dp, pp) in ((4, 2, 1), (4, 4, 1), (1, 16, 1))


def uses_step5_ll_fused_restore(config) -> bool:
    p = config.parallel_config
    # Step5-VL stores the language model settings in its text subconfig.
    text_config = getattr(config.model_config, "hf_text_config", None)
    if text_config is None:
        hf_config = config.model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
    return bool(
        p.enable_expert_parallel
        and p.all2all_backend == "deepep_low_latency"
        and supports_step5_ll_topology(
            p.tensor_parallel_size, p.data_parallel_size, p.pipeline_parallel_size
        )
        and not p.enable_dbo
        and not config.compilation_config.pass_config.enable_sp
        and getattr(text_config, "fp32_residual_connection", False)
    )


def allows_static_step5_ll_eplb(config=None):
    if os.getenv("HCU_STEP5_LL_STATIC_EPLB", "0") != "1":
        return False
    from vllm.config import get_current_vllm_config_or_none
    from vllm_hcu.patch.config import get_hcu_config

    cfg = config if hasattr(config, "parallel_config") else None
    if cfg is None:
        cfg = getattr(config, "_hcu_vllm_config", None)
    if cfg is None:
        cfg = get_current_vllm_config_or_none()
    if cfg is None or not is_step5_config(cfg):
        return False
    p = cfg.parallel_config
    hcu = get_hcu_config(cfg)
    return bool(
        p.enable_eplb
        and p.enable_expert_parallel
        and p.all2all_backend == "deepep_low_latency"
        and supports_step5_ll_topology(
            p.tensor_parallel_size, p.data_parallel_size, p.pipeline_parallel_size
        )
        and not p.enable_dbo
        and not cfg.compilation_config.pass_config.enable_sp
        and not p.eplb_config.use_async
        and p.eplb_config.num_redundant_experts == 0
        and hcu.eplb_disable_rearrange
        and bool(hcu.expert_map_path) != bool(hcu.expert_map_record_path)
    )


def _map_int64_ids(self, topk_ids):
    """Reuse the native mapper with INT64 input; avoid a per-layer cast copy."""
    import torch
    from vllm.model_executor.layers.fused_moe.router.base_router import (
        BaseRouter,
        _eplb_map_and_record_i32_kernel,
    )
    from vllm.v1.worker.ubatching import dbo_current_ubatch_id

    state = self.eplb_state
    if state is None or topk_ids.dtype != torch.int64:
        return BaseRouter._apply_eplb_mapping(self, topk_ids)
    if topk_ids.numel() == 0:
        return topk_ids
    ids = topk_ids.contiguous()
    out = torch.empty_like(ids)
    load = state.expert_load_view
    assert load is not None and load.is_contiguous()
    count = state.logical_replica_count.contiguous()
    mapping = state.logical_to_physical_map.contiguous()
    unpadded = state.num_unpadded_tokens_tensors[dbo_current_ubatch_id()]
    # The upstream Triton kernel loads through a typed pointer and converts
    # the loaded ID to int64. Its historical i32 name does not restrict input.
    _eplb_map_and_record_i32_kernel[((ids.numel() + 255) // 256,)](
        ids,
        count,
        mapping,
        out,
        load,
        state.should_record_tensor,
        unpadded,
        count.shape[0],
        mapping.shape[1],
        load.shape[0],
        ids.numel(),
        ids.shape[-1],
        HAS_NUM_UNPADDED=unpadded is not None,
        BLOCK_SIZE=256,
    )
    return out


def install_int64_eplb_mapping(router, config):
    """Opt in only for the validated Step5 static-placement LL topology."""
    if os.getenv("HCU_STEP5_LL_EPLB_I64", "0") != "1":
        return
    if not allows_static_step5_ll_eplb(config):
        raise ValueError(
            "INT64 EPLB mapping requires static Step5 LL placement "
            "with TP4/DP2, TP4/DP4 or TP1/DP16"
        )
    from types import MethodType

    router._apply_eplb_mapping = MethodType(_map_int64_ids, router)


def uses_step5_deepep(config):
    """Select by the actual backend, independently of a deployment geometry."""
    parallel = getattr(config, "moe_parallel_config", None)
    if parallel is not None:
        return bool(
            getattr(parallel, "use_deepep_auto_kernels", False)
            or getattr(parallel, "all2all_backend", "")
            in ("deepep_auto", "deepep_high_throughput", "deepep_low_latency")
        )
    from vllm_hcu.patch.config import get_hcu_config

    return bool(
        (
            getattr(config, "additional_config", None) is not None
            and get_hcu_config(config).deepep_auto
        )
        or getattr(getattr(config, "parallel_config", None), "all2all_backend", "")
        in ("deepep_auto", "deepep_high_throughput", "deepep_low_latency")
    )


def _scheme_field(value, key, default=None):
    item = (
        value.get(key, default)
        if isinstance(value, dict)
        else getattr(value, key, default)
    )
    return getattr(item, "value", item)


def _validate_channel_int_scheme(scheme, bits, storage_format):
    field = _scheme_field

    w = scheme.get("weights")
    a = scheme.get("input_activations")
    if not (
        scheme.get("format") == storage_format
        and field(w, "num_bits") == bits
        and field(w, "type") == "int"
        and field(w, "strategy") == "channel"
        and field(w, "symmetric") is True
        and not field(w, "dynamic", False)
        and field(w, "group_size") in (None, -1)
        and field(w, "actorder") is None
        and field(a, "num_bits") == 8
        and field(a, "type") == "int"
        and field(a, "strategy") == "token"
        and field(a, "symmetric") is True
        and field(a, "dynamic") is True
        and field(a, "group_size") in (None, -1)
        and field(a, "actorder") is None
        and scheme.get("output_activations") is None
    ):
        raise ValueError(
            f"Step5 requires {storage_format} symmetric channel INT{bits} "
            "weights and dynamic token INT8 activations"
        )


def validate_gptq_w4a8_scheme(scheme):
    """Accept only the channel INT4 / dynamic token INT8 storage contract."""
    _validate_channel_int_scheme(scheme, 4, "pack-quantized")


def validate_dense_w8a8_scheme(scheme):
    """Dense INT8 is stored directly, not in the experts' packed INT4 ABI."""
    _validate_channel_int_scheme(scheme, 8, "int-quantized")


def _validate_mixed_targets(group, bits):
    """Keep the native INT4 expert adapter scoped to routed expert targets.

    A broad ``Linear`` target is valid for a legacy INT4-only checkpoint, but
    is ambiguous in a mixed checkpoint. Actual per-expert resolution is still
    checked by Step5GPTQW4A8Config.get_quant_method; this check must not turn a
    dense INT8 group into an INT4 expert scheme or silently accept an unrelated
    quantization group.
    """
    targets = group.get("targets", [])
    if not targets:
        raise ValueError("Step5 mixed INT4/INT8 groups need explicit module targets")
    for target in targets:
        if not isinstance(target, str):
            raise ValueError("Step5 mixed quantization targets must be strings")
        if target.startswith("re:"):
            try:
                re.compile(target[3:])
            except re.error as exc:
                raise ValueError(
                    f"Invalid Step5 quantization target: {target}"
                ) from exc
        # Both checkpoint names and vLLM mapped names retain these namespaces.
        scope = target.replace(r"\.", ".")
        expert = "moe.experts" in scope
        dense = any(
            name in scope
            for name in ("lm_head", "self_attn", "mlp", "share_expert", "shared_head")
        )
        if (bits == 4 and not expert) or (bits == 8 and (expert or not dense)):
            raise ValueError(
                f"Step5 mixed INT{bits} target has the wrong module scope: {target}"
            )


def compressed_w4a8_config(config):
    quant = getattr(config, "quant_config", None)
    raw = getattr(quant, "config", None)
    if raw is None:
        model = getattr(config, "model_config", config)
        raw = getattr(model.hf_config, "quantization_config", None)
    if not isinstance(raw, dict) or raw.get("quant_method") != "compressed-tensors":
        return False
    groups = raw.get("config_groups", {})
    if not groups:
        return False
    bits = {
        _scheme_field(group.get("weights"), "num_bits") for group in groups.values()
    }
    # An INT8-only checkpoint does not select the native INT4 MoE adapter.
    if 4 not in bits:
        return False
    mixed = bits != {4}
    for group in groups.values():
        group = dict(group, format=group.get("format", raw.get("format")))
        weight_bits = _scheme_field(group.get("weights"), "num_bits")
        if weight_bits == 4:
            validate_gptq_w4a8_scheme(group)
        elif weight_bits == 8:
            validate_dense_w8a8_scheme(group)
        else:
            raise ValueError(
                "Step5 mixed checkpoint supports only INT4 experts and INT8 dense weights"
            )
        if mixed:
            _validate_mixed_targets(group, weight_bits)
    return True


def profile_for_model(config):
    if not is_step5_config(config):
        return None
    phase = os.environ.get("VLLM_HCU_STEP5_W4A8_PROFILE", "")
    if not phase:
        return None
    if phase not in ("prefill", "decode", "auto"):
        raise ValueError("VLLM_HCU_STEP5_W4A8_PROFILE must be prefill, decode or auto")
    model = config.model_config
    hf = model.hf_config
    quantization = getattr(model, "quantization", None)
    if quantization is None:
        quantization = (getattr(hf, "quantization_config", None) or {}).get(
            "quant_method"
        )
    gptq = quantization == "compressed-tensors" and compressed_w4a8_config(config)
    if quantization != "slimquant_w4a8" and not gptq:
        raise ValueError(
            "Step5 native W4A8 requires SlimQuant or channel GPTQ W4A8 weights"
        )
    parallel = config.parallel_config
    if getattr(config, "lora_config", None) is not None:
        raise ValueError("Step5 native W4A8 LoRA has not been enabled")
    if uses_step5_deepep(config):
        if phase in ("decode", "auto") and (
            not parallel.enable_expert_parallel
            or parallel.tensor_parallel_size * parallel.data_parallel_size <= 1
        ):
            raise ValueError(
                "Step5 Decode requires distributed EP with DeepEP masked experts; "
                "local EP1 and ordinary TP contiguous experts are Prefill-only"
            )
        if (
            phase in ("decode", "auto")
            and getattr(parallel, "all2all_backend", "") == "deepep_high_throughput"
        ):
            raise ValueError(
                "Step5 Decode requires DeepEP low-latency masked experts; "
                "use deepep_low_latency or deepep_auto"
            )
        # TP/DP/EP/PP sizes belong to the framework. In particular, do not
        # require one eight-card deployment or a fixed local expert count.
        if parallel.enable_eplb and not allows_static_step5_ll_eplb(config):
            raise ValueError(
                "Step5 packed DeepGEMM weights require static expert placement"
            )
        return phase
    if phase != "prefill":
        raise ValueError(
            "Step5 Decode requires DeepEP low-latency masked experts; "
            "static AG/RS is supported only for Prefill"
        )
    expected = ((1, 8, 1),)
    if (
        (
            parallel.tensor_parallel_size,
            parallel.data_parallel_size,
            parallel.pipeline_parallel_size,
        )
        not in expected
        or parallel.all2all_backend != "allgather_reducescatter"
        or not parallel.enable_expert_parallel
        or parallel.enable_eplb
    ):
        raise ValueError(
            f"Step5 {phase} requires static TP/DP/PP in {expected}, with EP enabled"
        )
    return phase


def resolve_profile(moe):
    if not is_step5_owner(moe):
        return None
    phase = os.environ.get("VLLM_HCU_STEP5_W4A8_PROFILE", "")
    if not phase:
        return None
    activation = getattr(moe.activation, "value", moe.activation)
    if activation != "swiglustep":
        return None
    if phase not in ("prefill", "decode", "auto"):
        raise ValueError("VLLM_HCU_STEP5_W4A8_PROFILE must be prefill, decode or auto")
    parallel = moe.moe_parallel_config
    if uses_step5_deepep(moe):
        if phase in ("decode", "auto") and not getattr(
            parallel, "use_all2all_kernels", True
        ):
            raise ValueError(
                "Step5 Decode requires distributed EP with DeepEP masked experts; "
                "local contiguous experts are Prefill-only"
            )
        if (
            phase in ("decode", "auto")
            and getattr(parallel, "all2all_backend", "") == "deepep_high_throughput"
        ):
            raise ValueError(
                "Step5 Decode requires DeepEP low-latency masked experts; "
                "use deepep_low_latency or deepep_auto"
            )
        if parallel.enable_eplb and not allows_static_step5_ll_eplb(moe):
            raise ValueError(
                "Step5 packed DeepGEMM weights require static expert placement"
            )
        return "auto"
    if phase != "prefill":
        raise ValueError(
            "Step5 Decode requires DeepEP low-latency masked experts; "
            "static AG/RS is supported only for Prefill"
        )
    expected = ((8, 8),)
    if (
        parallel.enable_eplb
        or parallel.all2all_backend != "allgather_reducescatter"
        or (parallel.ep_size, parallel.dp_size) not in expected
    ):
        raise ValueError(
            f"Step5 native W4A8 requires static AG/RS with (EP, DP) in {expected}"
        )
    return phase


def static_agrs_local_experts(moe):
    """Complete local experts for the static AG/RS Prefill profile."""
    if resolve_profile(moe) != "prefill" or uses_step5_deepep(moe):
        raise ValueError("Step5 static expert shapes require an AG/RS Prefill profile")
    return 352 // moe.moe_parallel_config.ep_size


def padded_route_rows(tokens, topk, experts, alignment=64):
    routes = tokens * topk
    return (
        (routes + min(routes, experts) * (alignment - 1) + alignment - 1)
        // alignment
        * alignment
    )
