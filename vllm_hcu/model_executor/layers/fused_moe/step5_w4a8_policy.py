# SPDX-License-Identifier: Apache-2.0
"""Dependency-light selection policy; unrelated models retain their old backend."""

import os
import re
from vllm_hcu.runtime_compat.step5 import is_step5_config, is_step5_owner


def uses_step5_deepep(config):
    """Select by the actual backend, independently of a deployment geometry."""
    parallel = getattr(config, "moe_parallel_config", None)
    if parallel is not None:
        return bool(
            getattr(parallel, "use_deepep_auto_kernels", False)
            or getattr(parallel, "all2all_backend", "") in (
                "deepep_auto", "deepep_high_throughput", "deepep_low_latency"
            )
        )
    from vllm_hcu.patch.config import get_hcu_config

    return bool(
        (getattr(config, "additional_config", None) is not None
         and get_hcu_config(config).deepep_auto)
        or getattr(getattr(config, "parallel_config", None), "all2all_backend", "")
        in ("deepep_auto", "deepep_high_throughput", "deepep_low_latency")
    )


def _scheme_field(value, key, default=None):
    item = value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)
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
                raise ValueError(f"Invalid Step5 quantization target: {target}") from exc
        # Both checkpoint names and vLLM mapped names retain these namespaces.
        scope = target.replace(r"\.", ".")
        expert = "moe.experts" in scope
        dense = any(name in scope for name in (
            "lm_head", "self_attn", "mlp", "share_expert", "shared_head"
        ))
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
    bits = {_scheme_field(group.get("weights"), "num_bits")
            for group in groups.values()}
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
            raise ValueError("Step5 mixed checkpoint supports only INT4 experts and INT8 dense weights")
        if mixed:
            _validate_mixed_targets(group, weight_bits)
    return True


def profile_for_model(config):
    phase = os.environ.get("VLLM_HCU_STEP5_W4A8_PROFILE", "")
    if not phase or not is_step5_config(config):
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
        # TP/DP/EP/PP sizes belong to the framework. In particular, do not
        # require one eight-card deployment or a fixed local expert count.
        if parallel.enable_eplb:
            raise ValueError("Step5 packed DeepGEMM weights require static expert placement")
        return phase
    if phase == "auto":
        raise ValueError("Step5 auto profile requires DeepEP auto dispatch")
    expected = (1, 8, 1) if phase == "prefill" else (4, 2, 1)
    if (
        (
            parallel.tensor_parallel_size,
            parallel.data_parallel_size,
            parallel.pipeline_parallel_size,
        )
        != expected
        or not parallel.enable_expert_parallel
        or parallel.enable_eplb
    ):
        raise ValueError(f"Step5 {phase} requires static TP/DP/PP={expected}, EP8")
    return phase


def resolve_profile(moe):
    phase = os.environ.get("VLLM_HCU_STEP5_W4A8_PROFILE", "")
    if not phase or not is_step5_owner(moe):
        return None
    activation = getattr(moe.activation, "value", moe.activation)
    if activation != "swiglustep":
        return None
    if phase not in ("prefill", "decode", "auto"):
        raise ValueError("VLLM_HCU_STEP5_W4A8_PROFILE must be prefill, decode or auto")
    parallel = moe.moe_parallel_config
    if uses_step5_deepep(moe):
        if parallel.enable_eplb:
            raise ValueError("Step5 packed DeepGEMM weights require static expert placement")
        return "auto"
    if (
        parallel.enable_eplb
        or parallel.all2all_backend != "allgather_reducescatter"
        or parallel.ep_size != 8
        or parallel.dp_size != (8 if phase == "prefill" else 2)
    ):
        raise ValueError(
            "Step5 native W4A8 requires static EP8 AG/RS and matching P/D ranks"
        )
    return phase


def padded_route_rows(tokens, topk, experts, alignment=64):
    routes = tokens * topk
    return (
        (routes + min(routes, experts) * (alignment - 1) + alignment - 1)
        // alignment
        * alignment
    )


def w4a8_execution_plan(tokens, phase):
    """Select the mode and its routing alignment as one inseparable plan.

    Decode shapes include the C8-C64 verify-token family. Larger real-prefill
    batches and shapes outside the validated family retain MODE1002/pad64.
    No device routing counts are copied to the host for this decision.
    """
    mode = int(os.environ.get("HCU_STEP5_W4A8_MODE", "1002"))
    if mode not in (1002, 1006):
        raise ValueError("HCU_STEP5_W4A8_MODE must be 1002 or 1006")
    if mode == 1006 and phase == "decode" and tokens in (
        8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256
    ):
        return 1006, 128
    return 1002, 64
