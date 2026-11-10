# SPDX-License-Identifier: Apache-2.0
"""Dependency-light selection policy; unrelated models retain their old backend."""

import os
from vllm_hcu.runtime_compat.step5 import is_step5_config, is_step5_owner


def validate_gptq_w4a8_scheme(scheme):
    """Accept only the channel INT4 / dynamic token INT8 storage contract."""
    def field(value, key, default=None):
        item = value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)
        return getattr(item, "value", item)

    w = scheme.get("weights")
    a = scheme.get("input_activations")
    if not (
        scheme.get("format") == "pack-quantized"
        and field(w, "num_bits") == 4
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
        and scheme.get("output_activations") is None
    ):
        raise ValueError("Step5 GPTQ requires symmetric channel INT4 weights and dynamic token INT8 activations")


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
    for group in groups.values():
        validate_gptq_w4a8_scheme(dict(group, format=group.get("format", raw.get("format"))))
    return True


def profile_for_model(config):
    phase = os.environ.get("VLLM_HCU_STEP5_W4A8_PROFILE", "")
    if not phase or not is_step5_config(config):
        return None
    if phase not in ("prefill", "decode"):
        raise ValueError("VLLM_HCU_STEP5_W4A8_PROFILE must be prefill or decode")
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
    if getattr(config, "lora_config", None) is not None:
        raise ValueError("Step5 native W4A8 LoRA has not been enabled")
    return phase


def resolve_profile(moe):
    phase = os.environ.get("VLLM_HCU_STEP5_W4A8_PROFILE", "")
    if not phase or not is_step5_owner(moe):
        return None
    activation = getattr(moe.activation, "value", moe.activation)
    if activation != "swiglustep":
        return None
    if phase not in ("prefill", "decode"):
        raise ValueError("VLLM_HCU_STEP5_W4A8_PROFILE must be prefill or decode")
    parallel = moe.moe_parallel_config
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
