# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Select the HCU compressed-tensors W4A8 MoE without changing other schemes."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe"
)
PATCH_ID = "worker.op_opt.compressed_tensors.moe_w4a8"
TARGETS = (f"{TARGET_MODULE}.CompressedTensorsMoEMethod.get_moe_method",)
_CLASS_MARKER = "_vllm_hcu_moe_w4a8_applied"
_WRAPPER_MARKER = "_vllm_hcu_moe_w4a8_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    module = load_exact_module(TARGET_MODULE, module)
    cls = require_class(module, "CompressedTensorsMoEMethod", TARGETS[0])
    if already_applied(
        cls, _CLASS_MARKER, ((cls, "get_moe_method", TARGETS[0], _WRAPPER_MARKER),)
    ):
        return False
    original = require_callable(cls, "get_moe_method", TARGETS[0])
    require_exact_signature(
        original, TARGETS[0], positional=("quant_config", "layer", "layer_name")
    )

    @functools.wraps(original)
    def get_moe_method(quant_config, layer, layer_name):
        quant_config._add_fused_moe_to_target_scheme_map()
        schemes = [
            quant_config.get_scheme_dict(layer, layer_name + f".0.{proj}_proj")
            for proj in ("gate", "up", "down")
        ]
        scheme = schemes[0]
        if scheme is not None:
            weight = scheme.get("weights")
            activation = scheme.get("input_activations")
            if (
                weight is not None and activation is not None
                and weight.type == "int" and activation.type == "int"
                and weight.num_bits == 4 and activation.num_bits == 8
            ):
                if not all(s == scheme for s in schemes):
                    raise ValueError("All MoE projections must use the same W4A8 scheme")
                if scheme.get("format") != "pack-quantized":
                    raise ValueError("HCU W4A8 requires pack-quantized weights")
                from vllm_hcu.model_executor.layers.quantization.compressed_tensors_w4a8 import (
                    HCUCompressedTensorsW4A8MoEMethod,
                )

                return HCUCompressedTensorsW4A8MoEMethod(
                    weight, activation, layer.moe_config
                )
        return original(quant_config, layer, layer_name)

    setattr(get_moe_method, _WRAPPER_MARKER, True)
    cls.get_moe_method = staticmethod(get_moe_method)
    setattr(cls, _CLASS_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
