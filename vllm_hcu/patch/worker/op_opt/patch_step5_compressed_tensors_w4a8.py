# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route Step5 compressed-tensors W4A8 experts to the HCU SlimQuant method."""

from __future__ import annotations

import functools
import os
from types import ModuleType

import torch
from torch.nn import Parameter
from vllm.logger import init_logger

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

logger = init_logger(__name__)

TARGET_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe"
)
PATCH_ID = "worker.op_opt.step5.compressed_tensors.moe_w4a8"
TARGETS = (f"{TARGET_MODULE}.CompressedTensorsMoEMethod.get_moe_method",)
_CLASS_MARKER = "_vllm_hcu_step5_w4a8_applied"
_WRAPPER_MARKER = "_vllm_hcu_step5_w4a8_wrapper"


def compressed_tensors_w4a8_to_slimquant(packed: torch.Tensor) -> torch.Tensor:
    """Convert low-nibble-first offset INT4 in INT32 words to HCU signed bytes."""
    if packed.dtype is not torch.int32:
        raise TypeError(f"expected packed INT32 W4A8 weights, got {packed.dtype}")
    if packed.ndim != 2:
        raise ValueError(f"expected rank-2 packed weights, got {tuple(packed.shape)}")

    # Reuse the packed loader's byte-wise conversion. Expanding every nibble
    # to INT32 creates an 8x temporary and costly CPU OpenMP work per expert.
    # Both paths consume the same signed, high-nibble-first W4A8 byte layout.
    from vllm_hcu.model_executor.layers.quantization.ct_w4a8_step5 import (
        ct_int32_to_packed_int4,
    )

    return ct_int32_to_packed_int4(packed)


@functools.lru_cache(maxsize=1)
def _step5_method_class():
    from vllm.model_executor.utils import set_weight_attrs
    from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import (
        SlimQuantW4A8Int8AiterMoEMethod,
    )

    class Step5CompressedTensorsW4A8Method(SlimQuantW4A8Int8AiterMoEMethod):
        def create_weights(
            self,
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        ):
            super().create_weights(
                layer,
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                params_dtype,
                **extra_weight_attrs,
            )

            def register_checkpoint_alias(name, loader):
                parameter = Parameter(
                    torch.empty(0, dtype=torch.int32),
                    requires_grad=False,
                )
                layer.register_parameter(name, parameter)
                set_weight_attrs(parameter, extra_weight_attrs)
                parameter.weight_loader = loader

            def packed_loader(target_name):
                def load(param, loaded_weight, *args, **kwargs):
                    del param
                    converted = compressed_tensors_w4a8_to_slimquant(loaded_weight)
                    target = getattr(layer, target_name)
                    target_loader = getattr(target, "weight_loader", None)
                    if not callable(target_loader):
                        raise RuntimeError(
                            f"Step5 W4A8 target {target_name} has no weight loader"
                        )
                    result = target_loader(target, converted, *args, **kwargs)
                    return result if type(result) is bool else True

                return load

            def shape_loader(expected_shape):
                def load(param, loaded_shape, *args, **kwargs):
                    del param, args
                    if loaded_shape.dtype not in (torch.int32, torch.int64):
                        raise TypeError(
                            "compressed-tensors W4A8 weight_shape must be integer"
                        )
                    actual_shape = tuple(int(value) for value in loaded_shape.tolist())
                    if actual_shape != expected_shape:
                        raise ValueError(
                            "Step5 W4A8 checkpoint shape mismatch: "
                            f"expected {expected_shape}, got {actual_shape}"
                        )
                    success = kwargs.get("return_success", False)
                    return True if success else None

                return load

            register_checkpoint_alias(
                "w13_weight_packed",
                packed_loader("w13_weight"),
            )
            register_checkpoint_alias(
                "w2_weight_packed",
                packed_loader("w2_weight"),
            )
            global_intermediate_size = (
                intermediate_size_per_partition * self.moe.tp_size
            )
            register_checkpoint_alias(
                "w13_weight_shape",
                shape_loader((global_intermediate_size, hidden_size)),
            )
            register_checkpoint_alias(
                "w2_weight_shape",
                shape_loader((hidden_size, global_intermediate_size)),
            )

        _boltops = False

        def _uses_boltops(self) -> bool:
            """Packed INT4 W4A8 through BoltOps instead of the AITER MOE_C GEMMs.

            gfx936 launches the AITER MOE_C W4A8 GEMMs without ever executing them,
            so ``auto`` picks BoltOps there.  VLLM_HCU_STEP5_W4A8_BACKEND=aiter
            restores the old path.  DeepEP keeps AITER (its experts class still
            calls it).
            """

            uses_deepep_ht = getattr(self, "_uses_deepep_ht", None)
            if callable(uses_deepep_ht) and uses_deepep_ht():
                return False
            choice = os.environ.get("VLLM_HCU_STEP5_W4A8_BACKEND", "auto").strip().lower()
            if choice == "aiter":
                return False
            if choice == "boltops":
                return True
            if choice != "auto":
                raise ValueError(
                    "VLLM_HCU_STEP5_W4A8_BACKEND must be auto, boltops or aiter, "
                    f"got {choice!r}"
                )
            arch = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).gcnArchName.split(":")[0]
            return arch == "gfx936"

        def _process_boltops_weights(self, layer):
            from vllm_hcu.model_executor.layers.quantization.compressed_tensors_moe_runtime import (
                prepare_step5_boltops_w4a8_moe,
            )

            prepare_step5_boltops_w4a8_moe(self, layer)
            self._boltops = True
            logger.info_once(
                "Step5 CT W4A8: packed INT4 experts on BoltOps (no AITER MOE_C, no INT8 expansion)"
            )

        def process_weights_after_loading(self, layer):
            if self._uses_boltops():
                self._process_boltops_weights(layer)
            else:
                super().process_weights_after_loading(layer)
            for name in (
                "w13_weight_packed",
                "w2_weight_packed",
                "w13_weight_shape",
                "w2_weight_shape",
            ):
                layer.register_parameter(name, None)

        def apply(
            self,
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts,
            shared_experts_input,
            i_q=None,
            i_s=None,
            output_dtype=None,
            **kwargs,
        ):
            if self._boltops:
                if i_q is not None:
                    raise RuntimeError(
                        "Step5 W4A8 BoltOps does not consume INT8 AllGatherv input"
                    )
                # Shared experts and DP/TP reductions remain owned by MoERunner.
                del shared_experts, shared_experts_input
                from vllm_hcu.model_executor.layers.quantization.compressed_tensors_moe_runtime import (
                    apply_step5_boltops_w4a8_moe,
                )

                return apply_step5_boltops_w4a8_moe(
                    self, layer, x, topk_weights, topk_ids
                )
            return super().apply(
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts,
                shared_experts_input,
                i_q=i_q,
                i_s=i_s,
                output_dtype=output_dtype,
                **kwargs,
            )

    return Step5CompressedTensorsW4A8Method


def _is_step5_channel_w4a8(quant_config, layer, layer_name) -> bool:
    from compressed_tensors import CompressionFormat
    from compressed_tensors.quantization import (
        QuantizationStrategy,
        QuantizationType,
    )
    from vllm.config import get_current_vllm_config_or_none
    from vllm.platforms import current_platform
    from vllm_hcu.runtime_compat.step5 import is_step5_config

    config = get_current_vllm_config_or_none()
    if (
        not current_platform.is_rocm()
        or not is_step5_config(config)
        or getattr(layer.moe_config, "has_bias", False)
    ):
        return False

    quant_config._add_fused_moe_to_target_scheme_map()
    unfused_names = [
        layer_name + suffix
        for suffix in (".0.gate_proj", ".0.up_proj", ".0.down_proj")
    ]
    schemes = [quant_config.get_scheme_dict(layer, name) for name in unfused_names]
    scheme = schemes[-1]
    if scheme is None or any(item != scheme for item in schemes[:-1]):
        return False

    weight_quant = scheme.get("weights")
    input_quant = scheme.get("input_activations")
    if (
        scheme.get("format") != CompressionFormat.pack_quantized.value
        or weight_quant is None
        or input_quant is None
        or weight_quant.type != QuantizationType.INT
        or weight_quant.strategy != QuantizationStrategy.CHANNEL
        or input_quant.strategy != QuantizationStrategy.TOKEN
        or not quant_config._is_dynamic_token_w4a8_int(weight_quant, input_quant)
    ):
        return False
    return True


def apply_to_module(module: ModuleType) -> bool:
    target_module = load_exact_module(TARGET_MODULE, module)
    method_class = require_class(
        target_module,
        "CompressedTensorsMoEMethod",
        f"{TARGET_MODULE}.CompressedTensorsMoEMethod",
    )
    wrapped = (
        (
            method_class,
            "get_moe_method",
            TARGETS[0],
            _WRAPPER_MARKER,
        ),
    )
    if already_applied(method_class, _CLASS_MARKER, wrapped):
        return False

    original = require_callable(method_class, "get_moe_method", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("quant_config", "layer", "layer_name"),
    )

    @functools.wraps(original)
    def hcu_get_moe_method(quant_config, layer, layer_name):
        if _is_step5_channel_w4a8(quant_config, layer, layer_name):
            parallel = getattr(layer.moe_config, "moe_parallel_config", None)
            if getattr(parallel, "all2all_backend", None) == "deepep_low_latency":
                # The core_fix selector owns the Step5 packed INT4 LL kernel.
                # The legacy BoltOps path leaves moe_kernel unset and sends
                # MoERunner through unsupported dispatch_router_logits.
                return original(quant_config, layer, layer_name)
            method_cls = _step5_method_class()
            return method_cls(quant_config, layer.moe_config)
        return original(quant_config, layer, layer_name)

    setattr(hcu_get_moe_method, _WRAPPER_MARKER, True)
    setattr(method_class, "_vllm_hcu_original_get_moe_method", original)
    setattr(method_class, "get_moe_method", staticmethod(hcu_get_moe_method))
    setattr(method_class, _CLASS_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
    "compressed_tensors_w4a8_to_slimquant",
]
