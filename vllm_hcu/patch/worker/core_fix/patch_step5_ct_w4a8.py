# SPDX-License-Identifier: Apache-2.0
"""Select the Step5 CT INT4 adapter without editing installed vLLM files.

The adapter keeps experts packed as INT4 and uses the BoltOps W4A8 runtime.
Upstream selection is preserved outside Step5 ROCm W4A8 layers.
"""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe"
)
PATCH_ID = "worker.core_fix.step5.ct_w4a8"
TARGET_SYMBOL = f"{TARGET_MODULE}.CompressedTensorsMoEMethod.get_moe_method"
_MARKER = "_vllm_hcu_step5_ct_w4a8_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    cls = require_class(target, "CompressedTensorsMoEMethod", TARGET_SYMBOL)
    original = require_callable(cls, "get_moe_method", TARGET_SYMBOL)
    if getattr(cls, _MARKER, False):
        if not getattr(original, _MARKER, False):
            raise PatchCompatibilityError(f"stale Step5 W4A8 marker on {TARGET_SYMBOL}")
        return False
    if not isinstance(vars(cls).get("get_moe_method"), staticmethod):
        raise PatchCompatibilityError(f"{TARGET_SYMBOL} must be a staticmethod")
    require_exact_signature(
        original, TARGET_SYMBOL,
        positional=("quant_config", "layer", "layer_name"),
    )
    require_callable(target, "get_current_vllm_config", TARGET_MODULE)
    require_callable(getattr(target, "current_platform", None), "is_rocm", TARGET_MODULE)

    @functools.wraps(original)
    def get_moe_method(quant_config, layer, layer_name):
        if not target.current_platform.is_rocm():
            return original(quant_config, layer, layer_name)
        hf_config = target.get_current_vllm_config().model_config.hf_config
        if getattr(hf_config, "model_type", None) not in ("step5", "step5_vl"):
            return original(quant_config, layer, layer_name)

        # Match upstream's fused-expert lookup, including ignored projections
        # and rejection of inconsistent gate/up/down quantization schemes.
        quant_config._add_fused_moe_to_target_scheme_map()
        schemes = [
            quant_config.get_scheme_dict(layer, layer_name + suffix)
            for suffix in (".0.gate_proj", ".0.up_proj", ".0.down_proj")
        ]
        scheme = schemes[-1]
        if any(value != scheme for value in schemes[:-1]):
            raise ValueError(
                "All MoE projections need to have same "
                "quantization scheme but found multiple"
            )
        if scheme is None:
            return original(quant_config, layer, layer_name)
        weight = scheme.get("weights")
        activation = scheme.get("input_activations")
        if not quant_config._is_dynamic_token_w4a8_int(weight, activation):
            return original(quant_config, layer, layer_name)
        if (
            getattr(weight.strategy, "value", weight.strategy) != "channel"
            or not weight.symmetric
        ):
            raise ValueError(
                "Step5 HCU W4A8 adapter requires symmetric per-channel weights"
            )

        from vllm_hcu.model_executor.layers.quantization.ct_w4a8_step5 import (
            Step5CompressedW4A8MoEMethod,
        )

        return Step5CompressedW4A8MoEMethod(quant_config, layer.moe_config)

    setattr(get_moe_method, _MARKER, True)
    cls.get_moe_method = staticmethod(get_moe_method)
    setattr(cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
