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
        _deepgemm_experts = None

        def _uses_deepgemm(self) -> bool:
            return getattr(self.moe, "moe_backend", "auto") == "deep_gemm"

        def _process_deepgemm_weights(self, layer):
            # naive / all-gather dispatch hands the experts gathered tokens;
            # DeepEP high-throughput only moves tokens and gets its own
            # modular kernel (built below, after the weights are packed).
            parallel = getattr(self.moe, "moe_parallel_config", None)
            if getattr(parallel, "all2all_backend", None) not in (
                None,
                "naive",
                "allgather_reducescatter",
                "deepep_high_throughput",
            ):
                raise ValueError(
                    "Step5 W4A8 --moe-backend deep_gemm supports naive, "
                    "allgather_reducescatter or deepep_high_throughput "
                    "dispatch only"
                )
            if self._deepgemm_experts is not None:
                return
            from vllm_hcu.model_executor.layers.quantization.step5_w4a8_deepgemm import (
                Step5W4A8DeepGemmExperts,
            )

            self._supports_int8_allgatherv_input = False
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            experts = Step5W4A8DeepGemmExperts(self.moe, self.moe_quant_config)
            # Repacks layer.w13/w2_weight in place; HIPC may pad the gate/up
            # scale, so rebuild the config from the registered Parameters.
            experts.process_weights_after_loading(layer)
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            self._deepgemm_experts = experts
            if self._uses_deepep_ht():
                from vllm_hcu.model_executor.layers.quantization.step5_w4a8_deepgemm import (
                    make_step5_marlin_ht_kernel,
                )

                routing_tables = getattr(layer, "_expert_routing_tables", None)
                self.moe_kernel = make_step5_marlin_ht_kernel(
                    experts,
                    self.moe_quant_config,
                    self.moe,
                    routing_tables() if callable(routing_tables) else None,
                )
                fused_experts = getattr(self.moe_kernel, "fused_experts", None)
                getattr(fused_experts, "experts", fused_experts).process_weights_after_loading(
                    layer
                )
            logger.info_once(
                "Step5 CT W4A8: packed INT4 experts on lightop Marlin "
                "W4A8 grouped GEMM"
            )

        def _build_deepep_ht_kernel(self, layer):
            from vllm_hcu.model_executor.layers.fused_moe.experts.step5_aiter_w4a8_moe import (
                make_step5_aiter_w4a8_ht_kernel,
            )

            if self.moe_quant_config is None:
                self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            routing_tables = getattr(layer, "_expert_routing_tables", None)
            self.moe_kernel = make_step5_aiter_w4a8_ht_kernel(
                self,
                self.moe_quant_config,
                self.moe,
                routing_tables() if callable(routing_tables) else None,
            )
            fused_experts = getattr(self.moe_kernel, "fused_experts", None)
            experts = getattr(fused_experts, "experts", fused_experts)
            experts.process_weights_after_loading(layer)

        def _uses_deepep_ht(self) -> bool:
            parallel = getattr(self.moe, "moe_parallel_config", None)
            return getattr(parallel, "all2all_backend", None) == "deepep_high_throughput"

        def _uses_boltops(self) -> bool:
            """Packed INT4 W4A8 through BoltOps instead of the AITER MOE_C GEMMs.

            gfx936 launches the AITER MOE_C W4A8 GEMMs without ever executing them,
            so ``auto`` picks BoltOps there.  VLLM_HCU_STEP5_W4A8_BACKEND=aiter
            restores the old path.  DeepEP keeps AITER (its experts class still
            calls it).
            """

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
            activation = getattr(self.moe.activation, "value", self.moe.activation)
            if activation == "swiglustep" and getattr(layer, "swiglu_limit", None) is None:
                layer.swiglu_limit = 7.0
            if self._uses_deepgemm():
                self._process_deepgemm_weights(layer)
            elif self._uses_boltops():
                self._process_boltops_weights(layer)
                if self._uses_deepep_ht() and self.moe_kernel is None:
                    # DeepEP moves tokens; its experts class then calls BoltOps.
                    self._build_deepep_ht_kernel(layer)
            else:
                super().process_weights_after_loading(layer)
                if self._uses_deepep_ht() and self.moe_kernel is None:
                    # DeepEP moves tokens, so INT8 all-gather input does not apply.
                    self._supports_int8_allgatherv_input = False
                    self._build_deepep_ht_kernel(layer)
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
            if self._boltops and self.moe_kernel is None:
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
            experts = self._deepgemm_experts
            if experts is None or self.moe_kernel is not None:
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
            if i_q is not None:
                raise RuntimeError(
                    "Step5 W4A8 DeepGEMM does not consume INT8 AllGatherv input"
                )
            # Shared experts and DP/TP reductions remain owned by MoERunner.
            del shared_experts, shared_experts_input
            if getattr(layer, "apply_router_weight_on_input", False):
                raise ValueError("Step5 W4A8 DeepGEMM requires output-side routing")
            if x.ndim != 2 or topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
                raise ValueError("Step5 W4A8 DeepGEMM requires rank-2 x and top-k")
            if topk_ids.shape[0] != x.shape[0]:
                raise ValueError("Step5 W4A8 DeepGEMM token/top-k count mismatch")
            if x.shape[0] == 0:
                return torch.empty_like(x)
            from vllm_hcu.model_executor.layers.fused_moe.aiter_moe_dispatch import (
                resolve_aiter_expert_maps,
            )
            from vllm_hcu.model_executor.layers.quantization.step5_w4a8_deepgemm import (
                apply_step5_deepgemm,
            )

            global_num_experts = getattr(
                layer, "global_num_experts", getattr(self.moe, "num_experts", -1)
            )
            expert_map, _ = resolve_aiter_expert_maps(
                getattr(layer, "expert_map", getattr(layer, "_expert_map", None)),
                global_num_experts,
            )
            return apply_step5_deepgemm(
                experts,
                layer,
                x,
                topk_weights,
                topk_ids,
                expert_map,
                global_num_experts,
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
            backend = getattr(parallel, "all2all_backend", None)
            moe_backend = getattr(layer.moe_config, "moe_backend", "auto")
            if backend == "deepep_high_throughput":
                if moe_backend not in ("auto", "deep_gemm"):
                    raise ValueError(
                        "Step5 DeepEP HT requires --moe-backend auto or deep_gemm"
                    )
                method_cls = _step5_method_class()
                return method_cls(quant_config, layer.moe_config)
            if backend == "deepep_low_latency":
                # The core_fix selector owns the Step5 packed INT4 LL kernel,
                # but in worker processes it can be applied after this wrapper.
                from vllm_hcu.model_executor.layers.quantization.ct_w4a8_step5 import (
                    Step5CompressedW4A8MoEMethod,
                )

                return Step5CompressedW4A8MoEMethod(quant_config, layer.moe_config)
            if moe_backend == "deep_gemm":
                method_cls = _step5_method_class()
                return method_cls(quant_config, layer.moe_config)
            from vllm_hcu.model_executor.layers.quantization.ct_w4a8_step5_native_policy import (
                native_enabled,
            )

            if native_enabled():
                # This is the final Step5 selector after the core and generic
                # OP callbacks. Delegating ``original`` would be intercepted
                # by the generic LightOp W4A8 wrapper before reaching core.
                # Select the core adapter explicitly for this opt-in only.
                scheme = quant_config.get_scheme_dict(
                    layer, layer_name + ".0.down_proj"
                )
                weight = scheme["weights"]
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
