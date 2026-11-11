# SPDX-License-Identifier: Apache-2.0
"""Step5-only load adapter from compressed-tensors GPTQ to native W4A8.

Checkpoint INT32 words hold eight offset-binary nibbles, low nibble first.
The existing backend consumes signed high-nibble-first bytes and scales for
the INT8 (INT4 * 16) domain. Conversion is lossless, local-expert-only, and
done before the existing in-place HIPC packer. No checkpoint is rewritten.
"""

import re
import sys
import os

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts, UnquantizedFusedMoEMethod
from vllm_hcu.model_executor.layers.fused_moe.step5_w4a8_policy import (
    compressed_w4a8_config,
    profile_for_model,
    uses_step5_deepep,
    validate_gptq_w4a8_scheme,
)
from .slimquant_w4a8 import SlimQuantW4A8Int8AiterMoEMethod, SlimQuantW4A8Int8Config
from compressed_tensors.quantization import QuantizationArgs
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8 import CompressedTensorsW8A8Fp8MoEMethod


def repack_gptq_int4(weight):
    if weight.dtype != torch.int32 or weight.ndim != 2:
        raise ValueError("Step5 GPTQ weights must be rank-2 INT32 packed along K")
    if sys.byteorder != "little":
        raise RuntimeError("Step5 GPTQ byte conversion requires little-endian storage")
    raw = weight.contiguous().view(torch.uint8)
    swapped = ((raw & 15) << 4) | (raw >> 4)
    return torch.bitwise_xor(swapped, 0x88).view(torch.int8)


def quantize_bf16_expert(weight, device):
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("Step5 BF16-to-FP8 expects one rank-2 BF16 expert projection")
    x = weight.to(device=device, dtype=torch.float32)
    scale = (x.abs().amax(dim=-1, keepdim=True) / 448.0).clamp_min(1e-12)
    quantized = (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized, scale


class Step5BF16ToFP8MoEMethod(CompressedTensorsW8A8Fp8MoEMethod):
    """Reuse the existing channel-FP8 runtime with local load-time conversion."""
    def __init__(self, moe, layer_name):
        # Scope the FP8 oracle exception to this mixed-checkpoint adapter.
        # Other Channel-FP8 models retain their existing backend validation.
        moe._hcu_step5_deepep_mixed_fp8 = uses_step5_deepep(moe)
        weight = QuantizationArgs(num_bits=8, type="float", strategy="channel",
                                  symmetric=True, dynamic=False)
        activation = QuantizationArgs(num_bits=8, type="float", strategy="token",
                                      symmetric=True, dynamic=True)
        super().__init__(weight, activation, moe, layer_name=layer_name)

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **attrs):
        if not uses_step5_deepep(self.moe) and (num_experts, hidden_size, intermediate_size_per_partition) != (44, 4096, 1536):
            raise ValueError("Step5 mixed FP8 requires 44 complete experts per EP8 rank")
        super().create_weights(layer, num_experts, hidden_size,
                               intermediate_size_per_partition, params_dtype, **attrs)
        self._bf16_loaded = set()
        self._bf16_expected = num_experts * 3
        for name in ("w13_weight", "w2_weight"):
            parameter = getattr(layer, name)
            original = parameter.weight_loader

            def load(param, tensor, weight_name, shard_id, expert_id,
                     return_success=False, *, _name=name, _original=original):
                local = layer._map_global_expert_id_to_local_expert_id(expert_id)
                if local == -1:
                    return False if return_success else None
                full_intermediate = intermediate_size_per_partition * self.moe.moe_parallel_config.tp_size
                expected = ((hidden_size, full_intermediate)
                            if shard_id == "w2"
                            else (full_intermediate, hidden_size))
                if tuple(tensor.shape) != expected:
                    raise ValueError(f"Unexpected BF16 expert projection shape: {tuple(tensor.shape)}")
                quantized, scale = quantize_bf16_expert(tensor, param.device)
                success = _original(param, quantized, weight_name, shard_id=shard_id,
                                    expert_id=expert_id, return_success=True)
                scale_param = getattr(layer, _name + "_scale")
                scale_success = scale_param.weight_loader(
                    scale_param, scale, weight_name + "_scale", shard_id=shard_id,
                    expert_id=expert_id, return_success=True)
                if success and scale_success:
                    self._bf16_loaded.add((shard_id, local))
                return success and scale_success if return_success else None

            parameter.weight_loader = load

    def process_weights_after_loading(self, layer):
        if len(self._bf16_loaded) != self._bf16_expected:
            raise RuntimeError(f"Incomplete BF16-to-FP8 expert load: {len(self._bf16_loaded)}/{self._bf16_expected}")
        if uses_step5_deepep(self.moe):
            from vllm_hcu.model_executor.layers.fused_moe.step5_deepep import prepare_step5_deepep
            prepare_step5_deepep(self, layer, int4=False)
        else:
            from vllm_hcu.model_executor.layers.fused_moe.step5_fp8_agrs import prepare_step5_fp8_agrs
            prepare_step5_fp8_agrs(self, layer)


class Step5GPTQW4A8MoEMethod(SlimQuantW4A8Int8AiterMoEMethod):
    @property
    def supports_int8_allgatherv_input(self):
        if uses_step5_deepep(self.moe):
            return False
        return super().supports_int8_allgatherv_input

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **attrs):
        from vllm_hcu.ops.step5.w4a8 import gemm_backend, require_gemm_op

        if gemm_backend() == "deepgemm" or uses_step5_deepep(self.moe):
            # Fail before reading the 335 GiB checkpoint when an incompatible
            # DeepGEMM wheel is installed.
            require_gemm_op("deepgemm")
        if not uses_step5_deepep(self.moe) and (num_experts, hidden_size, intermediate_size_per_partition) != (44, 4096, 1536):
            raise ValueError("Step5 GPTQ requires 44 complete experts per EP8 rank")
        super().create_weights(layer, num_experts, hidden_size,
                               intermediate_size_per_partition, params_dtype, **attrs)
        self._gptq_loaded = set()
        self._gptq_expected = num_experts * 6
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            parameter = getattr(layer, name)
            original = parameter.weight_loader

            def load(param, tensor, weight_name, shard_id, expert_id,
                     return_success=False, *, _name=name, _original=original):
                local = layer._map_global_expert_id_to_local_expert_id(expert_id)
                if local == -1:
                    return False if return_success else None
                if "scale" in _name:
                    full_intermediate = intermediate_size_per_partition * self.moe.moe_parallel_config.tp_size
                    rows = hidden_size if shard_id == "w2" else full_intermediate
                    if tensor.dtype != torch.float32 or tuple(tensor.shape) != (rows, 1):
                        raise ValueError("Step5 GPTQ requires FP32 per-output-channel scales")
                    converted = tensor * (1.0 / 16.0)
                else:
                    full_intermediate = intermediate_size_per_partition * self.moe.moe_parallel_config.tp_size
                    shape = ((hidden_size, full_intermediate // 8)
                             if shard_id == "w2"
                             else (full_intermediate, hidden_size // 8))
                    if tuple(tensor.shape) != shape:
                        raise ValueError(f"Unexpected Step5 GPTQ packed weight shape: {tuple(tensor.shape)}")
                    converted = repack_gptq_int4(tensor)
                success = _original(param, converted, weight_name, shard_id=shard_id,
                                    expert_id=expert_id, return_success=True)
                if success:
                    self._gptq_loaded.add((_name, shard_id, local))
                return success if return_success else None

            parameter.weight_loader = load

    def process_weights_after_loading(self, layer):
        if len(self._gptq_loaded) != self._gptq_expected:
            raise RuntimeError(f"Incomplete Step5 GPTQ expert load: {len(self._gptq_loaded)}/{self._gptq_expected}")
        super().process_weights_after_loading(layer)


class Step5GPTQW4A8Config(SlimQuantW4A8Int8Config):
    """A scoped MoE config; the model's resolved quantization stays unchanged."""
    def __init__(self, source):
        super().__init__()
        self.source = source

    @classmethod
    def get_name(cls):
        return "compressed-tensors"

    def get_quant_method(self, layer, prefix):
        if not isinstance(layer, RoutedExperts):
            return self.source.get_quant_method(layer, prefix)
        prefix = prefix.removesuffix(".routed_experts")
        schemes = [self.source.get_scheme_dict(layer, prefix + suffix)
                   for suffix in (".0.gate_proj", ".0.up_proj", ".0.down_proj")]
        if not all(s == schemes[0] for s in schemes):
            raise ValueError("Step5 GPTQ gate/up/down quantization schemes must agree")
        if schemes[0] is None:
            if os.environ.get("VLLM_HCU_STEP5_FP8_BF16_MOE", "0") == "1":
                layer._hcu_step5_gptq_kind = "fp8"
                return Step5BF16ToFP8MoEMethod(layer.moe_config, prefix)
            layer._hcu_step5_gptq_kind = "bf16"
            return UnquantizedFusedMoEMethod(layer.moe_config)
        validate_gptq_w4a8_scheme(schemes[0])
        layer._hcu_step5_gptq_kind = "w4a8"
        return Step5GPTQW4A8MoEMethod(self, layer.moe_config)


def moe_quant_config(vllm_config):
    source = vllm_config.quant_config
    if (not source or source.get_name() != "compressed-tensors"
            or not compressed_w4a8_config(vllm_config)):
        return source
    if profile_for_model(vllm_config) not in ("prefill", "decode", "auto"):
        raise ValueError("Set VLLM_HCU_STEP5_W4A8_PROFILE=prefill, decode or auto")
    adapter = getattr(source, "_hcu_step5_gptq_adapter", None)
    if adapter is None:
        adapter = source._hcu_step5_gptq_adapter = Step5GPTQW4A8Config(source)
    return adapter


def checkpoint_names(weights, quantized_layers):
    """Rename packed payloads and validate/drop redundant shape metadata."""
    pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.moe\.experts\.(\d+)\."
                         r"(gate_proj|up_proj|down_proj)\.(weight_packed|weight_scale|weight_shape)$")
    for name, tensor in weights:
        match = pattern.search(name)
        if match:
            layer, expert, projection, suffix = match.groups()
            if int(layer) not in quantized_layers or not 0 <= int(expert) < 352:
                raise ValueError(f"GPTQ tensor does not belong to a declared quantized expert: {name}")
            if suffix == "weight_shape":
                expected = (4096, 1536) if projection == "down_proj" else (1536, 4096)
                if tensor.dtype != torch.int64 or tuple(tensor.shape) != (2,) or tuple(tensor.tolist()) != expected:
                    raise ValueError(f"Invalid Step5 GPTQ shape metadata: {name}")
                continue
            if suffix == "weight_packed":
                name = name.removesuffix("_packed")
        yield name, tensor
