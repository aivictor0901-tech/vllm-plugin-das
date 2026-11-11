from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import QuantizationStrategy, QuantizationType

from vllm_hcu.patch.worker.op_opt.patch_step5_compressed_tensors_w4a8 import (
    TARGET_MODULE,
    _step5_method_class,
    apply_to_module,
    compressed_tensors_w4a8_to_slimquant,
)


def _unpack_slimquant(packed):
    values = packed.view(torch.uint8)
    nibbles = torch.stack(
        ((values >> 4) & 0xF, values & 0xF),
        dim=-1,
    ).flatten(-2)
    return torch.where(nibbles >= 8, nibbles - 16, nibbles).to(torch.int8)


def test_compressed_tensors_w4a8_conversion_round_trips_signed_int4():
    logical = torch.tensor(
        [
            [-8, -7, -1, 0, 1, 6, 7, -4, 3, 5, -2, 4, 2, -3, 0, 7],
            [7, 0, -8, 1, -1, -7, 6, -2, 4, -4, 5, -3, 3, 2, 0, 1],
        ],
        dtype=torch.int8,
    )
    checkpoint = pack_to_int32(logical, num_bits=4)

    hcu_packed = compressed_tensors_w4a8_to_slimquant(checkpoint)

    assert hcu_packed.shape == (2, 8)
    assert hcu_packed.dtype is torch.int8
    torch.testing.assert_close(_unpack_slimquant(hcu_packed), logical)


def test_factory_routes_only_step5_channel_w4a8(monkeypatch):
    import vllm.config
    from vllm.platforms import current_platform

    step5_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="step5"))
    )
    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config_or_none",
        lambda: step5_config,
    )
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)

    scheme = {
        "format": CompressionFormat.pack_quantized.value,
        "weights": SimpleNamespace(
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.CHANNEL,
        ),
        "input_activations": SimpleNamespace(
            strategy=QuantizationStrategy.TOKEN,
        ),
    }

    class QuantConfig:
        def _add_fused_moe_to_target_scheme_map(self):
            pass

        def get_scheme_dict(self, layer, name):
            assert name.endswith((".0.gate_proj", ".0.up_proj", ".0.down_proj"))
            return scheme

        def _is_dynamic_token_w4a8_int(self, weight, activation):
            return (
                weight is scheme["weights"]
                and activation is scheme["input_activations"]
            )

    class CompressedTensorsMoEMethod:
        @staticmethod
        def get_moe_method(quant_config, layer, layer_name):
            return "original"

    module = ModuleType(TARGET_MODULE)
    module.CompressedTensorsMoEMethod = CompressedTensorsMoEMethod
    assert apply_to_module(module)

    layer = SimpleNamespace(moe_config=SimpleNamespace(has_bias=False, tp_size=1))
    selected = CompressedTensorsMoEMethod.get_moe_method(
        QuantConfig(), layer, "model.layers.0.moe.experts"
    )
    assert isinstance(selected, _step5_method_class())

    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="llama"))
        ),
    )
    assert (
        CompressedTensorsMoEMethod.get_moe_method(
            QuantConfig(), layer, "model.layers.0.moe.experts"
        )
        == "original"
    )


def test_step5_w4a8_loader_converts_weights_and_validates_shape_metadata():
    calls = []

    def weight_loader(param, loaded, *args, **kwargs):
        calls.append((param, loaded, args, kwargs))
        return kwargs.get("return_success", False)

    method_class = _step5_method_class()
    method = method_class(object(), type("Moe", (), {"tp_size": 1})())
    layer = nn.Module()
    method.create_weights(
        layer,
        num_experts=1,
        hidden_size=16,
        intermediate_size_per_partition=8,
        params_dtype=torch.float16,
        weight_loader=weight_loader,
    )
    logical = torch.tensor(
        [[-8, -1, 0, 7, 1, 2, 3, -4] * 2],
        dtype=torch.int8,
    )
    checkpoint = pack_to_int32(logical, num_bits=4)

    loaded = layer.w13_weight_packed.weight_loader(
        layer.w13_weight_packed,
        checkpoint,
        "w13_weight_packed",
        shard_id="w1",
        expert_id=0,
        return_success=True,
    )

    assert loaded is True
    assert calls[-1][0] is layer.w13_weight
    torch.testing.assert_close(_unpack_slimquant(calls[-1][1]), logical)
    assert calls[-1][2] == ("w13_weight_packed",)
    assert calls[-1][3] == {
        "shard_id": "w1",
        "expert_id": 0,
        "return_success": True,
    }

    shape_loaded = layer.w13_weight_shape.weight_loader(
        layer.w13_weight_shape,
        torch.tensor([8, 16], dtype=torch.int64),
        "w13_weight_shape",
        shard_id="w1",
        expert_id=0,
        return_success=True,
    )
    assert shape_loaded is True

    with pytest.raises(ValueError, match="checkpoint shape mismatch"):
        layer.w13_weight_shape.weight_loader(
            layer.w13_weight_shape,
            torch.tensor([9, 16], dtype=torch.int64),
            "w13_weight_shape",
            shard_id="w1",
            expert_id=0,
            return_success=True,
        )


def test_step5_expert_placement_remaps_per_expert_checkpoint_names(tmp_path):
    import os
    import subprocess
    import sys

    placement = tmp_path / "placement.pt"
    torch.save(
        {"physical_to_logical_map": torch.tensor([[2, 0, 1]], dtype=torch.int64)},
        placement,
    )
    # step5_text imports lazy-weight helpers from vLLM's linear module, which the
    # plugin replaces at registration; load it the way the worker does.
    script = """
import sys
import torch
from vllm_hcu import hcu_platform_register_model
hcu_platform_register_model()
from vllm_hcu.models import step5_text

step5_text.EXPERT_PLACEMENT = sys.argv[1]
tensors = [
    (f"model.layers.0.moe.experts.{expert}.gate_proj.weight_packed", tensor)
    for expert, tensor in enumerate(
        (torch.tensor([0]), torch.tensor([1]), torch.tensor([2]))
    )
]
result = list(step5_text._place_experts(tensors, [0]))
assert [name for name, _ in result] == [
    "model.layers.0.moe.experts.1.gate_proj.weight_packed",
    "model.layers.0.moe.experts.2.gate_proj.weight_packed",
    "model.layers.0.moe.experts.0.gate_proj.weight_packed",
], result
assert [int(tensor.item()) for _, tensor in result] == [0, 1, 2]
"""
    env = os.environ.copy()
    env.setdefault("VLLM_HCU_USE_CUSTOM_OPS", "1")
    result = subprocess.run(
        [sys.executable, "-c", script, str(placement)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
