"""Storage correctness and scoped mixed-precision selection; CPU only."""
import copy
from types import SimpleNamespace as NS

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32, unpack_from_int32
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
from vllm_hcu.model_executor.layers.fused_moe.step5_w4a8_policy import profile_for_model, validate_gptq_w4a8_scheme
from vllm_hcu.model_executor.layers.quantization import step5_gptq_w4a8 as runtime


def scheme():
    return {"format": "pack-quantized",
            "weights": {"num_bits": 4, "type": "int", "strategy": "channel", "symmetric": True, "dynamic": False},
            "input_activations": {"num_bits": 8, "type": "int", "strategy": "token", "symmetric": True, "dynamic": True}}


def test_packed_encoding_and_scale_preserve_dequantized_weights():
    values = torch.arange(-8, 8, dtype=torch.int8).repeat(3, 4)
    packed = pack_to_int32(values, 4)
    original = packed.clone()
    converted = runtime.repack_gptq_int4(packed)
    # Independent reference: compressed-tensors's unpacker versus signed,
    # high-nibble-first decoding of the runtime payload.
    decoded = unpack_from_int32(packed, 4, values.shape)
    raw = converted.view(torch.uint8).to(torch.int16)
    nibbles = torch.stack((raw >> 4, raw & 15), dim=-1).flatten(-2)
    signed = torch.where(nibbles >= 8, nibbles - 16, nibbles).to(torch.int8)
    torch.testing.assert_close(signed, decoded, rtol=0, atol=0)
    scale = torch.tensor([[.25], [.5], [1.]])
    torch.testing.assert_close(signed.float() * ((scale / 16) * 16), decoded.float() * scale, rtol=0, atol=0)
    torch.testing.assert_close(packed, original, rtol=0, atol=0)


@pytest.mark.parametrize("section,key,value", [
    ("weights", "num_bits", 8), ("weights", "symmetric", False),
    ("weights", "strategy", "group"), ("weights", "group_size", 128),
    ("weights", "actorder", "group"), ("input_activations", "dynamic", False),
    ("input_activations", "type", "float"), ("input_activations", "strategy", "tensor")])
def test_incompatible_formats_are_rejected(section, key, value):
    value_scheme = scheme(); value_scheme[section][key] = value
    with pytest.raises(ValueError):
        validate_gptq_w4a8_scheme(value_scheme)


@pytest.mark.parametrize("convert", [False, True])
@pytest.mark.parametrize("phase,tp,dp", [("decode", 4, 2), ("prefill", 1, 8)])
def test_profile_preserves_compressed_tensors_and_requires_topology(monkeypatch, convert, phase, tp, dp):
    monkeypatch.setenv("VLLM_HCU_STEP5_W4A8_PROFILE", phase)
    monkeypatch.setenv("VLLM_HCU_STEP5_FP8_BF16_MOE", str(int(convert)))
    raw = {"quant_method": "compressed-tensors", "config_groups": {"experts": scheme()}}
    cfg = NS(model_config=NS(hf_config=NS(model_type="step5", quantization_config=raw), quantization="compressed-tensors"),
             parallel_config=NS(tensor_parallel_size=tp, data_parallel_size=dp, pipeline_parallel_size=1,
                                enable_expert_parallel=True, enable_eplb=False), lora_config=None)
    assert profile_for_model(cfg) == phase
    assert cfg.model_config.quantization == "compressed-tensors"
    cfg.parallel_config.tensor_parallel_size = 4 if tp == 1 else 1
    with pytest.raises(ValueError):
        profile_for_model(cfg)


@pytest.mark.parametrize("phase,tp,dp", [("decode", 4, 2), ("prefill", 1, 8)])
def test_moe_quant_config_selects_phase_and_reuses_adapter(monkeypatch, phase, tp, dp):
    monkeypatch.setenv("VLLM_HCU_STEP5_W4A8_PROFILE", phase)
    raw = {"quant_method": "compressed-tensors", "format": "pack-quantized",
           "config_groups": {"experts": dict(scheme(), targets=["Linear"])}}
    source = CompressedTensorsConfig.from_config(copy.deepcopy(raw))
    cfg = NS(quant_config=source,
             model_config=NS(hf_config=NS(model_type="step5", quantization_config=raw),
                             quantization="compressed-tensors"),
             parallel_config=NS(tensor_parallel_size=tp, data_parallel_size=dp,
                                pipeline_parallel_size=1, enable_expert_parallel=True,
                                enable_eplb=False), lora_config=None)
    adapter = runtime.moe_quant_config(cfg)
    assert isinstance(adapter, runtime.Step5GPTQW4A8Config)
    assert adapter.source is source
    assert runtime.moe_quant_config(cfg) is adapter
    assert cfg.quant_config is source
    assert cfg.model_config.quantization == "compressed-tensors"


@pytest.mark.parametrize("convert", [False, True])
@pytest.mark.parametrize("phase,dp", [("decode", 2), ("prefill", 8)])
def test_ignore_regex_selects_bf16_or_fp8_without_changing_int4_targets(monkeypatch, convert, phase, dp):
    monkeypatch.setenv("VLLM_HCU_STEP5_W4A8_PROFILE", phase)
    monkeypatch.setenv("VLLM_HCU_STEP5_FP8_BF16_MOE", str(int(convert)))
    raw = {"format": "pack-quantized", "quant_method": "compressed-tensors",
           "ignore": [r"re:.*layers\.(88|89|90)\.moe\.experts.*"],
           "config_groups": {"experts": dict(scheme(), targets=[r"re:.*layers\.\d+\.moe\.experts(?:\.\d+\.(?:gate_proj|up_proj|down_proj))?$"])}}
    ct = CompressedTensorsConfig.from_config(copy.deepcopy(raw))
    original_ignore = list(ct.ignore)
    wrapped = runtime.Step5GPTQW4A8Config(ct)
    layer = object.__new__(RoutedExperts); torch.nn.Module.__init__(layer)
    layer.moe_config = NS(_hcu_step5_owner=True, activation="swiglustep",
                          moe_parallel_config=NS(enable_eplb=False, all2all_backend="allgather_reducescatter", ep_size=8, dp_size=dp))
    sentinel = object()
    bf16_sentinel = object()
    monkeypatch.setattr(runtime, "Step5BF16ToFP8MoEMethod", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(runtime, "UnquantizedFusedMoEMethod", lambda *args, **kwargs: bf16_sentinel)
    assert isinstance(wrapped.get_quant_method(layer, "model.layers.3.moe.experts.routed_experts"), runtime.Step5GPTQW4A8MoEMethod)
    method = wrapped.get_quant_method(layer, "model.layers.88.moe.experts")
    assert (method is sentinel) == convert
    assert (method is bf16_sentinel) == (not convert)
    assert layer._hcu_step5_gptq_kind == ("fp8" if convert else "bf16")
    assert ct.ignore == original_ignore


def test_checkpoint_names_validate_shape_metadata_and_leave_bf16_payloads():
    packed = torch.zeros(1536, 512, dtype=torch.int32)
    bf16 = torch.zeros(1, dtype=torch.bfloat16)
    prefix = "layers.3.moe.experts.0.gate_proj."
    source = [(prefix + "weight_packed", packed), (prefix + "weight_shape", torch.tensor([1536, 4096])),
              ("layers.88.moe.experts.0.gate_proj.weight", bf16)]
    result = list(runtime.checkpoint_names(iter(source), {3}))
    assert result[0][0] == prefix + "weight" and result[0][1] is packed
    assert result[1][1] is bf16
    with pytest.raises(ValueError):
        list(runtime.checkpoint_names([(prefix + "weight_shape", torch.tensor([1536, 4095]))], {3}))
    with pytest.raises(ValueError):
        list(runtime.checkpoint_names([(prefix + "weight_packed", packed)], {4}))


def test_bf16_to_fp8_matches_existing_channel_quantization_and_handles_zero():
    weight = torch.tensor([[0., 0., 0., 0.], [-5., -.1, .25, 8.]], dtype=torch.bfloat16)
    original = weight.clone()
    q, scale = runtime.quantize_bf16_expert(weight, "cpu")
    expected_scale = (weight.float().abs().amax(-1, keepdim=True) / 448.).clamp_min(1e-12)
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(q.float(), (weight.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float(), rtol=0, atol=0)
    torch.testing.assert_close(q.float() * scale, weight.float(), rtol=.065, atol=.001)
    torch.testing.assert_close(weight, original, rtol=0, atol=0)


def test_per_expert_placement_keeps_router_payloads_and_scales_in_one_order(tmp_path):
    import ast
    import re
    from pathlib import Path
    from collections.abc import Iterable
    source = Path(runtime.__file__).parents[3] / 'models/step5_text.py'
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if
             isinstance(n, ast.FunctionDef) and n.name == '_place_experts'
             or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_PLACED_TENSOR' for t in n.targets)]
    placement = tmp_path / 'placement.pt'
    order = torch.tensor([[2, 0, 3, 1]])
    torch.save({'physical_to_logical_map': order}, placement)
    ns = dict(torch=torch, re=re, Iterable=Iterable, EXPERT_PLACEMENT=str(placement), logger=NS(info_once=lambda *a: None))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    router = torch.arange(8).reshape(4, 2).float()
    weights = [('layers.3.moe.gate.weight', router), ('layers.3.moe.router_bias', torch.arange(4).float())]
    payloads = {}
    for e in range(4):
        for projection in ['gate_proj', 'up_proj', 'down_proj']:
            for suffix, dtype in [('weight_packed', torch.int32), ('weight_scale', torch.float32), ('weight_shape', torch.int64)]:
                name = f'layers.3.moe.experts.{e}.{projection}.{suffix}'
                payloads[name] = torch.tensor([e, e+1], dtype=dtype)
                weights.append((name, payloads[name]))
    out = dict(ns['_place_experts'](iter(weights), [3], per_expert=True))
    torch.testing.assert_close(out['layers.3.moe.gate.weight'], router[order[0]])
    torch.testing.assert_close(out['layers.3.moe.router_bias'], order[0].float())
    inverse = torch.argsort(order[0]).tolist()
    for name, payload in payloads.items():
        e = int(name.split('.')[4])
        mapped = name.replace(f'.experts.{e}.', f'.experts.{inverse[e]}.')
        assert out[mapped] is payload
    with pytest.raises(ValueError, match='packed experts'):
        list(ns['_place_experts'](iter(weights), [3]))
