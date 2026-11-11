"""Geometry and ownership regressions for the Step5 DeepEP adapter."""
from types import SimpleNamespace as NS
import pytest
import torch
from vllm_hcu.model_executor.layers.fused_moe import step5_w4a8_policy as policy
from vllm_hcu.model_executor.layers.fused_moe.step5_w4a8 import validate_loaded_model


def quantization():
    return {"quant_method":"compressed-tensors","config_groups":{"moe":{
        "format":"pack-quantized",
        "weights":{"num_bits":4,"type":"int","strategy":"channel","symmetric":True,"dynamic":False},
        "input_activations":{"num_bits":8,"type":"int","strategy":"token","symmetric":True,"dynamic":True}}}}


@pytest.mark.parametrize("tp,dp,pp",[(1,8,1),(2,4,1),(4,2,1),(8,1,1),(2,2,2),(1,1,1)])
def test_auto_profile_accepts_framework_parallel_geometry(monkeypatch,tp,dp,pp):
    monkeypatch.setenv("VLLM_HCU_STEP5_W4A8_PROFILE","auto")
    raw=quantization()
    cfg=NS(model_config=NS(hf_config=NS(model_type="step5",quantization_config=raw),
                           quantization="compressed-tensors"),
           parallel_config=NS(tensor_parallel_size=tp,data_parallel_size=dp,
                pipeline_parallel_size=pp,enable_expert_parallel=True,enable_eplb=False),
           additional_config={"hcu":{"deepep_auto":True}},lora_config=None)
    assert policy.profile_for_model(cfg)=="auto"
    assert cfg.model_config.quantization=="compressed-tensors"


@pytest.mark.parametrize("local_experts",[11,22,44,88,176])
def test_owner_validation_uses_stage_and_expert_geometry(monkeypatch,local_experts):
    monkeypatch.setenv("VLLM_HCU_STEP5_W4A8_PROFILE","auto")
    model=torch.nn.Module()
    layer=torch.nn.Module();model.add_module('expert',layer)
    layer._hcu_step5_gptq_kind='w4a8'
    layer._hcu_step5_w4a8_owner=True
    layer.register_parameter('w13_weight',torch.nn.Parameter(torch.zeros((local_experts,8,8),dtype=torch.int8),requires_grad=False))
    layer.register_parameter('w2_weight',torch.nn.Parameter(torch.zeros((local_experts,16,2),dtype=torch.int8),requires_grad=False))
    validate_loaded_model(model)
    # Distinct projections may never silently alias one storage owner.
    layer.w2_weight=layer.w13_weight
    with pytest.raises(RuntimeError,match='one packed owner'):
        validate_loaded_model(model)


def test_pp_stage_with_no_routed_experts_is_valid(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_STEP5_W4A8_PROFILE","auto")
    validate_loaded_model(torch.nn.Module())


def test_step5_tp_only_auto_creates_ep_communicator(monkeypatch):
    import vllm.config
    from tests.runtime_patch.test_worker_framework_opt import _fake_base_communicator_module
    from vllm_hcu.patch.worker.framework_opt import patch_base_device_communicator

    monkeypatch.setenv("VLLM_HCU_STEP5_W4A8_PROFILE", "auto")
    cfg = NS(additional_config={"hcu": {"deepep_auto": True}},
             model_config=NS(hf_config=NS(model_type="step5")),
             parallel_config=NS(tensor_parallel_size=8, data_parallel_size=1))
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: cfg)
    module = _fake_base_communicator_module()
    patch_base_device_communicator.apply_to_module(module)
    assert module.DeviceCommunicatorBase(object(), unique_name="ep:0").use_all2all
    assert not module.DeviceCommunicatorBase(object(), unique_name="tp:0").use_all2all
    cfg.model_config.hf_config.model_type = "qwen3_moe"
    assert not module.DeviceCommunicatorBase(object(), unique_name="ep:0").use_all2all


def test_step5_tp_only_auto_passes_platform_validation(monkeypatch):
    from tests.runtime_patch.test_platform_hcu_config import _validation_config
    from vllm_hcu.patch.config import HcuFeatureConfig
    from vllm_hcu.patch.platform.core_fix import patch_vllm_config

    monkeypatch.setenv("VLLM_HCU_STEP5_W4A8_PROFILE", "auto")
    cfg = _validation_config(HcuFeatureConfig(deepep_auto=True, moe_backend="deep_gemm"))
    cfg.model_config.hf_config = NS(model_type="step5")
    cfg.parallel_config.tensor_parallel_size = 8
    cfg.parallel_config.enable_expert_parallel = True
    cfg.parallel_config.all2all_backend = "deepep_low_latency"
    assert patch_vllm_config.validate_and_update_hcu_config(cfg).deepep_auto
