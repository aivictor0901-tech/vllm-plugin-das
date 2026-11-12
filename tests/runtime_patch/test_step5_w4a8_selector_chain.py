"""Regression coverage for the composed Step5 compressed-tensors selectors."""

import importlib
from types import ModuleType, SimpleNamespace as NS

import pytest
from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import QuantizationStrategy, QuantizationType


@pytest.mark.parametrize("model_type", ["step5", "step5_vl"])
@pytest.mark.parametrize("moe_backend", ["auto", "triton", "deep_gemm"])
@pytest.mark.parametrize(
    "all2all_backend", ["deepep_low_latency", "allgather_reducescatter"]
)
def test_step5_selector_chain_preserves_deepep_ll(
    monkeypatch, model_type, moe_backend, all2all_backend
):
    import vllm.config
    from vllm.platforms import current_platform
    from vllm_hcu.model_executor.layers.quantization.ct_w4a8_step5 import (
        Step5CompressedW4A8MoEMethod,
    )

    core = importlib.import_module(
        "vllm_hcu.patch.worker.core_fix.patch_step5_ct_w4a8"
    )
    bridge = importlib.import_module(
        "vllm_hcu.patch.worker.op_opt.patch_compressed_tensors_moe_w4a8_int8"
    )
    legacy = importlib.import_module(
        "vllm_hcu.patch.worker.op_opt.patch_step5_compressed_tensors_w4a8"
    )
    config = NS(model_config=NS(hf_config=NS(
        model_type=model_type,
        text_config=NS(model_type="step5"),
    )))
    monkeypatch.setattr(
        vllm.config, "get_current_vllm_config_or_none", lambda: config
    )
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    scheme = {
        "format": CompressionFormat.pack_quantized.value,
        "weights": NS(
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.CHANNEL,
            symmetric=True,
        ),
        "input_activations": NS(strategy=QuantizationStrategy.TOKEN),
    }
    quant = NS(
        _add_fused_moe_to_target_scheme_map=lambda: None,
        get_scheme_dict=lambda layer, name: scheme,
        _is_dynamic_token_w4a8_int=lambda weight, activation: True,
    )
    layer = NS(moe_config=NS(
        has_bias=False,
        activation="swiglustep",
        tp_size=4,
        dp_size=4,
        moe_backend=moe_backend,
        moe_parallel_config=NS(
            all2all_backend=all2all_backend,
            use_deepep_ll_kernels=all2all_backend == "deepep_low_latency",
        ),
    ))

    class CompressedTensorsMoEMethod:
        @staticmethod
        def get_moe_method(quant_config, layer, layer_name):
            raise AssertionError("Step5 W4A8 must be handled by an HCU selector")

    target = ModuleType(core.TARGET_MODULE)
    target.CompressedTensorsMoEMethod = CompressedTensorsMoEMethod
    target.current_platform = current_platform
    target.get_current_vllm_config = lambda: config
    # Match the worker dispatcher: core_fix, generic W4A8 bridge, Step5 op_opt.
    for adapter in (core, bridge, legacy):
        assert adapter.apply_to_module(target)
        assert not adapter.apply_to_module(target)
    selected = target.CompressedTensorsMoEMethod.get_moe_method(
        quant, layer, "model.layers.3.moe.experts"
    )
    if all2all_backend == "deepep_low_latency":
        assert isinstance(selected, Step5CompressedW4A8MoEMethod)
        assert selected._use_deepep
        assert selected._use_deepgemm == (moe_backend == "deep_gemm")
    else:
        assert isinstance(selected, legacy._step5_method_class())
