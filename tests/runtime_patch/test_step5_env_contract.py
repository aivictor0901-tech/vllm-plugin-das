# SPDX-License-Identifier: Apache-2.0
"""Step5 environment values must agree across config and model construction."""

import subprocess
import sys
import warnings
from types import ModuleType, SimpleNamespace

import pytest

from vllm_hcu.platforms import step5_envs


@pytest.mark.parametrize(
    "value, disabled",
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        (" FALSE ", False),
        ("1", True),
        ("true", True),
        ("TRUE", True),
        (" TrUe ", True),
        (" 1 ", True),
        ("yes", False),
    ],
)
def test_dsa_switch_controls_compilation_registration(monkeypatch, value, disabled):
    from vllm.config.compilation import CompilationConfig
    from vllm_hcu.models.step5_config import Step5Config

    for name in step5_envs.IGNORED_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    if value is None:
        monkeypatch.delenv("VLLM_STEP5_DISABLE_DSA", raising=False)
    else:
        monkeypatch.setenv("VLLM_STEP5_DISABLE_DSA", value)
    monkeypatch.setattr(CompilationConfig, "_attention_ops", [])

    Step5Config(sparse_config={"enabled": True})
    registered = "vllm::step4_dsa_index" in CompilationConfig._attention_ops
    getter = step5_envs.ENVIRONMENT_VARIABLES["VLLM_STEP5_DISABLE_DSA"]
    assert getter() is disabled
    assert registered is not disabled


def test_custom_ops_master_does_not_disable_dsa_semantics(monkeypatch):
    from vllm.config.compilation import CompilationConfig
    from vllm_hcu.models.step5_config import Step5Config

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.delenv("VLLM_STEP5_DISABLE_DSA", raising=False)
    monkeypatch.setattr(CompilationConfig, "_attention_ops", [])

    assert step5_envs.is_dsa_disabled() is False
    Step5Config(sparse_config={"enabled": True})
    assert "vllm::step4_dsa_index" in CompilationConfig._attention_ops


@pytest.mark.parametrize("name", step5_envs.IGNORED_ENVIRONMENT_VARIABLES)
def test_obsolete_settings_warn_only_when_explicitly_set(monkeypatch, name):
    from vllm_hcu.models.step5_config import Step5Config

    for key in step5_envs.IGNORED_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(key, raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Step5Config()
    # Other dependencies may emit unrelated deprecation warnings.
    assert not [
        str(warning.message)
        for warning in caught
        if issubclass(warning.category, UserWarning)
        and "is obsolete and has no effect" in str(warning.message)
    ]
    monkeypatch.setenv(name, "0")
    with pytest.warns(UserWarning, match=f"{name} is obsolete and has no effect"):
        Step5Config()
    assert name not in step5_envs.ENVIRONMENT_VARIABLES


def test_env_helpers_remain_dependency_light():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vllm_hcu.platforms.step5_envs; "
            "assert 'torch' not in sys.modules; "
            "assert 'vllm' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("architecture", "master", "fused_gate", "serialized", "expect_gate_split"),
    [
        ("Step5ForCausalLM", "1", "0", False, True),
        ("Step5ForCausalLM", "1", "0", True, True),
        ("Qwen3ForCausalLM", "1", "0", False, False),
        ("Step5ForCausalLM", "0", "0", False, False),
        ("Qwen3ForCausalLM", "1", "1", False, True),
    ],
)
def test_swa_int8_split_uses_bound_step5_compilation_owner(
    monkeypatch, architecture, master, fused_gate, serialized, expect_gate_split
):
    from vllm_hcu.patch.platform.core_fix import (
        patch_compilation_config,
        patch_step5_splitting_ops,
    )

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", master)
    monkeypatch.setenv("VLLM_HCU_STEP5_SWA_FUSED_QK", "1")
    monkeypatch.setenv("VLLM_HCU_STEP5_SWA_INT8_EPILOGUE", "1")
    monkeypatch.setenv("VLLM_HCU_STEP5_FUSED_GATE", fused_gate)

    class CompilationConfig:
        def __init__(self):
            self.splitting_ops = None

        def adjust_cudagraph_sizes_for_spec_decode(
            self, uniform_decode_query_len, tensor_parallel_size
        ):
            return uniform_decode_query_len

        def set_splitting_ops_for_v1(self, all2all_backend, data_parallel_size):
            self.splitting_ops = ["vllm::baseline"]

    target = ModuleType(patch_step5_splitting_ops.TARGET_MODULE)
    target.CUDAGraphMode = SimpleNamespace(
        NONE=object(),
        FULL=object(),
        PIECEWISE=object(),
        FULL_AND_PIECEWISE=object(),
        FULL_DECODE_ONLY=object(),
    )
    target.CompilationConfig = CompilationConfig
    patch_compilation_config.apply_to_module(target)
    patch_step5_splitting_ops.apply_to_module(target)
    compilation = CompilationConfig()
    if serialized:
        config = {
            "compilation_config": compilation,
            "model_config": {"architectures": [architecture]},
        }
    else:
        config = SimpleNamespace(
            additional_config={},
            compilation_config=compilation,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=[architecture])
            ),
        )
    patch_compilation_config.bind_hcu_config(config)
    compilation.set_splitting_ops_for_v1("allgather_reducescatter", 8)

    assert ("vllm::step5_swa_attention_gate_quant" in compilation.splitting_ops) is (
        expect_gate_split
    )
    assert ("vllm::step5_swa_qk_norm_rope_cache" in compilation.splitting_ops) is (
        master == "1"
    )
