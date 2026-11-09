"""CPU-only dispatch regression tests; run directly with Python or pytest."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def load_adapter():
    # Avoid importing unrelated core_fix adapters, which require torch/HCU.
    root = Path(__file__).resolve().parents[2] / "vllm_hcu/patch/worker/core_fix"
    package = ModuleType("_step5_ct_w4a8_tests")
    package.__path__ = [str(root)]
    name = package.__name__ + ".patch_step5_ct_w4a8"
    spec = importlib.util.spec_from_file_location(name, root / "patch_step5_ct_w4a8.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {package.__name__: package, name: module}):
        spec.loader.exec_module(module)
    return module


adapter = load_adapter()
ADAPTER_MODULE = "vllm_hcu.model_executor.layers.quantization.ct_w4a8_step5"


class Step5W4A8Tests(unittest.TestCase):
    def setUp(self):
        self.original_calls = []

        class CompressedTensorsMoEMethod:
            @staticmethod
            def get_moe_method(quant_config, layer, layer_name):
                self.original_calls.append((quant_config, layer, layer_name))
                return "upstream"

        self.target = ModuleType(adapter.TARGET_MODULE)
        self.target.CompressedTensorsMoEMethod = CompressedTensorsMoEMethod
        self.target.current_platform = SimpleNamespace(is_rocm=lambda: True)
        self.hf_config = SimpleNamespace(model_type="step5_vl")
        self.target.get_current_vllm_config = lambda: SimpleNamespace(
            model_config=SimpleNamespace(hf_config=self.hf_config)
        )
        self.weight = SimpleNamespace(strategy="channel", symmetric=True)
        self.scheme = {"weights": self.weight, "input_activations": object()}
        self.quant = SimpleNamespace(
            _add_fused_moe_to_target_scheme_map=Mock(),
            get_scheme_dict=Mock(return_value=self.scheme),
            _is_dynamic_token_w4a8_int=Mock(return_value=True),
        )
        self.layer = SimpleNamespace(moe_config=object())
        self.name = "language_model.model.layers.3.moe.experts"
        self.factory = Mock(return_value="step5-adapter")
        module = ModuleType(ADAPTER_MODULE)
        module.Step5CompressedW4A8MoEMethod = self.factory
        patcher = patch.dict(sys.modules, {ADAPTER_MODULE: module})
        patcher.start()
        self.addCleanup(patcher.stop)
        adapter.apply_to_module(self.target)

    def select(self):
        return self.target.CompressedTensorsMoEMethod.get_moe_method(
            self.quant, self.layer, self.name
        )

    def test_step5_uses_adapter_before_upstream_cpu_only_method(self):
        for model_type in ("step5", "step5_vl"):
            for strategy in ("channel", SimpleNamespace(value="channel")):
                with self.subTest(model_type=model_type, strategy=strategy):
                    self.hf_config.model_type = model_type
                    self.weight.strategy = strategy
                    self.assertEqual(self.select(), "step5-adapter")
        self.assertEqual(self.original_calls, [])
        self.factory.assert_called_with(self.quant, self.layer.moe_config)
        names = [call.args[1] for call in self.quant.get_scheme_dict.call_args_list[:3]]
        self.assertEqual(names, [self.name + s for s in (
            ".0.gate_proj", ".0.up_proj", ".0.down_proj"
        )])

    def test_other_models_and_platforms_bypass_adapter(self):
        self.hf_config.model_type = "qwen3_moe"
        self.assertEqual(self.select(), "upstream")
        self.hf_config.model_type = "step5_vl"
        self.target.current_platform.is_rocm = lambda: False
        self.assertEqual(self.select(), "upstream")
        self.quant._add_fused_moe_to_target_scheme_map.assert_not_called()
        self.factory.assert_not_called()

    def test_ignored_or_other_quantization_uses_upstream(self):
        self.quant.get_scheme_dict.return_value = None
        self.assertEqual(self.select(), "upstream")
        self.quant._is_dynamic_token_w4a8_int.assert_not_called()
        self.quant.get_scheme_dict.return_value = self.scheme
        self.quant._is_dynamic_token_w4a8_int.return_value = False
        self.assertEqual(self.select(), "upstream")
        self.factory.assert_not_called()

    def test_inconsistent_projection_schemes_fail_before_adapter(self):
        self.quant.get_scheme_dict.side_effect = [self.scheme, None, self.scheme]
        with self.assertRaisesRegex(ValueError, "same quantization scheme"):
            self.select()
        self.factory.assert_not_called()

    def test_unsupported_weight_layout_fails_explicitly(self):
        for strategy, symmetric in (("group", True), ("channel", False)):
            with self.subTest(strategy=strategy, symmetric=symmetric):
                self.weight.strategy = strategy
                self.weight.symmetric = symmetric
                with self.assertRaisesRegex(ValueError, "symmetric per-channel"):
                    self.select()
        self.factory.assert_not_called()

    def test_registration_is_idempotent_and_preserves_staticmethod(self):
        cls = self.target.CompressedTensorsMoEMethod
        method = cls.get_moe_method
        self.assertFalse(adapter.apply_to_module(self.target))
        self.assertIs(method, cls.get_moe_method)
        self.assertIsInstance(vars(cls)["get_moe_method"], staticmethod)

    def test_stale_marker_is_rejected(self):
        cls = self.target.CompressedTensorsMoEMethod
        cls.get_moe_method = staticmethod(lambda quant_config, layer, layer_name: None)
        with self.assertRaisesRegex(adapter.PatchCompatibilityError, "stale"):
            adapter.apply_to_module(self.target)

    def test_incompatible_upstream_signature_is_rejected(self):
        class ChangedAPI:
            @staticmethod
            def get_moe_method(quant_config, layer, renamed_parameter):
                pass

        self.target.CompressedTensorsMoEMethod = ChangedAPI
        with self.assertRaisesRegex(adapter.PatchCompatibilityError, "signature"):
            adapter.apply_to_module(self.target)


if __name__ == "__main__":
    unittest.main()
