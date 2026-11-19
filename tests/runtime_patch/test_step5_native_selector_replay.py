"""Replay actual selector callbacks in dispatcher order, with no Torch/GPU.

Only external Torch/vLLM dependency objects and the legacy/GPU method factories
are metadata doubles. All four callback bodies, validation helpers, the Step5
family predicate, native policy and core CT method constructor are real source.
"""
import ast
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / 'vllm_hcu/patch/worker'
QUANT = ROOT / 'vllm_hcu/model_executor/layers/quantization'
TARGET = ('vllm.model_executor.layers.quantization.compressed_tensors.'
          'compressed_tensors_moe.compressed_tensors_moe')


def callback_order():
    tree = ast.parse((WORKER / '__init__.py').read_text())
    values = {n.target.id: n.value for n in tree.body if isinstance(n, ast.AnnAssign)
              and isinstance(n.target, ast.Name)}
    output = []
    for group in values['_ALL_CALLBACK_GROUPS'].elts:
        for spec in values[group.id].elts:
            call = spec.args[0]
            category, name = (ast.literal_eval(arg) for arg in call.args)
            path = WORKER / category.replace('.', '/') / (name + '.py')
            source = ast.parse(path.read_text())
            target = next((n.value for n in source.body if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == 'TARGET_MODULE'
                                   for t in n.targets)), None)
            if target is not None:
                try:
                    if ast.literal_eval(target) == TARGET:
                        output.append((category, name))
                except (ValueError, TypeError):
                    pass
    return output


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def replay(flag='1', model_type='step5', activation_bits=8, dp=8, pcp=1, symmetric=True):
    modules = {}
    def package(name, path=None):
        result = ModuleType(name)
        result.__path__ = [] if path is None else [str(path)]
        modules[name] = result
        return result
    for name in ('vllm', 'vllm.model_executor', 'vllm.model_executor.layers',
                 'vllm.model_executor.layers.quantization', 'compressed_tensors'):
        package(name)
    for suffix, path in [('', ROOT / 'vllm_hcu'), ('.patch', WORKER.parent),
            ('.patch.worker', WORKER), ('.patch.worker.core_fix', WORKER / 'core_fix'),
            ('.patch.worker.op_opt', WORKER / 'op_opt'),
            ('.model_executor', QUANT.parent.parent), ('.model_executor.layers', QUANT.parent),
            ('.model_executor.layers.quantization', QUANT),
            ('.runtime_compat', ROOT / 'vllm_hcu/runtime_compat')]:
        package('vllm_hcu' + suffix, path)
    torch = package('torch')
    torch.Tensor = type('Tensor', (), {})
    torch.nn = package('torch.nn')
    class MetadataTensor:
        def __init__(self, shape, dtype):
            self.shape, self.dtype = tuple(shape), dtype
    class Parameter(MetadataTensor):
        def __init__(self, source, requires_grad):
            super().__init__(source.shape, source.dtype)
            self.requires_grad = requires_grad
    torch.nn.Parameter = Parameter
    torch.empty = torch.ones = lambda shape, dtype: MetadataTensor(shape, dtype)
    torch.from_numpy = lambda value: value
    for dtype in ('int8', 'int32', 'int64', 'float32', 'uint8', 'float16', 'bfloat16'):
        setattr(torch, dtype, dtype)
    logger = package('vllm.logger')
    logger.init_logger = lambda _: NS(info_once=lambda *a: None)
    fused = package('vllm.model_executor.layers.fused_moe')
    class Base:
        def __init__(self, moe):
            self.moe, self.moe_kernel = moe, None
    fused.FusedMoEMethodBase = Base
    fused.FusedMoeWeightScaleSupported = NS(CHANNEL=NS(value='channel'))
    quantconfig = package('vllm.model_executor.layers.fused_moe.config')
    quantconfig.FusedMoEQuantConfig = NS(make=lambda *a, **kw: kw)
    utils = package('vllm.model_executor.utils')
    def set_weight_attrs(param, attrs):
        for key, value in attrs.items():
            setattr(param, key, value)
    utils.set_weight_attrs = set_weight_attrs
    cfg = NS(model_config=NS(hf_config=NS(model_type=model_type,
             text_config=NS(model_type='step5' if model_type == 'step5_vl' else model_type),
             architectures=[model_type])))
    config = package('vllm.config')
    config.get_current_vllm_config_or_none = lambda: cfg
    platforms = package('vllm.platforms')
    platforms.current_platform = NS(is_rocm=lambda: True, device_type='cuda')
    modules['compressed_tensors'].CompressionFormat = NS(pack_quantized=NS(value='pack-quantized'))
    ctq = package('compressed_tensors.quantization')
    ctq.QuantizationStrategy = NS(CHANNEL='channel', TOKEN='token')
    ctq.QuantizationType = NS(INT='int')
    generic = package('vllm_hcu.model_executor.layers.quantization.compressed_tensors_w4a8')
    generic.HCUCompressedTensorsW4A8MoEMethod = lambda *a: 'generic-lightop'
    native_stub = package('vllm_hcu.model_executor.layers.quantization.ct_w4a8_step5_native')
    prepared = []
    native_stub.prepare_native_w4a8 = lambda layer: prepared.append(layer)
    with patch.dict(sys.modules, modules), patch.dict(os.environ, {'VLLM_STEP5_NATIVE_W4A8': flag}):
        prefix = 'vllm_hcu.model_executor.layers.quantization.'
        load('vllm_hcu.runtime_compat.step5', ROOT / 'vllm_hcu/runtime_compat/step5.py')
        load(prefix + 'ct_w4a8_step5_native_policy', QUANT / 'ct_w4a8_step5_native_policy.py')
        core_method = load(prefix + 'ct_w4a8_step5', QUANT / 'ct_w4a8_step5.py')
        class Upstream:
            @staticmethod
            def get_moe_method(quant_config, layer, layer_name):
                return 'upstream'
        target = ModuleType(TARGET)
        target.CompressedTensorsMoEMethod = Upstream
        target.current_platform = platforms.current_platform
        target.get_current_vllm_config = lambda: cfg
        adapters = []
        for category, name in callback_order():
            adapter = load('vllm_hcu.patch.worker.' + category + '.' + name,
                           WORKER / category / (name + '.py'))
            if name == 'patch_step5_compressed_tensors_w4a8':
                adapter._step5_method_class = lambda: lambda *a: 'legacy-slimquant'
            assert adapter.apply_to_module(target)
            assert not adapter.apply_to_module(target)
            adapters.append(adapter)
        weight = NS(type='int', strategy='channel', num_bits=4, symmetric=symmetric)
        activation = NS(type='int', strategy='token', num_bits=activation_bits)
        scheme = {'format': 'pack-quantized', 'weights': weight, 'input_activations': activation}
        quant = NS(_add_fused_moe_to_target_scheme_map=lambda: None,
            get_scheme_dict=lambda layer, name: scheme,
            _is_dynamic_token_w4a8_int=lambda w, a: w.num_bits == 4 and a.num_bits == 8)
        layer = NS(moe_config=NS(activation='swiglustep', has_bias=False, moe_backend='auto',
            moe_parallel_config=NS(all2all_backend='allgather_reducescatter', tp_size=1,
                                   dp_size=dp, pcp_size=pcp, ep_size=8, enable_eplb=False)))
        yield NS(select=lambda: target.CompressedTensorsMoEMethod.get_moe_method(
            quant, layer, 'model.layers.3.moe.experts'), method=core_method,
            prepared=prepared, layer=layer, adapters=adapters, torch=torch)


class SelectorReplay(unittest.TestCase):
    def test_real_dispatcher_order_contains_both_inner_interceptors(self):
        self.assertEqual(callback_order(), [
            ('core_fix', 'patch_step5_ct_w4a8'),
            ('op_opt', 'patch_compressed_tensors_moe_w4a8'),
            ('op_opt', 'patch_compressed_tensors_moe_w4a8_int8'),
            ('op_opt', 'patch_step5_compressed_tensors_w4a8')])

    def test_native_one_selects_real_core_class_and_reaches_native_postload(self):
        for model_type in ('step5', 'step5_vl'):
            with self.subTest(model_type=model_type), replay(model_type=model_type) as case:
                method = case.select()
                self.assertIsInstance(method, case.method.Step5CompressedW4A8MoEMethod)
                self.assertTrue(method._use_native_w4a8)
                layer = NS(_parameters={'w13_weight_packed': object(), 'w2_weight_packed': object()},
                           w13_weight_scale=object(), w2_weight_scale=object())
                method.process_weights_after_loading(layer)
                self.assertEqual(case.prepared, [layer])
                self.assertFalse(layer._parameters)

    def test_native_zero_preserves_legacy_final_selector(self):
        with replay(flag='0') as case:
            self.assertEqual(case.select(), 'legacy-slimquant')
            self.assertEqual(case.prepared, [])

    def test_legacy_and_new_loader_conversion_share_exact_canonical_bytes(self):
        import numpy as np
        # Cover every byte pattern, all sixteen nibble values, and leave the
        # memory-mapped-equivalent checkpoint array unchanged.
        words = np.arange(256, dtype=np.uint8).reshape(64, 4).view(np.int32)
        before = words.copy()
        with replay() as case:
            packed = NS(dtype=case.torch.int32, ndim=2, device=NS(type='cpu'),
                        contiguous=lambda: NS(numpy=lambda: words))
            old = case.adapters[-1].compressed_tensors_w4a8_to_slimquant(packed)
            new = case.method.ct_int32_to_packed_int4(packed)
            raw = np.arange(256, dtype=np.uint8).reshape(64, 4)
            expected = ((((raw & 15).astype(np.int16) - 8) & 15) << 4
                        | ((((raw >> 4).astype(np.int16) - 8) & 15)))
            np.testing.assert_array_equal(old.view(np.uint8), expected.astype(np.uint8))
            np.testing.assert_array_equal(new, old)
            np.testing.assert_array_equal(words, before)

    def test_selected_native_create_and_load_keep_names_shapes_scales_and_nonlocal_skip(self):
        import numpy as np
        with replay() as case:
            method = case.select()
            class Layer:
                def __init__(self):
                    self._parameters = {}
                def register_parameter(self, name, value):
                    self._parameters[name] = value
                    setattr(self, name, value)
                def _map_global_expert_id_to_local_expert_id(self, expert):
                    return expert if expert < 44 else -1
            layer, loaded = Layer(), []
            def original(param, value, weight_name, **kwargs):
                loaded.append((param, value, weight_name, kwargs))
                return True
            method.create_weights(layer, 44, 4096, 1536, 'bfloat16', weight_loader=original)
            self.assertIs(layer.w13_weight, layer.w13_weight_packed)
            self.assertEqual(layer.w13_weight.shape, (44, 3072, 2048))
            self.assertEqual(layer.w2_weight.shape, (44, 4096, 768))
            self.assertEqual(layer.w13_weight_scale.shape, (44, 3072, 1))
            self.assertEqual(layer.w13_weight_scale.dtype, case.torch.float32)
            self.assertEqual(layer.w13_weight_scale.quant_method, 'channel')
            words = np.array([[0x76543210]], dtype=np.int32)
            packed = NS(dtype=case.torch.int32, ndim=2, device=NS(type='cpu'),
                        contiguous=lambda: NS(numpy=lambda: words))
            load = layer.w13_weight_packed.weight_loader
            self.assertTrue(load(layer.w13_weight_packed, packed, 'prefix.w13_weight_packed',
                                 shard_id='w1', expert_id=3, return_success=True))
            self.assertEqual(loaded[0][2], 'prefix.w13_weight')
            self.assertEqual(loaded[0][3], dict(shard_id='w1', expert_id=3, return_success=True))
            self.assertEqual(layer._step5_ct_load_generation, 1)
            self.assertFalse(load(layer.w13_weight_packed, packed, 'prefix.w13_weight_packed',
                                  shard_id='w1', expert_id=77, return_success=True))
            self.assertEqual(len(loaded), 1)
            self.assertEqual(layer._step5_ct_load_generation, 1)

    def test_other_model_and_quantization_remain_on_original_chain(self):
        with replay(model_type='qwen_moe') as case:
            self.assertEqual(case.select(), 'generic-lightop')
        with replay(activation_bits=4) as case:
            self.assertEqual(case.select(), 'upstream')

    def test_native_cp8_selects_same_core_method_without_falling_back_to_boltops(self):
        with replay(dp=1, pcp=8) as case:
            method = case.select()
            self.assertIsInstance(method, case.method.Step5CompressedW4A8MoEMethod)
            self.assertTrue(method._use_native_w4a8)

    def test_native_rejects_unsupported_parallelism_and_asymmetric_weights(self):
        with replay(dp=1) as case, self.assertRaisesRegex(ValueError, 'TP1/EP8'):
            case.select()
        with replay(dp=8, pcp=8) as case, self.assertRaisesRegex(ValueError, 'TP1/EP8'):
            case.select()
        with replay(symmetric=False) as case, self.assertRaisesRegex(ValueError, 'symmetric per-channel'):
            case.select()


if __name__ == '__main__':
    unittest.main()
