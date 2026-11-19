# SPDX-License-Identifier: Apache-2.0
"""CPU-only owner/scale and route-capacity contracts; no Torch dependency.

The numeric GPU oracle lives in the separate paired MoE validation harness.
These tests exercise the real post-load function with storage metadata doubles
to catch repeated in-place packing, a second owner, and scale compensation.
"""
import ast
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'vllm_hcu/model_executor/layers/quantization'
spec = importlib.util.spec_from_file_location('native_policy', SOURCE / 'ct_w4a8_step5_native_policy.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class Parameter:
    next_pointer = 10000

    def __init__(self, shape, dtype='int8', value=None):
        self.shape, self.dtype, self.value = tuple(shape), dtype, value
        self.ndim, self._version = len(shape), 0
        self.device = SimpleNamespace(type='cuda', index=0)
        self.pointer = Parameter.next_pointer
        Parameter.next_pointer += 100
        self.weight_loader = lambda *args: None

    def data_ptr(self):
        return self.pointer

    def numel(self):
        result = 1
        for n in self.shape:
            result *= n
        return result

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=self.data_ptr,
                               nbytes=lambda: self.numel() * (4 if self.dtype == 'float32' else 1))

    def storage_offset(self):
        return 0

    def is_contiguous(self):
        return True

    def contiguous(self):
        return self

    def requires_grad_(self, value):
        return self

    def __truediv__(self, divisor):
        result = Parameter(self.shape, self.dtype, self.value / divisor)
        result.device = self.device
        return result


def layer():
    result = SimpleNamespace(w13_weight=Parameter((44, 3072, 2048)),
                             w2_weight=Parameter((44, 4096, 768)),
                             w13_weight_scale=Parameter((44, 3072, 1), 'float32', 0.25),
                             w2_weight_scale=Parameter((44, 4096, 1), 'float32', 0.5))
    for value in vars(result).values():
        value.device = result.w13_weight.device
    return result


def postload_function(pack):
    tree = ast.parse((SOURCE / 'ct_w4a8_step5_native.py').read_text())
    names = {'_weight_key', '_signature', '_tracked_version', '_no_reload', 'prepare_native_w4a8'}
    functions = ast.Module(body=[n for n in tree.body
                                if isinstance(n, ast.FunctionDef) and n.name in names],
                           type_ignores=[])
    namespace = {'torch': SimpleNamespace(
        int8='int8', float32='float32', no_grad=nullcontext,
        cuda=SimpleNamespace(get_device_properties=lambda _: SimpleNamespace(gcnArchName='gfx936:sramecc+:xnack-'))),
        '_PACKED_OWNERS': {}, '_require_kernel': lambda: (None, pack),
        'NATIVE_TOKEN_CHUNK': policy.NATIVE_TOKEN_CHUNK,
        'logger': SimpleNamespace(info_once=lambda *args: None)}
    exec(compile(functions, '<actual-native-postload>', 'exec'), namespace)
    return namespace['prepare_native_w4a8']


class NativeContract(unittest.TestCase):
    def test_optin_is_off_and_other_values_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(policy.native_enabled())
        with patch.dict(os.environ, {policy.FLAG: '1'}):
            self.assertTrue(policy.native_enabled())
        with patch.dict(os.environ, {policy.FLAG: 'auto'}):
            with self.assertRaises(ValueError):
                policy.native_enabled()

    def test_parallel_ownership_contract(self):
        good = dict(all2all_backend='allgather_reducescatter', tp_size=1,
                    dp_size=8, ep_size=8, enable_eplb=False)
        policy.validate_parallel(SimpleNamespace(**good), 'swiglustep', 'auto')
        policy.validate_parallel(SimpleNamespace(**(good | dict(dp_size=1, pcp_size=8))),
                                 'swiglustep', 'auto')
        for override in [dict(dp_size=1), dict(ep_size=16), dict(tp_size=2),
                         dict(dp_size=1, pcp_size=4), dict(pcp_size=8), dict(sp_size=2),
                         dict(enable_eplb=True), dict(all2all_backend='deepep_low_latency')]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                policy.validate_parallel(SimpleNamespace(**(good | override)), 'swiglustep', 'auto')

    def test_route_capacity_includes_skew_and_every_expert_padding(self):
        rng = random.Random(47893)
        for tokens in [1, 3, 17, 128, 1024, 8192, 65536]:
            routes = tokens * 8
            bound = policy.padded_route_rows(tokens, 8, 44)
            for scenario in range(12):
                if scenario == 0:
                    counts = [routes] + [0] * 43
                elif scenario == 1:
                    counts = [routes // 44 + (i < routes % 44) for i in range(44)]
                else:
                    # Random split, including empty/nonlocal routes.
                    cuts = sorted(rng.randrange(routes + 1) for _ in range(45))
                    counts = [cuts[i + 1] - cuts[i] for i in range(44)]
                actual = sum(((count + 63) // 64) * 64 for count in counts)
                self.assertLessEqual(actual, bound)
            self.assertEqual(bound % 64, 0)

    def test_fixed_route_alignment_and_worst_workspace(self):
        with self.assertRaises(ValueError):
            policy.padded_route_rows(32, 8, 44, alignment=128)
        rows = policy.padded_route_rows(65536, 8, 44)
        self.assertEqual(rows, 527104)
        storage = policy.workspace_bytes(rows)
        self.assertEqual(storage['INT8'], rows * 4096)
        self.assertEqual(storage['BF16'], rows * 4096 * 2)
        self.assertEqual(storage['total'], 6477053952)

    def test_fixed_token_chunks_cover_without_overlap_or_capacity_growth(self):
        self.assertEqual(policy.NATIVE_TOKEN_CHUNK, 8192)
        for tokens in (0, 1, 8191, 8192, 8193, 32767, 32768, 32769, 65536, 65539, 131073):
            spans = list(policy.token_chunk_ranges(tokens))
            self.assertEqual(sum(b - a for a, b in spans), tokens)
            previous = 0
            for start, end in spans:
                self.assertEqual(start, previous)
                self.assertGreater(end, start)
                self.assertLessEqual(end - start, 8192)
                previous = end
            self.assertEqual(previous, tokens)
        capped = policy.padded_route_rows(policy.NATIVE_TOKEN_CHUNK, 8, 44)
        self.assertEqual(policy.workspace_bytes(capped)['total'], 801 * 2 ** 20)
        with self.assertRaises(ValueError):
            list(policy.token_chunk_ranges(-1))

    def test_actual_forward_reuses_one_output_and_same_kernels_for_every_slice(self):
        # Execute the actual forward control flow with metadata-only tensors.
        # Only local imports are replaced by supplied doubles; allocation,
        # slicing, loop, launch arguments and direct output views are real.
        class Tensor:
            def __init__(self, shape, dtype, device, owner=None, start=0):
                self.shape, self.dtype, self.device = tuple(shape), dtype, device
                self.ndim = len(shape)
                self.owner, self.start = owner if owner is not None else self, start

            def __getitem__(self, key):
                assert isinstance(key, slice) and key.step is None
                begin = key.start or 0
                end = self.shape[0] if key.stop is None else key.stop
                return Tensor((end - begin, *self.shape[1:]), self.dtype,
                              self.device, self.owner, self.start + begin)

            def view(self, *shape):
                return Tensor(shape, self.dtype, self.device, self.owner, self.start)

            def to(self, dtype):
                assert self.dtype == dtype
                return self

            def contiguous(self):
                return self

            def numel(self):
                result = 1
                for value in self.shape:
                    result *= value
                return result

        class IgnoreImports(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return ast.copy_location(ast.Pass(), node)

        class Launch:
            def __init__(self, fn):
                self.fn = fn

            def __getitem__(self, grid):
                return self.fn

        tree = ast.parse((SOURCE / 'ct_w4a8_step5_native.py').read_text())
        apply = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == 'apply_native_w4a8')
        module = ast.fix_missing_locations(IgnoreImports().visit(
            ast.Module(body=[apply], type_ignores=[])))
        for tokens in (1, 8191, 8192, 8193, 32767, 32768, 32769, 65536, 65539):
            with self.subTest(tokens=tokens):
                device = SimpleNamespace(type='cuda', index=0)
                allocations, reserve_rows, quant_rows, gemms, activations, writes = [], [], [], [], [], []
                def allocate(shape, *, dtype, device):
                    out = Tensor(shape, dtype, device)
                    allocations.append(out)
                    return out
                workspace = SimpleNamespace(lock=nullcontext())
                def reserve(rows, hidden, intermediate, device):
                    reserve_rows.append(rows)
                    workspace.integer = Tensor((rows * hidden,), 'int8', device)
                    workspace.bf16 = Tensor((rows * hidden,), 'bfloat16', device)
                workspace.reserve = reserve
                def quantize(value):
                    quant_rows.append(value.shape[0])
                    return (Tensor(value.shape, 'int8', device),
                            Tensor((value.shape[0], 1), 'float32', device))
                def permute(a, scale, ids, mapping, experts, out):
                    self.assertEqual(experts, 44)
                    return (out, Tensor((out.shape[0], 1), 'float32', device),
                            Tensor((out.shape[0],), 'int32', device), ids, 64)
                def kernel(a, w, output, indices, config):
                    gemms.append((output.shape, dict(config)))
                def activate(*args, **kwargs):
                    activations.append((args[4:8], kwargs))
                def weighted(*args, **kwargs):
                    out = args[5]
                    self.assertIs(out, args[6])
                    self.assertIs(out.owner, allocations[0])
                    self.assertEqual(args[7], out.shape[0])
                    writes.append((out.start, out.start + out.shape[0]))
                current_layer = SimpleNamespace(
                    _step5_native_w4a8_signature='prepared',
                    w13_weight=Tensor((44, 3072, 2048), 'int8', device),
                    w2_weight=Tensor((44, 4096, 768), 'int8', device),
                    _step5_native_w13_scale=None, _step5_native_w2_scale=None,
                    _expert_map=Tensor((352,), 'int32', device),
                    global_num_experts=352, apply_router_weight_on_input=False)
                namespace = dict(torch=SimpleNamespace(
                    bfloat16='bfloat16', int32='int32', float32='float32', empty=allocate),
                    _signature=lambda _: 'prepared', _workspace=lambda _: workspace,
                    _require_kernel=lambda: (kernel, None), per_token_quant_hip=quantize,
                    permute64=permute, activation_persistent=Launch(activate),
                    _weighted_output=Launch(weighted), padded_route_rows=policy.padded_route_rows,
                    token_chunk_ranges=policy.token_chunk_ranges,
                    NATIVE_TOKEN_CHUNK=policy.NATIVE_TOKEN_CHUNK,
                    triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b))
                exec(compile(module, '<actual-native-forward>', 'exec'), namespace)
                result = namespace['apply_native_w4a8'](
                    current_layer, Tensor((tokens, 4096), 'bfloat16', device),
                    Tensor((tokens, 8), 'float32', device),
                    Tensor((tokens, 8), 'int32', device))
                spans = list(policy.token_chunk_ranges(tokens))
                self.assertEqual(writes, spans)
                self.assertEqual(len(allocations), 1)
                self.assertIs(result, allocations[0])
                self.assertEqual(result.shape, (tokens, 4096))
                self.assertEqual(quant_rows, [b - a for a, b in spans])
                self.assertEqual(reserve_rows, [policy.padded_route_rows(min(tokens, 8192), 8, 44)])
                self.assertEqual(len(gemms), len(spans) * 2)
                self.assertTrue(all(config == {'MODE': 1002} for _, config in gemms))
                self.assertEqual(len(activations), len(spans))
                self.assertTrue(all(args[1:] == (1536, 8, 512) for args, _ in activations))

    def test_every_reserved_expert_index_is_initialized(self):
        # Independent coverage oracle for the launch grid and stride. This
        # includes M8 where the old two-block scheme left832 poisoned slots.
        for tokens in [1, 3, 5, 8, 17, 32, 64, 127, 128, 129, 8192, 65536]:
            routes = tokens * 8
            groups = (routes + 1023) // 1024
            bound = policy.padded_route_rows(tokens, 8, 44)
            coverage = [0] * bound
            for group in range(groups):
                for start in range(group * 1024, bound, groups * 1024):
                    for i in range(start, min(start + 1024, bound)):
                        coverage[i] += 1
            self.assertEqual(set(coverage), {1}, (tokens, bound))
        # Verify the actual GPU source uses that full-BOUND stride loop, not
        # an independently correct oracle paired with stale GPU source.
        tree = ast.parse((SOURCE / 'ct_w4a8_step5_native_routing.py').read_text())
        histogram = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_histogram')
        loops = [n for n in ast.walk(histogram) if isinstance(n, ast.For)]
        self.assertTrue(any(isinstance(n.iter, ast.Call) and len(n.iter.args) == 3
                            and isinstance(n.iter.args[1], ast.Name)
                            and n.iter.args[1].id == 'BOUND' for n in loops))

    def test_untracked_parameter_rejected_before_pack_or_scale_derivation(self):
        class InferenceParameter(Parameter):
            def __getattribute__(self, name):
                if name == '_version':
                    raise RuntimeError('Inference tensors do not track version counter')
                return super().__getattribute__(name)
        calls = []
        prepare = postload_function(lambda w: calls.append(w) or w)
        x = layer()
        x.w13_weight = InferenceParameter((44, 3072, 2048))
        with self.assertRaisesRegex(RuntimeError, 'outside torch.inference_mode'):
            prepare(x)
        self.assertEqual(calls, [])
        self.assertFalse(hasattr(x, '_step5_native_w13_scale'))

    def test_pack_one_owner_scale_once_and_repeated_prepare(self):
        calls = []
        def pack(weight):
            calls.append(weight.data_ptr())
            weight._version += 1
            return weight
        prepare = postload_function(pack)
        x = layer()
        pointers = [x.w13_weight.data_ptr(), x.w2_weight.data_ptr()]
        prepare(x)
        self.assertEqual(calls, pointers)
        self.assertEqual(x.w13_weight_scale.value, 0.25)
        self.assertEqual(x.w2_weight_scale.value, 0.5)
        self.assertEqual(x._step5_native_w13_scale.value, 0.25 / 16)
        self.assertEqual(x._step5_native_w2_scale.value, 0.5 / 16)
        derived = x._step5_native_w13_scale
        prepare(x)
        self.assertEqual(calls, pointers)
        self.assertIs(derived, x._step5_native_w13_scale)
        with self.assertRaises(RuntimeError):
            x.w13_weight.weight_loader(None)
        x.w13_weight_scale._version += 1
        with self.assertRaises(RuntimeError):
            prepare(x)

    def test_shared_weight_storage_not_packed_twice(self):
        calls = []
        def pack(weight):
            calls.append(weight.data_ptr())
            weight._version += 1
            return weight
        prepare = postload_function(pack)
        first = layer()
        second = SimpleNamespace(**vars(first))
        prepare(first)
        prepare(second)
        self.assertEqual(len(calls), 2)

    def test_reject_second_weight_owner(self):
        def allocates_copy(weight):
            return Parameter(weight.shape)
        with self.assertRaisesRegex(RuntimeError, 'second full GPU weight owner'):
            postload_function(allocates_copy)(layer())


if __name__ == '__main__':
    unittest.main()
