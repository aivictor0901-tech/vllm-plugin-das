# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for Step5 local-Q/full-KV SWA, without device dependencies.

Native Triton arithmetic is not emulated or certified here. A numerical CPU
attention oracle stands in for its documented absolute-position interface.
These tests exercise the actual new dispatch and shared gather/restore code.
"""

from __future__ import annotations

import importlib.util
import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


swa = _load("step5_swa_pcp_cpu_contract", "vllm_hcu/v1/attention/step5_swa_pcp.py")
shared = _load("step5_pcp_cpu_contract", "vllm_hcu/v1/step5_pcp.py")


class Tensor:
    """Token-axis storage/view double; no GPU or attention math in this type."""

    def __init__(self, values, shape=None, *, dtype="float", storage=None, indices=None):
        self.storage = list(values) if storage is None else storage
        self.indices = list(range(len(values))) if indices is None else list(indices)
        self.shape = tuple(shape or (len(self.indices),))
        self.dtype = dtype
        self.ndim = len(self.shape)

    @property
    def values(self):
        return [self.storage[i] for i in self.indices]

    def __getitem__(self, key):
        expanded = isinstance(key, tuple)
        if expanded:
            key = key[0]
        if isinstance(key, int):
            return self.storage[self.indices[key]]
        selected = self.indices[key]
        shape = (len(selected), 1, 1) if expanded else (len(selected), *self.shape[1:])
        return Tensor([], shape, dtype=self.dtype, storage=self.storage, indices=selected)

    def numel(self):
        return len(self.indices)

    def contiguous(self):
        return self

    def assign(self, values):
        if len(values) != len(self.indices):
            raise AssertionError("CPU kernel wrote a different token count")
        for index, value in zip(self.indices, values):
            self.storage[index] = value
        return self

    def copy_(self, source):
        return self.assign(source.values)

    def masked_fill_(self, mask, value):
        return self.assign([value if flag else old
                            for flag, old in zip(mask.values, self.values)])

    def zero_(self):
        return self.assign([0] * len(self.indices))

    def new_zeros(self, shape):
        return Tensor([0] * shape[0], shape, dtype=self.dtype)


def _cpu_attention(q, keys, values, position, window):
    lo = max(0, position - window)
    scores = [q * keys[i] for i in range(lo, position + 1)]
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    return sum(w * values[i] for w, i in zip(weights, range(lo, position + 1))) / sum(weights)


def _kernel(calls, kind):
    def run(q, k, v, out, cu, ends, table, *tail):
        calls.append((kind, q.shape[0], tuple(cu.values), tuple(ends.values)))
        window = tail[-1]
        result = []
        for row, (start, stop) in enumerate(zip(cu.values, cu.values[1:])):
            req = table.values[row]
            for local, value in enumerate(q.values[start:stop]):
                position = ends.values[row] - (stop - start) + local
                result.append(_cpu_attention(value, k[req], v[req], position, window))
        out.assign(result)
    return run


def _metadata(lengths, ends, requests=None):
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return SimpleNamespace(
        num_actual_tokens=cu[-1],
        max_query_len=max(lengths, default=0),
        query_start_loc=Tensor(cu),
        seq_lens=Tensor(ends),
        block_table=Tensor(requests or [0] * len(lengths), (len(lengths), 1)),
        causal=True, use_cascade=False,
    )


class Step5SwaPCPContracts(unittest.TestCase):
    def _cache_update_method(self, writes):
        # Execute the actual backend entry point while replacing only external
        # device/cache dependencies. This catches double cache writes or an
        # accidental fall-through to the generic rank-major PCP gather.
        tree = ast.parse((ROOT / "vllm_hcu/v1/attention/backends/flash_attn.py").read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "FlashAttentionImpl")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "do_kv_cache_update")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
        namespace = {
            "AttentionType": SimpleNamespace(ENCODER_ONLY="encoder_only", ENCODER="encoder"),
            "_split_kv_cache": lambda cache, size: ("K-cache", "V-cache"),
            "reshape_and_cache_flash": lambda *args: writes.append(args),
            "get_pcp_group": lambda: self.fail("Step5 path reached generic PCP gather"),
        }
        exec(compile(ast.fix_missing_locations(module), "<actual FlashAttentionImpl KV entry>", "exec"), namespace)
        return namespace["do_kv_cache_update"]

    def test_cp8_dual_chunks_use_absolute_positions_and_full_window(self):
        # Earlier cached chunks + a new 256-token chunk, split into 16 pieces.
        computed, scheduled, width, window = 73, 256, 16, 31
        keys = [[math.sin(i / 13) for i in range(computed + scheduled)]]
        values = [[math.cos(i / 17) for i in range(computed + scheduled)]]
        expected = [_cpu_attention((i % 11) / 7, keys[0], values[0], computed + i, window)
                    for i in range(scheduled)]
        restored = [None] * scheduled
        for rank in range(8):
            starts = [rank * width, (15 - rank) * width]
            indices = [i for start in starts for i in range(start, start + width)]
            q = Tensor([(i % 11) / 7 for i in indices], (32, 1, 1))
            out = Tensor([float("nan")] * 32, (32, 1, 1))
            md = _metadata([width, width], [computed + s + width for s in starts])
            ctx = SimpleNamespace(local_num_tokens=32, local_is_padding=None)
            calls = []
            swa.forward_step5_swa_pcp(q, keys, values, out, md, ctx,
                softmax_scale=1.0, window_left=window,
                decode_kernel=_kernel(calls, "decode"), prefill_kernel=_kernel(calls, "prefill"))
            self.assertEqual(calls[0][0:2], ("prefill", 32))
            for index, value in zip(indices, out.values):
                self.assertIsNone(restored[index])
                restored[index] = value
        for actual, reference in zip(restored, expected):
            self.assertAlmostEqual(actual, reference, places=12)

    def test_replicated_decode_mtp_rows_use_no_query_gather(self):
        keys = [[i / 17 for i in range(35)], [i / 9 for i in range(49)]]
        values = [[i / 19 for i in range(35)], [i / 31 for i in range(49)]]
        q = Tensor([0.2, 0.3, 0.4, 0.5], (4, 1, 1))
        for rank in range(8):
            calls = []
            out = Tensor([0] * 4, (4, 1, 1))
            md = _metadata([1, 3], [35, 49], [0, 1])
            ctx = SimpleNamespace(local_num_tokens=4, local_is_padding=None)
            swa.forward_step5_swa_pcp(q, keys, values, out, md, ctx,
                softmax_scale=1.0, window_left=7,
                decode_kernel=_kernel(calls, "decode"), prefill_kernel=_kernel(calls, "prefill"))
            reference = [_cpu_attention(q.values[0], keys[0], values[0], 34, 7)]
            reference += [_cpu_attention(q.values[i + 1], keys[1], values[1], 46 + i, 7)
                          for i in range(3)]
            self.assertEqual(calls, [("decode", 4, (0, 1, 4), (35, 49))])
            for actual, expected in zip(out.values, reference):
                self.assertAlmostEqual(actual, expected, places=12)

    def test_virtual_padding_and_trailing_output_are_zeroed(self):
        q = Tensor([1, 1, 0, 0], (4, 1, 1))
        out = Tensor([float("nan")] * 6, (6, 1, 1))
        md = _metadata([2, 2], [8, 2])
        ctx = SimpleNamespace(local_num_tokens=4,
                              local_is_padding=Tensor([False, False, True, True]))
        def nan_kernel(*args):
            args[3].assign([2, 3, float("nan"), float("nan")])
        swa.forward_step5_swa_pcp(q, {}, {}, out, md, ctx,
            softmax_scale=1, window_left=7, decode_kernel=nan_kernel, prefill_kernel=nan_kernel)
        self.assertEqual(out.values, [2, 3, 0, 0, 0, 0])

    def test_global_cache_slots_follow_restore_order_not_rank_order(self):
        # Each rank stores a replicated decode token, which the restore map
        # selects once, followed by front/back prefill pieces in rank order.
        key_shards = [90, 10, 13, 90, 11, 12]
        value_shards = [900, 100, 130, 900, 110, 120]
        restore = Tensor([0, 1, 4, 5, 2], dtype="int64")
        slots = Tensor([173, 904, 50, 412, 7], dtype="int64")
        calls = []
        def gather(tensor, dim):
            calls.append(tensor.values)
            return Tensor(key_shards if len(calls) == 1 else value_shards)
        ctx = SimpleNamespace(group=SimpleNamespace(world_size=2, all_gather=gather),
            local_num_tokens=3, local_padded_tokens=3, global_num_tokens=5,
            has_global_prefill=True, global_restore_indices=restore,
            global_attn_metadata=SimpleNamespace(slot_mapping=slots))
        fake_torch = SimpleNamespace(int64="int64",
            index_select=lambda tensor, dim, indices: Tensor([tensor.values[i] for i in indices.values]))
        with patch.dict(sys.modules, {"torch": fake_torch, "vllm_hcu.v1.step5_pcp": shared}):
            key, value, actual_slots = swa.gather_step5_swa_cache_inputs(
                Tensor([90, 10, 13]), Tensor([900, 100, 130]), ctx)
        self.assertEqual(key.values, [90, 10, 11, 12, 13])
        self.assertEqual(value.values, [900, 100, 110, 120, 130])
        self.assertIs(actual_slots, slots)
        self.assertEqual(len(calls), 2)

    def test_decode_cache_writes_do_not_collect(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("replicated decode must not all-gather")
        slots = Tensor([99, 21])
        ctx = SimpleNamespace(group=SimpleNamespace(world_size=8, all_gather=forbidden),
            local_num_tokens=2, local_padded_tokens=4, global_num_tokens=2,
            has_global_prefill=False, global_attn_metadata=SimpleNamespace(slot_mapping=slots))
        with patch.dict(sys.modules, {"vllm_hcu.v1.step5_pcp": shared}):
            key, value, actual_slots = swa.gather_step5_swa_cache_inputs(
                Tensor([3, 5, 0, 0]), Tensor([7, 9, 0, 0]), ctx)
        self.assertEqual(key.values, [3, 5])
        self.assertEqual(value.values, [7, 9])
        self.assertIs(actual_slots, slots)

    def test_invalid_virtual_rows_fail_before_kernel(self):
        md = _metadata([2], [7])
        md.block_table = Tensor([0, 0], (2, 1))
        ctx = SimpleNamespace(local_num_tokens=2, local_is_padding=None)
        def forbidden(*args):
            self.fail("invalid metadata reached a kernel")
        with self.assertRaisesRegex(RuntimeError, "block rows"):
            swa.forward_step5_swa_pcp(Tensor([1, 2]), {}, {}, Tensor([0, 0]), md, ctx,
                softmax_scale=1, window_left=7, decode_kernel=forbidden, prefill_kernel=forbidden)

    def test_backend_writes_cache_once_with_original_slots(self):
        writes = []
        method = self._cache_update_method(writes)
        self_obj = SimpleNamespace(attn_type="decoder", _step5_pcp=True,
                                   use_pcp=True, pcp_world_size=8, kv_cache_dtype="fp8_e4m3")
        layer = SimpleNamespace(layer_name="model.layers.3.self_attn.attn", _k_scale=1, _v_scale=1)
        global_key, global_value, slots = Tensor([1, 2, 3]), Tensor([4, 5, 6]), Tensor([17, 2, 43])
        contexts = []
        context = object()
        def resolve(*args, **kwargs):
            contexts.append(kwargs)
            return context
        def gather(key, value, ctx):
            self.assertIs(ctx, context)
            return global_key, global_value, slots
        with patch.dict(sys.modules, {
            "vllm_hcu.model_executor.layers.attention.pcp": SimpleNamespace(in_replicated_mtp_batch=lambda: False),
            "vllm_hcu.v1.step5_pcp": SimpleNamespace(get_step5_pcp_metadata=resolve),
            "vllm_hcu.v1.attention.step5_swa_pcp": SimpleNamespace(gather_step5_swa_cache_inputs=gather),
        }):
            method(self_obj, layer, Tensor([1], (1, 1, 192)), Tensor([4], (1, 1, 192)), None, Tensor([999] * 8))
        self.assertEqual(contexts, [{"layer_name": layer.layer_name}])
        self.assertEqual(len(writes), 1)
        self.assertIs(writes[0][0], global_key)
        self.assertIs(writes[0][1], global_value)
        self.assertIs(writes[0][4], slots)

    def test_backend_missing_context_fails_except_replicated_mtp(self):
        for replicated in (False, True):
            writes = []
            method = self._cache_update_method(writes)
            self_obj = SimpleNamespace(attn_type="decoder", _step5_pcp=True,
                use_pcp=True, pcp_world_size=8, kv_cache_dtype="fp8_e4m3")
            layer = SimpleNamespace(layer_name="attn", _k_scale=1, _v_scale=1)
            key, value, slots = Tensor([1], (1, 1, 192)), Tensor([4], (1, 1, 192)), Tensor([9])
            with patch.dict(sys.modules, {
                "vllm_hcu.model_executor.layers.attention.pcp": SimpleNamespace(in_replicated_mtp_batch=lambda: replicated),
                "vllm_hcu.v1.step5_pcp": SimpleNamespace(get_step5_pcp_metadata=lambda **kwargs: None),
                "vllm_hcu.v1.attention.step5_swa_pcp": swa,
            }):
                if replicated:
                    method(self_obj, layer, key, value, None, slots)
                    self.assertEqual(len(writes), 1)
                    self.assertIs(writes[0][0], key)
                else:
                    with self.assertRaisesRegex(RuntimeError, "metadata is missing"):
                        method(self_obj, layer, key, value, None, slots)
                    self.assertEqual(writes, [])


if __name__ == "__main__":
    unittest.main()
