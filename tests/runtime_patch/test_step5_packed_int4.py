"""Packed Step5 loader tests; opt into GPU numerics with STEP5_INT4_GPU_TEST=1."""

import os
from types import SimpleNamespace
import unittest

import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm_hcu.model_executor.layers.quantization.ct_w4a8_step5 import (
    Step5CompressedW4A8MoEMethod,
    ct_int32_to_packed_int4,
)


def unpack_signed_pairs(packed):
    raw = packed.view(torch.uint8).to(torch.int16)
    pairs = torch.stack((raw >> 4, raw & 15), dim=-1).flatten(-2)
    return torch.where(pairs >= 8, pairs - 16, pairs).to(torch.int8)


def make_layer(hidden=128, intermediate=64, experts=4, tp_size=1, tp_rank=0,
               mapping=None, activation="swiglustep", device="cpu"):
    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    parallel = SimpleNamespace(tp_size=tp_size, all2all_backend="allgather_reducescatter")
    layer.moe_config = SimpleNamespace(
        activation=activation, moe_backend="auto", has_bias=False,
        moe_parallel_config=parallel, tp_rank=tp_rank, is_act_and_mul=True,
    )
    layer.quant_config = SimpleNamespace(get_name=lambda: "compressed-tensors")
    layer.global_num_experts = len(mapping) if mapping is not None else experts
    layer._expert_map = None if mapping is None else torch.tensor(mapping, dtype=torch.int32, device=device)
    layer.expert_map_manager = SimpleNamespace(
        map_global_to_local=lambda e: e if mapping is None else mapping[e]
    )
    layer.apply_router_weight_on_input = False
    method = Step5CompressedW4A8MoEMethod(layer.quant_config, layer.moe_config)
    layer.quant_method = method
    with torch.device(device):
        method.create_weights(layer, experts, hidden, intermediate // tp_size,
                              torch.bfloat16, weight_loader=layer.weight_loader)
    return layer


def load_expert(layer, expert, gate, up, down, sgate, sup, sdown):
    for name, shard, weight, scale in (
        ("w13", "w1", gate, sgate), ("w13", "w3", up, sup),
        ("w2", "w2", down, sdown),
    ):
        param = getattr(layer, name + "_weight")
        param.weight_loader(param, pack_to_int32(weight, 4), name + "_weight_packed",
                            shard, expert, return_success=True)
        param = getattr(layer, name + "_weight_scale")
        param.weight_loader(param, scale, name + "_weight_scale", shard, expert)


class PackedInt4Tests(unittest.TestCase):
    def test_all_nibbles_roundtrip_without_expanding_storage(self):
        weights = torch.arange(-8, 8, dtype=torch.int8).repeat(8, 4)
        packed = pack_to_int32(weights, 4)
        original = packed.clone()
        converted = ct_int32_to_packed_int4(packed)
        # Independent byte contract: weights (-8, -7) become nibbles 0x8, 0x9.
        self.assertEqual(converted.view(torch.uint8)[0, 0].item(), 0x89)
        torch.testing.assert_close(unpack_signed_pairs(converted), weights)
        torch.testing.assert_close(packed, original)
        self.assertEqual(converted.numel(), weights.numel() // 2)
        self.assertEqual(converted.numel() * converted.element_size(), packed.numel() * 4)
        batch = ct_int32_to_packed_int4(torch.stack((packed, packed)))
        torch.testing.assert_close(unpack_signed_pairs(batch), torch.stack((weights, weights)))

    def test_invalid_checkpoint_format_rejected(self):
        for value in (torch.zeros(8, dtype=torch.int32), torch.zeros(8, 8, dtype=torch.int8)):
            with self.assertRaises(ValueError):
                ct_int32_to_packed_int4(value)

    def test_real_loader_tp_and_ep(self):
        torch.manual_seed(91)
        hidden, size = 64, 64
        gate = torch.randint(-8, 8, (size, hidden), dtype=torch.int8)
        up = torch.randint(-8, 8, (size, hidden), dtype=torch.int8)
        down = torch.randint(-8, 8, (hidden, size), dtype=torch.int8)
        sg, su, sd = [torch.rand(n, 1) for n in (size, size, hidden)]
        for rank in (0, 1):
            with self.subTest(tp_rank=rank):
                layer = make_layer(hidden, size, experts=2, tp_size=2, tp_rank=rank,
                                   mapping=[-1, 0, -1, 1])
                load_expert(layer, 3, gate, up, down, sg, su, sd)
                sl = slice(rank * size // 2, (rank + 1) * size // 2)
                torch.testing.assert_close(
                    unpack_signed_pairs(layer.w13_weight[1]), torch.cat((gate[sl], up[sl])))
                torch.testing.assert_close(unpack_signed_pairs(layer.w2_weight[1]), down[:, sl])
                torch.testing.assert_close(layer.w13_weight_scale[1], torch.cat((sg[sl], su[sl])))
                torch.testing.assert_close(layer.w2_weight_scale[1], sd)
                param = layer.w13_weight
                before = param.clone()
                self.assertFalse(param.weight_loader(
                    param, pack_to_int32(gate, 4), "w13_weight_packed", "w1", 0, True))
                torch.testing.assert_close(param, before)
                names = dict(layer.named_parameters())
                self.assertIn("w13_weight_packed", names)
                self.assertNotIn("w13_weight", names)

    def test_postload_keeps_packed_parameters_and_original_scales(self):
        layer = make_layer()
        params = {n: getattr(layer, n) for n in (
            "w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"
        )}
        pointers = {n: p.data_ptr() for n, p in params.items()}
        for _ in range(2):
            layer.quant_method.process_weights_after_loading(layer)
        for name, param in params.items():
            self.assertIs(getattr(layer, name), param)
            self.assertEqual(param.data_ptr(), pointers[name])
        self.assertEqual(layer.w13_weight.numel() + layer.w2_weight.numel(), 4 * 3 * 64 * 128 // 2)
        self.assertNotIn("w13_weight_packed", dict(layer.named_parameters()))
        torch.testing.assert_close(layer.w13_weight_scale, torch.ones_like(layer.w13_weight_scale))

    def test_unsupported_modes_fail_instead_of_expanding(self):
        layer = make_layer()
        for attr, value in (("moe_backend", "deep_gemm"), ("activation", "gelu"), ("has_bias", True)):
            old = getattr(layer.moe_config, attr)
            setattr(layer.moe_config, attr, value)
            with self.assertRaises(ValueError):
                Step5CompressedW4A8MoEMethod(layer.quant_config, layer.moe_config)
            setattr(layer.moe_config, attr, old)

    @unittest.skipUnless(os.environ.get("STEP5_INT4_GPU_TEST") == "1", "GPU test opt-in")
    def test_gpu_packed_matches_int8_reference_and_graph_replay(self):
        from boltops.fused_moe import fused_experts
        torch.manual_seed(13)
        device = "cuda:0"
        hidden, size, experts = 128, 64, 4
        for activation in ("silu", "swiglustep"):
            for mapping in (None, [-1, 0, -1, 1]):
                local_ids = list(range(experts)) if mapping is None else [1, 3]
                layer = make_layer(hidden, size, len(local_ids), mapping=mapping,
                                   activation=activation, device=device)
                w1 = torch.randint(-8, 8, (len(local_ids), 2 * size, hidden), dtype=torch.int8)
                w2 = torch.randint(-8, 8, (len(local_ids), hidden, size), dtype=torch.int8)
                s1 = torch.rand(len(local_ids), 2 * size, 1) * .04 + .01
                s2 = torch.rand(len(local_ids), hidden, 1) * .04 + .01
                for local, global_id in enumerate(local_ids):
                    load_expert(layer, global_id, w1[local, :size], w1[local, size:],
                                w2[local], s1[local, :size], s1[local, size:], s2[local])
                layer.quant_method.process_weights_after_loading(layer)
                rw1, rw2 = w1.to(device), w2.to(device)
                for tokens in (1, 17, 64):
                    with self.subTest(activation=activation, mapping=mapping, tokens=tokens):
                        x = (torch.randn(tokens, hidden, device=device) * 3).to(torch.bfloat16)
                        ids = torch.stack((torch.arange(tokens, device=device) % experts,
                                           (torch.arange(tokens, device=device) + 1) % experts), dim=1).int()
                        weights = torch.softmax(torch.randn(tokens, 2, device=device), dim=1)
                        expected = fused_experts(
                            x, rw1, rw2, weights, ids, activation=activation,
                            use_int8_w8a8=True, per_channel_quant=True,
                            w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
                            expert_map=layer._expert_map, global_num_experts=experts,
                            gemm1_limit=7.0 if activation == "swiglustep" else None,
                        )
                        actual = layer.quant_method.apply(layer, x, weights, ids)
                        torch.cuda.synchronize()
                        torch.testing.assert_close(actual, expected, atol=.04, rtol=.02)
                        if tokens == 17 and activation == "swiglustep" and mapping is not None:
                            # Warm up separately, then verify capture uses the same packed owners.
                            stream = torch.cuda.Stream()
                            stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(stream):
                                for _ in range(2):
                                    layer.quant_method.apply(layer, x, weights, ids)
                            torch.cuda.current_stream().wait_stream(stream)
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                replay_out = layer.quant_method.apply(layer, x, weights, ids)
                            graph.replay()
                            torch.cuda.synchronize()
                            torch.testing.assert_close(replay_out, actual, atol=.04, rtol=.02)
                            self.assertEqual(layer.w13_weight.numel(), len(local_ids) * size * hidden)


if __name__ == "__main__":
    unittest.main()
