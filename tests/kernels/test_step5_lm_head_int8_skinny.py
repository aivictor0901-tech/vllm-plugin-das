# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Benchmark the INT8 skinny lm_head GEMM against the two existing paths.

Same logical weight for all three, as in test_step5_lm_head_int8_vs_bf16_timing:
one INT8 tensor with per-output-channel scales, and its bf16 fold.

  bf16 torch   hipBLASLt bf16 baseline
  bf16 skinny  what the head runs today (skinny_lm_head.skinny_linear)
  int8 w8a8    hipBLASLt W8A8 (int8_runtime.apply_int8_linear)
  int8 skinny  skinny_lm_head_int8.skinny_w8a8_linear

The point of the last column: INT8 halves the weight bytes, and the skinny
schedule is what lets it keep the bf16 rate while doing so.
"""

import pytest
import torch

from vllm_hcu.model_executor.layers import skinny_lm_head, skinny_lm_head_int8
from vllm_hcu.models.step5_text import _dequantize_int8_channel

pytestmark = pytest.mark.hcu

# Step5 lm_head shard under TP8: 128896 / 8 rows, padded as vLLM pads it.
SHARD_N = 16112
HIDDEN_K = 4096
DECODE_ROWS = (1, 8, 16, 32, 64)


def _int8_weight(rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(0)
    quantized = torch.randint(
        -127, 128, (rows, cols), dtype=torch.int8, generator=generator
    )
    scale = (
        torch.rand(rows, 1, generator=generator, dtype=torch.float32) * 1e-3 + 1e-4
    )
    return quantized.cuda(), scale.cuda()


def _time_ms(run, iters: int = 20, warmup: int = 3) -> tuple[float, torch.Tensor]:
    for _ in range(warmup):
        out = run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters, out


@torch.inference_mode()
def test_install_int8_storage_replaces_the_head_weight(monkeypatch):
    """The head's bf16 weight is swapped for int8 weight + fp32 channel scale.

    Uses a stand-in module carrying the two things the real head provides: a bf16
    ``weight`` and ``weight_loader``. What matters is the shape, dtype and
    parameter attributes the vocab shard loader will see.
    """
    import torch.nn as nn

    from vllm_hcu.model_executor.layers import skinny_lm_head_int8

    # vLLM's parameter classes read the TP rank at construction time; a real
    # worker has it initialized, a bare test process does not.
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 3
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 8
    )

    rows, cols = 64, 128
    calls = []

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.tp_rank = 3
            self.tp_size = 8
            self.weight = nn.Parameter(
                torch.empty(rows, cols, dtype=torch.bfloat16), requires_grad=False
            )

        def weight_loader(self, param, loaded_weight):
            calls.append((param.shape, loaded_weight.shape))
            param.data.copy_(loaded_weight)

    head = Head()
    assert skinny_lm_head_int8.install_int8_storage(head)
    assert head.weight.dtype is torch.int8
    assert tuple(head.weight.shape) == (rows, cols)
    assert head.weight_scale.dtype is torch.float32
    assert tuple(head.weight_scale.shape) == (rows, 1)
    # The vocab shard loader shards the weight on dim 1 and the scale on dim 0.
    assert head.weight.input_dim == 1 and head.weight.output_dim == 0
    assert head.weight_scale.output_dim == 0
    assert head.weight.tp_rank == 3 and head.weight.tp_size == 8
    assert head.weight_scale.tp_rank == 3 and head.weight_scale.tp_size == 8

    assert skinny_lm_head_int8.HcuLmHeadInt8Storage.matches(head)
    # A bf16 head must not be reported as INT8 storage.
    head.weight = nn.Parameter(
        torch.empty(rows, cols, dtype=torch.bfloat16), requires_grad=False
    )
    assert not skinny_lm_head_int8.HcuLmHeadInt8Storage.matches(head)


@torch.inference_mode()
def test_skinny_w8a8_close_to_w8a8_and_bf16(capsys):
    if not torch.cuda.is_available():
        pytest.skip("requires HCU")

    quantized, scale = _int8_weight(SHARD_N, HIDDEN_K)
    folded = _dequantize_int8_channel("lm_head.weight", quantized, scale).contiguous()

    from vllm_hcu.model_executor.layers.quantization.int8_runtime import (
        apply_int8_linear,
    )

    rows = []
    for m in DECODE_ROWS:
        x = torch.randn(m, HIDDEN_K, device="cuda", dtype=torch.bfloat16) * 0.1
        reference = torch.nn.functional.linear(x, folded)

        assert skinny_lm_head.usable(x, folded)
        assert skinny_lm_head_int8.usable_int8(x, quantized, scale.reshape(-1))

        baseline_ms, _ = _time_ms(lambda: torch.nn.functional.linear(x, folded))
        skinny_ms, skinny_out = _time_ms(
            lambda: skinny_lm_head.skinny_linear(x, folded)
        )
        w8a8_ms, w8a8_out = _time_ms(
            lambda: apply_int8_linear(x, quantized, scale, torch.bfloat16)
        )
        i8skinny_ms, i8skinny_out = _time_ms(
            lambda: skinny_lm_head_int8.skinny_w8a8_linear(
                x, quantized, scale.reshape(-1)
            )
        )

        for name, out in (
            ("bf16 skinny", skinny_out),
            ("int8 w8a8", w8a8_out),
            ("int8 skinny", i8skinny_out),
        ):
            assert torch.isfinite(out).all(), f"{name} produced non-finite output"

        # The two INT8 paths must agree: same activation quantization, same
        # integer weight, only the accumulation order differs.
        torch.testing.assert_close(
            i8skinny_out, w8a8_out, rtol=2e-2, atol=2e-2
        )
        # Against the folded head only the quantization of the input and of the
        # weight differ, which is the checkpoint's own ~1% budget.
        torch.testing.assert_close(
            i8skinny_out, reference, rtol=5e-2, atol=5e-2
        )

        def err(out):
            return ((out - reference).abs().amax() / reference.abs().amax().clamp_min(1e-6)).item()

        rows.append(
            (
                m,
                baseline_ms,
                skinny_ms,
                w8a8_ms,
                i8skinny_ms,
                err(i8skinny_out),
                quantized.numel(),
                folded.numel() * 2,
            )
        )

    with capsys.disabled():
        print(
            f"\nlm_head shard {SHARD_N} x {HIDDEN_K} "
            f"(bf16 {SHARD_N * HIDDEN_K * 2 / 2**20:.0f} MiB, "
            f"int8 {SHARD_N * HIDDEN_K / 2**20:.0f} MiB)"
        )
        print(
            f"{'M':>4} {'bf16 torch':>11} {'bf16 skinny':>12} {'int8 w8a8':>11} "
            f"{'int8 skinny':>12} {'bf16 TB/s':>10} {'i8w8a8 TB/s':>12} "
            f"{'i8skin TB/s':>12} {'speedup':>8} {'relerr':>9}"
        )
        for m, base, skin, w8, i8, e, i8bytes, bf16bytes in rows:
            print(
                f"{m:>4} {base * 1e3:>9.1f}us {skin * 1e3:>10.1f}us "
                f"{w8 * 1e3:>9.1f}us {i8 * 1e3:>10.1f}us "
                f"{bf16bytes / (skin * 1e-3) / 1e12:>10.2f} "
                f"{i8bytes / (w8 * 1e-3) / 1e12:>12.2f} "
                f"{i8bytes / (i8 * 1e-3) / 1e12:>12.2f} "
                f"{skin / i8:>7.2f}x {e:>9.2e}"
            )
