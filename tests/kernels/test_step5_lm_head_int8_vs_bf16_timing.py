# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""lm_head at decode: the bf16 skinny GEMM against the INT8 W8A8 path.

MIXED_DENSE_INT8 folds the mixed checkpoint's INT8 lm_head back to bf16
(``_fold_int8_lm_head``) so the head keeps running through the bf16 skinny
kernel. This compares the two candidates on the real TP8 shard shape with the
*same logical weight*: one stored bf16 after folding, one kept INT8 and run
through hipBLASLt W8A8.

Both are weight-stream bound at these shapes, so the interesting number is
effective bandwidth over the weight bytes -- INT8 moves half as many bytes,
bf16 reaches a higher fraction of peak. The timing test prints instead of
asserting a budget; runner contention makes a threshold here noise.
"""

import pytest
import torch

from vllm_hcu.model_executor.layers import skinny_lm_head
from vllm_hcu.models.step5_text import (
    _dequantize_int8_channel,
    _fold_int8_dense,
    _fold_int8_lm_head,
)

pytestmark = pytest.mark.hcu

# Step5 lm_head shard under TP8, as in skinny_lm_head.py's own measurements.
SHARD_N = 32224
HIDDEN_K = 4096
# Decode batch sizes the skinny kernel is selected for (usable() caps at 64).
DECODE_ROWS = (1, 8, 16, 32, 64)


def _int8_weight(rows: int, cols: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel INT8 weight, as the checkpoint stores it."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    quantized = torch.randint(
        -127, 128, (rows, cols), dtype=torch.int8, generator=generator
    )
    # Channel scales of the magnitude a 4096-wide bf16 head quantizes to.
    scale = (
        torch.rand(rows, 1, generator=generator, dtype=torch.float32) * 1e-3 + 1e-4
    )
    return quantized.to(device), scale.to(device)


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
def test_fold_int8_lm_head_matches_channel_dequantization():
    """The fold is order-independent and reproduces the fp32 reference."""
    quantized, scale = _int8_weight(64, 256, "cpu")
    expected = (quantized.to(torch.float32) * scale).to(torch.bfloat16)

    for order in (
        [("lm_head.weight", quantized), ("lm_head.weight_scale", scale)],
        [("lm_head.weight_scale", scale), ("lm_head.weight", quantized)],
    ):
        folded = dict(_fold_int8_lm_head(iter(order)))
        assert set(folded) == {"lm_head.weight"}
        assert folded["lm_head.weight"].dtype == torch.bfloat16
        assert torch.equal(folded["lm_head.weight"], expected)

    # A bf16 head (the fp8 checkpoints) must stream through untouched.
    plain = torch.zeros(8, 16, dtype=torch.bfloat16)
    passed = dict(_fold_int8_lm_head(iter([("lm_head.weight", plain)])))
    assert passed["lm_head.weight"] is plain


@torch.inference_mode()
def test_fold_int8_dense_folds_projections_and_skips_packed_experts():
    """The dequant path folds dense INT8 and leaves packed experts alone."""
    quantized, scale = _int8_weight(32, 128, "cpu")
    expected = (quantized.to(torch.float32) * scale).to(torch.bfloat16)

    attention = "model.layers.3.self_attn.q_proj"
    head = "lm_head"
    # An INT4 expert pair: INT32 weight_packed plus its own scale.
    expert = "model.layers.3.moe.gate_proj"
    packed = torch.zeros(4, 8, 2, dtype=torch.int32)
    expert_scale = torch.ones(4, 8, 1, dtype=torch.float32)
    norm = torch.ones(16, dtype=torch.bfloat16)

    stream = [
        (f"{attention}.weight", quantized),
        (f"{attention}.weight_scale", scale),
        ("model.layers.3.input_layernorm.weight", norm),
        (f"{expert}.weight_packed", packed),
        (f"{expert}.weight_scale", expert_scale),
        (f"{head}.weight", quantized),
        (f"{head}.weight_scale", scale),
    ]
    folded = dict(_fold_int8_dense(iter(stream)))

    assert torch.equal(folded[f"{attention}.weight"], expected)
    assert torch.equal(folded[f"{head}.weight"], expected)
    assert f"{attention}.weight_scale" not in folded
    assert f"{head}.weight_scale" not in folded
    # The expert pair and the norm must reach the loader unchanged.
    assert folded[f"{expert}.weight_packed"] is packed
    assert folded[f"{expert}.weight_scale"] is expert_scale
    assert folded["model.layers.3.input_layernorm.weight"] is norm


@torch.inference_mode()
def test_lm_head_int8_vs_bf16_timing(capsys):
    if not torch.cuda.is_available():
        pytest.skip("requires HCU")

    quantized, scale = _int8_weight(SHARD_N, HIDDEN_K, "cuda")
    # Exactly the weight _fold_int8_lm_head hands to ParallelLMHead.
    folded = _dequantize_int8_channel("lm_head.weight", quantized, scale)
    assert folded.dtype == torch.bfloat16
    folded = folded.contiguous()

    weight_mib = SHARD_N * HIDDEN_K
    rows = []
    for m in DECODE_ROWS:
        x = torch.randn(m, HIDDEN_K, device="cuda", dtype=torch.bfloat16) * 0.1
        assert skinny_lm_head.usable(x, folded), (
            f"the skinny kernel must accept the folded head at M={m}"
        )

        reference = torch.nn.functional.linear(x, folded)
        baseline_ms, _ = _time_ms(lambda: torch.nn.functional.linear(x, folded))
        skinny_ms, skinny_out = _time_ms(
            lambda: skinny_lm_head.skinny_linear(x, folded)
        )
        assert torch.isfinite(skinny_out).all()
        # fp32 accumulate in both, so only the bf16 output rounding differs.
        torch.testing.assert_close(
            skinny_out, reference, rtol=2e-2, atol=2e-2
        )

        int8_ms = None
        int8_error = None
        try:
            from vllm_hcu.model_executor.layers.quantization.int8_runtime import (
                apply_int8_linear,
            )

            int8_ms, int8_out = _time_ms(
                lambda: apply_int8_linear(x, quantized, scale, torch.bfloat16)
            )
            assert torch.isfinite(int8_out).all()
            scale_of = reference.abs().amax().clamp_min(1e-6)
            int8_error = ((int8_out - reference).abs().amax() / scale_of).item()
        except Exception as exc:  # lightop W8A8 unavailable on this runner
            int8_note = f"{type(exc).__name__}: {exc}"
        else:
            int8_note = ""

        rows.append(
            (m, baseline_ms, skinny_ms, int8_ms, int8_error, int8_note, weight_mib)
        )

    with capsys.disabled():
        print(
            f"\nlm_head shard {SHARD_N} x {HIDDEN_K} "
            f"(bf16 weight {weight_mib * 2 / 2**20:.0f} MiB, "
            f"int8 weight {weight_mib / 2**20:.0f} MiB)"
        )
        print(
            f"{'M':>4} {'bf16 torch':>12} {'bf16 skinny':>12} {'int8 w8a8':>12} "
            f"{'skinny TB/s':>12} {'int8 TB/s':>12} {'int8 relerr':>12}"
        )
        for m, baseline_ms, skinny_ms, int8_ms, int8_error, note, nk in rows:
            skinny_tbs = nk * 2 / (skinny_ms * 1e-3) / 1e12
            int8_cell = "n/a" if int8_ms is None else f"{int8_ms * 1e3:.1f}us"
            int8_tbs = "n/a" if int8_ms is None else f"{nk / (int8_ms * 1e-3) / 1e12:.2f}"
            err_cell = "n/a" if int8_error is None else f"{int8_error:.2e}"
            print(
                f"{m:>4} {baseline_ms * 1e3:>10.1f}us {skinny_ms * 1e3:>10.1f}us "
                f"{int8_cell:>12} {skinny_tbs:>12.2f} {int8_tbs:>12} {err_cell:>12}"
            )
            if note:
                print(f"     int8 path unavailable: {note}")
