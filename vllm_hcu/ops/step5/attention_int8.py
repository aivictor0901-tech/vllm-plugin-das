# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 compiled attention gate and dynamic INT8 input quantization.

The production Inductor gate computes sigmoid and multiplication in FP32, then
writes BF16 before LightOp quantizes the row. Keep that BF16 rounding boundary
inside the fused kernel. Eager PyTorch additionally rounds sigmoid to BF16 and
is a different reference; the model must retain its original eager path.

This operation produces only an INT8 activation and its FP32 per-token scale.
The owner still executes the existing native hipBLASLt o_proj and TP reduction.
"""
from __future__ import annotations

import torch
from triton.language.extra import libdevice

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


ATTN_INT8_MAX_M = 1024


@triton.jit
def _gate_quant_int8_kernel(
    x_ptr, gate_ptr, xq_ptr, xs_ptr,
    stride_gt,
    NH: tl.constexpr, D: tl.constexpr, B: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    width: tl.constexpr = NH * D
    # The measured 3072/1536 widths have three power-of-two segments. Match
    # LightOp's reduction and division order, including the 1e-10 amax floor.
    x0 = tl.load(x_ptr + row * width + col).to(tl.float32)
    x1 = tl.load(x_ptr + row * width + B + col).to(tl.float32)
    x2 = tl.load(x_ptr + row * width + 2 * B + col).to(tl.float32)
    g0 = tl.load(gate_ptr + row * stride_gt + col // D).to(tl.float32)
    g1 = tl.load(gate_ptr + row * stride_gt + (B + col) // D).to(tl.float32)
    g2 = tl.load(gate_ptr + row * stride_gt + (2 * B + col) // D).to(tl.float32)
    y0 = (x0 * tl.sigmoid(g0)).to(tl.bfloat16).to(tl.float32)
    y1 = (x1 * tl.sigmoid(g1)).to(tl.bfloat16).to(tl.float32)
    y2 = (x2 * tl.sigmoid(g2)).to(tl.bfloat16).to(tl.float32)
    am = tl.maximum(tl.maximum(tl.abs(y0), tl.abs(y1)), tl.abs(y2))
    amax = tl.maximum(tl.max(am, 0), 1e-10)
    inv = 127 / amax
    tl.store(xq_ptr + row * width + col, libdevice.nearbyint(y0 * inv).to(tl.int8))
    tl.store(xq_ptr + row * width + B + col, libdevice.nearbyint(y1 * inv).to(tl.int8))
    tl.store(xq_ptr + row * width + 2 * B + col, libdevice.nearbyint(y2 * inv).to(tl.int8))
    tl.store(xs_ptr + row, amax / 127)


def usable_gate_quant_int8(x: torch.Tensor, gate: torch.Tensor) -> bool:
    """Static metadata guard for the measured TP4/TP8 BF16 attention shards."""
    return (
        x.is_cuda and x.ndim == 2 and x.dtype == torch.bfloat16
        and x.is_contiguous() and 0 <= x.shape[0] <= ATTN_INT8_MAX_M
        and gate.ndim == 2 and gate.dtype == torch.bfloat16
        and gate.shape[0] == x.shape[0] and gate.shape[1] in (8, 16)
        and x.shape[1] == gate.shape[1] * 192
        and gate.stride(1) == 1 and gate.stride(0) >= gate.shape[1]
        and gate.device == x.device
    )


def _gate_quant_int8(x: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xq = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    xs = torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device)
    if x.shape[0] != 0:
        _gate_quant_int8_kernel[(x.shape[0],)](
            x, gate, xq, xs, gate.stride(0),
            NH=gate.shape[1], D=192, B=x.shape[1] // 3,
            num_warps=2, num_stages=1,
        )
    return xq, xs


def _gate_quant_int8_fake(x: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty(x.shape, dtype=torch.int8, device=x.device),
        torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device),
    )


direct_register_custom_op(
    op_name="step5_attention_gate_quant_int8", op_func=_gate_quant_int8,
    mutates_args=[], fake_impl=_gate_quant_int8_fake,
)


def gate_quant_int8(x: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the compiled BF16 gate followed by LightOp per-token INT8 quant.

    Unsupported owners must use their ordinary path rather than silently
    changing precision, shape or quantization scheme. The model additionally
    guards the symmetric dynamic INT8 o_proj provider and compiled execution.
    """
    if not usable_gate_quant_int8(x, gate):
        raise ValueError("unsupported Step5 attention INT8 gate dtype, layout or dimensions")
    return torch.ops.vllm.step5_attention_gate_quant_int8(x, gate)


__all__ = ["ATTN_INT8_MAX_M", "gate_quant_int8", "usable_gate_quant_int8"]
