"""Dynamic per-token fp8 (e4m3) quantization as one Triton program per row.

Bit-identical to torch.ops._C.dynamic_per_token_scaled_fp8_quant without a
scale upper bound (scale = max(amax / 448, 1 / (448 * 512)), q = x / scale),
and faster at decode sizes: 3.1us vs 4.6us for 64 x 4096 bf16 (the Step5 MoE
input, quantized before the DP dispatch). Gated by VLLM_HCU_TRITON_TOKEN_FP8_QUANT.
"""

import os

import torch

from vllm.triton_utils import tl, triton
from vllm_hcu.platforms import envs as henvs

ENABLED = henvs.optional_custom_op_enabled(
    os.environ.get("VLLM_HCU_TRITON_TOKEN_FP8_QUANT", "0") == "1"
)
_MAX_K = 8192


@triton.jit
def _token_fp8_quant_kernel(x_ptr, stride_x, q_ptr, s_ptr, K: tl.constexpr, BK: tl.constexpr):
    t = tl.program_id(0)
    offs = tl.arange(0, BK)
    m = offs < K
    x = tl.load(x_ptr + t.to(tl.int64) * stride_x + offs, mask=m, other=0.0).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x), axis=0) / 448.0, 1.0 / (448.0 * 512.0))
    q = tl.minimum(tl.maximum(x / s, -448.0), 448.0).to(tl.float8e4nv)
    tl.store(q_ptr + t.to(tl.int64) * K + offs, q, mask=m)
    tl.store(s_ptr + t, s)


def can_use(x: torch.Tensor) -> bool:
    return (
        ENABLED
        and x.dim() == 2
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.stride(-1) == 1
        and x.shape[-1] <= _MAX_K
        and x.is_cuda
    )


def token_fp8_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x [M, K] -> (fp8 [M, K], fp32 scale [M, 1])."""
    M, K = x.shape
    q = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(M, 1, dtype=torch.float32, device=x.device)
    if M > 0:
        _token_fp8_quant_kernel[(M,)](
            x, x.stride(0), q, s, K=K, BK=triton.next_power_of_2(K), num_warps=8
        )
    return q, s


@triton.jit
def _masked_token_fp8_quant_kernel(
    x_ptr, q_ptr, s_ptr, counts_ptr, stride_e, stride_t,
    CAP: tl.constexpr, K: tl.constexpr, BK: tl.constexpr,
    ROW_LANES: tl.constexpr,
):
    expert = tl.program_id(0)
    count = tl.minimum(tl.maximum(tl.load(counts_ptr + expert), 0), CAP)
    offs = tl.arange(0, BK)
    for row in range(tl.program_id(1), count, ROW_LANES):
        x = tl.load(x_ptr + expert.to(tl.int64) * stride_e
                    + row.to(tl.int64) * stride_t + offs,
                    mask=offs < K, other=0.).to(tl.float32)
        # Keep the existing token_fp8_quant rounding and minimum scale.
        scale = tl.maximum(tl.max(tl.abs(x), axis=0) / 448.,
                           1. / (448. * 512.))
        q = tl.minimum(tl.maximum(x / scale, -448.), 448.).to(tl.float8e4nv)
        offset = expert.to(tl.int64) * CAP + row
        tl.store(q_ptr + offset * K + offs, q, mask=offs < K)
        tl.store(s_ptr + offset, scale)


def masked_token_fp8_quant(x: torch.Tensor, counts: torch.Tensor):
    """Quantize only live LL expert rows, with no host count readback.

    Input is BF16/FP16 [E, capacity, K]; padding in the returned buffers is
    unspecified and must only be consumed by count-masked expert kernels.
    """
    if (x.dim() != 3 or x.dtype not in (torch.bfloat16, torch.float16)
            or not x.is_cuda or x.stride(-1) != 1 or x.shape[-1] > _MAX_K
            or counts.shape != (x.shape[0],) or counts.dtype != torch.int32
            or counts.device != x.device or not counts.is_contiguous()):
        raise ValueError("Invalid Step5 LL per-token FP8 input/count layout")
    e, cap, k = x.shape
    q = torch.empty((e, cap, k), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((e, cap, 1), dtype=torch.float32, device=x.device)
    if e and cap:
        _masked_token_fp8_quant_kernel[(e, 8)](
            x, q, scales, counts, x.stride(0), x.stride(1),
            CAP=cap, K=k, BK=triton.next_power_of_2(k), ROW_LANES=8,
            num_warps=4)
    return q, scales
