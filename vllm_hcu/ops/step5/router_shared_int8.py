# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode router plus INT8 shared gate/up and clamped SwiGLU.

The router always reads the original BF16 activation and BF16 weight, and
accumulates FP32 logits. Only the shared gate/up uses per-token symmetric INT8
activations and checkpoint INT8 weights. Integer split-K partials remain INT32
until the complete sum, so large dot products are not rounded via FP32 scratch.

Gate/up results round to BF16 before the activation, matching the materialized
W8A8 linear. The returned activation is BF16; the existing shared down projection
still performs its required input quantization and INT8 GEMM.
"""

from __future__ import annotations

import math

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .router_gemm import RSG_MAX_M

RSG_INT8_MAX_M = RSG_MAX_M
# Tuned for the Step5 TP4 decode shard K=4096, router N=352, shared I=384.
_BM, _BN, _BK, _SPLIT_K, _WARPS, _STAGES = 32, 64, 128, 4, 4, 2


@triton.jit
def _router_shared_int8_kernel(
    x_ptr, xq_ptr, xs_ptr, wr_ptr, wg_ptr, wgs_ptr,
    logits_ptr, act_ptr, partials_ptr, counters_ptr,
    M, K, sxm, swr,
    NR: tl.constexpr, I: tl.constexpr, LIMIT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_m = tl.cdiv(M, BM)
    R_TILES: tl.constexpr = tl.cdiv(NR, BN)
    pid_n = pid // num_m
    pid_m = pid % num_m
    rm = pid_m * BM + tl.arange(0, BM)
    mmask = rm < M
    k_per = K // SPLIT_K
    rk = pid_k * k_per + tl.arange(0, BK)
    tile = tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
    num_tiles = tl.num_programs(0)
    base = partials_ptr + (pid_k * num_tiles + pid) * 2 * BM * BN
    if pid_n < R_TILES:
        rn = pid_n * BN + tl.arange(0, BN)
        xp = x_ptr + rm[:, None] * sxm + rk[None, :]
        wp = wr_ptr + rk[:, None] * swr + rn[None, :]
        acc = tl.zeros((BM, BN), tl.float32)
        for _ in range(0, k_per, BK):
            a = tl.load(xp, mask=mmask[:, None], other=0.0)
            b = tl.load(wp, mask=rn[None, :] < NR, other=0.0)
            acc += tl.dot(a, b)
            xp += BK
            wp += BK * swr
        # Scratch is INT32 for exact shared GEMMs. Router FP32 bits are stored
        # unchanged in its disjoint slots and reinterpreted on reduction.
        tl.store(base + tile, acc.to(tl.int32, bitcast=True))
        tl.debug_barrier()
        done = tl.atomic_add(counters_ptr + pid, 1, sem="acq_rel", scope="gpu")
        if done == SPLIT_K - 1:
            total = tl.zeros((BM, BN), tl.float32)
            for s in tl.static_range(SPLIT_K):
                bits = tl.load(partials_ptr + (s * num_tiles + pid) * 2 * BM * BN + tile)
                total += bits.to(tl.float32, bitcast=True)
            tl.store(logits_ptr + rm[:, None] * NR + rn[None, :], total,
                     mask=mmask[:, None] & (rn[None, :] < NR))
            tl.atomic_xchg(counters_ptr + pid, 0, sem="relaxed", scope="gpu")
    else:
        rn = (pid_n - R_TILES) * BN + tl.arange(0, BN)
        nmask = rn < I
        qxp = xq_ptr + rm[:, None] * K + rk[None, :]
        gp = wg_ptr + rn[None, :] * K + rk[:, None]
        acc_g = tl.zeros((BM, BN), tl.int32)
        acc_u = tl.zeros((BM, BN), tl.int32)
        for _ in range(0, k_per, BK):
            aq = tl.load(qxp, mask=mmask[:, None], other=0)
            g = tl.load(gp, mask=nmask[None, :], other=0)
            u = tl.load(gp + I * K, mask=nmask[None, :], other=0)
            acc_g += tl.dot(aq, g)
            acc_u += tl.dot(aq, u)
            qxp += BK
            gp += BK
        tl.store(base + tile, acc_g)
        tl.store(base + BM * BN + tile, acc_u)
        tl.debug_barrier()
        done = tl.atomic_add(counters_ptr + pid, 1, sem="acq_rel", scope="gpu")
        if done == SPLIT_K - 1:
            tg = tl.zeros((BM, BN), tl.int32)
            tu = tl.zeros((BM, BN), tl.int32)
            for s in tl.static_range(SPLIT_K):
                b2 = partials_ptr + (s * num_tiles + pid) * 2 * BM * BN
                tg += tl.load(b2 + tile)
                tu += tl.load(b2 + BM * BN + tile)
            xs = tl.load(xs_ptr + rm, mask=mmask, other=0.0)
            gs = tl.load(wgs_ptr + rn, mask=nmask, other=0.0)
            us = tl.load(wgs_ptr + I + rn, mask=nmask, other=0.0)
            # Native W8A8 first multiplies the two FP32 scales. Preserve this
            # order and its BF16 rounding boundary, including halfway cases.
            gate = (tg.to(tl.float32) * (xs[:, None] * gs[None, :])).to(tl.bfloat16).to(tl.float32)
            up = (tu.to(tl.float32) * (xs[:, None] * us[None, :])).to(tl.bfloat16).to(tl.float32)
            y = tl.minimum(tl.sigmoid(gate) * gate, LIMIT) * tl.minimum(
                tl.maximum(up, -LIMIT), LIMIT
            )
            tl.store(act_ptr + rm[:, None] * I + rn[None, :], y.to(tl.bfloat16),
                     mask=mmask[:, None] & nmask[None, :])
            tl.atomic_xchg(counters_ptr + pid, 0, sem="relaxed", scope="gpu")


def make_router_shared_int8_workspace(
    device: torch.device, NR: int, I: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate per-layer split-K storage before graph capture.

    Do not share a workspace between concurrently executing layers/streams.
    Router FP32 bit patterns and shared INT32 sums occupy separate tile slots.
    """
    if NR <= 0 or I <= 0:
        raise ValueError("router and shared dimensions must be positive")
    max_m = max(RSG_INT8_MAX_M, _BM)
    tiles = triton.cdiv(max_m, _BM) * (triton.cdiv(NR, _BN) + triton.cdiv(I, _BN))
    return (
        torch.empty(_SPLIT_K * tiles * 2 * _BM * _BN,
                    device=device, dtype=torch.int32),
        torch.zeros(tiles, device=device, dtype=torch.int32),
    )


def usable_router_shared_int8(
    x: torch.Tensor, w_router: torch.Tensor, w_gate_up: torch.Tensor,
    weight_scale: torch.Tensor, limit: float,
) -> bool:
    """Static eligibility; unsupported callers keep their ordinary INT8 path."""
    return (
        RSG_INT8_MAX_M > 0
        and x.is_cuda and x.ndim == 2 and x.dtype == torch.bfloat16
        and x.is_contiguous() and 0 <= x.shape[0] <= RSG_INT8_MAX_M
        and x.shape[1] > 0 and x.shape[1] % (_BK * _SPLIT_K) == 0
        and w_router.ndim == 2 and w_router.dtype == torch.bfloat16
        and w_router.shape[0] == x.shape[1] and w_router.shape[1] > 0
        and w_router.is_contiguous()
        and w_gate_up.ndim == 2 and w_gate_up.dtype == torch.int8
        and w_gate_up.shape[0] > 0 and w_gate_up.shape[0] % 2 == 0
        and w_gate_up.shape[1] == x.shape[1] and w_gate_up.is_contiguous()
        and weight_scale.dtype == torch.float32 and weight_scale.is_contiguous()
        and weight_scale.numel() == w_gate_up.shape[0]
        and x.device == w_router.device == w_gate_up.device == weight_scale.device
        and math.isfinite(limit) and limit > 0
    )


def _router_shared_int8(
    x: torch.Tensor, w_router: torch.Tensor, w_gate_up: torch.Tensor,
    weight_scale: torch.Tensor, partials: torch.Tensor, counters: torch.Tensor,
    limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from lightop.quant import per_token_quant_int8

    M, K = x.shape
    NR, I = w_router.shape[1], w_gate_up.shape[0] // 2
    logits = torch.empty((M, NR), device=x.device, dtype=torch.float32)
    act = torch.empty((M, I), device=x.device, dtype=torch.bfloat16)
    if M == 0:
        return logits, act
    xq, xs = per_token_quant_int8(x)
    tiles = triton.cdiv(M, _BM) * (triton.cdiv(NR, _BN) + triton.cdiv(I, _BN))
    _router_shared_int8_kernel[(tiles, _SPLIT_K)](
        x, xq, xs, w_router, w_gate_up, weight_scale,
        logits, act, partials, counters, M, K, x.stride(0), w_router.stride(0),
        NR=NR, I=I, LIMIT=limit,
        BM=_BM, BN=_BN, BK=_BK, SPLIT_K=_SPLIT_K,
        num_warps=_WARPS, num_stages=_STAGES,
    )
    return logits, act


def _router_shared_int8_fake(
    x: torch.Tensor, w_router: torch.Tensor, w_gate_up: torch.Tensor,
    weight_scale: torch.Tensor, partials: torch.Tensor, counters: torch.Tensor,
    limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty((x.shape[0], w_router.shape[1]), device=x.device, dtype=torch.float32),
        torch.empty((x.shape[0], w_gate_up.shape[0] // 2), device=x.device, dtype=torch.bfloat16),
    )


direct_register_custom_op(
    op_name="step5_router_shared_int8", op_func=_router_shared_int8,
    mutates_args=["partials", "counters"], fake_impl=_router_shared_int8_fake,
)


def router_shared_swiglustep_int8(
    x: torch.Tensor, w_router: torch.Tensor, w_gate_up: torch.Tensor,
    weight_scale: torch.Tensor, limit: float,
    workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 router logits and BF16 activation from checkpoint INT8 weights.

    A model layer supplies its own workspace. Direct eager callers may omit it;
    direct graph callers must prepare it before capture, avoiding a zero-fill
    initialization on the first graph invocation.
    """
    if not usable_router_shared_int8(x, w_router, w_gate_up, weight_scale, limit):
        raise ValueError("unsupported Step5 router/shared INT8 dtype, layout or dimensions")
    NR, I = w_router.shape[1], w_gate_up.shape[0] // 2
    if workspace is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare a private router/shared INT8 workspace before capture")
        workspace = make_router_shared_int8_workspace(x.device, NR, I)
    partials, counters = workspace
    tiles = triton.cdiv(x.shape[0], _BM) * (triton.cdiv(NR, _BN) + triton.cdiv(I, _BN))
    if (
        partials.dtype != torch.int32 or counters.dtype != torch.int32
        or partials.device != x.device or counters.device != x.device
        or not partials.is_contiguous() or not counters.is_contiguous()
        or counters.numel() < tiles
        or partials.numel() < _SPLIT_K * tiles * 2 * _BM * _BN
    ):
        raise ValueError("invalid private router/shared INT8 workspace")
    return torch.ops.vllm.step5_router_shared_int8(
        x, w_router, w_gate_up, weight_scale, partials, counters, float(limit)
    )


__all__ = [
    "RSG_INT8_MAX_M", "make_router_shared_int8_workspace",
    "router_shared_swiglustep_int8", "usable_router_shared_int8",
]
