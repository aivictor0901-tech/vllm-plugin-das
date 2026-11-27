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
import os

import torch
from triton.language.extra import libdevice

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .router_gemm import RSG_MAX_M

RSG_INT8_MAX_M = RSG_MAX_M
# Tuned for the Step5 TP4 decode shard K=4096, router N=352, shared I=384.
_BM, _BN, _BK, _SPLIT_K, _WARPS, _STAGES = 32, 64, 128, 4, 4, 2


def _shared_block_m(rows: int, width: int, router_tp_size: int = 1) -> int:
    """Opt-in candidate for full shared weights on small token-owner batches.

    Keep I384 on BM32: its native LightOp scratch consumer has a fixed layout.
    This runs inside the real-shape runtime ops, not only at Dynamo warmup.
    """
    if (os.getenv("HCU_STEP5_REPLICATED_SHARED_RS_AG", "0") == "1"
            and os.getenv("HCU_STEP5_SHARED_SMALL_M_BM16", "0") == "1"
            and width == 1536 and router_tp_size == 1 and 0 < rows <= 16):
        from vllm_hcu.model_executor.layers.fused_moe.step5_replicated_shared import owner_execution_active
        if owner_execution_active():
            return 16
    return _BM


@triton.jit
def _router_shared_int8_kernel(
    x_ptr, xq_ptr, xs_ptr, wr_ptr, wg_ptr, wgs_ptr,
    logits_ptr, act_ptr, partials_ptr, counters_ptr,
    M, K, sxm, swr,
    NR: tl.constexpr, I: tl.constexpr, LIMIT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    DEFER_SHARED: tl.constexpr = False,
    ROUTER_TP_SIZE: tl.constexpr = 1, ROUTER_TP_RANK: tl.constexpr = 0,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_m = tl.cdiv(M, BM)
    R_TILES: tl.constexpr = tl.cdiv(NR, BN)
    owner_rows = tl.cdiv(M, ROUTER_TP_SIZE)
    owner_start = ROUTER_TP_RANK * owner_rows
    if ROUTER_TP_SIZE == 1:
        pid_n = pid // num_m
        pid_m = pid % num_m
        rm = pid_m * BM + tl.arange(0, BM)
        mmask = rm < M
        num_tiles = tl.num_programs(0)
    else:
        owner_m = tl.cdiv(owner_rows, BM)
        router_tiles = R_TILES * owner_m
        if pid < router_tiles:
            pid_n = pid // owner_m
            pid_m = pid % owner_m
            rm = owner_start + pid_m * BM + tl.arange(0, BM)
            mmask = (rm < M) & (rm < owner_start + owner_rows)
        else:
            shared_pid = pid - router_tiles
            pid_n = R_TILES + shared_pid // num_m
            pid_m = shared_pid % num_m
            rm = pid_m * BM + tl.arange(0, BM)
            mmask = rm < M
        # The shared epilogue indexes the original, full-row split-K layout.
        pid = pid_n * num_m + pid_m
        num_tiles = num_m * (R_TILES + tl.cdiv(I, BN))
    k_per = K // SPLIT_K
    rk = pid_k * k_per + tl.arange(0, BK)
    tile = tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
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
            local_row = rm - owner_start
            if ROUTER_TP_SIZE > 1:
                # Match the zeros from padding before the former rank slice.
                total = tl.where(mmask[:, None], total, 0.)
            tl.store(logits_ptr + local_row[:, None] * NR + rn[None, :], total,
                     mask=(local_row < owner_rows)[:, None] & (rn[None, :] < NR))
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
        if not DEFER_SHARED:
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
    xq: torch.Tensor | None = None, xs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from lightop.quant import per_token_quant_int8

    M, K = x.shape
    NR, I = w_router.shape[1], w_gate_up.shape[0] // 2
    logits = torch.empty((M, NR), device=x.device, dtype=torch.float32)
    act = torch.empty((M, I), device=x.device, dtype=torch.bfloat16)
    if M == 0:
        return logits, act
    if xq is None:
        xq, xs = per_token_quant_int8(x)
    bm = _shared_block_m(M, I)
    tiles = triton.cdiv(M, bm) * (triton.cdiv(NR, _BN) + triton.cdiv(I, _BN))
    _router_shared_int8_kernel[(tiles, _SPLIT_K)](
        x, xq, xs, w_router, w_gate_up, weight_scale,
        logits, act, partials, counters, M, K, x.stride(0), w_router.stride(0),
        NR=NR, I=I, LIMIT=limit,
        BM=bm, BN=_BN, BK=_BK, SPLIT_K=_SPLIT_K,
        num_warps=2 if bm == 16 else _WARPS, num_stages=_STAGES,
    )
    return logits, act


def _router_shared_int8_fake(
    x: torch.Tensor, w_router: torch.Tensor, w_gate_up: torch.Tensor,
    weight_scale: torch.Tensor, partials: torch.Tensor, counters: torch.Tensor,
    limit: float,
    xq: torch.Tensor | None = None, xs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty((x.shape[0], w_router.shape[1]), device=x.device, dtype=torch.float32),
        torch.empty((x.shape[0], w_gate_up.shape[0] // 2), device=x.device, dtype=torch.bfloat16),
    )


direct_register_custom_op(
    op_name="step5_router_shared_int8", op_func=_router_shared_int8,
    mutates_args=["partials", "counters"], fake_impl=_router_shared_int8_fake,
)


@triton.jit
def _shared_int8_wide_epilogue_kernel(
    source, xscale, wscale, output, scales,
    M: tl.constexpr, I: tl.constexpr, NR: tl.constexpr, LIMIT: tl.constexpr,
    FROM_PARTIALS: tl.constexpr, B: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, SPLIT_K: tl.constexpr,
):
    """TP1 full-row epilogue; preserve both BF16 rounding boundaries."""
    row = tl.program_id(0)
    col = tl.arange(0, B)
    mask = col < I
    if FROM_PARTIALS:
        num_m: tl.constexpr = (M + BM - 1) // BM
        tiles: tl.constexpr = num_m * ((NR + BN - 1) // BN + (I + BN - 1) // BN)
        pid = (tl.cdiv(NR, BN) + col // BN) * num_m + row // BM
        offset = pid * 2 * BM * BN + (row % BM) * BN + col % BN
        gate_sum = tl.full((B,), 0, tl.int32)
        up_sum = tl.full((B,), 0, tl.int32)
        for split in tl.static_range(SPLIT_K):
            base = source + split * tiles * 2 * BM * BN + offset
            gate_sum += tl.load(base, mask, other=0)
            up_sum += tl.load(base + BM * BN, mask, other=0)
        xs = tl.load(xscale + row)
        gs = tl.load(wscale + col, mask, other=0.)
        us = tl.load(wscale + I + col, mask, other=0.)
        gate = (gate_sum.to(tl.float32) * (xs * gs)).to(tl.bfloat16).to(tl.float32)
        up = (up_sum.to(tl.float32) * (xs * us)).to(tl.bfloat16).to(tl.float32)
    else:
        gate = tl.load(source + row * 2 * I + col, mask, other=0.).to(tl.float32)
        up = tl.load(source + row * 2 * I + I + col, mask, other=0.).to(tl.float32)
    value = (tl.minimum(tl.sigmoid(gate) * gate, LIMIT)
             * tl.minimum(tl.maximum(up, -LIMIT), LIMIT)).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.where(mask, tl.abs(value), 0.), 0), 1e-10)
    quant = libdevice.nearbyint(value * (127. / amax)).to(tl.int8)
    tl.store(output + row * I + col, quant, mask)
    tl.store(scales + row, amax / 127.)


def _shared_int8_shape(input, xscale, wscale):
    if input.dtype == torch.int32:
        if xscale is None or wscale is None:
            raise ValueError("shared split-K epilogue requires input and weight scales")
        rows, width = xscale.shape[0], wscale.numel() // 2
    else:
        if input.dtype != torch.bfloat16 or input.ndim != 2:
            raise ValueError("shared epilogue requires BF16 gate/up or INT32 split-K input")
        rows, width = input.shape[0], input.shape[1] // 2
    if width not in (384, 1536):
        raise ValueError("shared INT8 epilogue supports TP4 (384) and TP1 (1536)")
    return rows, width


def _shared_int8_epilogue(
    input: torch.Tensor, xscale: torch.Tensor | None = None,
    wscale: torch.Tensor | None = None, router_columns: int = 352, limit: float = 7.0,
    block_m: int = 32, prefill_dense: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = _shared_int8_shape(input, xscale, wscale)
    if prefill_dense and (
        width != 1536 or input.dtype != torch.bfloat16 or not input.is_cuda
        or xscale is not None or wscale is not None
    ):
        raise ValueError("Step5 dense Prefill epilogue requires BF16 GPU gate/up")
    if block_m not in (16, 32) or (width == 384 and block_m != 32):
        raise ValueError("shared epilogue requires BM32 for I384 or BM16/BM32 for I1536")
    if width == 384:
        # Keep the validated TP4 native operation and its launch unchanged.
        from lightop.moe import shared_gateup_int8_epilogue
        return shared_gateup_int8_epilogue(input, xscale, wscale, router_columns, limit)
    if not input.is_contiguous():
        raise ValueError("shared epilogue requires contiguous input")
    if input.dtype == torch.int32:
        tiles = triton.cdiv(rows, block_m) * (triton.cdiv(router_columns, _BN)
                                        + triton.cdiv(width, _BN))
        if (input.numel() < _SPLIT_K * tiles * 2 * block_m * _BN
                or xscale.shape != (rows, 1) or wscale.numel() != 2 * width
                or xscale.dtype != torch.float32 or wscale.dtype != torch.float32
                or not xscale.is_contiguous() or not wscale.is_contiguous()
                or xscale.device != input.device or wscale.device != input.device):
            raise ValueError("invalid shared split-K workspace or scales")
    elif input.shape[1] != 2 * width:
        raise ValueError("invalid shared gate/up width")
    q = input.new_empty((rows, width), dtype=torch.int8)
    scale = input.new_empty((rows, 1), dtype=torch.float32)
    if rows and prefill_dense:
        from vllm_hcu.model_executor.layers.fused_moe.step5_w4a8_kernels import (
            activation_persistent,
        )

        activation_persistent[(min(triton.cdiv(rows, 4), 1024),)](
            input, q, scale, None, ROWS=rows, I=width, BM=4, CH=width // 3,
            DENSE=True, LIMIT=limit, num_warps=4, num_stages=1,
        )
    elif rows:
        _shared_int8_wide_epilogue_kernel[(rows,)](
            input, xscale, wscale, q, scale, M=rows, I=width,
            NR=router_columns, LIMIT=limit, FROM_PARTIALS=input.dtype == torch.int32,
            B=triton.next_power_of_2(width), BM=block_m, BN=_BN, SPLIT_K=_SPLIT_K,
            num_warps=4, num_stages=1,
        )
    return q, scale


def _shared_int8_epilogue_fake(
    input, xscale=None, wscale=None, router_columns=352, limit=7.0,
    block_m=32, prefill_dense=False,
):
    rows, width = _shared_int8_shape(input, xscale, wscale)
    if block_m not in (16, 32) or (width == 384 and block_m != 32):
        raise ValueError("shared epilogue requires BM32 for I384 or BM16/BM32 for I1536")
    return (
        input.new_empty((rows, width), dtype=torch.int8),
        input.new_empty((rows, 1), dtype=torch.float32),
    )


direct_register_custom_op(
    op_name="step5_shared_int8_epilogue", op_func=_shared_int8_epilogue,
    mutates_args=[], fake_impl=_shared_int8_epilogue_fake,
)


def _router_shared_int8_quant(
    x: torch.Tensor, w_router: torch.Tensor, w_gate_up: torch.Tensor,
    weight_scale: torch.Tensor, partials: torch.Tensor, counters: torch.Tensor,
    limit: float, xq: torch.Tensor | None = None, xs: torch.Tensor | None = None,
    router_tp_size: int = 1, router_tp_rank: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from lightop.quant import per_token_quant_int8

    m, k = x.shape
    nr = w_router.shape[1]
    width = w_gate_up.shape[0] // 2
    if not 0 <= router_tp_rank < router_tp_size:
        raise ValueError("invalid router TP rank or size")
    owner_rows = triton.cdiv(m, router_tp_size)
    logits = x.new_empty((owner_rows, nr), dtype=torch.float32)
    if m == 0:
        q = x.new_empty((m, width), dtype=torch.int8)
        scale = x.new_empty((m, 1), dtype=torch.float32)
        return logits, q, scale
    if xq is None:
        xq, xs = per_token_quant_int8(x)
    bm = _shared_block_m(m, width, router_tp_size)
    tiles = (triton.cdiv(owner_rows, bm) * triton.cdiv(nr, _BN)
             + triton.cdiv(m, bm) * triton.cdiv(width, _BN))
    _router_shared_int8_kernel[(tiles, _SPLIT_K)](
        x, xq, xs, w_router, w_gate_up, weight_scale,
        logits, x, partials, counters, m, k, x.stride(0), w_router.stride(0),
        NR=nr, I=width, LIMIT=limit, BM=bm, BN=_BN, BK=_BK, SPLIT_K=_SPLIT_K,
        DEFER_SHARED=True, ROUTER_TP_SIZE=router_tp_size, ROUTER_TP_RANK=router_tp_rank,
        num_warps=2 if bm == 16 else _WARPS, num_stages=_STAGES,
    )
    q, scale = torch.ops.vllm.step5_shared_int8_epilogue(
        partials, xs, weight_scale, nr, limit, bm,
    )
    return logits, q, scale


def _router_shared_int8_quant_fake(
    x, w_router, w_gate_up, weight_scale, partials, counters, limit, xq=None, xs=None,
    router_tp_size=1, router_tp_rank=0,
):
    owner_rows = (x.shape[0] + router_tp_size - 1) // router_tp_size
    return (
        x.new_empty((owner_rows, w_router.shape[1]), dtype=torch.float32),
        x.new_empty((x.shape[0], w_gate_up.shape[0] // 2), dtype=torch.int8),
        x.new_empty((x.shape[0], 1), dtype=torch.float32),
    )


direct_register_custom_op(
    op_name="step5_router_shared_int8_quant", op_func=_router_shared_int8_quant,
    mutates_args=["partials", "counters"], fake_impl=_router_shared_int8_quant_fake,
)


def router_shared_swiglustep_int8(
    x: torch.Tensor, w_router: torch.Tensor, w_gate_up: torch.Tensor,
    weight_scale: torch.Tensor, limit: float,
    workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
    x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None,
    quantized_output: bool = False,
    router_tp_size: int = 1, router_tp_rank: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
    """Return FP32 router logits and shared activation from checkpoint INT8 weights.

    The default activation is BF16. With quantized_output=True, return its
    INT8 payload and full-row FP32 scale after preserving BF16 rounding.

    A model layer supplies its own workspace. Direct eager callers may omit it;
    direct graph callers must prepare it before capture, avoiding a zero-fill
    initialization on the first graph invocation.
    """
    if not usable_router_shared_int8(x, w_router, w_gate_up, weight_scale, limit):
        raise ValueError("unsupported Step5 router/shared INT8 dtype, layout or dimensions")
    xq, xs = (None, None) if x_and_scale_quanted is None else x_and_scale_quanted
    if xq is not None and (
        xq.shape != x.shape or xq.dtype != torch.int8
        or xs is None or xs.shape != (x.shape[0], 1) or xs.dtype != torch.float32
        or xq.device != x.device or xs.device != x.device
        or not xq.is_contiguous() or not xs.is_contiguous()
    ):
        raise ValueError("invalid prequantized router/shared INT8 input")
    if (xq is None) != (xs is None):
        raise ValueError("prequantized router/shared input requires both payload and scale")
    NR, I = w_router.shape[1], w_gate_up.shape[0] // 2
    if workspace is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare a private router/shared INT8 workspace before capture")
        workspace = make_router_shared_int8_workspace(x.device, NR, I)
    partials, counters = workspace
    bm = _shared_block_m(x.shape[0], I, router_tp_size)
    tiles = triton.cdiv(x.shape[0], bm) * (triton.cdiv(NR, _BN) + triton.cdiv(I, _BN))
    if (
        partials.dtype != torch.int32 or counters.dtype != torch.int32
        or partials.device != x.device or counters.device != x.device
        or not partials.is_contiguous() or not counters.is_contiguous()
        or counters.numel() < tiles
        or partials.numel() < _SPLIT_K * tiles * 2 * bm * _BN
    ):
        raise ValueError("invalid private router/shared INT8 workspace")
    if not quantized_output and (router_tp_size, router_tp_rank) != (1, 0):
        raise ValueError("sharded router logits require the quantized shared epilogue")
    if quantized_output:
        if I not in (384, 1536):
            raise ValueError("shared INT8 epilogue requires a TP4 or TP1 shard")
        logits, q, scale = torch.ops.vllm.step5_router_shared_int8_quant(
            x, w_router, w_gate_up, weight_scale, partials, counters, float(limit), xq, xs,
            router_tp_size, router_tp_rank,
        )
        return logits, (q, scale)
    return torch.ops.vllm.step5_router_shared_int8(
        x, w_router, w_gate_up, weight_scale, partials, counters, float(limit), xq, xs
    )


__all__ = [
    "RSG_INT8_MAX_M", "make_router_shared_int8_workspace",
    "router_shared_swiglustep_int8", "usable_router_shared_int8",
]
