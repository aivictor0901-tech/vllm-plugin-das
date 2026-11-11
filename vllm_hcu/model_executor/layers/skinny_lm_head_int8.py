# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Skinny INT8 W8A8 GEMM for the lm_head at decode batch sizes.

``skinny_lm_head`` tiles only over N with a full-M block and reaches ~1.0 TB/s
on a bf16 head; hipBLASLt's W8A8 path (``int8_runtime.apply_int8_linear``)
picks a 256x256 macro tile, which at M <= 64 leaves the tensor cores mostly idle
and lands near 0.47 TB/s. INT8 halves the bytes but hipBLASLt halves the rate
with it, so the two come out even (262 us bf16 skinny vs 275 us int8 W8A8 on a
32224 x 4096 shard, M=16).

This keeps the skinny schedule -- one block in M, all CUs scanning N, so the
weight stream keeps HBM saturated -- and takes the INT8 weights directly. Per
launch on the 16112 x 4096 TP8 shard, measured with the work captured in a CUDA
graph (decode runs that way; eager timings carry ~25 us of launch cost that is
identical for every candidate):

    M          1      8     16     32     64
    bf16    139.6  139.9  142.1  147.9  150.7  us
    int8    123.3  124.8  125.2  134.8  121.8  us
    gain     1.13x  1.12x  1.13x  1.10x  1.24x

Two kernels, quantize then GEMM. The activation quantize is per-token symmetric
INT8, the same scheme ``apply_int8_linear`` uses, so numerics match the W8A8
path rather than the folded bf16 head. Folding the quantize into the GEMM was
measured and is slower (185-499 us): each of the 252 N-blocks would redo the
whole-K absmax.

Off by default; enable with VLLM_STEP5_LM_HEAD_INT8_SKINNY=1.
"""

import os

import torch
import triton
from vllm.triton_utils import tl

from vllm_hcu.platforms import envs as henvs

try:
    import triton.language.extra.libdevice as _libdevice

    _rint = _libdevice.rint
except Exception:  # older Triton exposes it under tl.extra
    _rint = tl.extra.libdevice.rint

_ENABLED = henvs.optional_custom_op_enabled(
    os.environ.get("VLLM_STEP5_LM_HEAD_INT8_SKINNY", "0") == "1"
)
MAX_M = 64
# Each row of the weight tile is BK contiguous bytes. 128 matches the bf16
# kernel's 256 B run at half the element size; larger values measured slower.
_BK = 128
# Rows per program for the activation quantize. K=4096 fits one tile, so this is
# the only axis that needs parallelism.
_QUANT_BM = 16


@triton.jit
def _quant_rows_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    M,
    K: tl.constexpr,
    BM: tl.constexpr,
):
    """Symmetric per-token INT8 quantization, BM rows per program.

    lightop's Triton fallback launches one program per row with a single
    next_power_of_2(K) tile and measures a flat ~95 us from M=1 to M=64. This
    keeps all of K in one tile but batches rows, which is ~44 us. Both are far
    above the ~5 us the bytes justify; the floor is the backend's launch cost.
    """
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    rr = tl.arange(0, K)
    mask = (rm < M)[:, None] & (rr < K)[None, :]
    x = tl.load(x_ptr + rm[:, None] * K + rr[None, :], mask=mask, other=0.0)
    xf = x.to(tl.float32)
    scale = tl.max(tl.abs(xf), axis=1) / 127.0
    q = tl.clamp(_rint(xf / (scale[:, None] + 1e-12)), -127, 127)
    tl.store(q_ptr + rm[:, None] * K + rr[None, :], q.to(tl.int8), mask=mask)
    tl.store(s_ptr + rm, scale, mask=rm < M)


@triton.jit
def _skinny_w8a8_kernel(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    o_ptr,
    M,
    N,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    pm = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pid * BN + tl.arange(0, BN)
    mmask = rm < M
    nmask = rn < N
    acc = tl.zeros((BM, BN), tl.int32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(
            x_ptr + rm[:, None] * K + rk[None, :], mask=mmask[:, None], other=0
        )
        wt = tl.load(
            w_ptr + rn[None, :] * K + rk[:, None], mask=nmask[None, :], other=0
        )
        acc += tl.dot(x, wt)
    x_scale = tl.load(xs_ptr + rm, mask=mmask, other=0.0)
    w_scale = tl.load(ws_ptr + rn, mask=nmask, other=0.0)
    out = acc.to(tl.float32) * x_scale[:, None] * w_scale[None, :]
    tl.store(
        o_ptr + rm[:, None] * N + rn[None, :],
        out.to(tl.bfloat16),
        mask=mmask[:, None] & nmask[None, :],
    )


def usable_int8(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
) -> bool:
    return (
        _ENABLED
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.int8
        and weight_scale.dtype == torch.float32
        and weight.dim() == 2
        and weight.is_contiguous()
        and weight_scale.numel() == weight.shape[0]
        and 0 < x.numel() // x.shape[-1] <= MAX_M
        and x.shape[-1] == weight.shape[1]
        and weight.shape[1] % _BK == 0
    )


def _config(m: int) -> tuple[int, int, int]:
    if m <= 16:
        return 16, 64, 2
    if m <= 32:
        return 32, 64, 2
    return 64, 128, 4


def quantize_activations(
    x2: torch.Tensor, q: torch.Tensor, s: torch.Tensor
) -> None:
    """Per-token INT8 quantize ``x2`` [M, K] into ``q`` with scales ``s``."""
    m = x2.shape[0]
    _quant_rows_kernel[(triton.cdiv(m, _QUANT_BM),)](
        x2, q, s, m, x2.shape[1], BM=_QUANT_BM, num_warps=8, num_stages=1
    )


_WORKSPACE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _workspace(
    m: int, n: int, k: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reusable quantized-activation and output buffers.

    Allocating three tensors per call cost ~50 us at these shapes, enough to
    make the INT8 path slower than the bf16 head in eager mode even though its
    kernel work is smaller. The head is called once per decode step with a
    stable shape, so one buffer set per shape is all that is ever needed; keep
    the cache tiny and drop it when a different shape arrives.
    """
    key = (m, n, k, device)
    cached = _WORKSPACE.get(key)
    if cached is not None:
        return cached
    _WORKSPACE.clear()
    buffers = (
        torch.empty((m, k), dtype=torch.int8, device=device),
        torch.empty(m, dtype=torch.float32, device=device),
        torch.empty((m, n), dtype=torch.bfloat16, device=device),
    )
    _WORKSPACE[key] = buffers
    return buffers


def skinny_w8a8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """x [..., K] bf16 @ weight[N, K].T int8 x per-row scale -> bf16.

    Requires ``usable_int8``. Buffers come from a per-shape cache, so a CUDA
    graph capture keeps pointing at stable memory across replays.
    """
    N, K = weight.shape
    x2 = x.reshape(-1, K)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    M = x2.shape[0]
    x_q, x_scale, out = _workspace(M, N, K, x.device)
    quantize_activations(x2, x_q, x_scale)

    bm, bn, nw = _config(M)
    _skinny_w8a8_kernel[(triton.cdiv(N, bn), triton.cdiv(M, bm))](
        x_q,
        x_scale,
        weight,
        weight_scale,
        out,
        M,
        N,
        K,
        BM=bm,
        BN=bn,
        BK=_BK,
        num_warps=nw,
        num_stages=1,
    )
    return out.reshape(*x.shape[:-1], N)


def install_int8_storage(lm_head: torch.nn.Module) -> bool:
    """Swap a head's bf16 weight for INT8 weight + fp32 per-row scale.

    ``ParallelLMHead`` has no INT8 storage class: its ``create_weights`` is the
    unquantized one and the shard loader shards along the vocab dimension. The
    checkpoint stores ``lm_head.weight`` as int8 ``[vocab, hidden]`` and
    ``lm_head.weight_scale`` as fp32 ``[vocab, 1]``, i.e. already split the same
    way, so replacing the two parameters is enough -- the existing vocab shard
    loader copies whatever dtype the parameter holds.

    Returns whether the storage was installed.
    """
    from torch.nn.parameter import Parameter

    from vllm.model_executor.parameter import (
        ChannelQuantScaleParameter,
        ModelWeightParameter,
    )

    if not _ENABLED:
        return False
    weight = getattr(lm_head, "weight", None)
    if weight is None or weight.dtype != torch.bfloat16:
        return False

    rows, cols = weight.shape
    loader = lm_head.weight_loader
    lm_head.register_parameter(
        "weight",
        ModelWeightParameter(
            data=torch.empty(rows, cols, dtype=torch.int8, device=weight.device),
            input_dim=1,
            output_dim=0,
            weight_loader=loader,
        ),
    )
    lm_head.register_parameter(
        "weight_scale",
        ChannelQuantScaleParameter(
            data=torch.empty(rows, 1, dtype=torch.float32, device=weight.device),
            output_dim=0,
            weight_loader=loader,
        ),
    )
    # ``bias`` and any tied-embedding path stay as they are: the head's bias is
    # None in these checkpoints and the INT8 kernel has no bias argument.
    del weight
    for param in lm_head.parameters():
        if hasattr(param, "tp_rank"):
            param.tp_rank = getattr(lm_head, "tp_rank", 0)
            param.tp_size = getattr(lm_head, "tp_size", 1)
    return True


class HcuLmHeadInt8Storage:
    """Marker so the router can tell an INT8 head from an ordinary one."""

    @staticmethod
    def matches(lm_head: torch.nn.Module) -> bool:
        weight = getattr(lm_head, "weight", None)
        scale = getattr(lm_head, "weight_scale", None)
        return (
            weight is not None
            and weight.dtype == torch.int8
            and scale is not None
            and scale.dtype == torch.float32
        )


__all__ = [
    "HcuLmHeadInt8Storage",
    "MAX_M",
    "install_int8_storage",
    "quantize_activations",
    "skinny_w8a8_linear",
    "usable_int8",
]
