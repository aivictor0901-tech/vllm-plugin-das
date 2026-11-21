# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Decode lm_head W8A8 GEMM for M <= 64, using checkpoint INT8 weights.

Per-token symmetric INT8 quantization precedes the GEMM so its whole-K absmax
is computed once, matching the native W8A8 input contract. Enable with
VLLM_STEP5_LM_HEAD_INT8_SKINNY=1; other shapes use the native W8A8 fallback.
"""

import os
from collections import OrderedDict

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
# Keep weight reads contiguous and bound quantization register pressure.
_BK = 256
_QUANT_BK = 1024


@triton.jit
def _quant_rows_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    M,
    K: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    """Find each row's absmax, then quantize with a bounded [BM, BK] tile.

    Two passes avoid holding the whole [BM, K] FP32 activation in registers.
    """
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    mmask = rm < M
    amax = tl.zeros((BM,), tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(
            x_ptr + rm[:, None] * K + rk[None, :],
            mask=mmask[:, None] & (rk[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x), axis=1))
    # Match LightOp's public per_token_quant_int8 exactly, including the
    # nonzero scale for zero/tiny rows and its multiplication order.
    amax = tl.maximum(amax, 1e-10)
    scale = amax / 127.0
    tl.store(s_ptr + rm, scale, mask=mmask)
    inv = 127.0 / amax
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        xf = tl.load(
            x_ptr + rm[:, None] * K + rk[None, :],
            mask=mmask[:, None] & (rk[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        q = tl.clamp(_rint(xf * inv[:, None]), -127, 127)
        tl.store(
            q_ptr + rm[:, None] * K + rk[None, :],
            q.to(tl.int8),
            mask=mmask[:, None] & (rk[None, :] < K),
        )


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
        and x.ndim >= 2
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.int8
        and weight_scale.dtype == torch.float32
        and weight.dim() == 2
        and weight.is_contiguous()
        and weight.device == x.device == weight_scale.device
        and weight_scale.is_contiguous()
        and weight_scale.numel() == weight.shape[0]
        and 0 < x.numel() // x.shape[-1] <= MAX_M
        and x.shape[-1] == weight.shape[1]
        and weight.shape[1] % _BK == 0
    )


def _config(m: int) -> tuple[int, int, int]:
    """Return the tuned (BM, BN, num_warps) for the decode batch size."""
    if m <= 16:
        return 16, 128, 2
    if m <= 32:
        return 32, 128, 2
    return 64, 128, 4


def quantize_activations(
    x2: torch.Tensor, q: torch.Tensor, s: torch.Tensor
) -> None:
    """Per-token INT8 quantize ``x2`` [M, K] into ``q`` with scales ``s``."""
    m = x2.shape[0]
    # BM=4 with 4 warps was within 1 us of the best config at every M, so one
    # shape serves the whole decode range instead of a per-M table.
    _quant_rows_kernel[(triton.cdiv(m, 4),)](
        x2, q, s, m, x2.shape[1], BM=4, BK=_QUANT_BK, num_warps=4, num_stages=1
    )


_WORKSPACE: OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
_MAX_WORKSPACES = 16


def _workspace(
    m: int, n: int, k: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reuse eager scratch only; each invocation owns its output.

    Captures allocate scratch from their graph pool, so different graphs and
    different heads cannot alias one global workspace when replayed on other
    streams. Eager calls may share scratch only on the same CUDA stream, where
    the quantize/GEMM pairs execute in order. Keeping output out of this cache
    also preserves ordinary functional semantics when a caller retains logits.
    """
    cached = None
    if not torch.cuda.is_current_stream_capturing():
        key = (device, torch.cuda.current_stream(device).cuda_stream, m, k)
        cached = _WORKSPACE.get(key)
        if cached is None:
            cached = (
                torch.empty((m, k), dtype=torch.int8, device=device),
                torch.empty(m, dtype=torch.float32, device=device),
            )
            _WORKSPACE[key] = cached
            if len(_WORKSPACE) > _MAX_WORKSPACES:
                _WORKSPACE.popitem(last=False)
        else:
            _WORKSPACE.move_to_end(key)
    if cached is None:
        cached = (
            torch.empty((m, k), dtype=torch.int8, device=device),
            torch.empty(m, dtype=torch.float32, device=device),
        )
    out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    return cached[0], cached[1], out


def skinny_w8a8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """x [..., K] bf16 @ weight[N, K].T int8 x per-row scale -> bf16.

    Requires ``usable_int8``. Graph captures own their buffers; eager calls
    reuse stream-local quantization scratch and always return distinct logits.
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
    lm_head._step5_int8_storage = True
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
