# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx936 Step5 SWA: E4M3FN KV storage, BF16 dot, FP32 online softmax.

Logical cache layout is [page, token, kv_head, 192], with explicit strides.
Each program handles a small query tile and reads only its sliding window.
This does not materialize a BF16 copy of the complete KV cache.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm_hcu.v1.attention.backends.swa_decode import _fp8e4m3_to_bf16


@triton.jit
def _swa_prefill_kernel(
    Q, K, V, O, CU, LEN, BT,
    scale,
    sqt, sqh, skb, skt, skh, svb, svt, svh, sot, soh, sbt,
    GROUP: tl.constexpr, PAGE: tl.constexpr, WINDOW: tl.constexpr,
    QT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    request = tl.program_id(0)
    K = K.to(tl.pointer_type(tl.uint8))
    V = V.to(tl.pointer_type(tl.uint8))
    kv_head = tl.program_id(1)
    tile = tl.program_id(2)
    q_start = tl.load(CU + request)
    q_len = tl.load(CU + request + 1) - q_start
    first = tile * QT
    if first >= q_len:
        return
    seq_len = tl.load(LEN + request)
    rows = tl.arange(0, BM)
    tok = first + rows // GROUP
    head = kv_head * GROUP + rows % GROUP
    valid_row = tok < q_len
    pos = seq_len - q_len + tok
    d0 = tl.arange(0, 128)
    d1 = 128 + tl.arange(0, 64)
    qp = Q + (q_start + tok)[:, None] * sqt + head[:, None] * sqh
    qa = tl.load(qp + d0[None, :], valid_row[:, None], other=0)
    qb = tl.load(qp + d1[None, :], valid_row[:, None], other=0)
    acc0 = tl.zeros((BM, 128), tl.float32)
    acc1 = tl.zeros((BM, 64), tl.float32)
    m = tl.full((BM,), float('-inf'), tl.float32)
    den = tl.zeros((BM,), tl.float32)
    start = tl.maximum(seq_len - q_len + first - WINDOW, 0)
    start = start // BN * BN
    end = tl.minimum(seq_len - q_len + first + QT, seq_len)
    ns = tl.arange(0, BN)
    for base in range(start, end, BN):
        n = base + ns
        # BN divides PAGE; one physical page serves the entire key tile.
        page = tl.load(BT + request * sbt + base // PAGE).to(tl.int64)
        kp = K + page * skb + (n % PAGE)[None, :] * skt + kv_head * skh
        vp = V + page * svb + (n % PAGE)[:, None] * svt + kv_head * svh
        ka = _fp8e4m3_to_bf16(tl.load(kp + d0[:, None], (n < seq_len)[None, :], other=0))
        kb = _fp8e4m3_to_bf16(tl.load(kp + d1[:, None], (n < seq_len)[None, :], other=0))
        va = _fp8e4m3_to_bf16(tl.load(vp + d0[None, :], (n < seq_len)[:, None], other=0))
        vb = _fp8e4m3_to_bf16(tl.load(vp + d1[None, :], (n < seq_len)[:, None], other=0))
        scores = (tl.dot(qa, ka) + tl.dot(qb, kb)) * (scale * 1.4426950408889634)
        valid = valid_row[:, None] & (n[None, :] <= pos[:, None])
        valid &= n[None, :] >= pos[:, None] - WINDOW
        valid &= n[None, :] < seq_len
        scores = tl.where(valid, scores, float('-inf'))
        new_m = tl.maximum(m, tl.max(scores, 1))
        safe_m = tl.where(new_m == float('-inf'), 0., new_m)
        alpha = tl.exp2(m - safe_m)
        p = tl.exp2(scores - safe_m[:, None])
        den = den * alpha + tl.sum(p, 1)
        pb = p.to(tl.bfloat16)
        acc0 = acc0 * alpha[:, None] + tl.dot(pb, va)
        acc1 = acc1 * alpha[:, None] + tl.dot(pb, vb)
        m = new_m
    inv = 1. / tl.where(den > 0, den, 1.)
    op = O + (q_start + tok)[:, None] * sot + head[:, None] * soh
    tl.store(op + d0[None, :], acc0 * inv[:, None], valid_row[:, None])
    tl.store(op + d1[None, :], acc1 * inv[:, None], valid_row[:, None])


def swa_prefill_fp8_d192(q, k, v, out, cu_seqlens_q, seqused_k,
                         block_table, max_seqlen_q, softmax_scale, window_left):
    """Causal SWA with unit KV scales; also supports mixed prefill/decode."""
    group = q.shape[1] // k.shape[2]
    if not (
        q.dtype == out.dtype == torch.bfloat16
        and k.dtype == v.dtype == torch.float8_e4m3fn
        and q.shape == out.shape and k.shape == v.shape
        and q.shape[-1] == k.shape[-1] == 192
        and q.shape[1] % k.shape[2] == 0
        and group == 16
        and k.shape[1] in (64, 128)
        and all(x.stride(-1) == 1 for x in (q, k, v, out))
        and window_left >= 0
    ):
        raise ValueError('gfx936 E4M3 SWA requires BF16 Q/output, D192, '
                         'GQA16, page64/128 and unit KV scales')
    if q.shape[0] == 0:
        return out
    # Keep a fixed 64-row dot tile; do not scale LDS use with prompt length.
    qt = 64 // group
    _swa_prefill_kernel[(cu_seqlens_q.numel() - 1, k.shape[2],
                         triton.cdiv(max_seqlen_q, qt))](
        q, k, v, out, cu_seqlens_q, seqused_k, block_table, softmax_scale,
        q.stride(0), q.stride(1), *k.stride()[:3], *v.stride()[:3],
        out.stride(0), out.stride(1), block_table.stride(0),
        GROUP=group, PAGE=k.shape[1], WINDOW=window_left,
        QT=qt, BM=64, BN=32, num_warps=4, num_stages=1,
    )
    return out
