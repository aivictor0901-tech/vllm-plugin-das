# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 main-attention Q/K GemmaRMSNorm and cached RoPE in one launch.

This operator consumes local heads, independently of TP/DP/EP topology. It
does not communicate, update the KV cache, or touch sparse-indexer metadata.
The compiled Step5 path keeps the normalized values in FP32 through RoPE;
``round_norm`` also provides the explicit intermediate cast for parity tests
and callers which preserve eager precision boundaries.
"""

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _attention_norm_rope_head(
    X, W, C, P, O, token, head, stride_x, stride_o, stride_c, stride_p,
    eps, D: tl.constexpr, R: tl.constexpr, NEOX: tl.constexpr,
    ROUND_NORM: tl.constexpr, B: tl.constexpr,
):
    d = tl.arange(0, B)
    x = tl.load(X + token * stride_x + head * D + d, d < D, 0).to(tl.float32)
    w = tl.load(W + d, d < D, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + eps)
    y = (x * inv) * (w + 1.0)
    if ROUND_NORM:
        y = y.to(O.dtype.element_ty).to(tl.float32)
    if R > 0:
        if NEOX:
            first = d < R // 2
            partner = tl.where(first, d + R // 2, d - R // 2)
            angle = d % (R // 2)
        else:
            first = d % 2 == 0
            partner = d ^ 1
            angle = d // 2
        # Gather from registers; do not reload the projection or statistics.
        other = tl.gather(y, tl.where(d < R, partner, 0), axis=0)
        pos = tl.load(P + token * stride_p).to(tl.int64)
        cos = tl.load(C + pos * stride_c + angle, d < R, 0).to(tl.float32)
        sin = tl.load(C + pos * stride_c + R // 2 + angle, d < R, 0).to(tl.float32)
        rotated = y * cos + tl.where(first, -other, other) * sin
        y = tl.where(d < R, rotated, y)
    tl.store(O + token * stride_o + head * D + d, y, d < D)


@triton.jit
def _attention_qk_norm_rope_kernel(
    Q, K, QW, KW, C, P, QO, KO, sq, sk, so_q, so_k, sc, sp,
    eq, ek, NQ: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    NEOX: tl.constexpr, ROUND_NORM: tl.constexpr, B: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    if head < NQ:
        _attention_norm_rope_head(
            Q, QW, C, P, QO, token, head, sq, so_q, sc, sp, eq,
            D, R, NEOX, ROUND_NORM, B,
        )
    else:
        _attention_norm_rope_head(
            K, KW, C, P, KO, token, head - NQ, sk, so_k, sc, sp, ek,
            D, R, NEOX, ROUND_NORM, B,
        )


def _fake(
    q: torch.Tensor, k: torch.Tensor, q_weight: torch.Tensor,
    k_weight: torch.Tensor, cache: torch.Tensor, positions: torch.Tensor,
    head_dim: int, rotary_dim: int, q_eps: float, k_eps: float,
    is_neox_style: bool = True, round_norm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty(q.shape, device=q.device, dtype=q.dtype), torch.empty(
        k.shape, device=k.device, dtype=k.dtype
    )


def attention_qk_norm_rope(
    q: torch.Tensor, k: torch.Tensor, q_weight: torch.Tensor,
    k_weight: torch.Tensor, cache: torch.Tensor, positions: torch.Tensor,
    head_dim: int, rotary_dim: int, q_eps: float, k_eps: float,
    is_neox_style: bool = True, round_norm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Q/K are row-strided [tokens, local_heads * head_dim], without mutation."""
    if not tensors_supported(q, k, q_weight, k_weight, cache, positions,
                             head_dim, rotary_dim):
        raise ValueError("Unsupported Step5 Q/K norm+RoPE tensor layout")
    qo, ko = _fake(q, k, q_weight, k_weight, cache, positions, head_dim,
                   rotary_dim, q_eps, k_eps, is_neox_style, round_norm)
    if q.shape[0] == 0:
        return qo, ko
    nq, nk = q.shape[1] // head_dim, k.shape[1] // head_dim
    _attention_qk_norm_rope_kernel[(q.shape[0], nq + nk)](
        q, k, q_weight, k_weight, cache, positions, qo, ko,
        q.stride(0), k.stride(0), qo.stride(0), ko.stride(0),
        cache.stride(0), positions.stride(0), q_eps, k_eps,
        NQ=nq, D=head_dim, R=rotary_dim, NEOX=is_neox_style,
        ROUND_NORM=round_norm, B=triton.next_power_of_2(head_dim),
        enable_fp_fusion=False,
    )
    return qo, ko


def tensors_supported(q, k, qw, kw, cache, positions, d, r) -> bool:
    """Only layout/semantic guards; head counts need not be multiples of a tile."""
    return (
        d > 0 and 0 <= r <= d and r % 2 == 0
        and q.ndim == k.ndim == 2 and q.shape[0] == k.shape[0]
        and q.shape[1] > 0 and k.shape[1] > 0
        and q.shape[1] % d == k.shape[1] % d == 0
        and q.stride(1) == k.stride(1) == 1
        and q.dtype == k.dtype and q.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and qw.ndim == kw.ndim == 1 and qw.shape[0] == kw.shape[0] == d
        and qw.stride(0) == kw.stride(0) == 1
        and qw.dtype == kw.dtype == torch.float32
        and cache.ndim == 2 and cache.shape[1] == r and cache.stride(1) == 1
        and cache.dtype == q.dtype
        and positions.ndim == 1 and positions.shape[0] == q.shape[0]
        and positions.dtype in (torch.int32, torch.int64)
        and all(t.device == q.device for t in (k, qw, kw, cache, positions))
        and q.is_cuda
    )


def can_use(module, q, k, positions) -> bool:
    rope = module.rotary_emb
    # The replacement targets the native operations fused by Inductor. Keep
    # eager/different precision policies on the original framework path.
    if not torch.compiler.is_compiling():
        return False
    from torch._inductor import config as inductor_config

    if inductor_config.emulate_precision_casts:
        return False
    return (
        module.use_rope
        and hasattr(rope, "cos_sin_cache")
        and hasattr(rope, "is_neox_style")
        and tensors_supported(
            q, k, module.q_norm.weight, module.k_norm.weight,
            rope.cos_sin_cache, positions, module.head_dim, rope.rotary_dim,
        )
    )


direct_register_custom_op(
    op_name="step5_attention_qk_norm_rope",
    op_func=attention_qk_norm_rope,
    mutates_args=[],
    fake_impl=_fake,
)
