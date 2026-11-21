"""Opt-in Step5 W4A8 backend for the existing modular EP/DP execution path.

Selected by the normal SlimQuant method only for the explicit Step5 profile.
"""

from functools import lru_cache
import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from vllm.logger import init_logger

from vllm_hcu.runtime_compat.step5 import padded_route_rows, static_agrs_local_experts

logger = init_logger(__name__)


@triton.jit
def activation_masked(
    x,
    q,
    scale,
    counts,
    CAP: tl.constexpr,
    I: tl.constexpr,
    INT8: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr = 8,
):
    """Existing Step5 activation math over valid expert rows only."""
    expert = tl.program_id(0)
    for row in range(tl.program_id(1), tl.minimum(tl.load(counts + expert), CAP), ROWS):
        col = tl.arange(0, BLOCK)
        offset = (expert.to(tl.int64) * CAP + row) * 2 * I + col
        gate = tl.load(x + offset, mask=col < I, other=0).to(tl.float32)
        up = tl.load(x + offset + I, mask=col < I, other=0).to(tl.float32)
        value = (
            (
                tl.minimum(gate * tl.sigmoid(gate), 7.0)
                * tl.minimum(tl.maximum(up, -7.0), 7.0)
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        maximum = tl.max(tl.abs(value), 0)
        s = tl.where(maximum > 0, maximum / (127.0 if INT8 else 448.0), 1.0)
        v = value / s
        if INT8:
            v = tl.minimum(tl.maximum(libdevice.nearbyint(v), -127.0), 127.0)
        tl.store(q + (expert.to(tl.int64) * CAP + row) * I + col, v, mask=col < I)
        tl.store(scale + expert * CAP + row, s)


@triton.jit
def _row_value(ptr, row, col, mask, I: tl.constexpr):
    offset = row[:, None].to(tl.int64) * (2 * I) + col[None, :]
    gate = tl.load(ptr + offset, mask=mask, other=0).to(tl.float32)
    up = tl.load(ptr + offset + I, mask=mask, other=0).to(tl.float32)
    return (
        (
            tl.minimum(gate * tl.sigmoid(gate), 7.0)
            * tl.minimum(tl.maximum(up, -7.0), 7.0)
        )
        .to(tl.bfloat16)
        .to(tl.float32)
    )


@triton.jit
def activation_persistent(
    gateup,
    quant,
    scales,
    experts,
    ROWS: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    CH: tl.constexpr,
):
    tl.static_assert(I == 3 * CH)
    col = tl.arange(0, CH)
    step = tl.num_programs(0) * BM
    for start in range(tl.program_id(0) * BM, ROWS, step):
        row = start + tl.arange(0, BM)
        inside = row < ROWS
        valid = inside & (tl.load(experts + row, mask=inside, other=-1) >= 0)
        if tl.sum(valid.to(tl.int32), 0) > 0:
            mask = valid[:, None]
            y0 = _row_value(gateup, row, col, mask, I)
            y1 = _row_value(gateup, row, col + CH, mask, I)
            y2 = _row_value(gateup, row, col + 2 * CH, mask, I)
            maximum = tl.maximum(
                tl.maximum(tl.max(tl.abs(y0), 1), tl.max(tl.abs(y1), 1)),
                tl.max(tl.abs(y2), 1),
            )
            scale = tl.where(maximum > 0, maximum / 127.0, 1.0)
            offset = row[:, None].to(tl.int64) * I + col[None, :]
            q0 = tl.minimum(
                tl.maximum(libdevice.nearbyint(y0 / scale[:, None]), -127.0), 127.0
            ).to(tl.int8)
            q1 = tl.minimum(
                tl.maximum(libdevice.nearbyint(y1 / scale[:, None]), -127.0), 127.0
            ).to(tl.int8)
            q2 = tl.minimum(
                tl.maximum(libdevice.nearbyint(y2 / scale[:, None]), -127.0), 127.0
            ).to(tl.int8)
            tl.store(quant + offset, q0, mask=mask)
            tl.store(quant + offset + CH, q1, mask=mask)
            tl.store(quant + offset + 2 * CH, q2, mask=mask)
            tl.store(scales + row, tl.where(valid, scale, 1.0), mask=inside)
        else:
            tl.store(scales + row, 1.0, mask=inside)


@triton.jit
def _small_route_meta(
    ids,
    mapping,
    inverse,
    expert_ids,
    ROUTES: tl.constexpr,
    GLOBAL_E: tl.constexpr,
    E: tl.constexpr,
    ALIGN: tl.constexpr,
    BOUND: tl.constexpr,
    BLOCK_R: tl.constexpr,
    EBINS: tl.constexpr,
    BLOCK_BOUND: tl.constexpr,
):
    expert = tl.program_id(0)
    r = tl.arange(0, BLOCK_R)
    inside = r < ROUTES
    gid = tl.load(ids + r, mask=inside, other=-1)
    local = tl.load(
        mapping + gid, mask=inside & (gid >= 0) & (gid < GLOBAL_E), other=-1
    )
    local = tl.where((local >= 0) & (local < E), local, E)
    counts = tl.histogram(local, EBINS)
    e = tl.arange(0, EBINS)
    padded = tl.where(e < E, tl.cdiv(counts, ALIGN) * ALIGN, 0)
    if expert < E:
        start = tl.sum(tl.where(e < expert, padded, 0), 0)
        n = tl.sum(tl.where(e == expert, counts, 0), 0)
        npad = tl.cdiv(n, ALIGN) * ALIGN
        mine = inside & (local == expert)
        position = start + tl.cumsum(mine.to(tl.int32), 0) - 1
        tl.store(inverse + r, position, mask=mine)
        tl.store(expert_ids + position, expert, mask=mine)
        j = tl.arange(0, ALIGN)
        pad = start + n + j
        tl.store(expert_ids + pad, -1, mask=n + j < npad)
    else:
        tl.store(inverse + r, -1, mask=inside & (local == E))
        total = tl.sum(padded, 0)
        tail = tl.arange(0, BLOCK_BOUND)
        tl.store(expert_ids + tail, -1, mask=(tail >= total) & (tail < BOUND))


@triton.jit
def _small_route_gather(
    a,
    scale,
    inverse,
    out,
    out_scale,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    route = tl.program_id(0)
    position = tl.load(inverse + route)
    if position >= 0:
        token = route // TOPK
        col = tl.arange(0, BLOCK)
        value = tl.load(a + token.to(tl.int64) * K + col, mask=col < K, other=0)
        tl.store(out + position.to(tl.int64) * K + col, value, mask=col < K)
        tl.store(out_scale + position, tl.load(scale + token * SCALE_STRIDE))


def small_permute(
    a, scale, ids, mapping, experts, out, global_experts=352, alignment=64
):
    routes = ids.numel()
    bound = padded_route_rows(ids.shape[0], ids.shape[1], experts, alignment)
    assert out.shape == (bound, a.shape[1]) and 0 < routes <= 2048
    indices = torch.empty((bound,), device=a.device, dtype=torch.int32)
    inverse = torch.empty_like(ids)
    scales = torch.empty((bound, 1), device=a.device, dtype=torch.float32)
    _small_route_meta[(experts + 1,)](
        ids,
        mapping,
        inverse,
        indices,
        routes,
        global_experts,
        experts,
        alignment,
        bound,
        triton.next_power_of_2(routes),
        triton.next_power_of_2(experts + 1),
        triton.next_power_of_2(bound),
        num_warps=1 if routes <= 256 else 8,
    )
    _small_route_gather[(routes,)](
        a,
        scale,
        inverse,
        out,
        scales,
        a.shape[1],
        ids.shape[1],
        scale.stride(0),
        triton.next_power_of_2(a.shape[1]),
        num_warps=4,
    )
    return out, scales, indices, inverse, alignment


@triton.jit
def _activation_row(gateup, quant, scales, row, I: tl.constexpr, BLOCK: tl.constexpr):
    col = tl.arange(0, BLOCK)
    g = tl.load(gateup + row * (2 * I) + col, col < I, other=0).to(tl.float32)
    u = tl.load(gateup + row * (2 * I) + I + col, col < I, other=0).to(tl.float32)
    y = (
        (tl.minimum(g * tl.sigmoid(g), 7.0) * tl.minimum(tl.maximum(u, -7.0), 7.0))
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    maximum = tl.max(tl.where(col < I, tl.abs(y), 0), 0)
    scale = tl.where(maximum > 0, maximum / 127.0, 1.0)
    q = tl.minimum(tl.maximum(libdevice.nearbyint(y / scale), -127.0), 127.0).to(
        tl.int8
    )
    tl.store(quant + row * I + col, q, col < I)
    tl.store(scales + row, scale)


@triton.jit
def _activation(gateup, quant, scales, experts, I: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    if tl.load(experts + row) >= 0:
        _activation_row(gateup, quant, scales, row, I, BLOCK)
    else:
        tl.store(scales + row, 1.0)


@triton.jit
def _activation_routes(
    gateup,
    quant,
    scales,
    inverse,
    experts,
    ROWS: tl.constexpr,
    ROUTES: tl.constexpr,
    I: tl.constexpr,
    BLOCK: tl.constexpr,
    PAD_BLOCK: tl.constexpr,
):
    route = tl.program_id(0)
    if route < ROUTES:
        row = tl.load(inverse + route).to(tl.int32)
        if row >= 0:
            _activation_row(gateup, quant, scales, row, I, BLOCK)
    else:
        # This CTA writes only invalid rows, disjoint from every valid-route
        # CTA. Preserve the scale=1 ABI without an additional fill kernel.
        for start in range(tl.cdiv(ROWS, PAD_BLOCK)):
            pad_row = start * PAD_BLOCK + tl.arange(0, PAD_BLOCK)
            invalid = tl.load(experts + pad_row, pad_row < ROWS, other=0) < 0
            tl.store(scales + pad_row, 1.0, (pad_row < ROWS) & invalid)


@triton.jit
def _weighted_output(
    a,
    inverse,
    weights,
    mapping,
    ids,
    output,
    shared,
    M: tl.constexpr,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    GLOBAL_E: tl.constexpr,
    HAS_MAP: tl.constexpr,
    ADD_SHARED: tl.constexpr,
    SHARED_START: tl.constexpr,
    SHARED_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for slot in tl.static_range(0, TOPK):
        e = tl.load(ids + row * TOPK + slot)
        valid = (e >= 0) & (e < GLOBAL_E)
        if HAS_MAP:
            local = tl.load(mapping + e, mask=valid, other=-1)
            valid = valid & (local >= 0)
        pos = tl.load(inverse + row * TOPK + slot, mask=valid, other=-1)
        # Padded all-local65536x8 routes can exceed2^31 elements at H4096.
        # Promote before multiplying, not after an overflowing int32 product.
        value = tl.load(
            a + pos.to(tl.int64) * K + cols,
            mask=valid & (pos >= 0) & (cols < K),
            other=0,
        ).to(tl.float32)
        # DeepEP may leave inactive Top-K slots unspecified. Mask the weight
        # as well as the value: 0 * NaN would otherwise poison a valid row.
        w = tl.load(weights + row * TOPK + slot, mask=valid & (pos >= 0), other=0.0)
        acc += value * w
    # Match the separate BF16 routed-output/shared-add rounding boundary.
    rounded = acc.to(output.dtype.element_ty)
    if ADD_SHARED:
        own = (row >= SHARED_START) & (row < SHARED_START + SHARED_ROWS)
        value = tl.load(
            shared + (row - SHARED_START) * K + cols, mask=own & (cols < K), other=0
        ).to(tl.float32)
        rounded = (rounded.to(tl.float32) + value).to(output.dtype.element_ty)
    tl.store(output + row * K + cols, rounded, cols < K)


@triton.jit
def _histogram(
    ids,
    mapping,
    partial,
    indices,
    inverse,
    R: tl.constexpr,
    E: tl.constexpr,
    G: tl.constexpr,
    BOUND: tl.constexpr,
    NB: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
):
    group = tl.program_id(0)
    r = group * B + tl.arange(0, B)
    gid = tl.load(ids + r, mask=r < R, other=-1)
    local = tl.load(mapping + gid, mask=(r < R) & (gid >= 0) & (gid < G), other=-1)
    local = tl.where((local >= 0) & (local < E), local, E)
    counts = tl.histogram(local, H)
    e = tl.arange(0, H)
    tl.store(partial + group * E + e, counts, mask=e < E)
    tl.store(inverse + r, -1, mask=r < R)
    tl.store(indices + r, -1, mask=r < BOUND)
    tail = r + NB * B
    tl.store(indices + tail, -1, mask=tail < BOUND)


@triton.jit
def _block_prefixes(
    partial,
    prefix,
    E: tl.constexpr,
    NB: tl.constexpr,
    RB: tl.constexpr,
    EH: tl.constexpr,
):
    r = tl.arange(0, RB)
    e = tl.arange(0, EH)
    values = tl.load(
        partial + r[:, None] * E + e[None, :],
        mask=(r[:, None] < NB) & (e[None, :] < E),
        other=0,
    )
    counts = tl.sum(values, axis=0)
    padded = tl.cdiv(counts, 64) * 64
    starts = tl.cumsum(padded, axis=0) - padded
    block_starts = starts[None, :] + tl.cumsum(values, axis=0) - values
    tl.store(
        prefix + r[:, None] * E + e[None, :],
        block_starts,
        mask=(r[:, None] < NB) & (e[None, :] < E),
    )


@triton.jit
def _assign_positions(
    ids,
    mapping,
    prefix,
    inverse,
    indices,
    R: tl.constexpr,
    E: tl.constexpr,
    G: tl.constexpr,
    B: tl.constexpr,
):
    group = tl.program_id(0)
    expert = tl.program_id(1)
    r = group * B + tl.arange(0, B)
    inside = r < R
    gid = tl.load(ids + r, mask=inside, other=-1)
    local = tl.load(mapping + gid, mask=inside & (gid >= 0) & (gid < G), other=-1)
    mine = inside & (local == expert)
    if tl.sum(mine.to(tl.int32), 0) > 0:
        start = tl.load(prefix + group * E + expert)
        pos = start + tl.cumsum(mine.to(tl.int32), axis=0) - 1
        tl.store(inverse + r, pos, mask=mine)
        tl.store(indices + pos, expert, mask=mine)


@triton.jit
def _gather_tokens(
    a,
    scale,
    inverse,
    out,
    out_scale,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    slots = tl.arange(0, triton.next_power_of_2(TOPK))
    positions = tl.load(inverse + token * TOPK + slots, mask=slots < TOPK, other=-1)
    if tl.sum((positions >= 0).to(tl.int32), 0) > 0:
        value = tl.load(a + token.to(tl.int64) * K + col, mask=col < K, other=0)
        s = tl.load(scale + token)
        for slot in tl.static_range(0, TOPK):
            pos = tl.load(inverse + token * TOPK + slot)
            if pos >= 0:
                tl.store(out + pos.to(tl.int64) * K + col, value, mask=col < K)
                tl.store(out_scale + pos, s)


def permute64(a, scale, ids, mapping, experts, out):
    assert a.is_contiguous() and ids.is_contiguous() and mapping.is_contiguous()
    assert scale.is_contiguous() and scale.shape[1] == 1 and a.dtype == torch.int8
    routes = ids.numel()
    groups = triton.cdiv(routes, 1024)
    bound = padded_route_rows(ids.shape[0], ids.shape[1], experts)
    assert out.shape == (bound, a.shape[1])
    partial = torch.empty((groups, experts), device=a.device, dtype=torch.int32)
    prefix = torch.empty_like(partial)
    indices = torch.empty((bound,), device=a.device, dtype=torch.int32)
    inverse = torch.empty_like(ids)
    scales = torch.empty((bound, 1), device=a.device, dtype=torch.float32)
    _histogram[(groups,)](
        ids,
        mapping,
        partial,
        indices,
        inverse,
        routes,
        experts,
        mapping.numel(),
        bound,
        groups,
        1024,
        triton.next_power_of_2(experts + 1),
        num_warps=4,
    )
    _block_prefixes[(1,)](
        partial,
        prefix,
        experts,
        groups,
        triton.next_power_of_2(groups),
        triton.next_power_of_2(experts),
        num_warps=4,
    )
    _assign_positions[(groups, experts)](
        ids,
        mapping,
        prefix,
        inverse,
        indices,
        routes,
        experts,
        mapping.numel(),
        1024,
        num_warps=4,
    )
    _gather_tokens[(ids.shape[0],)](
        a,
        scale,
        inverse,
        out,
        scales,
        a.shape[1],
        ids.shape[1],
        triton.next_power_of_2(a.shape[1]),
        num_warps=4,
    )
    return out, scales, indices, inverse, 64


def _generation(layer):
    return tuple(
        (id(t), t.data_ptr(), t._version, tuple(t.shape))
        for t in (
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
        )
    )


def _no_reload(*args, **kwargs):
    raise RuntimeError("W4A8 packed layout: restart from checkpoint for reload or EPLB")


@lru_cache(maxsize=4)
def _make_experts_class(phase, backend="deepgemm"):
    if phase != "prefill":
        raise ValueError("Step5 W4A8 contiguous experts support Prefill only")
    route_activation = os.environ.get("HCU_STEP5_W4A8_ROUTE_ACTIVATION", "0") == "1"
    # Initialize the public package before HCU replacement modules, avoiding
    # the partial deep_gemm_utils import cycle seen in the original probe.
    import vllm.model_executor.layers.fused_moe  # noqa: F401 - initialize before HCU adapters
    from vllm_hcu.model_executor.layers.quantization import (
        slimquant_w4a8_deepgemm_runtime as dg,
    )
    from vllm.model_executor.layers.fused_moe.utils import _resize_cache
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm_hcu.model_executor.layers.fused_moe.deep_gemm_api_runtime import (
        require_gemm_op,
    )

    op = require_gemm_op(backend)

    def expert_gemm(a, b, output, indices, config):
        op(
            a[0],
            b[0],
            output,
            a[1].contiguous(),
            b[1].contiguous(),
            indices.contiguous(),
            config["MODE"],
        )
        return output

    class Step5W4A8Experts(dg.DeepEPDeepGemmW4A8ContiguousExperts):
        _hcu_step5_owner = True
        _hcu_step5_w4a8 = True
        _hcu_step5_w4a8_gemm_backend = backend

        @staticmethod
        def _supports_activation(activation):
            return activation == MoEActivation.SWIGLUSTEP

        def process_weights_after_loading(self, layer):
            # Reuse the common validator and packer, but retain the Parameters
            # as owners here: the base method deliberately packs clones for
            # reload support. Changing that policy for other models would be
            # unsafe, and a second packed owner exceeds this profile's budget.
            dg._validate_w4a8_channel_weights(layer)
            old = getattr(layer, "_hcu_step5_w4a8_owner", None)
            if old is not None:
                if old != _generation(layer):
                    raise RuntimeError(
                        "Packed W4 owner changed; reload/EPLB unsupported"
                    )
            else:
                for parameter in (layer.w13_weight, layer.w2_weight):
                    before = (
                        id(parameter),
                        parameter.data_ptr(),
                        parameter.untyped_storage().nbytes(),
                        tuple(parameter.shape),
                    )
                    if not parameter.is_contiguous() or parameter.storage_offset():
                        raise ValueError(
                            "Single-owner W4 packing requires contiguous complete storage"
                        )
                    with torch.no_grad():
                        packed = dg.pack_w4a8_moe_hipc_weight(parameter)
                    after = (
                        id(parameter),
                        packed.data_ptr(),
                        packed.untyped_storage().nbytes(),
                        tuple(packed.shape),
                    )
                    if before != after:
                        raise RuntimeError(
                            "W4A8 repack allocated a second weight owner"
                        )
                layer._hcu_step5_w4a8_owner = _generation(layer)
                for parameter in (
                    layer.w13_weight,
                    layer.w2_weight,
                    layer.w13_weight_scale,
                    layer.w2_weight_scale,
                ):
                    parameter.requires_grad_(False)
                    if callable(getattr(parameter, "weight_loader", None)):
                        parameter.weight_loader = _no_reload
            self._deepgemm_w13, self._deepgemm_w2 = layer.w13_weight, layer.w2_weight

        def workspace_shapes(
            self,
            M,
            N,
            K,
            topk,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        ):
            # This private adapter stores only INT8 gathered/activation data in
            # workspace13; gateup/down BF16 data lives in workspace2. Keep the
            # framework's BF16 dtype but reserve the exact byte capacity.
            base = super().workspace_shapes(
                M,
                N,
                K,
                topk,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
                activation,
            )
            if (N, K) != (3072, 4096):
                return base
            if expert_tokens_meta is None:
                aligned = padded_route_rows(M, topk, local_num_experts)
            else:
                from vllm_hcu.model_executor.layers.quantization import (
                    slimquant_w4a8_deepgemm_runtime as dg,
                )

                aligned = dg.compute_aligned_M(
                    M=M,
                    num_topk=topk,
                    local_num_experts=local_num_experts,
                    alignment=self.ALIGNMENT,
                    expert_tokens_meta=expert_tokens_meta,
                )
            # The final output can alias workspace13 after GEMM2, so also
            # retain M*K BF16 elements when the allocator reuses this buffer.
            first_elements = max(triton.cdiv(aligned * max(K, N // 2), 2), M * K)
            return ((triton.cdiv(first_elements, K), K), (aligned, max(N, K)), (M, K))

        def apply(
            self,
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            a1q_scale,
            a2_scale,
            workspace13,
            workspace2,
            expert_tokens_meta,
            apply_router_weight_on_input,
        ):
            if apply_router_weight_on_input or not self._supports_activation(
                activation
            ):
                raise ValueError(
                    "Step5 W4 requires output-side weighting and SwigluStep7"
                )
            if hidden_states.dtype != torch.int8 or a1q_scale is None:
                raise ValueError(
                    "Modular prepare must dispatch paired INT8 activations and FP32 scales"
                )
            if hidden_states.shape[0] == 0:
                return
            if getattr(expert_tokens_meta, "_hcu_deepep_ht_local_counts", False):
                expert_tokens_meta._hcu_step5_compact_rows = True
            E, _, N, K, topk = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
            use_small = (
                0 < hidden_states.shape[0] <= 256
                and expert_map is not None
                and topk_ids.is_contiguous()
                and expert_map.is_contiguous()
                and a1q_scale.is_contiguous()
                and expert_tokens_meta is None
            )
            use_large64 = (
                not use_small
                and expert_tokens_meta is None
                and expert_map is not None
                and a1q_scale.is_contiguous()
            )
            aligned = dg.compute_aligned_M(
                M=topk_ids.shape[0],
                num_topk=topk,
                local_num_experts=E,
                alignment=self.ALIGNMENT,
                expert_tokens_meta=expert_tokens_meta,
            )
            if use_small:
                aligned = padded_route_rows(hidden_states.shape[0], topk, E)
            elif use_large64:
                aligned = padded_route_rows(hidden_states.shape[0], topk, E)
            gathered = _resize_cache(workspace13.view(dtype=torch.int8), (aligned, K))
            fused_scatter = getattr(
                expert_tokens_meta, "_hcu_step5_fused_scatter", None
            )
            if fused_scatter is not None:
                aq, scales, indices, inverse = fused_scatter
                if aq.dtype != torch.int8 or aq.shape[1] != K:
                    raise ValueError(
                        "Step5 W4A8 fused scatter requires matching INT8 expert rows"
                    )
                aligned = aq.shape[0]
            elif use_small:
                if os.getenv("VLLM_HCU_STEP5_LIGHTOP_INT8_PERMUTE", "0") == "1":
                    from lightop.moe import ep_permute_int8_small

                    aq, scales, indices, inverse = ep_permute_int8_small(
                        hidden_states,
                        self._ensure_2d_scale(a1q_scale),
                        topk_ids,
                        expert_map,
                        E,
                        64,
                        gathered,
                    )
                else:
                    aq, scales, indices, inverse, _ = small_permute(
                        hidden_states,
                        self._ensure_2d_scale(a1q_scale),
                        topk_ids,
                        expert_map,
                        E,
                        gathered,
                        global_num_experts,
                        64,
                    )
            elif use_large64:
                aq, scales, indices, inverse, _ = permute64(
                    hidden_states,
                    self._ensure_2d_scale(a1q_scale),
                    topk_ids,
                    expert_map,
                    E,
                    gathered,
                )
            else:
                aq, scales, indices, inverse, _ = dg.deepgemm_moe_permute(
                    aq=hidden_states,
                    aq_scale=self._ensure_2d_scale(a1q_scale),
                    topk_ids=topk_ids,
                    local_num_experts=E,
                    expert_map=expert_map,
                    expert_tokens_meta=expert_tokens_meta,
                    aq_out=gathered,
                )
            indices = indices.to(torch.int32).contiguous()
            gateup = _resize_cache(workspace2, (aligned, N))
            expert_gemm(
                (aq, scales),
                (self._deepgemm_w13, self.w1_scale),
                gateup,
                indices,
                config={"MODE": 1002},
            )
            intermediate = _resize_cache(
                workspace13.view(dtype=torch.int8), (aligned, N // 2)
            )
            a_scale = torch.empty(
                (aligned, 1), device=hidden_states.device, dtype=torch.float32
            )
            if route_activation:
                _activation_routes[(topk_ids.numel() + 1,)](
                    gateup,
                    intermediate,
                    a_scale,
                    inverse,
                    indices,
                    aligned,
                    topk_ids.numel(),
                    N // 2,
                    triton.next_power_of_2(N // 2),
                    8192,
                    num_warps=4,
                )
            elif hidden_states.shape[0] >= 1024 and N // 2 == 1536:
                activation_persistent[(min(triton.cdiv(aligned, 8), 1024),)](
                    gateup,
                    intermediate,
                    a_scale,
                    indices,
                    aligned,
                    N // 2,
                    8,
                    512,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            else:
                _activation[(aligned,)](
                    gateup,
                    intermediate,
                    a_scale,
                    indices,
                    N // 2,
                    triton.next_power_of_2(N // 2),
                    num_warps=4,
                )
            down = _resize_cache(workspace2, (aligned, K))
            expert_gemm(
                (intermediate, a_scale),
                (self._deepgemm_w2, self.w2_scale),
                down,
                indices,
                config={"MODE": 1002},
            )
            # Same provider/state protocol as FP8 fused experts. It declines
            # while DP-pipe owns combine or shared output is not ready. The
            # existing finalize hook then avoids a second shared addition.
            from vllm_hcu.model_executor.layers.fused_moe import ep_fused_moe

            provider = getattr(ep_fused_moe, "shared_add_provider", None)
            addition = provider(output) if provider is not None else None
            shared, start = addition if addition is not None else (output, 0)
            _weighted_output[(output.shape[0], triton.cdiv(K, 2048))](
                down,
                inverse,
                topk_weights,
                expert_map if expert_map is not None else topk_ids,
                topk_ids,
                output,
                shared,
                output.shape[0],
                K,
                topk,
                global_num_experts,
                expert_map is not None,
                addition is not None,
                start,
                shared.shape[0] if addition is not None else 0,
                2048,
                num_warps=4,
                enable_fp_fusion=False,
            )
            if fused_scatter is not None:
                from .step5_deepep import finish_scatter

                finish_scatter(expert_tokens_meta)

    return Step5W4A8Experts


def prepare_step5_w4a8(method, layer, phase):
    """Select modular W4 before model load; no legacy monolithic hook allowed."""
    import vllm.model_executor.layers.fused_moe  # noqa: F401 - initialize before HCU adapters
    from vllm_hcu.runtime_compat.step5 import uses_step5_deepep

    if uses_step5_deepep(method.moe):
        from .step5_deepep import prepare_step5_deepep

        prepare_step5_deepep(method, layer, int4=True)
        return
    if phase != "prefill":
        raise ValueError("Step5 W4A8 Decode requires DeepEP masked experts")
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk

    from vllm_hcu.model_executor.layers.fused_moe.deep_gemm_api_runtime import (
        gemm_backend,
        require_gemm_op,
    )

    backend = gemm_backend()
    Experts = _make_experts_class(phase, backend)

    activation = getattr(method.moe.activation, "value", method.moe.activation)
    if activation != "swiglustep":
        raise ValueError("Step5 W4A8 experts require SwigluStep activation")
    parallel = method.moe.moe_parallel_config
    if parallel.enable_eplb or parallel.all2all_backend != "allgather_reducescatter":
        raise ValueError("The static Prefill W4 profile requires AG/RS without EPLB")
    local_experts = static_agrs_local_experts(method.moe)
    if tuple(layer.w13_weight.shape) != (local_experts, 3072, 2048) or tuple(
        layer.w2_weight.shape
    ) != (local_experts, 4096, 768):
        raise ValueError(f"Expected Step5 EP{parallel.ep_size} packed W4 dimensions")
    if getattr(layer, "swiglu_limit", 7.0) not in (None, 7.0, 7):
        raise ValueError("Step5 W4 activation clamp must remain7")
    if getattr(layer, "_hcu_step5_w4a8_owner", None) is not None:
        if layer._hcu_step5_w4a8_owner != _generation(layer):
            raise RuntimeError("Packed W4 generation changed")
        if method.moe_kernel is not None:
            previous = getattr(layer, "_hcu_step5_w4a8_backend", {}).get(
                "gemm_backend", "deepgemm"
            )
            if previous != backend:
                raise RuntimeError(
                    "Changing Step5 W4A8 GEMM backend requires a server restart"
                )
            return
    quant = method.get_fused_moe_quant_config(layer)
    prepare = maybe_make_prepare_finalize(
        moe=method.moe,
        quant_config=quant,
        routing_tables=None,
        allow_new_interface=True,
        use_monolithic=False,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
        MoEPrepareAndFinalizeNaiveDPEPModular,
    )

    if type(prepare) is not MoEPrepareAndFinalizeNaiveDPEPModular:
        raise RuntimeError(
            "Expected modular naive DP/EP; refuse silent loss of DP pipeline"
        )
    experts = Experts(method.moe, quant)
    experts.process_weights_after_loading(layer)
    method.moe_quant_config = quant
    method.moe_kernel = mk.FusedMoEKernel(prepare, experts)
    logger.info_once(
        "[step5-w4a8] gemm_backend=%s, Prefill MODE=1002/pad64, static EP%d",
        backend,
        parallel.ep_size,
    )
    # The modular prepare owns quantize/dispatch and the modular kernel
    # owns pipe/DBO/combine. Outer naive dispatch must therefore be off.
    if not method.supports_internal_mk:
        raise RuntimeError("Framework did not recognize internal modular ownership")
    layer._hcu_step5_w4a8_backend = {
        "experts": type(experts).__name__,
        "prepare": type(prepare).__name__,
        "activation_dtype": "int8",
        "weight_bits": 4,
        "gemm_backend": backend,
        "gemm_op": str(require_gemm_op(backend)),
        "gemm_mode": 1002,
        "reduction_block": 2048,
        "small_route_alignment": 64,
        "prefill_activation": "persistent8x512",
        "profile": phase,
        "shared_add_protocol": "EP_COMBINE provider/finalize",
        "weight_scale": "canonical checkpoint s8, no x16 copy",
    }


def validate_loaded_model(model):
    modules = tuple(model.modules())
    layers = [
        module
        for module in modules
        if getattr(module, "_hcu_step5_w4a8_owner", None) is not None
    ]
    storage = {}
    for layer in layers:
        for weight in (layer.w13_weight, layer.w2_weight):
            storage[(weight.device.index, weight.untyped_storage().data_ptr())] = (
                weight.untyped_storage().nbytes()
            )
    declared = [m for m in modules if hasattr(m, "_hcu_step5_gptq_kind")]
    import os

    if os.environ.get("VLLM_HCU_STEP5_W4A8_PROFILE") == "auto" or any(
        hasattr(m, "_hcu_step5_deepep") for m in modules
    ):
        # PP stages own only their local layers. EP degree and padded physical
        # matrices determine the storage; no eight-rank byte constant applies.
        expected = {
            id(m)
            for m in modules
            if getattr(m, "_hcu_step5_w4a8_declared", False)
            or getattr(m, "_hcu_step5_gptq_kind", None) == "w4a8"
        }
        expected_bytes = sum(
            w.numel() * w.element_size()
            for m in layers
            for w in (m.w13_weight, m.w2_weight)
        )
        if (
            {id(m) for m in layers} != expected
            or len(storage) != 2 * len(layers)
            or sum(storage.values()) != expected_bytes
        ):
            raise RuntimeError(
                "Step5 DeepEP requires one packed owner per local W4 expert matrix"
            )
        logger.info(
            "[step5-deepep] verified local W4 layers=%d, mixed layers=%d, packed bytes=%d",
            len(layers),
            sum(m._hcu_step5_gptq_kind in ("bf16", "fp8") for m in declared),
            sum(storage.values()),
        )
        return
    expected_layers = 88
    if declared:
        expected = {id(m) for m in declared if m._hcu_step5_gptq_kind == "w4a8"}
        if len(declared) != 88 or not expected or expected != {id(m) for m in layers}:
            raise RuntimeError(
                "Step5 GPTQ packed owners must match all declared W4A8 experts"
            )
        for m in declared:
            if m._hcu_step5_gptq_kind == "bf16" and (
                m.w13_weight.dtype != torch.bfloat16
                or m.w2_weight.dtype != torch.bfloat16
            ):
                raise RuntimeError(
                    "Step5 GPTQ ignored experts must retain BF16 weights"
                )
            if m._hcu_step5_gptq_kind == "fp8" and (
                m.w13_weight.dtype != torch.float8_e4m3fn
                or m.w2_weight.dtype != torch.float8_e4m3fn
                or m.w13_weight_scale.dtype != torch.float32
                or m.w2_weight_scale.dtype != torch.float32
            ):
                raise RuntimeError(
                    "Step5 mixed FP8 experts require FP8 weights and FP32 scales"
                )
        expected_layers = len(expected)
    # EP16 owns 22 full experts per rank; EP8 owns 44. Keep the strict
    # single-storage invariant while accounting for the local expert count.
    local_counts = {int(m.w13_weight.shape[0]) for m in layers}
    shapes_valid = len(local_counts) == 1 and local_counts <= {22, 44}
    expected_bytes = 0
    for m in layers:
        e = int(m.w13_weight.shape[0])
        shapes_valid = shapes_valid and (
            tuple(m.w13_weight.shape) == (e, 3072, 2048)
            and tuple(m.w2_weight.shape) == (e, 4096, 768)
            and m.w13_weight.dtype == m.w2_weight.dtype == torch.int8
        )
        expected_bytes += e * (3072 * 2048 + 4096 * 768)
    if (
        not shapes_valid
        or len(layers) != expected_layers
        or len(storage) != expected_layers * 2
        or sum(storage.values()) != expected_bytes
    ):
        raise RuntimeError(
            "Step5 W4A8 must have one packed owner per declared EP8/EP16 expert layer"
        )
    logger.info(
        "[step5-w4a8] verified %d W4A8 layers, %d FP8 layers, %d BF16 layers, packed bytes=%d",
        expected_layers,
        sum(m._hcu_step5_gptq_kind == "fp8" for m in declared),
        sum(m._hcu_step5_gptq_kind == "bf16" for m in declared),
        sum(storage.values()),
    )
