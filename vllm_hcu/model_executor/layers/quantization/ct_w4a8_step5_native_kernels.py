# SPDX-License-Identifier: Apache-2.0
# Derived from frozen NMZ plugin626ffb5; only BF16 SwiGLUStep7 and masked reduction.
import triton
import triton.language as tl
from triton.language.extra import libdevice


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
        w = tl.load(weights + row * TOPK + slot, mask=valid & (pos >= 0), other=0.)
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
