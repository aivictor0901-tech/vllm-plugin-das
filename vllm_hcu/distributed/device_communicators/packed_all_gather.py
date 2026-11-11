# SPDX-License-Identifier: Apache-2.0
"""Byte-exact Step5 DP dispatch packing for uniform decode batches.

All peers must supply the same tensor count, dtypes, trailing dimensions and
uniform row sizes, as required by the underlying NCCL all-gather. Local strides
are normalized instead of choosing a different collective on one peer.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _pack4(A, B, C, D, P, N0: tl.constexpr, N1: tl.constexpr,
           N2: tl.constexpr, N3: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v0 = tl.load(A + i, i < N0, other=0)
    v1 = tl.load(B + i - N0, (i >= N0) & (i < N0 + N1), other=0)
    v2 = tl.load(C + i - N0 - N1,
                 (i >= N0 + N1) & (i < N0 + N1 + N2), other=0)
    v3 = tl.load(D + i - N0 - N1 - N2,
                 (i >= N0 + N1 + N2) & (i < N0 + N1 + N2 + N3), other=0)
    value = tl.where(i < N0, v0, tl.where(i < N0 + N1, v1,
                     tl.where(i < N0 + N1 + N2, v2, v3)))
    tl.store(P + i, value, i < N0 + N1 + N2 + N3)


@triton.jit
def _unpack4(P, A, B, C, D, N0: tl.constexpr, N1: tl.constexpr,
             N2: tl.constexpr, N3: tl.constexpr, WORLD: tl.constexpr,
             BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    stride: tl.constexpr = N0 + N1 + N2 + N3
    rank, offset = i // stride, i % stride
    valid = i < WORLD * stride
    value = tl.load(P + i, valid, other=0)
    tl.store(A + rank * N0 + offset, value, valid & (offset < N0))
    tl.store(B + rank * N1 + offset - N0, value,
             valid & (offset >= N0) & (offset < N0 + N1))
    tl.store(C + rank * N2 + offset - N0 - N1, value,
             valid & (offset >= N0 + N1) & (offset < N0 + N1 + N2))
    tl.store(D + rank * N3 + offset - N0 - N1 - N2, value,
             valid & (offset >= N0 + N1 + N2) & (offset < stride))


def _step5_dispatch_layout(inputs):
    """Recognize transport layouts without inspecting tensor values or strides."""
    if not isinstance(inputs, (list, tuple)) or len(inputs) not in (2, 4):
        return None
    if any(not isinstance(t, torch.Tensor) or t.ndim != 2 for t in inputs):
        return None
    x = inputs[0]
    rows = x.shape[0]
    if x.shape[1] != 4096 or not 1 <= rows <= 512:
        return None
    if len(inputs) == 2:
        logits = inputs[1]
        if (x.dtype not in (torch.bfloat16, torch.float16)
                or logits.dtype not in (torch.float32, torch.bfloat16, torch.float16)
                or logits.shape != (rows, 352)):
            return None
        return "float-router"
    if x.element_size() != 1:
        return None
    # Native prepare/finalize supplies [x, weights, ids, scale]. The optional
    # INT8 transport adapter supplies [x, scale, weights, ids]. Preserve order.
    if inputs[1].shape == (rows, 1):
        scale, weights, ids = inputs[1:]
    else:
        weights, ids, scale = inputs[1:]
    if (weights.dtype != torch.float32 or scale.dtype != torch.float32
            or ids.dtype not in (torch.int32, torch.int64)
            or weights.shape != (rows, 8) or ids.shape != (rows, 8)
            or scale.shape != (rows, 1)):
        return None
    return "byte-topk"


def packed_step5_all_gatherv(comm, inputs, original, dim=0, sizes=None):
    """Return tensors after one collective, or None for unsupported layouts.

    The caller enables this only for Step5 DP groups. All peers must agree on
    dtypes/trailing shapes, as with the original collectives. When sizes are
    supplied, decide uniformity from the complete vector on every rank before
    packing. Unequal, empty, or larger batches use the original AllGatherv path.
    """
    world = comm.world_size
    if dim != 0 or not 2 <= world <= 16:
        return None
    if sizes is not None and (
        len(sizes) != world or not sizes or any(s != sizes[0] for s in sizes)
    ):
        return None
    if _step5_dispatch_layout(inputs) is None:
        return None
    x = inputs[0]
    rows = x.shape[0]
    if sizes is not None and sizes[0] != rows:
        return None
    if not x.is_cuda or any(t.device != x.device for t in inputs):
        raise ValueError("packed Step5 all-gather requires one GPU device per rank")
    inputs = [tensor.contiguous() for tensor in inputs]
    counts = tuple(t.numel() * t.element_size() for t in inputs)
    views = [t.view(torch.uint8).reshape(-1) for t in inputs]
    # Reuse the byte kernels for two fields. Zero-length fields have no loads
    # or stores; constexpr specialization removes them from the generated code.
    padded_counts = (*counts, *((0,) * (4 - len(inputs))))
    padded_views = views + [views[0]] * (4 - len(inputs))
    total = sum(counts)
    packed = torch.empty(total, dtype=torch.uint8, device=x.device)
    _pack4[(triton.cdiv(total, 4096),)](
        *padded_views, packed, *padded_counts,
        BLOCK=4096, num_warps=4,
    )
    gathered = original(comm, packed, dim=0, sizes=None)
    if gathered.dtype != torch.uint8 or gathered.numel() != world * total:
        raise RuntimeError("packed Step5 all-gather returned an invalid byte buffer")
    outputs = [
        torch.empty((world * rows, *t.shape[1:]), dtype=t.dtype, device=t.device)
        for t in inputs
    ]
    output_views = [t.view(torch.uint8).reshape(-1) for t in outputs]
    padded_outputs = output_views + [output_views[0]] * (4 - len(inputs))
    _unpack4[(triton.cdiv(world * total, 4096),)](
        gathered, *padded_outputs,
        *padded_counts, WORLD=world, BLOCK=4096, num_warps=4,
    )
    return outputs
