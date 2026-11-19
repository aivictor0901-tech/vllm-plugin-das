# SPDX-License-Identifier: Apache-2.0
"""Dependency-light policy for the isolated BMZ expert-compute experiment."""

import os

FLAG = "VLLM_STEP5_NATIVE_W4A8"
# Internal expert-compute capacity only; scheduler/AgRs/chunk-prefill unchanged.
NATIVE_TOKEN_CHUNK = 8192


def native_enabled():
    value = os.environ.get(FLAG, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{FLAG} must be0 or1")
    return value == "1"


def validate_parallel(parallel, activation, backend, has_bias=False):
    """Do not take ownership of AgRs dispatch, combine, PP or shared experts."""
    if activation != "swiglustep" or has_bias:
        raise ValueError("Native CT W4A8 requires bias-free SwiGLUStep7")
    if backend not in ("auto", "triton"):
        raise ValueError("Native CT W4A8 replaces the AgRs auto/triton expert compute only")
    if parallel is None or getattr(parallel, "all2all_backend", None) != "allgather_reducescatter":
        raise ValueError("Native CT W4A8 requires the original AgRs outer dispatch")
    if getattr(parallel, "enable_eplb", False):
        raise ValueError("Native packed weights require static expert ownership")
    topology = (getattr(parallel, "tp_size", None),
                getattr(parallel, "pcp_size", 1),
                getattr(parallel, "dp_size", None),
                getattr(parallel, "ep_size", None))
    # FusedMoEParallelConfig keeps PCP/DP widths after flattening their
    # expert ownership into EP. CP8 and DP8 therefore share the same
    # eight-expert-rank AgRs contract; attention/PP scheduling stays outside
    # this expert-compute method.
    if topology not in ((1, 1, 8, 8), (1, 8, 1, 8)):
        raise ValueError("Native Step5 requires TP1/EP8 with either DP8 or PCP8/DP1")
    if getattr(parallel, "sp_size", 1) != 1:
        raise ValueError("Native Step5 does not support additional sequence parallelism")


def padded_route_rows(tokens, topk, experts, alignment=64):
    """Worst-case bound even when every route belongs to this EP rank."""
    if min(tokens, topk, experts) <= 0 or alignment != 64:
        raise ValueError("Positive dimensions and fixed64-row alignment required")
    routes = tokens * topk
    return ((routes + min(routes, experts) * (alignment - 1) + alignment - 1)
            // alignment * alignment)


def workspace_bytes(rows, hidden=4096, intermediate=1536):
    """Two serially reused buffers, independent of model layer count."""
    if rows <= 0:
        return {"INT8": 0, "BF16": 0, "total": 0}
    integer = rows * max(hidden, intermediate)
    bf16 = 2 * rows * max(hidden, 2 * intermediate)
    return {"INT8": integer, "BF16": bf16, "total": integer + bf16}


def token_chunk_ranges(tokens):
    """Every token follows the same native path with a fixed capacity cap."""
    if tokens < 0:
        raise ValueError("Token count must be nonnegative")
    for start in range(0, tokens, NATIVE_TOKEN_CHUNK):
        yield start, min(start + NATIVE_TOKEN_CHUNK, tokens)
