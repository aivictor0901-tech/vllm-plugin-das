# SPDX-License-Identifier: Apache-2.0
"""Step5 local-query PCP runtime contract.

The runner attaches one contract to each local attention metadata object after
it has built the original global request metadata.  Only token-axis tensors
are gathered here.  Q, attention output, dense projections and residuals stay
rank-local; this module never executes a replicated attention forward.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import sys
import time
from typing import Any


@dataclass(frozen=True)
class Step5PCPQueryRun:
    global_req_idx: int
    local_start: int
    global_start: int
    num_tokens: int


@dataclass(frozen=True)
class Step5PCPMetadata:
    group: Any
    global_attn_metadata: Any
    global_restore_indices: Any
    local_num_tokens: int
    local_padded_tokens: int
    global_num_tokens: int
    has_global_prefill: bool
    local_is_padding: Any | None = None
    local_query_runs: tuple[Step5PCPQueryRun, ...] = ()

    def __post_init__(self) -> None:
        if int(self.group.world_size) <= 1:
            raise ValueError("Step5 PCP metadata requires a real PCP group")
        if not 0 <= int(self.group.rank_in_group) < int(self.group.world_size):
            raise ValueError("Step5 PCP group rank is out of range")
        if not 0 <= self.local_num_tokens <= self.local_padded_tokens:
            raise ValueError("invalid Step5 PCP local token counts")
        if self.global_num_tokens < 0:
            raise ValueError("invalid Step5 PCP global token count")
        if int(self.global_restore_indices.numel()) != self.global_num_tokens:
            raise ValueError("Step5 PCP restore indices must cover the global batch")
        global_actual = getattr(self.global_attn_metadata, "num_actual_tokens", None)
        if global_actual is not None and int(global_actual) != self.global_num_tokens:
            raise ValueError("Step5 PCP global metadata/token count mismatch")
        if self.local_is_padding is not None and int(self.local_is_padding.numel()) < self.local_num_tokens:
            raise ValueError("Step5 PCP padding mask is shorter than the local batch")


def step5_pcp_enabled(vllm_config: object) -> bool:
    """Use the frozen, model-specific platform predicate when opted in."""
    if os.environ.get("VLLM_STEP5_PCP", "0") != "1":
        return False
    from vllm_hcu.v1.step5_pcp_config import is_step5_pcp

    return bool(is_step5_pcp(vllm_config))


def get_step5_pcp_metadata(
    metadata: object | None = None, *, layer_name: str | None = None,
) -> Step5PCPMetadata | None:
    """Resolve a per-layer contract, including cache-update calls without MD.

    A runner creates these objects only for ``is_step5_pcp(config)``.  The
    independent environment gate keeps an accidentally retained metadata
    object from changing a feature-off invocation.
    """
    if os.environ.get("VLLM_STEP5_PCP", "0") != "1":
        return None
    if metadata is None:
        from vllm.forward_context import get_forward_context

        metadata = get_forward_context().attn_metadata
    if isinstance(metadata, list):
        if len(metadata) != 1:
            raise ValueError("Step5 PCP does not support attention ubatches")
        metadata = metadata[0]
    if isinstance(metadata, dict):
        if layer_name is not None:
            metadata = metadata.get(layer_name)
        else:
            contexts = {id(ctx): ctx for value in metadata.values()
                        if (ctx := getattr(value, "step5_pcp", None)) is not None}
            if not contexts:
                return None
            if len(contexts) != 1:
                raise ValueError("Step5 PCP cache metadata lookup requires layer_name")
            return next(iter(contexts.values()))
    context = getattr(metadata, "step5_pcp", None)
    if context is not None and not isinstance(context, Step5PCPMetadata):
        raise TypeError("invalid Step5 PCP attention metadata contract")
    return context


_PCPDBG_COUNT = 0


_PCP_GATHER_TIMING = {
    "calls": 0,
    "pad_s": 0.0,
    "gather_s": 0.0,
    "idx_s": 0.0,
    "total_s": 0.0,
    "by_sig": {},
}


def _record_pcp_gather_timing(tensor, context, width, pad_s, gather_s, idx_s, total_s):
    state = _PCP_GATHER_TIMING
    state["calls"] += 1
    state["pad_s"] += pad_s
    state["gather_s"] += gather_s
    state["idx_s"] += idx_s
    state["total_s"] += total_s
    sig = (tuple(int(x) for x in tensor.shape[1:]), int(width),
           int(context.group.world_size))
    entry = state["by_sig"].setdefault(
        sig, {"calls": 0, "pad": 0.0, "gather": 0.0, "idx": 0.0, "total": 0.0}
    )
    entry["calls"] += 1
    entry["pad"] += pad_s
    entry["gather"] += gather_s
    entry["idx"] += idx_s
    entry["total"] += total_s
    if state["calls"] % 128 != 0:
        return
    top = sorted(state["by_sig"].items(), key=lambda kv: -kv[1]["total"])[:3]
    top_str = "; ".join(
        "%s:n=%d,tot=%.3f,g=%.3f,i=%.3f,p=%.3f"
        % (key, val["calls"], val["total"], val["gather"], val["idx"], val["pad"])
        for key, val in top
    )
    print(
        "PCP_GATHER_TIMING pid=%d rank=%d calls=%d pad_s=%.3f gather_s=%.3f "
        "idx_s=%.3f total_s=%.3f top=[%s]"
        % (
            os.getpid(),
            int(context.group.rank_in_group),
            state["calls"],
            state["pad_s"],
            state["gather_s"],
            state["idx_s"],
            state["total_s"],
            top_str,
        ),
        flush=True,
        file=sys.stderr,
    )


def gather_global_tokens(tensor: Any, context: Step5PCPMetadata) -> Any:
    """Gather equal-width local rows and restore the original request order.

    Replicated decode rows occur on every rank; the manager's restore index
    selects their rank-zero copy exactly once.  Decode-only execution already
    has the original batch on each rank, so it needs no collective.
    """
    if tensor.ndim < 1 or int(tensor.shape[0]) < context.local_num_tokens:
        raise ValueError("Step5 PCP tensor is shorter than the local token batch")
    if not context.has_global_prefill:
        if context.global_num_tokens > context.local_num_tokens:
            raise ValueError("replicated Step5 PCP decode batch is incomplete")
        return tensor[:context.global_num_tokens]
    import torch

    timing = bool(os.environ.get("VLLM_STEP5_PCP_GATHER_TIMING"))
    total_t0 = time.perf_counter() if timing else 0.0
    pad_s = 0.0
    gather_s = 0.0
    idx_s = 0.0

    width = context.local_padded_tokens
    pad_t0 = time.perf_counter() if timing else 0.0
    if int(tensor.shape[0]) < width:
        padded = tensor.new_zeros((width, *tensor.shape[1:]))
        padded[:context.local_num_tokens].copy_(tensor[:context.local_num_tokens])
    else:
        padded = tensor[:width].contiguous()
    if timing:
        pad_s = time.perf_counter() - pad_t0

    gather_t0 = time.perf_counter() if timing else 0.0
    gathered = context.group.all_gather(padded, dim=0)
    if timing:
        gather_s = time.perf_counter() - gather_t0
    if int(gathered.shape[0]) != width * int(context.group.world_size):
        raise RuntimeError("Step5 PCP gather returned an invalid token width")
    indices = context.global_restore_indices
    global _PCPDBG_COUNT
    if (os.environ.get("VLLM_STEP5_PCP_DEBUG")
            and context.global_num_tokens == 43
            and _PCPDBG_COUNT < 8):
        _PCPDBG_COUNT += 1
        try:
            rows = gathered.reshape(int(gathered.shape[0]), -1)
            occ = (rows != 0).any(dim=1).to(torch.uint8).cpu().tolist()
            print(
                "PCPDBG local=%d padded=%d global=%d world=%d in=%s out=%s "
                "idx=%s occ=%s"
                % (
                    context.local_num_tokens,
                    context.local_padded_tokens,
                    context.global_num_tokens,
                    int(context.group.world_size),
                    tuple(tensor.shape),
                    tuple(gathered.shape),
                    [int(x) for x in indices[:16].cpu().tolist()],
                    "".join(str(x) for x in occ),
                ),
                flush=True,
                file=sys.stderr,
            )
        except Exception as exc:
            print("PCPDBG_ERR %r" % (exc,), flush=True, file=sys.stderr)
    if indices.dtype != torch.int64:
        raise TypeError("Step5 PCP restore indices must use int64")
    if context.global_num_tokens == 0:
        return gathered[:0]
    idx_t0 = time.perf_counter() if timing else 0.0
    result = torch.index_select(gathered, 0, indices)
    if timing:
        idx_s = time.perf_counter() - idx_t0
        _record_pcp_gather_timing(
            tensor,
            context,
            width,
            pad_s,
            gather_s,
            idx_s,
            time.perf_counter() - total_t0,
        )
    return result


def global_cache_slot_mapping(context: Step5PCPMetadata) -> Any:
    slots = context.global_attn_metadata.slot_mapping
    if int(slots.shape[0]) != context.global_num_tokens:
        raise ValueError("Step5 PCP global KV slots do not match the original batch")
    return slots


def mask_local_padding(output: Any, context: Step5PCPMetadata) -> Any:
    """Keep virtual padding finite before it reaches routing or a residual."""
    if context.local_is_padding is None:
        return output
    import torch

    if context.local_is_padding.dtype != torch.bool:
        raise TypeError("Step5 PCP local padding mask must use bool")
    count = min(int(output.shape[0]), int(context.local_is_padding.shape[0]))
    mask = context.local_is_padding[:count]
    mask = mask.reshape((count,) + (1,) * (output.ndim - 1))
    output[:count].masked_fill_(mask, 0)
    return output


def scoring_rows(metadata: Any, region_size: int) -> list[dict[str, int]]:
    """Build local score geometry without touching global pending state.

    Two DualChunkSwap intervals from one request are separate virtual rows.
    Their ``seq_len - qlen`` is the absolute start of that interval, including
    already cached chunks.  This pure host contract is also exercised by CPU
    tests; the DSA adapter adds the physical summary-slot tensors.
    """
    if region_size <= 0:
        raise ValueError("DSA region size must be positive")
    starts = metadata.query_start_loc_cpu.tolist()
    seq_lens = metadata.seq_lens_cpu.tolist()
    num_reqs = int(metadata.num_decodes) + int(metadata.num_prefills)
    if len(starts) < num_reqs + 1 or len(seq_lens) < num_reqs:
        raise ValueError("Step5 PCP local score metadata has incomplete rows")
    result = []
    for row in range(num_reqs):
        start, stop = int(starts[row]), int(starts[row + 1])
        qlen = stop - start
        if qlen < 0:
            raise ValueError("Step5 PCP query rows must be monotonic")
        if qlen == 0:
            continue
        past = int(seq_lens[row]) - qlen
        if past < 0:
            raise ValueError("Step5 PCP query row has negative absolute position")
        result.append(dict(row=row, past=past, qlen=qlen, tok_start=start,
                           nscore=(past + qlen - 1) // region_size))
    return result


__all__ = [
    "Step5PCPMetadata", "Step5PCPQueryRun", "step5_pcp_enabled",
    "get_step5_pcp_metadata", "gather_global_tokens", "global_cache_slot_mapping",
    "mask_local_padding", "scoring_rows",
]
