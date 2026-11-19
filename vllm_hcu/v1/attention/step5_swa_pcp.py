# SPDX-License-Identifier: Apache-2.0
"""Step5 PCP SWA: rank-local queries against a replicated paged KV cache.

The PCP manager represents each contiguous local query segment as a virtual
request. Its sequence end is the last absolute query position plus one, not
the original request's sequence end. The native SWA kernels can therefore
apply their usual ``seq_len - query_len + row`` causal/window rule unchanged.
K/V writes use original request order and original slots; Q is never gathered.
"""

from __future__ import annotations


def gather_step5_swa_cache_inputs(key, value, context):
    """Gather each logical K/V token once and pair it with its original slot.

    Replicated decode rows are deduplicated by the shared Step5 PCP restore
    map. All ranks consequently store the same full cache, with no rank-major
    slot assumptions and no second generic PCP cache gather.
    """
    from vllm_hcu.v1.step5_pcp import (
        gather_global_tokens,
        global_cache_slot_mapping,
    )

    global_key = gather_global_tokens(key, context)
    global_value = gather_global_tokens(value, context)
    slots = global_cache_slot_mapping(context)
    expected = int(context.global_num_tokens)
    if not (
        global_key.shape[0] == global_value.shape[0] == slots.shape[0] == expected
    ):
        raise RuntimeError("Step5 SWA PCP K/V and original slot counts disagree")
    return global_key, global_value, slots


def forward_step5_swa_pcp(
    query,
    key_cache,
    value_cache,
    output,
    metadata,
    context,
    *,
    softmax_scale,
    window_left,
    decode_kernel,
    prefill_kernel,
):
    """Run native full-cache SWA on local virtual rows without LSE merging.

    ``metadata`` must be the manager's local metadata. The global metadata in
    ``context`` belongs only to cache writes: passing it to attention would
    shift a middle chunk's queries to the end of the complete request.
    """
    tokens = int(metadata.num_actual_tokens)
    if not (0 <= tokens <= int(context.local_num_tokens)):
        raise RuntimeError("Step5 SWA PCP local query token count is invalid")
    if query.shape[0] < tokens or output.shape[0] < tokens:
        raise RuntimeError("Step5 SWA PCP local query/output storage is too short")
    if metadata.query_start_loc.shape[0] != metadata.seq_lens.shape[0] + 1:
        raise RuntimeError("Step5 SWA PCP virtual request lengths disagree")
    if metadata.block_table.shape[0] != metadata.seq_lens.shape[0]:
        raise RuntimeError("Step5 SWA PCP virtual request block rows disagree")
    if metadata.causal is not True or metadata.use_cascade or window_left < 0:
        raise RuntimeError("Step5 SWA PCP requires causal non-cascade SWA")

    args = (
        query[:tokens],
        key_cache,
        value_cache,
        output[:tokens],
        metadata.query_start_loc,
        metadata.seq_lens,
        metadata.block_table,
    )
    if tokens:
        if metadata.max_query_len <= 4:
            decode_kernel(*args, softmax_scale, window_left)
        else:
            prefill_kernel(
                *args, metadata.max_query_len, softmax_scale, window_left
            )

    # The manager gives padding-only rows no real cache or output ownership.
    # Clear after attention: zero * NaN would leave NaNs in the residual stream.
    padding = getattr(context, "local_is_padding", None)
    if padding is not None:
        if padding.shape[0] < tokens:
            raise RuntimeError("Step5 SWA PCP padding mask is shorter than Q")
        output[:tokens].masked_fill_(padding[:tokens, None, None], 0)
    if output.shape[0] > tokens:
        output[tokens:].zero_()
    return output
