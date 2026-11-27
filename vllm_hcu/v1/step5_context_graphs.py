# SPDX-License-Identifier: Apache-2.0
"""Opt-in context-width variants of Step5 MRv2 target FULL graphs.

The upstream token/request descriptor is preserved as the dispatch candidate.
Only the final FULL lookup gains a context dimension. DP agreement carries that
dimension in the existing CPU collective; no additional collective is issued.
"""
from __future__ import annotations

import os
from dataclasses import make_dataclass
from functools import lru_cache


def parse_context_buckets(value: str, max_model_len: int) -> tuple[int, ...]:
    """Region counts, not token counts. Empty means the legacy single graph."""
    if not value.strip():
        return ()
    if max_model_len <= 0:
        raise ValueError("Step5 context graphs require a positive model length")
    fields = value.split(",")
    if any(not item.strip().isascii() or not item.strip().isdecimal()
           for item in fields):
        raise ValueError("HCU_STEP5_CONTEXT_GRAPH_BUCKETS requires positive integer regions")
    requested = sorted({int(item) for item in fields})
    if requested[0] <= 0:
        raise ValueError("Step5 context graph buckets must be positive")
    limit = (max_model_len + 7) // 8
    # A shorter model limit needs a terminal bucket at its own limit, rather
    # than silently losing graph coverage between the last bucket and the cap.
    return tuple(sorted({min(size, limit) for size in requested}))


def choose_context_bucket(upper: int | None, buckets: tuple[int, ...]) -> int:
    if upper is None or upper < 0:
        return 0
    return next((size for size in buckets if upper <= size * 8), 0)


@lru_cache(maxsize=None)
def _descriptor_type(base):
    return make_dataclass(
        "Step5ContextGraphDescriptor", [("context_regions", int, 0)],
        bases=(base,), frozen=True,
    )


def context_descriptor(desc, regions):
    from dataclasses import fields

    cls = _descriptor_type(type(desc))
    return cls(**{field.name: getattr(desc, field.name) for field in fields(desc)},
               context_regions=regions)


def configure_context_graphs(runner, *, profile_only=False):
    """Expand only target FULL descriptors; leave PW and draft graphs intact."""
    value = os.environ.get("HCU_STEP5_CONTEXT_GRAPH_BUCKETS", "")
    if not value.strip():
        return False
    from vllm.config.compilation import CUDAGraphMode
    from vllm_hcu.platforms import envs as henvs

    if not henvs.custom_ops_enabled():
        return False
    buckets = parse_context_buckets(value, runner.max_model_len)
    parallel = runner.parallel_config
    if (parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or getattr(parallel, "decode_context_parallel_size", 1) != 1
            or getattr(runner, "use_dcp", False)
            or getattr(runner, "pcp_manager", None) is not None
            or runner.lora_config is not None
            or getattr(runner, "ubatch_runner", None) is not None
            or getattr(runner, "adaptive_verification", None) is not None):
        raise ValueError("Step5 context graphs require PP1/PCP1/DCP1 without LoRA/DBO/adaptive verification")
    if not getattr(runner, "_hsl_enabled", False):
        raise ValueError("Step5 context graphs require the host sequence-length mirror")
    hot = int(os.environ.get("HCU_STEP5_CONTEXT_GRAPH_REGIONS", "0"))
    width = int(os.environ.get("VLLM_STEP4_DSA_GRAPH_REGIONS", "0"))
    if hot != width or hot < buckets[-1]:
        raise ValueError("Both Step5 graph-region limits must agree and cover every context bucket")
    manager = runner.cudagraph_manager
    if manager is None or not manager._capture_descs.get(CUDAGraphMode.FULL):
        raise ValueError("Step5 context buckets require target FULL CUDA graphs")
    if hasattr(manager, "_step5_context_buckets"):
        raise RuntimeError("Step5 context graph manager was configured twice")
    from vllm_hcu.model_executor.layers.step4_dsa import Step4DSAMetadataBuilder

    states = []
    for groups in runner.attn_groups:
        for group in groups:
            builder = group.get_metadata_builder(0)
            if isinstance(builder, Step4DSAMetadataBuilder):
                state = builder._dsa_state()
                if state is not None and not any(state is old for old in states):
                    states.append(state)
    if len(states) != 1:
        raise ValueError("Step5 context graphs require exactly one target DSA state")
    manager._step5_context_states = states
    manager._step5_context_buckets = buckets
    manager._step5_context_profile_only = profile_only
    base_descs = manager._capture_descs[CUDAGraphMode.FULL]
    # Largest context and token shapes first, preserving graph-pool reuse.
    manager._capture_descs[CUDAGraphMode.FULL] = [
        context_descriptor(desc, regions)
        for regions in reversed(buckets) for desc in base_descs
    ]
    manager._step5_context_lookup = {
        (desc, regions): context_descriptor(desc, regions)
        for desc in base_descs for regions in buckets
    }
    runner._step5_context_regions = None
    return True


def activate_capture_context(manager, desc, *, prepare=False):
    regions = getattr(desc, "context_regions", 0)
    if not regions:
        return
    for state in manager._step5_context_states:
        # First synthetic eager pass may be the allocation pass. The captured
        # factory is invoked again after that pass and sets the correct view.
        if state.summary is not None:
            state.select_context_graph_width(regions, warmup=prepare)


def _select_runtime_graph(runner, desc, regions):
    from vllm.config.compilation import CUDAGraphMode

    runner._step5_context_regions = regions
    if desc.cg_mode != CUDAGraphMode.FULL:
        # Eager metadata can grow to the actual history through the legacy
        # fallback path. No uncaptured context variant is ever replayed.
        return desc
    manager = runner.cudagraph_manager
    selected = manager._step5_context_lookup.get((desc, regions))
    if selected is None or selected not in manager.graphs:
        raise RuntimeError(f"Uncaptured Step5 context graph: regions={regions}, batch={desc}")
    for state in manager._step5_context_states:
        if state.summary is None:
            raise RuntimeError("Step5 context graph lost its DSA allocation")
        state.select_context_graph_width(regions)
    return selected


def dispatch_step5_cg_and_sync_dp(
    manager, num_reqs, num_tokens, uniform_token_count, dp_size, dp_rank,
    *, runner, batch_state=None, dummy_run=False, max_query_len=None, need_eager=False,
    num_active_loras=0, parallel_config=None, allow_ubatching=False,
    uniform_decode=False,
):
    """Legacy dispatch unchanged unless context variants are configured.

    This opt-in PP1 path mirrors MRv2's six-row CPU agreement with one appended
    row for context-region votes. DPSyncState stays upstream-compatible, so MTP
    continues to reuse target padding without another collective.
    """
    from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp

    if manager is None or not hasattr(manager, "_step5_context_buckets"):
        return dispatch_cg_and_sync_dp(
            manager, num_reqs, num_tokens, uniform_token_count, dp_size, dp_rank,
            max_query_len=max_query_len, need_eager=need_eager,
            num_active_loras=num_active_loras, parallel_config=parallel_config,
            allow_ubatching=allow_ubatching, uniform_decode=uniform_decode,
        )
    if allow_ubatching:
        raise ValueError("Step5 context graph dispatch does not support DBO")
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.dp_utils import DPSyncState
    from vllm_hcu.v1.step5_hot_context import conservative_seq_upper_bound

    buckets = manager._step5_context_buckets
    upper = (0 if dummy_run else
             conservative_seq_upper_bound(runner, batch_state))
    regions = choose_context_bucket(upper, buckets)
    runner._step5_hot_context_real_batch = not dummy_run
    runner._step5_hot_context_upper = upper
    need_eager = need_eager or regions == 0
    if need_eager:
        desired = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE, num_tokens=num_tokens, num_reqs=num_reqs,
            num_active_loras=num_active_loras,
        )
    else:
        desired = manager.dispatch(
            num_reqs, num_tokens, uniform_token_count,
            num_active_loras=num_active_loras, max_query_len=max_query_len,
        )
    if dp_size == 1:
        return _select_runtime_graph(runner, desired, regions), None

    import torch
    import torch.distributed as dist
    from vllm.distributed.parallel_state import get_dp_group

    votes = torch.zeros(7, dp_size, dtype=torch.int32, device="cpu")
    votes[0][dp_rank] = num_tokens
    votes[1][dp_rank] = desired.cg_mode.value
    votes[2][dp_rank] = uniform_token_count or 0
    votes[3][dp_rank] = max_query_len if max_query_len is not None else -1
    votes[4][dp_rank] = 0  # DBO is explicitly unsupported by this opt-in path.
    votes[5][dp_rank] = num_reqs
    votes[6][dp_rank] = regions
    dist.all_reduce(votes, group=get_dp_group().cpu_group)
    tokens = votes[0]
    if bool(torch.all(tokens == 0)):
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE, num_tokens=0, num_reqs=0,
        ), None
    agreed_uniform = int(votes[2][0])
    if not agreed_uniform or not bool(torch.all(votes[2] == agreed_uniform)):
        agreed_uniform = None
    regions = int(votes[6].max())
    mode = CUDAGraphMode(int(votes[1].min()))
    if mode == CUDAGraphMode.NONE:
        desc = BatchExecutionDescriptor(
            cg_mode=mode, num_tokens=num_tokens, num_reqs=num_reqs,
            num_active_loras=desired.num_active_loras,
        )
        sync = DPSyncState(tokens, agreed_uniform, True, int(votes[5].max()))
    else:
        query_bound = int(votes[3].max()) if bool(torch.all(votes[3] != -1)) else None
        desc = manager.dispatch(
            num_reqs, int(tokens.max()), agreed_uniform,
            num_active_loras=num_active_loras, max_query_len=query_bound,
        )
        tokens[:] = desc.num_tokens
        sync = DPSyncState(
            tokens, agreed_uniform, False,
            desc.num_reqs if desc.cg_mode == CUDAGraphMode.FULL
            and desc.num_reqs is not None else int(votes[5].max()),
        )
    return _select_runtime_graph(runner, desc, regions), sync
