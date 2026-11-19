# SPDX-License-Identifier: Apache-2.0
"""Bind Step5 local-query metadata to the original PCP request geometry."""
from __future__ import annotations

from copy import copy
from functools import lru_cache
import sys
from types import MethodType

from vllm_hcu.v1.step5_pcp_config import is_step5_pcp, validate_step5_pcp_config


@lru_cache(None)
def _step5_backend(base):
    if getattr(base, "_step5_pcp_backend_adapter", False):
        return base

    class Step5PCPBackend(base):
        _step5_pcp_backend_adapter = True

        @classmethod
        def supports_pcp(cls):
            return True

    # The HCU allocator identifies FlashAttention by its owning module.
    # Preserve that identity and publish an importable, distinct class name.
    name = "Step5PCP" + base.__name__
    Step5PCPBackend.__name__ = Step5PCPBackend.__qualname__ = name
    Step5PCPBackend.__module__ = base.__module__
    module = sys.modules[base.__module__]
    if hasattr(module, name):
        raise RuntimeError(f"Step5 PCP backend binding already exists: {base.__module__}.{name}")
    setattr(module, name, Step5PCPBackend)
    return Step5PCPBackend


def install_step5_pcp(runner):
    """Admit only the backends implemented by the Step5 PCP adapters."""
    if not is_step5_pcp(runner.vllm_config):
        return
    validate_step5_pcp_config(runner.vllm_config)
    from vllm.config import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm_hcu.model_executor.layers.step4_dsa import Step4SparseAttentionBackend
    from vllm_hcu.v1.attention.backends.flash_attn import HcuFlashAttentionBackend

    layers = get_layers_from_vllm_config(runner.vllm_config, AttentionLayerBase)
    if not layers:
        raise RuntimeError("Step5 PCP found no registered attention layers")
    counts = [0, 0]
    for layer in layers.values():
        backend = layer.get_attn_backend()
        if issubclass(backend, Step4SparseAttentionBackend):
            counts[0] += 1
        elif issubclass(backend, HcuFlashAttentionBackend):
            if not getattr(layer.impl, "_step5_pcp", False):
                raise RuntimeError("Step5 PCP SWA adapter was not constructed")
            counts[1] += 1
        else:
            raise RuntimeError(f"Unreviewed Step5 PCP backend: {backend}")
        backend = _step5_backend(backend)
        layer.get_attn_backend = MethodType(lambda self, b=backend: b, layer)
    if not all(counts):
        raise RuntimeError(f"Step5 PCP expected DSA and SWA layers: {counts}")
    runner.model_state._step5_pcp_runner = runner


def prepare_step5_pcp_attn(
    state, input_batch, cudagraph_mode, block_tables, slot_mappings,
    attn_groups, kv_cache_config, for_capture=False, ubatch_idx=0,
):
    from vllm_hcu.model_executor.layers.attention.pcp import (
        in_replicated_mtp_batch, logical_pcp_metadata_scope,
    )
    from vllm_hcu.v1.step5_metadata import prepare_step5_attn
    from vllm_hcu.v1.step5_pcp import Step5PCPMetadata, Step5PCPQueryRun

    def build(batch, tables, slots):
        # The native kernels consume full KV with ordinary absolute sequence
        # lengths. PCP ownership is carried separately in Step5PCPMetadata.
        with logical_pcp_metadata_scope(1):
            return prepare_step5_attn(
                state, batch, cudagraph_mode, tables, slots,
                attn_groups, kv_cache_config, for_capture, ubatch_idx,
            )

    if in_replicated_mtp_batch():
        # Restored global MTP and explicitly scoped dummy profiling have no
        # target CP contract. Their slots are global or invalid respectively.
        return build(input_batch, block_tables, slot_mappings)
    if for_capture or ubatch_idx:
        raise RuntimeError("Step5 PCP target supports eager non-ubatched attention")
    runner = getattr(state, "_step5_pcp_runner", None)
    manager = getattr(runner, "pcp_manager", None)
    if manager is None or getattr(state, "_step5_pcp_local_batch", None) is not input_batch:
        raise RuntimeError("Step5 PCP target is missing its partitioned input batch")
    global_batch = manager._global_batch
    if global_batch is None or global_batch is input_batch:
        raise RuntimeError("Step5 PCP target is missing original global request rows")
    global_tables, global_slots = manager.prepare_global_attn()
    global_metadata = build(global_batch, global_tables, global_slots)
    width = int(manager._padded_num_tokens)
    if manager._global_has_prefill:
        start = int(manager.pcp_rank) * width
        local_slots = slot_mappings[:, start:start + int(input_batch.num_tokens)]
    else:
        local_slots = global_slots[:, :int(input_batch.num_tokens)]
    local_descriptor = copy(input_batch)
    # Eager PCP queries use the local segment lengths, not the inherited
    # global dispatch descriptor (which can launch mostly empty SWA tiles).
    local_descriptor.max_query_len = int(input_batch.num_scheduled_tokens.max())
    local_metadata = build(local_descriptor, block_tables, local_slots)
    if local_metadata.keys() != global_metadata.keys():
        raise RuntimeError("Step5 PCP global and local attention layers differ")
    query_runs = tuple(
        Step5PCPQueryRun(
            segment.global_req_idx, segment.local_slice.start,
            segment.global_slice.start, segment.num_actual_tokens,
        ) for segment in manager._local_segments if segment.num_actual_tokens
    )
    seen = {}
    for name, local in local_metadata.items():
        global_md = global_metadata[name]
        previous = seen.get(id(local))
        if previous is not None:
            if previous.global_attn_metadata is not global_md:
                raise RuntimeError("Step5 PCP shared local metadata has conflicting global rows")
            continue
        if int(local.num_actual_tokens) != int(input_batch.num_tokens):
            raise RuntimeError("Step5 PCP local metadata token count changed")
        context = Step5PCPMetadata(
            group=manager._pcp_group,
            global_attn_metadata=global_md,
            global_restore_indices=manager._hidden_restore_idx,
            local_num_tokens=int(input_batch.num_tokens),
            local_padded_tokens=width,
            global_num_tokens=int(global_batch.num_tokens),
            has_global_prefill=bool(manager._global_has_prefill),
            local_is_padding=input_batch.is_padding,
            local_query_runs=query_runs,
        )
        local.step5_pcp = context
        # Generic PCP Q/K LSE merging must not interpret this metadata.
        if hasattr(local, "pcp_world_size"):
            local.pcp_world_size = 1
        if hasattr(local, "pcp_plan"):
            local.pcp_plan = None
        seen[id(local)] = context
    if any(getattr(md, "step5_pcp", None) is None for md in local_metadata.values()):
        raise RuntimeError("Step5 PCP target attention contract is incomplete")
    return local_metadata
