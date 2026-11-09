"""Synthetic KV and DSA state for Step5 DecodeBenchConnector only.

This does not reconstruct real prefill outputs. Filling is intentionally done
before attention metadata construction, outside graph capture/replay, and once
per connector fill event. No state is fabricated for IFB or Mooncake requests.
"""

from __future__ import annotations

import math

import torch


def _fill_rows(tensor, rows, mean, std, *, fp8_bytes=False):
    """Fill block rows without assigning numeric 0.015 to a uint8 cache.

    Bound temporary storage for long prompts. Byte copies also avoid depending
    on float8 advanced-index assignment support in the installed PyTorch.
    """
    dtype = torch.float8_e4m3fn if fp8_bytes else tensor.dtype
    byte_copy = fp8_bytes or dtype == torch.float8_e4m3fn
    dest = tensor.view(torch.uint8) if byte_copy else tensor
    for start in range(0, len(rows), 64):
        selected = rows[start:start + 64]
        indices = torch.tensor(selected, dtype=torch.long, device=tensor.device)
        shape = (len(selected), *tensor.shape[1:])
        if std:
            values = torch.empty(shape, dtype=torch.float32, device=tensor.device)
            values.normal_(mean, std)
        else:
            values = torch.full(shape, mean, dtype=torch.float32, device=tensor.device)
        if byte_copy:
            values.clamp_(-448.0, 448.0)
        values = values.to(dtype)
        dest.index_copy_(0, indices, values.view(torch.uint8) if byte_copy else values)


def prepare_decode_bench(runner, scheduler_output):
    config = runner.vllm_config
    transfer = config.kv_transfer_config
    if transfer is None or transfer.kv_connector != "DecodeBenchConnector":
        return
    from vllm_hcu.runtime_compat.step5 import is_step5_config

    if not is_step5_config(config):
        return
    metadata = scheduler_output.kv_connector_metadata
    if metadata is None or not metadata.reqs_to_fill:
        return
    if config.cache_config.enable_prefix_caching:
        raise ValueError("Step5 DecodeBench requires --no-enable-prefix-caching.")
    for name in ("pipeline_parallel_size", "prefill_context_parallel_size",
                 "decode_context_parallel_size"):
        if getattr(config.parallel_config, name, 1) != 1:
            raise ValueError(f"Step5 DecodeBench currently requires {name}=1.")
    if config.cache_config.cache_dtype != "fp8_e4m3":
        raise ValueError("Step5 DecodeBench currently requires fp8_e4m3 KV.")
    from vllm_hcu.model_executor.layers.step4_dsa import _INDEXERS

    worker = runner.kv_connector.kv_connector.connector_worker
    caches = worker.kv_caches
    if caches is None:
        raise RuntimeError("DecodeBench KV caches are not registered.")
    mean, std = float(worker.fill_mean), float(worker.fill_std)
    if not math.isfinite(mean) or not math.isfinite(std) or std < 0:
        raise ValueError("DecodeBench fill parameters must be finite, with std >= 0.")

    # Build/validate the entire fill plan before modifying cache memory.
    plans = []
    groups = runner.kv_cache_config.kv_cache_groups
    for req_id, (block_groups, past) in metadata.reqs_to_fill.items():
        if past <= 0 or len(block_groups) != len(groups):
            raise ValueError(f"Invalid Step5 DecodeBench fill metadata for {req_id}.")
        for group_idx, (group, block_ids) in enumerate(zip(groups, block_groups)):
            for name in worker.group_to_layers[group_idx]:
                cache = caches[name]
                if not isinstance(cache, torch.Tensor) or cache.ndim not in (4, 5):
                    raise ValueError(f"Unsupported Step5 KV layout: {name}.")
                if cache.dtype not in (torch.uint8, torch.float8_e4m3fn):
                    raise ValueError(f"Expected E4M3 byte storage for {name}.")
                kernel_block = cache.shape[2]
                ratio, remainder = divmod(group.kv_cache_spec.block_size, kernel_block)
                if remainder or ratio < 1:
                    raise ValueError(f"Invalid logical/kernel block sizes for {name}.")
                physical = [block * ratio + i for block in block_ids for i in range(ratio)]
                if any(block < 0 or block >= cache.shape[0] for block in physical):
                    raise ValueError(f"DecodeBench block outside cache capacity: {name}.")
                indexer = _INDEXERS.get(name)
                if indexer is not None:
                    if cache.ndim != 4 or len(physical) < math.ceil(past / kernel_block):
                        raise ValueError(f"Incomplete DSA block table for {name}.")
                plans.append((name, cache, physical, past, indexer))

    # Settle retirement from the previous forward before a recycled physical
    # block acquires new synthetic pending state. Do not advance pending.step.
    states = {}
    for _, cache, _, _, indexer in plans:
        if indexer is not None:
            state = indexer.state
            if id(state) not in states:
                state.ensure_allocated(cache, cache.device)
                state.end_step()
                states[id(state)] = state

    for _, cache, physical, past, indexer in plans:
        _fill_rows(cache, physical, mean, std, fp8_bytes=True)
        if indexer is None:
            continue
        state = indexer.state
        rpp = state.regions_per_page
        slots = [block * rpp + i for block in physical for i in range(rpp)]
        _fill_rows(state.summary[indexer.slot], slots, mean, std)
        # Remove stale partial rows from previous owners of these pages.
        # Shared-layer calls must keep a row created for this request's tail.
        prefix = past % state.geometry.region_size
        tail = (physical[past // state.block_size] * rpp
                + (past % state.block_size) // state.geometry.region_size) if prefix else None
        for slot in slots:
            if slot != tail:
                state.pending.release(slot)
        if tail is not None:
            row = state.pending.acquire(tail)
            _fill_rows(state.pending_key[indexer.slot], [row], mean, std)
            _fill_rows(state.pending_z[indexer.slot], [row], mean, std)

    # Copies and the forward use the current stream. Clearing only after all
    # fills succeed prevents native pre_forward() from refilling raw uint8 KV
    # with truncated numeric values, or resetting state on later decode steps.
    metadata.reqs_to_fill.clear()
