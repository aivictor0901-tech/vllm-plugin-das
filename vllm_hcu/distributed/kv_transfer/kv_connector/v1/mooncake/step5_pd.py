"""Step5 P/D state handoff for matching TP, PP=PCP=DCP=1 instances.

Ordinary KV pages alone do not contain the DSA summaries or partial regions.
Summaries use the existing block-addressed RDMA transfer. Small partial-region
buffers travel in the control response and are restored on the worker thread
before reporting receive completion to the scheduler.
"""

from __future__ import annotations

def enabled(config) -> bool:
    transfer = getattr(config, "kv_transfer_config", None)
    return bool(transfer and transfer.kv_connector_extra_config.get("step5_pd"))


def validate_config(config) -> None:
    from vllm_hcu.runtime_compat.step5 import is_step5_config

    if not is_step5_config(config):
        raise ValueError("step5_pd requires a Step5 model.")
    parallel = config.parallel_config
    # PP is allowed: each producer stage owns a layer subset, pushes the DSA
    # tails for those layers only, and the consumer unions the responses.
    for name in ("prefill_context_parallel_size", "decode_context_parallel_size"):
        if getattr(parallel, name, 1) != 1:
            raise ValueError(f"step5_pd requires {name}=1.")
    if config.cache_config.enable_prefix_caching:
        raise ValueError("step5_pd currently requires --no-enable-prefix-caching.")
    if config.cache_config.cache_dtype != "fp8_e4m3":
        raise ValueError("step5_pd currently requires fp8_e4m3 on both P and D.")
    if config.kv_transfer_config.kv_load_failure_policy != "fail":
        raise ValueError("step5_pd requires kv_load_failure_policy=fail.")
    # Retain the region containing N-1 even when the full prefill closes it.
    # The existing speculative rollback machinery owns these retained rows.
    spec = config.speculative_config
    if spec is None or spec.method != "mtp" or spec.num_speculative_tokens < 1:
        raise ValueError("step5_pd currently requires MTP on both P and D.")


class Step5StateTransfer:
    def __init__(self, worker, caches):
        import torch
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        from vllm_hcu.model_executor.layers.step4_dsa import _INDEXERS

        self.worker = worker
        self.layers = {name: _INDEXERS[name] for name in caches if name in _INDEXERS}
        if not self.layers:
            raise ValueError("step5_pd found no DSA indexer states to transfer.")
        self.caches = caches
        self.received = {}
        self.views = {}
        # The DSA summary is one proxy vector per region, with no KV-group axis
        # (sparse_indexer_num_k_heads=1), so it is identical on every TP rank.
        # These synthetic regions must therefore be copied whole to each
        # consumer rank instead of being sliced by tp_ratio.
        self.replicated_regions: set[str] = set()
        ratio = worker._physical_blocks_per_logical_kv_block
        for name, indexer in self.layers.items():
            cache = caches[name]
            if cache.ndim != 4:
                raise ValueError("step5_pd requires v0.28 four-dimensional DSA pages.")
            state = indexer.state
            state.pd_keep_prefill_tail = worker.is_kv_producer
            state.ensure_allocated(cache, cache.device)
            count, rpp = cache.shape[0], state.regions_per_page
            summary = state.summary[indexer.slot]
            view = summary.reshape(count, rpp, 1, state.geometry.proxy_dim).permute(0, 2, 1, 3)
            key = name + ".step5_summary_v1"
            self.views[key] = view
            self.replicated_regions.add(key)
            worker._layer_specs[key] = FullAttentionSpec(
                block_size=rpp * ratio, num_kv_heads=1,
                head_size=state.geometry.proxy_dim // 2, dtype=torch.float8_e4m3fn,
            )
            worker._layer_group_indices[key] = worker._layer_group_indices[name]

    def _slot(self, name, groups, past):
        indexer = self.layers[name]
        state = indexer.state
        group = self.worker._layer_group_indices[name]
        physical = self.worker._logical_to_kernel_block_ids(groups)[group]
        block_index, offset = divmod(past, state.block_size)
        if block_index >= len(physical):
            raise ValueError("Step5 P/D tail lies outside the transferred block table.")
        block = physical[block_index]
        if not 0 <= block < self.caches[name].shape[0]:
            raise ValueError("Step5 P/D tail block ID is outside cache capacity.")
        return block * state.regions_per_page + offset // state.geometry.region_size

    def capture(self, groups, prompt_tokens):
        """Called on the P worker before another forward can recycle rows."""
        tails = {}
        past = prompt_tokens - 1
        for name, indexer in self.layers.items():
            state = indexer.state
            prefix = past % state.geometry.region_size
            if not prefix:
                tails[name] = b""
                continue
            slot = self._slot(name, groups, past)
            row = state.pending.get(slot)
            import torch
            data = torch.stack((state.pending_key[indexer.slot, row, :prefix],
                                state.pending_z[indexer.slot, row, :prefix]))
            tails[name] = data.contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return tails

    def restore(self, groups, prompt_tokens, tails):
        """Called on D's worker thread, never on the networking event loop.

        With producer PP>1, ``tails`` arrives unioned from every stage: each
        stage captured the layers it owns and ``capture`` always emits an
        entry per owned layer, so the union still covers ``self.layers``
        exactly. The completeness check therefore stays strict.
        """
        import torch

        if set(tails) != set(self.layers):
            raise ValueError("Step5 P/D response does not contain every DSA layer.")
        past = prompt_tokens - 1
        prepared = []
        for name, indexer in self.layers.items():
            state = indexer.state
            prefix = past % state.geometry.region_size
            expected = 2 * prefix * state.geometry.proxy_dim * state.pending_key.element_size()
            payload = tails[name]
            if not isinstance(payload, bytes) or len(payload) != expected:
                raise ValueError(f"Invalid Step5 P/D tail payload for {name}.")
            if not prefix:
                continue
            slot = self._slot(name, groups, past)
            values = torch.frombuffer(bytearray(payload), dtype=state.pending_key.dtype).reshape(
                2, prefix, state.geometry.proxy_dim)
            prepared.append((indexer, slot, prefix, values))
        for indexer, slot, prefix, values in prepared:
            state = indexer.state
            row = state.pending.acquire(slot)
            state.pending_key[indexer.slot, row, :prefix].copy_(values[0])
            state.pending_z[indexer.slot, row, :prefix].copy_(values[1])
        # Publish readiness only after all device writes have completed.
        if prepared and next(iter(self.caches.values())).is_cuda:
            torch.cuda.current_stream().synchronize()
