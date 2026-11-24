"""Step5 P/D state handoff for matching-TP instances (DCP=1 on D; plain PCP
without DSA cache layer-split is allowed on P — see validate_config()).

Ordinary KV pages alone do not contain the DSA summaries or partial regions.
Summaries use the existing block-addressed RDMA transfer. Small partial-region
buffers travel in the control response and are restored on the worker thread
before reporting receive completion to the scheduler.
"""

from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

_DIAG_LOG = "/workspace/step5/pd_split/capture_diag.log"
_DIAG_VERBOSE = "/workspace/step5/pd_split/.capture_diag_verbose"


def _diag(message: str) -> None:
    try:
        import time

        with open(_DIAG_LOG, "a", encoding="utf-8") as handle:
            handle.write("%.3f %s" % (time.time(), message) + chr(10))
    except Exception:  # noqa: BLE001
        pass


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
    #
    # Decode-side context parallelism is not allowed: restore()'s tail-slot
    # math assumes D's own local block table already covers the whole
    # sequence, which a context-parallel D would violate.
    if getattr(parallel, "decode_context_parallel_size", 1) != 1:
        raise ValueError("step5_pd requires decode_context_parallel_size=1.")
    # Plain prefill-context-parallelism (PCP, no --enable-dsa-cache-layer-split)
    # is allowed on the producer: PCPContextManager's zig-zag chunk assignment
    # (pcp_manager.py rank_segments()) always gives PCP rank 0 the chunk that
    # contains the final prompt token, for any pcp_size. So only PCP rank 0
    # ever needs to run capture()/send a DSA tail; MooncakeConnectorWorker's
    # existing `_pcp_dummy_sender` gating (built for the ordinary KV-block
    # path) is reused on the producer to suppress capture() on the other
    # ranks, and the consumer only ever queries PCP rank 0's address when the
    # producer isn't using DSA cache layer-split.
    #
    # DSA cache layer-split (--enable-dsa-cache-layer-split) uses a different
    # ownership model (each PCP rank durably owns a contiguous layer range,
    # not a token range) that this capture()/restore() tail math has not been
    # analyzed against, so that combination stays rejected.
    if getattr(parallel, "prefill_context_parallel_size", 1) != 1:
        from vllm_hcu.v1.dsa_cache_layer_split import (
            is_dsa_cache_layer_split_requested,
        )

        if is_dsa_cache_layer_split_requested(config):
            raise ValueError(
                "step5_pd does not support prefill_context_parallel_size>1 "
                "together with --enable-dsa-cache-layer-split."
            )
    if config.cache_config.enable_prefix_caching:
        raise ValueError("step5_pd currently requires --no-enable-prefix-caching.")
    if config.cache_config.cache_dtype != "fp8_e4m3":
        raise ValueError("step5_pd currently requires fp8_e4m3 on both P and D.")
    if config.kv_transfer_config.kv_load_failure_policy != "fail":
        raise ValueError("step5_pd requires kv_load_failure_policy=fail.")
    # Retain the region containing N-1 even when the full prefill closes it.
    # That retention is driven by pd_keep_prefill_tail on the producer (see
    # step4_dsa.py), independent of speculative decoding, so MTP is optional.
    # If speculation IS enabled it must be the built-in MTP path this handoff
    # was validated against.
    spec = config.speculative_config
    if spec is not None and (
        getattr(spec, "method", None) != "mtp"
        or int(getattr(spec, "num_speculative_tokens", 0)) < 1
    ):
        raise ValueError("step5_pd requires built-in MTP when speculation is enabled.")


_TAIL_SKIP_WARNED = False


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
        released: dict[tuple[int, int], tuple[object, int]] = {}
        past = prompt_tokens - 1
        for name, indexer in self.layers.items():
            state = indexer.state
            prefix = past % state.geometry.region_size
            if not prefix:
                tails[name] = b""
                continue
            slot = self._slot(name, groups, past)
            table = state.pending
            mapped = len(table._row_of_slot)
            free = sum(1 for s in table._slot_of_row if s < 0)
            if os.environ.get("STEP5_PD_CAPTURE_LOG", "0") == "1" or os.path.exists(_DIAG_VERBOSE):
                logger.info(
                    "step5_pd capture: layer=%s prompt=%s prefix=%s slot=%s "
                    "table_step=%s capacity=%s mapped=%s free=%s row=%s",
                    name, prompt_tokens, prefix, slot, table.step,
                    table.capacity, mapped, free,
                    table._row_of_slot.get(int(slot), -1),
                )
            try:
                row = table.get(slot)
            except RuntimeError:
                sched = getattr(state, "_schedule", None)
                try:
                    plans = [
                        f"row={p.row} past={p.past} qlen={p.qlen} tail={p.tail} "
                        f"pin={p.pending_row_in} pout={p.pending_row_out} "
                        f"keep={p.keep_rows}"
                        for p in (sched.requests if sched is not None else [])
                    ]
                except Exception:  # noqa: BLE001
                    plans = ["<unavailable>"]
                logger.error(
                    "step5_pd capture miss: layer=%s prompt_tokens=%s past=%s "
                    "prefix=%s slot=%s layer_slot=%s groups0=%s "
                    "table_step=%s capacity=%s mapped=%s free=%s row=%s "
                    "plans=%s",
                    name, prompt_tokens, past, prefix, slot, indexer.slot,
                    groups[0] if groups else None, table.step, table.capacity,
                    mapped, free, table._row_of_slot.get(int(slot), -1), plans,
                )
                _diag(
                    "MISS layer=%s prompt=%s past=%s prefix=%s slot=%s "
                    "layer_slot=%s groups0=%s step=%s cap=%s mapped=%s free=%s "
                    "row=%s plans=%s"
                    % (name, prompt_tokens, past, prefix, slot, indexer.slot,
                       groups[0] if groups else None, table.step, table.capacity,
                       mapped, free, table._row_of_slot.get(int(slot), -1), plans)
                )
                raise
            import torch
            data = torch.stack((state.pending_key[indexer.slot, row, :prefix],
                                state.pending_z[indexer.slot, row, :prefix]))
            tails[name] = data.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            released[(id(table), slot)] = (table, slot)
        # The request is finished and its tail has been snapshotted, so the
        # pending row is dead. Releasing it here (instead of leaving it to
        # least-recently-touched recycling) keeps the table proportional to
        # in-flight requests; a leaked row per finished request eventually
        # fills the table and evicts rows that a later capture still needs.
        for table, slot in released.values():
            table.release(slot)
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
            global _TAIL_SKIP_WARNED
            try:
                slot = self._slot(name, groups, past)
            except ValueError:
                # The decoder replays the whole trailing region from a region
                # boundary (see _get_remote_prefill_token_count), so the
                # producer tail is redundant. For prompts whose last partial
                # region starts in a block the pull metadata did not carry
                # (a long prompt's final block), skip it instead of failing.
                if not _TAIL_SKIP_WARNED:
                    _TAIL_SKIP_WARNED = True
                    logger.warning(
                        "step5_pd: producer tail for %s falls outside the "
                        "transferred block table; the region is replayed "
                        "locally, so the tail is skipped.",
                        name,
                    )
                continue
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
