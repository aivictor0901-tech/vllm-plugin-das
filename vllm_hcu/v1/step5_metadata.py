# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Step5 attention metadata and the retained decode graph that consumes it."""

from __future__ import annotations

import ctypes as C
import os
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch

from vllm.v1.attention.backend import (
    CommonAttentionMetadata as UpstreamCommonAttentionMetadata,
)

from vllm.v1.worker.gpu.attn_utils import (
    Any,
    AttentionGroup,
    KVCacheConfig,
    Mapping,
    ModelSpecificAttnMetadata,
    Sequence,
    torch,
)


@dataclass
class CommonAttentionMetadata(UpstreamCommonAttentionMetadata):
    """Host block table mirror scoped to Step5 metadata construction."""

    block_table_np: object | None = None

    def unpadded(self, num_actual_tokens, num_actual_reqs):
        result = super().unpadded(num_actual_tokens, num_actual_reqs)
        result.block_table_np = (
            self.block_table_np[:num_actual_reqs]
            if self.block_table_np is not None
            else None
        )
        return result


def build_attn_metadata(
    attn_groups: list[list[AttentionGroup]],
    num_reqs: int,
    num_tokens: int,
    query_start_loc_gpu: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    max_query_len: int,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    block_tables: Sequence[torch.Tensor],
    slot_mappings: torch.Tensor,
    kv_cache_config: KVCacheConfig,
    seq_lens_cpu_upper_bound: torch.Tensor | None = None,
    dcp_local_seq_lens: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    is_prefilling: torch.Tensor | None = None,
    mm_req_doc_ranges: dict[int, list[tuple[int, int]]] | None = None,
    model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    for_cudagraph_capture: bool = False,
    causal: bool | torch.Tensor | Mapping[int, bool] = True,
    rswa_prefix_lens: torch.Tensor | None = None,
    ubatch_idx: int = 0,
    seq_lens_cpu: torch.Tensor | None = None,
    num_computed_tokens_cpu: torch.Tensor | None = None,
    block_tables_np: Sequence[Any] | None = None,
) -> dict[str, Any]:
    seq_lens = seq_lens[:num_reqs]
    # Exact host mirrors (see InputBatch.seq_lens_cpu); never used for capture,
    # where the dummy batch has none.
    if for_cudagraph_capture:
        seq_lens_cpu = num_computed_tokens_cpu = block_tables_np = None
    if seq_lens_cpu is not None:
        seq_lens_cpu = seq_lens_cpu[:num_reqs]
    if num_computed_tokens_cpu is not None:
        num_computed_tokens_cpu = num_computed_tokens_cpu[:num_reqs]
    if dcp_local_seq_lens is not None:
        dcp_local_seq_lens = dcp_local_seq_lens[:num_reqs]
    if seq_lens_cpu_upper_bound is not None:
        seq_lens_cpu_upper_bound = seq_lens_cpu_upper_bound[:num_reqs]

    attn_metadata: dict[str, Any] = {}
    num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
    for i in range(num_kv_cache_groups):
        block_table = block_tables[i]
        slot_mapping = slot_mappings[i]
        # Per-group causal for hybrid drafters (mixed SWA/full attention).
        group_causal = (
            causal if isinstance(causal, (bool, torch.Tensor)) else causal.get(i, True)
        )

        common_attn_metadata_extra_kwargs = (
            model_specific_attn_metadata.get_extra_common_attn_kwargs(i, num_reqs)
            if model_specific_attn_metadata is not None
            else {}
        )
        # Model-specific metadata (e.g. Mamba hybrid) may supply its own
        # padding-aware is_prefilling, which takes precedence over the default.
        group_is_prefilling = common_attn_metadata_extra_kwargs.pop(
            "is_prefilling", is_prefilling
        )
        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            max_seq_len=max_seq_len,
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            causal=group_causal,
            dcp_local_seq_lens=dcp_local_seq_lens,
            positions=positions,
            is_prefilling=group_is_prefilling,
            mm_req_doc_ranges=mm_req_doc_ranges,
            rswa_prefix_lens=rswa_prefix_lens,
            _seq_lens_cpu=seq_lens_cpu,
            _num_computed_tokens_cpu=num_computed_tokens_cpu,
            block_table_np=(
                block_tables_np[i][:num_reqs] if block_tables_np is not None else None
            ),
            **common_attn_metadata_extra_kwargs,
        )

        for attn_group in attn_groups[i]:
            attn_metadata_builder = attn_group.get_metadata_builder(ubatch_idx)
            if for_cudagraph_capture:
                metadata = attn_metadata_builder.build_for_cudagraph_capture(
                    common_attn_metadata
                )
            else:
                attn_metadata_extra_kwargs = (
                    model_specific_attn_metadata.get_extra_attn_kwargs(
                        attn_metadata_builder,
                        num_reqs,
                    )
                    if model_specific_attn_metadata is not None
                    else {}
                )
                metadata = attn_metadata_builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=common_attn_metadata,
                    **attn_metadata_extra_kwargs,
                )
            for layer_name in attn_group.layer_names:
                attn_metadata[layer_name] = metadata
    return attn_metadata


from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.attn_utils import compute_mm_prefix_ranges


def prepare_step5_attn(
    self,
    input_batch: InputBatch,
    cudagraph_mode: CUDAGraphMode,
    block_tables: tuple[torch.Tensor, ...],
    slot_mappings: torch.Tensor,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    for_capture: bool = False,
    ubatch_idx: int = 0,
) -> dict[str, Any]:
    if cudagraph_mode == CUDAGraphMode.FULL:
        # Use padded sizes - padding is handled by model_runner.prepare_attn.
        num_reqs = input_batch.num_reqs_after_padding
        num_tokens = input_batch.num_tokens_after_padding
    else:
        # For piecewise cudagraphs and eager, use unpadded sizes.
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens
    query_start_loc_cpu = torch.from_numpy(
        input_batch.query_start_loc_np[: num_reqs + 1]
    )
    query_start_loc_gpu = input_batch.query_start_loc[: num_reqs + 1]
    max_query_len = input_batch.max_query_len
    if max_query_len is None:
        max_query_len = input_batch.num_scheduled_tokens.max().item()
    seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
    if for_capture:
        # Capture with worst-case max_seq_len so the graph is valid at any replay.
        max_seq_len = self.max_model_len
    else:
        # This is an explicit CPU mirror, so NumPy avoids constructing a
        # temporary CPU tensor and dispatching reduction/item operators.
        max_seq_len = int(np.max(seq_lens_cpu_upper_bound.numpy()[:num_reqs]))
    req_doc_ranges: dict[int, list[tuple[int, int]]] | None = None
    if (
        self.supports_mm_inputs
        and self.encoder_cache is not None
        and self.model_config.is_mm_prefix_lm
    ):
        req_doc_ranges = compute_mm_prefix_ranges(
            req_ids=input_batch.req_ids,
            mm_features=self.encoder_cache.mm_features,
            sliding_window=self.model_config.get_sliding_window(),
        )
    attn_metadata = build_attn_metadata(
        attn_groups=attn_groups,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        query_start_loc_gpu=query_start_loc_gpu,
        query_start_loc_cpu=query_start_loc_cpu,
        max_query_len=max_query_len,
        seq_lens=input_batch.seq_lens,
        max_seq_len=max_seq_len,
        block_tables=block_tables,
        slot_mappings=slot_mappings,
        kv_cache_config=kv_cache_config,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
        positions=input_batch.positions,
        is_prefilling=torch.from_numpy(input_batch.is_prefilling_np),
        mm_req_doc_ranges=req_doc_ranges,
        for_cudagraph_capture=for_capture,
        rswa_prefix_lens=input_batch.prompt_lens,
        ubatch_idx=ubatch_idx,
        seq_lens_cpu=getattr(input_batch, "seq_lens_cpu", None),
        num_computed_tokens_cpu=getattr(input_batch, "num_computed_tokens_cpu", None),
        block_tables_np=getattr(input_batch, "block_tables_np", None),
    )
    return attn_metadata


# Partition a retained HIP graph at an empty dependency marker. The original
# stays alive, so kernels, parameters and allocations keep their ownership.
class _HipGraphs:
    def __init__(self):
        self.lib = C.CDLL("libamdhip64.so")
        pointer = C.c_void_p
        array = C.POINTER(pointer)
        size = C.POINTER(C.c_size_t)
        signatures = {
            "hipGraphGetNodes": [pointer, array, size],
            "hipGraphGetEdges": [pointer, array, array, size],
            "hipGraphNodeGetType": [pointer, C.POINTER(C.c_int)],
            "hipGraphClone": [array, pointer],
            "hipGraphNodeFindInClone": [array, pointer, pointer],
            "hipGraphDestroyNode": [pointer],
            "hipGraphDestroy": [pointer],
            "hipGraphExecDestroy": [pointer],
            "hipGraphInstantiate": [array, pointer, array, pointer, C.c_size_t],
            "hipGraphLaunch": [pointer, pointer],
            "hipStreamGetCaptureInfo_v2": [
                pointer,
                C.POINTER(C.c_int),
                C.POINTER(C.c_ulonglong),
                array,
                C.POINTER(array),
                size,
            ],
            "hipGraphAddEmptyNode": [array, pointer, array, C.c_size_t],
            "hipStreamUpdateCaptureDependencies": [
                pointer,
                array,
                C.c_size_t,
                C.c_uint,
            ],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, C.c_int
        self.lib.hipGetErrorString.argtypes = [C.c_int]
        self.lib.hipGetErrorString.restype = C.c_char_p

    def call(self, name, *args):
        error = getattr(self.lib, name)(*args)
        if error:
            reason = self.lib.hipGetErrorString(error).decode()
            raise RuntimeError(f"{name}: {error} ({reason})")

    def marker(self, stream):
        status, capture_id = C.c_int(), C.c_ulonglong()
        graph, deps, count = C.c_void_p(), C.POINTER(C.c_void_p)(), C.c_size_t()
        self.call(
            "hipStreamGetCaptureInfo_v2",
            stream,
            C.byref(status),
            C.byref(capture_id),
            C.byref(graph),
            C.byref(deps),
            C.byref(count),
        )
        if status.value != 1 or not graph.value:
            raise RuntimeError(
                "Step5 boundary marker requires an active stream capture"
            )
        dependencies = (C.c_void_p * count.value)(
            *[deps[i] for i in range(count.value)]
        )
        marker = C.c_void_p()
        self.call("hipGraphAddEmptyNode", C.byref(marker), graph, dependencies, count)
        dependency = (C.c_void_p * 1)(marker.value)
        # Set (replace), rather than add, the stream's current dependency set.
        self.call("hipStreamUpdateCaptureDependencies", stream, dependency, 1, 0)
        return graph.value, marker.value

    def topology(self, graph):
        count = C.c_size_t()
        self.call("hipGraphGetNodes", graph, None, C.byref(count))
        nodes = (C.c_void_p * count.value)()
        self.call("hipGraphGetNodes", graph, nodes, C.byref(count))
        types = {}
        for node in nodes:
            kind = C.c_int()
            self.call("hipGraphNodeGetType", node, C.byref(kind))
            # DTK's stream-captured Torch/HSA kernels have vendor type200.
            # They remain opaque: clone their native nodes and never rewrite
            # kernel parameters. The multistream replay test audits this path.
            # Host callbacks, external semaphores, graph allocations and
            # conditional nodes require a separate ownership/dependency audit.
            # Audited immediately before partitioning, after collecting all nodes.
            types[node] = kind.value
        edges = C.c_size_t()
        self.call("hipGraphGetEdges", graph, None, None, C.byref(edges))
        source, target = (C.c_void_p * edges.value)(), (C.c_void_p * edges.value)()
        self.call("hipGraphGetEdges", graph, source, target, C.byref(edges))
        return types, list(zip(source, target))

    def clone_subset(self, graph, nodes, keep):
        clone = C.c_void_p()
        self.call("hipGraphClone", C.byref(clone), graph)
        try:
            # Find every clone node before destroying anything: cloning lookup
            # must not depend on an already-mutated topology.
            mapped = {}
            for node in nodes:
                result = C.c_void_p()
                self.call("hipGraphNodeFindInClone", C.byref(result), node, clone)
                mapped[node] = result.value
            for node, copied in mapped.items():
                if node not in keep:
                    self.call("hipGraphDestroyNode", copied)
            executable = C.c_void_p()
            self.call("hipGraphInstantiate", C.byref(executable), clone, None, None, 0)
            return _NativeGraph(self, clone.value, executable.value)
        except BaseException:
            self.call("hipGraphDestroy", clone)
            raise


class _NativeGraph:
    def __init__(self, api, graph, executable):
        self.api, self.graph, self.executable = api, graph, executable

    def replay(self, stream):
        self.api.call("hipGraphLaunch", self.executable, stream)

    def close(self):
        if self.executable:
            self.api.call("hipGraphExecDestroy", self.executable)
            self.executable = None
        if self.graph:
            self.api.call("hipGraphDestroy", self.graph)
            self.graph = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _partition_graph(api, graph, marker):
    types, edges = api.topology(graph)
    if any(t not in (0, 1, 2, 4, 5, 6, 7, 200) for t in types.values()):
        from collections import Counter

        raise RuntimeError(
            f"Unsupported Step5 graph topology: {dict(Counter(types.values()))}"
        )
    if types.get(marker) != 5:
        raise RuntimeError("Step5 graph boundary is not the expected empty node")
    parents = {node: [] for node in types}
    for source, target in edges:
        parents[target].append(source)
    ancestors, todo = set(), list(parents[marker])
    while todo:
        node = todo.pop()
        if node not in ancestors:
            ancestors.add(node)
            todo.extend(parents[node])
    suffix = set(types) - ancestors - {marker}
    if not ancestors or not suffix:
        raise RuntimeError("Step5 graph boundary produced an empty partition")
    if any(source in suffix and target in ancestors for source, target in edges):
        raise RuntimeError("Step5 prefix is not closed under graph dependencies")
    prefix_graph = api.clone_subset(graph, types, ancestors)
    try:
        suffix_graph = api.clone_subset(graph, types, suffix)
    except BaseException:
        prefix_graph.close()
        raise
    return (
        prefix_graph,
        suffix_graph,
        dict(
            original_nodes=len(types),
            prefix_nodes=len(ancestors),
            suffix_nodes=len(suffix),
            marker_nodes=1,
        ),
    )


ENABLED = os.environ.get("HCU_STEP5_METADATA_OVERLAP", "0") == "1"
# The public switch selects the complete validated metadata/overlap pipeline.
EARLY_SUFFIX = ENABLED
_capture = None
_metadata_consumer = ContextVar("step5_metadata_consumer", default=None)


class _RetainedGraph:
    def __init__(self, capture, original, *args, **kwargs):
        kwargs["keep_graph"] = True
        self.graph = original(*args, **kwargs)
        self.capture = capture
        self.marker = None
        self.prefix = self.suffix = None
        self.pending = False
        self.suffix_submitted = False
        self.stream = None
        self.stats = None
        self.profile = False
        self.prefix_replays = self.suffix_replays = self.full_replays = 0

    def __getattr__(self, name):
        return getattr(self.graph, name)

    def capture_begin(self, *args, **kwargs):
        if self.capture.active is not None:
            raise RuntimeError("Nested Step5 full-graph capture")
        result = self.graph.capture_begin(*args, **kwargs)
        self.capture.active = self
        return result

    def capture_end(self):
        try:
            return self.graph.capture_end()
        finally:
            self.capture.active = None

    def finalize(self):
        self.graph.instantiate()
        if self.marker is not None:
            raw, marker = self.marker
            if raw != self.graph.raw_cuda_graph():
                raise RuntimeError("Step5 marker belongs to a different graph")
            self.prefix, self.suffix, self.stats = _partition_graph(
                self.capture.api, raw, marker
            )
        self.capture = None

    def start(self):
        if self.pending or self.prefix is None:
            raise RuntimeError("Step5 prefix replay is missing or already pending")
        scope = (
            torch.profiler.record_function("step5_prefix_graph")
            if self.profile
            else nullcontext()
        )
        with scope:
            self.stream = torch.cuda.current_stream()
            self.prefix.replay(self.stream.cuda_stream)
        self.pending = True
        self.suffix_submitted = False
        self.prefix_replays += 1

    def submit_suffix(self):
        if not self.pending or self.suffix_submitted:
            raise RuntimeError(
                "Step5 suffix is missing a prefix or was submitted twice"
            )
        scope = (
            torch.profiler.record_function("step5_suffix_graph")
            if self.profile
            else nullcontext()
        )
        with scope:
            self.suffix.replay(self.stream.cuda_stream)
        self.suffix_submitted = True
        self.suffix_replays += 1

    def replay(self):
        if self.pending:
            if not self.suffix_submitted:
                self.submit_suffix()
            self.pending = self.suffix_submitted = False
        else:
            self.graph.replay()
            self.full_replays += 1


class _Proxy:
    def __init__(self, original, **overrides):
        self.original, self.overrides = original, overrides

    def __getattr__(self, name):
        return (
            self.overrides[name]
            if name in self.overrides
            else getattr(self.original, name)
        )


class _Capture:
    def __init__(self):
        self.active = None
        self.api = _HipGraphs()
        self.graphs = []

    def create(self, *args, **kwargs):
        result = _RetainedGraph(self, torch.cuda.CUDAGraph, *args, **kwargs)
        self.graphs.append(result)
        return result


@contextmanager
def retain_graphs(manager):
    """Intercept this manager's constructors, without changing global torch."""
    global _capture
    from vllm.v1.worker.gpu import cudagraph_utils

    if _capture is not None:
        raise RuntimeError("Overlapping Step5 capture contexts")
    original = cudagraph_utils.torch
    capture = _Capture()
    cudagraph_utils.torch = _Proxy(
        original,
        cuda=_Proxy(original.cuda, CUDAGraph=capture.create),
    )
    _capture = capture
    try:
        yield
    finally:
        cudagraph_utils.torch = original
        _capture = None
    for graph in capture.graphs:
        graph.finalize()
    pairs = {
        desc: graph
        for desc, graph in manager.graphs.items()
        if isinstance(graph, _RetainedGraph) and graph.prefix is not None
    }
    if not pairs:
        raise RuntimeError("No Step5 metadata boundary found in captured target graphs")
    manager._step5_boundary_pairs = pairs
    manager._step5_boundary_enabled = True
    manager._step5_boundary_early = EARLY_SUFFIX


def mark_indexer_boundary():
    capture = _capture
    if capture is None or capture.active is None:
        return
    graph = capture.active
    if graph.marker is not None:
        raise RuntimeError("More than one first-indexer boundary in a target graph")
    graph.marker = capture.api.marker(torch.cuda.current_stream().cuda_stream)


def start_prefix(runner, desc):
    manager = runner.cudagraph_manager
    if not getattr(manager, "_step5_boundary_enabled", False):
        return False
    graph = manager._step5_boundary_pairs.get(desc)
    if graph is None:
        raise RuntimeError(f"Missing Step5 prefix for full-graph descriptor {desc}")
    from vllm.model_executor.offloader.base import get_offloader

    get_offloader().sync_prev_onload()
    graph.start()
    return True


@contextmanager
def metadata_consumer_scope(graph):
    if not graph.pending:
        raise RuntimeError("Metadata consumer requires a submitted prefix")
    token = _metadata_consumer.set(graph)
    try:
        yield
    finally:
        _metadata_consumer.reset(token)


def early_consumer_active():
    return _metadata_consumer.get() is not None


def metadata_ready():
    graph = _metadata_consumer.get()
    if graph is not None:
        graph.submit_suffix()
