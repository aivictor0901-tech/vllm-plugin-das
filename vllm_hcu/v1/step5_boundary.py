# SPDX-License-Identifier: Apache-2.0
"""Submit metadata-independent graph work before building the DSA schedule."""

import os
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

import torch

from .hip_graph_partition import HipGraphs, partition

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
            self.prefix, self.suffix, self.stats = partition(
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
        self.api = HipGraphs()
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
