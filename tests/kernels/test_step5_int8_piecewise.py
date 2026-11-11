# SPDX-License-Identifier: Apache-2.0
"""Step5 INT8 compile dispatch, graph replay and quantization regression tests.

Uses the production LightOp contract and capture sizes from int8_dense.py.
"""
from contextlib import contextmanager

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm_hcu.ops.step5 import int8_dense as dense

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires gfx936 HCU"
)


@contextmanager
def mode(value):
    with override_forward_context(ForwardContext(
        no_compile_layers={}, attn_metadata={}, slot_mapping={},
        cudagraph_runtime_mode=value,
    )):
        yield


def weights(k, n):
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx936"):
        pytest.skip("gfx936 INT8 path")
    torch.manual_seed(73)
    w = (torch.randn(k, n, device="cuda", dtype=torch.bfloat16) / k**0.5)
    q, scale = dense.prepare_weight(w)
    assert q is not None and scale is not None
    return w, q, scale


def accuracy(actual, reference):
    assert actual.shape == reference.shape
    assert actual.dtype == torch.bfloat16 and actual.is_contiguous()
    assert torch.isfinite(actual).all()
    rel = torch.linalg.vector_norm(actual.float() - reference.float()) / torch.linalg.vector_norm(reference.float()).clamp_min(1e-6)
    assert rel.item() < 0.025, rel.item()


@pytest.mark.parametrize('k,n', sorted(dense.SHAPES))
def test_all_capture_sizes_and_bf16_fallback(k, n):
    w, q, scale = weights(k, n)
    with mode(CUDAGraphMode.PIECEWISE):
        for m in dense.TOKENS:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            assert dense.graph_enabled(x)
            accuracy(dense._int8_impl(x, q, scale, n), x @ w)
            use_int8 = n != 16 and (
                (k, n) != (3456, 4096)
                or m in dense._PIECEWISE_SWA_TOKENS
            )
            expected = dense._int8_impl(x, q, scale, n) if use_int8 else x @ w
            torch.testing.assert_close(dense.int8_linear(x, q, scale, n, w), expected, rtol=0, atol=0)
        for m in (0, 3, 17, 513, 1024):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            assert not dense.graph_enabled(x)
            torch.testing.assert_close(dense.int8_linear(x, q, scale, n, w), x @ w, rtol=0, atol=0)
    with mode(CUDAGraphMode.NONE):
        x = torch.randn(64, k, device="cuda", dtype=torch.bfloat16)
        assert not dense.graph_enabled(x)
        torch.testing.assert_close(dense.int8_linear(x, q, scale, n, w), x @ w, rtol=0, atol=0)


@pytest.mark.parametrize('n', [4096, 5012])
def test_compile_in_none_then_piecewise_capture_and_replay(n, monkeypatch):
    k = 3072 if n == 4096 else 4096
    w, q, scale = weights(k, n)
    observed = []
    original = dense._int8_impl
    def tracked(x, weight, scales, width):
        observed.append(x.shape[0])
        return original(x, weight, scales, width)
    monkeypatch.setattr(dense, '_int8_impl', tracked)
    graphs = []
    def backend(gm, inputs):
        assert any(node.target in (torch.ops.vllm.step5_int8_dense,
                                  torch.ops.vllm.step5_int8_dense.default)
                   for node in gm.graph.nodes), gm.code
        graphs.append(gm)
        return torch._inductor.compile(gm, inputs)
    def forward(x):
        if dense.graph_enabled(x):
            return dense.int8_linear(x, q, scale, n, w)
        return x @ w
    compiled = torch.compile(forward, backend=backend, fullgraph=True, dynamic=True)
    x = torch.randn(8, k, device='cuda', dtype=torch.bfloat16)
    with mode(CUDAGraphMode.NONE):
        torch.testing.assert_close(compiled(x), x @ w, rtol=0, atol=0)
    assert graphs and not observed
    for runtime_mode in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL):
        with mode(runtime_mode):
            for m in (16, 64, 368, 512):
                x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
                expected = original(x, q, scale, n)
                torch.testing.assert_close(compiled(x), expected, rtol=0, atol=0)
                assert observed[-1] == m
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = compiled(x)
                x.normal_()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(out, original(x, q, scale, n), rtol=0, atol=0)
                accuracy(out, x @ w)
    with mode(CUDAGraphMode.PIECEWISE):
        x = torch.randn(513, k, device='cuda', dtype=torch.bfloat16)
        count = len(observed)
        torch.testing.assert_close(compiled(x), x @ w, rtol=0, atol=0)
        assert len(observed) == count


@pytest.mark.parametrize('k,n,m', [(4096, 16, 64), (3456, 4096, 128)])
def test_piecewise_slow_shapes_keep_bf16(k, n, m, monkeypatch):
    w, q, scale = weights(k, n)
    def unexpected(*args):
        pytest.fail("PIECEWISE slow shape must use BF16")
    monkeypatch.setattr(dense, '_int8_impl', unexpected)
    compiled = torch.compile(
        lambda x: dense.int8_linear(x, q, scale, n, w),
        fullgraph=True, dynamic=True,
    )
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    with mode(CUDAGraphMode.NONE):
        compiled(x)
    with mode(CUDAGraphMode.PIECEWISE):
        torch.testing.assert_close(compiled(x), x @ w, rtol=0, atol=0)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = compiled(x)
        x.normal_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, x @ w, rtol=0, atol=0)
