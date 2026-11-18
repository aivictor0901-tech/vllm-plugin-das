# SPDX-License-Identifier: Apache-2.0
"""Compare LightOp replacement with the original Step5 Triton arithmetic."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires target GPU")


def original(x, ids):
    from vllm_hcu.model_executor.layers.fused_moe import ep_fused_moe as ep
    q = torch.empty((x.shape[0], 1536), device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
    if x.shape[0]:
        ep._ep_swiglustep_quant_deepgemm_kernel[(min(4096, (x.shape[0] + 1) // 2),)](
            q, scale, x, ids, x.shape[0], D=1536, LIMIT=original.limit,
            DTYPE_MAX=448., BLOCK_M=2, CH=512, num_warps=1, num_stages=1)
    return q, scale


def compare(x, ids, limit, q=None):
    from vllm_hcu.model_executor.layers.fused_moe.ep_fused_moe import ep_swiglustep_quant_deepgemm
    original.limit = limit
    expected, es = original(x, ids)
    if q is None:
        q = torch.empty_like(expected)
    got, gs = ep_swiglustep_quant_deepgemm(x, ids, q, limit)
    assert got.data_ptr() == q.data_ptr()
    assert torch.isfinite(gs).all()
    padding = ids < 0
    assert torch.count_nonzero(got.view(torch.uint8)[padding]).item() == 0
    torch.testing.assert_close(gs[padding], es[padding], rtol=0, atol=0)
    torch.testing.assert_close(gs, es, rtol=.008, atol=1.e-20)
    a, b = got.float() * gs, expected.float() * es
    relative = (a - b).norm() / b.norm().clamp_min(1.e-30)
    assert relative.item() < .003
    return got, gs


@pytest.mark.parametrize("limit", [1., 7.])
@pytest.mark.parametrize("distribution", ["normal", "zero", "tiny", "wide", "boundary"])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_replacement_parity(limit, distribution, id_dtype):
    torch.manual_seed(732)
    x = torch.randn(257, 3072, device="cuda", dtype=torch.bfloat16)
    if distribution == "zero": x.zero_()
    elif distribution == "tiny": x.mul_(1.e-7)
    elif distribution == "wide": x.mul_(100.)
    elif distribution == "boundary":
        values = torch.tensor([-20., -limit, -.1, 0., limit, limit + .25, 20., 1.],
                              device="cuda", dtype=x.dtype)
        x.copy_(values.repeat(x.numel() // values.numel()).reshape_as(x))
    ids = torch.arange(257, device="cuda", dtype=id_dtype) % 16
    ids[::3] = -1
    ids[1::17] = -2
    x[ids < 0] = float("nan")  # Padding must not be read.
    compare(x, ids, limit)


def test_empty_and_no_triton_dispatch(monkeypatch):
    from vllm_hcu.model_executor.layers.fused_moe import ep_fused_moe as ep
    def forbidden(*a, **kw):
        raise AssertionError("production must use LightOp")
    monkeypatch.setattr(ep, "_ep_swiglustep_quant_deepgemm_kernel", forbidden)
    for n in [0, 1, 17]:
        x = torch.zeros(n, 3072, device="cuda", dtype=torch.bfloat16)
        ids = torch.zeros(n, device="cuda", dtype=torch.int32)
        q = torch.empty(n, 1536, device="cuda", dtype=torch.float8_e4m3fn)
        out, scale = ep.ep_swiglustep_quant_deepgemm(x, ids, q)
        assert out.shape == (n, 1536) and scale.shape == (n, 1)
        assert torch.count_nonzero(out.view(torch.uint8)).item() == 0


def test_stream_graph_padding_updates():
    from lightop.activation import step5_swiglustep_fp8_quant
    x = torch.zeros(17, 3072, device="cuda", dtype=torch.bfloat16)
    ids = torch.full((17,), -1, device="cuda", dtype=torch.int32)
    q = torch.empty(17, 1536, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty(17, 1, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step5_swiglustep_fp8_quant(x, ids, output=q, scales=scales)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            step5_swiglustep_fp8_quant(x, ids, output=q, scales=scales)
        x.fill_(2.)
        ids[::2] = 0
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    ref, rs = compare(x, ids, 7.)
    torch.testing.assert_close(q.float(), ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(scales, rs, atol=0, rtol=0)
