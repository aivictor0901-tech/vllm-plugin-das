"""Backend selection, dependency guards and graph metadata without GPU execution."""
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.ops.step5 import w4a8 as runtime


@pytest.fixture(autouse=True)
def clear_dependency_cache():
    runtime._require_deepgemm_kernel.cache_clear()
    yield
    runtime._require_deepgemm_kernel.cache_clear()


@pytest.mark.parametrize("invalid_backend", ["native", "typo"])
def test_backend_defaults_to_deepgemm_and_rejects_other_names(monkeypatch, invalid_backend):
    monkeypatch.delenv("VLLM_HCU_STEP5_W4A8_GEMM_BACKEND", raising=False)
    assert runtime.gemm_backend() == "deepgemm"
    monkeypatch.setenv("VLLM_HCU_STEP5_W4A8_GEMM_BACKEND", "deepgemm")
    assert runtime.gemm_backend() == "deepgemm"
    monkeypatch.setenv("VLLM_HCU_STEP5_W4A8_GEMM_BACKEND", invalid_backend)
    with pytest.raises(ValueError, match="only deepgemm"):
        runtime.gemm_backend()
    with pytest.raises(ValueError, match="only deepgemm"):
        runtime.require_gemm_op(invalid_backend)


def test_default_dispatch_does_not_require_the_removed_plugin_gemm(monkeypatch):
    monkeypatch.delenv("VLLM_HCU_STEP5_W4A8_GEMM_BACKEND", raising=False)
    kernel = object()
    monkeypatch.setitem(sys.modules, "deepgemm", SimpleNamespace(
        m_grouped_w4a8_gemm_nt_contiguous_hipc=kernel))
    monkeypatch.setitem(sys.modules, "deepgemm.version", SimpleNamespace(
        build_code_object_version=5, build_target_arch="gfx938"))
    original_import = runtime.importlib.import_module

    def without_native_extension(name, *args, **kwargs):
        if name == "vllm_hcu.hcu_ops":
            raise AssertionError("W4A8 must not require the removed native GEMM")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(runtime.importlib, "import_module", without_native_extension)
    assert runtime.require_gemm_op() is runtime.register_deepgemm_op()


@pytest.mark.parametrize("code_object", [None, 4, 6])
def test_unvalidated_wheel_is_rejected_before_a_kernel_call(monkeypatch, code_object):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dependency checking must not execute a kernel")

    monkeypatch.setitem(sys.modules, "deepgemm", SimpleNamespace(
        m_grouped_w4a8_gemm_nt_contiguous_hipc=forbidden))
    monkeypatch.setitem(sys.modules, "deepgemm.version", SimpleNamespace(
        build_code_object_version=code_object, build_target_arch="gfx938"))
    with pytest.raises(RuntimeError, match="CO5"):
        runtime._require_deepgemm_kernel()


def test_validated_wheel_is_selected_without_a_gpu_call(monkeypatch):
    kernel = object()
    monkeypatch.setitem(sys.modules, "deepgemm", SimpleNamespace(
        m_grouped_w4a8_gemm_nt_contiguous_hipc=kernel))
    monkeypatch.setitem(sys.modules, "deepgemm.version", SimpleNamespace(
        build_code_object_version=5, build_target_arch="gfx938"))
    assert runtime._require_deepgemm_kernel() is kernel


def test_wrapper_preserves_packed_owners_and_scale_values(monkeypatch):
    a = torch.zeros(64, 64, dtype=torch.int8)
    w = torch.zeros(2, 32, 32, dtype=torch.int8)
    out = torch.empty(64, 32, dtype=torch.bfloat16)
    xs = torch.full((64, 1), .125)
    ws = torch.full((2, 32, 1), .25)
    ids = torch.zeros(64, dtype=torch.int32)
    seen = []

    def kernel(inputs, weights, output, indices, *, config):
        seen.append((inputs, weights, output, indices, config))

    monkeypatch.setattr(runtime, "_require_deepgemm_kernel", lambda: kernel)
    assert runtime._deepgemm_gemm(a, w, out, xs, ws, ids) is None
    inputs, weights, output, indices, config = seen[0]
    assert inputs[0] is a and inputs[1] is xs
    assert weights[0] is w and weights[1] is ws
    assert output is out and indices is ids and config == {"MODE": 1002}
    torch.testing.assert_close(xs, torch.full_like(xs, .125), rtol=0, atol=0)
    torch.testing.assert_close(ws, torch.full_like(ws, .25), rtol=0, atol=0)


def tensors(m=16):
    return (
        torch.empty(m, 4096, dtype=torch.int8, device="meta"),
        torch.empty(44, 3072, 2048, dtype=torch.int8, device="meta"),
        torch.empty(m, 3072, dtype=torch.bfloat16, device="meta"),
        torch.empty(m, 1, dtype=torch.float32, device="meta"),
        torch.empty(44, 3072, 1, dtype=torch.float32, device="meta"),
        torch.empty(m, dtype=torch.int32, device="meta"),
    )


def test_meta_dispatch_preserves_output_and_mutation_schema():
    op = runtime.register_deepgemm_op()
    args = tensors()
    assert op(*args) is None
    output_arg = next(arg for arg in op._schema.arguments if arg.name == "output")
    assert output_arg.alias_info is not None and output_arg.alias_info.is_write
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "vllm::step5_deepgemm_w4a8_gemm", "CUDA")
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "vllm::step5_deepgemm_w4a8_gemm", "Meta")


def test_unbacked_batch_can_be_traced_without_deepgemm_or_gpu():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    op = runtime.register_deepgemm_op()
    mode = FakeTensorMode(shape_env=ShapeEnv())
    with mode:
        m = mode.shape_env.create_unbacked_symint()
        assert op(*tensors(m)) is None


def test_torch_compile_keeps_the_mutating_gemm_as_one_graph_op():
    op = runtime.register_deepgemm_op()
    graphs = []

    def capture(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def forward(*args):
        op(*args)
        return args[2]

    compiled = torch.compile(forward, backend=capture, fullgraph=True)
    args = tensors()
    assert compiled(*args) is args[2]
    assert len(graphs) == 1
    assert sum(node.target == op for node in graphs[0].graph.nodes) == 1
