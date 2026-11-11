# SPDX-License-Identifier: Apache-2.0
"""Transport correctness, fallback, and graph replay for Step5 dispatch packing."""

from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.distributed.device_communicators.packed_all_gather import (
    _step5_dispatch_layout,
    packed_step5_all_gatherv,
)


def test_other_shapes_and_variable_sizes_do_not_enter_a_collective():
    def forbidden(*args, **kwargs):
        raise AssertionError("fallback must not enter the packed collective")

    comm = SimpleNamespace(world_size=4)
    inputs = [torch.zeros(513, 4096, dtype=torch.uint8),
              torch.zeros(513, 8), torch.zeros(513, 8, dtype=torch.int32),
              torch.ones(513, 1)]
    assert packed_step5_all_gatherv(comm, inputs, forbidden) is None
    inputs = [torch.zeros(64, 4096, dtype=torch.uint8),
              torch.zeros(64, 8), torch.zeros(64, 8, dtype=torch.int32),
              torch.ones(64, 1)]
    assert packed_step5_all_gatherv(comm, inputs, forbidden, sizes=[64, 32, 64, 64]) is None
    assert packed_step5_all_gatherv(comm, inputs, forbidden, dim=1) is None


@pytest.mark.parametrize("rows", [1, 16, 64, 256, 512])
@pytest.mark.parametrize("scale_first", [False, True])
def test_int8_transport_field_orders(rows, scale_first):
    x = torch.empty(rows, 4096, dtype=torch.int8)
    weights = torch.empty(rows, 8, dtype=torch.float32)
    ids = torch.empty(rows, 8, dtype=torch.int64)
    scale = torch.empty(rows, 1, dtype=torch.float32)
    fields = [scale, weights, ids] if scale_first else [weights, ids, scale]
    assert _step5_dispatch_layout([x, *fields]) == "byte-topk"


@pytest.mark.parametrize("logit_dtype", [torch.float32, torch.bfloat16])
def test_w4a8_float_interface_and_all_peer_size_fallback(logit_dtype):
    x = torch.empty(64, 4096, dtype=torch.bfloat16)
    logits = torch.empty(64, 352, dtype=logit_dtype)
    inputs = [x, logits]
    assert _step5_dispatch_layout(inputs) == "float-router"

    def forbidden(*args, **kwargs):
        raise AssertionError("ineligible ranks must not enter a packed collective")

    # Every peer sees the same unequal vector, including the empty rank.
    for rank, rows in enumerate([0, 64, 64, 64]):
        local = [torch.empty(rows, 4096, dtype=torch.bfloat16),
                 torch.empty(rows, 352, dtype=logit_dtype)]
        comm = SimpleNamespace(world_size=4, rank=rank)
        assert packed_step5_all_gatherv(
            comm, local, forbidden, sizes=[0, 64, 64, 64]
        ) is None
    assert _step5_dispatch_layout([x, logits[:, :8]]) is None
    assert _step5_dispatch_layout([x.to(torch.int8), logits]) is None


def test_native_adapter_enables_packing_only_for_opted_in_dp_groups(monkeypatch):
    import vllm.config
    from vllm_hcu.patch.worker.framework_opt import patch_cuda_communicator as adapter

    class Communicator:
        def __init__(self, cpu_group, device=None, device_group=None, unique_name="",
                     global_ranks=None, global_world_size=None, tcp_store_group=None,
                     use_all2all=False):
            self.world_size = cpu_group
            self.use_all2all = use_all2all

        def all_gatherv(self, input_, dim=0, sizes=None):
            return input_

        def destroy(self):
            pass

    module = ModuleType(adapter.TARGET_MODULE)
    module.CudaCommunicator = Communicator
    monkeypatch.setattr(adapter, '_maybe_init_ep_custom_allreduce', lambda *args: None)
    monkeypatch.setattr(adapter, '_maybe_init_dp_ipc_all_gather', lambda obj, name: setattr(obj, '_hcu_ipc_ag', None))
    monkeypatch.setattr(vllm.config, 'get_current_vllm_config_or_none', lambda: None)
    monkeypatch.setattr(adapter.henvs, 'optional_custom_op_enabled', lambda requested: requested)
    assert adapter.apply_to_module(module)
    monkeypatch.delenv('VLLM_HCU_DP_PACKED_AG', raising=False)
    assert not Communicator(4, unique_name='dp:0')._hcu_dp_packed_ag
    monkeypatch.setenv('VLLM_HCU_DP_PACKED_AG', '1')
    assert not Communicator(4, unique_name='dp:0')._hcu_dp_packed_ag
    owner=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type='step5')))
    monkeypatch.setattr(vllm.config, 'get_current_vllm_config_or_none', lambda: owner)
    monkeypatch.setattr(adapter, 'get_hcu_config', lambda _:SimpleNamespace(deepep_auto=False))
    assert Communicator(4, unique_name='dp:0')._hcu_dp_packed_ag
    assert not Communicator(4, unique_name='tp:0')._hcu_dp_packed_ag
    assert not Communicator(1, unique_name='dp:0')._hcu_dp_packed_ag
    monkeypatch.setattr(adapter.henvs, 'optional_custom_op_enabled', lambda requested: False)
    assert not Communicator(4, unique_name='dp:0')._hcu_dp_packed_ag


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_bytes_and_changed_input_graph_replay(id_dtype, strided):
    torch.manual_seed(6479)
    raw = torch.randint(0, 256, (64, 4096), device="cuda", dtype=torch.uint8)
    inputs = [raw.view(torch.float8_e4m3fn),
              torch.randn(64, 8, device="cuda"),
              torch.randint(0, 352, (64, 8), device="cuda", dtype=id_dtype),
              torch.rand(64, 1, device="cuda")]
    if strided:
        inputs = [torch.stack((t, t), dim=-1)[..., 0] for t in inputs]
        assert not inputs[0].is_contiguous()
    calls = []

    def gather(comm, tensor, dim=0, sizes=None):
        calls.append((tensor.dtype, dim, sizes))
        return tensor.repeat(comm.world_size)

    comm = SimpleNamespace(world_size=4)

    def check(outputs):
        for src, out in zip(inputs, outputs):
            expected = torch.cat([src.contiguous()] * 4, dim=0)
            assert src.dtype == out.dtype
            assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))

    check(packed_step5_all_gatherv(comm, inputs, gather))
    assert calls == [(torch.uint8, 0, None)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = packed_step5_all_gatherv(comm, inputs, gather)
    for value in [0, 1]:
        inputs[0].fill_(value)
        for output in outputs:
            output.zero_()
        graph.replay()
        torch.cuda.synchronize()
        check(outputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("rows", [16, 64, 256])
@pytest.mark.parametrize("layout", ["bf16-fp32", "bf16-bf16", "int8-scale-first"])
@pytest.mark.parametrize("strided", [False, True])
def test_dispatch_bytes_rank_order_and_changed_input_graph_replay(rows, layout, strided):
    torch.manual_seed(83)
    if layout.startswith("bf16"):
        # Arbitrary bytes include special floating-point values. The transport
        # must preserve payload bits instead of casting to a common dtype.
        raw = torch.randint(0, 256, (rows, 8192), device="cuda", dtype=torch.uint8)
        x = raw.view(torch.bfloat16)
        dtype = torch.float32 if layout == "bf16-fp32" else torch.bfloat16
        width = 352 * torch.empty((), dtype=dtype).element_size()
        logits = torch.randint(0, 256, (rows, width), device="cuda", dtype=torch.uint8)
        inputs = [x, logits.view(dtype)]
    else:
        inputs = [torch.randint(-128, 128, (rows, 4096), device="cuda", dtype=torch.int8),
                  torch.rand(rows, 1, device="cuda"),
                  torch.randn(rows, 8, device="cuda"),
                  torch.randint(0, 352, (rows, 8), device="cuda", dtype=torch.int64)]
    if strided:
        inputs = [torch.stack((t, t), dim=-1)[..., 0] for t in inputs]
    comm = SimpleNamespace(world_size=4)
    calls = []

    def gather(comm, tensor, dim=0, sizes=None):
        calls.append((tensor.dtype, dim, sizes))
        # Give each simulated rank different bytes to detect rank/field swaps.
        return torch.cat([tensor.bitwise_xor(rank) for rank in range(comm.world_size)])

    def check(outputs):
        for source, output in zip(inputs, outputs):
            source_bytes = source.contiguous().view(torch.uint8)
            expected = torch.cat([source_bytes.bitwise_xor(rank) for rank in range(4)])
            assert output.dtype == source.dtype
            assert output.shape == (4 * rows, source.shape[1])
            assert torch.equal(output.view(torch.uint8), expected)

    check(packed_step5_all_gatherv(comm, inputs, gather, sizes=[rows] * 4))
    assert calls == [(torch.uint8, 0, None)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = packed_step5_all_gatherv(comm, inputs, gather, sizes=[rows] * 4)
    for value in (0, 1):
        for tensor in inputs:
            tensor.fill_(value)
        for output in outputs:
            output.zero_()
        graph.replay()
        torch.cuda.synchronize()
        check(outputs)
