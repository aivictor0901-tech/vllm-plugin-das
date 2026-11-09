# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Step5 packed INT4 W4A8 MoE through BoltOps: argument contract of the runtime."""

from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.quantization import (
    compressed_tensors_moe_runtime as runtime,
)

EXPERTS, HIDDEN, INTER = 4, 16, 8


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return torch.full_like(args[0], 3)


@pytest.fixture
def experts(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(runtime, "_step5_boltops_packed_experts", lambda: recorder)
    return recorder


def _setup(monkeypatch, activation="swiglustep", **layer_attrs):
    w1 = torch.zeros(EXPERTS, 2 * INTER, HIDDEN // 2, dtype=torch.int8)
    w2 = torch.zeros(EXPERTS, HIDDEN, INTER // 2, dtype=torch.int8)
    monkeypatch.setattr(
        runtime,
        "_slimquant_w4a8_metadata",
        lambda method, layer: (w1, w2, HIDDEN, activation),
    )
    w1_scale = torch.rand(EXPERTS, 2 * INTER)
    w2_scale = torch.rand(EXPERTS, HIDDEN)
    method = SimpleNamespace(
        moe=SimpleNamespace(num_experts=EXPERTS),
        moe_quant_config=SimpleNamespace(w1_scale=w1_scale, w2_scale=w2_scale),
    )
    layer = SimpleNamespace(**layer_attrs)
    return method, layer, w1, w2, w1_scale, w2_scale


def _inputs(tokens=3, topk=2, dtype=torch.bfloat16):
    hidden = torch.randn(tokens, HIDDEN).to(dtype)
    ids = torch.arange(tokens * topk).reshape(tokens, topk) % EXPERTS
    weights = torch.full((tokens, topk), 0.5, dtype=torch.float16)
    return hidden, weights, ids


def test_apply_passes_packed_weights_and_unscaled_checkpoint_scales(
    monkeypatch, experts
):
    method, layer, w1, w2, w1_scale, w2_scale = _setup(monkeypatch)
    hidden, weights, ids = _inputs()

    out = runtime.apply_step5_boltops_w4a8_moe(method, layer, hidden, weights, ids)

    assert torch.equal(out, torch.full_like(hidden, 3))
    (args, kwargs), = experts.calls
    assert args[1] is w1 and args[2] is w2
    assert args[0].dtype == torch.bfloat16 and args[0].is_contiguous()
    assert args[3].dtype == torch.float32 and args[4].dtype == torch.int32
    # The checkpoint scale is the true scale: it must not get the x16 of the
    # unpacked fallback path.
    assert kwargs["w1_scale"] is w1_scale and kwargs["w2_scale"] is w2_scale
    assert kwargs["use_int4_w4a8"] is True
    assert kwargs["per_channel_quant"] is True
    assert kwargs["is_gated"] is True
    assert kwargs["inplace"] is False
    assert kwargs["output_dtype"] == torch.bfloat16
    assert kwargs["routed_scaling_factor"] == 1.0


@pytest.mark.parametrize(
    "activation, swiglu_limit, expected",
    [
        ("swiglustep", None, 7.0),
        ("swiglustep", 0, 7.0),
        ("swiglustep", 5.5, 5.5),
        ("silu", 5.5, None),
    ],
)
def test_apply_clamp_limit_only_applies_to_swiglustep(
    monkeypatch, experts, activation, swiglu_limit, expected
):
    method, layer, *_ = _setup(
        monkeypatch, activation=activation, swiglu_limit=swiglu_limit
    )
    hidden, weights, ids = _inputs()

    runtime.apply_step5_boltops_w4a8_moe(method, layer, hidden, weights, ids)

    (_, kwargs), = experts.calls
    assert kwargs["activation"] == activation
    assert kwargs["gemm1_limit"] == expected


def test_apply_global_expert_count_precedence(monkeypatch, experts):
    method, layer, *_ = _setup(monkeypatch)
    hidden, weights, ids = _inputs()
    runtime.apply_step5_boltops_w4a8_moe(method, layer, hidden, weights, ids)
    assert experts.calls[-1][1]["global_num_experts"] == EXPERTS

    expert_map = torch.arange(32, dtype=torch.int32)
    layer.global_num_experts = 32
    layer._expert_map = expert_map
    runtime.apply_step5_boltops_w4a8_moe(method, layer, hidden, weights, ids)
    kwargs = experts.calls[-1][1]
    assert kwargs["global_num_experts"] == 32
    assert kwargs["expert_map"] is expert_map


def test_apply_empty_batch_skips_the_kernel(monkeypatch, experts):
    method, layer, *_ = _setup(monkeypatch)
    hidden = torch.empty(0, HIDDEN, dtype=torch.bfloat16)
    weights = torch.empty(0, 2)
    ids = torch.empty(0, 2, dtype=torch.int64)

    out = runtime.apply_step5_boltops_w4a8_moe(method, layer, hidden, weights, ids)

    assert out.shape == hidden.shape and out.dtype == hidden.dtype
    assert experts.calls == []


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda h, w, i: (h[0], w, i), "rank-2"),
        (lambda h, w, i: (h, w[:, :1], i), "rank-2"),
        (lambda h, w, i: (h, w[:2], i[:2]), "rank-2"),
        (lambda h, w, i: (h.float(), w, i), "BF16/FP16"),
        (lambda h, w, i: (torch.randn(3, HIDDEN + 2).bfloat16(), w, i), "logical K"),
    ],
)
def test_apply_rejects_invalid_inputs(monkeypatch, experts, mutate, message):
    method, layer, *_ = _setup(monkeypatch)
    hidden, weights, ids = mutate(*_inputs())

    with pytest.raises(runtime.HcuCompressedTensorsMoeError, match=message):
        runtime.apply_step5_boltops_w4a8_moe(method, layer, hidden, weights, ids)

    assert experts.calls == []


def test_apply_requires_scales_in_the_quant_config(monkeypatch, experts):
    method, layer, *_ = _setup(monkeypatch)
    method.moe_quant_config = SimpleNamespace(w1_scale=None, w2_scale=None)
    hidden, weights, ids = _inputs()

    with pytest.raises(runtime.HcuCompressedTensorsMoeError, match="w1_scale"):
        runtime.apply_step5_boltops_w4a8_moe(method, layer, hidden, weights, ids)


def test_prepare_keeps_packed_weights_and_marks_native_layout(monkeypatch, experts):
    method, layer, w1, w2, *_ = _setup(monkeypatch)
    quant_config = object()
    method.get_fused_moe_quant_config = lambda owner: quant_config
    marked = []
    monkeypatch.setattr(runtime, "mark_aiter_moe_native_layout", marked.append)
    layer.w13_weight, layer.w2_weight = w1, w2

    runtime.prepare_step5_boltops_w4a8_moe(method, layer)

    assert method._supports_int8_allgatherv_input is False
    assert method.moe_quant_config is quant_config
    assert marked == [layer]
    # No expansion to INT8: the packed tensors are untouched.
    assert layer.w13_weight is w1 and layer.w2_weight is w2


def test_prepare_fails_at_load_when_boltops_lacks_the_interface(monkeypatch):
    method, layer, *_ = _setup(monkeypatch)

    def missing():
        raise RuntimeError("BoltOps packed W4A8 swiglustep interface is missing")

    monkeypatch.setattr(runtime, "_step5_boltops_packed_experts", missing)

    with pytest.raises(RuntimeError, match="interface is missing"):
        runtime.prepare_step5_boltops_w4a8_moe(method, layer)
    assert not hasattr(method, "_supports_int8_allgatherv_input")
