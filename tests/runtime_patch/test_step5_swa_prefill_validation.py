# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""gfx936 Step5 SWA prefill: host-side validation and launch geometry (no device)."""

import pytest
import torch

from vllm_hcu.v1.attention.backends import swa_prefill

HEADS, KV_HEADS, DIM, PAGE = 16, 1, 192, 64


class _Kernel:
    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append((grid, args, kwargs))

        return launch


@pytest.fixture
def kernel(monkeypatch):
    fake = _Kernel()
    monkeypatch.setattr(swa_prefill, "_swa_prefill_kernel", fake)
    return fake


def _tensors(tokens=3, heads=HEADS, kv_heads=KV_HEADS, page=PAGE, dim=DIM):
    q = torch.zeros(tokens, heads, dim, dtype=torch.bfloat16)
    k = torch.zeros(2, page, kv_heads, dim, dtype=torch.float8_e4m3fn)
    v = torch.zeros_like(k)
    out = torch.zeros_like(q)
    cu = torch.tensor([0, tokens], dtype=torch.int32)
    seqused = torch.tensor([tokens], dtype=torch.int32)
    table = torch.zeros(1, 2, dtype=torch.int32)
    return q, k, v, out, cu, seqused, table


def _call(q, k, v, out, cu, seqused, table, max_q=3, window=127):
    return swa_prefill.swa_prefill_fp8_d192(
        q, k, v, out, cu, seqused, table, max_q, 0.125, window
    )


def test_valid_call_launches_one_program_grid_per_request_head_and_query_tile(kernel):
    tensors = _tensors(tokens=10)

    result = _call(*tensors, max_q=10)

    assert result is tensors[3]
    (grid, args, kwargs), = kernel.launches
    # 64-row dot tile over 16 query heads per KV head: 4 query tokens per tile.
    assert grid == (1, KV_HEADS, 3)
    assert kwargs["GROUP"] == 16 and kwargs["QT"] == 4
    assert kwargs["PAGE"] == PAGE and kwargs["WINDOW"] == 127
    assert kwargs["BM"] == 64 and kwargs["BN"] == 32


@pytest.mark.parametrize("page", [64, 128])
def test_supported_page_sizes(kernel, page):
    _call(*_tensors(page=page))
    assert kernel.launches[-1][2]["PAGE"] == page


def test_window_of_zero_is_allowed(kernel):
    _call(*_tensors(), window=0)
    assert kernel.launches[-1][2]["WINDOW"] == 0


def test_empty_query_returns_output_without_launching(kernel):
    tensors = _tensors(tokens=0)
    assert _call(*tensors, max_q=0) is tensors[3]
    assert kernel.launches == []


def _with(index, value):
    def mutate(tensors):
        tensors = list(tensors)
        tensors[index] = value(tensors[index])
        return tensors

    return mutate


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_with(0, lambda q: q.float()), id="q-not-bf16"),
        pytest.param(_with(3, lambda o: o.half()), id="out-not-bf16"),
        pytest.param(_with(1, lambda k: k.to(torch.bfloat16)), id="k-not-e4m3fn"),
        pytest.param(_with(2, lambda v: v.to(torch.bfloat16)), id="v-not-e4m3fn"),
        pytest.param(lambda t: _tensors(dim=128), id="head-dim-not-192"),
        pytest.param(lambda t: _tensors(heads=8), id="group-not-16"),
        pytest.param(lambda t: _tensors(heads=32, kv_heads=1), id="group-32"),
        pytest.param(lambda t: _tensors(heads=24, kv_heads=16), id="heads-not-divisible"),
        pytest.param(lambda t: _tensors(page=32), id="page-32"),
        pytest.param(_with(3, lambda o: o[:2].contiguous()), id="out-shape-mismatch"),
        pytest.param(_with(2, lambda v: v[:1].contiguous()), id="kv-shape-mismatch"),
        pytest.param(
            _with(0, lambda q: q.permute(0, 2, 1).contiguous().permute(0, 2, 1)),
            id="q-last-dim-strided",
        ),
    ],
)
def test_unsupported_inputs_are_rejected_before_launch(kernel, mutate):
    tensors = mutate(_tensors())

    with pytest.raises(ValueError, match="gfx936 E4M3 SWA requires"):
        _call(*tensors)

    assert kernel.launches == []


def test_negative_window_is_rejected(kernel):
    with pytest.raises(ValueError, match="gfx936 E4M3 SWA requires"):
        _call(*_tensors(), window=-1)
    assert kernel.launches == []
