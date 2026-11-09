"""Synthetic Step5 state must exist before the real DSA metadata builder."""
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_hcu.v1.step5_decode_bench import prepare_decode_bench


def setup_runner(monkeypatch, past, device='cpu', ratio=1, std=0.0):
    from vllm_hcu.model_executor.layers import step4_dsa as dsa

    state = dsa.Step4DSAState(2, dsa.Step4DSAGeometry(), 16, torch.bfloat16,
                             max_num_reqs=4, max_model_len=16384, spec_query_len=4)
    state.block_size, state.regions_per_page = 64, 8
    state.summary = torch.zeros((2, 144 * 8, 1, 256), dtype=torch.float8_e4m3fn, device=device)
    state.pending_key = torch.zeros((2, 17, 8, 256), dtype=torch.bfloat16, device=device)
    state.pending_z = torch.zeros_like(state.pending_key)
    names = ['model.layers.3.self_attn', 'model.layers.7.self_attn', 'model.layers.0.self_attn']
    caches = {name: torch.zeros((144, 1, 64, 8), dtype=torch.uint8, device=device) for name in names[:2]}
    # Exercise head-major 5-D SWA views with padding between physical pages.
    raw = torch.zeros((144, 2 * 64 * 8 + 16), dtype=torch.uint8, device=device)
    caches[names[2]] = raw.as_strided((144, 2, 64, 1, 8), (1040, 512, 8, 512, 1))
    monkeypatch.setattr(dsa, '_INDEXERS', {name: NS(state=state, slot=i) for i, name in enumerate(names[:2])})
    blocks = list(range(1, (past + 64 * ratio - 1) // (64 * ratio) + 1))
    worker = NS(kv_caches=caches, fill_mean=0.015, fill_std=std, group_to_layers={0: names})
    cfg = NS(model_config=NS(hf_config=NS(model_type='step5')),
             kv_transfer_config=NS(kv_connector='DecodeBenchConnector'),
             cache_config=NS(enable_prefix_caching=False, cache_dtype='fp8_e4m3'),
             parallel_config=NS(pipeline_parallel_size=1, prefill_context_parallel_size=1,
                                decode_context_parallel_size=1))
    runner = NS(vllm_config=cfg, kv_connector=NS(kv_connector=NS(connector_worker=worker)),
                kv_cache_config=NS(kv_cache_groups=[NS(kv_cache_spec=NS(block_size=64 * ratio))]))
    output = NS(kv_connector_metadata=NS(reqs_to_fill={'r': ((blocks,), past)}))
    physical = [b * ratio + i for b in blocks for i in range(ratio)]
    return runner, output, state, physical, raw


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('ratio', [1, 2])
@pytest.mark.parametrize('past', [1, 7, 8, 63, 64, 65, 8191])
def test_initial_fill_and_real_spec_metadata(monkeypatch, device, ratio, past):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('GPU required')
    runner, output, state, physical, raw = setup_runner(monkeypatch, past, device, ratio)
    if device == 'cpu':
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
        monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    # Reserve an unrelated live request and confirm it survives initialization.
    other_row = state.pending.acquire(0)
    state.pending_key[:, other_row].fill_(3)
    table = torch.tensor([physical + [140]], dtype=torch.int32)
    qsl = torch.tensor([0, 1], dtype=torch.int32)
    md = NS(num_decodes=1, num_decode_tokens=1, query_start_loc_cpu=qsl,
            query_start_loc=qsl, seq_lens_cpu=torch.tensor([past + 1], dtype=torch.int32),
            block_table=table, block_table_np=table.numpy())
    if past % 8:
        with pytest.raises(RuntimeError, match='no pending tokens'):
            state._spec_decode_schedule(md)
    prepare_decode_bench(runner, output)
    assert not output.kv_connector_metadata.reqs_to_fill
    assert state.pending.get(0) == other_row
    assert torch.all(state.pending_key[:, other_row] == 3)
    expected = torch.tensor(0.015).to(torch.float8_e4m3fn).float().item()
    for cache in runner.kv_connector.kv_connector.connector_worker.kv_caches.values():
        assert torch.all(cache[physical[0]].view(torch.float8_e4m3fn).float() == expected)
        assert torch.count_nonzero(cache[0]) == 0
    assert torch.count_nonzero(raw[:, 1024:]) == 0
    assert torch.all(state.summary[:, physical[0] * 8].float() == expected)
    if past % 8:
        slot = physical[past // 64] * 8 + (past % 64) // 8
        row = state.pending.get(slot)
        assert torch.all(state.pending_key[:, row, :past % 8] == torch.tensor(0.015, dtype=torch.bfloat16))
    # Same scheduling operation that failed in the server log now succeeds.
    sched = state._spec_decode_schedule(md)
    assert sched.spec
    before = state.summary.view(torch.uint8).clone()
    prepare_decode_bench(runner, output)  # no native duplicate fill on pre_forward
    assert torch.equal(before, state.summary.view(torch.uint8))


@pytest.mark.parametrize('connector', [None, 'MooncakeConnector'])
def test_ifb_and_pd_are_untouched(connector):
    cfg = NS(kv_transfer_config=None if connector is None else NS(kv_connector=connector))
    prepare_decode_bench(NS(vllm_config=cfg), NS())


def test_non_step5_is_untouched():
    cfg = NS(kv_transfer_config=NS(kv_connector='DecodeBenchConnector'),
             model_config=NS(hf_config=NS(model_type='other')))
    prepare_decode_bench(NS(vllm_config=cfg), NS())


def test_invalid_blocks_fail_before_writes(monkeypatch):
    runner, out, state, _, _ = setup_runner(monkeypatch, 63)
    out.kv_connector_metadata.reqs_to_fill['r'] = (([10000],), 63)
    with pytest.raises(ValueError, match='capacity'):
        prepare_decode_bench(runner, out)
    assert torch.count_nonzero(state.summary.view(torch.uint8)) == 0
    assert out.kv_connector_metadata.reqs_to_fill


def test_random_fill_and_recycled_pending(monkeypatch):
    runner, out, state, physical, _ = setup_runner(monkeypatch, 63, std=0.01)
    stale = physical[0] * 8
    state.pending.acquire(stale)
    prepare_decode_bench(runner, out)
    with pytest.raises(RuntimeError, match='no pending tokens'):
        state.pending.get(stale)
    assert state.summary[0].float().std() > 0
    runner.kv_connector.kv_connector.connector_worker.fill_std = 0
    out.kv_connector_metadata.reqs_to_fill = {'new': (([1],), 63)}
    prepare_decode_bench(runner, out)
    row = state.pending.get(15)
    assert torch.all(state.pending_key[:, row] == torch.tensor(0.015, dtype=torch.bfloat16))
