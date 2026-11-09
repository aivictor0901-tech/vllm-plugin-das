"""Step5 PD state transfer and native Mooncake request semantics."""

import asyncio
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_hcu.distributed.kv_transfer.kv_connector.v1.mooncake import step5_pd as pd


def _manager(monkeypatch, *, producer, device="cpu"):
    from vllm_hcu.model_executor.layers import step4_dsa as dsa

    names = [f"model.layers.{i}.self_attn" for i in range(2)]
    state = NS(
        geometry=NS(region_size=8, proxy_dim=4), regions_per_page=2, block_size=16,
        summary=torch.zeros((2, 8, 1, 4), dtype=torch.float8_e4m3fn, device=device),
        pending_key=torch.zeros((2, 8, 8, 4), dtype=torch.bfloat16, device=device),
        pending_z=torch.zeros((2, 8, 8, 4), dtype=torch.bfloat16, device=device),
        pending=dsa.Step4PendingTable(8), ensure_allocated=lambda *args: None,
    )
    monkeypatch.setattr(dsa, "_INDEXERS", {
        name: NS(state=state, slot=i) for i, name in enumerate(names)
    })
    caches = {name: torch.zeros((4, 1, 16, 8), dtype=torch.uint8, device=device) for name in names}
    worker = NS(_physical_blocks_per_logical_kv_block=1, is_kv_producer=producer,
                _layer_specs={}, _layer_group_indices={name: 0 for name in names},
                _logical_to_kernel_block_ids=lambda groups: groups)
    return pd.Step5StateTransfer(worker, caches), state


@pytest.mark.parametrize("prompt_tokens", [2, 8, 9, 16, 17, 31, 32, 33])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tail_restore_remaps_physical_blocks_and_preserves_bytes(monkeypatch, prompt_tokens, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    p, ps = _manager(monkeypatch, producer=True, device=device)
    d, ds = _manager(monkeypatch, producer=False, device=device)
    source_blocks, dest_blocks = [[0, 2, 1]], [[3, 1, 2]]
    past, prefix = prompt_tokens - 1, (prompt_tokens - 1) % 8
    logical_region = past // 8
    pslot = source_blocks[0][past // 16] * 2 + logical_region % 2
    dslot = dest_blocks[0][past // 16] * 2 + logical_region % 2
    original_key = torch.arange(64, device=device).reshape(2, 8, 4).to(torch.bfloat16)
    original_z = -original_key
    if prefix:
        row = ps.pending.acquire(pslot)
        ps.pending_key[:, row] = original_key
        ps.pending_z[:, row] = original_z
    # Force a different pending-buffer row on the receiver as well.
    ds.pending.acquire(0)
    tails = p.capture(source_blocks, prompt_tokens)
    d.restore(dest_blocks, prompt_tokens, tails)
    if prefix:
        row = ds.pending.get(dslot)
        assert row != 0
        assert torch.equal(ds.pending_key[:, row, :prefix], original_key[:, :prefix])
        assert torch.equal(ds.pending_z[:, row, :prefix], original_z[:, :prefix])
    assert len(p.views) == 2
    assert next(iter(p.views.values())).untyped_storage().data_ptr() == ps.summary.untyped_storage().data_ptr()
    with pytest.raises(ValueError, match="every DSA layer"):
        d.restore(dest_blocks, prompt_tokens, {})
    bad = dict(tails)
    bad[next(iter(bad))] += b"x"
    with pytest.raises(ValueError, match="payload"):
        d.restore(dest_blocks, prompt_tokens, bad)


@pytest.mark.parametrize("pd_mode", [False, True])
def test_prefill_retains_only_last_closed_region_for_pd(pd_mode):
    from vllm_hcu.model_executor.layers.step4_dsa import Step4DSAState, Step4DSAGeometry

    state = Step4DSAState(1, Step4DSAGeometry(), 8, torch.bfloat16, spec_query_len=4)
    state.regions_per_page = 8
    state.pd_keep_prefill_tail = pd_mode
    table = torch.tensor([[3]], dtype=torch.int32)
    seq = torch.tensor([32], dtype=torch.int32)
    md = NS(query_start_loc_cpu=torch.tensor([0, 32], dtype=torch.int32),
            seq_lens_cpu=seq, seq_lens=seq, block_table=table,
            block_table_np=table.numpy(), num_decodes=0, num_prefills=1)
    schedule = state._general_schedule(md)
    keep = schedule.requests[0].keep_rows
    assert len(keep) == int(pd_mode)
    if pd_mode:
        assert keep == [(state.pending.get(3 * 8 + 3), 24)]


def test_receive_is_restored_on_worker_before_scheduler_readiness(monkeypatch):
    import concurrent.futures
    from vllm_hcu.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector as mc

    worker = object.__new__(mc.MooncakeConnectorWorker)
    worker.shutdown = lambda: None
    worker.is_kv_producer, worker.is_kv_consumer = False, True
    worker.receiver_loop = None
    worker._kv_load_error = None
    worker.finished_recving_reqs = set()
    worker.tp_rank = 0
    calls = []
    worker._step5_state = NS(received={}, restore=lambda *args: calls.append(args))
    pull = mc.PullReqMeta("d", "x", [[3]], "p", "http://p:8998",
                         pull_tasks_count=1, step5_prompt_tokens=17)
    response = mc.MooncakeXferResponse(status=mc.MooncakeXferResponseStatus.FINISH,
                                      ok_reqs=["d"], step5_tails={"d": {"layer": b""}})
    worker.process_pulling_result(response, {"d": pull})
    assert not calls

    def submit(coro, loop):
        future = concurrent.futures.Future()
        future.set_result(asyncio.run(coro))
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)
    assert worker.get_finished() == (None, {"d"})
    assert calls == [([[3]], 17, {"layer": b""})]
    assert worker._step5_state.received == {}


def test_protocol_rejects_missing_state_without_publishing_ready():
    from vllm_hcu.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector as mc

    worker = object.__new__(mc.MooncakeConnectorWorker)
    worker.shutdown = lambda: None
    worker._step5_state = NS(received={})
    worker._kv_load_error = None
    worker.finished_recving_reqs = set()
    pull = mc.PullReqMeta("d", "x", [[3]], "p", "http://p:8998",
                         pull_tasks_count=1, step5_prompt_tokens=17)
    worker.process_pulling_result(mc.MooncakeXferResponse(
        status=mc.MooncakeXferResponseStatus.FINISH, ok_reqs=["d"]), {"d": pull})
    assert worker._kv_load_error is not None
    assert not worker.finished_recving_reqs


def test_pd_client_uses_real_prefill_and_same_token_ids(monkeypatch):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[2] / "examples/step5_pd_request.py"
    spec = importlib.util.spec_from_file_location("pd_client", path)
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    calls = []

    def request(url, payload=None, rank=None):
        calls.append((url, payload, rank))
        if url.endswith("/tokenize"):
            return {"tokens": [1, 2, 3]}
        if url.endswith("/query"):
            return {"2": {"engine_id": "actual-p-dp2"}}
        return {"choices": [{"text": "answer"}]}

    monkeypatch.setattr(client, "request", request)
    client.run(NS(prompt="test", model="step5", prefill="http://p:1", bootstrap="http://p:2",
                  decode="http://d:3", p_rank=2, d_rank=1, max_tokens=16))
    _, p, prank = calls[2]
    _, d, drank = calls[3]
    assert prank == 2 and drank == 1
    assert p["prompt"] == d["prompt"] == [1, 2, 3]
    assert p["max_tokens"] == 1 and p["ignore_eos"] is True
    assert p["kv_transfer_params"]["do_remote_decode"] is True
    assert d["kv_transfer_params"]["do_remote_prefill"] is True
    assert d["kv_transfer_params"]["remote_engine_id"] == "actual-p-dp2"
    assert p["kv_transfer_params"]["transfer_id"] == d["kv_transfer_params"]["transfer_id"]
    assert set(p["kv_transfer_params"]) == {"transfer_id", "do_remote_decode"}
    assert set(d["kv_transfer_params"]) == {
        "transfer_id", "do_remote_prefill", "remote_engine_id", "remote_bootstrap_addr",
    }


@pytest.mark.parametrize("producer", [False, True])
@pytest.mark.parametrize("params", [None, {}, {"do_remote_prefill": False},
                                     {"_step5_received": True}])
def test_ordinary_and_recomputed_requests_keep_native_behavior(producer, params):
    from vllm_hcu.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector as mc

    scheduler = object.__new__(mc.MooncakeConnectorScheduler)
    scheduler.step5_pd_enabled = True
    scheduler.is_kv_producer, scheduler.is_kv_consumer = producer, not producer
    scheduler._has_mamba = False
    req = NS(kv_transfer_params=params, num_prompt_tokens=17,
             num_computed_tokens=0, max_tokens=256)
    assert scheduler.get_num_new_matched_tokens(req, 0) == (0, False)


@pytest.mark.parametrize("producer", [False, True])
def test_standard_handoff_uses_engine_prompt_length(producer):
    from vllm_hcu.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector as mc

    scheduler = object.__new__(mc.MooncakeConnectorScheduler)
    scheduler.step5_pd_enabled = True
    scheduler.is_kv_producer, scheduler.is_kv_consumer = producer, not producer
    scheduler._has_mamba = False
    scheduler._reqs_need_recv, scheduler._reqs_need_send = {}, {}
    scheduler._reqs_not_processed = set()
    params = {"transfer_id": "transfer"}
    if producer:
        params["do_remote_decode"] = True
    else:
        params.update(do_remote_prefill=True, remote_engine_id="p",
                      remote_bootstrap_addr="http://p:8998")
    req = NS(kv_transfer_params=params, num_prompt_tokens=17, prompt_token_ids=list(range(17)),
             num_computed_tokens=0, max_tokens=1 if producer else 256)
    expected = (0, False) if producer else (17, True)
    assert scheduler.get_num_new_matched_tokens(req, 0) == expected
    # The client does not supply extension fields. Even if it does, lengths
    # used for state capture/restore must come from the tokenized request.
    params["step5_prompt_tokens"] = 999
    pending = scheduler._reqs_need_send if producer else scheduler._reqs_need_recv
    pending["req"] = (req, [[3, 4]])
    meta = scheduler.build_connector_meta(None)
    assert meta.step5_prompt_tokens == {"req": 17}
    if not producer:
        assert meta.reqs_to_recv["p"]["req"].step5_prompt_tokens == 17
    else:
        req.max_tokens = 2
        with pytest.raises(ValueError, match="exactly one token"):
            scheduler.get_num_new_matched_tokens(req, 0)
