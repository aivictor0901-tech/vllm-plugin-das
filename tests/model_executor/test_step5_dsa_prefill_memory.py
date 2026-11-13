# SPDX-License-Identifier: Apache-2.0
"""Compare bounded planning against the plugin's original GPU scorer/selector.

The reference is Step4Indexer._score + build_plan from this checkout, including
causal masking, physical page mapping, tie breaking, and group-major layout.
"""
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.hcu
@pytest.mark.parametrize("fp8", [True, False])
@pytest.mark.parametrize("case", ["empty_history", "mixed", "128k"])
def test_chunked_plan_matches_original_and_bounds_peak(monkeypatch, fp8, case):
    if not torch.cuda.is_available():
        pytest.skip("requires HCU")
    from vllm_hcu.model_executor.layers import step4_dsa as dsa

    monkeypatch.setattr(dsa, "PREFILL_FP8_INDEXER", fp8)
    monkeypatch.setattr(dsa, "INDEXER_Q_CHUNK", 0)
    monkeypatch.setattr(dsa, "DENSE_EQUIV", False)
    torch.manual_seed(42)
    nd = int(case == "mixed")
    sizes = {"empty_history": [(0, 7)], "mixed": [(4097, 1), (19, 137), (32768, 91)],
             "128k": [(129021, 2051)]}[case]
    groups, heads, dim = (4, 4, 256) if case == "128k" else (2, 2, 256)
    topk, chunk = (512, 1024) if case == "128k" else (31, 64)
    tokens = sum(n for _, n in sizes)
    pages = (max(past + n for past, n in sizes) + 15) // 16
    table = torch.randperm(len(sizes) * pages, device="cuda").reshape(len(sizes), pages).int()
    summary = torch.randn(len(sizes) * pages * 2, 1, dim, device="cuda").to(torch.float8_e4m3fn)
    q = torch.randn(tokens, groups, heads, dim, device="cuda").to(torch.float8_e4m3fn).to(torch.bfloat16)
    weights = torch.randn(tokens, groups, heads, device="cuda", dtype=torch.bfloat16)
    plans, positions, starts = [], [], [0]
    for row, (past, n) in enumerate(sizes):
        nscore = (past + n - 1) // 8
        regions = torch.arange(nscore, device="cuda")
        slots = table[row, regions // 2].long() * 2 + regions % 2
        plans.append(SimpleNamespace(row=row, past=past, qlen=n, tok_start=starts[-1],
                                     nscore=nscore, score_slots=slots))
        starts.append(starts[-1] + n)
        positions.append(torch.arange(past, past + n, device="cuda", dtype=torch.int32))
    md = SimpleNamespace(num_decodes=nd, num_decode_tokens=nd,
                         num_prefills=len(sizes)-nd, num_actual_tokens=tokens,
                         positions=torch.cat(positions), block_table=table,
                         seq_lens=torch.tensor([p+n for p,n in sizes], device="cuda", dtype=torch.int32),
                         query_start_loc=torch.tensor(starts, device="cuda", dtype=torch.int32))
    sched = SimpleNamespace(requests=plans, spec=False, decode=False)
    indexer = object.__new__(dsa.Step4Indexer)
    indexer.slot = 0
    indexer.state = SimpleNamespace(summary=summary.unsqueeze(0), regions_per_page=2)
    indexer.num_kv_groups = groups
    indexer.geometry = SimpleNamespace(topk=topk)
    indexer.weight_prescale = 0.125
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    prefill, decode, width = indexer._score(md, sched, q, weights)
    reference = dsa.build_plan(md, prefill, decode, topk=topk, num_kv_groups=groups,
                               regions_per_page=2)
    torch.cuda.synchronize()
    original_peak = torch.cuda.max_memory_allocated() - base
    del prefill, decode
    # Keep the small final reference on CPU so it does not inflate the measured
    # chunked peak, and demand exact selector outputs, not just similar scores.
    expected = tuple(x.cpu() for x in reference.prefill)
    expected_decode = None if reference.decode is None else tuple(x.cpu() for x in reference.decode)
    del reference
    # Check the compiled kernel actually used by the planner, rather than a
    # separately chosen test tile. Driver scratch is outside PyTorch's allocator
    # peak statistics and caused a real HSA OOM at the end of 128K prefill.
    compiled_scores = []
    if case == "128k" and fp8:
        from vllm_hcu.model_executor.layers import step4_dsa_kernels as kernels
        original_kernel = kernels._prefill_indexer_logits_kernel
        class CaptureKernel:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    compiled = original_kernel[grid](*args, **kwargs)
                    compiled_scores.append(compiled)
                    return compiled
                return launch
        monkeypatch.setattr(kernels, "_prefill_indexer_logits_kernel", CaptureKernel())
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    actual, actual_width = indexer._score_prefill_chunked(md, sched, q, weights, chunk_size=chunk)
    torch.cuda.synchronize()
    chunked_peak = torch.cuda.max_memory_allocated() - base
    if case == "128k" and fp8 and "gfx936" in torch.cuda.get_device_properties(0).gcnArchName:
        assert compiled_scores
        assert all(kernel.n_spills == 0 for kernel in compiled_scores)
    assert actual_width == max(p.nscore for p in plans if p.row >= nd)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0)
    if nd:
        no_prefill, decode, _ = indexer._score(md, sched, q, weights, skip_prefill=True)
        assert no_prefill is None
        decode_plan = dsa.build_plan(md, None, decode, topk=topk, num_kv_groups=groups, regions_per_page=2)
        for a, b in zip(decode_plan.decode, expected_decode):
            torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0)
    if case == "128k":
        print(f"fp8={fp8} original_peak_MiB={original_peak/2**20:.1f} chunked_peak_MiB={chunked_peak/2**20:.1f}")
        assert chunked_peak < original_peak * 0.75


@pytest.mark.parametrize("capturing", [False, True])
def test_prefill_attention_releases_only_eager_plan(monkeypatch, capturing):
    from vllm_hcu.model_executor.layers import step4_dsa as dsa

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(dsa, "FLASH_ATTN_DSA_STAGES", set())
    monkeypatch.setattr(dsa, "BOLTOPS_STAGES", set())
    md = SimpleNamespace(num_actual_tokens=2, num_decodes=0, num_decode_tokens=0,
                         num_prefills=1, positions=torch.arange(2), slot_mapping=torch.arange(2))
    plan = SimpleNamespace(md=md, prefill=(torch.zeros(2, 2, dtype=torch.int32),
                          torch.ones(2, dtype=torch.int32), torch.zeros(2, 2, dtype=torch.int32),
                          torch.zeros(2, dtype=torch.int32)))
    layer = SimpleNamespace(step4_sparse_plan=plan, layer_name="test")
    impl = object.__new__(dsa.Step4DSAImpl)
    impl.num_kv_heads, impl.scale = 1, 1.0
    q = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4)
    impl._write_kv = lambda *args: (q, q)
    def attend(query, *args, **kwargs):
        assert layer.step4_sparse_plan is (plan if capturing else None)
        return query + 1, torch.empty(0)
    monkeypatch.setattr(dsa, "sparse_attention_prefill", attend)
    result = impl.forward(layer, q, q, q, q, md, torch.empty_like(q))
    torch.testing.assert_close(result, q + 1)
    assert layer.step4_sparse_plan is (plan if capturing else None)
