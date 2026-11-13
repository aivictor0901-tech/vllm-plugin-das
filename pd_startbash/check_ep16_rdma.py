#!/usr/bin/env python3
"""Cross-node RCCL correctness/bandwidth probe using installed PyTorch APIs."""
import datetime
import os
import time
import torch
import torch.distributed as dist

torch.cuda.set_device(0)
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=180))
rank = dist.get_rank()
x = torch.full((8 * 1024 * 1024,), rank + 1, device="cuda", dtype=torch.float32)
dist.all_reduce(x)
expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
assert bool(torch.all(x == expected).item()), "all_reduce correctness failed"
x.zero_()
for _ in range(3):
    dist.all_reduce(x)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    dist.all_reduce(x)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print({"rank": rank, "correct": True, "payload_MiB": x.numel() * x.element_size() / 2**20,
       "iterations": 10, "seconds": elapsed,
       "payload_GB_per_second": x.numel() * x.element_size() * 10 / elapsed / 1e9,
       "NCCL_NET": os.environ.get("NCCL_NET")}, flush=True)
dist.destroy_process_group()
