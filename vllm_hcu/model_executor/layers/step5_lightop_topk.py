# SPDX-License-Identifier: Apache-2.0
"""LightOp selects IDs; the separate metadata operator sorts and packs them."""
import torch

_IDENTITY = {}
_MAX_REGIONS = 1 << 17


def lightop_region_topk_ids(logits, lengths, *, topk):
    # Step5's checkpoint uses K=512. Keep other geometries on their prior
    # selector until their native API and numerical behavior are verified.
    if topk != 512 or logits.shape[1] > _MAX_REGIONS:
        return None
    out = torch.empty((logits.shape[0],topk),device=logits.device,dtype=torch.int32)
    if not logits.shape[0]:
        return out
    import lightop
    key=logits.device
    if key not in _IDENTITY:
        # The expanded view has row stride zero. LightOp explicitly accepts
        # row-contiguous page tables and honors their row stride, including 0.
        # Keep the owner alive for every CUDA Graph using this device.
        _IDENTITY[key]=torch.arange(_MAX_REGIONS,device=key,dtype=torch.int32)[None,:]
    table=_IDENTITY[key].expand(logits.shape[0],-1)
    # Unit pages + identity transform make the output logical region IDs.
    # This does not apply Step5's physical mapping or append its current region.
    lightop.topk_transform_512(logits,lengths,table,out,1)
    return out
