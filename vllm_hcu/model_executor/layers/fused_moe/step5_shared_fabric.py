# SPDX-License-Identifier: Apache-2.0
"""Optional local Fabric transport for replicated Step5 owner collectives.

Initialization is collective and happens during model construction, before
capture. Unsupported groups retain the ordinary vLLM RS/AG transport.
"""
from __future__ import annotations

import logging
import os


def initialize_shared_fabric(group) -> None:
    if hasattr(group, '_step5_shared_fabric'):
        return
    group._step5_shared_fabric = None
    if (os.environ.get('HCU_STEP5_REPLICATED_SHARED_FABRIC', '0') != '1'
            or group.world_size not in (2, 4, 8)):
        return
    import torch
    import torch.distributed as dist
    from vllm.distributed.parallel_state import in_the_same_node_as

    if not all(in_the_same_node_as(group.cpu_group, source_rank=0)):
        return
    communicator = group.device_communicator
    device = communicator.device
    factory = None
    reason = None
    try:
        architecture = torch.cuda.get_device_properties(device).gcnArchName.split(':', 1)[0]
        if architecture != 'gfx938':
            reason = f'unsupported architecture {architecture}'
        else:
            from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
            required = ('should_custom_rs', 'reduce_scatter', 'should_custom_ag',
                        'all_gather_unreg', 'close')
            if all(callable(getattr(CustomAllreduce, name, None)) for name in required):
                factory = CustomAllreduce
            else:
                reason = 'installed AITER lacks the required RS/AG interface'
    except (ImportError, AttributeError) as error:
        reason = str(error)
    reasons = [None] * group.world_size
    dist.all_gather_object(reasons, reason, group=group.cpu_group)
    if any(reason is not None for reason in reasons):
        logging.getLogger(__name__).info('Step5 shared Fabric unavailable: %s', reasons)
        return
    # As with the existing DP Fabric adapter, transport is fixed at creation.
    # Never fall back locally after a collective allocation has started.
    overrides = {'AITER_AR_TRANSPORT': 'fabric', 'AITER_AR_ENABLE_REG_CAPTURE': '0'}
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        fabric = factory(group.cpu_group, device, max_size=32 * 1024 * 1024)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if fabric.disabled or fabric.transport != 'fabric':
        raise RuntimeError('Requested Step5 shared Fabric transport did not initialize')
    original_destroy = communicator.destroy

    def destroy():
        current = group._step5_shared_fabric
        group._step5_shared_fabric = None
        try:
            if current is not None:
                current.close()
        finally:
            original_destroy()

    communicator.destroy = destroy
    group._step5_shared_fabric = fabric
    print(f'[step5_shared_fabric] initialized TP={group.world_size} '
          f'rank={group.rank_in_group} transport={fabric.transport}', flush=True)
