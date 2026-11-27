# SPDX-License-Identifier: Apache-2.0
"""Opaque shape dispatch and serial Step5 LL counter lifetime.

The counter path is opt-in, fixed LL and DBO-off on supported topologies. Both native
counters are reset by the existing shared/routed restore kernel after receive
completion, including on empty ranks. No native synchronization is removed.
"""
import functools
import os
import threading

import torch
from torch.utils._python_dispatch import TorchDispatchMode

_pending = threading.local()


def _router_shared(x: torch.Tensor, xq: torch.Tensor | None,
                   xs: torch.Tensor | None, layer_name: str,
                   tp_size: int, num_experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    from .moe_runner import get_layer_from_name
    layer = get_layer_from_name(layer_name)
    return layer._hcu_step5_ll_router_shared(x, xq, xs)


def _router_shared_fake(x, xq, xs, layer_name, tp_size, num_experts):
    return (torch.empty_like(x),
            x.new_empty(((x.shape[0] + tp_size - 1) // tp_size, num_experts),
                        dtype=torch.float32))


@functools.lru_cache(maxsize=1)
def router_shared_op():
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(op_name="step5_ll_router_shared", op_func=_router_shared,
                              fake_impl=_router_shared_fake, mutates_args=[])
    return torch.ops.vllm.step5_ll_router_shared.default


class _CounterAllocation(TorchDispatchMode):
    """Replace only the verified native single-element INT32 zero allocation."""
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor
        self.count = 0
        self.seen = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        device = kwargs.get("device")
        if device is not None:
            device = torch.device(device)
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
        if func is torch.ops.aten.zeros.default:
            self.seen.append((str(args), str(kwargs)))
        if (func is torch.ops.aten.zeros.default and tuple(args[0]) == (1,)
                and kwargs.get("dtype") == torch.int32
                and device == self.tensor.device):
            self.count += 1
            if self.count != 1:
                raise RuntimeError("Step5 LL native counter allocation ABI changed")
            return self.tensor
        return func(*args, **kwargs)


class CounterState:
    def __init__(self, device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Initialize Step5 LL counters before CUDA graph capture")
        self.values = torch.zeros(2, dtype=torch.int32, device=device)
        self.slots = (self.values[:1], self.values[1:])
        self.phase = None
        self.stream = None
        self.valid = True

    def invoke(self, slot, fn, *args, **kwargs):
        if not self.valid:
            raise RuntimeError("Step5 LL counter state invalid after interrupted forward")
        if kwargs.get("return_recv_hook") is not True or kwargs.get("async_finish", False):
            raise RuntimeError("Fused LL counter reset requires synchronous recv hooks")
        stream = torch.cuda.current_stream(self.values.device).cuda_stream
        if slot == 0:
            if getattr(_pending, "state", None) is not None or self.phase is not None:
                raise RuntimeError("Overlapping LL forwards require independent counter state")
            self.stream = stream
            _pending.state = self
        elif getattr(_pending, "state", None) is not self or self.phase != 0 or self.stream != stream:
            raise RuntimeError("LL combine must follow dispatch on the same stream")
        mode = _CounterAllocation(self.slots[slot])
        try:
            # DBO/SP are rejected when this state is installed. In serial
            # serving there is no work between send and receive hooks, so let
            # native LL perform both phases in one launch and retain the
            # upstream hook interface with a no-op only after native completion.
            serial = os.getenv("HCU_STEP5_LL_SERIAL", "0") == "1"
            if serial:
                kwargs = dict(kwargs, return_recv_hook=False)
            with mode:
                result = fn(*args, **kwargs)
            if serial:
                if result[-1] is not None:
                    raise RuntimeError("Synchronous native LL unexpectedly returned a recv hook")
                result = (*result[:-1], lambda: None)
            if mode.count != 1:
                raise RuntimeError(f"Step5 LL native counter allocation ABI changed: {mode.seen}")
            self.phase = slot
            return result
        except BaseException:
            self.valid = False
            raise


def install_counter_reset(buffer):
    if os.getenv("HCU_STEP5_LL_FUSED_COUNTER_RESET", "0") != "1":
        return
    from vllm.config import get_current_vllm_config
    from .step5_ll_topology import uses_step5_ll_fused_restore
    cfg = get_current_vllm_config()
    from .step5_replicated_shared import uses_replicated_shared_ll_restore
    if not (uses_step5_ll_fused_restore(cfg)
            or uses_replicated_shared_ll_restore(cfg)):
        raise ValueError("Fused LL counters require fixed LL TP4/DP2, TP4/DP4 or "
                         "TP1/DP16, PP1, EP, FP32 residuals, DBO/SP off")
    if hasattr(buffer, "_hcu_step5_counter_state"):
        return
    state = CounterState(torch.device("cuda", torch.cuda.current_device()))
    for slot, name in enumerate(("low_latency_dispatch", "low_latency_combine")):
        original = getattr(buffer, name)
        @functools.wraps(original)
        def wrapped(*args, _slot=slot, _fn=original, **kwargs):
            return state.invoke(_slot, _fn, *args, **kwargs)
        setattr(buffer, name, wrapped)
    buffer._hcu_step5_counter_state = state


def take_pending_counters(device):
    state = getattr(_pending, "state", None)
    if state is None:
        return None
    if (not state.valid or state.phase != 1 or state.values.device != device
            or state.stream != torch.cuda.current_stream(device).cuda_stream):
        raise RuntimeError("LL counter reset must follow receive on the same stream")
    # GPU reset is ordered by the caller's existing restore kernel. Graph
    # replay retains these exact pointers and stream dependencies.
    return state.values


def finish_pending_counter_reset():
    state = getattr(_pending, "state", None)
    if state is not None:
        state.phase = None
        _pending.state = None
