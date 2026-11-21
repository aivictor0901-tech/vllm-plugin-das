# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Step5 router dispatch and low-latency Decode helpers."""

import functools
import importlib
import os
import threading

import torch

from vllm_hcu.platforms import envs as henvs


_pending = threading.local()


def router_bias_func(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    router_bias: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
    nan_row_i_out: int = 0,
    indices_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch Step5 biased top-k to the model-scoped Triton kernel."""
    del hidden_states
    if indices_dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "Step5 router indices_dtype must be torch.int32 or torch.int64, "
            f"got {indices_dtype}."
        )

    # The Triton kernel also runs on ROCm/HCU (is_cuda is True there) and is
    # 7-10x faster than the eager fallback below, which runs inside the opaque
    # moe_forward op and so never gets fused by Inductor.
    if not henvs.optional_custom_op_enabled() or not gating_output.is_cuda:
        if router_bias is None:
            raise ValueError("Step5 router_bias must be provided.")
        scores = gating_output.float().sigmoid()
        _, topk_ids = torch.topk(
            scores + router_bias.to(device=gating_output.device, dtype=torch.float32),
            k=topk,
            dim=-1,
        )
        topk_weights = scores.gather(1, topk_ids)
        if renormalize:
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights *= routed_scaling_factor

        invalid_rows = gating_output.isnan().any(dim=-1, keepdim=True)
        topk_weights = torch.where(invalid_rows, 0, topk_weights)
        topk_ids = torch.where(invalid_rows, nan_row_i_out, topk_ids)
        return topk_weights, topk_ids.to(indices_dtype)

    router_bias_triton_func = importlib.import_module(
        "vllm_hcu.ops.step5.router_bias"
    ).router_bias_triton_func
    if gating_output.stride(-1) != 1:
        gating_output = gating_output.contiguous()
    return router_bias_triton_func(
        gating_output,
        router_bias,
        topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        nan_row_i_out=nan_row_i_out,
        indices_dtype=indices_dtype,
    )


def _router_shared(
    x: torch.Tensor,
    xq: torch.Tensor | None,
    xs: torch.Tensor | None,
    layer_name: str,
    tp_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm_hcu.model_executor.layers.fused_moe.moe_runner import get_layer_from_name

    layer = get_layer_from_name(layer_name)
    return layer._hcu_step5_ll_router_shared(x, xq, xs)


def _router_shared_fake(x, xq, xs, layer_name, tp_size, num_experts):
    return (
        torch.empty_like(x),
        x.new_empty(
            ((x.shape[0] + tp_size - 1) // tp_size, num_experts), dtype=torch.float32
        ),
    )


@functools.lru_cache(maxsize=1)
def router_shared_op():
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="step5_ll_router_shared",
        op_func=_router_shared,
        fake_impl=_router_shared_fake,
        mutates_args=[],
    )
    return torch.ops.vllm.step5_ll_router_shared.default


@functools.lru_cache(maxsize=1)
def _counter_allocation_type():
    """Load the private Torch dispatch API only for fused LL counters."""
    from torch.utils._python_dispatch import TorchDispatchMode

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
            if (
                func is torch.ops.aten.zeros.default
                and tuple(args[0]) == (1,)
                and kwargs.get("dtype") == torch.int32
                and device == self.tensor.device
            ):
                self.count += 1
                if self.count != 1:
                    raise RuntimeError("Step5 LL native counter allocation ABI changed")
                return self.tensor
            return func(*args, **kwargs)

    return _CounterAllocation


class CounterState:
    """Track one fixed-LL dispatch and combine through restore on a single stream."""

    def __init__(self, device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Initialize Step5 LL counters before CUDA graph capture")
        self._allocation_type = _counter_allocation_type()
        self.values = torch.zeros(2, dtype=torch.int32, device=device)
        self.slots = (self.values[:1], self.values[1:])
        self.phase = None
        self.stream = None
        self.valid = True

    def invoke(self, slot, fn, *args, **kwargs):
        if not self.valid:
            raise RuntimeError(
                "Step5 LL counter state invalid after interrupted forward"
            )
        if kwargs.get("return_recv_hook") is not True or kwargs.get(
            "async_finish", False
        ):
            raise RuntimeError("Fused LL counter reset requires synchronous recv hooks")
        stream = torch.cuda.current_stream(self.values.device).cuda_stream
        if slot == 0:
            if getattr(_pending, "state", None) is not None or self.phase is not None:
                raise RuntimeError(
                    "Overlapping LL forwards require independent counter state"
                )
            self.stream = stream
            _pending.state = self
        elif (
            getattr(_pending, "state", None) is not self
            or self.phase != 0
            or self.stream != stream
        ):
            raise RuntimeError("LL combine must follow dispatch on the same stream")
        mode = self._allocation_type(self.slots[slot])
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
                    raise RuntimeError(
                        "Synchronous native LL unexpectedly returned a recv hook"
                    )
                result = (*result[:-1], lambda: None)
            if mode.count != 1:
                raise RuntimeError(
                    f"Step5 LL native counter allocation ABI changed: {mode.seen}"
                )
            self.phase = slot
            return result
        except BaseException:
            self.valid = False
            raise


def install_counter_reset(buffer):
    if os.getenv("HCU_STEP5_LL_FUSED_COUNTER_RESET", "0") != "1":
        return
    from vllm.config import get_current_vllm_config
    from vllm_hcu.runtime_compat.step5 import uses_step5_ll_fused_restore

    cfg = get_current_vllm_config()
    if not uses_step5_ll_fused_restore(cfg):
        raise ValueError(
            "Fused LL counters require fixed LL TP4/DP2, TP4/DP4 or "
            "TP1/DP16, PP1, EP, FP32 residuals, DBO/SP off"
        )
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
    if (
        not state.valid
        or state.phase != 1
        or state.values.device != device
        or state.stream != torch.cuda.current_stream(device).cuda_stream
    ):
        raise RuntimeError("LL counter reset must follow receive on the same stream")
    # GPU reset is ordered by the caller's existing restore kernel. Graph
    # replay retains these exact pointers and stream dependencies.
    return state.values


def finish_pending_counter_reset():
    state = getattr(_pending, "state", None)
    if state is not None:
        state.phase = None
        _pending.state = None
