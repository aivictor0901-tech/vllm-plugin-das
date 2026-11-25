# SPDX-License-Identifier: Apache-2.0
"""Opt-in replicated shared experts with attention RS and owner-output AG.

Only the target model's MoE layers use this layout. Attention still consumes
replicated rows; the residual and both expert branches operate on the identical
contiguous TP-owned token slice. Dense FFNs and MTP retain their original path.
"""
from __future__ import annotations

import functools
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar

import torch
import torch.nn.functional as F

_OWNER_EXECUTION = ContextVar("step5_replicated_shared_owner", default=False)


def owner_execution_active() -> bool:
    return _OWNER_EXECUTION.get()


def uses_replicated_shared_ll_restore(config) -> bool:
    """Keep LL reuse on all layers sharing the new path's communication buffer.

    Draft layers retain sharded shared weights, but must run the existing
    generic fused restore to reset reused counters after each receive.
    """
    return replicated_shared_enabled(config, "")


@contextmanager
def owner_execution():
    """Scope shape tuning to the actual new owner-row callback, including errors."""
    token = _OWNER_EXECUTION.set(True)
    try:
        yield
    finally:
        _OWNER_EXECUTION.reset(token)


@functools.lru_cache(maxsize=None)
def _fallback(reason: str) -> bool:
    logging.getLogger(__name__).warning(
        "Step5 replicated shared RS/AG is inactive; using the original layout: %s",
        reason,
    )
    return False


def replicated_shared_enabled(config, prefix: str) -> bool:
    if os.environ.get("HCU_STEP5_REPLICATED_SHARED_RS_AG", "0") != "1":
        return False
    # MTP reuses Step5DecoderLayer, but has a different sampling/forward ABI.
    if "mtp_block" in prefix:
        return False
    from vllm_hcu.platforms import envs as henvs
    if not henvs.custom_ops_enabled():
        return False
    from vllm_hcu.patch.config import get_hcu_config

    parallel = config.parallel_config
    text = getattr(config.model_config, "hf_text_config", None)
    if text is None:
        hf = config.model_config.hf_config
        text = getattr(hf, "text_config", hf)
    sizes = (parallel.tensor_parallel_size, parallel.data_parallel_size,
             parallel.pipeline_parallel_size)
    if (parallel.tensor_parallel_size == 1
            and parallel.data_parallel_size == 1):
        return _fallback("TP1/DP1 already owns complete token rows and weights")
    if (not all(isinstance(size, int) and not isinstance(size, bool) and size > 0
                for size in sizes)
            or not parallel.enable_expert_parallel
            or parallel.all2all_backend != "deepep_low_latency"
            or parallel.enable_dbo
            or config.compilation_config.pass_config.enable_sp
            or not getattr(text, "fp32_residual_connection", False)
            or getattr(parallel, "prefill_context_parallel_size", 1) != 1
            or getattr(parallel, "decode_context_parallel_size", 1) != 1
            or get_hcu_config(config).enable_custom_sp
            or getattr(config, "lora_config", None) is not None):
        return _fallback(
            "this layout needs fixed DeepEP LL, EP, FP32 residuals, "
            "PCP1/DCP1, and DBO/SP/LoRA disabled"
        )
    return True


def owner_residual(residual: torch.Tensor, tp_size: int, tp_rank: int) -> torch.Tensor:
    """Select the same rows as reduce_scatter, with NaN skip rows on padding.

    NaN padding preserves the existing Step5 router's -1 expert sentinel.
    Zero rows would otherwise dispatch fake tokens to valid experts.
    """
    rows = (residual.shape[0] + tp_size - 1) // tp_size
    start = min(tp_rank * rows, residual.shape[0])
    end = min((tp_rank + 1) * rows, residual.shape[0])
    # Keep the slice with its dynamic bounds across vLLM FX split points.
    # Tensor getitem is reassigned to the input's partition by split_graph.
    local = residual.narrow(0, start, end - start)
    return F.pad(
        local, (0, 0, 0, rows - local.shape[0]),
        value=float("nan"),
    )


def validate_replicated_shared_projection(mlp, intermediate_size: int) -> None:
    gate_up, down = mlp.gate_up_proj, mlp.down_proj
    if (gate_up.tp_size != 1 or down.tp_size != 1
            or gate_up.output_size_per_partition != 2 * intermediate_size
            or down.input_size_per_partition != intermediate_size
            or down.reduce_results):
        raise ValueError("Step5 shared projections did not retain full replicated weights")


def _tp_group(tp_size: int, group_name: str):
    from vllm.distributed import get_tp_group
    group = get_tp_group()
    if group.world_size != tp_size or group.unique_name != group_name:
        raise RuntimeError("Step5 replicated shared TP group changed")
    return group


def _attention_rs(x: torch.Tensor, tp_size: int, group_name: str) -> torch.Tensor:
    group = _tp_group(tp_size, group_name)
    if x.dim() != 2:
        raise ValueError("Step5 attention RS requires a 2D row-parallel output")
    if tp_size == 1:
        # Opaque custom-op outputs must not alias inputs, including TP1.
        return x.clone()
    if x.shape[0] == 0:
        return torch.empty_like(x)
    rows = (x.shape[0] + tp_size - 1) // tp_size
    padding = rows * tp_size - x.shape[0]
    padded = F.pad(x, (0, 0, 0, padding)) if padding else x
    padded = padded.contiguous()
    fabric = getattr(group, "_step5_shared_fabric", None)
    if fabric is not None and fabric.should_custom_rs(padded, 0):
        out = torch.empty((rows, x.shape[1]), dtype=x.dtype, device=x.device)
        fabric.reduce_scatter(padded, out, dim=0)
        return out
    return group.reduce_scatter(padded, dim=0)


def _attention_rs_fake(x, tp_size, group_name):
    return x.new_empty(((x.shape[0] + tp_size - 1) // tp_size, x.shape[1]))


def _finish_owner(shared: torch.Tensor, routed: torch.Tensor,
                  residual: torch.Tensor, num_tokens: int,
                  tp_size: int, group_name: str) -> torch.Tensor:
    group = _tp_group(tp_size, group_name)
    if shared.dim() != 2 or routed.dim() != 2 or residual.dim() != 2:
        raise ValueError("Step5 replicated shared expects 2D owner tensors")
    rows = (num_tokens + tp_size - 1) // tp_size
    expected = (rows, residual.shape[-1])
    if (tuple(shared.shape) != expected or tuple(routed.shape) != expected
            or tuple(residual.shape) != expected
            or not shared.is_contiguous() or not routed.is_contiguous()):
        raise ValueError("Step5 replicated shared token-owner layout mismatch")
    from .step5_deepep import _ll_shared_routed_partial
    from .step5_ll_runtime import take_pending_counters, finish_pending_counter_reset

    out = torch.empty(expected, dtype=torch.float32, device=shared.device)
    counters = take_pending_counters(shared.device)
    if out.numel() or counters is not None:
        # Reuse the existing restore kernel and its verified counter lifecycle.
        # TP=1 here means BOTH inputs contain complete local-owner results;
        # it does not change the actual distributed TP group.
        _ll_shared_routed_partial[(max(1, (out.numel() + 1023) // 1024),)](
            shared, routed, out, N=rows, LOCAL_N=rows, H=expected[1],
            TP=1, RANK=0, BLOCK=1024, COUNTERS=counters,
            RESET_COUNTERS=counters is not None, num_warps=4,
        )
    finish_pending_counter_reset()
    out.add_(residual.float())
    if tp_size == 1 or not out.numel():
        return out[:num_tokens]
    # Gather AFTER the FP32 residual addition, so a second residual gather is
    # unnecessary and the next target layer sees the normal replicated input.
    fabric = getattr(group, "_step5_shared_fabric", None)
    if fabric is not None and fabric.should_custom_ag(out, 0):
        return fabric.all_gather_unreg(out, dim=0)[:num_tokens]
    return group.all_gather(out, dim=0)[:num_tokens]


def _finish_owner_fake(shared, routed, residual, num_tokens, tp_size, group_name):
    return shared.new_empty((num_tokens, shared.shape[1]), dtype=torch.float32)


@functools.lru_cache(maxsize=1)
def attention_rs_op():
    try:
        return torch.ops.vllm.step5_replicated_shared_attention_rs.default
    except AttributeError:
        from vllm.utils.torch_utils import direct_register_custom_op
        direct_register_custom_op(
            op_name="step5_replicated_shared_attention_rs", op_func=_attention_rs,
            fake_impl=_attention_rs_fake, mutates_args=[],
        )
        return torch.ops.vllm.step5_replicated_shared_attention_rs.default


@functools.lru_cache(maxsize=1)
def finish_owner_op():
    try:
        return torch.ops.vllm.step5_replicated_finish_owner.default
    except AttributeError:
        from vllm.utils.torch_utils import direct_register_custom_op
        direct_register_custom_op(
            op_name="step5_replicated_finish_owner", op_func=_finish_owner,
            fake_impl=_finish_owner_fake, mutates_args=[],
        )
        return torch.ops.vllm.step5_replicated_finish_owner.default