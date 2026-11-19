# SPDX-License-Identifier: Apache-2.0
"""PCP-consistent Step5 MTP decisions before draft/state advancement.

The target attention runs with local Q and full per-rank caches.  The existing
runner restores the global batch for MTP, then every last-PP PCP worker runs
the same global draft rows.  Greedy draft decisions and target sampled tokens
plus rejection counts must be identical before the next draft or cache update.

This is a decision-consistency adapter, not a numerical-error correction.
Probabilistic draft distributions and adaptive verification remain fail-closed.
No runner, global class or non-Step5 model is patched by importing this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
from typing import Any


@dataclass(frozen=True)
class Step5PCPMTPSyncState:
    group: Any
    speculator: Any
    num_speculative_tokens: int


def validate_step5_pcp_mtp_sync_config(vllm_config: object) -> None:
    """Preflight called only within the Step5 PCP configuration branch."""
    spec = getattr(vllm_config, "speculative_config", None)
    if spec is None:
        return
    if getattr(spec, "method", None) != "mtp":
        raise ValueError("Step5 PCP decision sync requires built-in MTP")
    if not bool(getattr(vllm_config.model_config, "enforce_eager", False)):
        raise ValueError("Step5 PCP MTP decision sync requires eager execution")
    if getattr(spec, "draft_sample_method", "greedy") != "greedy":
        raise ValueError(
            "Step5 PCP MTP decision sync supports greedy drafts only; "
            "probabilistic draft logits caches are not synchronized"
        )
    if bool(getattr(spec, "enable_adaptive_verification", False)):
        raise ValueError("Step5 PCP MTP decision sync does not cover adaptive verification")
    if int(getattr(spec, "num_speculative_tokens", 0)) not in (1, 2, 3):
        raise ValueError("Step5 PCP MTP decision sync requires one to three draft tokens")


def _in_replicated_mtp_batch() -> bool:
    from vllm_hcu.model_executor.layers.attention.pcp import in_replicated_mtp_batch

    return bool(in_replicated_mtp_batch())


def _broadcast_draft_tokens(tokens: Any, group: Any) -> Any:
    import torch

    if tokens.ndim != 1 or tokens.dtype not in (torch.int32, torch.int64):
        raise TypeError("Step5 greedy MTP must return a one-dimensional integer tensor")
    if tokens.numel() == 0:
        return tokens
    # The coordinator's src is group-local: in CP8/PP2 the last-PP source is
    # global rank 8, not global rank 0. Consume the returned buffer explicitly.
    return group.broadcast(tokens.contiguous(), src=0)


def install_step5_pcp_mtp_sync(runner: object, *, group: Any | None = None) -> bool:
    """Call after speculator and distributed groups exist (KV init is suitable).

    Only the final PP stage owns a speculator.  The normal PP sampled-token
    and draft-token broadcasts distribute these already-synchronized decisions
    to the preceding stages; this adapter adds no second PP communication.
    """
    from vllm_hcu.v1.step5_pcp_config import is_step5_pcp

    config = runner.vllm_config
    if not is_step5_pcp(config):
        return False
    spec = getattr(config, "speculative_config", None)
    if spec is None:
        return False
    validate_step5_pcp_mtp_sync_config(config)
    speculator = getattr(runner, "speculator", None)
    if speculator is None:
        return False
    if not bool(getattr(runner, "is_last_pp_rank", True)):
        raise ValueError("Step5 PCP MTP speculator must belong to the final PP stage")
    if getattr(speculator, "draft_logits", None) is not None:
        raise ValueError("Step5 PCP greedy draft sync requires draft_logits=None")
    if group is None:
        from vllm.distributed import get_pcp_group

        group = get_pcp_group()
    if int(group.world_size) != int(config.parallel_config.prefill_context_parallel_size):
        raise RuntimeError("Step5 PCP MTP group size disagrees with the configured PCP size")
    if not 0 <= int(group.rank_in_group) < int(group.world_size):
        raise RuntimeError("Step5 PCP MTP group rank is out of range")
    previous = getattr(runner, "_step5_pcp_mtp_sync", None)
    if previous is not None:
        if previous.group is not group or previous.speculator is not speculator:
            raise RuntimeError("Step5 PCP MTP group/speculator changed after sync installation")
        return False
    if getattr(speculator, "_step5_pcp_mtp_sync", None) is not None:
        raise RuntimeError("Step5 PCP MTP speculator is already owned by another runner")
    original = speculator.sample_draft
    state = Step5PCPMTPSyncState(group, speculator, int(spec.num_speculative_tokens))

    @functools.wraps(original)
    def sample_draft(*args, **kwargs):
        # A runtime guard also catches a cache policy changed after preflight.
        if getattr(speculator, "draft_logits", None) is not None:
            raise RuntimeError("Step5 PCP MTP probabilistic draft cache appeared after installation")
        tokens = original(*args, **kwargs)
        if _in_replicated_mtp_batch():
            tokens = _broadcast_draft_tokens(tokens, group)
        # generate_step5_drafts writes this result to draft_tokens and consumes
        # it as the very next substep's input. Final propose()-only sync is late.
        return tokens

    speculator.sample_draft = sample_draft
    speculator._step5_pcp_mtp_sync = state
    runner._step5_pcp_mtp_sync = state
    return True


def sync_step5_pcp_sampler_output(runner: object, out: Any) -> Any:
    """Call immediately after super().sample(), before any state update.

    Preserve the original tuple, SamplerOutput and count buffers because the
    caller, async output copy, PP handler and postprocess_sampled may alias
    them.  One payload commits token IDs, sampled lengths and rejection lengths
    together.  Unused token columns are sent as -1, never uninitialized bytes.
    """
    state = getattr(runner, "_step5_pcp_mtp_sync", None)
    if state is None or not _in_replicated_mtp_batch():
        return out
    import torch

    output, num_sampled, num_rejected = out
    sampled = output.sampled_token_ids
    if num_sampled is None or num_rejected is None:
        raise RuntimeError("Step5 MTP sampler must return sampled and rejected counts")
    if sampled.ndim != 2:
        raise ValueError("Step5 MTP sampled tokens must have shape [requests, width]")
    n, width = sampled.shape
    if num_sampled.shape != (n,) or num_rejected.shape != (n,):
        raise ValueError("Step5 MTP sampled/rejected counts do not match the global batch")
    if any(t.dtype not in (torch.int32, torch.int64)
           for t in (sampled, num_sampled, num_rejected)):
        raise TypeError("Step5 MTP decision tensors must use integer dtypes")
    if sampled.device != num_sampled.device or sampled.device != num_rejected.device:
        raise ValueError("Step5 MTP decision tensors must use the same device")
    if n == 0:
        return out
    columns = torch.arange(width, device=sampled.device)
    valid_tokens = sampled.masked_fill(columns[None, :] >= num_sampled[:, None], -1)
    payload = torch.cat((valid_tokens.reshape(-1).to(torch.int64),
                         num_sampled.to(torch.int64), num_rejected.to(torch.int64)))
    payload = state.group.broadcast(payload, src=0)
    if payload.shape != (n * width + 2 * n,):
        raise RuntimeError("Step5 PCP MTP broadcast returned an invalid decision payload")
    end = n * width
    sampled.copy_(payload[:end].view(n, width))
    num_sampled.copy_(payload[end:end + n])
    num_rejected.copy_(payload[end + n:end + 2 * n])
    output.num_sampled = num_sampled
    output.num_rejected = num_rejected
    return out


__all__ = [
    "Step5PCPMTPSyncState", "validate_step5_pcp_mtp_sync_config",
    "install_step5_pcp_mtp_sync", "sync_step5_pcp_sampler_output",
]
