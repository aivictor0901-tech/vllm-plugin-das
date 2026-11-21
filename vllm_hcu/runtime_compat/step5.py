# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Dependency-light Step5 family predicates and lazy LL compatibility helpers."""

import os


def is_step5_config(config):
    model = getattr(config, "model_config", config)
    hf = getattr(model, "hf_config", model)
    text = getattr(hf, "text_config", hf)
    return getattr(text, "model_type", "") in ("step5", "step5_mtp") or any(
        name.startswith("Step5") for name in (getattr(hf, "architectures", None) or ())
    )


def is_step5_owner(owner=None):
    """Resolve ownership at construction time; unknown owners never opt in."""
    marker = getattr(owner, "_hcu_step5_owner", None)
    if marker is not None:
        return marker is True
    config = getattr(owner, "_hcu_vllm_config", None)
    if config is None:
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
    return config is not None and is_step5_config(config)


def supports_step5_ll_topology(tp: int, dp: int, pp: int = 1) -> bool:
    # EP is TP * DP: 8 ranks for the one-host deployment, otherwise 16.
    return (tp, dp, pp) in ((4, 2, 1), (4, 4, 1), (1, 16, 1))


def uses_step5_ll_fused_restore(config) -> bool:
    p = config.parallel_config
    # Step5-VL stores the language model settings in its text subconfig.
    text_config = getattr(config.model_config, "hf_text_config", None)
    if text_config is None:
        hf_config = config.model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
    return bool(
        p.enable_expert_parallel
        and p.all2all_backend == "deepep_low_latency"
        and supports_step5_ll_topology(
            p.tensor_parallel_size, p.data_parallel_size, p.pipeline_parallel_size
        )
        and not p.enable_dbo
        and not config.compilation_config.pass_config.enable_sp
        and getattr(text_config, "fp32_residual_connection", False)
    )


def allows_static_step5_ll_eplb(config=None):
    if os.getenv("HCU_STEP5_LL_STATIC_EPLB", "0") != "1":
        return False
    from vllm.config import get_current_vllm_config_or_none
    from vllm_hcu.patch.config import get_hcu_config

    cfg = config if hasattr(config, "parallel_config") else None
    if cfg is None:
        cfg = getattr(config, "_hcu_vllm_config", None)
    if cfg is None:
        cfg = get_current_vllm_config_or_none()
    if cfg is None or not is_step5_config(cfg):
        return False
    p = cfg.parallel_config
    hcu = get_hcu_config(cfg)
    return bool(
        p.enable_eplb
        and p.enable_expert_parallel
        and p.all2all_backend == "deepep_low_latency"
        and supports_step5_ll_topology(
            p.tensor_parallel_size, p.data_parallel_size, p.pipeline_parallel_size
        )
        and not p.enable_dbo
        and not cfg.compilation_config.pass_config.enable_sp
        and not p.eplb_config.use_async
        and p.eplb_config.num_redundant_experts == 0
        and hcu.eplb_disable_rearrange
        and bool(hcu.expert_map_path) != bool(hcu.expert_map_record_path)
    )


def _map_int64_ids(self, topk_ids):
    """Reuse the native mapper with INT64 input; avoid a per-layer cast copy."""
    import torch
    from vllm.model_executor.layers.fused_moe.router.base_router import (
        BaseRouter,
        _eplb_map_and_record_i32_kernel,
    )
    from vllm.v1.worker.ubatching import dbo_current_ubatch_id

    state = self.eplb_state
    if state is None or topk_ids.dtype != torch.int64:
        return BaseRouter._apply_eplb_mapping(self, topk_ids)
    if topk_ids.numel() == 0:
        return topk_ids
    ids = topk_ids.contiguous()
    out = torch.empty_like(ids)
    load = state.expert_load_view
    assert load is not None and load.is_contiguous()
    count = state.logical_replica_count.contiguous()
    mapping = state.logical_to_physical_map.contiguous()
    unpadded = state.num_unpadded_tokens_tensors[dbo_current_ubatch_id()]
    # The upstream Triton kernel loads through a typed pointer and converts
    # the loaded ID to int64. Its historical i32 name does not restrict input.
    _eplb_map_and_record_i32_kernel[((ids.numel() + 255) // 256,)](
        ids,
        count,
        mapping,
        out,
        load,
        state.should_record_tensor,
        unpadded,
        count.shape[0],
        mapping.shape[1],
        load.shape[0],
        ids.numel(),
        ids.shape[-1],
        HAS_NUM_UNPADDED=unpadded is not None,
        BLOCK_SIZE=256,
    )
    return out


def install_int64_eplb_mapping(router, config):
    """Opt in only for the validated Step5 static-placement LL topology."""
    if os.getenv("HCU_STEP5_LL_EPLB_I64", "0") != "1":
        return
    if not allows_static_step5_ll_eplb(config):
        raise ValueError(
            "INT64 EPLB mapping requires static Step5 LL placement "
            "with TP4/DP2, TP4/DP4 or TP1/DP16"
        )
    from types import MethodType

    router._apply_eplb_mapping = MethodType(_map_int64_ids, router)
