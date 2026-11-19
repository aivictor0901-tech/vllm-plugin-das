"""Opt-in contract for Step5 local-query/full-cache prefill CP.

This path is isolated from legacy Step5 PD and other model families. Full KV
is replicated within a PCP group; only target prefill queries are partitioned.
"""
from __future__ import annotations

import os


def _step5_family(config: object) -> bool:
    model = getattr(config, "model_config", config)
    hf = getattr(model, "hf_config", model)
    text = getattr(hf, "text_config", hf)
    architectures = getattr(hf, "architectures", ()) or getattr(model, "architectures", ()) or ()
    return getattr(text, "model_type", "") in ("step5", "step5_mtp") or any(
        str(name).startswith("Step5") for name in architectures
    )


def is_step5_pcp(config: object) -> bool:
    parallel = getattr(config, "parallel_config", None)
    return bool(
        os.environ.get("VLLM_STEP5_PCP", "0") == "1"
        and _step5_family(config)
        and int(getattr(parallel, "prefill_context_parallel_size", 1)) > 1
    )


def validate_step5_pcp_config(config: object) -> None:
    if not is_step5_pcp(config):
        raise ValueError("Step5 PCP requires its model family, PCP>1 and VLLM_STEP5_PCP=1")
    parallel, model = config.parallel_config, config.model_config
    if not getattr(config, "use_v2_model_runner", False):
        raise ValueError("Step5 PCP requires Model Runner V2")
    topology = (
        int(parallel.tensor_parallel_size), int(parallel.prefill_context_parallel_size),
        int(parallel.pipeline_parallel_size), int(parallel.data_parallel_size),
        int(parallel.decode_context_parallel_size),
    )
    tp, pcp, pp, dp, dcp = topology
    if tp != 1 or pcp not in (2, 4, 8) or pp not in (1, 2) or dp != 1 or dcp != 1:
        raise ValueError("Step5 PCP supports TP1/PCP2,4,8/PP1,2/DP1/DCP1; got " + str(topology))
    if not parallel.enable_expert_parallel:
        raise ValueError("Step5 PCP requires expert parallelism over its PCP group")
    if parallel.all2all_backend != "allgather_reducescatter":
        raise ValueError("Step5 PCP currently requires allgather_reducescatter")
    if getattr(parallel, "enable_eplb", False):
        raise ValueError("Step5 PCP does not yet support EPLB")
    if not getattr(model, "enforce_eager", False):
        raise ValueError("Step5 PCP requires eager execution until graph replay is validated")
    if getattr(model, "quantization", None) != "compressed-tensors":
        raise ValueError("Step5 PCP currently targets the packed mixed INT4/INT8 checkpoint")
    mm = getattr(model, "multimodal_config", None)
    if getattr(model, "is_multimodal_model", False) and not getattr(mm, "language_model_only", False):
        raise ValueError("Step5 PCP requires --language-model-only")
    if getattr(config, "lora_config", None) is not None:
        raise ValueError("Step5 PCP does not support LoRA")
    if getattr(config, "kv_transfer_config", None) is not None:
        raise ValueError("Step5 PCP currently supports integrated IFB only; legacy PD is unchanged")
    cache = config.cache_config
    if getattr(cache, "enable_prefix_caching", False):
        raise ValueError("Step5 PCP prefix-cache ownership is not validated")
    if getattr(cache, "kv_offloading_size", None) is not None:
        raise ValueError("Step5 PCP does not support KV offload")
    if getattr(cache, "cache_dtype", None) not in ("fp8_e4m3", "fp8"):
        raise ValueError("Step5 PCP currently requires FP8 E4M3 KV")
    if int(getattr(parallel, "cp_kv_cache_interleave_size", 1)) != 1:
        raise ValueError("Step5 PCP requires full non-interleaved KV caches")
    spec = getattr(config, "speculative_config", None)
    if spec is not None and (
        getattr(spec, "method", None) != "mtp"
        or int(getattr(spec, "num_speculative_tokens", 0)) not in (1, 2, 3)
    ):
        raise ValueError("Step5 PCP supports built-in MTP1..3 only")
    from vllm_hcu.v1.step5_pcp_mtp import validate_step5_pcp_mtp_sync_config

    validate_step5_pcp_mtp_sync_config(config)


__all__ = ["is_step5_pcp", "validate_step5_pcp_config"]
