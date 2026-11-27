# SPDX-License-Identifier: Apache-2.0
"""Shared eligibility for Step5's serial LL optimizations."""


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
