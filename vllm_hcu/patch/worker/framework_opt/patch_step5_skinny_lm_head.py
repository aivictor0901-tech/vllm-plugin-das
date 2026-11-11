# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations
from types import ModuleType

import torch

from vllm_hcu.runtime_compat.step5_patch_contract import (
    already_applied,
    remember_targets,
    require_signatures,
)
from ._common import load_exact_module, require_callable

TARGET_MODULE = "vllm.model_executor.layers.logits_processor"
PATCH_ID = "worker.framework_opt.step5_skinny_lm_head"
TARGETS = ("vllm.model_executor.layers.logits_processor.LogitsProcessor._apply_head",)
_MARKER = "_vllm_hcu_step5_skinny_lm_head_applied"

SIGNATURES = (
    (
        "vllm.model_executor.layers.logits_processor.LogitsProcessor._apply_head",
        ("self", "lm_head", "hidden_states", "embedding_bias"),
    ),
)
REVIEWED_SOURCES = ()


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    import functools
    import os
    from vllm_hcu.platforms import envs as henvs

    cls = getattr(target, "LogitsProcessor", None)
    original = require_callable(cls, "_apply_head", TARGETS[0])

    @functools.wraps(original)
    def apply_head(self, lm_head, hidden_states, embedding_bias=None):
        # Only processors created by the Step5 models opt into this kernel.
        if (
            henvs.optional_custom_op_enabled(
                os.environ.get("STEP5_SKINNY_LM_HEAD", "1") != "0"
            )
            and getattr(self, "_step5_skinny_lm_head", False)
            and (self.head_dtype is None or self.head_dtype == hidden_states.dtype)
        ):
            from vllm_hcu.model_executor.layers import skinny_lm_head
            from vllm_hcu.model_executor.layers import skinny_lm_head_int8

            if embedding_bias is None and skinny_lm_head_int8.HcuLmHeadInt8Storage.matches(
                lm_head
            ):
                if skinny_lm_head_int8.usable_int8(
                    hidden_states, lm_head.weight, lm_head.weight_scale
                ):
                    target.logger.info_once(
                        "lm_head: using the INT8 skinny Triton GEMM for M <= %d",
                        skinny_lm_head_int8.MAX_M,
                    )
                    return skinny_lm_head_int8.skinny_w8a8_linear(
                        hidden_states, lm_head.weight, lm_head.weight_scale.reshape(-1)
                    )
                # Outside the skinny range the INT8 head has no GEMM of its own
                # here; dequantize for this call rather than running an
                # unquantized GEMM against int8 storage. M > 64 is prefill, so
                # the extra copy is amortized over many tokens.
                from vllm_hcu.models.step5_text import _dequantize_int8_channel

                target.logger.warning_once(
                    "lm_head: INT8 skinny GEMM covers M <= %d only; "
                    "dequantizing for larger batches",
                    skinny_lm_head_int8.MAX_M,
                )
                head = _dequantize_int8_channel(
                    "lm_head.weight", lm_head.weight, lm_head.weight_scale
                )
                return torch.nn.functional.linear(hidden_states, head).to(
                    torch.bfloat16
                )

            if (
                embedding_bias is None
                and type(lm_head.quant_method)
                in (target.UnquantizedEmbeddingMethod, target.UnquantizedLinearMethod)
                and skinny_lm_head.usable(hidden_states, lm_head.weight)
            ):
                target.logger.info_once(
                    "lm_head: using the skinny Triton GEMM for M <= %d",
                    skinny_lm_head.MAX_M,
                )
                return skinny_lm_head.skinny_linear(hidden_states, lm_head.weight)
        return original(self, lm_head, hidden_states, embedding_bias)

    cls._apply_head = apply_head
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
