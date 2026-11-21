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
    unquantized_types = tuple(
        kind
        for kind in (
            getattr(target, "UnquantizedEmbeddingMethod", None),
            getattr(target, "UnquantizedLinearMethod", None),
        )
        if isinstance(kind, type)
    )

    @functools.wraps(original)
    def apply_head(self, lm_head, hidden_states, embedding_bias=None):
        step5_head = getattr(self, "_step5_skinny_lm_head", False)
        skinny_enabled = henvs.optional_custom_op_enabled(
            os.environ.get("STEP5_SKINNY_LM_HEAD", "1") != "0"
        )
        # INT8 storage is a loading decision. Disabling an optional skinny
        # kernel must never hand its raw integer weights to the unquantized
        # ParallelLMHead method. All non-skinny cases still use true W8A8.
        int8_storage = getattr(lm_head, "_step5_int8_storage", False) or (
            type(getattr(lm_head, "quant_method", None)) in unquantized_types
        )
        if (
            step5_head
            and int8_storage
            and getattr(lm_head.weight, "dtype", None) == torch.int8
        ):
            from vllm_hcu.model_executor.layers import skinny_lm_head_int8
            from vllm_hcu.model_executor.layers.quantization.int8_runtime import (
                apply_int8_linear,
            )

            if not skinny_lm_head_int8.HcuLmHeadInt8Storage.matches(lm_head):
                raise ValueError("Step5 INT8 lm_head is missing FP32 channel scales")
            compute_dtype = self.head_dtype or hidden_states.dtype
            if compute_dtype not in (torch.bfloat16, torch.float16):
                raise ValueError(
                    "Step5 INT8 lm_head requires BF16 or FP16 activations, got "
                    f"{compute_dtype}"
                )
            if hidden_states.dtype != compute_dtype:
                hidden_states = hidden_states.to(compute_dtype)
            n, k = lm_head.weight.shape
            if hidden_states.ndim < 2 or hidden_states.shape[-1] != k:
                raise ValueError("Step5 INT8 lm_head activation shape does not match weight")
            if hidden_states.numel() == 0:
                return hidden_states.new_empty((*hidden_states.shape[:-1], n))
            if (
                skinny_enabled
                and embedding_bias is None
                and skinny_lm_head_int8.usable_int8(
                    hidden_states, lm_head.weight, lm_head.weight_scale
                )
            ):
                target.logger.info_once(
                    "lm_head: using the INT8 skinny Triton GEMM for M <= %d",
                    skinny_lm_head_int8.MAX_M,
                )
                return skinny_lm_head_int8.skinny_w8a8_linear(
                    hidden_states, lm_head.weight, lm_head.weight_scale.reshape(-1)
                )
            return apply_int8_linear(
                hidden_states,
                lm_head.weight,
                lm_head.weight_scale,
                compute_dtype,
                bias=embedding_bias,
            )

        # Native quantized heads (including MTP) own their post-load layout and
        # dispatch. In particular master-off may transpose their INT8 weights.
        if step5_head and getattr(lm_head.weight, "dtype", None) == torch.int8:
            return original(self, lm_head, hidden_states, embedding_bias)

        # Only processors created by the Step5 models opt into this kernel.
        if (
            skinny_enabled
            and step5_head
            and (self.head_dtype is None or self.head_dtype == hidden_states.dtype)
        ):
            from vllm_hcu.model_executor.layers import skinny_lm_head
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
