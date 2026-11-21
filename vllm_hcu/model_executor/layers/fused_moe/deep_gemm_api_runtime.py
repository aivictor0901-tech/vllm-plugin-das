# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU DeepGEMM package/API compatibility for upstream lazy utilities."""

import functools
import importlib
import inspect
import os


@functools.cache
def accepts_ue8m0(impl):
    try:
        params = inspect.signature(impl).parameters
    except (TypeError, ValueError):
        return True
    return "disable_ue8m0_cast" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def install_api(target):
    original_import = target._import_deep_gemm
    original_get = target.get_mk_alignment_for_contiguous_layout
    original_set = target.set_mk_alignment_for_contiguous_layout

    @functools.wraps(original_import)
    def import_deep_gemm():
        module = original_import()
        if module is not None:
            return module
        try:
            return importlib.import_module("deepgemm")
        except ImportError:
            return None

    @functools.wraps(original_get)
    def get_alignment():
        target._lazy_init()
        if (
            getattr(target, "_get_mk_alignment_for_contiguous_layout_impl", None)
            is None
        ):
            return [16, 16]
        return original_get()

    @functools.wraps(original_set)
    def set_alignment(value):
        dg = import_deep_gemm()
        if dg is not None and hasattr(dg, "set_mk_alignment_for_contiguous_layout"):
            return original_set(value)

    def make_grouped(name, impl_name):
        original = getattr(target, name)

        @functools.wraps(original)
        def grouped(*args, **kwargs):
            target._lazy_init()
            impl = getattr(target, impl_name)
            if impl is None:
                return original(*args, **kwargs)
            if accepts_ue8m0(impl):
                kwargs["disable_ue8m0_cast"] = not target.is_deep_gemm_e8m0_used()
            return impl(*args, **kwargs)

        return grouped

    target._import_deep_gemm = import_deep_gemm
    target.get_mk_alignment_for_contiguous_layout = get_alignment
    target.set_mk_alignment_for_contiguous_layout = set_alignment
    for name, impl in (
        ("m_grouped_fp8_gemm_nt_contiguous", "_grouped_impl"),
        ("m_grouped_fp8_fp4_gemm_nt_contiguous", "_grouped_fp4_impl"),
        ("fp8_m_grouped_gemm_nt_masked", "_grouped_masked_impl"),
    ):
        setattr(target, name, make_grouped(name, impl))


# Step5 W4A8 custom-op registration uses the same lazy DeepGEMM API boundary.


def gemm_backend():
    backend = os.environ.get("VLLM_HCU_STEP5_W4A8_GEMM_BACKEND", "deepgemm")
    if backend != "deepgemm":
        raise ValueError("VLLM_HCU_STEP5_W4A8_GEMM_BACKEND supports only deepgemm")
    return backend


@functools.lru_cache(maxsize=1)
def _require_deepgemm_kernel():
    try:
        module = importlib.import_module("deepgemm")
        version = importlib.import_module("deepgemm.version")
        kernel = module.m_grouped_w4a8_gemm_nt_contiguous_hipc
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "Step5 DeepGEMM requires the contiguous W4A8 HIPC API"
        ) from error
    if (
        getattr(version, "build_code_object_version", None) != 5
        or getattr(version, "build_target_arch", None) != "gfx938"
    ):
        raise RuntimeError(
            "Step5 DeepGEMM requires the validated gfx938 CO5 wheel "
            "(deepgemm 2.1.0+das.dtk2610.4e0b87b.gfx938.co5). "
            "The default AICC build fails to launch HIPC kernels on this runtime."
        )
    return kernel


@functools.lru_cache(maxsize=1)
def register_deepgemm_op():
    import torch

    try:
        return torch.ops.vllm.step5_deepgemm_w4a8_gemm.default
    except AttributeError:
        from vllm.utils.torch_utils import direct_register_custom_op

        def _deepgemm_gemm(
            input: torch.Tensor,
            weight: torch.Tensor,
            output: torch.Tensor,
            input_scale: torch.Tensor,
            weight_scale: torch.Tensor,
            expert_ids: torch.Tensor,
            mode: int = 1002,
        ) -> None:
            # Weights already have their single-owner HIPC layout. Scales have already
            # been converted by the checkpoint loader; never repack or rescale here.
            _require_deepgemm_kernel()(
                (input, input_scale),
                (weight, weight_scale),
                output,
                expert_ids,
                config={"MODE": mode},
            )

        def _deepgemm_gemm_fake(
            input: torch.Tensor,
            weight: torch.Tensor,
            output: torch.Tensor,
            input_scale: torch.Tensor,
            weight_scale: torch.Tensor,
            expert_ids: torch.Tensor,
            mode: int = 1002,
        ) -> None:
            # The output is caller-owned. Keep this free of data-dependent shape guards
            # so FakeTensor and unbacked symbolic batch sizes can trace the operation.
            return None

        direct_register_custom_op(
            op_name="step5_deepgemm_w4a8_gemm",
            op_func=_deepgemm_gemm,
            mutates_args=["output"],
            fake_impl=_deepgemm_gemm_fake,
        )
    return torch.ops.vllm.step5_deepgemm_w4a8_gemm.default


def require_gemm_op(backend=None):
    backend = gemm_backend() if backend is None else backend
    if backend != "deepgemm":
        raise ValueError("Step5 W4A8 GEMM backend supports only deepgemm")
    _require_deepgemm_kernel()
    return register_deepgemm_op()
