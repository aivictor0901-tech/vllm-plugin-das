# SPDX-License-Identifier: Apache-2.0
"""Graph-compatible Step5 W4A8 GEMMs using the validated DeepGEMM wheel."""
from functools import lru_cache
import importlib
import os

import torch


def gemm_backend():
    backend = os.environ.get("VLLM_HCU_STEP5_W4A8_GEMM_BACKEND", "deepgemm")
    if backend != "deepgemm":
        raise ValueError("VLLM_HCU_STEP5_W4A8_GEMM_BACKEND supports only deepgemm")
    return backend


@lru_cache(maxsize=1)
def _require_deepgemm_kernel():
    try:
        module = importlib.import_module("deepgemm")
        version = importlib.import_module("deepgemm.version")
        kernel = module.m_grouped_w4a8_gemm_nt_contiguous_hipc
    except (ImportError, AttributeError) as error:
        raise RuntimeError("Step5 DeepGEMM requires the contiguous W4A8 HIPC API") from error
    if (getattr(version, "build_code_object_version", None) != 5
            or getattr(version, "build_target_arch", None) != "gfx938"):
        raise RuntimeError(
            "Step5 DeepGEMM requires the validated gfx938 CO5 wheel "
            "(deepgemm 2.1.0+das.dtk2610.4e0b87b.gfx938.co5). "
            "The default AICC build fails to launch HIPC kernels on this runtime."
        )
    return kernel


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
        (input, input_scale), (weight, weight_scale), output, expert_ids,
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


@lru_cache(maxsize=1)
def register_deepgemm_op():
    try:
        return torch.ops.vllm.step5_deepgemm_w4a8_gemm.default
    except AttributeError:
        from vllm.utils.torch_utils import direct_register_custom_op

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
