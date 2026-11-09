# SPDX-License-Identifier: Apache-2.0
"""Step5 W8A8 INT8 dense projections for FULL and PIECEWISE graphs on gfx936.

Sources: local ops/step5/fp8_dense.py (padding and custom-op/fake registration),
model_executor/layers/quantization/int8_runtime.py (LightOp INT8 contract), and
step5-int8-dense-gfx936-tp4.patch (weight preparation and FULL-graph path).
BF16 weights remain available for eager execution and unsupported token sizes.
"""
from functools import cache

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger("vllm.step5.int8_dense")
SHAPES = frozenset(((4096, 3472), (4096, 5012), (4096, 3456),
                    (4096, 16), (3072, 4096), (4096, 6912), (3456, 4096)))
# Covers the default 51 capture sizes from the Step5 FULL_AND_PIECEWISE log.
# Other sizes retain BF16; compile_sizes is not a capture-size allowlist.
TOKENS = (1, 2, 4, *range(8, 257, 8), *range(272, 513, 16))
# gfx936 graph-replay measurements: SWA o_proj has a kernel performance cliff.
# Admit only measured sizes with >10% margin; the N=16 gate stays BF16 in PW.
_PIECEWISE_SWA_TOKENS = frozenset((
    24, 32, 40, 56, 64, 72, 80, 88,
    272, 288, 304, 320, 336, *range(368, 513, 16),
))
_WARMED = set()
_ANNOUNCED = set()


@cache
def _backend():
    from lightop.quant import per_token_quant_int8
    from lightop.gemm_ops import hipblaslt_w8a8_gemm
    return per_token_quant_int8, hipblaslt_w8a8_gemm


def _runtime_enabled(x):
    return (
        x.shape[0] in TOKENS
        and is_forward_context_available()
        and get_forward_context().cudagraph_runtime_mode in (
            CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE
        )
    )


def _use_int8(x, n):
    if not _runtime_enabled(x):
        return False
    if get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
        if n == 16:
            return False
        if (x.shape[1], n) == (3456, 4096):
            return x.shape[0] in _PIECEWISE_SWA_TOKENS
    return True


def graph_enabled(x):
    if x.ndim != 2 or x.dtype != torch.bfloat16:
        return False
    # Dynamo often traces with mode NONE and a symbolic M. Always retain the
    # opaque dispatch op in that graph; choose INT8/BF16 at execution time.
    if torch.compiler.is_compiling():
        return True
    return _runtime_enabled(x)


def _int8_impl(x, weight, scale, n):
    quant, gemm = _backend()
    m, k = x.shape
    if m == 0:
        return x.new_empty((0, n))
    xq, xs = quant(x)
    status, out = gemm(xq, weight, xs, scale, m, weight.shape[0], k,
                       "NT", torch.bfloat16)
    if status is not True or out.shape != (m, weight.shape[0]):
        raise RuntimeError("Step5 INT8 hipBLASLt returned invalid output")
    # The fake implementation promises a contiguous logical [M,N] output.
    return out[:, :n].contiguous()


def _step5_int8_dense(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
    bf16_weight: torch.Tensor,
) -> torch.Tensor:
    if _use_int8(x, bf16_weight.shape[1]):
        mode = get_forward_context().cudagraph_runtime_mode
        if mode not in _ANNOUNCED:
            logger.info("Step5 INT8 dense executing in %s mode (first M=%s, K=%s, N=%s)",
                        mode.name, x.shape[0], x.shape[1], bf16_weight.shape[1])
            _ANNOUNCED.add(mode)
        return _int8_impl(x, weight, scale, bf16_weight.shape[1])
    return torch.matmul(x, bf16_weight)


def _step5_int8_dense_fake(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
    bf16_weight: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], bf16_weight.shape[1]))


direct_register_custom_op(
    op_name="step5_int8_dense", op_func=_step5_int8_dense,
    mutates_args=[], fake_impl=_step5_int8_dense_fake,
)


def int8_linear(x, weight, scale, n, bf16_weight):
    """Opaque dynamic dispatch; BF16 fallback weight is in [K,N] layout."""
    assert n == bf16_weight.shape[1]
    return torch.ops.vllm.step5_int8_dense(x, weight, scale, bf16_weight)


@torch.no_grad()
def prepare_weight(w_kn):
    """Cache INT8 [Np,K] and FP32 [Np,1] scales; preserve the BF16 source."""
    if (w_kn.ndim != 2 or tuple(w_kn.shape) not in SHAPES
            or w_kn.dtype != torch.bfloat16 or not w_kn.is_cuda):
        return None, None
    arch = torch.cuda.get_device_properties(w_kn.device).gcnArchName
    if arch.split(":", 1)[0] != "gfx936":
        return None, None
    k, n = w_kn.shape
    npad = ((n + 255) // 256) * 256
    q = torch.zeros((npad, k), dtype=torch.int8, device=w_kn.device)
    scale = torch.ones((npad, 1), dtype=torch.float32, device=w_kn.device)
    for lo in range(0, n, 512):
        w = w_kn[:, lo:lo + 512].t().float()
        s = (w.abs().amax(dim=-1, keepdim=True) / 127.).clamp_min(1.e-12)
        q[lo:lo + len(w)] = (w / s).round().clamp(-127, 127).to(torch.int8)
        scale[lo:lo + len(w)] = s
    # LightOp must initialize its shape-specific plans outside capture.
    for m in TOKENS:
        key = (w_kn.device.index, m, npad, k)
        if key not in _WARMED:
            x = torch.zeros((m, k), dtype=torch.bfloat16, device=w_kn.device)
            _int8_impl(x, q, scale, n)
            _WARMED.add(key)
    return q, scale
