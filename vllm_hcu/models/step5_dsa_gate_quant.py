# SPDX-License-Identifier: Apache-2.0
"""Consume DSA's returned BF16 output directly in the existing INT8 gate.

No FlashAttention changes or new numerical kernels. Caller-owned quantized
outputs cross the attention graph boundary; the native temporary remains local
to that boundary. The original indexer and backend own all metadata/KV updates.
"""
import os

import torch
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm_hcu.platforms import envs as henvs
from vllm_hcu.ops.step5.attention_int8 import (
    _gate_quant_int8_kernel,
    usable_gate_quant_int8,
)

OP_NAME = "step5_dsa_gate_quant_no_copy"
ENABLED = henvs.optional_custom_op_enabled(
    os.getenv("VLLM_HCU_STEP5_DSA_GATE_QUANT_NO_COPY", "0") == "1"
)


def configured_for(module) -> bool:
    """Reuse the existing INT8 provider; do not select a new quantization path."""
    attn = module.attn
    return (
        ENABLED and module.is_sparse_layer and module.int8_gate_quant
        and not module.fused_gate
        and callable(getattr(attn.impl, "forward_gate_quant_no_copy", None))
        # Attention.forward can own query quantization / scale calibration.
        # Do not bypass those responsibilities on configurations needing them.
        and getattr(attn, "query_quant", None) is None
        and not getattr(attn, "calculate_kv_scales", False)
    )


def require_graph_boundary() -> None:
    try:
        config = get_current_vllm_config()
    except AssertionError:
        return  # Standalone operator tests have no serving config.
    compilation = config.compilation_config
    if compilation.cudagraph_mode == CUDAGraphMode.NONE:
        return
    if "vllm::" + OP_NAME not in (compilation.splitting_ops or []):
        raise RuntimeError(
            "DSA no-copy gate requires vllm::step5_dsa_gate_quant_no_copy "
            "in explicit splitting_ops (automatically added for default configs)"
        )


def can_use(module, q, gate) -> bool:
    # The existing fast gate only runs on the compiled path: eager sigmoid
    # has a different BF16 rounding boundary and must retain its old behavior.
    return (
        module.dsa_gate_quant_no_copy and torch.compiler.is_compiling()
        and gate is not None and usable_gate_quant_int8(q, gate)
    )


def write_quantized_rows(attention_output, gate, xq, xs, valid: int) -> None:
    """Use exactly the existing kernel; no intermediate BF16 output copy."""
    if valid == 0:
        return
    flat = attention_output[:valid].view(valid, -1)
    if not usable_gate_quant_int8(flat, gate[:valid]):
        raise ValueError("DSA no-copy output violates the existing INT8 gate contract")
    _gate_quant_int8_kernel[(valid,)](
        flat, gate, xq, xs, gate.stride(0),
        NH=gate.shape[1], D=192, B=xq.shape[1] // 3,
        num_warps=2, num_stages=1,
    )


def _op_impl(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    gate: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    metadata, layer, kv_cache, _ = get_attention_context(_resolve_layer_name(layer_name))
    if metadata is None:
        xq.zero_()
        xs.fill_(1.0)
        return
    layer.impl.forward_gate_quant_no_copy(
        layer, query, key, value, kv_cache, metadata, gate, xq, xs
    )


def _op_fake(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    gate: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    # Never read the real attention context while tracing.
    return None


if ENABLED:
    direct_register_custom_op(
        op_name=OP_NAME, op_func=_op_impl, mutates_args=["xq", "xs"], fake_impl=_op_fake,
    )


def forward(module, q, k, v, gate):
    # These allocations belong to the compiled caller, not the opaque op.
    # The following graph piece must see the same captured xq/xs addresses.
    xq = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    xs = torch.empty((q.shape[0], 1), dtype=torch.float32, device=q.device)
    torch.ops.vllm.step5_dsa_gate_quant_no_copy(
        q.view(-1, module.num_heads, module.head_dim),
        k.view(-1, module.num_kv_heads, module.head_dim),
        v.view(-1, module.num_kv_heads, module.head_dim),
        gate, xq, xs, _encode_layer_name(module.attn.layer_name),
    )
    return xq, xs
