# SPDX-License-Identifier: Apache-2.0
"""Keep Step5's FP32 residual intact at the V2 runner's PP copy boundary.

The method is loaded in isolation like the other CPU model-helper tests, so
this regression does not require loading HCU operators or model weights.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _make_buffer(fp32_residual, dtype):
    source = Path(__file__).resolve().parents[2] / "vllm_hcu/models/step5_text.py"
    model = next(
        node for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "Step5Model"
    )
    method = next(
        node for node in model.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "make_empty_intermediate_tensors"
    )
    namespace = {"torch": torch, "IntermediateTensors": dict}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    model_state = SimpleNamespace(
        config=SimpleNamespace(hidden_size=4),
        fp32_residual_connection=fp32_residual,
    )
    return namespace["make_empty_intermediate_tensors"](
        model_state, 3, dtype, torch.device("cpu")
    )["hidden_states"]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_pp_copy_preserves_fp32_residual(dtype):
    buffer = _make_buffer(True, dtype)
    # Values intentionally lose precision in BF16/FP16. Match the V2 runner's
    # copy of received activations into its preallocated intermediate buffer.
    incoming = torch.tensor([[1.0001, -1.0001, 1.003, -1.003]]).repeat(2, 1)
    assert not torch.equal(incoming.to(dtype).float(), incoming)
    buffer[:2].copy_(incoming)
    assert buffer.dtype == torch.float32
    assert buffer.shape == (3, 4)
    assert torch.equal(buffer[:2], incoming)
    assert torch.count_nonzero(buffer[2]) == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_pp_without_fp32_residual_keeps_requested_dtype(dtype):
    buffer = _make_buffer(False, dtype)
    assert buffer.dtype == dtype
    assert buffer.shape == (3, 4)
    assert torch.count_nonzero(buffer) == 0
