# SPDX-License-Identifier: Apache-2.0
"""Step5 attention is built without bf16 storage and quantized layer by layer."""

import os
import subprocess
import sys

_PRELUDE = r"""
import torch
from torch import nn
from vllm_hcu import hcu_platform_register_model
hcu_platform_register_model()
from vllm_hcu.models import step5_text as m
import vllm.model_executor.layers.linear as hcu_linear
assert hcu_linear.__name__ and hasattr(hcu_linear, "lazy_weight_storage")


class FakeAttention(nn.Module):
    def __init__(self, expected):
        super().__init__()
        self.expected = frozenset(expected)
        self.events = []

    def materialize_bf16_attention(self):
        self.events.append("materialize")

    def expected_checkpoint_weights(self):
        return self.expected

    def quantize_loaded_attention(self):
        self.events.append("quantize")


class FakeLayer(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.self_attn = attention


m.Step5Attention = FakeAttention


def run(layers, names):
    log = []
    stream = ((name, torch.zeros(1)) for name in names)
    for name, _ in m._quantize_attention_as_loaded(layers, stream):
        log.append((name, [list(layer.self_attn.events) for layer in layers]))
    return log
"""


def _run(body: str) -> None:
    env = os.environ.copy()
    env.setdefault("VLLM_HCU_USE_CUSTOM_OPS", "1")
    result = subprocess.run(
        [sys.executable, "-c", _PRELUDE + body],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_lazy_weight_has_shape_but_no_storage_until_materialized():
    _run(r"""
with hcu_linear.lazy_weight_storage():
    weight = hcu_linear._empty_weight(4096, 128, dtype=torch.bfloat16)
assert tuple(weight.shape) == (4096, 128)
assert weight.untyped_storage().nbytes() == weight.element_size()
assert hcu_linear.is_lazy_weight(weight)

param = nn.Parameter(weight, requires_grad=False)
assert hcu_linear.materialize_lazy_weight(param)
assert tuple(param.shape) == (4096, 128)
assert param.is_contiguous()
assert not hcu_linear.is_lazy_weight(param)
assert not hcu_linear.materialize_lazy_weight(param)
""")


def test_weights_are_real_outside_the_lazy_context_and_after_errors():
    _run(r"""
weight = hcu_linear._empty_weight(8, 8, dtype=torch.bfloat16)
assert not hcu_linear.is_lazy_weight(weight)
assert weight.untyped_storage().nbytes() == 8 * 8 * weight.element_size()

try:
    with hcu_linear.lazy_weight_storage():
        raise RuntimeError
except RuntimeError:
    pass
assert not hcu_linear.is_lazy_weight(
    hcu_linear._empty_weight(8, 8, dtype=torch.bfloat16)
)
""")


def test_unquantized_linear_allocates_no_weight_inside_lazy_context():
    _run(r"""
import vllm.model_executor.parameter as parameter_module

parameter_module.get_tensor_model_parallel_rank = lambda: 0
parameter_module.get_tensor_model_parallel_world_size = lambda: 1
method = hcu_linear.UnquantizedLinearMethod()


def create(lazy):
    layer = nn.Module()
    with hcu_linear.lazy_weight_storage(lazy):
        method.create_weights(
            layer, 256, [512], 256, 512, torch.bfloat16,
            weight_loader=lambda *args, **kwargs: None,
        )
    return layer


lazy, real = create(True), create(False)
assert hcu_linear.is_lazy_weight(lazy.weight)
assert not hcu_linear.is_lazy_weight(real.weight)
assert lazy.weight.shape == real.weight.shape
assert lazy.weight.untyped_storage().nbytes() < real.weight.untyped_storage().nbytes()
""")


def test_layer_is_quantized_after_its_last_expected_tensor_is_loaded():
    _run(r"""
attention = FakeAttention({"q_proj", "k_proj", "v_proj", "o_proj", "g_proj"})
names = [
    "model.layers.0.self_attn.q_norm.weight",
    "model.layers.0.self_attn.g_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
]
log = run([FakeLayer(attention)], names)
events = {name: seen[0] for name, seen in log}
assert events["model.layers.0.self_attn.g_proj.weight"] == ["materialize"]
assert "quantize" not in events["model.layers.0.self_attn.v_proj.weight"]
assert events["model.layers.0.mlp.gate_proj.weight"][-1] == "quantize"
assert attention.events.count("quantize") == 1
""")


def test_quantization_waits_for_indexer_tensors_when_they_are_expected():
    _run(r"""
attention = FakeAttention({
    "q_proj", "k_proj", "v_proj", "o_proj",
    "sparse_indexer_q", "sparse_indexer_k", "sparse_indexer_z", "sparse_indexer_w",
})
base = "model.layers.3.self_attn."
names = [base + f"{k}.weight" for k in ("q_proj", "k_proj", "v_proj", "o_proj")]
names += [base + f"sparse_indexer_{k}.weight" for k in "qkzw"]
names.append("model.layers.3.mlp.up_proj.weight")
layers = [FakeLayer(FakeAttention(())) for _ in range(3)] + [FakeLayer(attention)]
seen = {name: per_layer[3] for name, per_layer in run(layers, names)}
assert "quantize" not in seen[base + "o_proj.weight"]
assert "quantize" not in seen[base + "sparse_indexer_z.weight"]
assert seen["model.layers.3.mlp.up_proj.weight"][-1] == "quantize"
""")


def test_last_tensor_of_the_stream_still_completes_its_layer():
    _run(r"""
attention = FakeAttention({"q_proj", "k_proj", "v_proj", "o_proj"})
names = [f"model.layers.0.self_attn.{k}.weight" for k in ("q_proj", "k_proj", "v_proj", "o_proj")]
run([FakeLayer(attention)], names)
assert attention.events.count("quantize") == 1
""")


def test_layers_outside_the_model_are_passed_through():
    _run(r"""
attention = FakeAttention({"q_proj"})
log = run(
    [FakeLayer(attention)],
    ["model.layers.92.self_attn.q_proj.weight", "lm_head.weight"],
)
assert [name for name, _ in log] == [
    "model.layers.92.self_attn.q_proj.weight",
    "lm_head.weight",
]
assert attention.events == []
""")
