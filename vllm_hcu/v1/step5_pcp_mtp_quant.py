# SPDX-License-Identifier: Apache-2.0
"""Isolated CT aliases for the three unfused Step5 PCP MTP INT8 linears.

Checkpoint targets name ``model.layers.N.self_attn/MLP`` while the MTP
module's parameter namespace inserts ``mtp_block``. The checkpoint already
has aliases for packed QKV/gate-up and its shared heads. Add only the missing
g_proj, o_proj and down_proj aliases to a per-MTP-layer configuration copy.
The VL target model maps literal checkpoint rules under ``language_model``
before the MTP model is constructed; its draft prefixes remain ``model.*``.
"""
from __future__ import annotations

from copy import copy, deepcopy
from collections.abc import Mapping


_SUFFIXES = ("self_attn.g_proj", "self_attn.o_proj", "mlp.down_proj")


def _field(value, name):
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _is_channel_int8(scheme):
    weights = scheme.get("weights") if isinstance(scheme, Mapping) else None
    kind = _field(weights, "type")
    strategy = _field(weights, "strategy")
    return (
        _field(weights, "num_bits") == 8
        and getattr(kind, "value", kind) == "int"
        and getattr(strategy, "value", strategy) == "channel"
    )


def _canonical_prefixes(prefix):
    # Step5ForConditionalGeneration inherits Step3p7's exact model. ->
    # language_model.model. mapper. CT applies that mapper in place to literal
    # targets AND ignores, while regex targets are deliberately unchanged.
    # Accept both observed namespaces without remapping the shared CT object or
    # changing any checkpoint/internal module prefix.
    if prefix.startswith("model.layers."):
        return (prefix, "language_model." + prefix)
    if prefix.startswith("language_model.model.layers."):
        return (prefix, prefix.removeprefix("language_model."))
    return (prefix,)


def step5_pcp_mtp_quant_config(vllm_config, prefix):
    """Return a per-layer config copy; leave target and non-PCP paths intact."""
    from vllm_hcu.v1.step5_pcp_config import is_step5_pcp

    if not is_step5_pcp(vllm_config):
        return vllm_config
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer,
    )

    quant = vllm_config.quant_config
    if not isinstance(quant, CompressedTensorsConfig):
        raise TypeError("Step5 PCP MTP INT8 aliases require compressed-tensors")
    local_quant = copy(quant)
    # get_scheme_dict may fill a missing format in a scheme dict. Isolate
    # nested metadata too, so even that native behavior cannot mutate target
    # quantization state or another MTP layer's rule table.
    local_quant.target_scheme_map = deepcopy(quant.target_scheme_map)
    local_quant.ignore = list(quant.ignore)
    local_quant.packed_modules_mapping = deepcopy(quant.packed_modules_mapping)
    added = []
    for suffix in _SUFFIXES:
        canonicals = tuple(f"{name}.{suffix}" for name in _canonical_prefixes(prefix))
        runtime = f"{prefix}.mtp_block.{suffix}"
        if runtime in local_quant.target_scheme_map:
            continue
        sources = [(name, local_quant.target_scheme_map[name]) for name in canonicals
                   if name in local_quant.target_scheme_map]
        if not sources:
            continue
        scheme = sources[0][1]
        if any(other != scheme for _, other in sources[1:]):
            raise ValueError(f"Step5 PCP MTP conflicting canonical CT rules: {canonicals}")
        # Do not turn an explicitly BF16/ignored or non-INT8 checkpoint rule
        # into INT8. Generic rules and existing runtime aliases retain their
        # original priority because these exact aliases are appended last.
        if scheme is None or not _is_channel_int8(scheme):
            continue
        # Include the selected mapped source and both runtime namespaces. An
        # ignore that CT already mapped must still protect the draft alias.
        ignored_names = (*canonicals, *(f"{name}.mtp_block.{suffix}"
                                        for name in _canonical_prefixes(prefix)))
        if any(should_ignore_layer(name, ignore=local_quant.ignore,
                                   fused_mapping=local_quant.packed_modules_mapping)
               for name in ignored_names):
            continue
        local_quant.target_scheme_map[runtime] = deepcopy(scheme)
        added.append(runtime)
    local_config = copy(vllm_config)
    local_config.quant_config = local_quant
    local_config._step5_pcp_mtp_quant_aliases = tuple(added)
    return local_config


__all__ = ["step5_pcp_mtp_quant_config"]
