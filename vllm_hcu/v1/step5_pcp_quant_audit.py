"""Bounded read-only Step5 PCP model/quantization metadata for worker RPC.

This module does not patch/register anything. The caller passes an already
loaded runner. No tensor contents, scalar values, device copies, kernels,
synchronization, SMI calls or distributed collectives are used.
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Mapping
from typing import Any

_MISSING = object()
_SUFFIXES = ("self_attn.g_proj", "self_attn.o_proj", "mlp.down_proj")
_MAX_MODULES = 16384
_MAX_DIRECT = 16
_MAX_JSON_CHARS = 60000


def _member(obj: Any, name: str, default: Any = None) -> Any:
    """Read stored attributes/Module registries without invoking properties."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    try:
        state = object.__getattribute__(obj, "__dict__")
    except (AttributeError, TypeError):
        return default
    if name == "__dict__":
        return state
    if name in state:
        return state[name]
    for bucket in ("_modules", "_parameters", "_buffers"):
        values = state.get(bucket)
        if isinstance(values, Mapping) and name in values:
            return values[name]
    return default


def _path(obj: Any, path: str) -> Any:
    for name in path.split("."):
        obj = _member(obj, name)
        if obj is None:
            break
    return obj


def _type_name(obj: Any) -> str | None:
    if obj is None:
        return None
    cls = type(obj)
    return (cls.__module__ + "." + cls.__qualname__)[:192]


def _tensor_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {"present": False}
    result: dict[str, Any] = {"present": True, "python_type": _type_name(value)}
    try:
        # Tensor shape/dtype/device are metadata; never touch .data or values.
        shape = tuple(value.shape)
        if len(shape) > 8 or any(type(dim) is not int for dim in shape):
            return {**result, "metadata_error": "non_static_or_excessive_shape"}
        result.update(dtype=str(value.dtype)[:96], shape=list(shape),
                      device=str(value.device)[:48])
    except (AttributeError, TypeError, RuntimeError, ValueError) as error:
        result["metadata_error"] = type(error).__name__
    return result


def _direct_metadata(module: Any, bucket: str) -> dict[str, Any]:
    values = _member(module, bucket, {})
    if not isinstance(values, Mapping):
        return {"count": 0, "entries": {}, "unavailable": True}
    names = [name for name in values if isinstance(name, str)]
    def priority(name: str) -> tuple[int, str]:
        return ((0 if name == "weight" else 1 if name == "weight_scale" else
                 2 if name == "weight_scale_inv" else 3 if "scale" in name else 4), name)
    selected = sorted(names, key=priority)[:_MAX_DIRECT]
    return {"count": len(names), "truncated": len(names) > len(selected),
            "entries": {name[:128]: _tensor_metadata(values[name]) for name in selected}}


def _linear_metadata(name: str, module: Any) -> dict[str, Any]:
    quant = _member(module, "quant_method")
    params = _direct_metadata(module, "_parameters")
    buffers = _direct_metadata(module, "_buffers")
    # Some post-load scale caches are plain attributes, not registered buffers.
    state = _member(module, "__dict__", {})
    if not isinstance(state, dict):
        try:
            state = object.__getattribute__(module, "__dict__")
        except (AttributeError, TypeError):
            state = {}
    registered = set(params["entries"]) | set(buffers["entries"])
    extra_names = sorted(key for key in state if isinstance(key, str)
                         and "scale" in key.lower() and key not in registered)
    scale = _member(module, "weight_scale", _MISSING)
    return {"name": name[:256], "module_type": _type_name(module),
            "quant_method_type": _type_name(quant),
            "module_scheme_type": _type_name(_member(module, "scheme")),
            "quant_method_scheme_type": _type_name(_member(quant, "scheme")),
            "quant_method_private_scheme_type": _type_name(_member(quant, "_scheme")),
            "weight_scale_attribute_present": scale is not _MISSING and scale is not None,
            "parameters": params, "buffers": buffers,
            "scale_attributes": {key[:128]: _tensor_metadata(state[key])
                                 for key in extra_names[:_MAX_DIRECT]},
            "scale_attributes_truncated": len(extra_names) > _MAX_DIRECT}


def _matching_linears(model: Any) -> tuple[list[tuple[str, Any]], dict[str, Any]]:
    """Walk the stored nn.Module tree, equivalent to named_modules paths.

    Do not invoke model-defined traversal/forward/property implementations.
    """
    pending = [("", model)]
    seen: set[int] = set()
    matches = []
    while pending and len(seen) < _MAX_MODULES:
        name, module = pending.pop()
        if module is None or id(module) in seen:
            continue
        seen.add(id(module))
        if any(name == suffix or name.endswith("." + suffix) for suffix in _SUFFIXES):
            matches.append((name, module))
        children = _member(module, "_modules", {})
        if isinstance(children, Mapping):
            pending.extend((name + "." + key if name else key, child)
                           for key, child in reversed(list(children.items()))
                           if isinstance(key, str))
    return matches, {"visited_modules": len(seen), "walk_truncated": bool(pending)}


def _layer_id(name: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.([0-9]+)(?:\.|$)", name)
    return int(match.group(1)) if match else None


def _parallel_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        state = importlib.import_module("vllm.distributed.parallel_state")
    except Exception as error:
        return {"error_type": type(error).__name__, "distributed_rank": None}
    for key, getter in (("world", "get_world_group"), ("TP", "get_tp_group"),
                        ("PCP", "get_pcp_group"), ("PP", "get_pp_group"),
                        ("EP", "get_ep_group")):
        try:
            group = getattr(state, getter)()
            ranks = _member(group, "ranks")
            plain_ranks = (isinstance(ranks, (list, tuple))
                           and all(type(rank) is int for rank in ranks))
            record = {name: _member(group, name) if type(_member(group, name)) is int else None
                      for name in ("rank", "rank_in_group", "world_size")}
            record.update(ranks=list(ranks[:256]) if plain_ranks else None,
                          ranks_truncated=bool(plain_ranks and len(ranks) > 256),
                          source="existing_group_metadata_no_collective")
            result[key] = record
        except Exception as error:
            result[key] = {"error_type": type(error).__name__}
    world = result.get("world", {})
    result["distributed_rank"] = world.get("rank")
    result["distributed_world_size"] = world.get("world_size")
    return result


def audit_step5_pcp_quantization(runner: Any) -> dict[str, Any]:
    """Return <=60k JSON characters of rank/model/quantization metadata.

    PP stages without a speculator are reported as absent, not fabricated.
    Scale existence/dtype/shape does not prove that checkpoint values loaded.
    """
    target = _member(runner, "model", _MISSING)
    target_path = "runner.model"
    if target is _MISSING or target is None:
        target = _path(runner, "model_state.model")
        target_path = "runner.model_state.model"
    speculator = _member(runner, "speculator")
    draft = _member(speculator, "model")
    result: dict[str, Any] = {"status": "READ_ONLY_METADATA", "schema_version": 1,
        "runner_type": _type_name(runner), "parallel": _parallel_metadata(),
        "target": {"exists": target is not None, "path": target_path,
                   "type": _type_name(target)},
        "speculator": {"exists": speculator is not None, "type": _type_name(speculator)},
        "speculator_model": {"exists": draft is not None, "path": "runner.speculator.model",
                             "type": _type_name(draft)},
        "scope": "Metadata only; no tensor values, transfers, kernels, synchronization or collectives"}
    target_matches, walk = _matching_linears(target)
    layers = [_layer_id(name) for name, _ in target_matches]
    first = min((index for index in layers if index is not None), default=None)
    selected = [(name, module) for name, module in target_matches if _layer_id(name) == first][:3]
    result["target"].update(walk, first_local_layer_id=first,
                            linears=[_linear_metadata(name, module) for name, module in selected])
    draft_matches, walk = _matching_linears(draft)
    draft_matches.sort(key=lambda pair: (_layer_id(pair[0]) if _layer_id(pair[0]) is not None else -1, pair[0]))
    chosen = draft_matches[:9]
    layer_ids = sorted({_layer_id(name) for name, _ in chosen if _layer_id(name) is not None})
    coverage = {str(index): [suffix for suffix in _SUFFIXES
                            if any(_layer_id(name) == index and name.endswith(suffix) for name, _ in chosen)]
                for index in layer_ids}
    result["speculator_model"].update(walk, expected_linear_count=9,
        matching_linear_count=len(draft_matches), linears_truncated=len(draft_matches) > 9,
        observed_layer_ids=layer_ids, layer_suffix_coverage=coverage,
        complete_three_layers_nine_linears=(len(layer_ids) == 3 and len(draft_matches) == 9
            and all(len(suffixes) == 3 for suffixes in coverage.values()) and not walk["walk_truncated"]),
        linears=[_linear_metadata(name, module) for name, module in chosen])
    encoded = json.dumps(result, ensure_ascii=True, separators=(",", ":"))
    if len(encoded) > _MAX_JSON_CHARS:
        return {"status": "METADATA_LIMIT_EXCEEDED", "schema_version": 1,
                "parallel": result["parallel"], "limit_characters": _MAX_JSON_CHARS,
                "observed_characters": len(encoded), "quantization_proof_available": False}
    return result
