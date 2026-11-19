# SPDX-License-Identifier: Apache-2.0
"""Opt-in CT INT4/INT8 expert compute; original BF16 AgRs owns all transport.

Checkpoint CT INT4 scales describe signed values[-8,7]. The contiguous HIPC
Marlin kernel consumes their high-nibble INT8 domain[-128,112], so the derived
kernel scales are checkpoint_scale/16 exactly once. Disk checkpoints and the
canonical scale Parameters remain unchanged. HIPC weights replace their own
GPU storage in-place; no full canonical GPU copy is retained.
"""

from functools import lru_cache
import importlib
import threading
import weakref

import torch
import triton

from vllm.logger import init_logger
from .ct_w4a8_step5_native_policy import (
    NATIVE_TOKEN_CHUNK, padded_route_rows, token_chunk_ranges, workspace_bytes,
)

logger = init_logger(__name__)
_PACKED_OWNERS = weakref.WeakValueDictionary()
_WORKSPACES = {}
_WORKSPACE_LOCK = threading.RLock()


def _weight_key(weight):
    return (weight.device.type, weight.device.index,
            weight.untyped_storage().data_ptr(),
            weight.untyped_storage().nbytes(), tuple(weight.shape))


def _signature(layer):
    return (getattr(layer, "_step5_ct_load_generation", 0), tuple(
        (id(t), t.data_ptr(), _tracked_version(t), tuple(t.shape), t.dtype)
        for t in (layer.w13_weight, layer.w2_weight,
                  layer.w13_weight_scale, layer.w2_weight_scale)))


def _tracked_version(tensor):
    try:
        return tensor._version
    except RuntimeError as error:
        # Never fabricate a version for InferenceTensor, silently relax the
        # immutability guard, or make a second full GPU copy to convert it.
        raise RuntimeError(
            "Native CT W4A8 requires versioned weight/scale Parameters created "
            "outside torch.inference_mode(); the loader must preserve that "
            "contract. No in-place packing has been performed."
        ) from error


def _no_reload(*args, **kwargs):
    raise RuntimeError("Native CT W4A8 is packed in-place; restart from checkpoint for reload/EPLB")


@lru_cache(maxsize=1)
def _require_kernel():
    module = importlib.import_module("deepgemm")
    kernel = getattr(module, "m_grouped_w4a8_gemm_nt_contiguous_hipc", None)
    pack = getattr(module, "pack_w4a8_moe_hipc_weight", None)
    if not callable(kernel) or not callable(pack):
        raise RuntimeError("Native CT W4A8 requires contiguous W4A8 HIPC GEMM and single-owner pack APIs")
    version = importlib.import_module("deepgemm.version")
    # Legacy V5 wheels do not publish these two fields. Their ELF/launch
    # validation belongs to the predeployment GPU unit receipt; absence is
    # not proof that the existing package lacks the operation.
    co = getattr(version, "build_code_object_version", None)
    target = getattr(version, "build_target_arch", None)
    if co is not None and co != 5:
        raise RuntimeError("Native CT W4A8 requires Code Object V5")
    if target is not None and target != "gfx936":
        raise RuntimeError(f"Native BMZ experiment requires a gfx936 wheel, got {target}")
    return kernel, pack


def prepare_native_w4a8(layer):
    """Post-load conversion. Repeated calls must not repack an HIPC owner."""
    before_signature = _signature(layer)  # Reject untracked tensors before mutation.
    old = getattr(layer, "_step5_native_w4a8_signature", None)
    if old is not None:
        if old != before_signature:
            raise RuntimeError("Native CT W4A8 parameters changed after packing")
        return layer._step5_native_w4a8_info
    weights = (layer.w13_weight, layer.w2_weight)
    scales = (layer.w13_weight_scale, layer.w2_weight_scale)
    w1, w2 = weights
    if (w1.dtype != torch.int8 or w2.dtype != torch.int8
            or w1.ndim != 3 or w2.ndim != 3 or w1.shape[0] != w2.shape[0]
            or w1.shape[1] != w2.shape[2] * 4 or w2.shape[1] != w1.shape[2] * 2):
        raise ValueError("Native CT W4A8 requires canonical[E,N,K/2] packed expert matrices")
    if w1.device.type != "cuda" or w1.device != w2.device:
        raise ValueError("Native CT W4A8 requires colocated GPU expert weights")
    if w2.shape[1] % 512 or w2.shape[2] * 2 != 1536:
        raise ValueError("This fixed native kernel experiment requires Step5 intermediate1536 and hidden%512=0")
    for weight, scale in zip(weights, scales):
        if (scale.dtype != torch.float32 or scale.ndim not in (2, 3)
                or tuple(scale.shape[:2]) != tuple(weight.shape[:2])
                or (scale.ndim == 3 and scale.shape[2] != 1)
                or scale.device != weight.device):
            raise ValueError("Native CT W4A8 requires original FP32 per-output-channel scales")
        if (not weight.is_contiguous() or weight.storage_offset()
                or weight.untyped_storage().nbytes() != weight.numel()):
            raise ValueError("Native pack requires a complete contiguous single-owner INT8 storage")
    architecture = torch.cuda.get_device_properties(w1.device).gcnArchName.split(':', 1)[0]
    if architecture != "gfx936":
        raise RuntimeError(f"BMZ native experiment requires gfx936, got {architecture}")
    _, pack = _require_kernel()
    with torch.no_grad():
        # Tiny derived scales only; no expanded weight representation.
        layer._step5_native_w13_scale = (scales[0] / 16.0).contiguous()
        layer._step5_native_w2_scale = (scales[1] / 16.0).contiguous()
        for weight in weights:
            key = _weight_key(weight)
            owner = _PACKED_OWNERS.get(key)
            if owner is None:
                before = _weight_key(weight)
                packed = pack(weight)
                if _weight_key(packed) != before or packed.data_ptr() != weight.data_ptr():
                    raise RuntimeError("Native pack allocated a second full GPU weight owner")
                _PACKED_OWNERS[key] = weight
            weight.requires_grad_(False)
            if callable(getattr(weight, "weight_loader", None)):
                weight.weight_loader = _no_reload
    layer._step5_native_w4a8_info = {
        "backend": "contiguous W4A8 HIPC / Marlin", "MODE": 1002,
        "activation": "BF16 SwiGLUStep7 -> token INT8",
        "weight_scale": "original CT scale/16, once",
        "weight_storage": "single-owner in-place HIPC N32",
        "dispatch_combine": "original MoERunner BF16 AgRs, unchanged",
        "alignment": 64, "architecture": architecture,
        "internal_token_chunk": NATIVE_TOKEN_CHUNK,
    }
    layer._step5_native_w4a8_signature = _signature(layer)
    logger.info_once("Step5 CT native W4A8: fixed MODE1002, gfx936, single-owner pack, original scale/16; outer AgRs unchanged")
    return layer._step5_native_w4a8_info


class _Workspace:
    def __init__(self):
        self.integer = None
        self.bf16 = None
        self.lock = threading.RLock()

    def reserve(self, rows, hidden, intermediate, device):
        sizes = workspace_bytes(rows, hidden, intermediate)
        if self.integer is None or self.integer.numel() < sizes["INT8"]:
            self.integer = torch.empty(sizes["INT8"], dtype=torch.int8, device=device)
        if self.bf16 is None or self.bf16.numel() * 2 < sizes["BF16"]:
            self.bf16 = torch.empty(sizes["BF16"] // 2, dtype=torch.bfloat16, device=device)


def _workspace(x):
    # A shared pool across local model layers, not one multi-GiB allocation
    # per layer. Distinct streams get distinct storage; the enqueue lock also
    # prevents two host threads interleaving operations on the same stream.
    stream = torch.cuda.current_stream(x.device).cuda_stream
    key = (x.device.index, stream)
    with _WORKSPACE_LOCK:
        return _WORKSPACES.setdefault(key, _Workspace())


def native_workspace_inventory():
    with _WORKSPACE_LOCK:
        entries = list(_WORKSPACES.items())
    result = []
    for key, value in entries:
        with value.lock:
            result.append({"device": key[0], "stream": key[1],
                "INT8_bytes": 0 if value.integer is None else value.integer.numel(),
                "BF16_bytes": 0 if value.bf16 is None else value.bf16.numel() * 2})
    return result


def release_native_workspaces(device=None):
    """Drop this current HIP stream's disposable expert scratch owners.

    Called at the real logits/head phase, after expert work has been enqueued.
    The fixed-chunk native path uses each pool only on its allocating stream;
    dropping its last Python owners permits ordered caching-allocator reuse on
    that same stream without a device sync or empty_cache. Returned expert
    outputs own independent storage. Other streams, weights and scales stay
    untouched. This lifecycle is validated for the existing eager deployment.
    """
    device = (torch.device("cuda", torch.cuda.current_device()) if device is None
              else torch.device(device))
    if device.type != "cuda":
        raise ValueError("Native W4A8 scratch belongs to a CUDA/HIP device")
    index = torch.cuda.current_device() if device.index is None else device.index
    stream = torch.cuda.current_stream(device).cuda_stream
    with _WORKSPACE_LOCK:
        workspace = _WORKSPACES.get((index, stream))
    if workspace is None:
        return 0
    # Keep the pool and its lock in the registry: an apply which already took
    # this pool reference must share the same lock on its next reserve. Never
    # hold the registry lock while waiting for an enqueue owner to finish.
    with workspace.lock:
        released = (0 if workspace.integer is None else workspace.integer.numel())
        released += (0 if workspace.bf16 is None else workspace.bf16.numel() * 2)
        workspace.integer = None
        workspace.bf16 = None
    return released


def apply_native_w4a8(layer, x, topk_weights, topk_ids):
    """Expert output only. Shared add and all communication stay with caller."""
    if getattr(layer, "_step5_native_w4a8_signature", None) != _signature(layer):
        raise RuntimeError("Native CT W4A8 was not prepared, or its packed parameters changed")
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != layer.w2_weight.shape[1]:
        raise ValueError("Native CT W4A8 requires BF16[M,H] input")
    if (topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape
            or topk_ids.shape[0] != x.shape[0] or topk_ids.shape[1] != 8):
        raise ValueError("Native CT W4A8 requires matching top8 routes per token")
    if getattr(layer, "apply_router_weight_on_input", False):
        raise ValueError("Native CT W4A8 requires output-side FP32 routing weights")
    if x.shape[0] == 0:
        return torch.empty_like(x)
    mapping = getattr(layer, "_expert_map", None)
    global_experts = getattr(layer, "global_num_experts", layer.w13_weight.shape[0])
    if (mapping is None or mapping.ndim != 1 or mapping.numel() != global_experts
            or mapping.dtype != torch.int32 or mapping.device != x.device):
        raise ValueError("Native CT W4A8 requires the existing global-to-local INT32 expert map")
    if global_experts != 352 or layer.w13_weight.shape[0] != 44:
        raise ValueError("This experiment requires352 global experts and44 local experts")
    if not (x.device == topk_ids.device == topk_weights.device == layer.w13_weight.device):
        raise ValueError("Native CT W4A8 tensors must be colocated")
    from boltops.fused_moe.triton.moe_compat import per_token_quant_hip
    from .ct_w4a8_step5_native_routing import permute64
    from .ct_w4a8_step5_native_kernels import activation_persistent, _weighted_output

    kernel, _ = _require_kernel()
    hidden, intermediate = x.shape[1], layer.w2_weight.shape[2] * 2
    workspace = _workspace(x)
    # One owning output allocation, regardless of the number of internal
    # slices. Each slice writes directly to its disjoint view. Row-contiguous
    # output also keeps the pointer-only reduction correct for strided x.
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    with workspace.lock:
        capacity = min(x.shape[0], NATIVE_TOKEN_CHUNK)
        workspace.reserve(padded_route_rows(capacity, 8, 44),
                          hidden, intermediate, x.device)
        for start, end in token_chunk_ranges(x.shape[0]):
            current = x[start:end]
            ids = topk_ids[start:end].to(torch.int32).contiguous()
            weights = topk_weights[start:end].to(torch.float32).contiguous()
            rows = padded_route_rows(end - start, 8, 44)
            quantized, input_scale = per_token_quant_hip(current.contiguous())
            gathered = workspace.integer[:rows * hidden].view(rows, hidden)
            aq, scales, indices, inverse, _ = permute64(
                quantized, input_scale.contiguous(), ids,
                mapping.contiguous(), 44, gathered)
            gateup = workspace.bf16[:rows * 2 * intermediate].view(rows, 2 * intermediate)
            kernel((aq, scales), (layer.w13_weight, layer._step5_native_w13_scale),
                   gateup, indices, config={"MODE": 1002})
            active = workspace.integer[:rows * intermediate].view(rows, intermediate)
            # Identical implementation for every chunk and every MTP tail.
            activation_persistent[(min(triton.cdiv(rows, 8), 1024),)](
                gateup, active, scales, indices, rows, intermediate, 8, 512,
                num_warps=4, enable_fp_fusion=False)
            down = workspace.bf16[:rows * hidden].view(rows, hidden)
            kernel((active, scales), (layer.w2_weight, layer._step5_native_w2_scale),
                   down, indices, config={"MODE": 1002})
            target = output[start:end]
            _weighted_output[(end - start, triton.cdiv(hidden, 2048))](
                down, inverse, weights, mapping, ids, target, target,
                end - start, hidden, 8, global_experts, True, False, 0, 0, 2048,
                num_warps=4, enable_fp_fusion=False)
            # Drop the previous slice's temporary owners before allocating
            # the next quantized input/metadata. The current stream owns the
            # queued use, so allocator reuse preserves HIP enqueue ordering.
            del current, ids, weights, quantized, input_scale, gathered
            del aq, scales, indices, inverse, gateup, active, down, target
    return output
