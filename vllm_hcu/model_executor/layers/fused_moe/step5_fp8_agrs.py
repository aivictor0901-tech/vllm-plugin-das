# SPDX-License-Identifier: Apache-2.0
"""DeepGEMM FP8 experts behind the unchanged Step5 AG/RS prepare/finalize."""
import math
import os
import torch
import triton
import triton.language as tl
import vllm.model_executor.layers.fused_moe  # initialize the public package first
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from .step5_deepep import Step5FP8HTExperts, gemm_op
from .step5_fp8_gemm import pack_decode_weight
from .step5_w4a8_kernels import _small_route_gather, _weighted_output, activation_masked


@triton.jit
def _masked_route_meta(ids, mapping, inverse, counts, ROUTES: tl.constexpr,
                       GLOBAL_E: tl.constexpr, E: tl.constexpr,
                       CAP: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    r = tl.arange(0, BLOCK)
    gid = tl.load(ids + r, mask=r < ROUTES, other=-1)
    local = tl.load(mapping + gid,
        mask=(r < ROUTES) & (gid >= 0) & (gid < GLOBAL_E), other=-1)
    mine = (r < ROUTES) & (local == expert)
    position = expert * CAP + tl.cumsum(mine.to(tl.int32), 0) - 1
    tl.store(inverse + r, position, mask=mine)
    if expert == 0:
        tl.store(inverse + r, -1, mask=(r < ROUTES) & ((local < 0) | (local >= E)))
    tl.store(counts + expert, tl.sum(mine.to(tl.int32), 0))


class Step5FP8AGRSExperts(Step5FP8HTExperts):
    """Use contiguous1000 for P; one fixed masked mode for both D GEMMs."""
    CHUNK = 256

    def __init__(self, moe_config, quant_config):
        super().__init__(moe_config, quant_config)
        phase = os.environ.get("VLLM_HCU_STEP5_W4A8_PROFILE")
        if phase not in ("prefill", "decode"):
            raise ValueError("Step5 FP8 AG/RS requires a prefill or decode profile")
        self.decode = phase == "decode"
        if self.decode:
            self.WEIGHT_LAYOUT = "step5_fp8_decode"

    def _pack_channel_weights(self, w13, w2):
        if self.decode:
            return pack_decode_weight(w13), pack_decode_weight(w2)
        return super()._pack_channel_weights(w13, w2)

    def workspace_shapes(self, M, N, K, topk, global_num_experts,
                         local_num_experts, expert_tokens_meta, activation):
        if not self.decode:
            return super().workspace_shapes(M,N,K,topk,global_num_experts,
                local_num_experts,expert_tokens_meta,activation)
        # Bound by a fixed token tile, including pathological duplicate routes.
        cap = triton.cdiv(min(M, self.CHUNK) * topk, 64) * 64
        rows = local_num_experts * cap
        return ((rows, max(N,K)), (rows, max(N,K)), (M,K))

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
              activation, global_num_experts, expert_map, a1q_scale, a2_scale,
              workspace13, workspace2, expert_tokens_meta, apply_router_weight_on_input):
        if not self.decode:
            return super().apply(output,hidden_states,w1,w2,topk_weights,topk_ids,
                activation,global_num_experts,expert_map,a1q_scale,a2_scale,
                workspace13,workspace2,expert_tokens_meta,apply_router_weight_on_input)
        if activation != MoEActivation.SWIGLUSTEP or apply_router_weight_on_input:
            raise ValueError("Step5 FP8 requires output-side weights and SwigluStep7")
        if hidden_states.dtype != torch.float8_e4m3fn or a1q_scale is None:
            raise ValueError("Step5 FP8 AG/RS expects FP8 activations and token scales")
        e,m,n,k,topk = self.moe_problem_size(hidden_states,w1,w2,topk_ids)
        if not m:
            return
        if expert_map is None:
            if e != global_num_experts:
                raise ValueError("Partial expert ownership requires an expert map")
            expert_map = torch.arange(e,device=hidden_states.device,dtype=torch.int32)
        cap = triton.cdiv(min(m,self.CHUNK) * topk,64) * 64
        aq = _resize_cache(workspace2.view(torch.float8_e4m3fn),(e,cap,k))
        scales = torch.empty((e,cap,1),device=aq.device,dtype=torch.float32)
        counts = torch.empty((e,),device=aq.device,dtype=torch.int32)
        gateup = _resize_cache(workspace13,(e,cap,n))
        # Keep final output independent of the modular allocator's aliases.
        # This also makes long real-prefill chunks in a D service safe.
        result = torch.empty_like(output)
        down = torch.empty((e,cap,k),device=aq.device,dtype=torch.bfloat16)
        op = gemm_op()
        for start in range(0,m,self.CHUNK):
            end = min(m,start+self.CHUNK)
            ids = topk_ids[start:end].contiguous()
            inverse = torch.empty_like(ids)
            _masked_route_meta[(e,)](ids,expert_map,inverse,counts,ids.numel(),
                global_num_experts,e,cap,triton.next_power_of_2(ids.numel()),num_warps=8)
            # Copy FP8 bytes without asking Triton to cast an integer mask
            # fill to gfx938's fp8e4nv type. Values/scales remain unchanged.
            _small_route_gather[(ids.numel(),)](hidden_states[start:end].view(torch.uint8),a1q_scale[start:end],
                inverse,aq.view(torch.uint8),scales,k,topk,a1q_scale.stride(0),triton.next_power_of_2(k),num_warps=4)
            expected = max(1,math.ceil((end-start)*topk/global_num_experts))
            op(aq,self._deepgemm_w13,gateup,scales,self.w1_scale.contiguous(),
                counts,expected,False,True)
            q = _resize_cache(workspace2.view(torch.float8_e4m3fn),(e,cap,n//2))
            activation_masked[(e,8)](gateup,q,scales,counts,cap,n//2,False,
                triton.next_power_of_2(n//2),num_warps=4)
            op(q,self._deepgemm_w2,down,scales,self.w2_scale.contiguous(),
                counts,expected,False,True)
            _weighted_output[(end-start,triton.cdiv(k,256))](down,inverse,
                topk_weights[start:end],expert_map,ids,result[start:end],result,
                end-start,k,topk,global_num_experts,True,False,0,0,256,num_warps=4)
        output.copy_(result)


def prepare_step5_fp8_agrs(method, layer):
    from vllm.model_executor.layers.fused_moe.all2all_utils import maybe_make_prepare_finalize
    quant = method.get_fused_moe_quant_config(layer)
    prepare = maybe_make_prepare_finalize(moe=method.moe,quant_config=quant,
        routing_tables=layer._expert_routing_tables(),allow_new_interface=True,use_monolithic=False)
    experts = Step5FP8AGRSExperts(method.moe,quant)
    experts.process_weights_after_loading(layer)
    method.moe_quant_config = quant
    method.moe_kernel = mk.FusedMoEKernel(prepare,experts)
