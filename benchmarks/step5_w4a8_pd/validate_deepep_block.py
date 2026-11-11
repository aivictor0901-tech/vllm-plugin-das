"""Real vLLM groups, checkpoint loaders, Step5MoE and automatic HT/LL dispatch.

Uses synthetic weights for one model block, never reads full checkpoint shards.
"""
import datetime,json,os,sys,traceback
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist

root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
rank=int(os.environ['LOCAL_RANK'])
tp=int(os.getenv('STEP5_TEST_TP','4'));pp=int(os.getenv('STEP5_TEST_PP','1'));dp=8//tp//pp
record=dict(rank=rank,tp=tp,dp=dp,pp=pp,passed=False,tests=[])
def save():(root/f'block-rank-{rank}.json').write_text(json.dumps(record,indent=2))
try:
    torch.set_num_threads(1);torch.cuda.set_device(rank);torch.set_default_dtype(torch.bfloat16)
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    from vllm.engine.arg_utils import EngineArgs
    from vllm.config import set_current_vllm_config
    from vllm.config.compilation import CUDAGraphMode
    from vllm_hcu.patch.worker import apply_worker_patches
    args=EngineArgs(model='/model/step-5-preview-channel-w4a8-GPTQ',
        quantization='compressed-tensors',tensor_parallel_size=tp,data_parallel_size=dp,
        pipeline_parallel_size=pp,enable_expert_parallel=True,
        distributed_executor_backend='mp' if pp>1 else 'external_launcher',
        max_model_len=1024,max_num_batched_tokens=128,max_num_seqs=16,
        disable_custom_all_reduce=True,all2all_backend='deepep_auto',moe_backend='deep_gemm',
        **(dict(data_parallel_rank=rank//(tp*pp)) if dp>1 else {}))
    cfg=args.create_engine_config()
    # The production Step3p7 multimodal wrapper constructs its text model
    # with this nested config. This probe instantiates the text block directly.
    if hasattr(cfg.model_config.hf_config,'text_config'):
        cfg.model_config.hf_config=cfg.model_config.hf_config.text_config
    with set_current_vllm_config(cfg):
        apply_worker_patches(cfg)
        from vllm.distributed import (init_distributed_environment,initialize_model_parallel,
            get_tp_group,get_dp_group,get_ep_group,destroy_model_parallel,destroy_distributed_environment)
        # torchrun already owns the global eight-rank rendezvous. Production
        # PP uses mp; avoid its additional per-DP world expansion in this
        # manually launched, single-block fixture.
        executor=cfg.parallel_config.distributed_executor_backend
        cfg.parallel_config.distributed_executor_backend='external_launcher'
        init_distributed_environment(world_size=8,rank=rank,local_rank=rank,
            distributed_init_method='env://',backend='nccl',timeout=datetime.timedelta(seconds=120))
        cfg.parallel_config.distributed_executor_backend=executor
        initialize_model_parallel(tensor_model_parallel_size=tp,pipeline_model_parallel_size=pp)
        from vllm.v1.worker.workspace import init_workspace_manager
        init_workspace_manager(torch.device('cuda',rank),1,1)
        from vllm_hcu.models.step5_text import FusedMoEBlock as Step5MoE,Step5DecoderLayer
        from vllm.forward_context import set_forward_context
        from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
        h,i=4096,1536
        for layer_id in [3,88]:
            with torch.device('cuda'):
                moe=Step5MoE(cfg,prefix=f'model.layers.{layer_id}.moe')
            layer=moe.routed_experts
            method=layer.quant_method
            record['tests'].append(dict(stage='constructed',layer=layer_id,
                method=type(method).__name__,ep=get_ep_group().world_size,
                local_experts=layer.w13_weight.shape[0],sp=moe.effective_sequence_parallel));save()
            # Structured sparse matrices, with real checkpoint storage types.
            g=torch.zeros((i,h),dtype=torch.int8,device='cpu')
            d=torch.zeros((h,i),dtype=torch.int8,device='cpu')
            ix=torch.arange(i,device='cpu');g[ix,ix]=1;d[ix,ix]=1
            first=moe.physical_expert_start;last=moe.physical_expert_end
            for gid in range(first,last):
                for projection,shard,w in [('gate_proj','w1',g),('up_proj','w3',g),('down_proj','w2',d)]:
                    name='w2_weight' if shard=='w2' else 'w13_weight'
                    p=getattr(layer,name)
                    tensor=pack_to_int32(w,4) if layer_id==3 else w.to(torch.bfloat16)
                    p.weight_loader(p,tensor,f'model.layers.{layer_id}.moe.experts.{gid}.{projection}.weight',
                                    shard_id=shard,expert_id=gid)
                    if layer_id==3:
                        scale=getattr(layer,name+'_scale')
                        value=torch.full((w.shape[0],1),1.,dtype=torch.float32,device='cpu')
                        scale.weight_loader(scale,value,f'model.layers.{layer_id}.moe.experts.{gid}.{projection}.weight_scale',
                                            shard_id=shard,expert_id=gid)
            method.process_weights_after_loading(layer)
            meta=layer._hcu_step5_deepep
            record['tests'].append(dict(stage='loaded_and_prepared',layer=layer_id,backend=meta));save()
            # Each TP rank owns different shared-expert weight columns.
            part=i//tp;tpr=get_tp_group().rank_in_group
            gu=torch.zeros((2*part,h),dtype=torch.bfloat16,device='cpu')
            down=torch.zeros((h,part),dtype=torch.bfloat16,device='cpu')
            local=torch.arange(part,device='cpu');global_col=local+tpr*part
            gu[local,global_col]=1;gu[local+part,global_col]=1;down[global_col,local]=1
            with torch.no_grad():
                param=moe.share_expert.gate_up_proj.weight
                param.weight_loader(param,g.to(torch.bfloat16),0)
                param.weight_loader(param,g.to(torch.bfloat16),1)
                param=moe.share_expert.down_proj.weight
                param.weight_loader(param,d.to(torch.bfloat16))
                moe.gate.weight.zero_()
                moe.router_bias.zero_()
                # One routed expert on each EP owner; the rest remain valid
                # loaded weights but are not selected in this fixture.
                moe.router_bias[torch.arange(get_ep_group().world_size,device='cuda')
                                *layer.w13_weight.shape[0]]=1
            ffn=SimpleNamespace(use_moe=True,moe=moe,_cast_for_residual=lambda x:x.float())
            gen=torch.Generator(device='cpu').manual_seed(538+get_dp_group().rank_in_group)
            x=(torch.randn((9,h),generator=gen,device='cpu',dtype=torch.float32)*.1).to(torch.bfloat16).cuda()
            def context(prefill,full=False):
                return set_forward_context(None,cfg,num_tokens=x.shape[0],
                    num_tokens_across_dp=torch.full((dp,),x.shape[0],dtype=torch.int64,device='cpu'),
                    batch_descriptor=SimpleNamespace(uniform=not prefill),
                    cudagraph_runtime_mode=CUDAGraphMode.FULL if full else CUDAGraphMode.NONE,
                    deepep_auto_is_prefilling=[prefill])
            def run():return Step5DecoderLayer._forward_ffn(ffn,x)
            def audited_run(prefill):
                events=[];restore=[]
                for label,group in [('tp',get_tp_group()),('dp',get_dp_group()),('ep',get_ep_group())]:
                    for name in ('all_reduce','all_gather','reduce_scatter',
                                 'all_gatherv','reduce_scatterv','dispatch_router_logits'):
                        if not hasattr(group,name):continue
                        original=getattr(group,name)
                        def wrapped(*a,_original=original,_name=label+'.'+name,**kw):
                            events.append(dict(op=_name));return _original(*a,**kw)
                        setattr(group,name,wrapped);restore.append((group,name,original))
                prepare=method.moe_kernel.prepare_finalize
                buffers={id(p.buffer):p.buffer for p in [prepare.ht_prepare_finalize,prepare.ll_prepare_finalize]}
                for buffer in buffers.values():
                    for name in ('dispatch','combine','low_latency_dispatch','low_latency_combine'):
                        original=getattr(buffer,name)
                        def wrapped(*a,_original=original,_name=name,**kw):
                            value=a[0] if a else kw.get('x')
                            if isinstance(value,tuple):value=value[0]
                            events.append(dict(op=_name,rows=value.shape[0]))
                            return _original(*a,**kw)
                        setattr(buffer,name,wrapped);restore.append((buffer,name,original))
                try:
                    with context(prefill):result=run().clone()
                finally:
                    for obj,name,original in reversed(restore):setattr(obj,name,original)
                dispatch='dispatch' if prefill else 'low_latency_dispatch'
                combine='combine' if prefill else 'low_latency_combine'
                assert sum(e['op']==dispatch for e in events)==1,events
                assert sum(e['op']==combine for e in events)==1,events
                assert not any(e['op'].startswith('dp.') or e['op'].endswith('all_reduce')
                               or e['op'].endswith('dispatch_router_logits') for e in events),events
                assert next(e['rows'] for e in events if e['op']==dispatch)==(x.shape[0]+tp-1)//tp
                record['tests'].append(dict(stage='communication_audit',layer=layer_id,
                    phase='HT' if prefill else 'LL',events=events));save()
                return result
            def compare(a,b,limit=.025):
                aa,bb=a.cpu().float(),b.cpu().float();delta=aa-bb
                nrmse=(delta.square().sum()/bb.square().sum().clamp_min(1e-30)).sqrt().item()
                assert torch.isfinite(aa).all() and nrmse<limit,(nrmse,delta.abs().max().item())
                return dict(nrmse=nrmse,max_abs=delta.abs().max().item())
            dist.barrier()
            ht=audited_run(True)
            torch.cuda.synchronize()
            ll=audited_run(False)
            torch.cuda.synchronize()
            record['tests'].append(dict(stage='real_block_HT_LL',layer=layer_id,**compare(ll,ht)));save()
            # Every loaded routed expert is the same identity projection.
            # Check against a separate formula to catch duplicated TP sums,
            # dropped routes, or two equally incorrect HT/LL implementations.
            ref=torch.zeros_like(x,device='cpu',dtype=torch.float32)
            xc=x.cpu().float()[:,:i]
            def qdq(value):
                bound=127 if layer_id==3 else 448
                qs=(value.abs().amax(-1,keepdim=True)/bound).clamp_min(1e-12)
                q=value/qs
                q=q.round().clamp(-127,127) if layer_id==3 else q.to(torch.float8_e4m3fn).float()
                return q*qs
            gate=qdq(x.cpu().float())[:,:i].bfloat16().float()
            act=(torch.clamp(gate*torch.sigmoid(gate),max=7)*torch.clamp(gate,-7,7)).bfloat16().float()
            routed_ref=qdq(act).bfloat16().float()
            shared_ref=(xc*torch.sigmoid(xc)*xc).bfloat16().float()
            scale=moe.routed_scaling_factor*(1 if cfg.model_config.hf_config.norm_expert_weight else .5*cfg.model_config.hf_config.moe_top_k)
            ref[:,:i]=shared_ref+routed_ref*scale
            record['tests'].append(dict(stage='independent_routed_and_shared_reference',layer=layer_id,
                **compare(ht,ref,limit=.025)));save()
            graph=torch.cuda.CUDAGraph();dist.barrier()
            with context(False,True):
                with torch.cuda.graph(graph):out=run()
                graph.replay()
            torch.cuda.synchronize()
            record['tests'].append(dict(stage='real_block_LL_graph',layer=layer_id,**compare(out,ll)));save()
            # Zero stays zero in either packed layout. Isolate shared TP
            # reduction and token restoration against an independent formula.
            with torch.no_grad():layer.w13_weight.zero_();layer.w2_weight.zero_()
            with context(True):shared=run().clone()
            ref=torch.zeros_like(x,device='cpu',dtype=torch.float32)
            xc=x.cpu().float()[:,:i]
            ref[:,:i]=(xc*torch.sigmoid(xc)*xc).to(torch.bfloat16).float()
            record['tests'].append(dict(stage='shared_TP_and_token_restore',layer=layer_id,
                **compare(shared,ref,limit=.02)));save()
            del graph,moe,layer,method,ffn
            torch.cuda.empty_cache()
        record['passed']=True;save();dist.barrier()
        destroy_model_parallel();destroy_distributed_environment()
    print(json.dumps(record),flush=True)
except BaseException as exc:
    record.update(error=repr(exc),traceback=traceback.format_exc());save()
    print(json.dumps(record),flush=True);sys.stdout.flush();sys.stderr.flush();os._exit(1)
