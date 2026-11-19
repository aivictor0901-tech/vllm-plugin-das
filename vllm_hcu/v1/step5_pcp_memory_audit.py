"""Bounded metadata and allocator counters from the actual Step5 worker."""
from collections import Counter
import os


def storage_inventory(named_tensors):
    """Deduplicate simultaneous owners, without reading tensor values."""
    owners, logical_bytes, entries = {}, 0, 0
    for name, tensor in named_tensors:
        entries += 1
        logical = int(tensor.numel()) * int(tensor.element_size())
        logical_bytes += logical
        storage = tensor.untyped_storage()
        key = (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))
        if key not in owners:
            owners[key] = {'bytes':key[2], 'device':key[0], 'dtype':str(tensor.dtype),
                           'shape':list(tensor.shape), 'alias_count':0, 'names':[]}
        row = owners[key]
        row['alias_count'] += 1
        if len(row['names']) < 4:
            row['names'].append(name)
    by_dtype = Counter()
    for row in owners.values():
        by_dtype[row['dtype']] += row['bytes']
    return {'tensor_entries':entries, 'logical_bytes_including_aliases':logical_bytes,
            'unique_storage_count':len(owners), 'unique_storage_bytes':sum(r['bytes'] for r in owners.values()),
            'unique_storage_bytes_by_dtype':dict(by_dtype),
            'largest_storages':sorted(owners.values(),key=lambda r:r['bytes'],reverse=True)[:16],
            'aliased_storages':sorted((r for r in owners.values() if r['alias_count']>1),
                                      key=lambda r:r['bytes'],reverse=True)[:16]}


def audit_memory(runner, device, reset_peaks=False):
    import torch
    from vllm_hcu.v1.step5_pcp_quant_audit import _parallel_metadata

    if reset_peaks and os.environ.get('VLLM_STEP5_MEMORY_AUDIT','0') != '1':
        raise ValueError('Peak reset requires an explicit isolated Step5 memory audit window')
    target = getattr(runner,'model',None)
    if target is None:
        target = getattr(getattr(runner,'model_state',None),'model',None)
    draft = getattr(getattr(runner,'speculator',None),'model',None)
    named, dsa_states, seen_states = [], [], set()
    for label, model in [('target',target),('draft',draft)]:
        if model is not None:
            for name,tensor in model.named_parameters(remove_duplicate=False):
                named.append((label+'.parameter.'+name,tensor))
            for name,tensor in model.named_buffers(remove_duplicate=False):
                named.append((label+'.buffer.'+name,tensor))
            for name,module in model.named_modules():
                state = getattr(module,'dsa_state',None)
                if state is None or id(state) in seen_states:
                    continue
                seen_states.add(id(state))
                prefix = label+'.dsa.'+name
                dsa_states.append({'name':prefix,'global_layer_count':state.num_layers,
                    'registered_storage_layers':sorted(state._registered),
                    'logical_to_storage':getattr(state,'_step5_pcp_layer_storage',None),
                    'rope_capacity_bound':getattr(state,'_step5_pcp_rope_cache_limit',0)})
                for field in ('summary','pending_key','pending_z','_decode_logits','_spec_raw',
                              '_spec_starts','_spec_table'):
                    value = getattr(state,field,None)
                    if isinstance(value,torch.Tensor):
                        named.append((prefix+'.'+field,value))
                for bucket in ('_buf','_spec_i32'):
                    for key,value in getattr(state,bucket,{}).items():
                        if isinstance(value,torch.Tensor):
                            named.append((prefix+'.'+bucket+'.'+str(key),value))
                for index,values in enumerate(getattr(state,'_rope',{}).values()):
                    for part,value in enumerate(values):
                        if isinstance(value,torch.Tensor):
                            named.append((prefix+'.rope.%d.%d'%(index,part),value))
    inventory = storage_inventory(named)
    del named
    # This metadata probe never synchronizes or transfers tensor values.
    # A reset is only used in a separate controller-verified idle window.
    if reset_peaks:
        torch.cuda.reset_peak_memory_stats(device)
    native = {'enabled':False,'pools':[]}
    if os.environ.get('VLLM_STEP5_NATIVE_W4A8','0') == '1':
        from vllm_hcu.model_executor.layers.quantization.ct_w4a8_step5_native import native_workspace_inventory
        native = {'enabled':True,'pools':native_workspace_inventory()}
    return {'scope':'model parameters/buffers, explicit DSA owners, native pools and allocator; not all CUDA-library allocations',
            'pid':os.getpid(),'parallel':_parallel_metadata(),'storage':inventory,'native':native,'dsa_states':dsa_states,
            'peak_reset':bool(reset_peaks),'allocator':{
                'allocated_bytes':torch.cuda.memory_allocated(device),
                'reserved_bytes':torch.cuda.memory_reserved(device),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)}}
