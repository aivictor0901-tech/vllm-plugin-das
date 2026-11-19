"""Serving context bounds must preserve MTP lookahead and legacy defaults."""
import ast
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2] / 'vllm_hcu/v1'
spec = importlib.util.spec_from_file_location('_pcp_memory', ROOT/'step5_pcp_memory.py')
memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory)


def dsa_methods():
    source = ROOT.parent/'model_executor/layers/step4_dsa.py'
    tree = ast.parse(source.read_text())
    owner = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Step4DSAState')
    names = {'register_layer','ensure_allocated'}
    nodes = [n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name in names]
    class Tensor:
        def __init__(self, shape, dtype='int8'):
            self.shape, self.dtype = tuple(shape), dtype
            self.ndim = len(shape)
        def numel(self):
            return math.prod(self.shape)
        def element_size(self):
            return 1 if self.dtype == 'fp8' else 2
        def __getitem__(self, index):
            return Tensor(self.shape[1:],self.dtype)
    torch = NS(float8_e4m3fn='fp8',cuda=NS(is_current_stream_capturing=lambda:False),
               zeros=lambda shape, dtype, device:Tensor(shape,dtype),
               zeros_like=lambda value:Tensor(value.shape,value.dtype))
    namespace = {'torch':torch,'logger':NS(info=lambda *a:None)}
    future = ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+nodes,type_ignores=[])),
                 '<actual-dsa-memory-methods>','exec'),namespace)
    return namespace,Tensor


class PCPMemory(unittest.TestCase):
    def test_ordinary_rotary_cache_keeps_defaults_and_frequency_parameters(self):
        cfg = NS(cp=True, model_config=NS(max_model_len=131073, enforce_eager=True),
                 speculative_config=NS(num_speculative_tokens=3))
        contract = ModuleType('vllm_hcu.v1.step5_pcp_config')
        contract.is_step5_pcp = lambda c: c.cp
        params = {'rope_type':'llama3','factor':1.,'original_max_position_embeddings':1048576,
                  'rope_theta':1e7,'partial_rotary_factor':1/3}
        before = dict(params)
        with patch.dict(sys.modules,{contract.__name__:contract}), patch.dict(os.environ,{},clear=True):
            self.assertEqual(memory.rotary_cache_position(cfg,1048576,params),1048576)
            os.environ['VLLM_STEP5_PCP_TRIM_ROTARY_CACHE']='1'
            self.assertEqual(memory.rotary_cache_position(cfg,1048576,params),131136)
            self.assertEqual(memory.rotary_cache_position(cfg,8192,params),8192)
            self.assertEqual(params,before)
            self.assertEqual(memory.indexer_rope_limit(cfg),0)
            cfg.cp=False
            self.assertEqual(memory.rotary_cache_position(cfg,1048576,params),1048576)

    def test_ordinary_rotary_rejects_unverified_scaling_and_graphs(self):
        cfg=NS(model_config=NS(max_model_len=131073,enforce_eager=True),
               speculative_config=NS(num_speculative_tokens=3))
        contract=ModuleType('vllm_hcu.v1.step5_pcp_config');contract.is_step5_pcp=lambda c:True
        with patch.dict(sys.modules,{contract.__name__:contract}),patch.dict(os.environ,{'VLLM_STEP5_PCP_TRIM_ROTARY_CACHE':'1'}):
            for params in ({'rope_type':'yarn'},{'rope_type':'llama3','factor':8.}):
                with self.assertRaises(ValueError):memory.rotary_cache_position(cfg,1048576,params)
            cfg.model_config.enforce_eager=False
            with self.assertRaises(ValueError):memory.rotary_cache_position(cfg,1048576,{'rope_type':'default'})

    def test_pp_local_storage_retains_unique_layer_ownership_and_schedule_order(self):
        api, Tensor = dsa_methods()
        for compact in (False,True):
            for slots in (range(11),range(11,23),(3,7,11,15),(15,3,11,7)):
                slots = list(slots)
                state = NS(_step5_pcp_compact_layers=compact,_registered=set(),summary=None,
                           num_layers=23,geometry=NS(region_size=8,proxy_dim=256),max_pending=80,
                           dtype='bf16',_allocate_graph_buffers=lambda *a:None)
                mapped = [api['register_layer'](state,s) for s in slots]
                self.assertEqual(mapped,list(range(len(slots))) if compact else slots)
                self.assertEqual(api['register_layer'](state,slots[0]),mapped[0])
                self.assertEqual(state.first_slot,mapped[slots.index(min(slots))])
                api['ensure_allocated'](state,Tensor((2,1024,64,4,128),'bf16'),'cuda:0')
                size = len(slots) if compact else 23
                self.assertEqual(state.summary.shape,(size,8192,1,256))
                self.assertEqual(state.pending_key.shape,(size,81,8,256))
                self.assertEqual(state.pending_z.shape,state.pending_key.shape)
                if compact:
                    with self.assertRaises(RuntimeError):
                        api['register_layer'](state,999)

    def test_default_off_and_other_model_pd_do_not_trim(self):
        contract = ModuleType('vllm_hcu.v1.step5_pcp_config')
        contract.is_step5_pcp = lambda c: c.cp
        cfg = NS(cp=True, model_config=NS(max_model_len=131073, enforce_eager=True),
                 speculative_config=NS(num_speculative_tokens=3))
        with patch.dict(sys.modules, {contract.__name__: contract}), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(memory.indexer_rope_limit(cfg), 0)
            os.environ['VLLM_STEP5_PCP_TRIM_INDEXER_ROPE'] = '1'
            cfg.cp = False
            self.assertEqual(memory.indexer_rope_limit(cfg), 0)

    def test_serving_limit_keeps_boundary_speculative_positions(self):
        contract = ModuleType('vllm_hcu.v1.step5_pcp_config')
        contract.is_step5_pcp = lambda c: True
        with patch.dict(sys.modules, {contract.__name__: contract}), patch.dict(
                os.environ, {'VLLM_STEP5_PCP_TRIM_INDEXER_ROPE': '1'}):
            for maximum in (8191, 8192, 8193, 16383, 16384, 16385, 131073):
                for drafts in range(4):
                    cfg = NS(model_config=NS(max_model_len=maximum, enforce_eager=True),
                             speculative_config=NS(num_speculative_tokens=drafts))
                    bound = memory.indexer_rope_limit(cfg)
                    self.assertGreaterEqual(bound, maximum + drafts + 1)
                    self.assertLess(bound, maximum + drafts + 65)
                    self.assertEqual(bound % 64, 0)
            cfg.model_config.enforce_eager = False
            with self.assertRaises(ValueError):
                memory.indexer_rope_limit(cfg)


if __name__ == '__main__':
    unittest.main()
