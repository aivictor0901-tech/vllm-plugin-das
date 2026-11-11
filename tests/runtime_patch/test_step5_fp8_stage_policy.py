"""CPU API contracts; real numerical/Graph checks live in the GPU benchmark."""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

PATH=Path(__file__).resolve().parents[2]/'vllm_hcu/model_executor/layers/fused_moe/step5_fp8_gemm.py'
spec=importlib.util.spec_from_file_location('step5_fp8_stage_policy_under_test',PATH)
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)


class TestFixedStagePolicy(unittest.TestCase):
    def setUp(self):
        policy.decode_mode.cache_clear()
        self.env=patch.dict(os.environ,{},clear=True);self.env.start()
        self.calls=[]
        module=ModuleType('deepgemm')
        for name in ['m_grouped_fp8_gemm_nt_contiguous','m_grouped_fp8_gemm_nt_masked','m_grouped_fp8_gemm_nt_masked_ll']:
            def call(*args,_name=name,**kwargs):self.calls.append((_name,args,kwargs))
            setattr(module,name,call)
        self.modules=patch.dict(sys.modules,{'deepgemm':module});self.modules.start()

    def tearDown(self):
        self.modules.stop();self.env.stop();policy.decode_mode.cache_clear()

    def gemm(self,m,decode,rank):
        policy.grouped_fp8(('a','as'),(SimpleNamespace(ndim=rank),'bs'),'out','counts',m,decode=decode)

    def test_decode_uses_one_mode_for_small_and_large_m(self):
        self.gemm(1,True,7);self.gemm(64,True,7)
        self.assertEqual([v[0] for v in self.calls],['m_grouped_fp8_gemm_nt_masked']*2)
        self.assertEqual([v[2]['config'] for v in self.calls],[{'MODE':1000}]*2)

    def test_prefill_is_independent_from_decode_mode(self):
        os.environ['HCU_STEP5_FP8_DECODE_MODE']='invalid'
        self.gemm(512,False,7)
        self.assertEqual(self.calls[0][0],'m_grouped_fp8_gemm_nt_contiguous')
        self.assertEqual(self.calls[0][2]['config'],{'MODE':1000})

    def test_wrong_weight_layout_is_rejected_before_launch(self):
        with self.assertRaisesRegex(ValueError,'rank-7'):
            self.gemm(8,True,6)
        self.assertFalse(self.calls)

    def test_mode_cannot_change_under_resident_packed_weights(self):
        self.assertEqual(policy.decode_mode(),1000)
        os.environ['HCU_STEP5_FP8_DECODE_MODE']='1101'
        self.assertEqual(policy.decode_mode(),1000)

    def test_ll_override_has_matching_abi(self):
        os.environ['HCU_STEP5_FP8_DECODE_MODE']='1101'
        self.gemm(8,True,6)
        self.assertEqual(self.calls[0][0],'m_grouped_fp8_gemm_nt_masked_ll')
        self.assertEqual(self.calls[0][2]['config'],{'MODE':1101})

    def test_unsupported_mode_fails_before_launch(self):
        os.environ['HCU_STEP5_FP8_DECODE_MODE']='999'
        with self.assertRaises(ValueError):self.gemm(8,True,6)
        self.assertFalse(self.calls)


if __name__=='__main__':unittest.main()
