# SPDX-License-Identifier: Apache-2.0
"""Stage-fixed DeepGEMM policy for Step5's three mixed FP8 expert layers."""
from functools import lru_cache
import os


@lru_cache(maxsize=1)
def decode_mode():
    mode = int(os.environ.get("HCU_STEP5_FP8_DECODE_MODE", "1000"))
    if mode not in (1000, 1001, 1101):
        raise ValueError("HCU_STEP5_FP8_DECODE_MODE must be 1000, 1001 or 1101")
    return mode


def pack_decode_weight(weight):
    import deepgemm
    if decode_mode() == 1101:
        # This byte permutation is different from gfx936's shuffled W6.
        # The latter has the same shape but produces incorrect FP8 results.
        from deepgemm.m_group_gemm import pack_int8_weight_enk_to_w6_low_latency_inplace
        return pack_int8_weight_enk_to_w6_low_latency_inplace(weight).detach()
    return deepgemm.marlin_fp8_masked_weight(weight).detach()


def grouped_fp8(a, b, output, layout, expected_m, *, decode):
    import deepgemm
    if not decode:
        return deepgemm.m_grouped_fp8_gemm_nt_contiguous(
            a, b, output, layout, config={"MODE": 1000})
    mode = decode_mode()
    fn = (deepgemm.m_grouped_fp8_gemm_nt_masked_ll if mode == 1101
          else deepgemm.m_grouped_fp8_gemm_nt_masked)
    expected_rank = 6 if mode == 1101 else 7
    if b[0].ndim != expected_rank:
        raise ValueError(f"Step5 FP8 decode MODE={mode} requires rank-{expected_rank} weights")
    return fn(a, b, output, layout, expected_m, config={"MODE": mode})
