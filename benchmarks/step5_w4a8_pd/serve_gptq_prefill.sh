#!/usr/bin/env bash
# Step5 GPTQ real Prefill, TP1 / DP8 / EP8; FP8-aligned chunk8192.
set -euo pipefail
export MODEL_PATH="${MODEL_PATH:-/model/step-5-preview-channel-w4a8-GPTQ}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
[[ -f "$MODEL_PATH/config.json" ]] || { echo "Missing model config: $MODEL_PATH" >&2; exit 2; }
export ACCUMULATE_DISPATCH_PROFILE_LEVEL=0
export VLLM_HCU_STEP5_W4A8_PROFILE=prefill
export VLLM_HCU_STEP5_W4A8_GEMM_BACKEND="${VLLM_HCU_STEP5_W4A8_GEMM_BACKEND:-deepgemm}"
export VLLM_HCU_STEP5_FP8_BF16_MOE="${VLLM_HCU_STEP5_FP8_BF16_MOE:-1}"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_USE_NN=1
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_HCU_STEP5_FP8_DENSE="${VLLM_HCU_STEP5_FP8_DENSE:-1}"
export VLLM_HCU_STEP5_FUSE_QKVG=1
export VLLM_HCU_STEP5_SWA_FUSED_QK=1
export VLLM_HCU_STEP5_FUSED_GATE=1
export VLLM_HCU_STEP5_QKV_INDEXER=1
export VLLM_HCU_TRITON_TOKEN_FP8_QUANT=1
export VLLM_HCU_EP_FUSED_MOE=1
export VLLM_HCU_EP_FUSED_COMBINE=1
case "${STEP5_DP_DISPATCH:-ipc}" in
  ipc) export VLLM_HCU_DP_CUSTOM_AG=1 VLLM_HCU_DP_PACKED_AG=0 ;;
  packed) export VLLM_HCU_DP_CUSTOM_AG=0 VLLM_HCU_DP_PACKED_AG=1 ;;
  *) echo 'STEP5_DP_DISPATCH must be ipc or packed' >&2; exit 2 ;;
esac
export HCU_STEP5_DP_FABRIC=0
export VLLM_HCU_HOST_SEQ_LENS=1
export HCU_STEP5_COMPACT_HOST_BLOCK_TABLES=1
export HCU_STEP5_CONTEXT_GRAPH_REGIONS="${STEP5_CONTEXT_GRAPH_REGIONS:-17152}"
export HCU_STEP5_BOUNDED_EP_WORKSPACE=1
export VLLM_HCU_STEP5_BF16_ROUTER=1
export VLLM_HCU_STEP5_FUSED_ROUTER_SHARED=1
export VLLM_STEP4_DSA_PREFILL_FP8_INDEXER=1
export HCU_MTP_PREFILL_TAIL="${HCU_MTP_PREFILL_TAIL:-1536}"
export HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS="${HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS:-4096}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
export HCU_STEP5_DENSE_TUNING="${HCU_STEP5_DENSE_TUNING:-0}"
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=1
export VLLM_HCU_USE_FUSE_MOE_GATE=1
export VLLM_HCU_USE_LIGHTOP_EP_SCATTER=1
export VLLM_HCU_USE_LIGHTOP_MOE_ALIGN=1
export VLLM_HCU_USE_CUSTOM_SILU_AND_MUL=1
export VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM=1
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_USE_FLASH_ATTN_FP8=1
export VLLM_USE_PD_SPLIT=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0
export HCU_DBO_MIN_TOKENS="${HCU_DBO_MIN_TOKENS:-8192}"
export HCU_DP_MOE_PIPE_MIN_TOKENS="${HCU_DP_MOE_PIPE_MIN_TOKENS:-16384}"
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${USE_STATIC_PLACEMENT:-1}" == 1 ]]; then
  export VLLM_STEP5_EXPERT_PLACEMENT="${STEP5_EXPERT_PLACEMENT:-$recipe_dir/assets/expert_placement_prefill.pt}"
  [[ -r "$VLLM_STEP5_EXPERT_PLACEMENT" ]] || { echo 'Missing expert placement' >&2; exit 2; }
else export VLLM_STEP5_EXPERT_PLACEMENT=""; fi
export HCU_EP_DIRECT_SUM="${HCU_EP_DIRECT_SUM:-16384}"
export EP_FUSED_MOE_TABLE="${STEP5_FP8_MOE_TABLE:-$recipe_dir/assets/step5_ep8_moe_table.json}"
[[ -r "$EP_FUSED_MOE_TABLE" ]] || { echo 'Missing FP8 MoE tuning table' >&2; exit 2; }
export VLLM_STEP4_DSA_GRAPH_REGIONS="$HCU_STEP5_CONTEXT_GRAPH_REGIONS"
export VLLM_HCU_PIN_MEMORY=1
export HIP_FORCE_DEV_KERNARG=1
export GPU_MAX_HW_QUEUES=4
export FA_DECODE_NUM_SPLITS=2
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/step5-prefill/cache/triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/workspace/step5-prefill/cache/vllm}"
unset STEP5_FAKE_RUN STEP5_PROFILE_RUN PYTHONPATH
unset VLLM_HCU_ALLOCATOR_LIMIT_GIB ROCBLAS_TENSILE_LIBPATH HIPBLASLT_TUNING_OVERRIDE_FILE
if [[ -n "${STEP5_TOPO_FILE:-}" ]]; then export NCCL_TOPO_FILE="$STEP5_TOPO_FILE"; else unset NCCL_TOPO_FILE; fi
if [[ -z "${SPECULATIVE_CONFIG:-}" ]]; then
  SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}'
fi
args=("${SERVER_PYTHON:-python3}" -m vllm.entrypoints.cli.main serve "$MODEL_PATH"
  --tokenizer "$TOKENIZER_PATH" --served-model-name step5 --reasoning-parser step3p5
  --host "${HOST:-0.0.0.0}" --port "${PORT:-18132}" --api-server-count 1
  --quantization compressed-tensors --tensor-parallel-size 1 --data-parallel-size 8
  --enable-expert-parallel --all2all-backend allgather_reducescatter --moe-backend triton
  --max-model-len "${MAX_MODEL_LEN:-262144}" --max-num-batched-tokens "${CHUNK_SIZE:-8192}"
  --max-num-seqs "${MAX_NUM_SEQS:-64}" --max-cudagraph-capture-size 64
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.875}" --kv-cache-dtype fp8_e4m3
  --no-enable-prefix-caching --limit-mm-per-prompt '{"image":0}'
  --speculative-config "$SPECULATIVE_CONFIG"
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","compile_sizes":[16,64],"custom_ops":["+swiglustep_and_mul"]}')
if [[ "${1:-}" == --dry-run ]]; then
  printf 'VLLM_HCU_STEP5_W4A8_PROFILE=%s\nVLLM_HCU_STEP5_FP8_BF16_MOE=%s\nVLLM_HCU_STEP5_W4A8_GEMM_BACKEND=%s\n' "$VLLM_HCU_STEP5_W4A8_PROFILE" "$VLLM_HCU_STEP5_FP8_BF16_MOE" "$VLLM_HCU_STEP5_W4A8_GEMM_BACKEND"
  printf 'STEP5_DP_DISPATCH=%s\nHCU_STEP5_COMPACT_HOST_BLOCK_TABLES=%s\nHCU_STEP5_CONTEXT_GRAPH_REGIONS=%s\n' "${STEP5_DP_DISPATCH:-ipc}" "$HCU_STEP5_COMPACT_HOST_BLOCK_TABLES" "$HCU_STEP5_CONTEXT_GRAPH_REGIONS"
  printf '%q ' "${args[@]}"; printf '\n'
  exit 0
fi
exec "${args[@]}" "$@"
