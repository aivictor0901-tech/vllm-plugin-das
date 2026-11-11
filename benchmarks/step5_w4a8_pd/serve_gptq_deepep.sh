#!/usr/bin/env bash
set -euo pipefail
# Unified Step5 W4A8: HT for prefill/mixed batches, LL for pure decode.
# TP/DP/PP are runtime inputs. EP size follows vLLM's parallel groups.
export VLLM_HCU_STEP5_W4A8_PROFILE=auto
export VLLM_HCU_STEP5_W4A8_GEMM_BACKEND=deepgemm
export VLLM_HCU_STEP5_FP8_BF16_MOE=1
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=1
export VLLM_HCU_STEP5_FP8_DENSE=1
export VLLM_HCU_STEP5_BF16_ROUTER=1
export VLLM_HCU_STEP5_FUSED_GATE=1
export VLLM_HCU_STEP5_FUSED_ROUTER_SHARED=1
export VLLM_HCU_STEP5_FUSE_QKVG=1
export VLLM_HCU_STEP5_QKV_INDEXER=1
export VLLM_HCU_STEP5_SWA_FUSED_QK=1
export VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM=1
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_USE_FLASH_ATTN_FP8=1
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_USE_NN=1
export VLLM_USE_PD_SPLIT=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_HCU_DP_CUSTOM_AG=0
export VLLM_HCU_DP_PACKED_AG=0
export VLLM_HCU_EP_FUSED_COMBINE=0
export HCU_DBO_MIN_TOKENS=0
export HCU_DP_MOE_PIPE_MIN_TOKENS=0
export HCU_STEP5_DEEPEP_PROFILE_DBO_OUTPUT=0
export HCU_STEP5_DEEPEP_DBO_BUFFER_PAIR=0
export HCU_STEP5_DEEPEP_FUSED_SCATTER="${HCU_STEP5_DEEPEP_FUSED_SCATTER:-1}"
export HCU_STEP5_COMPACT_HOST_BLOCK_TABLES=1
export HCU_STEP5_BOUNDED_EP_WORKSPACE=1
export HCU_MTP_PREFILL_TAIL=1536
export HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS=4096
export HCU_STEP5_CONTEXT_GRAPH_REGIONS="${HCU_STEP5_CONTEXT_GRAPH_REGIONS:-17152}"
export VLLM_STEP4_DSA_GRAPH_REGIONS="$HCU_STEP5_CONTEXT_GRAPH_REGIONS"
export VLLM_STEP4_DSA_PREFILL_FP8_INDEXER=1
export ACCUMULATE_DISPATCH_PROFILE_LEVEL=0
export ROCSHMEM_IPC_MNVL="${ROCSHMEM_IPC_MNVL:-1}"
export ROCSHMEM_IPC_MNVL_NPES="${ROCSHMEM_IPC_MNVL_NPES:-16}"
export DEEPEP_ENABLE_LL_LAYERED_OPT="${DEEPEP_ENABLE_LL_LAYERED_OPT:-1}"
export FA_DECODE_NUM_SPLITS="${FA_DECODE_NUM_SPLITS:-2}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-4}"
export HIP_FORCE_DEV_KERNARG=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

model="${MODEL_PATH:-/model/step-5-preview-channel-w4a8-GPTQ}"
spec="${SPECULATIVE_CONFIG:-}"
if [[ -z "$spec" ]]; then
    spec='{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}'
fi
compile="${COMPILATION_CONFIG:-}"
if [[ -z "$compile" ]]; then
    compile='{"cudagraph_mode":"FULL_AND_PIECEWISE","compile_sizes":[16,64],"custom_ops":["+swiglustep_and_mul"]}'
fi
args=("${SERVER_PYTHON:-python3}" -m vllm.entrypoints.cli.main serve "$model"
    --tokenizer "$model" --served-model-name step5 --reasoning-parser step3p5
    --host "${HOST:-127.0.0.1}" --port "${PORT:-18134}" --api-server-count 1
    --quantization compressed-tensors
    --tensor-parallel-size "${TP:-4}" --data-parallel-size "${DP:-2}"
    --pipeline-parallel-size "${PP:-1}" --enable-expert-parallel
    --all2all-backend deepep_auto --moe-backend deep_gemm
    --max-num-batched-tokens "${CHUNK_SIZE:-8192}"
    --max-model-len "${MAX_MODEL_LEN:-139264}" --max-num-seqs "${MAX_NUM_SEQS:-64}"
    --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE:-256}"
    --kv-cache-dtype fp8_e4m3 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
    --no-enable-prefix-caching --limit-mm-per-prompt '{"image":0}'
    --speculative-config "$spec" --safetensors-load-strategy lazy
    --compilation-config "$compile" --cudagraph-metrics)
if [[ -n "${KV_TRANSFER_CONFIG:-}" ]]; then args+=(--kv-transfer-config "$KV_TRANSFER_CONFIG"); fi
if [[ "${1:-}" == --dry-run ]]; then printf '%q ' "${args[@]}"; printf '\n'; exit 0; fi
exec "${args[@]}" "$@"
