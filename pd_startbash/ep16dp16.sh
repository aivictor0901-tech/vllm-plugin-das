#!/usr/bin/env bash
# Adapted from the user-provided pd_startbash launchers for vllm-plugin-das.
# Combined prefill/decode service: head/worker are node roles, not P/D roles.
set -euo pipefail

# Prefer this checkout over the image's preinstalled plugin in every child.
PLUGIN_ROOT=${PLUGIN_ROOT:-/my_code/vllm-plugin-das}
export PYTHONPATH="$PLUGIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  printf 'Usage: %s {head|worker} [additional vLLM arguments...]\n' "$0" >&2
  printf 'Set HEAD_NODE_IP to override the default head address (10.1.12.67).\n' >&2
}

if (($# == 0)); then
  usage
  exit 2
fi

ROLE=$1
shift
case "$ROLE" in
  head|worker) ;;
  *)
    usage
    exit 2
    ;;
esac

# Keep each invocation's log, and expose stable links for live inspection.
LOG_DIR=${LOG_DIR:-/my_code/pd_log/ep16dp16}
RUN_ID=${RUN_ID:-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RUN_ID}_${ROLE}.log"
printf 'Log file: %s\n' "$LOG_FILE"
ln -sfn "$(basename -- "$LOG_FILE")" "$LOG_DIR/${ROLE}.latest.log"
exec >> "$LOG_FILE" 2>&1
printf 'Started %s node=%s role=%s plugin=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "${DEPLOY_NODE:-$HOSTNAME}" "$ROLE" "$PLUGIN_ROOT"

HEAD_NODE_IP=${HEAD_NODE_IP:-10.1.12.67}
DP_RPC_PORT=${DP_RPC_PORT:-13345}
MODEL_PATH=${MODEL_PATH:-/models/Step-5-Preview-Channel-INT4-w4a8-INT8-w8a8}

if [[ ! -d "$MODEL_PATH" ]]; then
  printf 'Model directory does not exist: %s\n' "$MODEL_PATH" >&2
  exit 1
fi

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# DSA score tensors grow with context; preserve large blocks for reuse.
# Reclaim inactive blocks before they starve HSA's non-PyTorch allocations.
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-max_split_size_mb:256,garbage_collection_threshold:0.85}
# Bound DSA scores and selector workspace without changing the scheduling budget.
export VLLM_STEP4_DSA_PREFILL_PLAN_CHUNK=${VLLM_STEP4_DSA_PREFILL_PLAN_CHUNK:-1024}
# BoltOps fused_moe.py exposes this token-axis chunk size (default 32768).
export TRITON_FUSED_MOE_CHUNK_SIZE=${TRITON_FUSED_MOE_CHUNK_SIZE:-4096}
# The image's SHCA provider is present but is not in libibverbs.d.
# /dev/infiniband must also be mapped into the container (see create script).
export RDMAV_DRIVERS=${RDMAV_DRIVERS:-shca}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-shca_0,shca_1,shca_2,shca_3}
# Fail explicitly if IB is unavailable instead of benchmarking the 10Gb NIC.
# Built-in IB skips SHCA; use the provider shipped with this image.
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-/opt/rccl-rdma-sharp-plugins/lib/librccl-net-shca.so}
export NCCL_NET="${NCCL_NET:-NCCL RDMA Plugin v8}"
if [[ ! -d /dev/infiniband || ! -r "$NCCL_NET_PLUGIN" ]]; then
  printf 'SHCA needs /dev/infiniband and plugin %s; see create_step5_container.sh.\n' "$NCCL_NET_PLUGIN" >&2
  exit 1
fi
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
export NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE:-$LOG_DIR/${RUN_ID}_nccl_${ROLE}_%h_%p.log}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eno1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}
export HSA_FORCE_FINE_GRAIN_PCIE=1
export VLLM_USE_V2_MODEL_RUNNER=1

export HCU_DP_MOE_PIPE_MIN_TOKENS=16384
export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS-8}
export HCU_EP_DIRECT_SUM=${HCU_EP_DIRECT_SUM-16384}
export HCU_DBO_MIN_TOKENS=${HCU_DBO_MIN_TOKENS:-0}

export GPU_MAX_HW_QUEUES=4
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=1
export VLLM_HCU_USE_FUSE_MOE_GATE=1
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_HCU_USE_LIGHTOP_EP_SCATTER=1
export VLLM_HCU_USE_CUSTOM_SILU_AND_MUL=1
export VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_HCU_USE_LIGHTOP_MOE_ALIGN=1
export FA_DECODE_NUM_SPLITS=2
export VLLM_HCU_EP_FUSED_MOE=1
export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0
export HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS=4096
export VLLM_HCU_EP_FUSED_COMBINE=1
export VLLM_HCU_DP_CUSTOM_AG=1
# This checkpoint already has INT8 dense weights; do not requantize as FP8.
export VLLM_STEP5_MIXED_DENSE_INT8=1
export VLLM_STEP5_LM_HEAD_INT8_SKINNY=1
export VLLM_HCU_STEP5_FP8_DENSE=0
export VLLM_HCU_STEP5_INT8_DENSE=0
export VLLM_HCU_STEP5_FP8_BF16_MOE=0

unset ROCBLAS_TENSILE_LIBPATH
unset HIPBLASLT_TUNING_OVERRIDE_FILE

export VLLM_HCU_STEP5_SWA_FUSED_QK=1
export VLLM_HCU_STEP5_FUSED_GATE=1
export VLLM_HCU_TRITON_TOKEN_FP8_QUANT=1
export VLLM_HCU_STEP5_QKV_INDEXER=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
# Upstream execute_model/sample_tokens RPC defaults to 300 s; EP16 with
# 16 x 8192 scheduled tokens can exceed it while still computing normally.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/opt/step5_runtime/triton_cache}
mkdir -p "$TRITON_CACHE_DIR"
export VLLM_HCU_PIN_MEMORY=1
export HIP_FORCE_DEV_KERNARG=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_STEP4_DSA_GRAPH_REGIONS=8704
export AITER_PTQ_I8_BACKEND=triton
export HCU_STEP5_DEEPEP_PROFILE_DBO_OUTPUT=1
# Diagnostics: dump rocSHMEM's auto-selected HCA/NIC topology at Buffer init.
export ROCSHMEM_PRINT_TOPO=${ROCSHMEM_PRINT_TOPO:-1}
export ACCUMULATE_DISPATCH_PROFILE_LEVEL=0

COMMON_ARGS=(
  --served-model-name step5
  --reasoning-parser step3p5
  --tensor-parallel-size 1
  --distributed-executor-backend mp
  --data-parallel-size 16
  --data-parallel-size-local 8
  --data-parallel-address "$HEAD_NODE_IP"
  --data-parallel-rpc-port "$DP_RPC_PORT"
  --enable-expert-parallel
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}"
  # 131072 input tokens plus the single benchmark output token.
  --max-model-len "${MAX_MODEL_LEN:-131073}"
  --max-num-seqs 32
  --no-enable-prefix-caching
  --kv-cache-dtype fp8_e4m3
  --limit-mm-per-prompt '{"image": 0}'
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.82}"
  --enable-ep-weight-filter
  --enforce-eager
  --all2all-backend allgather_reducescatter
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/my_code/pd_log/profile","torch_profiler_record_shapes": true,"torch_profiler_with_stack": true}'
  # --moe-backend deep_gemm
)

if [[ "$ROLE" == head ]]; then
  exec vllm serve "$MODEL_PATH" \
    --host "${API_HOST:-127.0.0.1}" \
    --port "${API_PORT:-18666}" \
    --api-server-count 1 \
    "${COMMON_ARGS[@]}" \
    "$@"
else
  exec vllm serve "$MODEL_PATH" \
    --headless \
    --data-parallel-start-rank 8 \
    "${COMMON_ARGS[@]}" \
    "$@"
fi
