#!/usr/bin/env bash
# Based on the user's original container command; RDMA needs the extra device
# mapping and unlimited locked memory. Keep model mounts read-only.
set -euo pipefail
IMAGE=${IMAGE:-42.228.13.241:5000/jenkins/model_test_env/vllm:0.28.1rc1-ubuntu22.04-dtk2610-py3.12-sf_b075-20261103-1558}
NAME=${NAME:-step-5_zyq}
CODE_DIR=${CODE_DIR:-/public/home/xdb4_10676/zhengyq}
if docker inspect "$NAME" >/dev/null 2>&1; then
  printf 'Container %s already exists; inspect its /dev/infiniband mapping before reuse.\n' "$NAME" >&2
  exit 1
fi
docker run -d --network=host --ipc=host --shm-size=16G \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  --device=/dev/infiniband --ulimit memlock=-1:-1 \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$CODE_DIR:/my_code" \
  -v /public/share/rogon_admin/models:/models:ro \
  --name "$NAME" --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  "$IMAGE" /bin/bash -lc 'exec sleep infinity'
