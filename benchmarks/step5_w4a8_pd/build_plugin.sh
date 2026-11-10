#!/usr/bin/env bash
set -euo pipefail
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$recipe_dir/../.." && pwd)"
export PYTORCH_ROCM_ARCH=gfx938
export MAX_JOBS="${MAX_JOBS:-2}"
exec python3 -m pip install --no-build-isolation -e "$repo_dir"
