#!/usr/bin/env bash
set -euo pipefail
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$recipe_dir/../.." && pwd)"
export VLLM_HCU_BUILD_STEP5_DENSE=1
export PYTORCH_ROCM_ARCH=gfx938
export MAX_JOBS="${MAX_JOBS:-2}"
# Mounted host-user files can be built by container root. setuptools' Git
# finder sanitizes GIT_CONFIG_* variables, so trust only this exact checkout
# in the builder account's config when its owner differs.
if [[ "$(stat -c %u "$repo_dir")" != "$EUID" ]]; then
  build_trusted_paths="$(git config --global --get-all safe.directory || true)"
  case $'\n'"$build_trusted_paths"$'\n' in
    *$'\n'"$repo_dir"$'\n'*) ;;
    *) git config --global --add safe.directory "$repo_dir" ;;
  esac
fi
exec "${SERVER_PYTHON:-python3}" -m pip install --no-deps --no-build-isolation --force-reinstall "$repo_dir"
