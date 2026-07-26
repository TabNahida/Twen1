#!/usr/bin/env bash
# Install the pinned Qwen3.5 linear-attention CUDA fast path into .venv.
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${root_dir}/.venv/bin/python"
uv_bin="${UV_BIN:-$(command -v uv)}"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
build_root="$(mktemp -d "${TMPDIR:-/tmp}/twen-cuda-kernels.XXXXXX")"
fla_commit="2e38c1fab332174d056928feaf29f8c5fd5ac550"
conv_commit="4f6ae4e26ae5fe8af9372f8d312ab25cc4595223"

if [[ ! -x "${python_bin}" ]]; then
  echo "missing virtual environment: ${python_bin}" >&2
  exit 66
fi
if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
  echo "missing CUDA compiler: ${cuda_home}/bin/nvcc" >&2
  exit 69
fi
"${python_bin}" "${root_dir}/scripts/check_cuda_toolchain.py" \
  --cuda-home "${cuda_home}" --compile >/dev/null

"${root_dir}/scripts/with_github_proxy.sh" git clone --quiet \
  https://github.com/fla-org/flash-linear-attention.git "${build_root}/fla"
git -C "${build_root}/fla" checkout --quiet "${fla_commit}"
"${root_dir}/scripts/with_github_proxy.sh" git clone --quiet \
  https://github.com/Dao-AILab/causal-conv1d.git "${build_root}/causal-conv1d"
git -C "${build_root}/causal-conv1d" checkout --quiet "${conv_commit}"

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" "${uv_bin}" pip install \
  --python "${python_bin}" --no-deps --no-build-isolation "${build_root}/fla"

# v1.6.2.post1 has no wheel for upstream torch 2.11+cu130.  Its source build
# includes an explicit sm_120 target; pin the compiler and avoid trying an
# incompatible release wheel first.
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" \
CUDA_HOME="${cuda_home}" \
PATH="${cuda_home}/bin:${PATH}" \
MAX_JOBS="${MAX_JOBS:-4}" \
CAUSAL_CONV1D_FORCE_BUILD=TRUE \
"${uv_bin}" pip install --python "${python_bin}" --no-deps --no-build-isolation \
  "${build_root}/causal-conv1d"

CUDA_HOME="${cuda_home}" \
CUDA_PATH="${cuda_home}" \
CUDACXX="${cuda_home}/bin/nvcc" \
PATH="${cuda_home}/bin:${PATH}" \
TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-/tmp/tilelang-cache}" \
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache}" \
"${python_bin}" -c '
from transformers.utils import (
    is_causal_conv1d_available,
    is_flash_linear_attention_available,
)

if not is_flash_linear_attention_available() or not is_causal_conv1d_available():
    raise SystemExit("Transformers does not see both Qwen3.5 CUDA fast-path packages")
import causal_conv1d
import fla
from tilelang import env as tilelang_env

print(f"flash-linear-attention={fla.__version__}")
print(f"causal-conv1d={causal_conv1d.__version__}")
print(f"tilelang CUDA_HOME={tilelang_env.CUDA_HOME}")
'

echo "temporary sources retained at ${build_root}"
