#!/usr/bin/env bash
# Run one command with a compiler/header-coherent CUDA toolkit for TileLang JIT.
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${root_dir}/.venv/bin/python"
cuda_home="${TWEN_CUDA_HOME:-${CUDA_HOME:-/usr/local/cuda}}"

if [[ ! -x "${python_bin}" ]]; then
  echo "missing virtual environment: ${python_bin}" >&2
  exit 66
fi
if [[ "$#" -eq 0 ]]; then
  echo "usage: scripts/with_cuda_toolchain.sh COMMAND [ARG ...]" >&2
  exit 64
fi

toolchain_report="$(${python_bin} "${root_dir}/scripts/check_cuda_toolchain.py" \
  --cuda-home "${cuda_home}")" || {
  echo "${toolchain_report}" >&2
  exit 69
}

cuda_home="$(${python_bin} -c 'import json,sys; print(json.load(sys.stdin)["toolchain"]["cuda_home"])' \
  <<<"${toolchain_report}")"
export CUDA_HOME="${cuda_home}"
export CUDA_PATH="${cuda_home}"
export CUDACXX="${cuda_home}/bin/nvcc"
export PATH="${cuda_home}/bin:${PATH}"
# TileLang 0.1.9 compiles on the coherent 13.2 toolkit, but its full T=4096
# gated-delta backward is not alignment-safe on sm_120.  Keep Triton as the
# production default while allowing an explicit FLA_TILELANG=1 for isolated
# compiler/kernel experiments.
export FLA_TILELANG="${FLA_TILELANG:-0}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-${root_dir}/.cache/tilelang}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${root_dir}/.cache/triton}"
mkdir -p "${TILELANG_CACHE_DIR}" "${TRITON_CACHE_DIR}"

exec "$@"
