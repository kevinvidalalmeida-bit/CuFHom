#!/usr/bin/env bash
# CUDA user-space libraries installed in this repository's virtual environment.
CUFHOM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUDA_WHEEL_LIBS="$CUFHOM_ROOT/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$CUFHOM_ROOT/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:$CUFHOM_ROOT/.venv/lib/python3.12/site-packages/nvidia/cufft/lib:$CUFHOM_ROOT/.venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib"
export LD_LIBRARY_PATH="$CUDA_WHEEL_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
