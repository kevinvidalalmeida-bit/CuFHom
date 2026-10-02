#!/usr/bin/env bash
# One sudo invocation collects counters; the NVIDIA driver stays loaded.
set -euo pipefail
CUFHOM_PROFILE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CUFHOM_PROFILE_ROOT"
CUFHOM_PROFILE_NCU=/usr/local/cuda-13.0/bin/ncu
CUFHOM_PROFILE_PYTHON="$CUFHOM_PROFILE_ROOT/.venv/bin/python"
for CUFHOM_PROFILE_FILE in "$CUFHOM_PROFILE_NCU" "$CUFHOM_PROFILE_PYTHON" \
  benchmarks/performance_audit_20261001/evidence.json.gz \
  benchmarks/performance_audit_20261001/material.json \
  benchmarks/results/performance_audit_20261001/fiber_geometries/sfrp_vf15_ar10/N192/phase.npy; do
  test -f "$CUFHOM_PROFILE_FILE" || { echo "Missing: $CUFHOM_PROFILE_FILE"; exit 1; }
done
if [[ "${1:-}" == --check ]]; then
  echo 'Profiler, virtual environment, audit evidence and fiber inputs are present.'
  exit 0
fi
if (( EUID != 0 )); then
  echo "Run: sudo bash $CUFHOM_PROFILE_ROOT/benchmarks/profile_gpu_counters.sh" >&2
  exit 1
fi
umask 077
if [[ "${1:-}" == --resume && $# == 2 ]]; then
  CUFHOM_PROFILE_OUT="$(realpath -- "$2")"
  if [[ "$CUFHOM_PROFILE_OUT" != "$CUFHOM_PROFILE_ROOT/benchmarks/results/ncu_counters_20261001_"* \
    || ! -f "$CUFHOM_PROFILE_OUT/manifest.json" ]]; then
    echo 'Resume requires an existing counter directory under benchmarks/results/.' >&2
    exit 1
  fi
elif (( $# == 0 )); then
  CUFHOM_PROFILE_OUT="$(mktemp -d "$CUFHOM_PROFILE_ROOT/benchmarks/results/ncu_counters_20261001_XXXXXX")"
else
  echo 'Usage: profile_gpu_counters.sh [--resume COUNTER_DIRECTORY]' >&2
  exit 1
fi
function cufhom_profile_cleanup {
  if [[ -n "${SUDO_UID:-}" && -n "${SUDO_GID:-}" ]]; then
    chown -hR -- "$SUDO_UID:$SUDO_GID" "$CUFHOM_PROFILE_OUT"
  fi
}
trap cufhom_profile_cleanup EXIT
source benchmarks/reference_solvers/env_gpu.sh
export OPENBLAS_NUM_THREADS=1
export CUPY_CACHE_DIR="$CUFHOM_PROFILE_OUT/cupy_cache"
export NUMBA_CACHE_DIR="$CUFHOM_PROFILE_OUT/numba_cache"
export PYTHONPYCACHEPREFIX="$CUFHOM_PROFILE_OUT/pycache"
echo "Saving counters in: $CUFHOM_PROFILE_OUT"
"$CUFHOM_PROFILE_PYTHON" benchmarks/profile_gpu_counters.py \
  --ncu "$CUFHOM_PROFILE_NCU" --output "$CUFHOM_PROFILE_OUT"
echo "Finished. Counter reports: $CUFHOM_PROFILE_OUT"
