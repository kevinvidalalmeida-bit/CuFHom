#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
CUFHOM_LIB_DIRS=""
for CUFHOM_LIB in "$PWD"/.venv/lib/python*/site-packages/nvidia/*/lib; do
    CUFHOM_LIB_DIRS="${CUFHOM_LIB_DIRS:+$CUFHOM_LIB_DIRS:}$CUFHOM_LIB"
done
export LD_LIBRARY_PATH="$CUFHOM_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
exec .venv/bin/python "$@"
