#!/usr/bin/env bash
# Source after activating the environment containing CuPy and NVIDIA libraries.
CUFHOM_CUDA_LIBS="$(python - <<'PYCODE'
import pathlib
import site
print(':'.join(str(path) for root in site.getsitepackages()
               for path in sorted(pathlib.Path(root).glob('nvidia/*/lib'))))
PYCODE
)"
if [[ -n "$CUFHOM_CUDA_LIBS" ]]; then
    export LD_LIBRARY_PATH="$CUFHOM_CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset CUFHOM_CUDA_LIBS

# NVRTC also needs a CUDA include root when only NVIDIA wheels are installed.
# Keep explicit CUDA_PATH values and system Toolkit installations intact.
if [[ -z "${CUDA_PATH:-}" ]] && ! command -v nvcc >/dev/null 2>&1 && [[ ! -d /usr/local/cuda ]]; then
    CUFHOM_CUDA_ROOT="$(python - <<'PYCODE'
import pathlib
import site
for root in site.getsitepackages():
    runtime = pathlib.Path(root) / 'nvidia' / 'cuda_runtime'
    if (runtime / 'include' / 'cuda_runtime.h').is_file():
        print(runtime)
        break
PYCODE
)"
    if [[ -n "$CUFHOM_CUDA_ROOT" ]]; then
        export CUDA_PATH="$CUFHOM_CUDA_ROOT"
    fi
    unset CUFHOM_CUDA_ROOT
fi
