#!/usr/bin/env bash
# Source after activating the environment containing CuPy and NVIDIA libraries.
CUFHOM_CUDA_LIBS="$(python - <<'PYCODE'
import pathlib, site
print(':'.join(str(path) for root in site.getsitepackages()
               for path in sorted(pathlib.Path(root).glob('nvidia/*/lib'))))
PYCODE
)"
if [[ -n "$CUFHOM_CUDA_LIBS" ]]; then
    export LD_LIBRARY_PATH="$CUFHOM_CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset CUFHOM_CUDA_LIBS
